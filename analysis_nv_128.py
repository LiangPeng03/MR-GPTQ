"""
关键测试程序: macro_scatter.png, macro_block.png, nvfp4_all_layers_summary.png
"""
import os, sys, torch, numpy as np
import matplotlib.pyplot as plt
import matplotlib
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use('Agg')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.quantization.quantizer import Quantizer
from src.transforms.transforms import build_transform

def collect_activations(model, layers_to_hook, calib_data, device):
    activations = {}
    for idx in layers_to_hook:
        layer = model.model.layers[idx].mlp.down_proj
        cache = []
        def hook(m, i, o): cache.append(i[0].detach().cpu())
        handle = layer.register_forward_hook(hook)
        with torch.no_grad():
            model(**calib_data)
        handle.remove()
        
        W = layer.weight.detach().cpu().float()
        activations[idx] = {
            'X': cache[0].view(-1, cache[0].shape[-1]).float(),
            'W': W
        }
        torch.cuda.empty_cache()
    return activations

def quantize_group(values, quantizer, device):
    """对一?group 的值进行量化，返回量化后的值和 scale"""
    x = values.to(device).unsqueeze(0)  # (1, group_size)
    scales, zeros = quantizer.get_quantization_params(x)
    x_q = quantizer(x, scales, zeros)
    return x_q.squeeze(0).cpu(), scales.squeeze().cpu().item()

def analyze_block(block_values, quantizer_nv, quantizer_mx, transform, device):

    block = block_values.clone()
    
    # --- 旋转 ---
    block_rot = transform(block.unsqueeze(0).to(device)).squeeze(0).cpu()
    
    # 辅助函数：量化整?block
    def quantize_full_block(data, quantizer, gs, is_locked=False, lock_ref_data=None):
        x = data.to(device).unsqueeze(0) # (1, 128)
        
        # 重置 quantizer 状?
        quantizer._track_global_scale = (quantizer.scale_precision.value == "e4m3")
        quantizer.global_scale = torch.tensor([float("inf")], dtype=torch.float32).to(device)
        
        if is_locked and quantizer.scale_precision.value == "e4m3":
            # 锁定 global_scale ?lock_ref_data 的最大?
            from src.quantization.quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX
            from src.quantization.quantizer import get_reciprocal
            act_max_val = lock_ref_data.abs().max().to(torch.float32).view(1)
            locked_scale = FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(act_max_val)
            quantizer.global_scale = locked_scale.to(device)
            quantizer._track_global_scale = False
            
        scales, zeros = quantizer.get_quantization_params(x)
        x_q = quantizer(x, scales, zeros)
        
        x_q = x_q.squeeze(0).cpu() # (128,)
        scales = scales.squeeze().cpu() # (n_groups,)
        
        n_groups = 128 // gs
        groups_info = []
        for g in range(n_groups):
            vals = data[g*gs : (g+1)*gs].float()
            q_vals = x_q[g*gs : (g+1)*gs].float()
            scale = scales[g].item() if scales.dim() > 0 else scales.item()
            mse = (q_vals - vals).pow(2).mean().item()
            groups_info.append({
                "values": vals.numpy(),
                "quantized": q_vals.numpy(),
                "scale": scale,
                "mse": mse,
                "max_abs": vals.abs().max().item()
            })
        total_mse = np.mean([g["mse"] for g in groups_info])
        return groups_info, total_mse
    
    def compute_stats(g_info, is_rotated, raw_data):
        q_vals = np.concatenate([g["quantized"] for g in g_info])
        err2 = (q_vals - raw_data)**2
        max_idx = np.argmax(np.abs(block.numpy()))
        mse_max = err2[max_idx]
        mse_other = (np.sum(err2) - mse_max) / 127.0
        return mse_max, mse_other
    
    results = {}
    
    # 1. NVFP4 No Rotation
    g_info, mse = quantize_full_block(block, quantizer_nv, 16)
    mse_max, mse_other = compute_stats(g_info, False, block.numpy())
    results["nvfp4_norot"] = {"groups": g_info, "total_mse": mse, "name": "NVFP4 No Rotation", "mse_max": mse_max, "mse_other": mse_other}
    
    # 2. NVFP4 Rotated (Unlocked)
    g_info, mse = quantize_full_block(block_rot, quantizer_nv, 16)
    mse_max, mse_other = compute_stats(g_info, True, block_rot.numpy())
    results["nvfp4_rot"] = {"groups": g_info, "total_mse": mse, "name": "NVFP4 Rotated (Unlocked)", "mse_max": mse_max, "mse_other": mse_other}
    
    # 3. NVFP4 Rotated (Locked to original block)
    g_info, mse = quantize_full_block(block_rot, quantizer_nv, 16, is_locked=True, lock_ref_data=block)
    mse_max, mse_other = compute_stats(g_info, True, block_rot.numpy())
    results["nvfp4_rot_locked"] = {"groups": g_info, "total_mse": mse, "name": "NVFP4 Rotated (Locked)", "mse_max": mse_max, "mse_other": mse_other}
    
    # 4. MXFP4 No Rotation
    g_info, mse = quantize_full_block(block, quantizer_mx, 32)
    mse_max, mse_other = compute_stats(g_info, False, block.numpy())
    results["mxfp4_norot"] = {"groups": g_info, "total_mse": mse, "name": "MXFP4 No Rotation", "mse_max": mse_max, "mse_other": mse_other}
    
    # 5. MXFP4 Rotated
    g_info, mse = quantize_full_block(block_rot, quantizer_mx, 32)
    mse_max, mse_other = compute_stats(g_info, True, block_rot.numpy())
    results["mxfp4_rot"] = {"groups": g_info, "total_mse": mse, "name": "MXFP4 Rotated", "mse_max": mse_max, "mse_other": mse_other}
    
    return results, block.numpy(), block_rot.numpy()

def print_report(layer_idx, block_idx, token_idx, orig, rotated, results):
    print(f"\n{'='*80}")
    print(f"  Layer {layer_idx}, Block {block_idx} (channels {block_idx*128}-{(block_idx+1)*128-1}), Token {token_idx}")
    print(f"{'='*80}")

    max_pos = np.argmax(np.abs(orig))
    print(f"\n  原始�?(128 channels), Max |值| = {orig[max_pos]:.4f} @ position {max_pos}")
    print(f"  �?6: {np.array2string(orig[:16], precision=3, separator=', ')}")
    
    print(f"\n  旋转�?(Hadamard-128), Max |值| = {np.max(np.abs(rotated)):.4f}")
    print(f"  �?6: {np.array2string(rotated[:16], precision=3, separator=', ')}")
    
    for fmt in ["nvfp4", "mxfp4"]:
        norot = results[f"{fmt}_norot"]
        rot = results[f"{fmt}_rot"]
        rot_locked = results.get(f"{fmt}_rot_locked", None)
        n_groups = len(norot["groups"])
        gs = 128 // n_groups
        
        print(f"\n{'─'*90}")
        print(f"  {norot['name'].split(' No Rotation')[0]}")
        print(f"{'─'*90}")
        print(f"  {'Group':>7} | {'Condition':>12} | {'Scale':>10} | {'Max|Val|':>10} | {'Group MSE':>12} | �?个�?)")
        print(f"  {'─'*7}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*30}")
        
        for g in range(n_groups):
            g_norot = norot["groups"][g]
            g_rot = rot["groups"][g]
            
            # 标记�?Outlier 的组
            is_outlier = g_norot["max_abs"] > 2 * np.median([gi["max_abs"] for gi in norot["groups"]])
            marker = " �? if is_outlier else "
            
            vals_str = np.array2string(g_norot["values"][:6], precision=2, separator=',')
            print(f"  G{g:>5} | {'NoRot':>12} | {g_norot['scale']:>10.4f} | {g_norot['max_abs']:>10.4f} | {g_norot['mse']:>12.6f} | {vals_str}{marker}")
            
            vals_str_r = np.array2string(g_rot["values"][:6], precision=2, separator=',')
            print(f"  {'':>7} | {'Rot(Unlck)':>12} | {g_rot['scale']:>10.4f} | {g_rot['max_abs']:>10.4f} | {g_rot['mse']:>12.6f} | {vals_str_r}")
            
            if rot_locked:
                g_rot_l = rot_locked["groups"][g]
                vals_str_rl = np.array2string(g_rot_l["values"][:6], precision=2, separator=',')
                print(f"  {'':>7} | {'Rot(Locked)':>12} | {g_rot_l['scale']:>10.4f} | {g_rot_l['max_abs']:>10.4f} | {g_rot_l['mse']:>12.6f} | {vals_str_rl}")
            
            print(f"  {'─'*7}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*30}")
        
        ratio = rot["total_mse"] / (norot["total_mse"] + 1e-15)
        verdict = "�?WORSE" if ratio > 1 else "�?BETTER"
        locked_info = f"  Rot_Locked={rot_locked['total_mse']:.6f}" if rot_locked else ""
        print(f"  Total MSE:  NoRot={norot['total_mse']:.6f}  Rot={rot['total_mse']:.6f}{locked_info}  Ratio={ratio:.2f}× ({verdict})")

def compute_loss_zone_mse(values, quantized, scale):
    """Return per-zone squared-error contributions for the Figure 2b zones.

    Zones are defined on the normalized magnitude ``abs(x) / scale``:
      * medium: [2.25, 2.75], [3.25, 3.75], [4.25, 4.50], [5.50, 5.75]
      * high:   (4.50, 5.50)
      * low:    all remaining values

    ``total_mse`` is the mean over the complete 16-value group.  Zone values
    are sums of squared error so their percentages add up to 100%.
    """
    values = np.asarray(values, dtype=np.float64)
    quantized = np.asarray(quantized, dtype=np.float64)
    squared_error = (quantized - values) ** 2

    if scale > 0:
        normalized = np.abs(values) / float(scale)
    else:
        normalized = np.zeros_like(values)

    medium = (
        ((normalized >= 2.25) & (normalized <= 2.75))
        | ((normalized >= 3.25) & (normalized <= 3.75))
        | ((normalized >= 4.25) & (normalized <= 4.50))
        | ((normalized >= 5.50) & (normalized <= 5.75))
    )
    high = (normalized > 4.50) & (normalized < 5.50)
    low = ~(medium | high)

    contributions = {
        "low": float(squared_error[low].sum()),
        "medium": float(squared_error[medium].sum()),
        "high": float(squared_error[high].sum()),
    }
    total_sse = float(squared_error.sum())
    if total_sse > 0:
        percentages = {
            name: 100.0 * contribution / total_sse
            for name, contribution in contributions.items()
        }
    else:
        percentages = {name: 0.0 for name in contributions}

    return contributions, percentages, total_sse, float(squared_error.mean())


def print_loss_zone_mse(tag, condition, values, quantized, scale):
    contributions, percentages, total_sse, total_mse = compute_loss_zone_mse(
        values, quantized, scale
    )
    print(
        f"[SPECIAL_GROUP_ZONE_MSE] tag={tag} condition={condition} "
        f"group_mse={total_mse:.10f} total_sse={total_sse:.10f}"
    )
    print(
        "[SPECIAL_GROUP_ZONE_MSE] "
        f"low={percentages['low']:.4f}% (sse={contributions['low']:.10f}) | "
        f"medium={percentages['medium']:.4f}% (sse={contributions['medium']:.10f}) | "
        f"high={percentages['high']:.4f}% (sse={contributions['high']:.10f})"
    )


def print_special_group_values(layer_idx, block_idx, token_idx, tag, group_idx,
                               group_orig, group_rotated, results, force=False):
    """Print the two 16-value groups used by the summary figure.

    The diagnostic script selects the 16-channel group containing the largest
    original activation.  Keeping the printout here (rather than in the plot
    code) makes the exact floating-point values available for reproducing a
    paper figure on the remote machine.
    """
    if (not force and os.environ.get("PRINT_SPECIAL_GROUPS", "1").lower()
            in {"0", "false", "no"}):
        return

    def fmt(values):
        return np.array2string(
            np.asarray(values, dtype=np.float64),
            precision=10,
            separator=", ",
            max_line_width=240,
        )

    no = results["nvfp4_norot"]["groups"][group_idx]
    rot = results["nvfp4_rot"]["groups"][group_idx]
    locked = results["nvfp4_rot_locked"]["groups"][group_idx]

    print("\n" + "=" * 108)
    print(
        f"[SPECIAL_GROUP] layer={layer_idx} tag={tag} block128={block_idx} "
        f"token={token_idx} group16=G{group_idx}"
    )
    print(f"[SPECIAL_GROUP] before_rotation = {fmt(group_orig)}")
    print(f"[SPECIAL_GROUP] after_rotation  = {fmt(group_rotated)}")
    print(f"[SPECIAL_GROUP] q_norot         = {fmt(no['quantized'])}")
    print(f"[SPECIAL_GROUP] q_rot_unlocked  = {fmt(rot['quantized'])}")
    print(f"[SPECIAL_GROUP] q_rot_locked    = {fmt(locked['quantized'])}")
    print(
        f"[SPECIAL_GROUP] scale_norot={no['scale']:.10f} "
        f"mse_norot={no['mse']:.10f}"
    )
    print(
        f"[SPECIAL_GROUP] scale_rot_unlocked={rot['scale']:.10f} "
        f"mse_rot_unlocked={rot['mse']:.10f}"
    )
    print(
        f"[SPECIAL_GROUP] scale_rot_locked={locked['scale']:.10f} "
        f"mse_rot_locked={locked['mse']:.10f}"
    )
    print_loss_zone_mse(
        tag, "before_rotation", group_orig, no["quantized"], no["scale"]
    )
    print_loss_zone_mse(
        tag, "after_rotation", group_rotated, rot["quantized"], rot["scale"]
    )
    print("=" * 108)


def find_compressed_three_zone_worse_group(layer_idx, x_blocks,
                                           quantizer_nv, quantizer_mx,
                                           transform, device, max_block_tokens=8,
                                           min_zone_percent=3.0):
    """Find a range-compressed, post-MR-worse group with all three zones.

    We inspect only the top ``max_block_tokens`` (token, 128-channel-block)
    pairs by absolute activation.  Each pair has eight FP4 groups, so the
    search is bounded to 64 groups by default.  A valid illustration must:
    (1) shrink its 16-value range after MR, (2) increase group MSE, and
    (3) have visible low/medium/high post-MR error contributions.
    """
    pair_scores = x_blocks.abs().amax(dim=-1).flatten()
    top_pairs = torch.topk(pair_scores, k=min(max_block_tokens, pair_scores.numel())).indices.tolist()
    n_blocks = x_blocks.shape[1]
    scanned = 0
    eligible = []

    for pair_idx in top_pairs:
        token_idx = pair_idx // n_blocks
        block_idx = pair_idx % n_blocks
        results, orig, rotated = analyze_block(
            x_blocks[token_idx, block_idx, :], quantizer_nv, quantizer_mx,
            transform, device
        )
        for group_idx, (no, rot) in enumerate(zip(
                results["nvfp4_norot"]["groups"],
                results["nvfp4_rot"]["groups"],
        )):
            scanned += 1
            _, percentages, _, _ = compute_loss_zone_mse(
                rotated[group_idx * 16:(group_idx + 1) * 16],
                rot["quantized"],
                rot["scale"],
            )
            zone_floor = min(percentages.values())
            mse_increase = rot["mse"] - no["mse"]
            range_shrink = no["max_abs"] / max(rot["max_abs"], 1e-15)
            if (mse_increase <= 0 or zone_floor < min_zone_percent
                    or range_shrink <= 1.0):
                continue

            # Prefer a substantial MSE increase and clear three-zone bar,
            # while favouring a visibly compressed input range.  The cap keeps
            # one extreme range ratio from dominating all other evidence.
            score = mse_increase * (zone_floor / 100.0) * min(range_shrink, 10.0)
            eligible.append({
                "score": score,
                "token_idx": token_idx,
                "block_idx": block_idx,
                "group_idx": group_idx,
                "orig": orig,
                "rotated": rotated,
                "results": results,
                "mse_increase": mse_increase,
                "ratio": rot["mse"] / max(no["mse"], 1e-15),
                "range_shrink": range_shrink,
                "percentages": percentages,
            })

    print(
        f"[COMPRESSED_THREE_ZONE_SCAN] layer={layer_idx} "
        f"token_block_pairs={len(top_pairs)} groups_scanned={scanned} "
        f"min_zone_percent={min_zone_percent:.1f} eligible={len(eligible)}"
    )
    if not eligible:
        print("[COMPRESSED_THREE_ZONE_SCAN] No eligible group found; expand max_block_tokens or lower min_zone_percent.")
        return None

    best = max(eligible, key=lambda item: item["score"])
    print(
        f"[COMPRESSED_THREE_ZONE_CANDIDATE] token={best['token_idx']} "
        f"block128={best['block_idx']} group16=G{best['group_idx']} "
        f"range_shrink={best['range_shrink']:.4f}x "
        f"mse_increase={best['mse_increase']:.10f} "
        f"mse_ratio={best['ratio']:.4f} score={best['score']:.10f} "
        f"after_low={best['percentages']['low']:.2f}% "
        f"after_medium={best['percentages']['medium']:.2f}% "
        f"after_high={best['percentages']['high']:.2f}%"
    )
    return best


def find_stronger_figure_c_candidate(layer_idx, x_blocks, quantizer_nv,
                                     quantizer_mx, transform, device,
                                     max_block_tokens=12):
    """Find a more illustrative Figure-C counterexample in a bounded scan.

    The candidate must retain a *moderate* pre-MR outlier (rather than a
    single overwhelmingly dominant value), achieve visibly stronger range
    compression, become worse after rotation, and move most post-MR error
    into the medium/high-loss zones while retaining all three zones.
    """
    pair_scores = x_blocks.abs().amax(dim=-1).flatten()
    top_pairs = torch.topk(
        pair_scores, k=min(max_block_tokens, pair_scores.numel())
    ).indices.tolist()
    n_blocks = x_blocks.shape[1]
    scanned, eligible = 0, []

    for pair_idx in top_pairs:
        token_idx = pair_idx // n_blocks
        block_idx = pair_idx % n_blocks
        results, orig, rotated = analyze_block(
            x_blocks[token_idx, block_idx, :], quantizer_nv, quantizer_mx,
            transform, device
        )
        for group_idx, (no, rot) in enumerate(zip(
                results["nvfp4_norot"]["groups"],
                results["nvfp4_rot"]["groups"],
        )):
            scanned += 1
            start, end = group_idx * 16, (group_idx + 1) * 16
            orig_group = orig[start:end]
            rot_group = rotated[start:end]
            _, before_pct, _, _ = compute_loss_zone_mse(
                orig_group, no["quantized"], no["scale"]
            )
            _, after_pct, _, _ = compute_loss_zone_mse(
                rot_group, rot["quantized"], rot["scale"]
            )

            sorted_abs = np.sort(np.abs(orig_group))[::-1]
            outlier_ratio = float(sorted_abs[0] / max(sorted_abs[1], 1e-15))
            rot_abs = np.abs(rot_group)
            rot_max = float(rot_abs.max())
            small_fraction = float(np.mean(rot_abs <= 0.30 * rot_max))
            large_fraction = float(np.mean(rot_abs >= 0.65 * rot_max))
            mse_increase = rot["mse"] - no["mse"]
            range_shrink = no["max_abs"] / max(rot["max_abs"], 1e-15)
            medium_high = after_pct["medium"] + after_pct["high"]

            valid = (
                mse_increase > 0
                and range_shrink >= 2.0
                and 1.25 <= outlier_ratio <= 4.0
                and before_pct["low"] >= 70.0
                and min(after_pct.values()) >= 5.0
                and after_pct["medium"] >= 15.0
                and after_pct["high"] >= 20.0
                and medium_high >= 70.0
                and small_fraction >= 0.125
                and large_fraction >= 0.20
            )
            if not valid:
                continue

            score = (
                mse_increase * min(range_shrink, 8.0)
                * (medium_high / 100.0)
                * min(outlier_ratio, 2.5)
            )
            eligible.append({
                "score": score,
                "token_idx": token_idx,
                "block_idx": block_idx,
                "group_idx": group_idx,
                "orig": orig,
                "rotated": rotated,
                "results": results,
                "mse_increase": mse_increase,
                "ratio": rot["mse"] / max(no["mse"], 1e-15),
                "range_shrink": range_shrink,
                "outlier_ratio": outlier_ratio,
                "before_pct": before_pct,
                "after_pct": after_pct,
                "small_fraction": small_fraction,
                "large_fraction": large_fraction,
            })

    print(
        f"[STRONGER_FIGURE_C_SCAN] layer={layer_idx} "
        f"token_block_pairs={len(top_pairs)} groups_scanned={scanned} "
        f"eligible={len(eligible)}"
    )
    if not eligible:
        print(
            "[STRONGER_FIGURE_C_SCAN] No strict candidate found; "
            "the existing G2 remains the recommended counterexample."
        )
        return None

    ranked = sorted(eligible, key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranked[:5], start=1):
        pct = item["after_pct"]
        print(
            f"[STRONGER_FIGURE_C_TOP{rank}] token={item['token_idx']} "
            f"block128={item['block_idx']} group16=G{item['group_idx']} "
            f"shrink={item['range_shrink']:.3f}x "
            f"outlier_ratio={item['outlier_ratio']:.3f} "
            f"mse_ratio={item['ratio']:.3f} "
            f"after_low={pct['low']:.1f}% after_medium={pct['medium']:.1f}% "
            f"after_high={pct['high']:.1f}%"
        )
    return ranked[0]


def find_compressed_mse_better_group(layer_idx, x_blocks, quantizer_nv,
                                     quantizer_mx, transform, device):
    """Find a true-H16 group whose range and quantization MSE both decrease.

    Twelve representative token/block pairs are inspected (the four strongest
    plus eight amplitude quantiles), i.e. at most 96 FP4 groups.  This avoids
    an exhaustive activation sweep while looking beyond only extreme outliers.
    """
    pair_scores = x_blocks.abs().amax(dim=-1).flatten()
    ordered_pairs = torch.argsort(pair_scores, descending=True).tolist()
    n_pairs = len(ordered_pairs)
    ranks = list(range(min(4, n_pairs)))
    ranks.extend(round((n_pairs - 1) * fraction)
                 for fraction in (0.08, 0.20, 0.35, 0.50, 0.65, 0.80, 0.92, 0.98))
    pair_indices = [ordered_pairs[rank] for rank in dict.fromkeys(ranks)]
    n_blocks = x_blocks.shape[1]
    scanned, eligible = 0, []

    for pair_idx in pair_indices:
        token_idx, block_idx = divmod(pair_idx, n_blocks)
        results, orig, rotated = analyze_block(
            x_blocks[token_idx, block_idx, :], quantizer_nv, quantizer_mx,
            transform, device
        )
        for group_idx, (no, rot) in enumerate(zip(
                results["nvfp4_norot"]["groups"],
                results["nvfp4_rot"]["groups"],
        )):
            scanned += 1
            mse_reduction = no["mse"] - rot["mse"]
            range_shrink = no["max_abs"] / max(rot["max_abs"], 1e-15)
            if mse_reduction <= 0 or range_shrink <= 1.0 or no["mse"] < 1e-6:
                continue

            _, percentages, _, _ = compute_loss_zone_mse(
                rotated[group_idx * 16:(group_idx + 1) * 16],
                rot["quantized"], rot["scale"],
            )
            score = mse_reduction * min(range_shrink, 10.0)
            eligible.append({
                "score": score,
                "token_idx": token_idx,
                "block_idx": block_idx,
                "group_idx": group_idx,
                "orig": orig,
                "rotated": rotated,
                "results": results,
                "mse_reduction": mse_reduction,
                "ratio": rot["mse"] / max(no["mse"], 1e-15),
                "range_shrink": range_shrink,
                "percentages": percentages,
            })

    print(
        f"[COMPRESSED_MSE_BETTER_SCAN] layer={layer_idx} "
        f"token_block_pairs={len(pair_indices)} groups_scanned={scanned} "
        f"eligible={len(eligible)}"
    )
    if not eligible:
        print("[COMPRESSED_MSE_BETTER_SCAN] No eligible group found.")
        return None

    best = max(eligible, key=lambda item: item["score"])
    print(
        f"[COMPRESSED_MSE_BETTER_CANDIDATE] token={best['token_idx']} "
        f"block128={best['block_idx']} group16=G{best['group_idx']} "
        f"range_shrink={best['range_shrink']:.4f}x "
        f"mse_reduction={best['mse_reduction']:.10f} "
        f"mse_ratio={best['ratio']:.4f} "
        f"after_low={best['percentages']['low']:.2f}% "
        f"after_medium={best['percentages']['medium']:.2f}% "
        f"after_high={best['percentages']['high']:.2f}%"
    )
    return best


def plot_block(layer_idx, block_idx, token_idx, orig, rotated, results, output_path):
    """
    2×3 图：
    (0,0) NVFP4 NoRot  (0,1) NVFP4 Rot Unlocked  (0,2) NVFP4 Rot Locked
    (1,0) MXFP4 NoRot  (1,1) MXFP4 Rot           (1,2) Empty
    """
    fig, axes = plt.subplots(2, 3, figsize=(30, 10), sharex=True)
    
    group_colors_8 = plt.cm.Set2(np.linspace(0, 1, 8))
    group_colors_4 = plt.cm.Set1(np.linspace(0, 1, 4))
    
    configs = [
        (0, 0, "nvfp4_norot", "NVFP4 No Rotation", 16, group_colors_8, orig),
        (0, 1, "nvfp4_rot",   "NVFP4 Rotated (Unlocked)", 16, group_colors_8, rotated),
        (0, 2, "nvfp4_rot_locked", "NVFP4 Rotated (Locked)", 16, group_colors_8, rotated),
        (1, 0, "mxfp4_norot", "MXFP4 No Rotation", 32, group_colors_4, orig),
        (1, 1, "mxfp4_rot",   "MXFP4 Rotated",     32, group_colors_4, rotated),
    ]
    
    # Hide the empty subplot
    axes[1, 2].axis('off')
    
    for row, col, key, title, gs, colors, data in configs:
        ax = axes[row, col]
        res = results[key]
        n_groups = 128 // gs
        
        # 画每�?group �?bar (原始�?
        for g in range(n_groups):
            x_pos = np.arange(g*gs, (g+1)*gs)
            vals = data[g*gs:(g+1)*gs]
            q_vals = res["groups"][g]["quantized"]
            scale = res["groups"][g]["scale"]
            
            ax.bar(x_pos, vals, color=colors[g], alpha=0.5, width=1.0, label=f'G{g} s={scale:.3f}' if g < 4 else None)
            ax.scatter(x_pos, q_vals, color=colors[g], s=8, zorder=5, edgecolors='black', linewidths=0.3)
            
            # 画量化网格线 (只画正的)
            fp4_levels = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6]) * scale
            for lvl in fp4_levels:
                if lvl > 0 and lvl < ax.get_ylim()[1] * 1.2 if ax.get_ylim()[1] > 0 else True:
                    ax.hlines(lvl, g*gs-0.5, (g+1)*gs-0.5, colors=colors[g], alpha=0.2, linewidth=0.5, linestyles='--')
        
        mse = res["total_mse"]
        mse_max = res.get("mse_max", 0.0)
        mse_other = res.get("mse_other", 0.0)
        
        title_text = f"{title}\nTotal MSE = {mse:.6f}\nMax Pos MSE: {mse_max:.6f} | Others Mean MSE: {mse_other:.6f}"
        
        ax.set_title(title_text, fontsize=10)
        ax.set_ylabel("Value")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend(fontsize=6, ncol=4, loc='upper right')
        
        # 添加组分隔线
        for g in range(1, n_groups):
            ax.axvline(g*gs - 0.5, color='gray', linewidth=0.5, linestyle=':')
    
    axes[1, 0].set_xlabel("Channel Index (within 128-block)")
    axes[1, 1].set_xlabel("Channel Index (within 128-block)")
    axes[0, 2].set_xlabel("Channel Index (within 128-block)")
    
    fig.suptitle(f"Layer {layer_idx}, Block {block_idx}, Token {token_idx}\n"
                 f"Dots = Quantized Values | Bars = Original Values | Dashed = FP4 Grid",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Figure saved to {output_path}")

def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [0, 15, 31]
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model.eval()
    
    text = "Detailed shape analysis of LLM activations. " * 300
    calib_data = tokenizer(text, return_tensors="pt").to(device)
    activations = collect_activations(model, layers, calib_data, device)
    
    # 量化�?
    nv_q = Quantizer(bits=4, format="nvfp", granularity="group", group_size=16, symmetric=True, scale_precision="e4m3")
    mx_q = Quantizer(bits=4, format="mxfp", granularity="group", group_size=32, symmetric=True, scale_precision="e8m0")
    # Keep the established H128 path for micro-block and macro diagnostics.
    transform = build_transform("hadamard", size=128, group_size=128).to(device)
    # The all-summary figure is intentionally a true intra-group H16 example:
    # every FP4 quantization group is transformed independently by H16/sqrt(16).
    transform_summary_16 = build_transform("hadamard", size=16, group_size=16).to(device)
    
    # For Summary Plot
    all_nv_results = []

    for layer_idx in layers:
        X = activations[layer_idx]['X']  # (tokens, dim)
        dim = X.shape[-1]
        n_blocks = dim // 128
        
        # 找到含最�?Outlier �?block
        X_blocks = X[:, :n_blocks*128].reshape(X.shape[0], n_blocks, 128)
        block_max = X_blocks.abs().max(dim=-1).values.max(dim=0).values  # (n_blocks,)
        best_block = block_max.argmax().item()
        
        # 找到�?block �?outlier 最大的 token
        token_max = X_blocks[:, best_block, :].abs().max(dim=-1).values
        best_token = token_max.argmax().item()
        
        # 也选一�?普�? token (中位数附�?
        median_token = torch.argsort(token_max)[len(token_max)//2].item()
        
        for token_idx in [best_token, median_token]:
            block_vals = X_blocks[token_idx, best_block, :]  # (128,)
            # Existing block figures remain H128 diagnostics.
            results, orig, rotated = analyze_block(block_vals, nv_q, mx_q, transform, device)
            # print_report(layer_idx, best_block, token_idx, orig, rotated, results)
            
            tag = "outlier" if token_idx == best_token else "normal"
            if os.environ.get("SKIP_MICRO_PLOTS", "0").lower() not in {"1", "true", "yes"}:
                plot_block(layer_idx, best_block, token_idx, orig, rotated, results,
                           f"micro_block_L{layer_idx}_{tag}.png")

            # Only the all-summary figure and its printed representative-group
            # values use true independent H16 rotations.
            summary_results, summary_orig, summary_rotated = analyze_block(
                block_vals, nv_q, mx_q, transform_summary_16, device
            )
            
            # Find the group (16-size) that contains the max outlier
            group_idx = np.abs(summary_orig).argmax() // 16
            group_orig = summary_orig[group_idx*16 : (group_idx+1)*16]
            group_rotated = summary_rotated[group_idx*16 : (group_idx+1)*16]

            # The summary figure's two representative cases are the outlier
            # and median-token groups from the final inspected layer.  Print
            # their exact values so the paper illustration can use the same
            # data as this diagnostic run without dumping every layer/group.
            if layer_idx == layers[-1]:
                print_special_group_values(
                    layer_idx, best_block, token_idx, tag, group_idx,
                    group_orig, group_rotated, summary_results
                )
            
            # Compute group-level stats
            max_idx_grp = np.argmax(np.abs(group_orig))
            
            q_norot = summary_results['nvfp4_norot']['groups'][group_idx]['quantized']
            err2_norot = (q_norot - group_orig)**2
            norot_mse_max = err2_norot[max_idx_grp]
            norot_mse_other = (np.sum(err2_norot) - norot_mse_max) / 15.0
            
            q_rot = summary_results['nvfp4_rot']['groups'][group_idx]['quantized']
            err2_rot = (q_rot - group_rotated)**2
            rot_mse_max = err2_rot[max_idx_grp]
            rot_mse_other = (np.sum(err2_rot) - rot_mse_max) / 15.0
            
            q_rot_l = summary_results['nvfp4_rot_locked']['groups'][group_idx]['quantized']
            err2_rot_l = (q_rot_l - group_rotated)**2
            rot_l_mse_max = err2_rot_l[max_idx_grp]
            rot_l_mse_other = (np.sum(err2_rot_l) - rot_l_mse_max) / 15.0

            _, norot_zone_pct, _, _ = compute_loss_zone_mse(
                group_orig, q_norot,
                summary_results['nvfp4_norot']['groups'][group_idx]['scale'],
            )
            _, rot_zone_pct, _, _ = compute_loss_zone_mse(
                group_rotated, q_rot,
                summary_results['nvfp4_rot']['groups'][group_idx]['scale'],
            )
            
            # Save for summary plot (only the specific 16-channel group)
            all_nv_results.append({
                'layer': layer_idx,
                'tag': tag,
                'group_id': group_idx,
                'norot_mse': summary_results['nvfp4_norot']['groups'][group_idx]['mse'],
                'norot_mse_max': norot_mse_max,
                'norot_mse_other': norot_mse_other,
                'rot_mse': summary_results['nvfp4_rot']['groups'][group_idx]['mse'],
                'rot_mse_max': rot_mse_max,
                'rot_mse_other': rot_mse_other,
                'rot_locked_mse': summary_results['nvfp4_rot_locked']['groups'][group_idx]['mse'],
                'rot_locked_mse_max': rot_l_mse_max,
                'rot_locked_mse_other': rot_l_mse_other,
                'orig': group_orig,
                'rotated': group_rotated,
                'norot_quantized': q_norot,
                'norot_scale': summary_results['nvfp4_norot']['groups'][group_idx]['scale'],
                'rot_quantized': q_rot,
                'rot_scale': summary_results['nvfp4_rot']['groups'][group_idx]['scale'],
                'norot_zone_pct': norot_zone_pct,
                'rot_zone_pct': rot_zone_pct,
            })

    # A bounded, paper-figure-oriented search over the final layer's strongest
    # token/block pairs. It prints one replacement candidate but deliberately
    # leaves the existing summary figure unchanged until the candidate is
    # reviewed.
    if os.environ.get("SKIP_THREE_ZONE_SCAN", "0").lower() not in {"1", "true", "yes"}:
        three_zone_candidate = find_compressed_three_zone_worse_group(
            layer_idx, X_blocks, nv_q, mx_q, transform_summary_16, device
        )
        if three_zone_candidate is not None:
            candidate_group = three_zone_candidate["group_idx"]
            candidate_orig = three_zone_candidate["orig"][candidate_group * 16:(candidate_group + 1) * 16]
            candidate_rotated = three_zone_candidate["rotated"][candidate_group * 16:(candidate_group + 1) * 16]
            print_special_group_values(
                layer_idx, three_zone_candidate["block_idx"], three_zone_candidate["token_idx"],
                "compressed_three_zone_worse", candidate_group, candidate_orig,
                candidate_rotated, three_zone_candidate["results"],
            )
    else:
        print("[COMPRESSED_THREE_ZONE_SCAN] Skipped by request.")

    # Stricter Figure-C replacement search.  This is deliberately independent
    # of the original scan above, so it can be run by itself with
    # SKIP_THREE_ZONE_SCAN=1 and does not change any figure files.
    if os.environ.get("FIND_STRONGER_FIGURE_C_GROUP", "0").lower() in {"1", "true", "yes"}:
        max_pairs = max(1, int(os.environ.get("STRONGER_FIGURE_C_SCAN_PAIRS", "12")))
        stronger_candidate = find_stronger_figure_c_candidate(
            layer_idx, X_blocks, nv_q, mx_q, transform_summary_16, device,
            max_block_tokens=max_pairs,
        )
        if stronger_candidate is not None:
            candidate_group = stronger_candidate["group_idx"]
            candidate_orig = stronger_candidate["orig"][candidate_group * 16:(candidate_group + 1) * 16]
            candidate_rotated = stronger_candidate["rotated"][candidate_group * 16:(candidate_group + 1) * 16]
            print_special_group_values(
                layer_idx, stronger_candidate["block_idx"], stronger_candidate["token_idx"],
                "stronger_figure_c_worse", candidate_group, candidate_orig,
                candidate_rotated, stronger_candidate["results"], force=True,
            )

    if os.environ.get("FIND_MSE_BETTER_GROUP", "0").lower() in {"1", "true", "yes"}:
        better_candidate = find_compressed_mse_better_group(
            layer_idx, X_blocks, nv_q, mx_q, transform_summary_16, device
        )
        if better_candidate is not None:
            candidate_group = better_candidate["group_idx"]
            candidate_orig = better_candidate["orig"][candidate_group * 16:(candidate_group + 1) * 16]
            candidate_rotated = better_candidate["rotated"][candidate_group * 16:(candidate_group + 1) * 16]
            print_special_group_values(
                layer_idx, better_candidate["block_idx"], better_candidate["token_idx"],
                "compressed_mse_better", candidate_group, candidate_orig,
                candidate_rotated, better_candidate["results"], force=True,
            )

    if os.environ.get("STOP_AFTER_THREE_ZONE_SCAN", "0").lower() in {"1", "true", "yes"}:
        print("[COMPRESSED_THREE_ZONE_SCAN] Stopping after candidate scan by request.")
        return

    print_last_three_h16_groups(all_nv_results)

    if os.environ.get("SKIP_SUMMARY_PLOT", "0").lower() not in {"1", "true", "yes"}:
        plot_summary_comparison(all_nv_results, "nvfp4_all_layers_summary.png")
    else:
        print("[SUMMARY] Summary image generation skipped by request.")

    if os.environ.get("STOP_AFTER_SUMMARY", "0").lower() in {"1", "true", "yes"}:
        print("[SUMMARY] Stopping after nvfp4_all_layers_summary.png by request.")
        return
    
    # 宏观块级特征分析与散点图 (Phase 3 定量验证)
    analyze_macro_block_features(activations, nv_q, mx_q, transform, device)

def analyze_macro_block_features(activations, quantizer_nv, quantizer_mx, transform_128, device):
    """
    全矩阵宏观特征分析：
    同时分析 128-channel block 和 16-channel group。
    用象限散点图和Scale压缩图验证效果A（小弟获益）和效果B（老大吃亏）的普遍存在。
    """
    print("\n" + "="*80)
    print(" 🚀 RUNNING MACRO-DIAGNOSTIC STATISTICS (Decomposed A/B Effects)")
    print("="*80)
    
    from src.transforms.transforms import build_transform
    transform_16 = build_transform("hadamard", size=16, group_size=16).to(device)
    
    gs_nv = 16
    gs_mx = 32
    block_size_128 = 128
    block_size_16 = 16
    
    for layer_idx, data in activations.items():
        print(f"\n--- Layer {layer_idx} Macro Analysis ---")
        X = data['X'].to(device)
        n_tokens, dim = X.shape
        n_blocks_128 = (n_tokens * dim) // block_size_128
        n_groups_16 = (n_tokens * dim) // block_size_16
        
        def compute_macro_metrics(X, X_q_norot, scales_norot, X_rot, X_q_rot, scales_rot, gs, block_size):
            n_blocks = (n_tokens * dim) // block_size
            
            err2_norot = (X_q_norot - X).pow(2).view(n_blocks, block_size)
            mse_norot = err2_norot.mean(dim=1)
            
            err2_rot = (X_q_rot - X_rot).pow(2).view(n_blocks, block_size)
            mse_rot = err2_rot.mean(dim=1)
            
            delta_mse = (mse_rot - mse_norot).cpu().numpy()
            
            # Scale compression ratio (mean across sub-groups within the block)
            sr = (scales_rot / (scales_norot + 1e-15)).view(n_blocks, -1).mean(dim=1).cpu().numpy()
            
            # High Loss Ratio calculation
            def get_hlr(X_q, scales):
                X_q_reshaped = X_q.view(-1, gs)
                scales_reshaped = scales.view(-1, 1)
                val_normalized = (X_q_reshaped.abs() / (scales_reshaped + 1e-15))
                is_4 = (val_normalized - 4.0).abs() < 0.1
                is_6 = (val_normalized - 6.0).abs() < 0.1
                is_hl = is_4 | is_6
                # reshape back to blocks
                hl_per_block = is_hl.view(n_blocks, block_size).float().mean(dim=1)
                return hl_per_block.cpu().numpy()
                
            hlr_norot = get_hlr(X_q_norot, scales_norot)
            hlr_rot = get_hlr(X_q_rot, scales_rot)
            delta_hlr = hlr_rot - hlr_norot
            
            return sr, delta_hlr, delta_mse
        
        # ==========================================
        # 1. 128-Channel Block 分析 (MXFP4 & NVFP4 Had128)
        # ==========================================
        X_rot128 = transform_128(X)
        
        # MXFP4
        scales_mx, zeros_mx = quantizer_mx.get_quantization_params(X)
        X_q_mx = quantizer_mx(X, scales_mx, zeros_mx)
        scales_mx_rot128, zeros_mx_rot128 = quantizer_mx.get_quantization_params(X_rot128)
        X_q_mx_rot128 = quantizer_mx(X_rot128, scales_mx_rot128, zeros_mx_rot128)
        
        sr_mx128, d_hlr_mx128, d_mse_mx128 = compute_macro_metrics(
            X, X_q_mx, scales_mx, X_rot128, X_q_mx_rot128, scales_mx_rot128, gs_mx, block_size_128
        )
        
        # NVFP4 (with global scale lock for rot)
        scales_nv, zeros_nv = quantizer_nv.get_quantization_params(X)
        X_q_nv = quantizer_nv(X, scales_nv, zeros_nv)
        
        quantizer_nv._track_global_scale = False
        from src.quantization.quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX
        from src.quantization.quantizer import get_reciprocal
        act_max_val = X.abs().max().to(torch.float32).view(1)
        locked_scale = FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(act_max_val)
        quantizer_nv.global_scale = locked_scale.to(device)
        
        scales_nv_rot128, zeros_nv_rot128 = quantizer_nv.get_quantization_params(X_rot128)
        X_q_nv_rot128 = quantizer_nv(X_rot128, scales_nv_rot128, zeros_nv_rot128)
        
        sr_nv128, d_hlr_nv128, d_mse_nv128 = compute_macro_metrics(
            X, X_q_nv, scales_nv, X_rot128, X_q_nv_rot128, scales_nv_rot128, gs_nv, block_size_128
        )
        
        # ==========================================
        # 2. 16-Channel Group 分析 (NVFP4 Had16)
        # ==========================================
        X_rot16 = transform_16(X)
        scales_nv_rot16, zeros_nv_rot16 = quantizer_nv.get_quantization_params(X_rot16)
        X_q_nv_rot16 = quantizer_nv(X_rot16, scales_nv_rot16, zeros_nv_rot16)
        
        sr_nv16, d_hlr_nv16, d_mse_nv16 = compute_macro_metrics(
            X, X_q_nv, scales_nv, X_rot16, X_q_nv_rot16, scales_nv_rot16, gs_nv, block_size_16
        )
        
        quantizer_nv._track_global_scale = True
        
        # ==========================================
        # 3. Spatial Distribution Processing
        # ==========================================
        G_128 = dim // block_size_128
        G_16 = dim // block_size_16
        
        def compute_spatial_stats(delta_tot_np, G):
            matrix = delta_tot_np.reshape(n_tokens, G)
            prob_worse = (matrix > 0).sum(axis=0) / n_tokens * 100
            total_mse = matrix.sum(axis=0)
            return prob_worse, total_mse
            
        prob_worse_nv128, total_mse_nv128 = compute_spatial_stats(d_mse_nv128, G_128)
        prob_worse_mx128, total_mse_mx128 = compute_spatial_stats(d_mse_mx128, G_128)
        prob_worse_nv16, total_mse_nv16 = compute_spatial_stats(d_mse_nv16, G_16)
        
        # ==========================================
        # 4. Plotting New Mechanism Scatter and Spatial
        # ==========================================
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        
        def plot_mechanism_scatter(ax, sr, d_hlr, d_mse, title):
            # Define colormap: Green for negative (loss decrease), Red for positive (loss increase)
            import matplotlib.colors as mcolors
            max_abs = max(abs(d_mse.min()), abs(d_mse.max()), 1e-4)
            norm = mcolors.SymLogNorm(linthresh=1e-4, vmin=-max_abs, vmax=max_abs, base=10)
            scatter = ax.scatter(sr, d_hlr, c=d_mse, cmap='RdYlGn_r', alpha=0.6, s=10, norm=norm)
            
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Scale Ratio ($Scale_{Rot} / Scale_{NoRot}$)', fontweight='bold')
            ax.set_ylabel('$\Delta$ High-Loss Ratio (Rot - NoRot)', fontweight='bold')
            ax.set_xscale('log')
            ax.set_title(title, fontweight='bold')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('$\Delta$ Total MSE (Red = Worse, Green = Better)')
            
        def plot_spatial(ax, prob_worse, total_mse, G, title):
            ax.bar(range(G), prob_worse, color='salmon', width=1.0, alpha=0.8, label='Worsened (%)')
            ax.bar(range(G), 100 - prob_worse, bottom=prob_worse, color='mediumseagreen', width=1.0, alpha=0.8, label='Improved (%)')
            ax.axhline(50, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Channel Group Index', fontweight='bold')
            ax.set_ylabel('% of Tokens (Red=Worse, Green=Better)', fontweight='bold')
            ax.set_ylim(0, 100)
            ax.set_title(title, fontweight='bold')
            
            ax2 = ax.twinx()
            ax2.plot(range(G), total_mse, color='blue', alpha=0.7, linewidth=1.5, label='Total ΔMSE')
            ax2.set_yscale('symlog', linthresh=0.01)
            ax2.axhline(0, color='blue', linestyle=':', alpha=0.5)
            ax2.set_ylabel('Total $\Delta$MSE (symlog)', color='blue', fontweight='bold')
            
        plot_mechanism_scatter(axes[0, 0], sr_nv128, d_hlr_nv128, d_mse_nv128, 'NVFP4 (Had128) Mechanism Map')
        plot_mechanism_scatter(axes[0, 1], sr_mx128, d_hlr_mx128, d_mse_mx128, 'MXFP4 (Had128) Mechanism Map')
        plot_mechanism_scatter(axes[0, 2], sr_nv16, d_hlr_nv16, d_mse_nv16, 'NVFP4 (Had16) Mechanism Map')
        
        plot_spatial(axes[1, 0], prob_worse_nv128, total_mse_nv128, G_128, 'NVFP4 (Had128) Spatial Consistency')
        plot_spatial(axes[1, 1], prob_worse_mx128, total_mse_mx128, G_128, 'MXFP4 (Had128) Spatial Consistency')
        plot_spatial(axes[1, 2], prob_worse_nv16, total_mse_nv16, G_16, 'NVFP4 (Had16) Spatial Consistency')
        
        plt.suptitle(f'Layer {layer_idx}: Scale & High-Loss-Zone Impact on Total MSE', fontsize=18, fontweight='bold')
        plt.tight_layout()
        output_file = f"macro_scatter_L{layer_idx}.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  New scatter plot saved to {output_file}")
        
        del X, X_rot128, X_rot16
        del X_q_mx, X_q_mx_rot128
        del X_q_nv, X_q_nv_rot128, X_q_nv_rot16
        torch.cuda.empty_cache()

def print_last_three_h16_groups(all_results):
    """Print exact H16 values and loss-zone ratios for the final three rows."""
    def fmt(values):
        return np.array2string(np.asarray(values, dtype=np.float64), precision=10,
                               separator=", ", max_line_width=240)

    print("\n" + "=" * 108)
    print("[LAST_THREE_H16_GROUPS] Exact values and loss-zone MSE ratios")
    for group in all_results[-3:]:
        identity = f"layer={group['layer']} tag={group['tag']} group16=G{group['group_id']}"
        print(f"[LAST_THREE_H16_GROUP] {identity}")
        print(f"[LAST_THREE_H16_GROUP] before_rotation = {fmt(group['orig'])}")
        print(f"[LAST_THREE_H16_GROUP] q_before        = {fmt(group['norot_quantized'])}")
        print(
            f"[LAST_THREE_H16_GROUP] before_scale={group['norot_scale']:.10f} "
            f"before_mse={group['norot_mse']:.10f} | "
            f"low={group['norot_zone_pct']['low']:.4f}% "
            f"medium={group['norot_zone_pct']['medium']:.4f}% "
            f"high={group['norot_zone_pct']['high']:.4f}%"
        )
        print(f"[LAST_THREE_H16_GROUP] after_rotation  = {fmt(group['rotated'])}")
        print(f"[LAST_THREE_H16_GROUP] q_after         = {fmt(group['rot_quantized'])}")
        print(
            f"[LAST_THREE_H16_GROUP] after_scale={group['rot_scale']:.10f} "
            f"after_mse={group['rot_mse']:.10f} | "
            f"low={group['rot_zone_pct']['low']:.4f}% "
            f"medium={group['rot_zone_pct']['medium']:.4f}% "
            f"high={group['rot_zone_pct']['high']:.4f}%"
        )
        print("-" * 108)
    print("=" * 108)
    return

    # Retained below temporarily for source compatibility; unreachable.
    selected = all_results[-3:]
    zone_names = ["low", "medium", "high"]
    zone_labels = ["Low / safe", "Medium loss", "High loss"]
    zone_colors = ["#66b3ff", "#ffcc99", "#ff9999"]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    y_positions, labels, mse_values, pct_rows = [], [], [], []
    for group in selected:
        group_label = f"L{group['layer']} {group['tag']} G{group['group_id']}"
        for condition, mse_key, pct_key in [
            ("before MR", "norot_mse", "norot_zone_pct"),
            ("after MR", "rot_mse", "rot_zone_pct"),
        ]:
            labels.append(f"{group_label} — {condition}")
            mse_values.append(group[mse_key])
            pct_rows.append([group[pct_key][zone] for zone in zone_names])

    y_positions = np.arange(len(labels))[::-1]
    for row_idx, (y, percentages) in enumerate(zip(y_positions, pct_rows)):
        left = 0.0
        for percentage, color, label in zip(percentages, zone_colors, zone_labels):
            if percentage > 0:
                ax.barh(y, percentage, left=left, height=0.66,
                        color=color, edgecolor="white", linewidth=1.5,
                        label=None)
                if percentage >= 8:
                    ax.text(left + percentage / 2, y, f"{percentage:.1f}%",
                            ha="center", va="center", fontsize=10)
            left += percentage

        ax.text(102, y, f"MSE={mse_values[row_idx]:.5g}",
                va="center", fontsize=10)

    ax.set_xlim(0, 124)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Contribution to total squared error within the 16-value group")
    ax.set_title("H16 loss-zone MSE contribution: last three representative groups",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color)
                      for color in zone_colors]
    ax.legend(legend_handles, zone_labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print("\n[LAST_THREE_ZONE_BARS] Exact H16 loss-zone MSE contribution percentages")
    for label, mse, percentages in zip(labels, mse_values, pct_rows):
        print(
            f"[LAST_THREE_ZONE_BARS] {label} mse={mse:.10f} | "
            f"low={percentages[0]:.4f}% medium={percentages[1]:.4f}% "
            f"high={percentages[2]:.4f}%"
        )
    print(f"[LAST_THREE_ZONE_BARS] Figure saved to {output_path}")


def plot_summary_comparison(all_results, output_path):
    """
    汇总图片：对比所有层的 NVFP4 NoRot vs Rot (Unlocked) vs Rot (Locked)
    """
    n = len(all_results)
    fig, axes = plt.subplots(n, 3, figsize=(22, 4*n))
    
    for i, res in enumerate(all_results):
        layer = res['layer']
        tag = res['tag']
        gid = res['group_id']
        
        # NoRot Plot
        axes[i, 0].bar(range(16), res['orig'], color='skyblue', alpha=0.7)
        axes[i, 0].set_title(f"L{layer} ({tag}) G{gid} - H16 NoRot\nGrp MSE: {res['norot_mse']:.4f} | Max: {res['norot_mse_max']:.4f} | Oth: {res['norot_mse_other']:.4f}", fontsize=10)
        
        # Rot Plot (Unlocked)
        axes[i, 1].bar(range(16), res['rotated'], color='salmon', alpha=0.7)
        axes[i, 1].set_title(f"L{layer} ({tag}) G{gid} - H16 Rot (Unlck)\nGrp MSE: {res['rot_mse']:.4f} | Max: {res['rot_mse_max']:.4f} | Oth: {res['rot_mse_other']:.4f}", fontsize=10)
        
        # Rot Plot (Locked)
        axes[i, 2].bar(range(16), res['rotated'], color='mediumpurple', alpha=0.7)
        axes[i, 2].set_title(f"L{layer} ({tag}) G{gid} - H16 Rot (Lck)\nGrp MSE: {res['rot_locked_mse']:.4f} | Max: {res['rot_locked_mse_max']:.4f} | Oth: {res['rot_locked_mse_other']:.4f}", fontsize=10)
        
        for j in range(3):
            axes[i, j].axhline(0, color='black', linewidth=0.5)
            axes[i, j].set_xticks(range(16))
            axes[i, j].tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Summary figure saved to {output_path}")

if __name__ == "__main__":
    main()
