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
    transform = build_transform("hadamard", size=128, group_size=128).to(device)
    
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
            results, orig, rotated = analyze_block(block_vals, nv_q, mx_q, transform, device)
            # print_report(layer_idx, best_block, token_idx, orig, rotated, results)
            
            tag = "outlier" if token_idx == best_token else "normal"
            plot_block(layer_idx, best_block, token_idx, orig, rotated, results,
                       f"micro_block_L{layer_idx}_{tag}.png")
            
            # Find the group (16-size) that contains the max outlier
            group_idx = np.abs(orig).argmax() // 16
            group_orig = orig[group_idx*16 : (group_idx+1)*16]
            group_rotated = rotated[group_idx*16 : (group_idx+1)*16]
            
            # Compute group-level stats
            max_idx_grp = np.argmax(np.abs(group_orig))
            
            q_norot = results['nvfp4_norot']['groups'][group_idx]['quantized']
            err2_norot = (q_norot - group_orig)**2
            norot_mse_max = err2_norot[max_idx_grp]
            norot_mse_other = (np.sum(err2_norot) - norot_mse_max) / 15.0
            
            q_rot = results['nvfp4_rot']['groups'][group_idx]['quantized']
            err2_rot = (q_rot - group_rotated)**2
            rot_mse_max = err2_rot[max_idx_grp]
            rot_mse_other = (np.sum(err2_rot) - rot_mse_max) / 15.0
            
            q_rot_l = results['nvfp4_rot_locked']['groups'][group_idx]['quantized']
            err2_rot_l = (q_rot_l - group_rotated)**2
            rot_l_mse_max = err2_rot_l[max_idx_grp]
            rot_l_mse_other = (np.sum(err2_rot_l) - rot_l_mse_max) / 15.0
            
            # Save for summary plot (only the specific 16-channel group)
            all_nv_results.append({
                'layer': layer_idx,
                'tag': tag,
                'group_id': group_idx,
                'norot_mse': results['nvfp4_norot']['groups'][group_idx]['mse'],
                'norot_mse_max': norot_mse_max,
                'norot_mse_other': norot_mse_other,
                'rot_mse': results['nvfp4_rot']['groups'][group_idx]['mse'],
                'rot_mse_max': rot_mse_max,
                'rot_mse_other': rot_mse_other,
                'rot_locked_mse': results['nvfp4_rot_locked']['groups'][group_idx]['mse'],
                'rot_locked_mse_max': rot_l_mse_max,
                'rot_locked_mse_other': rot_l_mse_other,
                'orig': group_orig,
                'rotated': group_rotated
            })

    plot_summary_comparison(all_nv_results, "nvfp4_all_layers_summary.png")
    
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
        axes[i, 0].set_title(f"L{layer} ({tag}) G{gid} - NoRot\nGrp MSE: {res['norot_mse']:.4f} | Max: {res['norot_mse_max']:.4f} | Oth: {res['norot_mse_other']:.4f}", fontsize=10)
        
        # Rot Plot (Unlocked)
        axes[i, 1].bar(range(16), res['rotated'], color='salmon', alpha=0.7)
        axes[i, 1].set_title(f"L{layer} ({tag}) G{gid} - Rot (Unlck)\nGrp MSE: {res['rot_mse']:.4f} | Max: {res['rot_mse_max']:.4f} | Oth: {res['rot_mse_other']:.4f}", fontsize=10)
        
        # Rot Plot (Locked)
        axes[i, 2].bar(range(16), res['rotated'], color='mediumpurple', alpha=0.7)
        axes[i, 2].set_title(f"L{layer} ({tag}) G{gid} - Rot (Lck)\nGrp MSE: {res['rot_locked_mse']:.4f} | Max: {res['rot_locked_mse_max']:.4f} | Oth: {res['rot_locked_mse_other']:.4f}", fontsize=10)
        
        for j in range(3):
            axes[i, j].axhline(0, color='black', linewidth=0.5)
            axes[i, j].set_xticks(range(16))
            axes[i, j].tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Summary figure saved to {output_path}")

if __name__ == "__main__":
    main()
