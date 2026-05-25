"""
微观 Block 级别量化分析：
对同一个 128-channel Block，分别用 NVFP4(8组×16) 和 MXFP4(4组×32) 量化，
对比旋转前后的具体数值、scale、量化值和误差。
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
    """对一个 group 的值进行量化，返回量化后的值和 scale"""
    x = values.to(device).unsqueeze(0)  # (1, group_size)
    scales, zeros = quantizer.get_quantization_params(x)
    x_q = quantizer(x, scales, zeros)
    return x_q.squeeze(0).cpu(), scales.squeeze().cpu().item()

def analyze_block(block_values, quantizer_nv, quantizer_mx, transform, device):
    """
    block_values: (128,) 一个 token 的 128 个通道值
    返回 NVFP4 和 MXFP4 在旋转前后的详细量化结果
    """
    block = block_values.clone()
    
    # --- 旋转 ---
    block_rot = transform(block.unsqueeze(0).to(device)).squeeze(0).cpu()
    
    results = {}
    for label, fmt_q, gs, name in [
        ("nvfp4", quantizer_nv, 16, "NVFP4 (8 groups × 16)"),
        ("mxfp4", quantizer_mx, 32, "MXFP4 (4 groups × 32)")
    ]:
        n_groups = 128 // gs
        for rot_label, data in [("norot", block), ("rot", block_rot)]:
            groups_info = []
            for g in range(n_groups):
                vals = data[g*gs : (g+1)*gs].float()
                q_vals, scale = quantize_group(vals, fmt_q, device)
                mse = (q_vals - vals).pow(2).mean().item()
                groups_info.append({
                    "values": vals.numpy(),
                    "quantized": q_vals.numpy(),
                    "scale": scale,
                    "mse": mse,
                    "max_abs": vals.abs().max().item()
                })
            total_mse = np.mean([g["mse"] for g in groups_info])
            results[f"{label}_{rot_label}"] = {
                "groups": groups_info,
                "total_mse": total_mse,
                "name": name
            }
    
    return results, block.numpy(), block_rot.numpy()

def print_report(layer_idx, block_idx, token_idx, orig, rotated, results):
    """打印详细的数值报告"""
    print(f"\n{'='*80}")
    print(f"  Layer {layer_idx}, Block {block_idx} (channels {block_idx*128}-{(block_idx+1)*128-1}), Token {token_idx}")
    print(f"{'='*80}")
    
    # 原始值摘要
    max_pos = np.argmax(np.abs(orig))
    print(f"\n  原始值 (128 channels), Max |值| = {orig[max_pos]:.4f} @ position {max_pos}")
    print(f"  前16: {np.array2string(orig[:16], precision=3, separator=', ')}")
    
    # 旋转后值摘要
    print(f"\n  旋转后 (Hadamard-128), Max |值| = {np.max(np.abs(rotated)):.4f}")
    print(f"  前16: {np.array2string(rotated[:16], precision=3, separator=', ')}")
    
    for fmt in ["nvfp4", "mxfp4"]:
        norot = results[f"{fmt}_norot"]
        rot = results[f"{fmt}_rot"]
        n_groups = len(norot["groups"])
        gs = 128 // n_groups
        
        print(f"\n{'─'*80}")
        print(f"  {norot['name']}")
        print(f"{'─'*80}")
        print(f"  {'Group':>7} | {'Condition':>8} | {'Scale':>10} | {'Max|Val|':>10} | {'Group MSE':>12} | 前6个值")
        print(f"  {'─'*7}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*30}")
        
        for g in range(n_groups):
            g_norot = norot["groups"][g]
            g_rot = rot["groups"][g]
            
            # 标记含 Outlier 的组
            is_outlier = g_norot["max_abs"] > 2 * np.median([gi["max_abs"] for gi in norot["groups"]])
            marker = " ★" if is_outlier else ""
            
            vals_str = np.array2string(g_norot["values"][:6], precision=2, separator=',')
            print(f"  G{g:>5} | {'NoRot':>8} | {g_norot['scale']:>10.4f} | {g_norot['max_abs']:>10.4f} | {g_norot['mse']:>12.6f} | {vals_str}{marker}")
            
            vals_str_r = np.array2string(g_rot["values"][:6], precision=2, separator=',')
            print(f"  {'':>7} | {'Rot':>8} | {g_rot['scale']:>10.4f} | {g_rot['max_abs']:>10.4f} | {g_rot['mse']:>12.6f} | {vals_str_r}")
            print(f"  {'─'*7}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*30}")
        
        ratio = rot["total_mse"] / (norot["total_mse"] + 1e-15)
        verdict = "↑ WORSE" if ratio > 1 else "↓ BETTER"
        print(f"  Total MSE:  NoRot={norot['total_mse']:.6f}  Rot={rot['total_mse']:.6f}  Ratio={ratio:.2f}× ({verdict})")

def plot_block(layer_idx, block_idx, token_idx, orig, rotated, results, output_path):
    """
    2×2 图：
    (0,0) NVFP4 NoRot  (0,1) NVFP4 Rot
    (1,0) MXFP4 NoRot  (1,1) MXFP4 Rot
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
    
    group_colors_8 = plt.cm.Set2(np.linspace(0, 1, 8))
    group_colors_4 = plt.cm.Set1(np.linspace(0, 1, 4))
    
    configs = [
        (0, 0, "nvfp4_norot", "NVFP4 No Rotation", 16, group_colors_8, orig),
        (0, 1, "nvfp4_rot",   "NVFP4 Rotated",     16, group_colors_8, rotated),
        (1, 0, "mxfp4_norot", "MXFP4 No Rotation", 32, group_colors_4, orig),
        (1, 1, "mxfp4_rot",   "MXFP4 Rotated",     32, group_colors_4, rotated),
    ]
    
    for row, col, key, title, gs, colors, data in configs:
        ax = axes[row, col]
        res = results[key]
        n_groups = 128 // gs
        
        # 画每个 group 的 bar (原始值)
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
        ax.set_title(f"{title}\nMean Block MSE = {mse:.6f}", fontsize=12)
        ax.set_ylabel("Value")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend(fontsize=6, ncol=4, loc='upper right')
        
        # 添加组分隔线
        for g in range(1, n_groups):
            ax.axvline(g*gs - 0.5, color='gray', linewidth=0.5, linestyle=':')
    
    axes[1, 0].set_xlabel("Channel Index (within 128-block)")
    axes[1, 1].set_xlabel("Channel Index (within 128-block)")
    
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
    
    # 量化器
    nv_q = Quantizer(bits=4, format="nvfp", granularity="group", group_size=16, symmetric=True, scale_precision="e4m3")
    mx_q = Quantizer(bits=4, format="mxfp", granularity="group", group_size=32, symmetric=True, scale_precision="e8m0")
    transform = build_transform("hadamard", size=128, group_size=128).to(device)
    
    # For Summary Plot
    all_nv_results = []

    for layer_idx in layers:
        X = activations[layer_idx]['X']  # (tokens, dim)
        dim = X.shape[-1]
        n_blocks = dim // 128
        
        # 找到含最大 Outlier 的 block
        X_blocks = X[:, :n_blocks*128].reshape(X.shape[0], n_blocks, 128)
        block_max = X_blocks.abs().max(dim=-1).values.max(dim=0).values  # (n_blocks,)
        best_block = block_max.argmax().item()
        
        # 找到该 block 中 outlier 最大的 token
        token_max = X_blocks[:, best_block, :].abs().max(dim=-1).values
        best_token = token_max.argmax().item()
        
        # 也选一个"普通" token (中位数附近)
        median_token = torch.argsort(token_max)[len(token_max)//2].item()
        
        for token_idx in [best_token, median_token]:
            block_vals = X_blocks[token_idx, best_block, :]  # (128,)
            results, orig, rotated = analyze_block(block_vals, nv_q, mx_q, transform, device)
            print_report(layer_idx, best_block, token_idx, orig, rotated, results)
            
            tag = "outlier" if token_idx == best_token else "normal"
            plot_block(layer_idx, best_block, token_idx, orig, rotated, results,
                       f"micro_block_L{layer_idx}_{tag}.png")
            
            # Find the group (16-size) that contains the max outlier
            group_idx = np.abs(orig).argmax() // 16
            group_orig = orig[group_idx*16 : (group_idx+1)*16]
            group_rotated = rotated[group_idx*16 : (group_idx+1)*16]
            
            # Save for summary plot (only the specific 16-channel group)
            all_nv_results.append({
                'layer': layer_idx,
                'tag': tag,
                'group_id': group_idx,
                'norot_mse': results['nvfp4_norot']['groups'][group_idx]['mse'],
                'rot_mse': results['nvfp4_rot']['groups'][group_idx]['mse'],
                'orig': group_orig,
                'rotated': group_rotated
            })

    # 生成汇总图片
    plot_summary_comparison(all_nv_results, "nvfp4_all_layers_summary.png")
    
    # 宏观全矩阵统计
    run_macro_statistics(activations, nv_q, transform, device)
    
    # 生成宏观图表（图表一和图表三）
    plot_macro_visualizations(activations, nv_q, device)

def plot_summary_comparison(all_results, output_path):
    """
    汇总图片：对比所有层的 NVFP4 NoRot vs Rot 性能
    """
    n = len(all_results)
    fig, axes = plt.subplots(n, 2, figsize=(15, 4*n))
    
    for i, res in enumerate(all_results):
        layer = res['layer']
        tag = res['tag']
        gid = res['group_id']
        
        # NoRot Plot
        axes[i, 0].bar(range(16), res['orig'], color='skyblue', alpha=0.7)
        axes[i, 0].set_title(f"L{layer} ({tag}) G{gid} - NoRot\nMSE: {res['norot_mse']:.6f}")
        
        # Rot Plot
        axes[i, 1].bar(range(16), res['rotated'], color='salmon', alpha=0.7)
        axes[i, 1].set_title(f"L{layer} ({tag}) G{gid} - Rotated\nMSE: {res['rot_mse']:.6f}")
        
        for j in range(2):
            axes[i, j].axhline(0, color='black', linewidth=0.5)
            axes[i, j].set_xticks(range(16))
            axes[i, j].tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Summary figure saved to {output_path}")


def run_macro_statistics(activations, quantizer_nv, transform, device):
    """
    全矩阵宏观诊断：定量分析各种 Outlier 组的出现频率和损失贡献。
    重点评估 NVFP4 (group=16) 及其在 Hadamard 旋转前后的表现。
    """
    print("\n" + "="*80)
    print(" 🚀 RUNNING MACRO-DIAGNOSTIC STATISTICS (NVFP4 Output MSE Analysis)")
    print("="*80)
    
    gs = 16 # NVFP4 group size
    
    for layer_idx, data in activations.items():
        X = data['X']
        W = data['W']
        
        out_dim = W.shape[0]
        n_tokens = X.shape[0]
        n_groups_per_token = X.shape[1] // gs
        
        print(f"\n--- Layer {layer_idx} Macro Analysis ---")
        
        # 1. 计算 Spikiness (尖刺度) = 组内 max(|x|) / mean(|x|)
        X_groups = X.view(-1, gs)
        max_abs = X_groups.abs().max(dim=1).values
        mean_abs = X_groups.abs().mean(dim=1)
        spikiness = max_abs / (mean_abs + 1e-9)
        
        # 2. 计算量化 MSE
        X_dev = X.to(device)
        scales, zeros = quantizer_nv.get_quantization_params(X_dev)
        X_q = quantizer_nv(X_dev, scales, zeros)
        
        err_sq_norot = (X_q - X_dev).pow(2).view(-1, gs).cpu()
        
        # --- 计算期望输出级损失 (Expected Output MSE Contribution) ---
        # 提取权重的通道级强度，映射到输出空间
        W_g = W.view(out_dim, n_groups_per_token, gs).permute(1, 2, 0)
        W_norm2 = W_g.pow(2).sum(dim=2) # (G, gs)
        W_norm2_expanded = W_norm2.unsqueeze(0).expand(n_tokens, -1, -1).reshape(-1, gs)
        
        err_sq_out = err_sq_norot * W_norm2_expanded / out_dim
        output_mse_groups = err_sq_out.sum(dim=1) # (T*G,)
        
        # --- 组内误差归因 (Intra-Group Attribution based on Grid Levels) ---
        extreme_mask = spikiness >= 8
        mean_err_pct = None
        mean_cnt = None
        if extreme_mask.sum() > 0:
            X_q_ext = X_q.view(-1, gs)[extreme_mask].cpu()
            scales_ext = scales.view(-1, 1)[extreme_mask].cpu()
            err_ext = err_sq_out[extreme_mask]
            
            levels = X_q_ext.abs() / (scales_ext + 1e-15)
            
            mask_6x = levels >= 5.5
            mask_4x = (levels > 3.5) & (levels < 4.5)
            mask_05x = (levels > 0.25) & (levels < 0.75)
            mask_0x = levels < 0.25
            mask_rest = ~(mask_6x | mask_4x | mask_05x | mask_0x)
            
            total_err = err_ext.sum(dim=1).clamp(min=1e-15)
            
            pct_6x = ((err_ext * mask_6x).sum(dim=1) / total_err * 100).mean().item()
            pct_4x = ((err_ext * mask_4x).sum(dim=1) / total_err * 100).mean().item()
            pct_05x = ((err_ext * mask_05x).sum(dim=1) / total_err * 100).mean().item()
            pct_0x = ((err_ext * mask_0x).sum(dim=1) / total_err * 100).mean().item()
            pct_rest = ((err_ext * mask_rest).sum(dim=1) / total_err * 100).mean().item()
            
            mean_err_pct = [pct_6x, pct_4x, pct_05x, pct_0x, pct_rest]
            
            cnt_6x = mask_6x.sum(dim=1).float().mean().item()
            cnt_4x = mask_4x.sum(dim=1).float().mean().item()
            cnt_05x = mask_05x.sum(dim=1).float().mean().item()
            cnt_0x = mask_0x.sum(dim=1).float().mean().item()
            cnt_rest = mask_rest.sum(dim=1).float().mean().item()
            
            mean_cnt = [cnt_6x, cnt_4x, cnt_05x, cnt_0x, cnt_rest]
        
        # --- 计算旋转后的期望输出级损失 (Rotated Output MSE) ---
        X_rot = transform(X_dev)
        scales_r, zeros_r = quantizer_nv.get_quantization_params(X_rot)
        X_q_rot = quantizer_nv(X_rot, scales_r, zeros_r)
        X_q_orig_space = transform(X_q_rot) # Hadamard is symmetric self-inverse
        
        err_sq_rot = (X_q_orig_space - X_dev).pow(2).view(-1, gs).cpu()
        err_sq_out_rot = err_sq_rot * W_norm2_expanded / out_dim
        output_mse_rot_groups = err_sq_out_rot.sum(dim=1)
        
        # 释放显存
        del X_dev, X_q, X_rot, X_q_rot, X_q_orig_space
        torch.cuda.empty_cache()
        
        total_output_mse = output_mse_groups.sum().item()
        
        # 3. 按 Block 的最大尖刺度进行分桶统计 (128-Block Binning)
        output_mse_blocks = output_mse_groups.view(-1, 8).sum(dim=1)
        output_mse_rot_blocks = output_mse_rot_groups.view(-1, 8).sum(dim=1)
        
        spikiness_matrix = spikiness.view(-1, 8)
        block_spikiness_max = spikiness_matrix.max(dim=1).values
        
        bins = [1, 2, 4, 8, 10, 12, 14, 16, float('inf')]
        bin_names = ["1-2 (Smooth)", "2-4 (Mild)", "4-8 (Spiky)", "8-10 (High)", "10-12 (Extreme)", "12-14 (Severe)", "14-16 (Isolated)", "16+ (Catastrophic)"]
        
        print(f"\n  [128-Block Level Diagnostic (8 groups per block)]")
        print(f"{'Block Max Spikiness':>19} | {'Freq (%)':>8} | {'Loss Contrib(%)':>15} | {'Rot vs NoRot Ratio':>18} | Group Profile (Avg out of 8)")
        print("-" * 115)
        
        for i in range(len(bins)-1):
            low, high = bins[i], bins[i+1]
            mask = (block_spikiness_max >= low) & (block_spikiness_max < high)
            count = mask.sum().item()
            
            freq_pct = count / len(block_spikiness_max) * 100
            if count > 0:
                bin_mse_sum = output_mse_blocks[mask].sum().item()
                bin_mse_rot_sum = output_mse_rot_blocks[mask].sum().item()
                loss_contrib_pct = bin_mse_sum / total_output_mse * 100
                rot_ratio = bin_mse_rot_sum / (bin_mse_sum + 1e-15)
                
                # Profile of the 8 groups within these blocks
                spiky_in_bin = spikiness_matrix[mask] # (count, 8)
                c_smooth = (spiky_in_bin < 4).sum(dim=1).float().mean().item()
                c_spiky = ((spiky_in_bin >= 4) & (spiky_in_bin < 8)).sum(dim=1).float().mean().item()
                c_ext = (spiky_in_bin >= 8).sum(dim=1).float().mean().item()
                profile_str = f"{c_smooth:.1f} Smooth (<4), {c_spiky:.1f} Spiky (4-8), {c_ext:.1f} Extreme (>=8)"
            else:
                loss_contrib_pct = 0.0
                rot_ratio = 0.0
                profile_str = "N/A"
                
            print(f"{bin_names[i]:>19} | {freq_pct:>7.2f}% | {loss_contrib_pct:>14.2f}% | {rot_ratio:>17.2f}x | {profile_str}")
            
        if mean_err_pct is not None:
            print(f"\n  [Intra-Group Attribution for Extreme Groups (Spikiness >= 8)]")
            print(f"    Quantized to 6x scale           : {mean_err_pct[0]:>5.2f}% of Group OUTPUT MSE  (avg {mean_cnt[0]:.1f} elements)")
            print(f"    Quantized to 4x scale           : {mean_err_pct[1]:>5.2f}% of Group OUTPUT MSE  (avg {mean_cnt[1]:.1f} elements)")
            print(f"    Quantized to 0.5x scale         : {mean_err_pct[2]:>5.2f}% of Group OUTPUT MSE  (avg {mean_cnt[2]:.1f} elements)")
            print(f"    Quantized to 0x (Underflow)     : {mean_err_pct[3]:>5.2f}% of Group OUTPUT MSE  (avg {mean_cnt[3]:.1f} elements)")
            print(f"    Quantized to Others (1x-3x)     : {mean_err_pct[4]:>5.2f}% of Group OUTPUT MSE  (avg {mean_cnt[4]:.1f} elements)")


def plot_macro_visualizations(activations, quantizer_nv, device):
    """
    生成大一统双轴图表（图表一）和图表三（Scale分布对比）
    """
    print("\n" + "="*80)
    print(" 🎨 GENERATING COMPREHENSIVE MACRO VISUALIZATIONS")
    print("="*80)
    
    gs = 16
    bins = [1, 2, 4, 8, 10, 12, 14, 16, float('inf')]
    bin_names = ["1-2 (Smooth)", "2-4 (Mild)", "4-8 (Spiky)", "8-10 (High)", "10-12 (Extreme)", "12-14 (Severe)", "14-16 (Isolated)", "16+ (Catastrophic)"]
    short_bin_names = ["1-2", "2-4", "4-8", "8-10", "10-12", "12-14", "14-16", "16+"]
    
    for layer_idx, data in activations.items():
        print(f"  Generating plots for Layer {layer_idx}...")
        X = data['X']
        W = data['W']
        out_dim = W.shape[0]
        n_tokens = X.shape[0]
        n_groups_per_token = X.shape[1] // gs
        
        X_groups = X.view(-1, gs)
        max_abs = X_groups.abs().max(dim=1).values
        mean_abs = X_groups.abs().mean(dim=1)
        spikiness = max_abs / (mean_abs + 1e-9)
        
        X_dev = X.to(device)
        scales_norot, zeros_norot = quantizer_nv.get_quantization_params(X_dev)
        X_q = quantizer_nv(X_dev, scales_norot, zeros_norot)
        
        # --- 计算 Output MSE ---
        err_sq_norot = (X_q - X_dev).pow(2).view(-1, gs).cpu()
        W_g = W.view(out_dim, n_groups_per_token, gs).permute(1, 2, 0)
        W_norm2 = W_g.pow(2).sum(dim=2)
        W_norm2_expanded = W_norm2.unsqueeze(0).expand(n_tokens, -1, -1).reshape(-1, gs)
        
        err_sq_out = err_sq_norot * W_norm2_expanded / out_dim
        total_output_mse = err_sq_out.sum().item() + 1e-15
        
        # --- Chart 1: Comprehensive Dual-Axis Stacked Bar Chart ---
        levels_all = X_q.abs().view(-1, gs).cpu() / (scales_norot.view(-1, 1).cpu() + 1e-15)
        
        avg_mse_6x = []
        avg_mse_4x = []
        avg_mse_05x = []
        avg_mse_0x = []
        avg_mse_rest = []
        
        counts_6x = []
        counts_4x = []
        counts_05x = []
        counts_0x = []
        counts_rest = []
        
        freq_list = []
        valid_bins = []
        
        for i in range(len(bins)-1):
            mask = (spikiness >= bins[i]) & (spikiness < bins[i+1])
            if mask.sum() > 0:
                bin_err = err_sq_out[mask]
                bin_levels = levels_all[mask]
                
                n_groups = mask.sum().item()
                freq_list.append(n_groups / len(spikiness) * 100)
                
                # We now plot TOTAL OUTPUT MSE CONTRIBUTION (%) per bin
                err_6x = (bin_err * (bin_levels >= 5.5)).sum().item() / total_output_mse * 100
                err_4x = (bin_err * ((bin_levels > 3.5) & (bin_levels < 4.5))).sum().item() / total_output_mse * 100
                err_05x = (bin_err * ((bin_levels > 0.25) & (bin_levels < 0.75))).sum().item() / total_output_mse * 100
                err_0x = (bin_err * (bin_levels < 0.25)).sum().item() / total_output_mse * 100
                
                mask_rest = ~( (bin_levels >= 5.5) | ((bin_levels > 3.5) & (bin_levels < 4.5)) | ((bin_levels > 0.25) & (bin_levels < 0.75)) | (bin_levels < 0.25) )
                err_rest = (bin_err * mask_rest).sum().item() / total_output_mse * 100
                
                # 计算组内平均元素个数 (Average counts of elements per group in this bin)
                c_6x = (bin_levels >= 5.5).sum().item() / n_groups
                c_4x = ((bin_levels > 3.5) & (bin_levels < 4.5)).sum().item() / n_groups
                c_05x = ((bin_levels > 0.25) & (bin_levels < 0.75)).sum().item() / n_groups
                c_0x = (bin_levels < 0.25).sum().item() / n_groups
                c_rest = mask_rest.sum().item() / n_groups
                
                avg_mse_6x.append(err_6x)
                avg_mse_4x.append(err_4x)
                avg_mse_05x.append(err_05x)
                avg_mse_0x.append(err_0x)
                avg_mse_rest.append(err_rest)
                
                counts_6x.append(c_6x)
                counts_4x.append(c_4x)
                counts_05x.append(c_05x)
                counts_0x.append(c_0x)
                counts_rest.append(c_rest)
                
                valid_bins.append(short_bin_names[i])
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        x = np.arange(len(valid_bins))
        width = 0.6
        
        # Primary Y-axis: Stacked Bars for Output MSE Contribution %
        b1 = ax1.bar(x, avg_mse_6x, width, color='#2ca02c', label='Quantized to 6x (Outlier)')
        b2 = ax1.bar(x, avg_mse_4x, width, bottom=avg_mse_6x, color='#bcbd22', label='Quantized to 4x')
        
        bottom_rest = np.array(avg_mse_6x)+np.array(avg_mse_4x)
        b_rest = ax1.bar(x, avg_mse_rest, width, bottom=bottom_rest, color='#ff7f0e', label='Quantized to Others (1x-3x)')
        
        bottom_05x = bottom_rest+np.array(avg_mse_rest)
        b05 = ax1.bar(x, avg_mse_05x, width, bottom=bottom_05x, color='#9467bd', label='Quantized to 0.5x')
        
        bottom_0x = bottom_05x+np.array(avg_mse_05x)
        b0 = ax1.bar(x, avg_mse_0x, width, bottom=bottom_0x, color='#d62728', label='Quantized to 0 (Underflow Massacre)')
        
        # Add Text Labels inside bars for element counts
        # This solves the "is it 1 element or many?" question instantly
        def add_labels(bars, counts, threshold=1.0):
            for i, rect in enumerate(bars):
                height = rect.get_height()
                if height > threshold: # Only label if bar is tall enough to fit text
                    count_val = counts[i]
                    if count_val >= 0.1: # Only label if there is a meaningful count
                        ax1.text(rect.get_x() + rect.get_width()/2., 
                                 rect.get_y() + height/2.,
                                 f"n={count_val:.1f}",
                                 ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        
        # Threshold for adding text (e.g. at least 2% contribution height to fit text)
        add_labels(b1, counts_6x, threshold=2.0)
        add_labels(b2, counts_4x, threshold=2.0)
        add_labels(b_rest, counts_rest, threshold=2.0)
        add_labels(b05, counts_05x, threshold=2.0)
        add_labels(b0, counts_0x, threshold=2.0)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_bins, rotation=45, fontsize=10)
        ax1.set_ylabel('Total Output MSE Contribution (%)', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Spikiness Bin (Max / Mean)', fontweight='bold', fontsize=12)
        
        # Secondary Y-axis: Line plot for Frequency
        ax2 = ax1.twinx()
        ax2.plot(x, freq_list, color='black', marker='o', linewidth=2.5, markersize=8, linestyle='-', label='Frequency of Groups (%)')
        ax2.set_ylabel('Frequency of Groups (%)', color='black', fontweight='bold', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, max(max(freq_list)*1.2, 10)) # Ensure line doesn't cover top bars completely
        
        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=10)
        
        plt.title(f'Layer {layer_idx}: Comprehensive Output Error Diagnostic (Total = 100%)', pad=50, fontsize=15, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        plt.savefig(f"macro_comprehensive_L{layer_idx}.png", dpi=200, bbox_inches='tight')
        plt.close()
        
        # --- Chart 3: Scale Distribution ---
        scales_orig = scales_norot.view(-1).cpu().numpy()
        channel_max = X.abs().max(dim=0).values
        clustered_perm = torch.argsort(channel_max, descending=True)
        
        X_clustered = X[:, clustered_perm].to(device)
        scales_clustered, _ = quantizer_nv.get_quantization_params(X_clustered)
        scales_clustered = scales_clustered.view(-1).cpu().numpy()
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(scales_orig, bins=50, color='gray', alpha=0.7)
        plt.title(f'Layer {layer_idx}: Original Scales\n(No Reordering)')
        plt.xlabel('Group Scale')
        plt.ylabel('Frequency (Log Scale)')
        plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.subplot(1, 2, 2)
        plt.hist(scales_clustered, bins=50, color='green', alpha=0.7)
        plt.title(f'Layer {layer_idx}: Clustered Scales\n(Magnitude Clustering Reordering)')
        plt.xlabel('Group Scale')
        plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.suptitle(f"Layer {layer_idx}: Paradigm Shift - Rescuing the Scales", fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"macro_scale_distribution_L{layer_idx}.png", dpi=150)
        plt.close()
        
        del X_dev, X_q, X_clustered
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
