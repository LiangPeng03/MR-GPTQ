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
        activations[idx] = cache[0].view(-1, cache[0].shape[-1]).float()
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
        X = activations[layer_idx]  # (tokens, dim)
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


if __name__ == "__main__":
    main()
