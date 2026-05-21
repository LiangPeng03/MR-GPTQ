"""
MinMax 首尾配对通道重排序诊断分析
=================================
目的: 精确定位 MinMax 重排序为何在大模型上未能改善 PPL
分析维度:
  1. 逐层激活值量化 MSE (原始顺序 vs MinMax)
  2. 逐层权重量化 MSE (原始顺序 vs MinMax)
  3. 组级分解: Outlier组改善 vs 中间组恶化
  4. 组内 max/second 比率分布变化
"""
import os, sys, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.quantization.quantizer import Quantizer
from src.transforms.transforms import build_transform

# ======================== 核心算法 ========================

def compute_nvfp4_scale_first_perm(act_stat, w_norm, group_size=16):
    w_norm = w_norm.to(act_stat.device)
    N = act_stat.shape[0]
    n_groups = max(1, N // group_size)
    score = act_stat * w_norm
    score_sorted_idx = torch.argsort(score, descending=True)
    anchors = score_sorted_idx[:n_groups].tolist()
    groups = [[a] for a in anchors]
    group_lens = torch.ones(n_groups, dtype=torch.long, device=act_stat.device)
    FP4_GRID = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=act_stat.device)
    remaining_mask = torch.ones(N, dtype=torch.bool, device=act_stat.device)
    remaining_mask[anchors] = False
    remaining_idx = remaining_mask.nonzero(as_tuple=True)[0]
    remaining_score = score[remaining_idx]
    rem_sorted_idx = remaining_idx[torch.argsort(remaining_score, descending=True)]
    group_scales = torch.tensor([act_stat[a] / 6.0 + 1e-12 for a in anchors], device=act_stat.device)
    group_w_sums = torch.tensor([w_norm[a] for a in anchors], device=act_stat.device)
    for ch_idx in rem_sorted_idx.tolist():
        val = act_stat[ch_idx]
        w = w_norm[ch_idx]
        new_scales = torch.maximum(group_scales, torch.tensor(val / 6.0 + 1e-12, device=act_stat.device))
        normalized = val / new_scales
        distances = (normalized.unsqueeze(1) - FP4_GRID.unsqueeze(0)).abs()
        min_errs = distances.min(dim=1).values * new_scales
        damage = min_errs * w
        scale_diff = new_scales - group_scales
        damage += scale_diff * group_w_sums
        damage[group_lens >= group_size] = float('inf')
        best_g_idx = damage.argmin().item()
        groups[best_g_idx].append(ch_idx)
        group_lens[best_g_idx] += 1
        group_scales[best_g_idx] = new_scales[best_g_idx]
        group_w_sums[best_g_idx] += w
    perm = []
    for g in groups:
        perm.extend(g)
    return torch.tensor(perm, device=act_stat.device, dtype=torch.long)


def quantize_and_mse(x, quantizer, device):
    """
    对输入 x (1, dim) 量化并返回逐组 MSE
    返回: total_mse, per_group_mse_list, per_group_scale_list
    """
    x_dev = x.to(device)
    scales, zeros = quantizer.get_quantization_params(x_dev)
    x_q = quantizer(x_dev, scales, zeros)
    
    gs = quantizer.group_size
    n_groups = x.shape[-1] // gs
    
    per_group_mse = []
    per_group_scale = []
    for g in range(n_groups):
        orig_g = x[0, g*gs:(g+1)*gs].float()
        quant_g = x_q[0, g*gs:(g+1)*gs].float().cpu()
        mse = (orig_g - quant_g).pow(2).mean().item()
        per_group_mse.append(mse)
        
    scales_flat = scales.flatten().cpu().float()
    for g in range(n_groups):
        per_group_scale.append(scales_flat[g].item() if g < len(scales_flat) else 0)
    
    total_mse = np.mean(per_group_mse)
    return total_mse, per_group_mse, per_group_scale


# ======================== 数据收集 ========================

def collect_all_layer_data(model, tokenizer, device, n_layers):
    """收集所有层 down_proj 的激活值和权重"""
    text = "The large language model architecture involves sophisticated attention mechanisms. " * 300
    calib_data = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    activations = {}
    weights = {}
    
    for idx in range(n_layers):
        layer = model.model.layers[idx].mlp.down_proj
        weights[idx] = layer.weight.data.cpu().float()  # (out_dim, in_dim)
        
        cache = []
        def hook(m, i, o, c=cache): c.append(i[0].detach().cpu())
        handle = layer.register_forward_hook(hook)
        with torch.no_grad():
            model(**calib_data)
        handle.remove()
        # (tokens, dim)
        activations[idx] = cache[0].view(-1, cache[0].shape[-1]).float()
        torch.cuda.empty_cache()
    
    return activations, weights


# ======================== 逐层分析 ========================

def analyze_layer(layer_idx, act, weight, quantizer, device, group_size=16):
    """
    对一层做完整的 MinMax 诊断
    返回 dict:
      act_mse_orig, act_mse_minmax: 激活值总MSE
      wt_mse_orig, wt_mse_minmax: 权重总MSE
      per_group_act_orig, per_group_act_minmax: 逐组激活值MSE
      per_group_wt_orig, per_group_wt_minmax: 逐组权重MSE
      n_outlier_groups: Outlier组数
      perm: 排列
    """
    dim = act.shape[-1]
    
    # --- 1. 计算 P95 统计量 ---
    p95 = torch.quantile(act.abs(), 0.95, dim=0)
    w_norm = weight.float().norm(p=2, dim=0).to(p95.device)
    
    # --- 2. 计算 MinMax 排列 ---
    perm = compute_nvfp4_scale_first_perm(p95, w_norm, group_size)
    
    # --- 3. 激活值量化对比 ---
    # 取多行token做平均MSE（取前100行避免太慢）
    n_tokens = min(100, act.shape[0])
    
    act_mse_orig_list = []
    act_mse_minmax_list = []
    act_per_group_orig_all = []
    act_per_group_minmax_all = []
    
    for t in range(n_tokens):
        x_orig = act[t:t+1, :]  # (1, dim)
        x_minmax = x_orig[:, perm]
        
        mse_o, pg_o, _ = quantize_and_mse(x_orig, quantizer, device)
        mse_m, pg_m, _ = quantize_and_mse(x_minmax, quantizer, device)
        
        act_mse_orig_list.append(mse_o)
        act_mse_minmax_list.append(mse_m)
        act_per_group_orig_all.append(pg_o)
        act_per_group_minmax_all.append(pg_m)
    
    act_mse_orig = np.mean(act_mse_orig_list)
    act_mse_minmax = np.mean(act_mse_minmax_list)
    per_group_act_orig = np.mean(act_per_group_orig_all, axis=0)
    per_group_act_minmax = np.mean(act_per_group_minmax_all, axis=0)
    
    # --- 4. 权重量化对比 ---
    # 权重是 (out, in)，排列作用于 in 维度（列）
    # 取前 64 行做权重量化对比
    n_rows = min(64, weight.shape[0])
    wt_mse_orig_list = []
    wt_mse_minmax_list = []
    wt_per_group_orig_all = []
    wt_per_group_minmax_all = []
    
    for r in range(n_rows):
        w_orig = weight[r:r+1, :]  # (1, in_dim)
        w_minmax = w_orig[:, perm]
        
        mse_o, pg_o, _ = quantize_and_mse(w_orig, quantizer, device)
        mse_m, pg_m, _ = quantize_and_mse(w_minmax, quantizer, device)
        
        wt_mse_orig_list.append(mse_o)
        wt_mse_minmax_list.append(mse_m)
        wt_per_group_orig_all.append(pg_o)
        wt_per_group_minmax_all.append(pg_m)
    
    wt_mse_orig = np.mean(wt_mse_orig_list)
    wt_mse_minmax = np.mean(wt_mse_minmax_list)
    per_group_wt_orig = np.mean(wt_per_group_orig_all, axis=0)
    per_group_wt_minmax = np.mean(wt_per_group_minmax_all, axis=0)
    
    # --- 5. Outlier 组统计 ---
    sorted_p95 = p95[perm]
    groups = sorted_p95.view(-1, group_size)
    group_max = groups.max(dim=-1).values
    group_second = groups.topk(2, dim=-1).values[:, 1]
    median_max = group_max.median().item()
    n_outlier_groups = (group_max > 2 * median_max).sum().item()
    
    # 组内 max/second ratio
    ratio_orig_groups = p95.view(-1, group_size)
    ratio_orig = (ratio_orig_groups.max(dim=-1).values / (ratio_orig_groups.topk(2, dim=-1).values[:, 1] + 1e-9)).mean().item()
    ratio_minmax_groups = sorted_p95.view(-1, group_size)
    ratio_minmax = (ratio_minmax_groups.max(dim=-1).values / (ratio_minmax_groups.topk(2, dim=-1).values[:, 1] + 1e-9)).mean().item()
    
    return {
        'act_mse_orig': act_mse_orig,
        'act_mse_minmax': act_mse_minmax,
        'wt_mse_orig': wt_mse_orig,
        'wt_mse_minmax': wt_mse_minmax,
        'per_group_act_orig': per_group_act_orig,
        'per_group_act_minmax': per_group_act_minmax,
        'per_group_wt_orig': per_group_wt_orig,
        'per_group_wt_minmax': per_group_wt_minmax,
        'n_outlier_groups': n_outlier_groups,
        'n_total_groups': dim // group_size,
        'ratio_orig': ratio_orig,
        'ratio_minmax': ratio_minmax,
        'perm': perm,
    }


# ======================== 可视化 ========================

def plot_results(all_results, n_layers, output_prefix):
    """生成 3 张诊断图"""
    
    layers = list(range(n_layers))
    
    # ===== 图1: 逐层激活值 MSE =====
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    act_orig = [all_results[l]['act_mse_orig'] for l in layers]
    act_mm = [all_results[l]['act_mse_minmax'] for l in layers]
    
    ax = axes[0]
    x = np.arange(n_layers)
    w = 0.35
    ax.bar(x - w/2, act_orig, w, label='Original Order', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, act_mm, w, label='MinMax Order', color='coral', alpha=0.8)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Mean Activation MSE')
    ax.set_title('Activation Quantization MSE: Original vs MinMax (per layer)')
    ax.legend()
    ax.set_xticks(x)
    
    # 变化百分比
    act_change = [(m - o) / (o + 1e-15) * 100 for o, m in zip(act_orig, act_mm)]
    ax2 = axes[1]
    colors = ['green' if c < 0 else 'red' for c in act_change]
    ax2.bar(x, act_change, color=colors, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('MSE Change (%)')
    ax2.set_title('Activation MSE Change (negative = improved)')
    ax2.set_xticks(x)
    
    fig.tight_layout()
    fig.savefig(f'{output_prefix}_act_mse.png', dpi=150)
    print(f"  Saved: {output_prefix}_act_mse.png")
    plt.close(fig)
    
    # ===== 图2: 逐层权重 MSE =====
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    wt_orig = [all_results[l]['wt_mse_orig'] for l in layers]
    wt_mm = [all_results[l]['wt_mse_minmax'] for l in layers]
    
    ax = axes[0]
    ax.bar(x - w/2, wt_orig, w, label='Original Order', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, wt_mm, w, label='MinMax Order', color='coral', alpha=0.8)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Mean Weight MSE')
    ax.set_title('Weight Quantization MSE: Original vs MinMax (per layer)')
    ax.legend()
    ax.set_xticks(x)
    
    wt_change = [(m - o) / (o + 1e-15) * 100 for o, m in zip(wt_orig, wt_mm)]
    ax2 = axes[1]
    colors = ['green' if c < 0 else 'red' for c in wt_change]
    ax2.bar(x, wt_change, color=colors, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('MSE Change (%)')
    ax2.set_title('Weight MSE Change (negative = improved)')
    ax2.set_xticks(x)
    
    fig.tight_layout()
    fig.savefig(f'{output_prefix}_wt_mse.png', dpi=150)
    print(f"  Saved: {output_prefix}_wt_mse.png")
    plt.close(fig)
    
    # ===== 图3: 组级分解 (选 3 个代表层) =====
    repr_layers = [0, n_layers // 2, n_layers - 1]
    fig, axes = plt.subplots(len(repr_layers), 2, figsize=(20, 5 * len(repr_layers)))
    
    for i, l in enumerate(repr_layers):
        res = all_results[l]
        n_g = res['n_total_groups']
        
        # 激活值逐组对比
        ax = axes[i, 0]
        ax.plot(range(n_g), res['per_group_act_orig'], 'b-', alpha=0.5, label='Original', linewidth=0.8)
        ax.plot(range(n_g), res['per_group_act_minmax'], 'r-', alpha=0.5, label='MinMax', linewidth=0.8)
        ax.set_title(f'Layer {l}: Activation Per-Group MSE\n'
                     f'Outlier groups: {res["n_outlier_groups"]}/{n_g}  '
                     f'Ratio orig/mm: {res["ratio_orig"]:.1f}/{res["ratio_minmax"]:.1f}')
        ax.set_xlabel('Group Index (sorted by anchor magnitude in MinMax)')
        ax.set_ylabel('Group MSE')
        ax.legend(fontsize=8)
        
        # 权重逐组对比
        ax = axes[i, 1]
        ax.plot(range(n_g), res['per_group_wt_orig'], 'b-', alpha=0.5, label='Original', linewidth=0.8)
        ax.plot(range(n_g), res['per_group_wt_minmax'], 'r-', alpha=0.5, label='MinMax', linewidth=0.8)
        ax.set_title(f'Layer {l}: Weight Per-Group MSE')
        ax.set_xlabel('Group Index')
        ax.set_ylabel('Group MSE')
        ax.legend(fontsize=8)
    
    fig.tight_layout()
    fig.savefig(f'{output_prefix}_group_decompose.png', dpi=150)
    print(f"  Saved: {output_prefix}_group_decompose.png")
    plt.close(fig)


def print_summary(all_results, n_layers):
    """打印汇总表格"""
    print(f"\n{'='*100}")
    print(f"{'Layer':>6} | {'Act MSE Orig':>14} | {'Act MSE MM':>14} | {'Act Δ%':>8} | "
          f"{'Wt MSE Orig':>14} | {'Wt MSE MM':>14} | {'Wt Δ%':>8} | "
          f"{'Outliers':>8} | {'R_orig':>7} | {'R_mm':>7}")
    print(f"{'─'*6}─┼─{'─'*14}─┼─{'─'*14}─┼─{'─'*8}─┼─"
          f"{'─'*14}─┼─{'─'*14}─┼─{'─'*8}─┼─"
          f"{'─'*8}─┼─{'─'*7}─┼─{'─'*7}")
    
    total_act_improve = 0
    total_wt_change = 0
    
    for l in range(n_layers):
        r = all_results[l]
        act_chg = (r['act_mse_minmax'] - r['act_mse_orig']) / (r['act_mse_orig'] + 1e-15) * 100
        wt_chg = (r['wt_mse_minmax'] - r['wt_mse_orig']) / (r['wt_mse_orig'] + 1e-15) * 100
        total_act_improve += act_chg
        total_wt_change += wt_chg
        
        act_marker = "✓" if act_chg < -1 else ("✗" if act_chg > 1 else "≈")
        wt_marker = "✓" if wt_chg < -1 else ("✗" if wt_chg > 1 else "≈")
        
        print(f"  L{l:>3} | {r['act_mse_orig']:>14.8f} | {r['act_mse_minmax']:>14.8f} | {act_chg:>+7.2f}%{act_marker} | "
              f"{r['wt_mse_orig']:>14.8f} | {r['wt_mse_minmax']:>14.8f} | {wt_chg:>+7.2f}%{wt_marker} | "
              f"{r['n_outlier_groups']:>4}/{r['n_total_groups']:<3} | {r['ratio_orig']:>7.1f} | {r['ratio_minmax']:>7.1f}")
    
    print(f"\n  Average Act MSE Change: {total_act_improve/n_layers:+.2f}%")
    print(f"  Average Wt MSE Change:  {total_wt_change/n_layers:+.2f}%")


# ======================== Main ========================

def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    group_size = 16  # NVFP4
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} layers, hidden_size={model.config.hidden_size}")
    
    # 量化器 (NVFP4)
    quantizer = Quantizer(
        bits=4, format="nvfp", granularity="group", group_size=group_size,
        symmetric=True, scale_precision="e4m3"
    )
    
    print("Collecting activations and weights from all layers...")
    activations, weights = collect_all_layer_data(model, tokenizer, device, n_layers)
    
    # 释放模型显存
    del model
    torch.cuda.empty_cache()
    
    print("\nAnalyzing each layer...")
    all_results = {}
    for l in range(n_layers):
        print(f"  Layer {l}/{n_layers-1}...", end=" ", flush=True)
        result = analyze_layer(l, activations[l], weights[l], quantizer, device, group_size)
        all_results[l] = result
        
        act_chg = (result['act_mse_minmax'] - result['act_mse_orig']) / (result['act_mse_orig'] + 1e-15) * 100
        wt_chg = (result['wt_mse_minmax'] - result['wt_mse_orig']) / (result['wt_mse_orig'] + 1e-15) * 100
        print(f"Act: {act_chg:+.2f}%  Wt: {wt_chg:+.2f}%  Outliers: {result['n_outlier_groups']}/{result['n_total_groups']}")
    
    print_summary(all_results, n_layers)
    plot_results(all_results, n_layers, "minmax_diag")
    print("\nDone! Check minmax_diag_*.png files.")


if __name__ == "__main__":
    main()
