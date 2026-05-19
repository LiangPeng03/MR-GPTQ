"""
MXFP4 MinMax & Hadamard Unified Diagnostic Analysis
===================================================
目的: 精确定位 MinMax 重排序在 MXFP4 格式下 (Hadamard size 128, quantization group size 32)
为何在大模型上未能改善 PPL，并全面剖析其在旋转前后的误差传递特征。

分析维度:
  1. 逐层激活值量化 MSE & 输出空间有效误差 ||W @ delta||² & Hessian 加权误差 (原始 vs MinMax)
  2. 逐层权重量化 MSE (原始 vs MinMax)
  3. 组级分解: 旋转空间内 32-group Quantization MSE 与 scale 分布
  4. 块级分析: 128-size Hadamard 块内的 Outlier 数量统计 (证明 Outlier 分散效果)
  5. 重要性分桶: Top 10%, Mid 40%, Bottom 50% 通道误差的消长情况
"""
import os, sys, torch, numpy as np, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.quantization.quantizer import Quantizer
from fast_hadamard_transform import hadamard_transform

# ======================== 核心算法 ========================

def compute_truncated_minmax_perm(act_stat, group_size=128, outlier_ratio=0.005):
    """Truncated MinMax Strategy"""
    N = act_stat.shape[0]
    sorted_idx = torch.argsort(act_stat, descending=True)
    
    n_outliers = int(N * outlier_ratio)
    head, tail = 0, N - 1
    perm = []
    
    for _ in range(n_outliers):
        if head >= tail: break
        group = [sorted_idx[head].item()]
        head += 1
        for _ in range(group_size - 1):
            if head > tail: break
            group.append(sorted_idx[tail].item())
            tail -= 1
        perm.extend(group)
        
    remaining = [sorted_idx[i].item() for i in range(head, tail + 1)]
    remaining = sorted(remaining)
    
    perm.extend(remaining)
    return torch.tensor(perm, device=act_stat.device, dtype=torch.long)


def apply_hadamard_128(x, dim=-1):
    """在 128 大小的 Block 上应用 Fast Hadamard Transform"""
    orig_shape = x.shape
    device = x.device
    dtype = x.dtype
    
    # 确保在指定的维度上大小可以被 128 整除
    size = x.shape[dim]
    assert size % 128 == 0, f"Dimension size {size} is not divisible by 128."
    
    # 将指定维度移到最后
    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1)
    
    flat_shape = list(x.shape[:-1]) + [-1, 128]
    x_reshaped = x.reshape(-1, 128)
    
    scale = 1.0 / math.sqrt(128)
    x_rot = hadamard_transform(x_reshaped, scale=scale)
    
    x_out = x_rot.view(x.shape)
    if dim != -1 and dim != x.ndim - 1:
        x_out = x_out.transpose(dim, -1)
        
    return x_out.reshape(orig_shape).to(device=device, dtype=dtype)


def quantize_and_mse_mxfp4(x_rot, quantizer, device):
    """
    对已旋转空间内的 x_rot (1, dim) 进行 MXFP4 量化
    返回: total_mse, per_group_mse_list, per_group_scale_list
    """
    x_rot_dev = x_rot.to(device)
    scales, zeros = quantizer.get_quantization_params(x_rot_dev)
    x_rot_q = quantizer(x_rot_dev, scales, zeros)
    
    gs = quantizer.group_size  # 32
    n_groups = x_rot_dev.shape[-1] // gs
    
    per_group_mse = []
    per_group_scale = []
    for g in range(n_groups):
        orig_g = x_rot_dev[0, g*gs:(g+1)*gs].float()
        quant_g = x_rot_q[0, g*gs:(g+1)*gs].float()
        mse = (orig_g - quant_g).pow(2).mean().item()
        per_group_mse.append(mse)
        
    scales_flat = scales.flatten().cpu().float()
    for g in range(n_groups):
        per_group_scale.append(scales_flat[g].item() if g < len(scales_flat) else 0)
    
    total_mse = np.mean(per_group_mse)
    return total_mse, per_group_mse, per_group_scale, x_rot_q.cpu()


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

def analyze_layer(layer_idx, act, weight, quantizer, device):
    """
    针对 MXFP4 对一层进行完整的 MinMax & Hadamard 诊断
    """
    dim = act.shape[-1]
    n_tokens = min(50, act.shape[0])
    
    # 近似 Hessian 对角线
    hessian_diag = (act[:n_tokens] ** 2).mean(dim=0)
    
    # 1. 计算 P95 统计量并获得 Truncated MinMax 排列 (outlier_ratio=0.005)
    p95 = torch.quantile(act.abs(), 0.95, dim=0)
    perm = compute_truncated_minmax_perm(p95, group_size=128, outlier_ratio=0.001)
    inv_perm = torch.argsort(perm)
    
    # 权重子矩阵，用于快速评估输出空间误差
    n_out = min(128, weight.shape[0])
    W = weight[:n_out, :].float().to(device)  # (n_out, dim)
    
    # 区分通道重要性
    importance = hessian_diag * p95 ** 2
    imp_sorted = torch.argsort(importance, descending=True)
    top10_mask = torch.zeros(dim, dtype=torch.bool)
    top10_mask[imp_sorted[:dim // 10]] = True
    mid_mask = torch.zeros(dim, dtype=torch.bool)
    mid_mask[imp_sorted[dim // 10: dim // 2]] = True
    bottom_mask = ~top10_mask & ~mid_mask
    
    # 激活值量化容器
    act_metrics = {
        'mse_orig': [], 'mse_mm': [],
        'output_mse_orig': [], 'output_mse_mm': [],
        'hessian_err_orig': [], 'hessian_err_mm': [],
        'hessian_err_top10_orig': [], 'hessian_err_top10_mm': [],
        'hessian_err_mid_orig': [], 'hessian_err_mid_mm': [],
        'hessian_err_bottom_orig': [], 'hessian_err_bottom_mm': [],
    }
    
    per_group_act_orig_all = []
    per_group_act_mm_all = []
    per_group_scale_orig_all = []
    per_group_scale_mm_all = []
    
    # ---- 激活值逐行量化与误差评估 ----
    for t in range(n_tokens):
        x = act[t:t+1, :].to(device)  # (1, dim)
        
        # A. Baseline: Hadamard 128 Only
        x_rot = apply_hadamard_128(x)
        _, pg_mse_o, pg_scale_o, x_rot_q = quantize_and_mse_mxfp4(x_rot, quantizer, device)
        x_q_orig = apply_hadamard_128(x_rot_q.to(device))
        delta_orig = (x_q_orig - x).squeeze(0).cpu().float()
        
        # B. MinMax: Reordering + Hadamard 128
        x_perm = x[:, perm]
        x_perm_rot = apply_hadamard_128(x_perm)
        _, pg_mse_m, pg_scale_m, x_perm_rot_q = quantize_and_mse_mxfp4(x_perm_rot, quantizer, device)
        x_perm_q = apply_hadamard_128(x_perm_rot_q.to(device))
        x_q_mm = x_perm_q[:, inv_perm]
        delta_mm = (x_q_mm - x).squeeze(0).cpu().float()
        
        # 保存 rotated 空间内 32-group 的统计
        per_group_act_orig_all.append(pg_mse_o)
        per_group_act_mm_all.append(pg_mse_m)
        per_group_scale_orig_all.append(pg_scale_o)
        per_group_scale_mm_all.append(pg_scale_m)
        
        # 计算各种激活误差指标
        mse_orig = delta_orig.pow(2).mean().item()
        mse_mm = delta_mm.pow(2).mean().item()
        
        y_err_orig = (W @ delta_orig.to(device)).pow(2).mean().item()
        y_err_mm = (W @ delta_mm.to(device)).pow(2).mean().item()
        
        h_err_orig = (hessian_diag * delta_orig.pow(2)).sum().item()
        h_err_mm = (hessian_diag * delta_mm.pow(2)).sum().item()
        
        h_top10_orig = (hessian_diag[top10_mask] * delta_orig[top10_mask].pow(2)).sum().item()
        h_top10_mm = (hessian_diag[top10_mask] * delta_mm[top10_mask].pow(2)).sum().item()
        h_mid_orig = (hessian_diag[mid_mask] * delta_orig[mid_mask].pow(2)).sum().item()
        h_mid_mm = (hessian_diag[mid_mask] * delta_mm[mid_mask].pow(2)).sum().item()
        h_bot_orig = (hessian_diag[bottom_mask] * delta_orig[bottom_mask].pow(2)).sum().item()
        h_bot_mm = (hessian_diag[bottom_mask] * delta_mm[bottom_mask].pow(2)).sum().item()
        
        act_metrics['mse_orig'].append(mse_orig)
        act_metrics['mse_mm'].append(mse_mm)
        act_metrics['output_mse_orig'].append(y_err_orig)
        act_metrics['output_mse_mm'].append(y_err_mm)
        act_metrics['hessian_err_orig'].append(h_err_orig)
        act_metrics['hessian_err_mm'].append(h_err_mm)
        act_metrics['hessian_err_top10_orig'].append(h_top10_orig)
        act_metrics['hessian_err_top10_mm'].append(h_top10_mm)
        act_metrics['hessian_err_mid_orig'].append(h_mid_orig)
        act_metrics['hessian_err_mid_mm'].append(h_mid_mm)
        act_metrics['hessian_err_bottom_orig'].append(h_bot_orig)
        act_metrics['hessian_err_bottom_mm'].append(h_bot_mm)
        
    # ---- 权重量化对比 ----
    n_rows = min(64, weight.shape[0])
    wt_mse_orig_list = []
    wt_mse_minmax_list = []
    
    for r in range(n_rows):
        w = weight[r:r+1, :].to(device)  # (1, dim)
        
        # Baseline: Hadamard 128 Only
        w_rot = apply_hadamard_128(w)
        _, _, _, w_rot_q = quantize_and_mse_mxfp4(w_rot, quantizer, device)
        w_q_orig = apply_hadamard_128(w_rot_q.to(device))
        wt_mse_orig_list.append((w_q_orig - w).pow(2).mean().item())
        
        # MinMax: Reordering + Hadamard 128
        w_perm = w[:, perm]
        w_perm_rot = apply_hadamard_128(w_perm)
        _, _, _, w_perm_rot_q = quantize_and_mse_mxfp4(w_perm_rot, quantizer, device)
        w_perm_q = apply_hadamard_128(w_perm_rot_q.to(device))
        w_q_mm = w_perm_q[:, inv_perm]
        wt_mse_minmax_list.append((w_q_mm - w).pow(2).mean().item())
        
    # ---- 128-size Hadamard 块内 Outlier 数量统计 ----
    # 统计 P95 统计量在 128-block 里的分布情况，证明 Outlier 分散度
    median_p95 = p95.median().item()
    is_outlier = (p95 > 2 * median_p95)
    
    # 原始排列下每个 block (128) 内 Outlier 的数量
    outliers_per_block_orig = is_outlier.view(-1, 128).sum(dim=-1).cpu().numpy()
    
    # MinMax 排列下每个 block (128) 内 Outlier 的数量
    outliers_per_block_mm = is_outlier[perm].view(-1, 128).sum(dim=-1).cpu().numpy()
    
    result = {
        'wt_mse_orig': np.mean(wt_mse_orig_list),
        'wt_mse_minmax': np.mean(wt_mse_minmax_list),
        'per_group_act_orig': np.mean(per_group_act_orig_all, axis=0),
        'per_group_act_minmax': np.mean(per_group_act_mm_all, axis=0),
        'per_group_scale_orig': np.mean(per_group_scale_orig_all, axis=0),
        'per_group_scale_minmax': np.mean(per_group_scale_mm_all, axis=0),
        'outliers_per_block_orig': outliers_per_block_orig,
        'outliers_per_block_mm': outliers_per_block_mm,
        'n_total_groups': dim // 32,
        'n_total_blocks': dim // 128,
        'perm': perm,
    }
    
    # 合并激活指标的均值
    for key in act_metrics:
        result[key] = np.mean(act_metrics[key])
        
    return result


# ======================== 可视化 ========================

def plot_mxfp4_results(all_results, n_layers, output_prefix):
    """生成 MXFP4 专属的三张诊断大图"""
    layers = list(range(n_layers))
    x = np.arange(n_layers)
    
    # ===== 图1: Hessian 有效误差诊断 =====
    fig, axes = plt.subplots(3, 1, figsize=(16, 15))
    
    # Subplot 1: Plain MSE vs Output Error vs Hessian Error
    ax = axes[0]
    mse_changes = [(all_results[l]['mse_mm'] - all_results[l]['mse_orig']) / (all_results[l]['mse_orig'] + 1e-15) * 100 for l in layers]
    out_changes = [(all_results[l]['output_mse_mm'] - all_results[l]['output_mse_orig']) / (all_results[l]['output_mse_orig'] + 1e-15) * 100 for l in layers]
    hess_changes = [(all_results[l]['hessian_err_mm'] - all_results[l]['hessian_err_orig']) / (all_results[l]['hessian_err_orig'] + 1e-15) * 100 for l in layers]
    
    ax.plot(x, mse_changes, 'b-o', markersize=4, label='Plain MSE (Rotated space metric)', alpha=0.7)
    ax.plot(x, out_changes, 'r-s', markersize=4, label='Output Space Error ||WΔx||² (What matters)', alpha=0.7)
    ax.plot(x, hess_changes, 'g-^', markersize=4, label='Hessian-weighted Error Δx^T H Δx', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Change (%)')
    ax.set_title('MXFP4 Error Metric Changes: MinMax vs Baseline (negative = improved)')
    ax.legend()
    ax.set_xticks(x)
    
    # Subplot 2: Hessian 误差重要性通道消长
    ax = axes[1]
    top10_orig = [all_results[l]['hessian_err_top10_orig'] for l in layers]
    top10_mm = [all_results[l]['hessian_err_top10_mm'] for l in layers]
    mid_orig = [all_results[l]['hessian_err_mid_orig'] for l in layers]
    mid_mm = [all_results[l]['hessian_err_mid_mm'] for l in layers]
    bot_orig = [all_results[l]['hessian_err_bottom_orig'] for l in layers]
    bot_mm = [all_results[l]['hessian_err_bottom_mm'] for l in layers]
    
    top10_chg = [(m - o) / (o + 1e-15) * 100 for o, m in zip(top10_orig, top10_mm)]
    mid_chg = [(m - o) / (o + 1e-15) * 100 for o, m in zip(mid_orig, mid_mm)]
    bot_chg = [(m - o) / (o + 1e-15) * 100 for o, m in zip(bot_orig, bot_mm)]
    
    w = 0.25
    ax.bar(x - w, top10_chg, w, label='Top 10% Channels (Most Sensitive)', color='crimson', alpha=0.7)
    ax.bar(x, mid_chg, w, label='Middle 40% Channels', color='darkorange', alpha=0.7)
    ax.bar(x + w, bot_chg, w, label='Bottom 50% Channels (Least Sensitive)', color='forestgreen', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Hessian Error Change (%)')
    ax.set_title('MXFP4 Hessian Error Change by Channel Sensitivity Tiers')
    ax.legend()
    ax.set_xticks(x)
    
    # Subplot 3: 原始空间下的重要性桶对总 Hessian 误差的占比贡献
    ax = axes[2]
    top10_frac = [all_results[l]['hessian_err_top10_orig'] / (all_results[l]['hessian_err_orig'] + 1e-15) * 100 for l in layers]
    mid_frac = [all_results[l]['hessian_err_mid_orig'] / (all_results[l]['hessian_err_orig'] + 1e-15) * 100 for l in layers]
    bot_frac = [all_results[l]['hessian_err_bottom_orig'] / (all_results[l]['hessian_err_orig'] + 1e-15) * 100 for l in layers]
    
    ax.bar(x, top10_frac, label='Top 10% (Outliers)', color='crimson', alpha=0.7)
    ax.bar(x, mid_frac, bottom=top10_frac, label='Middle 40%', color='darkorange', alpha=0.7)
    ax.bar(x, bot_frac, bottom=[t + m for t, m in zip(top10_frac, mid_frac)], label='Bottom 50%', color='forestgreen', alpha=0.7)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Hessian Error Contribution (%)')
    ax.set_title('Channel Tier Error Contribution (Original Order Baseline)')
    ax.legend()
    ax.set_xticks(x)
    
    fig.tight_layout()
    fig.savefig(f'{output_prefix}_hessian_diag.png', dpi=150)
    print(f"  Saved: {output_prefix}_hessian_diag.png")
    plt.close(fig)
    
    # ===== 图2: 权重量化 MSE 逐层对比 =====
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    wt_orig = [all_results[l]['wt_mse_orig'] for l in layers]
    wt_mm = [all_results[l]['wt_mse_minmax'] for l in layers]
    
    ax = axes[0]
    ax.bar(x - 0.175, wt_orig, 0.35, label='Original + Hadamard', color='steelblue', alpha=0.8)
    ax.bar(x + 0.175, wt_mm, 0.35, label='MinMax + Hadamard', color='coral', alpha=0.8)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Mean Weight MSE')
    ax.set_title('MXFP4 Weight Quantization MSE (per layer)')
    ax.legend()
    ax.set_xticks(x)
    
    wt_change = [(m - o) / (o + 1e-15) * 100 for o, m in zip(wt_orig, wt_mm)]
    ax2 = axes[1]
    colors = ['green' if c < 0 else 'red' for c in wt_change]
    ax2.bar(x, wt_change, color=colors, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Weight MSE Change (%)')
    ax2.set_title('Weight MSE Change (negative = improved)')
    ax2.set_xticks(x)
    
    fig.tight_layout()
    fig.savefig(f'{output_prefix}_wt_mse.png', dpi=150)
    print(f"  Saved: {output_prefix}_wt_mse.png")
    plt.close(fig)
    
    # ===== 图3: 组级分解与块级 Outlier 隔离分析 (选3层) =====
    repr_layers = [0, n_layers // 2, n_layers - 1]
    fig, axes = plt.subplots(len(repr_layers), 3, figsize=(24, 5 * len(repr_layers)))
    
    for i, l in enumerate(repr_layers):
        res = all_results[l]
        n_g = res['n_total_groups']
        n_b = res['n_total_blocks']
        
        # Column 1: 32-group Quantization MSE in Rotated space
        ax = axes[i, 0]
        ax.plot(range(n_g), res['per_group_act_orig'], 'b-', alpha=0.6, label='Baseline Rotated', linewidth=0.8)
        ax.plot(range(n_g), res['per_group_act_minmax'], 'r-', alpha=0.6, label='MinMax Rotated', linewidth=0.8)
        ax.set_title(f'Layer {l}: 32-Group Act Quant MSE')
        ax.set_xlabel('32-Group Index')
        ax.set_ylabel('Quantization MSE')
        ax.legend(fontsize=8)
        
        # Column 2: 32-group Quantization Scale in Rotated space
        ax = axes[i, 1]
        ax.plot(range(n_g), res['per_group_scale_orig'], 'b-', alpha=0.6, label='Baseline Scale', linewidth=0.8)
        ax.plot(range(n_g), res['per_group_scale_minmax'], 'r-', alpha=0.6, label='MinMax Scale', linewidth=0.8)
        ax.set_title(f'Layer {l}: 32-Group Scale Distribution')
        ax.set_xlabel('32-Group Index')
        ax.set_ylabel('e8m0 Scale')
        ax.legend(fontsize=8)
        
        # Column 3: 128-size Hadamard Block-wise Outlier isolation statistics
        ax = axes[i, 2]
        ax.bar(np.arange(n_b) - 0.2, res['outliers_per_block_orig'], 0.4, label='Baseline', color='steelblue', alpha=0.7)
        ax.bar(np.arange(n_b) + 0.2, res['outliers_per_block_mm'], 0.4, label='MinMax (Isolated)', color='coral', alpha=0.7)
        ax.set_title(f'Layer {l}: Outlier Count per 128-Block\n(MinMax should yield flat distribution)')
        ax.set_xlabel('128-Block Index')
        ax.set_ylabel('Number of Outliers (>2*median)')
        ax.legend(fontsize=8)
        
    fig.tight_layout()
    fig.savefig(f'{output_prefix}_group_decompose.png', dpi=150)
    print(f"  Saved: {output_prefix}_group_decompose.png")
    plt.close(fig)


def print_summary(all_results, n_layers):
    """打印 MXFP4 评估汇总表"""
    print(f"\n{'='*120}")
    print(f"{'Layer':>5} | {'Act MSE Δ%':>10} | {'Output Err Δ%':>13} | {'Hess Err Δ%':>12} | "
          f"{'Wt MSE Δ%':>10} | {'Max Outl/Block (Orig -> MM)':>27}")
    print(f"{'─'*5}─┼─{'─'*10}─┼─{'─'*13}─┼─{'─'*12}─┼─"
          f"{'─'*10}─┼─{'─'*27}")
    
    total_act_mse = 0
    total_out_err = 0
    total_hess_err = 0
    total_wt_mse = 0
    
    for l in range(n_layers):
        r = all_results[l]
        act_chg = (r['mse_mm'] - r['mse_orig']) / (r['mse_orig'] + 1e-15) * 100
        out_chg = (r['output_mse_mm'] - r['output_mse_orig']) / (r['output_mse_orig'] + 1e-15) * 100
        hess_chg = (r['hessian_err_mm'] - r['hessian_err_orig']) / (r['hessian_err_orig'] + 1e-15) * 100
        wt_chg = (r['wt_mse_minmax'] - r['wt_mse_orig']) / (r['wt_mse_orig'] + 1e-15) * 100
        
        total_act_mse += act_chg
        total_out_err += out_chg
        total_hess_err += hess_chg
        total_wt_mse += wt_chg
        
        max_orig_block = r['outliers_per_block_orig'].max()
        max_mm_block = r['outliers_per_block_mm'].max()
        
        print(f"  L{l:>2} | {act_chg:>9.2f}% | {out_chg:>12.2f}% | {hess_chg:>11.2f}% | "
              f"{wt_chg:>9.2f}% | {max_orig_block:>10} -> {max_mm_block:<14}")
              
    print(f"\n  Average Act MSE Change:         {total_act_mse/n_layers:+.2f}%")
    print(f"  Average Output Space Err Change: {total_out_err/n_layers:+.2f}%")
    print(f"  Average Hessian Err Change:      {total_hess_err/n_layers:+.2f}%")
    print(f"  Average Weight MSE Change:       {total_wt_mse/n_layers:+.2f}%")
    print(f"{'='*120}")


# ======================== Main ========================

def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers. Specifying MXFP4 Quantizer (group_size=32)...")
    
    # MXFP4 量化器，group_size 32，采用 e8m0 比例精度
    quantizer = Quantizer(
        bits=4, format="mxfp", granularity="group", group_size=32,
        symmetric=True, scale_precision="e8m0"
    )
    
    print("Collecting activations and weights from all layers...")
    activations, weights = collect_all_layer_data(model, tokenizer, device, n_layers)
    
    # 释放大模型显存，以保证后面的计算非常流畅且没有 OOM
    del model
    torch.cuda.empty_cache()
    
    print("\nRunning full diagnostic for MXFP4 (Hadamard block 128, Group size 32)...")
    all_results = {}
    for l in range(n_layers):
        print(f"  Layer {l}/{n_layers-1}...", end=" ", flush=True)
        result = analyze_layer(l, activations[l], weights[l], quantizer, device)
        all_results[l] = result
        
        act_chg = (result['mse_mm'] - result['mse_orig']) / (result['mse_orig'] + 1e-15) * 100
        out_chg = (result['output_mse_mm'] - result['output_mse_orig']) / (result['output_mse_orig'] + 1e-15) * 100
        print(f"Act MSE: {act_chg:+.2f}% | Output Err: {out_chg:+.2f}%")
        
    print_summary(all_results, n_layers)
    plot_mxfp4_results(all_results, n_layers, "mxfp4_diag")
    print("\nMXFP4 diagnosis completed successfully! Visualizations saved with prefix 'mxfp4_diag'.")


if __name__ == "__main__":
    main()
