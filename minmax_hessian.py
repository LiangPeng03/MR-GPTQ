"""
MinMax 有效输出误差诊断
=======================
验证假设：MinMax 降低了"不重要通道"的 MSE，
但对模型真正关心的 Hessian 加权输出误差几乎无影响。

测量:
  1. 输出空间误差: ||W @ quant(x) - W @ x||²
  2. Hessian 加权误差: Δx^T H Δx (H 近似为 diag(X^T X))
  3. 按通道重要性分桶: 大通道 vs 小通道的贡献分解
"""
import os, sys, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.quantization.quantizer import Quantizer

def compute_minmax_perm(act_stat, group_size):
    N = act_stat.shape[0]
    sorted_idx = torch.argsort(act_stat, descending=True)
    head, tail = 0, N - 1
    perm = []
    while head < tail and len(perm) + group_size <= N:
        group = [sorted_idx[head].item()]
        head += 1
        for _ in range(group_size - 1):
            group.append(sorted_idx[tail].item())
            tail -= 1
        perm.extend(group)
    for i in range(head, tail + 1):
        perm.append(sorted_idx[i].item())
    return torch.tensor(perm, dtype=torch.long)


def quantize_row(x_row, quantizer, device):
    """量化一行激活值，返回量化后的值"""
    x = x_row.unsqueeze(0).to(device)  # (1, dim)
    scales, zeros = quantizer.get_quantization_params(x)
    x_q = quantizer(x, scales, zeros)
    return x_q.squeeze(0).cpu()


def analyze_layer_output_error(layer_idx, act, weight, quantizer, device, group_size=16):
    """
    测量层输出空间的有效误差
    act: (n_tokens, dim)
    weight: (out_dim, in_dim) 
    """
    dim = act.shape[-1]
    n_tokens = min(50, act.shape[0])
    
    # P95 统计量 & MinMax 排列
    p95 = torch.quantile(act.abs(), 0.95, dim=0)
    perm = compute_minmax_perm(p95, group_size)
    inv_perm = torch.argsort(perm)  # 逆排列，用于把 MinMax 量化结果映射回原始空间
    
    # 近似 Hessian 对角线 = Σ x_i² / n_tokens (per channel)
    hessian_diag = (act[:n_tokens] ** 2).mean(dim=0)  # (dim,)
    
    # Weight sub-matrix for output error computation (取前 128 行减少计算)
    n_out = min(128, weight.shape[0])
    W = weight[:n_out, :].float().to(device)  # (n_out, dim)
    
    # ---- 逐 token 测量 ----
    metrics = {
        'output_mse_orig': [], 'output_mse_mm': [],
        'hessian_err_orig': [], 'hessian_err_mm': [],
        'mse_orig': [], 'mse_mm': [],
        # 按通道重要性分桶
        'hessian_err_top10_orig': [], 'hessian_err_top10_mm': [],
        'hessian_err_mid_orig': [], 'hessian_err_mid_mm': [],
        'hessian_err_bottom_orig': [], 'hessian_err_bottom_mm': [],
    }
    
    # 通道重要性排序 (按 Hessian 对角线 × P95²)
    importance = hessian_diag * p95 ** 2
    imp_sorted = torch.argsort(importance, descending=True)
    top10_mask = torch.zeros(dim, dtype=torch.bool)
    top10_mask[imp_sorted[:dim // 10]] = True  # 前 10% 最重要的通道
    mid_mask = torch.zeros(dim, dtype=torch.bool)
    mid_mask[imp_sorted[dim // 10: dim // 2]] = True  # 中间 40%
    bottom_mask = ~top10_mask & ~mid_mask  # 后 50%
    
    for t in range(n_tokens):
        x = act[t].float()  # (dim,)
        
        # --- 原始顺序量化 ---
        x_q_orig = quantize_row(x, quantizer, device).float()
        delta_orig = x_q_orig - x  # (dim,)
        
        # --- MinMax 顺序量化 ---
        x_perm = x[perm]
        x_q_perm = quantize_row(x_perm, quantizer, device).float()
        # 映射回原始通道空间
        x_q_mm = x_q_perm[inv_perm]
        delta_mm = x_q_mm - x  # (dim,)
        
        # 1. 普通 MSE
        mse_orig = delta_orig.pow(2).mean().item()
        mse_mm = delta_mm.pow(2).mean().item()
        
        # 2. 输出空间误差: ||W @ Δx||²
        x_dev = x.to(device)
        delta_orig_dev = delta_orig.to(device)
        delta_mm_dev = delta_mm.to(device)
        
        y_err_orig = (W @ delta_orig_dev).pow(2).mean().item()
        y_err_mm = (W @ delta_mm_dev).pow(2).mean().item()
        
        # 3. Hessian 加权误差: Σ H[i] × Δx[i]²
        h_err_orig = (hessian_diag * delta_orig.pow(2)).sum().item()
        h_err_mm = (hessian_diag * delta_mm.pow(2)).sum().item()
        
        # 4. 按通道重要性分桶
        h_top10_orig = (hessian_diag[top10_mask] * delta_orig[top10_mask].pow(2)).sum().item()
        h_top10_mm = (hessian_diag[top10_mask] * delta_mm[top10_mask].pow(2)).sum().item()
        h_mid_orig = (hessian_diag[mid_mask] * delta_orig[mid_mask].pow(2)).sum().item()
        h_mid_mm = (hessian_diag[mid_mask] * delta_mm[mid_mask].pow(2)).sum().item()
        h_bot_orig = (hessian_diag[bottom_mask] * delta_orig[bottom_mask].pow(2)).sum().item()
        h_bot_mm = (hessian_diag[bottom_mask] * delta_mm[bottom_mask].pow(2)).sum().item()
        
        metrics['mse_orig'].append(mse_orig)
        metrics['mse_mm'].append(mse_mm)
        metrics['output_mse_orig'].append(y_err_orig)
        metrics['output_mse_mm'].append(y_err_mm)
        metrics['hessian_err_orig'].append(h_err_orig)
        metrics['hessian_err_mm'].append(h_err_mm)
        metrics['hessian_err_top10_orig'].append(h_top10_orig)
        metrics['hessian_err_top10_mm'].append(h_top10_mm)
        metrics['hessian_err_mid_orig'].append(h_mid_orig)
        metrics['hessian_err_mid_mm'].append(h_mid_mm)
        metrics['hessian_err_bottom_orig'].append(h_bot_orig)
        metrics['hessian_err_bottom_mm'].append(h_bot_mm)
    
    # 取平均
    result = {}
    for key in metrics:
        result[key] = np.mean(metrics[key])
    return result


def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    group_size = 16
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    
    quantizer = Quantizer(
        bits=4, format="nvfp", granularity="group", group_size=group_size,
        symmetric=True, scale_precision="e4m3"
    )
    
    # 收集激活值
    text = "The large language model architecture involves sophisticated attention mechanisms. " * 300
    calib_data = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    all_results = {}
    
    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx].mlp.down_proj
        weight = layer.weight.data.cpu().float()
        
        cache = []
        def hook(m, i, o, c=cache): c.append(i[0].detach().cpu())
        handle = layer.register_forward_hook(hook)
        with torch.no_grad():
            model(**calib_data)
        handle.remove()
        act = cache[0].view(-1, cache[0].shape[-1]).float()
        torch.cuda.empty_cache()
        
        result = analyze_layer_output_error(layer_idx, act, weight, quantizer, device, group_size)
        all_results[layer_idx] = result
        
        mse_chg = (result['mse_mm'] - result['mse_orig']) / (result['mse_orig'] + 1e-15) * 100
        out_chg = (result['output_mse_mm'] - result['output_mse_orig']) / (result['output_mse_orig'] + 1e-15) * 100
        hess_chg = (result['hessian_err_mm'] - result['hessian_err_orig']) / (result['hessian_err_orig'] + 1e-15) * 100
        
        print(f"  L{layer_idx:>2} | MSE: {mse_chg:>+7.2f}% | OutputErr: {out_chg:>+7.2f}% | HessianErr: {hess_chg:>+7.2f}%")
    
    # ===== 绘图 =====
    layers = list(range(n_layers))
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    # --- 图1: 三种误差指标的变化百分比 ---
    ax = axes[0]
    mse_changes = [(all_results[l]['mse_mm'] - all_results[l]['mse_orig']) / (all_results[l]['mse_orig'] + 1e-15) * 100 for l in layers]
    out_changes = [(all_results[l]['output_mse_mm'] - all_results[l]['output_mse_orig']) / (all_results[l]['output_mse_orig'] + 1e-15) * 100 for l in layers]
    hess_changes = [(all_results[l]['hessian_err_mm'] - all_results[l]['hessian_err_orig']) / (all_results[l]['hessian_err_orig'] + 1e-15) * 100 for l in layers]
    
    x = np.arange(n_layers)
    ax.plot(x, mse_changes, 'b-o', markersize=4, label='Plain MSE (what we optimized)', alpha=0.7)
    ax.plot(x, out_changes, 'r-s', markersize=4, label='Output Error ||WΔx||² (what matters)', alpha=0.7)
    ax.plot(x, hess_changes, 'g-^', markersize=4, label='Hessian-weighted Δx^T H Δx', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Change (%)')
    ax.set_title('MinMax vs Original: Three Error Metrics Compared\n(negative = improved)')
    ax.legend()
    ax.set_xticks(x)
    
    # --- 图2: Hessian 误差按通道重要性分解 ---
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
    ax.bar(x - w, top10_chg, w, label='Top 10% channels (most important)', color='red', alpha=0.7)
    ax.bar(x, mid_chg, w, label='Middle 40% channels', color='orange', alpha=0.7)
    ax.bar(x + w, bot_chg, w, label='Bottom 50% channels (least important)', color='green', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Hessian Error Change (%)')
    ax.set_title('Hessian Error Change by Channel Importance Tier\n(Red=important channels, Green=unimportant channels)')
    ax.legend(fontsize=8)
    ax.set_xticks(x)
    
    # --- 图3: 各通道重要性桶对总 Hessian 误差的贡献占比 ---
    ax = axes[2]
    top10_frac = [all_results[l]['hessian_err_top10_orig'] / (all_results[l]['hessian_err_orig'] + 1e-15) * 100 for l in layers]
    mid_frac = [all_results[l]['hessian_err_mid_orig'] / (all_results[l]['hessian_err_orig'] + 1e-15) * 100 for l in layers]
    bot_frac = [all_results[l]['hessian_err_bottom_orig'] / (all_results[l]['hessian_err_orig'] + 1e-15) * 100 for l in layers]
    
    ax.bar(x, top10_frac, label='Top 10%', color='red', alpha=0.7)
    ax.bar(x, mid_frac, bottom=top10_frac, label='Middle 40%', color='orange', alpha=0.7)
    ax.bar(x, bot_frac, bottom=[t + m for t, m in zip(top10_frac, mid_frac)], label='Bottom 50%', color='green', alpha=0.7)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Contribution to Total Hessian Error (%)')
    ax.set_title('Channel Importance Contribution to Total Error (Original Order)\n(Shows which channels ACTUALLY matter)')
    ax.legend(fontsize=8)
    ax.set_xticks(x)
    
    fig.tight_layout()
    fig.savefig('minmax_hessian_diag.png', dpi=150)
    print(f"\n  Saved: minmax_hessian_diag.png")
    
    # ===== 汇总 =====
    avg_mse = np.mean(mse_changes)
    avg_out = np.mean(out_changes)
    avg_hess = np.mean(hess_changes)
    print(f"\n{'='*60}")
    print(f"  Average Plain MSE Change:    {avg_mse:>+7.2f}%")
    print(f"  Average Output Error Change: {avg_out:>+7.2f}%")
    print(f"  Average Hessian Error Change: {avg_hess:>+7.2f}%")
    print(f"{'='*60}")
    
    if abs(avg_out) < abs(avg_mse) * 0.3:
        print("\n  ⚠ CONFIRMED: Output error change is much smaller than MSE change.")
        print("    → MinMax optimizes 'unimportant' channels; PPL is insensitive to this.")
    elif avg_out > 0:
        print("\n  ⚠ Output error INCREASED despite MSE decrease!")
        print("    → MinMax hurts model-sensitive channels.")
    else:
        print("\n  ✓ Output error also improved. The PPL issue may be in error propagation.")


if __name__ == "__main__":
    main()
