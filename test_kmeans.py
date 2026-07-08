"""
test_kmeans.py - 验证 K-Means FP4 通道重排序对 MSE 的降低效果
比较：
- Baseline: 原始通道顺序
- K-Means FP4: 通道重排后
分别统计：
- 激活矩阵 MSE
- 权重矩阵 MSE
- 输出结果 MSE (Weight * Act)
支持多层测试（Layer 0, 15, 31），每种矩阵独立统计。
与 test.sh 配置对齐。
"""
import torch
import gc
import copy
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.data_utils import get_data
from src.quantization.quantizer import get_reciprocal
from src.quantization.quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX

# ================================================================
# 基础工具
# ================================================================
class ForwardInterrupt(Exception): pass

class InputCollector(torch.nn.Module):
    def __init__(self, module, cpu_offload=False):
        super().__init__()
        self.module = module
        self.cpu_offload = cpu_offload
        self.input_args = []
        self.input_kwargs = []
    def forward(self, *input_args, **input_kwargs):
        if self.cpu_offload:
            def to_cpu(v):
                if isinstance(v, torch.Tensor): return v.cpu()
                if isinstance(v, tuple): return tuple(to_cpu(x) for x in v)
                if isinstance(v, list): return [to_cpu(x) for x in v]
                if isinstance(v, dict): return {k: to_cpu(val) for k, val in v.items()}
                return v
            input_args = to_cpu(input_args)
            input_kwargs = to_cpu(input_kwargs)
        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)
        raise ForwardInterrupt

def to_device(v, device):
    if isinstance(v, torch.Tensor): return v.to(device)
    if isinstance(v, tuple): return tuple(to_device(x, device) for x in v)
    if isinstance(v, list): return [to_device(x, device) for x in v]
    if isinstance(v, dict): return {k: to_device(val, device) for k, val in v.items()}
    return v

def maybe_first(obj):
    if isinstance(obj, tuple): return obj[0]
    return obj

def get_combined_weight(block, name):
    if name == "qkv": w = torch.cat([block.self_attn.q_proj.weight, block.self_attn.k_proj.weight, block.self_attn.v_proj.weight], dim=0)
    elif name == "o": w = block.self_attn.o_proj.weight
    elif name == "gate_up": w = torch.cat([block.mlp.gate_proj.weight, block.mlp.up_proj.weight], dim=0)
    elif name == "down": w = block.mlp.down_proj.weight
    else: return None
    return w.float()

def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    out = torch.where(x > 5.0, 6.0,
          torch.where(x >= 3.5, 4.0,
          torch.where(x >= 1.75, torch.round(x),
                                 torch.round(x * 2.0) * 0.5)))
    return out * sign

def compute_global_scale(x):
    act_max = x.abs().max().to(torch.float32).view(1)
    return (FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(act_max)).to(x.device)

def scale_to_e4m3(raw_scale, global_scale):
    return (raw_scale * global_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX) \
        .to(torch.float8_e4m3fn) \
        .to(torch.float32) \
        .mul(get_reciprocal(global_scale))

def compute_kmeans_fp4_perm(X_abs_T, group_size=16, channel_weights=None, act_alpha=0.0):
    dim = X_abs_T.shape[0]
    n_groups = dim // group_size
    device = X_abs_T.device
    grid_mults = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device, dtype=torch.float32)
    def compute_loss_distances(all_ch_abs, ref_ch_abs, chunk_size=2048, c_weights=None):
        ref = ref_ch_abs.unsqueeze(0)
        n_ch = all_ch_abs.shape[0]
        losses = torch.zeros(n_ch, device=device, dtype=torch.float32)
        for i in range(0, n_ch, chunk_size):
            chunk = all_ch_abs[i:i+chunk_size]
            p_max = torch.maximum(chunk, ref)
            p_min = torch.minimum(chunk, ref)
            scale = (p_max / 6.0).clamp(min=1e-10)
            norm = p_min / scale
            min_q = cast_to_fp4(norm) * scale
            mse = (p_min - min_q) ** 2
            if act_alpha > 0.0:
                act_weight = p_min.abs() ** act_alpha
                mse = mse * act_weight
            mse = mse.sum(dim=1)
            if c_weights is not None:
                mse = mse * c_weights[i:i+chunk_size]
            losses[i:i+chunk_size] = mse
        return losses

    if act_alpha > 0.0:
        ch_sums = (X_abs_T ** (1.0 + act_alpha)).sum(dim=1)
    else:
        ch_sums = X_abs_T.sum(dim=1)
    
    if channel_weights is not None:
        ch_sums = ch_sums * channel_weights

    accumulated_dist = torch.zeros(dim, device=device, dtype=torch.float32)
    seed_order = []
    first_seed = ch_sums.argmax().item()
    seed_order.append(first_seed)
    accumulated_dist[first_seed] = -float('inf')
    dists = compute_loss_distances(X_abs_T, X_abs_T[first_seed], c_weights=channel_weights)
    accumulated_dist += dists
    accumulated_dist[first_seed] = -float('inf')
    for k in range(1, n_groups):
        new_seed = accumulated_dist.argmax().item()
        seed_order.append(new_seed)
        accumulated_dist[new_seed] = -float('inf')
        dists = compute_loss_distances(X_abs_T, X_abs_T[new_seed], c_weights=channel_weights)
        accumulated_dist += dists
    opt_groups = [[s] for s in seed_order]
    group_maxes = X_abs_T[seed_order].clone()
    is_assigned = torch.zeros(dim, dtype=torch.bool, device=device)
    group_sizes = torch.ones(n_groups, device=device, dtype=torch.long)
    seed_set = set(seed_order)
    remaining = [c for c in range(dim) if c not in seed_set]
    remaining_sums = ch_sums[remaining]
    sorted_order = torch.argsort(remaining_sums, descending=True)
    remaining_sorted = [remaining[i] for i in sorted_order.tolist()]
    for c in remaining_sorted:
        c_abs = X_abs_T[c]
        ref = c_abs.unsqueeze(0)
        
        # Vectorized over all groups at once
        p_max = torch.maximum(ref, group_maxes)
        p_min = torch.minimum(ref, group_maxes)
        scale = (p_max / 6.0).clamp(min=1e-10)
        norm = p_min / scale
        min_q = cast_to_fp4(norm) * scale
        mse = (p_min - min_q) ** 2
        
        if act_alpha > 0.0:
            act_weight = p_min.abs() ** act_alpha
            mse = mse * act_weight
            
        losses = mse.sum(dim=1)
        if channel_weights is not None:
            losses = losses * channel_weights[c]
            
        losses[group_sizes >= group_size] = float('inf')
        best_g = losses.argmin().item()
        opt_groups[best_g].append(c)
        group_maxes[best_g] = torch.maximum(group_maxes[best_g], c_abs)
        group_sizes[best_g] += 1
    perm = []
    for g in opt_groups: perm.extend(g)
    return torch.tensor(perm, device=device, dtype=torch.long)

def simulate_nvfp4(tensor, group_size=16):
    orig_shape = tensor.shape
    t_groups = tensor.contiguous().view(-1, group_size)
    gs = compute_global_scale(tensor)
    abs_max = t_groups.abs().amax(dim=-1, keepdim=True)
    s_current = abs_max / 6.0
    s_current[s_current == 0] = 1.0
    s_e4m3 = scale_to_e4m3(s_current, gs)
    q_val = cast_to_fp4(t_groups / s_e4m3)
    dequant = q_val * s_e4m3
    return dequant.view(orig_shape)

def compute_group_diagnostics(tensor, group_size=16):
    """
    FP4 网格诊断指标。分析量化组内值落入高损失区(HLZ)的情况。
    
    FP4 网格 (归一化后): 0, 0.5, 1, 1.5, 2, 3, 4, 6
    高损失区定义 (归一化值 n = |v| / scale):
      - [2.25, 2.75]: 2~3 间隙中心，gap=1.0
      - [3.25, 3.75]: 3~4 间隙中心，gap=1.0
      - [4.50, 5.50]: 4~6 间隙中心，gap=2.0
    
    仅统计 "显著值" (归一化 > 2.0，即位于稀疏网格区域的大值)。
    
    Returns dict:
      hlzs       : 每行(或Token)平均在 HLZ 的原始绝对值累加之和 (High-Loss Zone Sum)
      hlz_mse_pct: HLZ 显著值贡献的 MSE 占总 MSE 的百分比
      submax_mse_pct: 非最大值的显著值(归一化>2)贡献的 MSE 占比
    """
    t = tensor.contiguous().view(-1, group_size).float()
    abs_t = t.abs()
    n_groups = t.shape[0]
    
    # 使用与 simulate_nvfp4 完全一致的 scale 计算
    gs = compute_global_scale(tensor)
    group_max = abs_t.amax(dim=-1, keepdim=True)
    raw_scale = group_max / 6.0
    raw_scale[raw_scale == 0] = 1.0
    scale = scale_to_e4m3(raw_scale, gs)
    
    # 归一化到 FP4 域 [0, ~6]
    normalized = abs_t / scale
    
    # FP4 量化
    grid_points = cast_to_fp4(normalized)
    
    # 每元素量化误差 (原始尺度)
    error_sq = ((normalized - grid_points) * scale) ** 2
    total_mse = error_sq.sum().item()
    
    # 显著值: 归一化 > 2.0 (位于稀疏网格区，gap >= 1.0)
    is_significant = normalized > 2.0
    
    # 区分"谁是组最大值"：每组第一个达到 group_max 的位置
    is_group_max = (abs_t == group_max)
    # 处理并列: 每组只保留第一个 max
    max_cumsum = is_group_max.cumsum(dim=-1)
    is_first_max = is_group_max & (max_cumsum == 1)
    
    # 高损失区: 各间隙的中心 50% 区域
    in_hlz = ((normalized > 2.25) & (normalized < 2.75)) | \
             ((normalized > 3.25) & (normalized < 3.75)) | \
             ((normalized > 4.25) & (normalized < 5.75))
    
    # HLZ 显著值: 在 HLZ 且显著，并且排除组最大值
    hlz_sig = in_hlz & is_significant & (~is_first_max)
    
    # 指标 1: HLZS (High-Loss Zone Sum) 原始尺度的绝对值之和平均到每组或每行
    # 这里我们采用平均到每组 (除以 n_groups)
    hlzs = (abs_t * hlz_sig.float()).sum().item() / n_groups
    
    # 指标 2: HLZ 显著值的 MSE 占比
    hlz_mse = (error_sq * hlz_sig.float()).sum().item()
    hlz_mse_pct = hlz_mse / (total_mse + 1e-20) * 100
    
    # 指标 3: 所有非最大值显著值的 MSE 占比
    is_submax_sig = is_significant & (~is_first_max)
    submax_mse = (error_sq * is_submax_sig.float()).sum().item()
    submax_mse_pct = submax_mse / (total_mse + 1e-20) * 100
    
    # 指标 4: 最大值的 MSE 占比
    max_mse = (error_sq * is_first_max.float()).sum().item()
    max_mse_pct = max_mse / (total_mse + 1e-20) * 100
    
    return {
        'hlzs': hlzs,
        'hlz_mse_pct': hlz_mse_pct,
        'submax_mse_pct': submax_mse_pct,
        'max_mse_pct': max_mse_pct,
    }

def optimize_channel_scales_coordinate_descent(X, W, weight_mse_ratio=2.0, group_size=16, top_k=5, num_rounds=3):
    """
    Coordinate Descent Joint Tree Search for NVFP4 Channel Scaling.
    Targets Top-K channels and optimizes them iteratively for num_rounds.
    """
    device = X.device
    in_features = X.shape[1]
    num_groups = in_features // group_size
    T = X.shape[0]
    OutDim = W.shape[0]
    
    S_g = torch.ones((num_groups, group_size), device=device) # [num_groups, 16]
    
    # 21 个候选缩放值 (0.5 到 1.5 之间，步长 0.05，包含 1.0)
    candidates = torch.linspace(0.5, 1.5, steps=21, device=device)
    num_cands = candidates.shape[0]
    
    X_g = X.view(T, num_groups, group_size)
    W_g = W.view(OutDim, num_groups, group_size)
    
    gs_w = compute_global_scale(W)
    
    def calc_mse_positive_inplace(mat_temp, gs=None):
        vmax = mat_temp.amax(dim=-1, keepdim=True)
        raw_scale = vmax.mul_(1/6.0).clamp_(min=1e-10)
        scale = scale_to_e4m3(raw_scale, gs) if gs is not None else raw_scale
        
        mat_temp.div_(scale)
        quantized = torch.where(mat_temp > 5.0, 6.0,
                      torch.where(mat_temp >= 3.5, 4.0,
                      torch.where(mat_temp >= 1.75, torch.round(mat_temp),
                                             torch.round(mat_temp * 2.0) * 0.5)))
        mat_temp.sub_(quantized).pow_(2).mul_(scale.pow(2))
        return mat_temp.sum(dim=(-3, -1))
        
    abs_X_g = X_g.abs()
    abs_W_g = W_g.abs()
    
    base_act_mse = calc_mse_positive_inplace(abs_X_g.clone()) # [num_groups]
    base_w_mse = calc_mse_positive_inplace(abs_W_g.clone(), gs=gs_w) # [num_groups]
    
    norm2_X = (X_g.float() ** 2).sum(dim=0) # [num_groups, 16]
    norm2_W = (W_g.float() ** 2).sum(dim=0) # [num_groups, 16]
    
    sensitivity_c = norm2_X * norm2_W # [num_groups, 16]
    topk_vals, topk_idx = torch.topk(sensitivity_c, k=top_k, dim=-1) # [num_groups, top_k]
    
    target_elements = 600_000_000 # 峰值显存控制
    max_dim = max(T, OutDim)
    chunk_G = max(1, target_elements // (max_dim * 16 * num_cands))
    
    for g_start in range(0, num_groups, chunk_G):
        g_end = min(g_start + chunk_G, num_groups)
        G_curr = g_end - g_start
        
        X_g_chunk = abs_X_g[:, g_start:g_end, :] # [T, G_curr, 16]
        W_g_chunk = abs_W_g[:, g_start:g_end, :] # [OutDim, G_curr, 16]
        topk_idx_chunk = topk_idx[g_start:g_end] # [G_curr, top_k]
        
        b_act = base_act_mse[g_start:g_end].unsqueeze(0) + 1e-12
        b_w = base_w_mse[g_start:g_end].unsqueeze(0) + 1e-12
        
        # 本地维护最优的 Scale 矩阵
        local_S = torch.ones((G_curr, group_size), device=device)
        
        for r in range(num_rounds):
            for k in range(top_k):
                # temp_S 继承当前的最佳 scale [num_cands, G_curr, 16]
                temp_S = local_S.unsqueeze(0).expand(num_cands, G_curr, group_size).clone()
                
                # 针对第 k 个目标通道，替换为 21 个候选值
                target_ch_idx = topk_idx_chunk[:, k] # [G_curr]
                c_expanded = candidates.unsqueeze(1).expand(num_cands, G_curr)
                idx_expanded = target_ch_idx.unsqueeze(0).unsqueeze(-1).expand(num_cands, G_curr, 1)
                temp_S.scatter_(2, idx_expanded, c_expanded.unsqueeze(-1))
                
                X_temp = X_g_chunk.unsqueeze(0) * temp_S.unsqueeze(1) # [21, T, G_curr, 16]
                W_temp = W_g_chunk.unsqueeze(0) / temp_S.unsqueeze(1) # [21, OutDim, G_curr, 16]
                
                act_mse = calc_mse_positive_inplace(X_temp) # [21, G_curr]
                w_mse = calc_mse_positive_inplace(W_temp, gs=gs_w) # [21, G_curr]
                
                score = (act_mse / b_act) + weight_mse_ratio * (w_mse / b_w) # [21, G_curr]
                
                # 取得 21 个候选中的最低分数及其索引
                min_score, min_idx = score.min(dim=0) # [G_curr]
                
                # 直接更新为当前找到的最优解（这保证了严格的单调下降或持平）
                best_c = candidates[min_idx] # [G_curr]
                local_S.scatter_(1, target_ch_idx.unsqueeze(1), best_c.unsqueeze(1))
                
        S_g[g_start:g_end] = local_S
        
    S_out = S_g.view(-1)
    
    # 打印统计信息
    mean_s = S_out.mean().item()
    min_s = S_out.min().item()
    max_s = S_out.max().item()
    pct_high = (S_out > 1.2).float().mean().item() * 100
    pct_low = (S_out < 0.8).float().mean().item() * 100
    print(f"      [Coord Descent Search] S_g Dist: Mean={mean_s:.3f}, Min={min_s:.3f}, Max={max_s:.3f} | >1.2: {pct_high:.1f}%, <0.8: {pct_low:.1f}%")
    
    return S_out


def run_kmeans_experiment(X, W, perm_km, group_size=16):
    results = {}
    
    # 1. True FP16 Output
    out_true = X @ W.T
    
    # 2. Baseline
    X_q_base = simulate_nvfp4(X, group_size)
    W_q_base = simulate_nvfp4(W, group_size)
    out_base = X_q_base @ W_q_base.T
    
    results["base_act_mse"] = ((X_q_base - X) ** 2).mean().item()
    results["base_w_mse"] = ((W_q_base - W) ** 2).mean().item()
    results["base_out_mse"] = ((out_base - out_true) ** 2).mean().item()
    
    # Baseline 诊断
    diag_base_act = compute_group_diagnostics(X, group_size)
    diag_base_w = compute_group_diagnostics(W, group_size)
    results["base_act_hlzs"] = diag_base_act['hlzs']
    results["base_act_hlz_mse_pct"] = diag_base_act['hlz_mse_pct']
    results["base_act_submax_mse_pct"] = diag_base_act['submax_mse_pct']
    results["base_act_max_mse_pct"] = diag_base_act['max_mse_pct']
    results["base_w_hlzs"] = diag_base_w['hlzs']
    results["base_w_hlz_mse_pct"] = diag_base_w['hlz_mse_pct']
    results["base_w_submax_mse_pct"] = diag_base_w['submax_mse_pct']
    results["base_w_max_mse_pct"] = diag_base_w['max_mse_pct']
    
    # 3. KMeans Permutation
    X_perm = X[:, perm_km]
    W_perm = W[:, perm_km]
    
    X_q_km = simulate_nvfp4(X_perm, group_size)
    W_q_km = simulate_nvfp4(W_perm, group_size)
    out_km = X_q_km @ W_q_km.T
    
    results["km_act_mse"] = ((X_q_km - X_perm) ** 2).mean().item()
    results["km_w_mse"] = ((W_q_km - W_perm) ** 2).mean().item()
    results["km_out_mse"] = ((out_km - out_true) ** 2).mean().item()
    
    # KMeans 诊断
    diag_km_act = compute_group_diagnostics(X_perm, group_size)
    diag_km_w = compute_group_diagnostics(W_perm, group_size)
    results["km_act_hlzs"] = diag_km_act['hlzs']
    results["km_act_hlz_mse_pct"] = diag_km_act['hlz_mse_pct']
    results["km_act_submax_mse_pct"] = diag_km_act['submax_mse_pct']
    results["km_act_max_mse_pct"] = diag_km_act['max_mse_pct']
    results["km_w_hlzs"] = diag_km_w['hlzs']
    results["km_w_hlz_mse_pct"] = diag_km_w['hlz_mse_pct']
    results["km_w_submax_mse_pct"] = diag_km_w['submax_mse_pct']
    results["km_w_max_mse_pct"] = diag_km_w['max_mse_pct']
    
    # 4. Scaled (KMeans + Coordinate Descent Search)
    S_gics = optimize_channel_scales_coordinate_descent(X_perm, W_perm, weight_mse_ratio=1.0, group_size=group_size, top_k=5, num_rounds=3)
    X_scaled = X_perm * S_gics
    W_scaled = W_perm / S_gics
    
    X_q_sc = simulate_nvfp4(X_scaled, group_size)
    W_q_sc = simulate_nvfp4(W_scaled, group_size)
    out_sc = X_q_sc @ W_q_sc.T
    
    results["sc_act_mse"] = ((X_q_sc - X_scaled) ** 2).mean().item()
    results["sc_w_mse"] = ((W_q_sc - W_scaled) ** 2).mean().item()
    results["sc_out_mse"] = ((out_sc - out_true) ** 2).mean().item()
    
    diag_sc_act = compute_group_diagnostics(X_scaled, group_size)
    diag_sc_w = compute_group_diagnostics(W_scaled, group_size)
    results["sc_act_hlzs"] = diag_sc_act['hlzs']
    results["sc_act_hlz_mse_pct"] = diag_sc_act['hlz_mse_pct']
    results["sc_act_submax_mse_pct"] = diag_sc_act['submax_mse_pct']
    results["sc_act_max_mse_pct"] = diag_sc_act['max_mse_pct']
    results["sc_w_hlzs"] = diag_sc_w['hlzs']
    results["sc_w_hlz_mse_pct"] = diag_sc_w['hlz_mse_pct']
    results["sc_w_submax_mse_pct"] = diag_sc_w['submax_mse_pct']
    results["sc_w_max_mse_pct"] = diag_sc_w['max_mse_pct']
    
    return results

# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--dataset_name_or_path", type=str, default="fineweb-edu")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--num_sequences", type=int, default=16)
    parser.add_argument("--kmeans_alpha", type=float, default=2.0)
    parser.add_argument("--kmeans_act_alpha", type=float, default=0.0)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    print(f"Loading calibration data ({args.num_sequences} sequences)...")
    calib_data = get_data(args.dataset_name_or_path, tokenizer, args.sequence_length, args.num_sequences, seed=42)
    
    model.config.use_cache = False
    model.requires_grad_(False)
    blocks = model.model.layers
    
    print("Capturing layer inputs (Batched to maximize GPU util)...")
    
    # 尝试将输入 batch 起来，提高 GPU 利用率
    if isinstance(calib_data, list) and all(isinstance(x, torch.Tensor) for x in calib_data):
        try:
            batched_sample = torch.cat(calib_data, dim=0)
            calib_data = [batched_sample]
        except Exception:
            pass
            
    blocks[0] = InputCollector(blocks[0], cpu_offload=False)
    model.get_input_embeddings().to(device)
    blocks[0] = blocks[0].to(device)
    for sample in calib_data:
        try:
            with torch.no_grad(): model(sample.to(device))
        except ForwardInterrupt: pass
    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module.cpu()
    model.get_input_embeddings().cpu()
    
    target_layers = [0, 15, 31]
    matrix_names = ["qkv", "o", "gate_up", "down"]
    
    # 全局统计累加器
    diag_keys = [
        "base_act_hlzs", "km_act_hlzs", "sc_act_hlzs",
        "base_act_hlz_mse_pct", "km_act_hlz_mse_pct", "sc_act_hlz_mse_pct",
        "base_act_submax_mse_pct", "km_act_submax_mse_pct", "sc_act_submax_mse_pct",
        "base_act_max_mse_pct", "km_act_max_mse_pct", "sc_act_max_mse_pct",
        "base_w_hlzs", "km_w_hlzs", "sc_w_hlzs",
        "base_w_hlz_mse_pct", "km_w_hlz_mse_pct", "sc_w_hlz_mse_pct",
        "base_w_submax_mse_pct", "km_w_submax_mse_pct", "sc_w_submax_mse_pct",
        "base_w_max_mse_pct", "km_w_max_mse_pct", "sc_w_max_mse_pct",
    ]
    stats = {mat: {
        "base_act_mse": [], "km_act_mse": [], "sc_act_mse": [],
        "base_w_mse": [], "km_w_mse": [], "sc_w_mse": [],
        "base_out_mse": [], "km_out_mse": [], "sc_out_mse": [],
        **{k: [] for k in diag_keys},
    } for mat in matrix_names}
    
    print("Running experiments...\n")
    
    for block_idx, block in enumerate(blocks):
        block = block.to(device)
        if block_idx not in target_layers:
            for i in range(len(input_args)):
                with torch.no_grad():
                    a = to_device(input_args[i], device)
                    k = to_device(input_kwargs[i], device)
                    out = block(*a, **k)
                    input_args[i] = (maybe_first(out).cpu(),) + input_args[i][1:]
            block = block.cpu(); continue
        
        print(f"  Processing Layer {block_idx}...")
        block_copy = copy.deepcopy(block).to(device)
        act_caches = {}
        def hook_factory(name):
            def _hook(_, inp, out):
                if name not in act_caches: act_caches[name] = []
                # 修复 OOM：立刻加上 .cpu() 将张量卸载到主存
                # 并限制截取上限：对于评估 K-Means 的 MSE，16条序列(32768个Tokens)已经足够代表真实分布
                if len(act_caches[name]) < 16:
                    act_caches[name].append(inp[0].detach().cpu().float().view(-1, inp[0].shape[-1]))
            return _hook
        hooks = []
        hooks.append(block_copy.self_attn.q_proj.register_forward_hook(hook_factory("qkv")))
        hooks.append(block_copy.self_attn.o_proj.register_forward_hook(hook_factory("o")))
        hooks.append(block_copy.mlp.gate_proj.register_forward_hook(hook_factory("gate_up")))
        hooks.append(block_copy.mlp.down_proj.register_forward_hook(hook_factory("down")))
        
        for i in range(len(input_args)):
            with torch.no_grad():
                a = to_device(input_args[i], device)
                k = to_device(input_kwargs[i], device)
                block_copy(*a, **k)
        for h in hooks: h.remove()
        
        for mat_name in matrix_names:
            if mat_name not in act_caches: continue
            X = torch.cat(act_caches[mat_name], dim=0).to(device)
            W = get_combined_weight(block_copy, mat_name).to(device)
            
            dim = X.shape[1]
            max_samples = 4096
            if X.shape[0] > max_samples:
                idx = torch.linspace(0, X.shape[0]-1, max_samples, dtype=torch.long, device=device)
                X_abs_T = X[idx].abs().T.float()
                X_sub = X[idx]
            else:
                X_abs_T = X.abs().T.float()
                X_sub = X
            
            print(f"    - Matrix: {mat_name} | Computing KMeans FP4 Permutation...")
            
            if args.kmeans_alpha > 0.0:
                W_norm = torch.norm(W, p=2, dim=0) # Shape: (In_Features,)
                channel_weights = W_norm ** args.kmeans_alpha
            else:
                channel_weights = None
                
            perm_km = compute_kmeans_fp4_perm(X_abs_T, group_size=16, channel_weights=channel_weights, act_alpha=args.kmeans_act_alpha)
            del X_abs_T
            
            res = run_kmeans_experiment(X_sub, W, perm_km, group_size=16)
            
            del X, X_sub, W; torch.cuda.empty_cache()
            
            s = stats[mat_name]
            for k in res:
                if k in s:
                    s[k].append(res[k])
        
        del act_caches; gc.collect(); torch.cuda.empty_cache()
        del block_copy
        
        for i in range(len(input_args)):
            with torch.no_grad():
                a = to_device(input_args[i], device)
                k = to_device(input_kwargs[i], device)
                out = block(*a, **k)
                input_args[i] = (maybe_first(out).cpu(),) + input_args[i][1:]
        block = block.cpu(); torch.cuda.empty_cache()

    # ================================================================
    # Helper: 打印一组 MSE + 诊断数据
    # ================================================================
    def print_mse_table(title, get_vals):
        """get_vals(mat, key) -> value"""
        W_WIDTH = 130
        print(f"\n  {title}")
        print(f"{'─'*W_WIDTH}")
        print(f"  {'Matrix':<8} │ {'Act MSE':^36} │ {'Weight MSE':^36} │ {'Output MSE':^36}")
        print(f"  {'':<8} │ {'Base':>8}   {'KM':>8}   {'Scaled':>8}  {'(Δ)':>5} │ {'Base':>8}   {'KM':>8}   {'Scaled':>8}  {'(Δ)':>5} │ {'Base':>8}   {'KM':>8}   {'Scaled':>8}  {'(Δ)':>5}")
        print(f"  {'─'*8}─┼{'─'*36}─┼{'─'*36}─┼{'─'*36}")
        for mat in matrix_names:
            ba = get_vals(mat, "base_act_mse"); ka = get_vals(mat, "km_act_mse"); sa = get_vals(mat, "sc_act_mse")
            bw = get_vals(mat, "base_w_mse"); kw = get_vals(mat, "km_w_mse"); sw = get_vals(mat, "sc_w_mse")
            bo = get_vals(mat, "base_out_mse"); ko = get_vals(mat, "km_out_mse"); so = get_vals(mat, "sc_out_mse")
            if ba is None: continue
            da = (sa-ba)/ba*100 if ba else 0
            dw = (sw-bw)/bw*100 if bw else 0
            do = (so-bo)/bo*100 if bo else 0
            print(f"  {mat:<8} │ {ba:>8.2e} {ka:>8.2e} {sa:>8.2e} {da:>+5.1f}% │ {bw:>8.2e} {kw:>8.2e} {sw:>8.2e} {dw:>+5.1f}% │ {bo:>8.2e} {ko:>8.2e} {so:>8.2e} {do:>+5.1f}%")
        print(f"  {'─'*8}─┴{'─'*36}─┴{'─'*36}─┴{'─'*36}")
    
    def print_diag_table(title, get_vals):
        """打印 FP4 Grid 诊断表 (分成两行打印，避免过长)"""
        W_WIDTH = 120
        print(f"\n  {title} - Activation")
        print(f"{'─'*W_WIDTH}")
        print(f"  {'Matrix':<8} │ {'HLZS':^25} │ {'HLZ MSE%':^25} │ {'SubMax MSE%':^25} │ {'Max MSE%':^25}")
        print(f"  {'':<8} │ {'Base':>4} {'KM':>4} {'Scaled':>6} {'(Δ)':>5} │ {'Base':>4} {'KM':>4} {'Scaled':>6} {'(Δ)':>5} │ {'Base':>4} {'KM':>4} {'Scaled':>6} {'(Δ)':>5} │ {'Base':>4} {'KM':>4} {'Scaled':>6} {'(Δ)':>5}")
        print(f"  {'─'*8}─┼{'─'*26}┼{'─'*26}┼{'─'*26}┼{'─'*26}")
        for mat in matrix_names:
            bac = get_vals(mat, "base_act_hlzs"); kac = get_vals(mat, "km_act_hlzs"); sac = get_vals(mat, "sc_act_hlzs")
            if bac is None: continue
            bam = get_vals(mat, "base_act_hlz_mse_pct"); kam = get_vals(mat, "km_act_hlz_mse_pct"); sam = get_vals(mat, "sc_act_hlz_mse_pct")
            bas = get_vals(mat, "base_act_submax_mse_pct"); kas = get_vals(mat, "km_act_submax_mse_pct"); sas = get_vals(mat, "sc_act_submax_mse_pct")
            bax = get_vals(mat, "base_act_max_mse_pct"); kax = get_vals(mat, "km_act_max_mse_pct"); sax = get_vals(mat, "sc_act_max_mse_pct")
            
            dac = (sac-bac)/bac*100 if bac else 0
            dam = (sam-bam)/bam*100 if bam else 0
            das = (sas-bas)/bas*100 if bas else 0
            dax = (sax-bax)/bax*100 if bax else 0
            print(f"  {mat:<8} │ {bac:>4.1f} {kac:>4.1f} {sac:>6.1f} {dac:>+4.0f}% │ {bam:>4.1f} {kam:>4.1f} {sam:>6.1f} {dam:>+4.0f}% │ {bas:>4.1f} {kas:>4.1f} {sas:>6.1f} {das:>+4.0f}% │ {bax:>4.1f} {kax:>4.1f} {sax:>6.1f} {dax:>+4.0f}%")
        print(f"  {'─'*8}─┴{'─'*26}┴{'─'*26}┴{'─'*26}┴{'─'*26}")
        
        print(f"\n  {title} - Weight")
        print(f"{'─'*W_WIDTH}")
        print(f"  {'Matrix':<8} │ {'HLZS':^25} │ {'HLZ MSE%':^25} │ {'SubMax MSE%':^25} │ {'Max MSE%':^25}")
        print(f"  {'':<8} │ {'Base':>4} {'KM':>4} {'Scaled':>6} {'(Δ)':>5} │ {'Base':>4} {'KM':>4} {'Scaled':>6} {'(Δ)':>5} │ {'Base':>4} {'KM':>4} {'Scaled':>6} {'(Δ)':>5} │ {'Base':>4} {'KM':>4} {'Scaled':>6} {'(Δ)':>5}")
        print(f"  {'─'*8}─┼{'─'*26}┼{'─'*26}┼{'─'*26}┼{'─'*26}")
        for mat in matrix_names:
            bwc = get_vals(mat, "base_w_hlzs"); kwc = get_vals(mat, "km_w_hlzs"); swc = get_vals(mat, "sc_w_hlzs")
            if bwc is None: continue
            bwm = get_vals(mat, "base_w_hlz_mse_pct"); kwm = get_vals(mat, "km_w_hlz_mse_pct"); swm = get_vals(mat, "sc_w_hlz_mse_pct")
            bws = get_vals(mat, "base_w_submax_mse_pct"); kws = get_vals(mat, "km_w_submax_mse_pct"); sws = get_vals(mat, "sc_w_submax_mse_pct")
            bwx = get_vals(mat, "base_w_max_mse_pct"); kwx = get_vals(mat, "km_w_max_mse_pct"); swx = get_vals(mat, "sc_w_max_mse_pct")
            
            dwc = (swc-bwc)/bwc*100 if bwc else 0
            dwm = (swm-bwm)/bwm*100 if bwm else 0
            dws = (sws-bws)/bws*100 if bws else 0
            dwx = (swx-bwx)/bwx*100 if bwx else 0
            print(f"  {mat:<8} │ {bwc:>4.1f} {kwc:>4.1f} {swc:>6.1f} {dwc:>+4.0f}% │ {bwm:>4.1f} {kwm:>4.1f} {swm:>6.1f} {dwm:>+4.0f}% │ {bws:>4.1f} {kws:>4.1f} {sws:>6.1f} {dws:>+4.0f}% │ {bwx:>4.1f} {kwx:>4.1f} {swx:>6.1f} {dwx:>+4.0f}%")
        print(f"  {'─'*8}─┴{'─'*26}┴{'─'*26}┴{'─'*26}┴{'─'*26}")
    
    # ================================================================
    # 按层展示
    # ================================================================
    W_WIDTH = 110
    print("\n\n" + "=" * W_WIDTH)
    print("  KMeans FP4 Channel Resort (W4A4 NVFP4) 效果验证  (按层展示)")
    print("=" * W_WIDTH)
    
    print("\n  [ FP4 诊断指标说明 ]")
    print("  1. HLZS (High-Loss Zone Sum):")
    print("     含义: 所有在归一化后落入 FP4 高损失区(2.25~2.75, 3.25~3.75, 4.25~5.75)的大值，其原始绝对值的平均总和。")
    print("     说明: 相比单纯的数量，HLZS 同时反映了这些危险值的影响力。该值越小，说明量化对数据的破坏越少。")
    print("\n  2. HLZ MSE% (High-Loss Zone MSE Percentage):")
    print("     含义: 上述落入高损失区的大值，它们产生的量化误差占该矩阵总 MSE 的百分比。")
    print("     说明: 占比越低，说明非均匀网格最粗糙的部分对整体误差的破坏越小。")
    print("\n  3. SubMax MSE% (Sub-Maximum MSE Percentage):")
    print("     含义: 排除组内绝对值最大的那一个元素后，剩下所有归一化后 > 2.0 的次大值，产生的量化误差占总 MSE 的百分比。")
    print("     说明: 占比高意味着误差由“多个次大值”主导，KMeans 重排效果会很明显；占比低意味着误差由最大值主导。")
    print("\n  4. Max MSE% (Maximum Element MSE Percentage):")
    print("     含义: 组内绝对值最大的那一个元素(它决定了整组的 scale)，其自身的量化误差占总 MSE 的百分比。")
    print("     说明: 占比极高(如 down 矩阵高达60%+)说明量化痛点是极端的“一枝独秀”，此时 KMeans 作用微弱，应当转向 Intra-Group AWQ (组内缩放) 压制最大值。")
    print("=" * W_WIDTH)
    
    for i, layer_idx in enumerate(target_layers):
        def get_layer_val(mat, key, idx=i):
            s = stats[mat]
            if idx >= len(s.get(key, [])): return None
            return s[key][idx]
        
        print_mse_table(f"[ Layer {layer_idx} ] MSE", get_layer_val)
        print_diag_table(f"[ Layer {layer_idx} ] FP4 Grid Diagnostics", get_layer_val)

    # ================================================================
    # 多层平均
    # ================================================================
    print("\n\n" + "=" * W_WIDTH)
    print("  KMeans FP4 Channel Resort (W4A4 NVFP4) 效果验证  (多层平均)")
    print("=" * W_WIDTH)
    
    def get_avg_val(mat, key):
        vals = stats[mat].get(key, [])
        if not vals: return None
        return np.mean(vals)
    
    print_mse_table("[Average] MSE", get_avg_val)
    print_diag_table("[Average] FP4 Grid Diagnostics", get_avg_val)

if __name__ == "__main__":
    main()
