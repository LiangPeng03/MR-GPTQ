"""
test_hypothesis.py - 验证 KMeans_FP4 通道重排序失效原因的三个猜想

猜想1 (过拟合): 用校准数据 A 半计算排序 → 在 A 上评估(闭卷) vs 在 B 上评估(开卷)
猜想2 (逐Token波动): 输出逐 Token 的 MSE 分布，看是否存在少数 Token 暴涨
猜想3 (排列稳定性): 用两组不同的校准数据分别计算排列，比较它们的一致性
"""
import torch
import gc
import copy
import argparse
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.data_utils import get_data
from src.quantization.quantizer import Quantizer, get_reciprocal

class ForwardInterrupt(Exception):
    pass

class InputCollector(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, cpu_offload: bool = False):
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
from src.quantization.quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX

def apply_real_nvfp4_quant(x, group_size=16):
    device = x.device
    nv_q = Quantizer(bits=4, format="nvfp", granularity="group", group_size=group_size, symmetric=True, scale_precision="e4m3")
    nv_q._track_global_scale = False
    act_max = x.abs().max().to(torch.float32).view(1)
    nv_q.global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(act_max)).to(device)
    x_16 = x.contiguous().view(-1, group_size)
    sc, z = nv_q.get_quantization_params(x_16)
    q = nv_q(x_16, sc, z)
    return q.view(x.shape)

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


# ================================================================
# KMeans_FP4 排序算法 (复用自 test_cf.py)
# ================================================================
def compute_kmeans_fp4_perm(X_abs_T, group_size=16):
    """
    输入: X_abs_T shape=(dim, N_tokens)，每行是一个通道的绝对值序列
    输出: perm tensor of channel indices
    """
    dim = X_abs_T.shape[0]
    n_groups = dim // group_size
    device = X_abs_T.device
    grid_mults = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device, dtype=torch.float32)
    
    def compute_loss_distances(all_ch_abs, ref_ch_abs, chunk_size=256):
        ref = ref_ch_abs.unsqueeze(0)
        n_ch = all_ch_abs.shape[0]
        losses = torch.zeros(n_ch, device=device, dtype=torch.float32)
        for i in range(0, n_ch, chunk_size):
            chunk = all_ch_abs[i:i+chunk_size]
            p_max = torch.maximum(chunk, ref)
            p_min = torch.minimum(chunk, ref)
            scale = (p_max / 6.0).clamp(min=1e-10)
            grid = scale.unsqueeze(-1) * grid_mults
            diff = (p_min.unsqueeze(-1) - grid).abs()
            min_q = grid.gather(-1, diff.argmin(-1, keepdim=True)).squeeze(-1)
            losses[i:i+chunk_size] = ((p_min - min_q) ** 2).sum(dim=1)
        return losses
    
    ch_sums = X_abs_T.sum(dim=1)
    accumulated_dist = torch.zeros(dim, device=device, dtype=torch.float32)
    seed_order = []
    first_seed = ch_sums.argmax().item()
    seed_order.append(first_seed)
    accumulated_dist[first_seed] = -float('inf')
    dists = compute_loss_distances(X_abs_T, X_abs_T[first_seed])
    accumulated_dist += dists
    accumulated_dist[first_seed] = -float('inf')
    
    for k in range(1, n_groups):
        new_seed = accumulated_dist.argmax().item()
        seed_order.append(new_seed)
        accumulated_dist[new_seed] = -float('inf')
        dists = compute_loss_distances(X_abs_T, X_abs_T[new_seed])
        accumulated_dist += dists
    
    opt_groups = [[s] for s in seed_order]
    group_maxes = X_abs_T[seed_order].clone()
    group_sizes = torch.ones(n_groups, device=device, dtype=torch.long)
    seed_set = set(seed_order)
    remaining = [c for c in range(dim) if c not in seed_set]
    remaining_sums = ch_sums[remaining]
    sorted_order = torch.argsort(remaining_sums, descending=True)
    remaining_sorted = [remaining[i] for i in sorted_order.tolist()]
    
    for c in remaining_sorted:
        c_abs = X_abs_T[c]
        losses = torch.zeros(n_groups, device=device, dtype=torch.float32)
        ref = c_abs.unsqueeze(0)
        for i in range(0, n_groups, 256):
            g_chunk = group_maxes[i:i+256]
            p_max = torch.maximum(ref, g_chunk)
            p_min = torch.minimum(ref, g_chunk)
            scale = (p_max / 6.0).clamp(min=1e-10)
            grid = scale.unsqueeze(-1) * grid_mults
            diff = (p_min.unsqueeze(-1) - grid).abs()
            min_q = grid.gather(-1, diff.argmin(-1, keepdim=True)).squeeze(-1)
            losses[i:i+256] = ((p_min - min_q) ** 2).sum(dim=1)
        losses[group_sizes >= group_size] = float('inf')
        best_g = losses.argmin().item()
        opt_groups[best_g].append(c)
        group_maxes[best_g] = torch.maximum(group_maxes[best_g], c_abs)
        group_sizes[best_g] += 1
    
    perm = []
    for g in opt_groups: perm.extend(g)
    return torch.tensor(perm, device=device, dtype=torch.long)


# ================================================================
# 评估函数: 计算逐 Token 的激活量化 MSE (仅 A 量化)
# ================================================================
def eval_per_token_act_mse(X, W, perm, group_size=16, chunk_size=4096):
    """
    给定 X (N_tokens, dim), W (M_out, dim), perm (dim,)
    返回逐 Token 的 MSE: shape=(N_tokens,)
    """
    N_tokens = X.shape[0]
    per_token_mse = torch.zeros(N_tokens, device=X.device, dtype=torch.float32)
    
    with torch.no_grad():
        W_perm = W[:, perm]
        
        for i in range(0, N_tokens, chunk_size):
            X_chunk = X[i:i+chunk_size]
            X_perm_chunk = X_chunk[:, perm]
            
            # FP16 output
            Y_fp16_chunk = X_chunk @ W.T
            
            # Quantized output
            X_q_chunk = apply_real_nvfp4_quant(X_perm_chunk, group_size=group_size)
            Y_q_A_chunk = X_q_chunk @ W_perm.T
            
            # MSE
            per_token_mse[i:i+chunk_size] = ((Y_fp16_chunk - Y_q_A_chunk) ** 2).mean(dim=1)
            
            del Y_fp16_chunk, X_perm_chunk, X_q_chunk, Y_q_A_chunk
        torch.cuda.empty_cache()
        
    return per_token_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--dataset_name_or_path", type=str, default="fineweb-edu")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--num_sequences", type=int, default=32)
    args = parser.parse_args()
    
    device = "cuda"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # ================================================================
    # 加载两组校准数据（不同 seed）用于验证
    # ================================================================
    print("Loading calibration data (Set A: seed=42, Set B: seed=123)...")
    calib_data_A = get_data(args.dataset_name_or_path, tokenizer, args.sequence_length, args.num_sequences, seed=42)
    calib_data_B = get_data(args.dataset_name_or_path, tokenizer, args.sequence_length, args.num_sequences, seed=123)
    
    model.config.use_cache = False
    model.requires_grad_(False)
    
    blocks = model.model.layers
    
    # ================================================================
    # 用集合 A 跑 forward 得到各层输入
    # ================================================================
    print("Capturing layer inputs with Set A...")
    blocks[0] = InputCollector(blocks[0], cpu_offload=False)
    model.get_input_embeddings().to(device)
    blocks[0] = blocks[0].to(device)
    for sample in calib_data_A:
        try:
            with torch.no_grad(): model(sample.to(device))
        except ForwardInterrupt:
            pass
    input_args_A = blocks[0].input_args
    input_kwargs_A = blocks[0].input_kwargs
    blocks[0] = blocks[0].module.cpu()
    
    # 用集合 B 跑 forward 得到各层输入
    print("Capturing layer inputs with Set B...")
    blocks[0] = InputCollector(blocks[0], cpu_offload=False)
    blocks[0] = blocks[0].to(device)
    for sample in calib_data_B:
        try:
            with torch.no_grad(): model(sample.to(device))
        except ForwardInterrupt:
            pass
    input_args_B = blocks[0].input_args
    input_kwargs_B = blocks[0].input_kwargs
    blocks[0] = blocks[0].module.cpu()
    model.get_input_embeddings().cpu()
    
    target_layers = [0, 15, 31]
    matrix_names = ["qkv", "o", "gate_up", "down"]
    
    for block_idx, block in enumerate(blocks):
        block = block.to(device)
        
        if block_idx not in target_layers:
            print(f"Skipping layer {block_idx} (FP16 Pass)...")
            for i in range(len(input_args_A)):
                with torch.no_grad():
                    args_on_dev = to_device(input_args_A[i], device)
                    kwargs_on_dev = to_device(input_kwargs_A[i], device)
                    out = block(*args_on_dev, **kwargs_on_dev)
                    out_hidden = maybe_first(out).cpu()
                    input_args_A[i] = (out_hidden,) + input_args_A[i][1:]
            for i in range(len(input_args_B)):
                with torch.no_grad():
                    args_on_dev = to_device(input_args_B[i], device)
                    kwargs_on_dev = to_device(input_kwargs_B[i], device)
                    out = block(*args_on_dev, **kwargs_on_dev)
                    out_hidden = maybe_first(out).cpu()
                    input_args_B[i] = (out_hidden,) + input_args_B[i][1:]
            block = block.cpu()
            continue
        
        print(f"\n{'='*70}")
        print(f"==== Layer {block_idx}: Hypothesis Verification ====")
        print(f"{'='*70}")
        
        # ============================================================
        # 收集激活值
        # ============================================================
        def collect_activations(block_module, input_args_list, input_kwargs_list):
            """Run forward and collect per-matrix activations."""
            act_caches = {}
            def hook_factory(name):
                def _hook(_, inp, out):
                    if name not in act_caches: act_caches[name] = []
                    act_caches[name].append(inp[0].detach().float().view(-1, inp[0].shape[-1]))
                return _hook
            
            hooks = []
            hooks.append(block_module.self_attn.q_proj.register_forward_hook(hook_factory("qkv")))
            hooks.append(block_module.self_attn.o_proj.register_forward_hook(hook_factory("o")))
            hooks.append(block_module.mlp.gate_proj.register_forward_hook(hook_factory("gate_up")))
            hooks.append(block_module.mlp.down_proj.register_forward_hook(hook_factory("down")))
            
            for i in range(len(input_args_list)):
                with torch.no_grad():
                    args_on_dev = to_device(input_args_list[i], device)
                    kwargs_on_dev = to_device(input_kwargs_list[i], device)
                    block_module(*args_on_dev, **kwargs_on_dev)
            for h in hooks: h.remove()
            
            result = {}
            for name, caches in act_caches.items():
                result[name] = torch.cat(caches, dim=0).to(device)
            return result
        
        block_copy = copy.deepcopy(block).to(device)
        
        print("  Collecting activations from Set A...")
        acts_A = collect_activations(block_copy, input_args_A, input_kwargs_A)
        print("  Collecting activations from Set B...")
        acts_B = collect_activations(block_copy, input_args_B, input_kwargs_B)
        
        del block_copy
        torch.cuda.empty_cache()
        
        for mat_name in matrix_names:
            if mat_name not in acts_A or mat_name not in acts_B:
                continue
            
            X_A = acts_A[mat_name]  # 用集合 A 的激活值
            X_B = acts_B[mat_name]  # 用集合 B 的激活值
            W = get_combined_weight(block, mat_name).to(device)
            dim = X_A.shape[1]
            
            print(f"\n  --- {mat_name} (dim={dim}) ---")
            
            # ============================================================
            # 计算排列
            # ============================================================
            # Baseline (identity)
            perm_baseline = torch.arange(dim, device=device, dtype=torch.long)
            
            # Perm from Set A
            max_samples = 4096
            if X_A.shape[0] > max_samples:
                subset_idx = torch.linspace(0, X_A.shape[0] - 1, max_samples, dtype=torch.long, device=device)
                X_abs_T_A = X_A[subset_idx].abs().T.float()
            else:
                X_abs_T_A = X_A.abs().T.float()
            perm_A = compute_kmeans_fp4_perm(X_abs_T_A, group_size=16)
            
            # Perm from Set B
            if X_B.shape[0] > max_samples:
                subset_idx = torch.linspace(0, X_B.shape[0] - 1, max_samples, dtype=torch.long, device=device)
                X_abs_T_B = X_B[subset_idx].abs().T.float()
            else:
                X_abs_T_B = X_B.abs().T.float()
            perm_B = compute_kmeans_fp4_perm(X_abs_T_B, group_size=16)
            
            del X_abs_T_A, X_abs_T_B
            torch.cuda.empty_cache()
            
            # ============================================================
            # 猜想 3: 排列稳定性 (先输出，因为不需要 MSE 计算)
            # ============================================================
            # 把两个排列都转换成 "group membership" 数组
            group_of_A = torch.zeros(dim, device=device, dtype=torch.long)
            group_of_B = torch.zeros(dim, device=device, dtype=torch.long)
            for g in range(dim // 16):
                for c in range(16):
                    group_of_A[perm_A[g * 16 + c]] = g
                    group_of_B[perm_B[g * 16 + c]] = g
            
            # 计算同组一致率：在 perm_A 中同组的通道对，有多少比例在 perm_B 中也同组？
            n_pairs_same = 0
            n_pairs_total = 0
            for g in range(dim // 16):
                members_A = perm_A[g * 16: (g + 1) * 16].tolist()
                for i in range(len(members_A)):
                    for j in range(i + 1, len(members_A)):
                        n_pairs_total += 1
                        if group_of_B[members_A[i]] == group_of_B[members_A[j]]:
                            n_pairs_same += 1
            
            perm_consistency = n_pairs_same / max(n_pairs_total, 1) * 100
            
            # 理论随机基线：一个组16个通道中随机抽2个同组的概率 = 15/(dim-1)
            random_baseline = 15.0 / (dim - 1) * 100
            
            print(f"  [猜想3] 排列稳定性:")
            print(f"    同组一致率 = {perm_consistency:.2f}%  (随机基线 = {random_baseline:.2f}%)")
            if perm_consistency < random_baseline * 3:
                print(f"    ⚠ 排列极不稳定! 接近随机水平 → 重排序高度依赖输入数据")
            elif perm_consistency > 50:
                print(f"    ✓ 排列较稳定 (>50%) → 排列不是主要问题")
            else:
                print(f"    △ 排列中等稳定 → 部分过拟合，但不完全随机")
            
            # ============================================================
            # 猜想 1: 闭卷 vs 开卷 (Cross-validation)
            # ============================================================
            print(f"\n  [猜想1] 闭卷 vs 开卷 (Cross-validation):")
            
            # Baseline MSE on both sets (用于百分比比较)
            mse_baseline_A = eval_per_token_act_mse(X_A, W, perm_baseline).mean().item()
            mse_baseline_B = eval_per_token_act_mse(X_B, W, perm_baseline).mean().item()
            
            # 闭卷：用 A 数据算排列 → 在 A 数据上评估
            mse_closed = eval_per_token_act_mse(X_A, W, perm_A).mean().item()
            closed_improve = (mse_closed - mse_baseline_A) / (mse_baseline_A + 1e-12) * 100
            
            # 开卷：用 A 数据算排列 → 在 B 数据上评估
            mse_open = eval_per_token_act_mse(X_B, W, perm_A).mean().item()
            open_improve = (mse_open - mse_baseline_B) / (mse_baseline_B + 1e-12) * 100
            
            # 交叉：用 B 数据算排列 → 在 A 数据上评估
            mse_cross = eval_per_token_act_mse(X_A, W, perm_B).mean().item()
            cross_improve = (mse_cross - mse_baseline_A) / (mse_baseline_A + 1e-12) * 100
            
            print(f"    Baseline MSE (A): {mse_baseline_A:.6e}")
            print(f"    Baseline MSE (B): {mse_baseline_B:.6e}")
            print(f"    闭卷 (train=A, eval=A): {mse_closed:.6e} ({closed_improve:+.2f}%)")
            print(f"    开卷 (train=A, eval=B): {mse_open:.6e} ({open_improve:+.2f}%)")
            print(f"    交叉 (train=B, eval=A): {mse_cross:.6e} ({cross_improve:+.2f}%)")
            
            gap = abs(closed_improve) - abs(open_improve)
            if gap > 5:
                print(f"    ⚠ 闭卷/开卷差距 = {gap:.1f}pp → 存在显著过拟合")
            else:
                print(f"    ✓ 闭卷/开卷差距 = {gap:.1f}pp → 过拟合不明显，排列可泛化")
            
            # ============================================================
            # 猜想 2: 逐 Token MSE 分布分析
            # ============================================================
            print(f"\n  [猜想2] 逐Token MSE分布 (A-only quant, eval on Set A):")
            
            per_token_baseline = eval_per_token_act_mse(X_A, W, perm_baseline)
            per_token_kmeans = eval_per_token_act_mse(X_A, W, perm_A)
            
            # 计算分位数
            percentiles = [50, 90, 95, 99, 100]
            print(f"    {'Percentile':>12} | {'Baseline':>14} | {'KMeans_FP4':>14} | {'Change':>10}")
            print(f"    {'-'*58}")
            
            worst_pct_change = 0
            for p in percentiles:
                if p == 100:
                    base_val = per_token_baseline.max().item()
                    km_val = per_token_kmeans.max().item()
                    label = "Max"
                else:
                    base_val = torch.quantile(per_token_baseline.float(), p / 100).item()
                    km_val = torch.quantile(per_token_kmeans.float(), p / 100).item()
                    label = f"P{p}"
                pct_change = (km_val - base_val) / (base_val + 1e-12) * 100
                print(f"    {label:>12} | {base_val:>14.6e} | {km_val:>14.6e} | {pct_change:>+9.2f}%")
                if p >= 95:
                    worst_pct_change = max(worst_pct_change, pct_change)
            
            # 统计变好和变差的 Token 比例
            improved = (per_token_kmeans < per_token_baseline).float().mean().item() * 100
            worsened = (per_token_kmeans > per_token_baseline).float().mean().item() * 100
            
            # 变差超过2倍的Token比例
            severely_worsened = (per_token_kmeans > per_token_baseline * 2).float().mean().item() * 100
            
            print(f"\n    变好的 Token: {improved:.1f}%")
            print(f"    变差的 Token: {worsened:.1f}%")
            print(f"    变差超过2倍: {severely_worsened:.2f}%")
            
            if worst_pct_change > 10:
                print(f"    ⚠ 尾部 Token MSE 恶化 {worst_pct_change:.1f}% → PPL 对极端值敏感，尾部恶化拖垮整体")
            elif improved < 50:
                print(f"    ⚠ 多数 Token 实际上变差了 → 重排序的收益不具有普遍性")
            else:
                print(f"    ✓ 多数 Token 受益，尾部也未明显恶化")
            
            del per_token_baseline, per_token_kmeans
            del X_A, X_B, W, perm_A, perm_B, perm_baseline
            torch.cuda.empty_cache()
        
        del acts_A, acts_B
        gc.collect()
        torch.cuda.empty_cache()
        
        # Advance FP16 state
        print(f"\n  Advancing FP16 state to next layer...")
        for i in range(len(input_args_A)):
            with torch.no_grad():
                args_on_dev = to_device(input_args_A[i], device)
                kwargs_on_dev = to_device(input_kwargs_A[i], device)
                out = block(*args_on_dev, **kwargs_on_dev)
                out_hidden = maybe_first(out).cpu()
                input_args_A[i] = (out_hidden,) + input_args_A[i][1:]
        for i in range(len(input_args_B)):
            with torch.no_grad():
                args_on_dev = to_device(input_args_B[i], device)
                kwargs_on_dev = to_device(input_kwargs_B[i], device)
                out = block(*args_on_dev, **kwargs_on_dev)
                out_hidden = maybe_first(out).cpu()
                input_args_B[i] = (out_hidden,) + input_args_B[i][1:]
        block = block.cpu()
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("All hypothesis tests completed.")
    print("=" * 70)

if __name__ == "__main__":
    main()
