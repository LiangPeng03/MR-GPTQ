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

def compute_kmeans_fp4_perm(X_abs_T, group_size=16):
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

def run_kmeans_experiment(X, W, perm_km, group_size=16):
    results = {}
    
    # 1. True FP16 Output
    # X shape: (num_tokens, in_features)
    # W shape: (out_features, in_features)
    out_true = X @ W.T
    
    # 2. Baseline
    X_q_base = simulate_nvfp4(X, group_size)
    W_q_base = simulate_nvfp4(W, group_size)
    out_base = X_q_base @ W_q_base.T
    
    results["base_act_mse"] = ((X_q_base - X) ** 2).mean().item()
    results["base_w_mse"] = ((W_q_base - W) ** 2).mean().item()
    results["base_out_mse"] = ((out_base - out_true) ** 2).mean().item()
    
    # 3. KMeans Permutation
    X_perm = X[:, perm_km]
    W_perm = W[:, perm_km]
    
    X_q_km = simulate_nvfp4(X_perm, group_size)
    W_q_km = simulate_nvfp4(W_perm, group_size)
    
    # The output mathematically matches because we permuted both columns of X and columns of W.
    out_km = X_q_km @ W_q_km.T
    
    results["km_act_mse"] = ((X_q_km - X_perm) ** 2).mean().item()
    results["km_w_mse"] = ((W_q_km - W_perm) ** 2).mean().item()
    results["km_out_mse"] = ((out_km - out_true) ** 2).mean().item()
    
    return results

# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--dataset_name_or_path", type=str, default="fineweb-edu")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--num_sequences", type=int, default=128)
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
    
    print("Capturing layer inputs...")
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
    stats = {mat: {
        "base_act_mse": [], "km_act_mse": [],
        "base_w_mse": [], "km_w_mse": [],
        "base_out_mse": [], "km_out_mse": [],
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
            perm_km = compute_kmeans_fp4_perm(X_abs_T, group_size=16)
            del X_abs_T
            
            res = run_kmeans_experiment(X_sub, W, perm_km, group_size=16)
            
            del X, X_sub, W; torch.cuda.empty_cache()
            
            s = stats[mat_name]
            s["base_act_mse"].append(res["base_act_mse"])
            s["km_act_mse"].append(res["km_act_mse"])
            s["base_w_mse"].append(res["base_w_mse"])
            s["km_w_mse"].append(res["km_w_mse"])
            s["base_out_mse"].append(res["base_out_mse"])
            s["km_out_mse"].append(res["km_out_mse"])
        
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
    # 精简全局总结
    # ================================================================
    W_WIDTH = 110
    print("\n\n" + "=" * W_WIDTH)
    print("  KMeans FP4 Channel Resort (W4A4 NVFP4) 效果验证  (多层平均: L0, L15, L31)")
    print("=" * W_WIDTH)

    print(f"\n{'─'*W_WIDTH}")
    print(f"  {'Matrix':<8} │ {'Act MSE':^28} │ {'Weight MSE':^28} │ {'Output MSE':^28}")
    print(f"  {'':<8} │ {'Base':>8}   {'KMeans':>8}   {'(Δ)':>6} │ {'Base':>8}   {'KMeans':>8}   {'(Δ)':>6} │ {'Base':>8}   {'KMeans':>8}   {'(Δ)':>6}")
    print(f"  {'─'*8}─┼{'─'*28}─┼{'─'*28}─┼{'─'*28}")
    
    for mat in matrix_names:
        s = stats[mat]
        ba = np.mean(s["base_act_mse"]); ka = np.mean(s["km_act_mse"])
        bw = np.mean(s["base_w_mse"]); kw = np.mean(s["km_w_mse"])
        bo = np.mean(s["base_out_mse"]); ko = np.mean(s["km_out_mse"])
        
        da = (ka-ba)/ba*100 if ba else 0
        dw = (kw-bw)/bw*100 if bw else 0
        do = (ko-bo)/bo*100 if bo else 0
        
        print(f"  {mat:<8} │ {ba:>8.2e}  {ka:>8.2e} {da:>+5.1f}% │ {bw:>8.2e}  {kw:>8.2e} {dw:>+5.1f}% │ {bo:>8.2e}  {ko:>8.2e} {do:>+5.1f}%")

    print(f"  {'─'*8}─┼{'─'*28}─┼{'─'*28}─┼{'─'*28}")

if __name__ == "__main__":
    main()
