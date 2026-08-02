"""
test_scale_opt.py - 验证"MSE-Optimal Scale"与闭式解(LSS)对NVFP4激活量化的改善效果
证明LSS对激活量化损失的减小能力以及增加的推理时间开销
"""
import torch
import gc
import copy
import argparse
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.data_utils import get_data
from src.quantization.quantizer import Quantizer, get_reciprocal
from src.quantization.quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX

class ForwardInterrupt(Exception):
    pass

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

# 方法1: MinMax (Baseline) - 量化+反量化
def quantize_minmax(x_groups, global_scale):
    abs_max = x_groups.abs().amax(dim=1, keepdim=True)
    raw_scales = abs_max / 6.0
    raw_scales[raw_scales == 0] = 1.0
    scales = scale_to_e4m3(raw_scales, global_scale)
    
    x_normalized = x_groups / scales
    x_q = cast_to_fp4(x_normalized)
    return x_q * scales

# 方法1: MinMax (Baseline) - 仅量化
def quantize_minmax_only(x_groups, global_scale):
    abs_max = x_groups.abs().amax(dim=1, keepdim=True)
    raw_scales = abs_max / 6.0
    raw_scales[raw_scales == 0] = 1.0
    scales = scale_to_e4m3(raw_scales, global_scale)
    
    x_normalized = x_groups / scales
    x_q = cast_to_fp4(x_normalized)
    return x_q, scales

# 方法2: LSS 闭式解 - 量化+反量化
def quantize_lss(x_groups, global_scale):
    abs_x = x_groups.abs()
    
    # 1. 基础 MinMax 寻找初步格点
    abs_max = abs_x.amax(dim=1, keepdim=True)
    s_0 = abs_max / 6.0
    s_0[s_0 == 0] = 1.0
    
    # 2. 试投格点
    x_norm = abs_x / s_0
    g = cast_to_fp4(x_norm).abs()
    
    # 3. 闭式解计算最佳 Scale
    num = (abs_x * g).sum(dim=1, keepdim=True)
    den = (g * g).sum(dim=1, keepdim=True)
    
    den[den == 0] = 1.0
    raw_scales = num / den
    raw_scales[abs_max == 0] = 1.0
    
    # 4. 转换精度并真正量化
    scales = scale_to_e4m3(raw_scales, global_scale)
    x_normalized = x_groups / scales
    x_q = cast_to_fp4(x_normalized)
    return x_q * scales

# 方法2: LSS 闭式解 - 仅量化
def quantize_lss_only(x_groups, global_scale):
    abs_x = x_groups.abs()
    
    # 1. 基础 MinMax 寻找初步格点
    abs_max = abs_x.amax(dim=1, keepdim=True)
    s_0 = abs_max / 6.0
    s_0[s_0 == 0] = 1.0
    
    # 2. 试投格点
    x_norm = abs_x / s_0
    g = cast_to_fp4(x_norm).abs()
    
    # 3. 闭式解计算最佳 Scale
    num = (abs_x * g).sum(dim=1, keepdim=True)
    den = (g * g).sum(dim=1, keepdim=True)
    
    den[den == 0] = 1.0
    raw_scales = num / den
    raw_scales[abs_max == 0] = 1.0
    
    # 4. 转换精度并真正量化
    scales = scale_to_e4m3(raw_scales, global_scale)
    x_normalized = x_groups / scales
    x_q = cast_to_fp4(x_normalized)
    return x_q, scales

# 方法3: 4/6 (Four Over Six) 自适应 block scale - 量化+反量化
def quantize_4o6(x_groups, global_scale, scale_rule='mse'):
    """
    Four Over Six (4/6): Adaptive block scale selection for NVFP4.
    对每个 group 同时尝试 scale=abs_max/6 (标准) 和 scale=abs_max/4，
    按给定 rule (mse/mae/abs_max) 逐 group 择优。
    """
    abs_x = x_groups.abs()
    abs_max = abs_x.amax(dim=1, keepdim=True)
    
    # Scale candidate 1: /6 (standard NVFP4)
    s_6 = abs_max / 6.0
    s_6[s_6 == 0] = 1.0
    scales_6 = scale_to_e4m3(s_6, global_scale)
    
    # Scale candidate 2: /4 (four over six)
    s_4 = abs_max / 4.0
    s_4[s_4 == 0] = 1.0
    scales_4 = scale_to_e4m3(s_4, global_scale)
    
    # Quantize using both scales
    x_q_6 = cast_to_fp4(x_groups / scales_6)
    x_deq_6 = x_q_6 * scales_6
    x_q_4 = cast_to_fp4(x_groups / scales_4)
    x_deq_4 = x_q_4 * scales_4
    
    # Per-group error comparison
    if scale_rule == 'mse':
        err_6 = ((x_groups - x_deq_6) ** 2).sum(dim=1)
        err_4 = ((x_groups - x_deq_4) ** 2).sum(dim=1)
    elif scale_rule == 'mae':
        err_6 = (x_groups - x_deq_6).abs().sum(dim=1)
        err_4 = (x_groups - x_deq_4).abs().sum(dim=1)
    elif scale_rule == 'abs_max':
        err_6 = (x_groups - x_deq_6).abs().max(dim=1).values
        err_4 = (x_groups - x_deq_4).abs().max(dim=1).values
    else:
        raise ValueError(f"Unknown scale_rule: {scale_rule}")
    
    # Select better scale per group
    select_4 = (err_4 < err_6).unsqueeze(1)
    x_q = torch.where(select_4, x_q_4, x_q_6)
    scales = torch.where(select_4, scales_4, scales_6)
    
    return x_q * scales

# 方法3: 4/6 (Four Over Six) - 仅量化
def quantize_4o6_only(x_groups, global_scale, scale_rule='mse'):
    """仅量化，输出 (quantized_values, scales) — 真实推理路径"""
    abs_x = x_groups.abs()
    abs_max = abs_x.amax(dim=1, keepdim=True)
    
    s_6 = abs_max / 6.0
    s_6[s_6 == 0] = 1.0
    scales_6 = scale_to_e4m3(s_6, global_scale)
    
    s_4 = abs_max / 4.0
    s_4[s_4 == 0] = 1.0
    scales_4 = scale_to_e4m3(s_4, global_scale)
    
    x_q_6 = cast_to_fp4(x_groups / scales_6)
    x_deq_6 = x_q_6 * scales_6
    x_q_4 = cast_to_fp4(x_groups / scales_4)
    x_deq_4 = x_q_4 * scales_4
    
    if scale_rule == 'mse':
        err_6 = ((x_groups - x_deq_6) ** 2).sum(dim=1)
        err_4 = ((x_groups - x_deq_4) ** 2).sum(dim=1)
    elif scale_rule == 'mae':
        err_6 = (x_groups - x_deq_6).abs().sum(dim=1)
        err_4 = (x_groups - x_deq_4).abs().sum(dim=1)
    elif scale_rule == 'abs_max':
        err_6 = (x_groups - x_deq_6).abs().max(dim=1).values
        err_4 = (x_groups - x_deq_4).abs().max(dim=1).values
    
    select_4 = (err_4 < err_6).unsqueeze(1)
    x_q = torch.where(select_4, x_q_4, x_q_6)
    scales = torch.where(select_4, scales_4, scales_6)
    
    return x_q, scales

# 方法4: LSS 3 轮迭代 - 量化+反量化
def quantize_lss_3round(x_groups, global_scale):
    abs_x = x_groups.abs()

    # 1. 基础 MinMax 寻找初步格点
    abs_max = abs_x.amax(dim=1, keepdim=True)
    s_0 = abs_max / 6.0
    s_0[s_0 == 0] = 1.0

    # 2. 试投格点
    x_norm = abs_x / s_0
    g = cast_to_fp4(x_norm).abs()

    # 3. 闭式解计算最佳 Scale (第1轮)
    num = (abs_x * g).sum(dim=1, keepdim=True)
    den = (g * g).sum(dim=1, keepdim=True)
    den[den == 0] = 1.0
    raw_scales = num / den
    raw_scales[abs_max == 0] = 1.0

    # 第2轮: 用第1轮的 scale 重新投格点
    scales_r1 = scale_to_e4m3(raw_scales, global_scale)
    x_norm_r2 = abs_x / scales_r1
    g_r2 = cast_to_fp4(x_norm_r2).abs()
    num_r2 = (abs_x * g_r2).sum(dim=1, keepdim=True)
    den_r2 = (g_r2 * g_r2).sum(dim=1, keepdim=True)
    den_r2[den_r2 == 0] = 1.0
    raw_scales_r2 = num_r2 / den_r2
    raw_scales_r2[abs_max == 0] = 1.0

    # 第3轮: 用第2轮的 scale 重新投格点
    scales_r2 = scale_to_e4m3(raw_scales_r2, global_scale)
    x_norm_r3 = abs_x / scales_r2
    g_r3 = cast_to_fp4(x_norm_r3).abs()
    num_r3 = (abs_x * g_r3).sum(dim=1, keepdim=True)
    den_r3 = (g_r3 * g_r3).sum(dim=1, keepdim=True)
    den_r3[den_r3 == 0] = 1.0
    raw_scales_r3 = num_r3 / den_r3
    raw_scales_r3[abs_max == 0] = 1.0

    # 最终量化
    scales = scale_to_e4m3(raw_scales_r3, global_scale)
    x_normalized = x_groups / scales
    x_q = cast_to_fp4(x_normalized)
    return x_q * scales


# 方法4: LSS 3 轮迭代 - 仅量化
def quantize_lss_3round_only(x_groups, global_scale):
    abs_x = x_groups.abs()

    abs_max = abs_x.amax(dim=1, keepdim=True)
    s_0 = abs_max / 6.0
    s_0[s_0 == 0] = 1.0

    x_norm = abs_x / s_0
    g = cast_to_fp4(x_norm).abs()

    num = (abs_x * g).sum(dim=1, keepdim=True)
    den = (g * g).sum(dim=1, keepdim=True)
    den[den == 0] = 1.0
    raw_scales = num / den
    raw_scales[abs_max == 0] = 1.0

    scales_r1 = scale_to_e4m3(raw_scales, global_scale)
    x_norm_r2 = abs_x / scales_r1
    g_r2 = cast_to_fp4(x_norm_r2).abs()
    num_r2 = (abs_x * g_r2).sum(dim=1, keepdim=True)
    den_r2 = (g_r2 * g_r2).sum(dim=1, keepdim=True)
    den_r2[den_r2 == 0] = 1.0
    raw_scales_r2 = num_r2 / den_r2
    raw_scales_r2[abs_max == 0] = 1.0

    scales_r2 = scale_to_e4m3(raw_scales_r2, global_scale)
    x_norm_r3 = abs_x / scales_r2
    g_r3 = cast_to_fp4(x_norm_r3).abs()
    num_r3 = (abs_x * g_r3).sum(dim=1, keepdim=True)
    den_r3 = (g_r3 * g_r3).sum(dim=1, keepdim=True)
    den_r3[den_r3 == 0] = 1.0
    raw_scales_r3 = num_r3 / den_r3
    raw_scales_r3[abs_max == 0] = 1.0

    scales = scale_to_e4m3(raw_scales_r3, global_scale)
    x_normalized = x_groups / scales
    x_q = cast_to_fp4(x_normalized)
    return x_q, scales


# 权重量化使用 MSE 搜索 (模拟真实 MR-GPTQ 行为)
# 搜索范围: init_raw_scales 的 0.5 ~ 1.1 倍
def quantize_weight_mse(w_groups, global_scale, scale_search_iters=100, scale_min_factor=0.5, scale_max_factor=1.1):
    abs_max = w_groups.abs().amax(dim=1, keepdim=True)
    init_raw_scales = abs_max / 6.0
    init_raw_scales[init_raw_scales == 0] = 1.0
    
    best_quantization_error = torch.full((w_groups.shape[0],), float("inf"), device=w_groups.device)
    best_scales = torch.zeros_like(init_raw_scales)
    
    for i in range(scale_search_iters + 1):
        scale_factor = scale_min_factor + i * (scale_max_factor - scale_min_factor) / scale_search_iters
        candidate_scales = scale_factor * init_raw_scales
        
        x_normalized = w_groups / candidate_scales
        x_q = cast_to_fp4(x_normalized)
        x_dequant = x_q * candidate_scales
        
        error = (w_groups - x_dequant).abs().pow(2.4).sum(dim=1)
        improved = error < best_quantization_error
        if improved.any():
            best_quantization_error[improved] = error[improved]
            best_scales[improved] = candidate_scales[improved]
            
    best_scales = scale_to_e4m3(best_scales, global_scale)
    x_normalized = w_groups / best_scales
    x_q = cast_to_fp4(x_normalized)
    best_dequant = x_q * best_scales
    
    return best_dequant

# 评估函数: 计算纯激活量化误差（分批处理，节省显存）
def eval_quant_error_only(X_list, quant_fn, global_scale, group_size=16, chunk_size=512):
    """X_list: list of activation tensors (already on device), or a single tensor.
    分批计算 MSE，避免一次性加载全部数据到 GPU。"""
    total_sq_err = 0.0
    total_n = 0
    with torch.no_grad():
        if isinstance(X_list, torch.Tensor):
            X_list = [X_list]
        for X_chunk in X_list:
            if X_chunk.numel() == 0:
                continue
            # 进一步切成 sub-chunks 以控制峰值显存
            N = X_chunk.shape[0]
            for start in range(0, N, chunk_size):
                X_sub = X_chunk[start:start+chunk_size]
                X_groups = X_sub.contiguous().view(-1, group_size)
                X_q = quant_fn(X_groups, global_scale).view(X_sub.shape)
                total_sq_err += ((X_sub - X_q) ** 2).sum().item()
                total_n += X_sub.numel()
    return total_sq_err / total_n if total_n > 0 else 0.0

def eval_output_mse(X_list, W, quant_fn, global_scale_X, global_scale_W, group_size=16, chunk_size=512):
    """X_list: list of activation tensors (already on device), or a single tensor.
    分批计算 output MSE，避免一次性加载全部数据到 GPU。"""
    total_sq_err = 0.0
    total_tokens = 0
    out_dim = W.shape[0]
    with torch.no_grad():
        W_groups = W.contiguous().view(-1, group_size)
        # 用 MSE 搜索量化权重（一次性，W 单层不会太大）
        W_q = quantize_weight_mse(W_groups, global_scale_W).view(W.shape)
        if isinstance(X_list, torch.Tensor):
            X_list = [X_list]
        for X_chunk in X_list:
            if X_chunk.numel() == 0:
                continue
            N = X_chunk.shape[0]
            for start in range(0, N, chunk_size):
                X_sub = X_chunk[start:start+chunk_size]
                Y_fp16 = X_sub @ W.T
                X_groups = X_sub.contiguous().view(-1, group_size)
                X_q = quant_fn(X_groups, global_scale_X).view(X_sub.shape)
                Y_q = X_q @ W_q.T
                total_sq_err += ((Y_fp16 - Y_q) ** 2).sum().item()
                total_tokens += X_sub.shape[0]
    return total_sq_err / (total_tokens * out_dim) if total_tokens > 0 else 0.0

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
    
    print("Loading calibration data...")
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
        except ForwardInterrupt:
            pass
    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module.cpu()
    model.get_input_embeddings().cpu()
    
    target_layers = [0, 15,31]
    matrix_names = ["qkv", "o", "gate_up", "down"]
    
    strategies = {
        "MinMax (baseline)": quantize_minmax,
        "LSS (Closed-form)": quantize_lss,
        "4/6 (MSE)": quantize_4o6,
        "LSS-3R (3-round)": quantize_lss_3round,
    }
    
    # 汇总表数据结构: summary[matrix_name][strategy_name] = {"act_sum": ..., "out_sum": ..., "count": ...}
    summary = {m: {s: {"act_sum": 0.0, "out_sum": 0.0, "count": 0} for s in strategies} for m in matrix_names}
    
    print("\n" + "=" * 110)
    print(f"{'Layer':<5} | {'Matrix':<8} | {'Strategy':<18} | {'Act Quant MSE':<20} | {'Output MSE(W4A4)':<20}")
    print("-" * 110)
    
    for block_idx, block in enumerate(blocks):
        block = block.to(device)
        if block_idx not in target_layers:
            for i in range(len(input_args)):
                with torch.no_grad():
                    args_on_dev = to_device(input_args[i], device)
                    kwargs_on_dev = to_device(input_kwargs[i], device)
                    out = block(*args_on_dev, **kwargs_on_dev)
                    out_hidden = maybe_first(out).cpu()
                    input_args[i] = (out_hidden,) + input_args[i][1:]
            block = block.cpu()
            continue
        
        block_copy = copy.deepcopy(block).to(device)
        act_caches = {}
        def hook_factory(name):
            def _hook(_, inp, out):
                if name not in act_caches: act_caches[name] = []
                act_caches[name].append(inp[0].detach().float().view(-1, inp[0].shape[-1]))
            return _hook
        
        hooks = []
        hooks.append(block_copy.self_attn.q_proj.register_forward_hook(hook_factory("qkv")))
        hooks.append(block_copy.self_attn.o_proj.register_forward_hook(hook_factory("o")))
        hooks.append(block_copy.mlp.gate_proj.register_forward_hook(hook_factory("gate_up")))
        hooks.append(block_copy.mlp.down_proj.register_forward_hook(hook_factory("down")))
        
        for i in range(len(input_args)):
            with torch.no_grad():
                args_on_dev = to_device(input_args[i], device)
                kwargs_on_dev = to_device(input_kwargs[i], device)
                block_copy(*args_on_dev, **kwargs_on_dev)
        for h in hooks: h.remove()
        del block_copy
        torch.cuda.empty_cache()
        
        for mat_name in matrix_names:
            if mat_name not in act_caches: continue
            # 分批计算 global_scale（只需 max，逐 chunk 累积即可）
            act_list = act_caches[mat_name]
            act_abs_max = 0.0
            for x_chunk in act_list:
                am = x_chunk.abs().max().item()
                if am > act_abs_max:
                    act_abs_max = am
            gs_X = (FP8_E4M3_MAX * FP4_E2M1_MAX / max(act_abs_max, 1e-15))
            gs_X = torch.tensor([gs_X], device=device)
            
            W = get_combined_weight(block, mat_name).to(device)
            gs_W = compute_global_scale(W)
            
            baseline_results = {}
            for strat_name, quant_fn in strategies.items():
                act_mse = eval_quant_error_only(act_list, quant_fn, gs_X)
                out_mse_wa = eval_output_mse(act_list, W, quant_fn, gs_X, gs_W)
                
                if strat_name == "MinMax (baseline)":
                    baseline_results = {"act": act_mse, "out_wa": out_mse_wa}
                    print(f"L{block_idx:<4} | {mat_name:<8} | {strat_name:<18} | {act_mse:.6e}           | {out_mse_wa:.6e}")
                else:
                    act_pct = (act_mse - baseline_results["act"]) / (baseline_results["act"] + 1e-15) * 100
                    outwa_pct = (out_mse_wa - baseline_results["out_wa"]) / (baseline_results["out_wa"] + 1e-15) * 100
                    print(f"{'':5} | {'':8} | {strat_name:<18} | {act_mse:.6e} ({act_pct:+.1f}%) | {out_mse_wa:.6e} ({outwa_pct:+.1f}%)")
                
                summary[mat_name][strat_name]["act_sum"] += act_mse
                summary[mat_name][strat_name]["out_sum"] += out_mse_wa
                summary[mat_name][strat_name]["count"] += 1
            del W, act_list
            torch.cuda.empty_cache()
        # 清空 activation caches
        for k in list(act_caches.keys()):
            act_caches[k] = None
        del act_caches
        
        for i in range(len(input_args)):
            with torch.no_grad():
                args_on_dev = to_device(input_args[i], device)
                kwargs_on_dev = to_device(input_kwargs[i], device)
                out = block(*args_on_dev, **kwargs_on_dev)
                out_hidden = maybe_first(out).cpu()
                input_args[i] = (out_hidden,) + input_args[i][1:]
        block = block.cpu()
        torch.cuda.empty_cache()
    
    # ========== 性能基准测试 (真实推理路径: 仅量化, 输出 scale + 量化值) ==========
    timing_strategies = {
        "MinMax (Q only)": quantize_minmax_only,
        "LSS (Q only)": quantize_lss_only,
        "4/6 (Q only)": quantize_4o6_only,
        "LSS-3R (Q only)": quantize_lss_3round_only,
    }
    
    dummy_x = torch.randn(4096 * 4096, device=device, dtype=torch.bfloat16).view(-1, 16)
    outlier_mask = torch.rand_like(dummy_x) > 0.99
    dummy_x[outlier_mask] *= 10.0 
    dummy_global_scale = torch.tensor([1.0], device=device)
    tensor_size_gb = dummy_x.numel() * dummy_x.element_size() / (1024**3)
    
    # --- 真实推理路径性能对比 (Quantize Only) ---
    print("\n" + "=" * 110)
    print("Performance Benchmark: Quantize Only (output scale + quantized values)")
    print("-" * 110)
    print(f"{'Strategy':<18} | {'Exec Time (ms)':<15} | {'Peak Extra Mem (MB)':<20} | {'Throughput (GB/s)'}")
    print("-" * 110)
    
    for strat_name in ["MinMax (Q only)", "LSS (Q only)", "4/6 (Q only)", "LSS-3R (Q only)"]:
        quant_fn = timing_strategies[strat_name]
        for _ in range(5):
            _ = quant_fn(dummy_x, dummy_global_scale)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        
        start_event.record()
        for _ in range(20):
            res = quant_fn(dummy_x, dummy_global_scale)
        end_event.record()
        torch.cuda.synchronize()
        
        time_ms = start_event.elapsed_time(end_event) / 20.0
        peak_mem = torch.cuda.max_memory_allocated() - mem_before
        peak_mem_mb = peak_mem / (1024 * 1024)
        throughput = tensor_size_gb / (time_ms / 1000.0) if time_ms > 0 else 0.0
        
        print(f"{strat_name:<18} | {time_ms:>13.3f}   | {peak_mem_mb:>17.2f}    | {throughput:>14.2f}")
        del res
        torch.cuda.empty_cache()

    # ========== 汇总表: 全局平均 MSE（与前面表同宽 110 字符） ==========
    strategy_order = ["MinMax (baseline)", "LSS (Closed-form)", "4/6 (MSE)", "LSS-3R (3-round)"]
    print("\n" + "=" * 110)
    print("Summary: Global Average MSE Across All Layers")
    print("-" * 110)
    print(f"{'Strategy':<18} | {'Avg Act Quant MSE':<20} | {'Avg Output MSE(W4A4)':<20}")
    print("-" * 110)

    # 先算 MinMax baseline 的全局平均
    baseline_act = 0.0
    baseline_out = 0.0
    baseline_cnt = 0
    for mat_name in matrix_names:
        entry = summary[mat_name]["MinMax (baseline)"]
        baseline_act += entry["act_sum"]
        baseline_out += entry["out_sum"]
        baseline_cnt += entry["count"]
    baseline_act /= baseline_cnt if baseline_cnt > 0 else 1
    baseline_out /= baseline_cnt if baseline_cnt > 0 else 1

    for strat_name in strategy_order:
        total_act, total_out, cnt = 0.0, 0.0, 0
        for mat_name in matrix_names:
            entry = summary[mat_name][strat_name]
            total_act += entry["act_sum"]
            total_out += entry["out_sum"]
            cnt += entry["count"]
        if cnt > 0:
            avg_act = total_act / cnt
            avg_out = total_out / cnt
            if strat_name == "MinMax (baseline)":
                print(f"{strat_name:<18} | {avg_act:<20.6e} | {avg_out:<20.6e}")
            else:
                act_pct = (avg_act - baseline_act) / (baseline_act + 1e-15) * 100
                out_pct = (avg_out - baseline_out) / (baseline_out + 1e-15) * 100
                print(f"{strat_name:<18} | {avg_act:.6e} ({act_pct:+.1f}%) | {avg_out:.6e} ({out_pct:+.1f}%)")
    print("=" * 110)

if __name__ == "__main__":
    main()
