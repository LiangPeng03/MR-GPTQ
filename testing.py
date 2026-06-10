"""
test_scale_opt.py - 验证"MSE-Optimal Scale"与闭式解(LSS)对NVFP4激活量化的改善效果
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

# 方法1: MinMax (Baseline)
def quantize_minmax(x_groups, global_scale):
    abs_max = x_groups.abs().amax(dim=1, keepdim=True)
    raw_scales = abs_max / 6.0
    raw_scales[raw_scales == 0] = 1.0
    scales = scale_to_e4m3(raw_scales, global_scale)
    
    x_normalized = x_groups / scales
    x_q = cast_to_fp4(x_normalized)
    return x_q * scales

# 方法2: LSS 闭式解
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

# 权重量化使用 MSE 搜索 (模拟真实 MR-GPTQ 行为)
def quantize_weight_mse(w_groups, global_scale, scale_search_iters=100, max_scale_shrink_factor=0.80):
    abs_max = w_groups.abs().amax(dim=1, keepdim=True)
    init_raw_scales = abs_max / 6.0
    init_raw_scales[init_raw_scales == 0] = 1.0
    
    best_quantization_error = torch.full((w_groups.shape[0],), float("inf"), device=w_groups.device)
    best_dequant = torch.zeros_like(w_groups)
    
    for i in range(scale_search_iters):
        scale_shrink_factor = 1 - i * max_scale_shrink_factor / scale_search_iters
        candidate_raw = scale_shrink_factor * init_raw_scales
        candidate_scales = scale_to_e4m3(candidate_raw, global_scale)
        
        x_normalized = w_groups / candidate_scales
        x_q = cast_to_fp4(x_normalized)
        x_dequant = x_q * candidate_scales
        
        error = ((w_groups - x_dequant) ** 2).sum(dim=1)
        improved = error < best_quantization_error
        if improved.any():
            best_quantization_error[improved] = error[improved]
            best_dequant[improved] = x_dequant[improved]
            
    return best_dequant

# 评估函数: 计算纯激活量化误差
def eval_quant_error_only(X, quant_fn, global_scale, group_size=16):
    with torch.no_grad():
        X_groups = X.contiguous().view(-1, group_size)
        X_q = quant_fn(X_groups, global_scale).view(X.shape)
        mse = ((X - X_q) ** 2).mean().item()
    return mse

def eval_output_mse(X, W, quant_fn, global_scale_X, global_scale_W, group_size=16, chunk_size=4096):
    N_tokens = X.shape[0]
    total_mse = 0.0
    with torch.no_grad():
        W_groups = W.contiguous().view(-1, group_size)
        # 用 MSE 搜索量化权重
        W_q = quantize_weight_mse(W_groups, global_scale_W).view(W.shape)
        for i in range(0, N_tokens, chunk_size):
            X_chunk = X[i:i+chunk_size]
            Y_fp16 = X_chunk @ W.T
            X_groups = X_chunk.contiguous().view(-1, group_size)
            X_q = quant_fn(X_groups, global_scale_X).view(X_chunk.shape)
            Y_q = X_q @ W_q.T
            total_mse += ((Y_fp16 - Y_q) ** 2).sum().item()
    return total_mse / (N_tokens * W.shape[0])

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
    }
    
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
            X = torch.cat(act_caches[mat_name], dim=0).to(device)
            W = get_combined_weight(block, mat_name).to(device)
            gs_X = compute_global_scale(X)
            gs_W = compute_global_scale(W)
            
            baseline_results = {}
            for strat_name, quant_fn in strategies.items():
                act_mse = eval_quant_error_only(X, quant_fn, gs_X)
                out_mse_wa = eval_output_mse(X, W, quant_fn, gs_X, gs_W)
                
                if strat_name == "MinMax (baseline)":
                    baseline_results = {"act": act_mse, "out_wa": out_mse_wa}
                    print(f"L{block_idx:<4} | {mat_name:<8} | {strat_name:<18} | {act_mse:.6e}           | {out_mse_wa:.6e}")
                else:
                    act_pct = (act_mse - baseline_results["act"]) / (baseline_results["act"] + 1e-15) * 100
                    outwa_pct = (out_mse_wa - baseline_results["out_wa"]) / (baseline_results["out_wa"] + 1e-15) * 100
                    print(f"{'':5} | {'':8} | {strat_name:<18} | {act_mse:.6e} ({act_pct:+.1f}%) | {out_mse_wa:.6e} ({outwa_pct:+.1f}%)")
            del X, W
            torch.cuda.empty_cache()
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
    
    print("\n" + "=" * 110)
    print("Performance Benchmark (Time & Memory on a large dummy matrix)")
    print("-" * 110)
    dummy_x = torch.randn(4096 * 4096, device=device, dtype=torch.bfloat16).view(-1, 16)
    outlier_mask = torch.rand_like(dummy_x) > 0.99
    dummy_x[outlier_mask] *= 10.0 
    dummy_global_scale = torch.tensor([1.0], device=device)
    
    print(f"{'Strategy':<18} | {'Exec Time (ms)':<15} | {'Peak Extra Mem (MB)':<20} | {'Throughput (GB/s)'}")
    print("-" * 110)
    tensor_size_gb = dummy_x.numel() * dummy_x.element_size() / (1024**3)
    
    for strat_name, quant_fn in strategies.items():
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
        throughput = tensor_size_gb / (time_ms / 1000.0)
        
        print(f"{strat_name:<18} | {time_ms:>13.3f}   | {peak_mem_mb:>17.2f}    | {throughput:>14.2f}")
        del res
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
