import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantization.quantizer import Quantizer
from src.quantization.quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX
from src.quantization.quantizer import get_reciprocal

def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda"
    layer_idx = 31
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model.eval()
    
    # 准备三个不同的数据集（校准集 A，测试集 B、C）
    text_a = "Detailed shape analysis of LLM activations. " * 300
    text_b = "The quick brown fox jumps over the lazy dog. Machine learning is transforming artificial intelligence research across the globe. " * 150
    text_c = "Data structures and algorithms form the foundation of modern computer science and software engineering practices. " * 200
    
    layer = model.model.layers[layer_idx].mlp.down_proj
    
    def get_activations(text):
        calib = tokenizer(text, return_tensors="pt").to(device)
        cache = []
        def hook(m, i, o): cache.append(i[0].detach())
        handle = layer.register_forward_hook(hook)
        with torch.no_grad():
            model(**calib)
        handle.remove()
        return cache[0].view(-1, cache[0].shape[-1]).float()
    
    X_a = get_activations(text_a)
    X_b = get_activations(text_b)
    X_c = get_activations(text_c)
    
    n_tokens_a, dim = X_a.shape
    GS = 16
    n_groups = dim // GS
    
    print(f"\nSet A (Calibration): {n_tokens_a} tokens")
    print(f"Set B (Test 1):      {X_b.shape[0]} tokens")
    print(f"Set C (Test 2):      {X_c.shape[0]} tokens")
    print("\n" + "="*80)
    print("CO-OCCURRENCE-AWARE STAGGERED REORDERING: COMPREHENSIVE COMPARISON")
    print("="*80)
    
    # 计算量化 MSE 的辅助函数
    def calc_mse(X):
        nv_q = Quantizer(bits=4, format="nvfp", granularity="group", group_size=16, symmetric=True, scale_precision="e4m3")
        nv_q._track_global_scale = False
        act_max = X.abs().max().to(torch.float32).view(1)
        nv_q.global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(act_max)).to(device)
        X_16 = X.view(-1, 16)
        sc, z = nv_q.get_quantization_params(X_16)
        q = nv_q(X_16, sc, z)
        return ((q - X_16)**2).mean().item()
        
    mse_a_orig = calc_mse(X_a)
    mse_b_orig = calc_mse(X_b)
    mse_c_orig = calc_mse(X_c)
    
    print(f"  [Baseline: Original Unordered Grouping]")
    print(f"    Set A MSE: {mse_a_orig:.6f}")
    print(f"    Set B MSE: {mse_b_orig:.6f}")
    print(f"    Set C MSE: {mse_c_orig:.6f}")
    print("-" * 80)
    
    flat_X_a = X_a.abs().view(-1)
    sorted_vals, _ = torch.sort(flat_X_a)
    
    def evaluate_co_occurrence_staggering(is_outlier_mask, method_name):
        # 计算每个通道的 Outlier 频率并降序排列
        ch_freq = is_outlier_mask.float().mean(dim=0)
        sorted_channels_list = torch.argsort(ch_freq, descending=True).tolist()
        
        opt_groups = [[] for _ in range(n_groups)]
        group_profiles = torch.zeros((n_groups, n_tokens_a), device=device, dtype=torch.float32)
        group_sizes = torch.zeros(n_groups, device=device, dtype=torch.long)
        
        # 贪心插入（错峰）
        for ch in sorted_channels_list:
            c_profile = is_outlier_mask[:, ch].float()
            penalties = torch.mv(group_profiles, c_profile)
            penalties[group_sizes >= GS] = float('inf')
            best_group = penalties.argmin().item()
            opt_groups[best_group].append(ch)
            group_profiles[best_group] += c_profile
            group_sizes[best_group] += 1
            
        opt_perm = []
        for g in opt_groups: opt_perm.extend(g)
        opt_perm = torch.tensor(opt_perm, device=device)
        
        # 测试重排序后的 MSE
        mse_a = calc_mse(X_a[:, opt_perm])
        mse_b = calc_mse(X_b[:, opt_perm])
        mse_c = calc_mse(X_c[:, opt_perm])
        
        imp_a = (mse_a_orig - mse_a) / mse_a_orig * 100
        imp_b = (mse_b_orig - mse_b) / mse_b_orig * 100
        imp_c = (mse_c_orig - mse_c) / mse_c_orig * 100
        
        print(f"  [{method_name}]")
        print(f"    Total Outliers Detected: {is_outlier_mask.sum().item():>7}")
        print(f"    Set A MSE: {mse_a:.6f} (Imp: {imp_a:+.2f}%)")
        print(f"    Set B MSE: {mse_b:.6f} (Imp: {imp_b:+.2f}%)")
        print(f"    Set C MSE: {mse_c:.6f} (Imp: {imp_c:+.2f}%)")
        print("-" * 80)

    # 我们要测试的分位数列表
    percentiles = [0.99, 0.95, 0.9375, 0.925, 0.90, 0.85]
    
    print("\n>>> PART 1: TOKEN-LEVEL PERCENTILES (Row-wise adaptive) <<<")
    for p in percentiles:
        mask = X_a.abs() > torch.quantile(X_a.abs().float(), p, dim=1, keepdim=True)
        evaluate_co_occurrence_staggering(mask, f"Token-level P{p*100:g}")
        
    print("\n>>> PART 2: GLOBAL MATRIX PERCENTILES (Fixed threshold) <<<")
    for p in percentiles:
        idx = int(p * sorted_vals.numel())
        th = sorted_vals[idx]
        mask = X_a.abs() > th
        evaluate_co_occurrence_staggering(mask, f"Global P{p*100:g}")

if __name__ == "__main__":
    main()
