import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def verify_diffusion():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.eval()
    
    text = "Detailed shape analysis of LLM activations. " * 50
    calib = tokenizer(text, return_tensors="pt").to(device)
    
    # Target layers to observe
    target_layers = [0, 7, 15, 23, 31]
    matrices = ["qkv", "o", "gate_up", "down"]
    
    caches = {}
    hooks = []
    
    def hook_factory(name):
        def hook(m, i, o):
            if name not in caches: caches[name] = []
            caches[name].append(i[0].detach().float().view(-1, i[0].shape[-1]))
        return hook
        
    for l_idx in target_layers:
        layer = model.model.layers[l_idx]
        hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_factory(f"L{l_idx}_qkv")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(hook_factory(f"L{l_idx}_o")))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(hook_factory(f"L{l_idx}_gate_up")))
        hooks.append(layer.mlp.down_proj.register_forward_hook(hook_factory(f"L{l_idx}_down")))
    
    print("Running forward pass...")
    with torch.no_grad():
        model(**calib)
        
    for h in hooks: h.remove()
    
    print("\n" + "="*90)
    print(f"{'Matrix':<12} | {'Max Freq':<10} | {'Top 1% Freq':<12} | {'Top 5% Freq':<12} | {'Top 10% Freq':<12} | {'Median Freq':<12}")
    print("-" * 90)
    
    for l_idx in target_layers:
        for mat in matrices:
            name = f"L{l_idx}_{mat}"
            if name not in caches: continue
            
            # For matrices like qkv and gate_up, the input is identical across the concatenated projections.
            # We just take the first cache entry which corresponds to the common input (hidden_states).
            X = caches[name][0]
            
            threshold = torch.quantile(X.abs(), 0.9375, dim=1, keepdim=True)
            is_outlier = (X.abs() > threshold).float()
            
            # Calculate outlier frequency per channel
            ch_freq = is_outlier.mean(dim=0)
            
            # Sort frequencies
            sorted_freq, _ = torch.sort(ch_freq, descending=True)
            
            dim = sorted_freq.shape[0]
            max_f = sorted_freq[0].item()
            top1_f = sorted_freq[max(0, dim // 100 - 1)].item()
            top5_f = sorted_freq[max(0, dim // 20 - 1)].item()
            top10_f = sorted_freq[max(0, dim // 10 - 1)].item()
            median_f = sorted_freq[dim // 2].item()
            
            print(f"{name:<12} | {max_f:>9.2%} | {top1_f:>11.2%} | {top5_f:>11.2%} | {top10_f:>11.2%} | {median_f:>11.2%}")
        print("-" * 90)
        
    print("\n" + "="*110)
    print("=== OUTLIER MAGNITUDE & ENERGY DIAGNOSTIC ===")
    print("="*110)
    print(f"{'Matrix':<12} | {'Mean/Median':<12} | {'> 10x Median':<12} | {'> 50x Median':<12} | {'Top 6.25% Energy':<18}")
    print("-" * 110)
    
    for l_idx in target_layers:
        for mat in matrices:
            name = f"L{l_idx}_{mat}"
            if name not in caches: continue
            
            X = caches[name][0] # shape (N_tokens, dim)
            X_abs = X.abs()
            token_means = X_abs.mean(dim=1, keepdim=True)
            token_medians = X_abs.median(dim=1, keepdim=True)[0].clamp(min=1e-7)
            
            mean_to_median = (token_means / token_medians).mean().item()
            
            # Count percentage of channels per token that exceed multiples of the MEDIAN
            pct_10x_med = (X_abs > 10 * token_medians).float().mean().item() * 100
            pct_50x_med = (X_abs > 50 * token_medians).float().mean().item() * 100
            
            # Energy concentration: What % of the total absolute sum is held by the Top 6.25%?
            k = max(1, int(X.shape[1] * 0.0625))
            topk_vals, _ = torch.topk(X_abs, k, dim=1)
            topk_energy = topk_vals.sum(dim=1)
            total_energy = X_abs.sum(dim=1).clamp(min=1e-7)
            energy_pct = (topk_energy / total_energy).mean().item() * 100
            
            print(f"{name:<12} | {mean_to_median:>11.2f}x | {pct_10x_med:>11.2f}% | {pct_50x_med:>11.2f}% | {energy_pct:>16.2f}%")
        print("-" * 110)
        
    print("\n" + "="*110)
    print("=== TOKEN-LEVEL OUTLIER CONGESTION (Using Global Absolute Threshold) ===")
    print("="*110)
    print("Threshold = The Top 6.25% Absolute Value of the ENTIRE matrix.")
    print("If outliers were perfectly distributed, every token would have exactly 256.")
    print(f"{'Matrix':<12} | {'Threshold':<12} | {'Avg >Thresh':<14} | {'Max >Thresh':<14} | {'% Tokens > 256':<20}")
    print("-" * 110)
    
    for l_idx in target_layers:
        for mat in matrices:
            name = f"L{l_idx}_{mat}"
            if name not in caches: continue
            
            X = caches[name][0]
            X_abs = X.abs()
            
            # Find the global threshold: top 6.25% of the entire matrix
            # Since tensor might be large, we can sample or just use flattening if memory permits
            # flatten() might OOM if too large, so let's subsample if needed, or just reshape
            kth = max(1, int(X.numel() * 0.9375))
            # using quantile on flattened is safe for sizes like 8192x4096 (33M elements)
            global_thresh = torch.quantile(X_abs.flatten().float(), 0.9375).item()
            
            # Count how many values per token exceed this GLOBAL absolute threshold
            outlier_counts = (X_abs > global_thresh).float().sum(dim=1)
            
            avg_outliers = outlier_counts.mean().item()
            max_outliers = outlier_counts.max().item()
            pct_over_budget = (outlier_counts > 256).float().mean().item() * 100
            
            print(f"{name:<12} | {global_thresh:>12.3f} | {avg_outliers:>14.1f} | {max_outliers:>14.0f} | {pct_over_budget:>19.2f}%")
        print("-" * 110)

if __name__ == "__main__":
    verify_diffusion()
