import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg') # 非交互模式

# 确保能引入项目内的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.quantization.quantizer import Quantizer
from src.transforms.transforms import build_transform

def compute_group_features(X):
    """计算 Top-2 能量占比"""
    abs_X = X.abs()
    top2_vals, _ = torch.topk(abs_X, k=2, dim=-1)
    Top1, Top2 = top2_vals[:, 0], top2_vals[:, 1]
    E_total = torch.sum(X**2, dim=1) + 1e-9
    
    features = {
        "top2_ratio": (Top1**2 + Top2**2) / E_total,
        "top2_energy": Top1**2 + Top2**2,
        "energy_max": Top1**2
    }
    return features

def analyze_layer_activation(layer_name, X, quantizer, transform):
    X_groups = X.view(-1, 16)
    if X_groups.shape[0] > 500000:
        indices = torch.randperm(X_groups.shape[0])[:500000].to(X.device)
        X_groups = X_groups[indices]
    
    # 特征提取
    features = compute_group_features(X_groups)
    
    # 量化误差
    scales, zeros = quantizer.get_quantization_params(X_groups)
    mse_norot = (quantizer(X_groups, scales, zeros) - X_groups).pow(2).mean(dim=-1)
    
    X_rot = transform(X_groups)
    scales_rot, zeros_rot = quantizer.get_quantization_params(X_rot)
    X_q_rot_inner = quantizer(X_rot, scales_rot, zeros_rot)
    
    # 反向旋转回来算最终误差
    X_q_rot = transform(X_q_rot_inner, inv_t=True)
    mse_rot = (X_q_rot - X_groups).pow(2).mean(dim=-1)
    
    # 旋转后最终量化损失 (Post-Rotation MSE)
    # 计算旋转后量化相较于原始输入的 MSE，损失必然 >= 0
    
    from scipy.stats import pearsonr, spearmanr
    feat_cpu = features["top2_energy"].cpu().numpy()
    mse_rot_cpu = mse_rot.cpu().numpy()
    weights = features["energy_max"].cpu().numpy()
    
    corr_p, p_p = pearsonr(feat_cpu, mse_rot_cpu)
    corr_s, p_s = spearmanr(feat_cpu, mse_rot_cpu)
    
    print(f"\n[{layer_name}] Correlation (Top-2 Energy vs Post-Rot MSE):")
    print(f"  Pearson: {corr_p:.4f} (p={p_p:.1e}) | Spearman: {corr_s:.4f} (p={p_s:.1e})")
    
    return {
        "layer_name": layer_name,
        "top2_energy": feat_cpu,
        "mse_rot": mse_rot_cpu,
        "energy_max": weights
    }

def plot_consolidated_top2(results_list):
    n_layers = len(results_list)
    fig, axes = plt.subplots(1, n_layers, figsize=(18, 5))
    if n_layers == 1: axes = [axes]
    
    for i, res in enumerate(results_list):
        ax = axes[i]
        x = res["top2_energy"]
        y = res["mse_rot"]
        w = res["energy_max"]
        
        # 1. 绘制背景散点 (采样 20000 点增加密度)
        sample_idx = np.random.choice(len(x), min(20000, len(x)), replace=False)
        ax.scatter(x[sample_idx], y[sample_idx], alpha=0.1, s=1, color='purple')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"{res['layer_name']}")
        ax.set_xlabel("Top-2 Energy ($Top_1^2 + Top_2^2$)")
        if i == 0: ax.set_ylabel("Post-Rotation MSE (Log Scale)")
        ax.grid(True, alpha=0.3, which="both", linestyle='--')


    fig.suptitle("Top-2 Energy vs Post-Rotation Quantization MSE", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("consolidated_top2_impact.png")
    print(f"\n【整合图表已生成】: consolidated_top2_impact.png")

def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.eval()

    quantizer = Quantizer(bits=4, symmetric=True, format="nvfp", granularity="group", group_size=16, scale_precision="e4m3")
    transform = build_transform("hadamard", group_size=16, device=device, dtype=torch.bfloat16)

    text = "Detailed validation of Top-2 energy concentration in LLM activations. " * 100
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    target_layers = [0, 15, 31]
    results_cache = []
    
    for layer_idx in target_layers:
        mod = model.model.layers[layer_idx].mlp.down_proj
        activation_cache = []
        def hook_fn(m, i, o): activation_cache.append(i[0].detach().float())
        handle = mod.register_forward_hook(hook_fn)
        
        with torch.no_grad(): model(**inputs)
        handle.remove()
        
        print(f"Analyzing Layer {layer_idx}...")
        act = activation_cache[0].view(-1, 16)
        res = analyze_layer_activation(f"Layer_{layer_idx}", act, quantizer, transform)
        results_cache.append(res)
        torch.cuda.empty_cache()
    
    plot_consolidated_top2(results_cache)

if __name__ == "__main__":
    main()
