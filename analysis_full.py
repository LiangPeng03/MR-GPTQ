import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

matplotlib.use('Agg') # 非交互模式

# 确保能引入项目内的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.quantization.quantizer import Quantizer
from src.transforms.transforms import build_transform

def compute_metrics(X, quantizer, transform):
    """
    计算特定矩阵输入的 |Delta L| 和 Top-2 Energy 相关性
    """
    X_groups = X.reshape(-1, 16)
    
    # 抽样以加速
    if X_groups.shape[0] > 50000:
        indices = torch.randperm(X_groups.shape[0])[:50000].to(X.device)
        X_groups = X_groups[indices]
    
    # 1. 原始量化 MSE
    scales, zeros = quantizer.get_quantization_params(X_groups)
    mse_norot = (quantizer(X_groups, scales, zeros) - X_groups).pow(2).mean(dim=-1)
    
    # 2. 旋转量化 MSE
    X_rot = transform(X_groups)
    scales_rot, zeros_rot = quantizer.get_quantization_params(X_rot)
    X_q_rot_inner = quantizer(X_rot, scales_rot, zeros_rot)
    mse_rot = (transform(X_q_rot_inner, inv_t=True) - X_groups).pow(2).mean(dim=-1)
    
    # 旋转后最终量化损失 (Post-Rotation MSE)
    # 计算旋转后量化相较于原始输入的 MSE，损失必然 >= 0
    mse_rot_cpu = mse_rot.cpu().numpy()
    
    # Top-2 绝对能量
    abs_X = X_groups.abs()
    top2_vals, _ = torch.topk(abs_X, k=2, dim=-1)
    top2_energy = (top2_vals[:, 0]**2 + top2_vals[:, 1]**2).cpu().numpy()
    
    p_corr, p_p = pearsonr(top2_energy, mse_rot_cpu)
    s_corr, p_s = spearmanr(top2_energy, mse_rot_cpu)
    
    return p_corr, p_p, s_corr, p_s, mse_rot.mean().item()

def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.eval()

    quantizer = Quantizer(bits=4, symmetric=True, format="nvfp", granularity="group", group_size=16, scale_precision="e4m3")
    transform = build_transform("hadamard", group_size=16, device=device, dtype=torch.bfloat16)

    text = "Systematic analysis of quantization trends across all layers. " * 50
    inputs = tokenizer(text, return_tensors="pt").to(device)

    module_types = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    results = {m: {"pearson": [], "pearson_p": [], "spearman": [], "spearman_p": [], "mean_mse": []} for m in module_types}
    
    num_layers = len(model.model.layers)
    
    for i in tqdm(range(num_layers), desc="Analyzing Layers"):
        layer = model.model.layers[i]
        targets = {
            "q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj, "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj, "up_proj": layer.mlp.up_proj, "down_proj": layer.mlp.down_proj,
        }
        
        layer_activations = {}
        def get_hook(name):
            def hook(m, i, o): layer_activations[name] = i[0].detach().float()
            return hook
        
        handles = [mod.register_forward_hook(get_hook(name)) for name, mod in targets.items()]
        with torch.no_grad(): model(**inputs)
        for h in handles: h.remove()
            
        for name in module_types:
            p_val, p_p, s_val, s_p, mean_mse = compute_metrics(layer_activations[name], quantizer, transform)
            results[name]["pearson"].append(p_val)
            results[name]["pearson_p"].append(p_p)
            results[name]["spearman"].append(s_val)
            results[name]["spearman_p"].append(s_p)
            results[name]["mean_mse"].append(mean_mse)
            
        torch.cuda.empty_cache()

    # === 整合绘图 (1x3 Layout) ===
    layers = np.arange(num_layers)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for m in module_types:
        axes[0].plot(layers, results[m]["pearson"], label=m, marker='o', markersize=3, alpha=0.7)
        axes[1].plot(layers, results[m]["spearman"], label=m, marker='s', markersize=3, alpha=0.7)
        axes[2].plot(layers, results[m]["mean_mse"], label=m, marker='^', markersize=3, alpha=0.7)

    axes[0].set_title("Pearson Correlation Trend\nTop-2 Energy vs Post-Rot MSE")
    axes[0].set_ylabel("Pearson $r$")
    axes[1].set_title("Spearman Correlation Trend\nTop-2 Energy vs Post-Rot MSE")
    axes[1].set_ylabel("Spearman $\\rho$")
    axes[2].set_title("Quantization Noise Trend\nMean Post-Rotation MSE")
    axes[2].set_ylabel("Mean MSE (Log)")
    axes[2].set_yscale('log')
    
    for i, ax in enumerate(axes):
        ax.set_xlabel("Layer Index")
        ax.grid(True, alpha=0.3)
        # 在相关系数图 (前两张) 中添加 0.3 和 0.6 的水平参考线
        if i < 2:
            ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Weak Threshold' if i==0 else "")
            ax.axhline(0.6, color='darkred', linestyle='--', alpha=0.5, label='Moderate Threshold' if i==0 else "")

    
    axes[2].legend(title="Module", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.suptitle("Full Model Quantization Sensitivity: Top-2 Energy vs Post-Rotation MSE", fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("consolidated_model_trends.png")
    print("\n【整合趋势图已保存】: consolidated_model_trends.png")

    # === 全模型统计 ===
    print("\n" + "="*60)
    print(f"{'Module Type':<12} | {'Avg Pearson':<12} | {'Avg P-val(P)':<12} | {'Avg Spearman':<12}")
    print("-" * 60)
    
    total_p, total_s = [], []
    for m in module_types:
        avg_p = np.mean(results[m]["pearson"])
        avg_pp = np.mean(results[m]["pearson_p"])
        avg_s = np.mean(results[m]["spearman"])
        print(f"{m:<12} | {avg_p:12.4f} | {avg_pp:12.1e} | {avg_s:12.4f}")
        total_p.append(avg_p)
        total_s.append(avg_s)
        
    print("-" * 60)
    print(f"{'OVERALL AVG':<12} | {np.mean(total_p):12.4f} | {'N/A':<12} | {np.mean(total_s):12.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
