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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.quantization.quantizer import Quantizer
from src.transforms.transforms import build_transform

def analyze_layer_shape(layer_name, X, quantizer, transform):
    """只负责计算特征和误差，返回数据字典"""
    X_groups = X.view(-1, 16)
    if X_groups.shape[0] > 500000:
        indices = torch.randperm(X_groups.shape[0])[:500000].to(X.device)
        X_groups = X_groups[indices]
    
    N_groups = X_groups.shape[0]
    
    # 特征提取
    abs_X = X_groups.abs()
    top3_vals, _ = torch.topk(abs_X, k=3, dim=-1)
    Top1, Top2, Top3 = top3_vals[:, 0], top3_vals[:, 1], top3_vals[:, 2]
    
    E_total = torch.sum(X_groups**2, dim=1)
    E_max = Top1**2
    Top2_energy_ratio = (Top1**2 + Top2**2) / (E_total + 1e-9)
    Top3_energy_ratio = (Top1**2 + Top2**2 + Top3**2) / (E_total + 1e-9)
    
    # 量化误差
    scales, zeros = quantizer.get_quantization_params(X_groups)
    mse_norot = (quantizer(X_groups, scales, zeros) - X_groups).pow(2).mean(dim=-1)
    
    X_rot = transform(X_groups)
    scales_rot, zeros_rot = quantizer.get_quantization_params(X_rot)
    mse_rot = (transform(quantizer(X_rot, scales_rot, zeros_rot), inv_t=True) - X_groups).pow(2).mean(dim=-1)
    
    delta_L = mse_rot - mse_norot
    
    R = Top2 / (Top1 + 1e-9)
    
    return {
        "layer_name": layer_name,
        "E_total": E_total.cpu().numpy(),
        "E_max": E_max.cpu().numpy(),
        "Ratio_top2_energy": Top2_energy_ratio.cpu().numpy(),
        "Ratio_top3_energy": Top3_energy_ratio.cpu().numpy(),
        "R": R.cpu().numpy(),
        "mse_rot": mse_rot.cpu().numpy(),
    }

def plot_consolidated_results(results_list):
    """生成三张 1x3 的整合图表"""
    n_layers = len(results_list)
    # 理论共振点 R 值
    theory_R = [1.0, 0.846, 0.714, 0.600, 0.500, 0.333, 0.200, 0.0]
    
    # --- 图表 1: 能量收敛漏斗图 (Convergence) ---
    fig1, axes1 = plt.subplots(1, n_layers, figsize=(18, 5))
    if n_layers == 1: axes1 = [axes1]
    for i, res in enumerate(results_list):
        ax = axes1[i]
        # X 轴改为旋转后最终量化损失 Post-Rotation MSE
        sample_size = min(50000, len(res["mse_rot"]))
        idx = np.random.choice(len(res["mse_rot"]), sample_size, replace=False)
        ax.scatter(res["mse_rot"][idx], res["Ratio_top2_energy"][idx], alpha=0.1, s=1, color='purple')
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)
        ax.axhline(0.95, color='red', linestyle='--', linewidth=1)
        ax.set_title(f"{res['layer_name']}")
        if i == 0: ax.set_ylabel("Top-2 Energy Ratio")
        ax.set_xlabel("Post-Rotation MSE")
        
        # 统计 MSE 数量级下的 Top-2 占比情况
        mse_vals = res["mse_rot"]
        print(f"\n[{res['layer_name']}] Post-Rotation MSE 数量级统计:")
        for m in [1e-5, 1e-4, 1e-3, 1e-2]:
            mask = mse_vals > m
            if mask.any():
                prop95 = (res["Ratio_top2_energy"][mask] > 0.95).mean()
                prop97_t3 = (res["Ratio_top3_energy"][mask] > 0.97).mean()
                print(f"  MSE > {m:<8} | Top-2 > 95%: {prop95:.2%} | Top-3 > 97%: {prop97_t3:.2%}")

    fig1.suptitle("Energy Convergence: High-MSE groups concentrate energy in Top-2 elements", fontsize=16)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig("consolidated_convergence.png")

    # --- 图表 2: 共振剖面图 (Resonance Comb) - 三行显示 ---
    # 第一行：全部组；第二行：Top-2 > 95%；第三行：Top-3 > 97%
    fig2, axes2 = plt.subplots(3, n_layers, figsize=(18, 15))
    if n_layers == 1: axes2 = np.expand_dims(axes2, axis=1) # 保持索引一致性
    
    for row in range(3):
        for i, res in enumerate(results_list):
            ax = axes2[row, i]
            
            if row == 0:
                # 全部组
                mask = np.ones_like(res["R"], dtype=bool)
                row_title = "All Groups"
                current_title = f"{res['layer_name']} ({row_title})"
                color = 'blue'
            elif row == 1:
                # Top-2 占比 > 95%
                mask = res["Ratio_top2_energy"] > 0.95
                row_title = "Top-2 > 95%"
                current_title = f"{res['layer_name']} ({row_title})"
                color = 'red'
            else:
                # Top-3 占比 > 97%
                mask = res["Ratio_top3_energy"] > 0.97
                row_title = "Top-3 > 97%"
                current_title = f"{res['layer_name']} ({row_title})"
                color = 'darkgreen'
                
            R_vals = res["R"][mask]
            mse_vals = res["mse_rot"][mask]
            weights = res["E_max"][mask]**2
            
            # 分箱计算
            bins = 100
            bin_edges = np.linspace(0, 1, bins + 1)
            bin_idx = np.digitize(R_vals, bin_edges) - 1
            
            binned_R = []
            binned_mse = []
            for b in range(bins):
                m = bin_idx == b
                if m.any():
                    w = weights[m]
                    binned_R.append(np.mean(R_vals[m]))
                    binned_mse.append(np.sum(mse_vals[m] * w) / (np.sum(w) + 1e-12))
            
            # 绘制底层散点
            sample_size = min(10000, len(R_vals))
            if sample_size > 0:
                idx = np.random.choice(len(R_vals), sample_size, replace=False)
                ax.scatter(R_vals[idx], mse_vals[idx], alpha=0.08, s=2, color='gray')
            
            # 绘制加权均值曲线
            ax.plot(binned_R, binned_mse, color=color, linewidth=2, label='Weighted Mean MSE')
            
            # 标记共振点
            for tr in theory_R:
                ax.axvline(tr, color='green', linestyle=':', alpha=0.4)
            
            ax.set_yscale('log')
            ax.set_title(current_title)
            if i == 0: ax.set_ylabel(f"{row_title}\nPost-Rot MSE")
            if row == 2: ax.set_xlabel("Ratio $R = |Top2|/|Top1|$")
            if i == n_layers-1: ax.legend()
            
    fig2.suptitle("Resonance Comb: Post-Rotation Quantization MSE sensitivity to R-ratio", fontsize=20)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig("consolidated_resonance_profile.png")



def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.eval()

    quantizer = Quantizer(bits=4, symmetric=True, format="nvfp", granularity="group", group_size=16, scale_precision="e4m3")
    transform = build_transform("hadamard", group_size=16, device=device, dtype=torch.bfloat16)

    # 增加文本长度以获得更多激活样本
    text = "Detailed shape analysis of LLM activations. " * 100
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
        
        print(f"Processing Layer {layer_idx}...")
        # 展平 batch 和 seq 维度
        act = activation_cache[0].view(-1, activation_cache[0].shape[-1])
        res = analyze_layer_shape(f"Layer_{layer_idx}", act, quantizer, transform)
        results_cache.append(res)
        torch.cuda.empty_cache()
    
    print("\nGenerating consolidated plots...")
    plot_consolidated_results(results_cache)
    print("Done. Check consolidated_convergence.png and consolidated_resonance_profile.png")

if __name__ == "__main__":
    main()
