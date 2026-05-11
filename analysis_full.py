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
    计算特定矩阵输入的 MSE 增量和 Energy Max 特征
    """
    # X shape: [1, seq_len, dim] -> [N, 16]
    X_groups = X.reshape(-1, 16)
    
    # 抽样以加速，58万组还是有点多，全量跑 224 个模块会很慢
    # 取 50000 组足够反映统计规律
    if X_groups.shape[0] > 50000:
        indices = torch.randperm(X_groups.shape[0])[:50000].to(X.device)
        X_groups = X_groups[indices]
    
    # 1. 原始量化 MSE
    scales, zeros = quantizer.get_quantization_params(X_groups)
    X_q_norot = quantizer(X_groups, scales, zeros)
    mse_norot = (X_q_norot - X_groups).pow(2).mean(dim=-1)
    
    # 2. 旋转量化 MSE
    X_rot = transform(X_groups)
    scales_rot, zeros_rot = quantizer.get_quantization_params(X_rot)
    X_q_rot_inner = quantizer(X_rot, scales_rot, zeros_rot)
    X_q_rot = transform(X_q_rot_inner, inv_t=True)
    mse_rot = (X_q_rot - X_groups).pow(2).mean(dim=-1)
    
    delta_L = mse_rot - mse_norot
    energy_max = torch.max(X_groups.abs(), dim=-1)[0] ** 2
    
    # 计算相关系数
    e_cpu = energy_max.cpu().numpy()
    d_cpu = delta_L.cpu().numpy()
    
    # 过滤掉非正值以防对数趋势干扰（虽然 Pearson 是线性的）
    p_corr, _ = pearsonr(e_cpu, d_cpu)
    s_corr, _ = spearmanr(e_cpu, d_cpu)
    
    # 返回相关性以及 Delta L 的极值
    return p_corr, s_corr, delta_L.max().item(), delta_L.min().item()

def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    # 初始化量化器和变换
    quantizer = Quantizer(bits=4, symmetric=True, format="nvfp", granularity="group", group_size=16, scale_precision="e4m3")
    transform = build_transform("hadamard", group_size=16, device=device, dtype=torch.bfloat16)

    # 准备输入 (稍微长一点增加覆盖面)
    text = "Machine learning quantization analysis across all transformer layers to identify outlier patterns. " * 30
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 定义要追踪的模块类型
    module_types = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    results = {m: {"pearson": [], "spearman": [], "max_dl": [], "min_dl": []} for m in module_types}
    
    num_layers = len(model.model.layers)
    
    for i in tqdm(range(num_layers), desc="Analyzing Layers"):
        layer = model.model.layers[i]
        # 建立当前层模块映射
        targets = {
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj,
            "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
        }
        
        layer_activations = {}
        def get_hook(name):
            def hook(m, i, o):
                # i[0] 是输入张量
                layer_activations[name] = i[0].detach().float()
            return hook
        
        # 注册 hooks
        handles = []
        for name, mod in targets.items():
            handles.append(mod.register_forward_hook(get_hook(name)))
            
        # 运行前向推理抓取激活
        with torch.no_grad():
            model(**inputs)
            
        # 移除 hooks
        for h in handles:
            h.remove()
            
        # 分析该层所有模块
        for name in module_types:
            act = layer_activations[name]
            p, s, max_dl, min_dl = compute_metrics(act, quantizer, transform)
            results[name]["pearson"].append(p)
            results[name]["spearman"].append(s)
            results[name]["max_dl"].append(max_dl)
            results[name]["min_dl"].append(min_dl)
            
        # 及时清理内存
        del layer_activations
        torch.cuda.empty_cache()

    # === 绘图 ===
    layers = np.arange(num_layers)
    
    # 图 1: Pearson (Linear)
    plt.figure(figsize=(12, 7))
    for m in module_types:
        plt.plot(layers, results[m]["pearson"], label=m, marker='o', markersize=4)
    plt.title("Linear Correlation Trend (Pearson): Max Energy vs Delta L")
    plt.xlabel("Layer Index")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.legend(title="Module Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("trend_pearson_all_layers.png")
    print("Pearson 趋势图已保存至: trend_pearson_all_layers.png")

    # 图 2: Spearman (Non-linear)
    plt.figure(figsize=(12, 7))
    for m in module_types:
        plt.plot(layers, results[m]["spearman"], label=m, marker='s', markersize=4)
    # 图 3: Delta L Range (Max - Min)
    # 展示每个模块内部量化损失波动的最大跨度，反映旋转对该模块的“冲击强度”
    plt.figure(figsize=(12, 7))
    for m in module_types:
        # 计算跨度：最大恶化减去最大获益
        diff_dl = np.array(results[m]["max_dl"]) - np.array(results[m]["min_dl"])
        plt.plot(layers, diff_dl, label=m, marker='o', markersize=4)
    
    plt.title("Hadamard Rotation Impact Range: (Max - Min) Delta L", fontsize=14)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Delta L Range (Log Scale)", fontsize=12)
    
    # 跨度一定是正数，直接使用 log 坐标可以最清晰地看到量级差异
    plt.yscale('log')
    plt.legend(title="Module Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig("trend_delta_l_range.png")
    print("影响跨度图 (Impact Range) 已保存至: trend_delta_l_range.png")

if __name__ == "__main__":
    main()
