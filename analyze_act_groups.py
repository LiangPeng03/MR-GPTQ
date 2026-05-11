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
    """
    计算每个 Group (大小16) 的统计学特征
    X shape: [N, 16]
    """
    features = {}
    
    # 1. 峰度 (Kurtosis): 衡量极端离群值的强度
    mu = X.mean(dim=-1, keepdim=True)
    var = X.var(dim=-1, keepdim=True) + 1e-6
    # 简化的峰度计算 E[(X-mu)^4] / var^2
    kurtosis = ((X - mu)**4).mean(dim=-1) / (var.squeeze(-1)**2)
    features["kurtosis"] = kurtosis
    
    # 2. Max-to-Mean Ratio: 最大绝对值 / 平均绝对值
    abs_X = X.abs()
    max_val = abs_X.max(dim=-1)[0]
    mean_val = abs_X.mean(dim=-1) + 1e-6
    features["max_to_mean_ratio"] = max_val / mean_val
    
    # 3. L1/L2 范数比: 衡量能量集中度/稀疏度 (比值越小越稀疏)
    l1_norm = abs_X.sum(dim=-1)
    l2_norm = torch.linalg.norm(X, dim=-1) + 1e-6
    features["l1_l2_ratio"] = l1_norm / l2_norm
    
    # 4. 零值主导率 (Zero-Domination): 评估在 FP4 缩放下，有多少元素会被归零
    # 在 FP4 E2M1 中，如果 abs(x)/scale <= 0.25，会被量化为 0
    # 我们用 max_val 来粗略估计 scale
    zeros_mask = abs_X < (0.25 * max_val.unsqueeze(-1))
    features["zeros_ratio"] = zeros_mask.float().mean(dim=-1)
    
    # 5. 孤狼绝对动能 (Outlier Absolute Energy): Max(|X|)^2
    # 理论预测：它与绝对误差增量 Delta L 呈极强正相关 (>0.85)
    features["energy_max"] = max_val ** 2
    
    # 6. 离群剥离比 (Max-to-Rest-Mean): Max / Mean(Rest)
    # 理论预测：它与相对损失率 Delta L_rel 呈强正相关
    rest_mean = (l1_norm - max_val) / 15.0
    features["max_to_rest_mean"] = max_val / (rest_mean + 1e-6)
    
    return features

def analyze_layer_activation(layer_name, X, quantizer, transform):
    print(f"\n{'='*50}\n分析层: {layer_name}\n{'='*50}")
    
    # 将激活值 Reshape 为 [N, 16]
    X_shape_orig = X.shape
    N_elements = X.numel()
    
    # 我们只抽取部分样本进行分析以防显存溢出 (比如最多取 100 万个 group)
    max_groups = 1000000
    X_groups = X.view(-1, 16)
    if X_groups.shape[0] > max_groups:
        # 随机采样以加速计算
        indices = torch.randperm(X_groups.shape[0])[:max_groups].to(X.device)
        X_groups = X_groups[indices]
    
    N_groups = X_groups.shape[0]
    print(f"总计分析组数 (Groups): {N_groups}")

    # 1. 提取未旋转特征
    features = compute_group_features(X_groups)
    
    # 2. 计算未旋转的量化 MSE
    # 让 Quantizer 内部处理分组逻辑
    scales, zeros = quantizer.get_quantization_params(X_groups)
    X_q_norot = quantizer(X_groups, scales, zeros)
    mse_norot = (X_q_norot - X_groups).pow(2).mean(dim=-1)
    
    # 3. 计算旋转后的量化 MSE
    # 注意: LLaMA 3 8B intermediate_size 不一定能整除 128
    # down_proj 的输入维度是 intermediate_size
    # 3. 计算旋转后的量化 MSE (Hadamard size = 16)
    # 因为 Hadamard size 也是 16，所以可以直接对 X_groups 进行变换
    X_rot = transform(X_groups)
    
    scales_rot, zeros_rot = quantizer.get_quantization_params(X_rot)
    X_q_rot_inner = quantizer(X_rot, scales_rot, zeros_rot)
    
    # 反向旋转回来算最终误差
    X_q_rot = transform(X_q_rot_inner, inv_t=True)
    mse_rot = (X_q_rot - X_groups).pow(2).mean(dim=-1)
    
    # 4. 计算误差增量 (Delta L)
    # Delta L > 0 代表旋转让误差变大了 (Bad)
    delta_L = mse_rot - mse_norot
    # 计算相对损失率 (Delta L_rel)
    delta_L_rel = delta_L / (mse_norot + 1e-12)
    
    # === 统计相关性 ===
    print("\n--- 特征与量化损失的相关系数与显著性 P 值 ---")
    from scipy.stats import pearsonr, spearmanr
    
    delta_L_cpu = delta_L.cpu().numpy()
    delta_L_rel_cpu = delta_L_rel.cpu().numpy()
    
    header = f"{'Feature':20} | {'Pearson (vs ΔL)':18} | {'Spearman (vs ΔL)':18}"
    print(header)
    print("-" * len(header))
    
    for feat_name, feat_vals in features.items():
        feat_vals_cpu = feat_vals.cpu().numpy()
        
        # Pearson 相关性 (线性)
        corr_p, p_p = pearsonr(feat_vals_cpu, delta_L_cpu)
        # Spearman 相关性 (秩相关/非线性单调)
        corr_s, p_s = spearmanr(feat_vals_cpu, delta_L_cpu)
        
        print(f"{feat_name:20} | {corr_p:8.4f} (p={p_p:.1e}) | {corr_s:8.4f} (p={p_s:.1e})")
        
    # === 对比 Good vs Bad Groups ===
    # 排序 Delta L
    sorted_indices = torch.argsort(delta_L)
    
    # 取 Top 5% 最差的组 (Bad Groups, Delta L 显著大于 0)
    # 取 Bottom 5% 最好的组 (Good Groups, Delta L < 0 或增加极少)
    top_k = int(N_groups * 0.05)
    good_indices = sorted_indices[:top_k]
    bad_indices = sorted_indices[-top_k:]
    
    print(f"\n--- Good Groups (Top 5% 受益于旋转) vs Bad Groups (Top 5% 旋转后变糟) ---")
    print(f"Avg Delta L -> Good: {delta_L[good_indices].mean().item():.2e} | Bad: {delta_L[bad_indices].mean().item():.2e}")
    
    for feat_name, feat_vals in features.items():
        good_val = feat_vals[good_indices].mean().item()
        bad_val = feat_vals[bad_indices].mean().item()
        print(f"{feat_name:20} -> Good: {good_val:.4f} | Bad: {bad_val:.4f}")
        
    # print("\n--- 恶化最严重的前 10 个组 (Top 10 Worst Bad Groups) ---")
    # worst_10_indices = sorted_indices[-10:].tolist()
    # worst_10_indices.reverse() # 从最坏的开始
    
    # for i, idx in enumerate(worst_10_indices):
    #     orig_group = X_groups[idx].cpu().numpy()
    #     rot_group = X_rot[idx].cpu().numpy()
    #     orig_mse = mse_norot[idx].item()
    #     rot_mse = mse_rot[idx].item()
    #     d_L = delta_L[idx].item()
        
    #     print(f"Top {i+1} Worst Group (Delta L = {d_L:.2e}):")
    #     print(f"  Orig MSE: {orig_mse:.2e} | Rot MSE: {rot_mse:.2e}")
    #     print(f"  Orig Values: {[float(f'{v:.4f}') for v in orig_group]}")
    #     print(f"  Rot  Values: {[float(f'{v:.4f}') for v in rot_group]}")
    #     print("-" * 60)

    print("\n--- 收益最高的前 5 个组 (Top 5 Best Good Groups) ---")
    best_5_indices = sorted_indices[:5].tolist()
    for i, idx in enumerate(best_5_indices):
        orig_group = X_groups[idx].cpu().numpy()
        rot_group = X_rot[idx].cpu().numpy()
        orig_mse = mse_norot[idx].item()
        rot_mse = mse_rot[idx].item()
        d_L = delta_L[idx].item()
        
        print(f"Top {i+1} Best Group (Delta L = {d_L:.2e}):")
        print(f"  Orig MSE: {orig_mse:.2e} | Rot MSE: {rot_mse:.2e}")
        print(f"  Orig Values: {[float(f'{v:.4f}') for v in orig_group]}")
        print(f"  Rot  Values: {[float(f'{v:.4f}') for v in rot_group]}")
        print("-" * 60)
        
    # === 分位数/分箱分析 (Decile Binning Analysis) ===
    print("\n--- 分位数分箱分析 (Decile Binning Analysis) ---")
    print("基于 'energy_max' 将组分为 10 个区间 (Deciles)，求每份的平均 Delta L。")
    print("目的：抵消随机舍入噪声，展现能量大小与量化损失的宏观非线性趋势。")
    
    energy_vals = features["energy_max"]
    sorted_energy_indices = torch.argsort(energy_vals)
    decile_size = N_groups // 10
    
    print(f"{'Decile (按能量升序)':<18} | {'Avg Energy Max':<18} | {'Avg Delta L':<15} | {'Avg Delta L_rel':<15}")
    print("-" * 75)
    
    for i in range(10):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < 9 else N_groups
        bin_indices = sorted_energy_indices[start_idx:end_idx]
        
        avg_energy = energy_vals[bin_indices].mean().item()
        avg_delta_l = delta_L[bin_indices].mean().item()
        avg_delta_l_rel = delta_L_rel[bin_indices].mean().item()
        
        print(f"D{i+1:<17} | {avg_energy:<18.4f} | {avg_delta_l:<15.2e} | {avg_delta_l_rel:<15.4f}")
        
    # === 绘图逻辑 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xscale('log')
    # 使用 symlog 可以在对数域同时展示正值（恶化）和负值（受益）
    # linthresh 定义了 0 附近的线性区间大小，建议设为最小 MSE 变化的量级
    ax.set_yscale('symlog', linthresh=1e-6)
    
    # 1. 绘制散点图 (随机采样 10000 个点)
    sample_size = min(10000, N_groups)
    sample_indices = torch.randperm(N_groups)[:sample_size]
    x_scatter = energy_vals[sample_indices].cpu().numpy()
    y_scatter = delta_L[sample_indices].cpu().numpy()
    
    # 现在不再过滤负值，直接全部画出
    ax.scatter(x_scatter, y_scatter, alpha=0.3, s=8, label='Groups (Sampled)', color='royalblue')
    
    # 2. 绘制分箱趋势线 (Decile Means)
    decile_energies = []
    decile_deltas = []
    for i in range(10):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < 9 else N_groups
        bin_indices = sorted_energy_indices[start_idx:end_idx]
        d_e = energy_vals[bin_indices].mean().item()
        d_l = delta_L[bin_indices].mean().item()
        
        decile_energies.append(d_e)
        decile_deltas.append(d_l)
        
    ax.plot(decile_energies, decile_deltas, marker='D', markersize=8, linestyle='-', color='red', linewidth=3, label='Decile Binning Means')
    
    ax.set_xlabel('Outlier Absolute Energy ($Max(|X|)^2$) [Log Scale]', fontsize=12)
    ax.set_ylabel('Quantization Loss Increase ($\Delta L$) [SymLog Scale]', fontsize=12)
    ax.set_title(f'Hadamard Rotation Impact: {layer_name} (SymLog)', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    plot_path = f"analysis_{layer_name}_binning.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"【图表已生成】: {plot_path}")
    
    print("\n")


def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    print(f"Loading model {model_path}...")
    
    # 仅在 CPU 上加载，然后手动移到 GPU 上处理特定层以防 OOM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 尝试加载模型
    # 如果 accelerate 有问题，我们不用 device_map="auto"，手动转到 device
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    if not hasattr(model, "hf_device_map"):
        model = model.to(device)
    
    # 构造一条简单的输入，或者从 fineweb-edu 拿一条长序列
    text = "Machine learning quantization is an important technique to reduce model size. " * 50
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 初始化量化器和变换矩阵
    quantizer = Quantizer(
        bits=4,
        symmetric=True,
        format="nvfp",
        granularity="group",
        observer="minmax",
        group_size=16,          # 内部被重置为 None
        scale_precision="e4m3"
    )
    
    transform = build_transform("hadamard", group_size=16, device=device, dtype=torch.bfloat16)

    # 我们要 Hook 层 1 和 31 的 down_proj
    # LLaMA 3 8B 层索引是 0 到 31
    target_layers = {"Layer_0": model.model.layers[0].mlp.down_proj, 
                     "Layer_15": model.model.layers[15].mlp.down_proj,
                     "Layer_31": model.model.layers[31].mlp.down_proj}
                     
    activations = {}
    
    def get_activation_hook(name):
        def hook(module, input, output):
            # 截取输入 x
            activations[name] = input[0].detach().clone()
        return hook

    handles = []
    for name, layer in target_layers.items():
        handles.append(layer.register_forward_hook(get_activation_hook(name)))
        
    print("Running forward pass to collect activations...")
    with torch.no_grad():
        model(**inputs)
        
    for h in handles:
        h.remove()
        
    # 开始分析
    for name, act in activations.items():
        analyze_layer_activation(name, act.to(device).float(), quantizer, transform)

if __name__ == "__main__":
    main()
