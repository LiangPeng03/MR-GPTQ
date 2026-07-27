import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import gc

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data_utils import get_data
from src.quantization.gics import compute_global_scale, scale_to_e4m3
from src.quantization.quant_ops import cast_to_fp4

def analyze_tensor_zones(tensor, group_size=16):
    device = tensor.device
    t = tensor.contiguous().view(-1, group_size).float()
    abs_t = t.abs() # 真实绝对值
    
    # 按照真实的 NVFP4 标准计算 scale
    gs = compute_global_scale(tensor)
    group_max = abs_t.amax(dim=-1, keepdim=True)
    raw_scale = group_max / 6.0
    raw_scale[raw_scale == 0] = 1.0
    # 这里的 scale 就是你提到的 Tensor scale * group scale 的 E4M3 最终映射结果
    scale = scale_to_e4m3(raw_scale, gs)
    
    # 归一化值，仅用于传入 cast_to_fp4 获取离散网格点
    normalized = abs_t / scale
    grid_points = cast_to_fp4(normalized)
    
    # 计算均方误差 (在真实尺度上计算：原始值 - 量化后重构值)
    error_sq = (abs_t - (grid_points * scale)) ** 2
    
    # 区间定义：完全按照你要求的“真实值与 scale 乘以比如 4.5 比较”的机制
    # 中损失区: 2.25~2.75, 3.25~3.75, 4.25~4.5, 5.5~5.75
    mask_med = ((abs_t >= scale * 2.25) & (abs_t <= scale * 2.75)) | \
               ((abs_t >= scale * 3.25) & (abs_t <= scale * 3.75)) | \
               ((abs_t >= scale * 4.25) & (abs_t <= scale * 4.50)) | \
               ((abs_t >= scale * 5.50) & (abs_t <= scale * 5.75))
               
    # 高损失区: 4.5~5.5
    mask_high = (abs_t > scale * 4.5) & (abs_t < scale * 5.5)
    
    # 其他区域: Safe/Other (包括了最大值本身，因为它约等于 scale * 6.0)
    mask_other = ~(mask_med | mask_high)
    
    # 统计数量
    count_med = mask_med.sum().item()
    count_high = mask_high.sum().item()
    count_other = mask_other.sum().item()
    
    # 统计真实尺度的 MSE
    mse_med = (error_sq * mask_med.float()).sum().item()
    mse_high = (error_sq * mask_high.float()).sum().item()
    mse_other = (error_sq * mask_other.float()).sum().item()
    
    return {
        "count": [count_other, count_med, count_high],
        "mse": [mse_other, mse_med, mse_high],
        "total_mse": mse_other + mse_med + mse_high
    }

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_id = "meta-llama/Meta-Llama-3-8B"
    print(f"Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    
    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # 获取 31 层 down_proj 权重
    print("Analyzing Layer 31 down_proj weights...")
    layer_idx = 31
    weight_tensor = model.model.layers[layer_idx].mlp.down_proj.weight.data
    weight_stats = analyze_tensor_zones(weight_tensor)
    
    # 获取激活值
    print("Collecting activations...")
    dataset = get_data("c4", tokenizer, 16, 2048)
    
    activations = []
    
    def hook_fn(module, inp, out):
        val = inp[0].detach()
        activations.append(val.cpu())
        
    hook = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i].unsqueeze(0).to(device)
            model(batch)
            if len(activations) >= 16:
                break
                
    hook.remove()
    
    print("Analyzing Layer 31 down_proj activations...")
    act_tensor = torch.cat(activations, dim=0).to(device)
    act_stats = analyze_tensor_zones(act_tensor)
    
    # 绘图
    print("Plotting charts...")
    labels = ['Safe/Low Loss', 'Medium Loss Zone', 'High Loss Zone']
    colors = ['#66b3ff', '#ffcc99', '#ff9999']
    
    fig = plt.figure(figsize=(14, 16))
    
    # 顶部画数轴 (色块全部画在主轴上方, 像比例尺一样)
    ax_scale = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_scale.set_xlim(0, 6.5)
    ax_scale.set_ylim(0, 1)
    ax_scale.axis('off')

    # 主轴线 (不超出 6.0)
    ax_scale.plot([0, 6.0], [0.5, 0.5], color='black', linewidth=2)

    # 在主轴上方添加损失区 (像尺子的色块, 单侧, 底边贴在轴线)
    med_zones = [(2.25, 0.5), (3.25, 0.5), (4.25, 0.25), (5.5, 0.25)]
    high_zones = [(4.5, 1.0)]

    # 先画中损失区
    for start, width in med_zones:
        rect = Rectangle((start, 0.5), width, 0.16,
                         facecolor=colors[1], alpha=0.75,
                         edgecolor='darkorange', linewidth=1.2)
        ax_scale.add_patch(rect)

    # 再画高损失区 (覆盖中损失区在 [4.5, 5.5] 的部分)
    for start, width in high_zones:
        rect = Rectangle((start, 0.5), width, 0.16,
                         facecolor=colors[2], alpha=0.75,
                         edgecolor='darkred', linewidth=1.2)
        ax_scale.add_patch(rect)

    # 标出每个色块的边界坐标 (在色块正上方, 用对应颜色)
    for start, width in med_zones:
        ax_scale.text(start, 0.68, f"{start:.2f}",
                      ha='center', va='bottom', fontsize=9,
                      color='darkorange', fontweight='bold')
        ax_scale.text(start + width, 0.68, f"{start + width:.2f}",
                      ha='center', va='bottom', fontsize=9,
                      color='darkorange', fontweight='bold')

    for start, width in high_zones:
        ax_scale.text(start, 0.68, f"{start:.2f}",
                      ha='center', va='bottom', fontsize=9,
                      color='darkred', fontweight='bold')
        ax_scale.text(start + width, 0.68, f"{start + width:.2f}",
                      ha='center', va='bottom', fontsize=9,
                      color='darkred', fontweight='bold')

    # 区域总标签 (在边界坐标与刻度数字之间)
    ax_scale.text(3.0, 0.79, "Medium Loss Zone",
                  color='darkorange', ha='center', va='bottom',
                  fontsize=12, fontweight='bold')
    ax_scale.text(5.0, 0.79, "High Loss Zone",
                  color='darkred', ha='center', va='bottom',
                  fontsize=12, fontweight='bold')

    # 标出 E2M1 锚点: 短竖线在横线上方 (色块之上), 黑色数字在横线下方
    grid_pts = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    for pt in grid_pts:
        # 短竖线: 在横线上方 0.50 到 0.55
        ax_scale.plot([pt, pt], [0.50, 0.55], color='black', linewidth=2)
        # 黑色数字: 在横线下方
        ax_scale.text(pt, 0.40, str(pt), ha='center', va='top',
                      fontsize=12, fontweight='bold')

    ax_scale.set_title("E2M1 FP4 Normalized Grid & Loss Zones", fontsize=16, pad=20)
    
    # 饼图
    ax_w_cnt = plt.subplot2grid((3, 2), (1, 0))
    ax_w_cnt.pie(weight_stats["count"], labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
    ax_w_cnt.set_title('Weights: Values Count Proportion\n')
    
    ax_w_mse = plt.subplot2grid((3, 2), (1, 1))
    ax_w_mse.pie(weight_stats["mse"], labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
    ax_w_mse.set_title(f'Weights: MSE Contribution Proportion\nTotal Absolute MSE: {weight_stats["total_mse"]:.2f}')
    
    ax_a_cnt = plt.subplot2grid((3, 2), (2, 0))
    ax_a_cnt.pie(act_stats["count"], labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
    ax_a_cnt.set_title('Activations: Values Count Proportion\n')
    
    ax_a_mse = plt.subplot2grid((3, 2), (2, 1))
    ax_a_mse.pie(act_stats["mse"], labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
    ax_a_mse.set_title(f'Activations: MSE Contribution Proportion\nTotal Absolute MSE: {act_stats["total_mse"]:.2f}')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "figure2b.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
