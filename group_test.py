import os
import sys
import torch
import numpy as np

# 确保能引入项目内的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.quantization.quantizer import Quantizer
from src.transforms.transforms import build_transform

def create_synthetic_tensors():
    """
    创建控制变量法所需的合成 16 维张量
    """
    base = 0.1
    tensors = {}
    
    # 辅助函数：快速生成基础张量
    def make_tensor(*peaks):
        t = torch.full((16,), base)
        for i, val in enumerate(peaks):
            t[i] = val
        return t

    # 1. 单峰组 (Single Peak)
    tensors["Single Peak (Positive)"] = make_tensor(100.0)
    tensors["Single Peak (Negative)"] = make_tensor(-100.0)
    
    # 2. 等量双峰组 (Double Peak - Equal)
    tensors["Double Peak (Same Sign)"] = make_tensor(100.0, 100.0)
    tensors["Double Peak (Opposite Sign)"] = make_tensor(100.0, -100.0)
    
    # 3. 异量双峰组 (Double Peak - Unequal)
    tensors["Double Peak (100, 50)"] = make_tensor(100.0, 50.0)
    tensors["Double Peak (100, -50)"] = make_tensor(100.0, -50.0)
    
    # 4. 三峰组 (Triple Peak)
    tensors["Triple Peak (Same Sign)"] = make_tensor(100.0, 100.0, 100.0)
    tensors["Triple Peak (Mixed Sign)"] = make_tensor(100.0, 100.0, -100.0)
    
    # 5. 真实采样组 (Real Samples from Images)
    tensors["Real Sample @!!(Same Sign)"] = torch.tensor([-0.4609,0.0552,-0.2139,0.063,0.0006,0.0032,1.5078,-0.0073,-0.2695,-25.75,0.0986,0.042,-0.016,-0.0486,-0.0055,0.017
    ])
    # 样本 1: 含有两个同号大值 (5.5938, 6.6562)
    tensors["Real Sample (Same Sign)"] = torch.tensor([
        -0.0635, 0.01, 5.5938, -0.2227, 6.6562, -0.0588, 0.0013, 0.0737, 
        -0.0693, -0.0002, 0.0021, -0.0723, -0.0361, -0.015, -0.0266, -0.0049
    ])
    
    # 样本 2: 含有异号大值 (1.5078, -25.75)
    tensors["Real Sample (Opposite Sign)"] = torch.tensor([
        -0.0635, 0.01, 5.5938, -0.2227, -6.6562, -0.0588, 0.0013, 0.0737, 
        -0.0693, -0.0002, 0.0021, -0.0723, -0.0361, -0.015, -0.0266, -0.0049
    ])
    tensors["Real Sample 0 (Same Sign)"] = torch.tensor([
        -0.0635, 0.01, 0.0738, -0.2227, 0.0562, -0.0588, 0.0013, 0.0737, 
        -0.0693, -0.0002, 0.0021, -0.0723, -0.0361, -0.015, -0.0266, -0.0049
    ])
    
    # 样本 2: 含有异号大值 (1.5078, -25.75)
    tensors["Real Sample 0 (Opposite Sign)"] = torch.tensor([
        -0.0635, 0.01, 0.0738, -0.2227, -0.0562, -0.0588, 0.0013, 0.0737, 
        -0.0693, -0.0002, 0.0021, -0.0723, -0.0361, -0.015, -0.0266, -0.0049
    ])
    tensors["Real Sample 3 (same Sign)"] = torch.tensor([
        -0.0635, 0.01, 5.5938, -0.2227, 6.6562, -0.0588, 5, 0.0737, 
        -0.0693, -0.0002, 0.0021, -0.0723, -0.0361, -0.015, -0.0266, -0.0049
    ])
    tensors["Real Sample 3 (opposite Sign)"] = torch.tensor([
        -0.0635, 0.01, 5.5938, -0.2227, -6.6562, -0.0588, -5, 0.0737, 
        -0.0693, -0.0002, 0.0021, -0.0723, -0.0361, -0.015, -0.0266, -0.0049
    ])
    tensors["Real Sample 4 (same Sign)"] = torch.tensor([
        -0.0635, 0.01, 5.5938, -0.2227, 6.6562, -0.0588, -5, 0.0737, 
        -0.0693, -4.8, 0.0021, -0.0723, -0.0361, -0.015, -0.0266, -0.0049
    ])
    tensors["Real Sample 4 (opposite Sign)"] = torch.tensor([
        -0.0635, 0.01, 5.5938, -0.2227, 6.6562, -0.0588, 5, 0.0737, 
        -0.0693, 4.8, 0.0021, -0.0723, -0.0361, -0.015, -0.0266, -0.0049
    ])
    tensors["Real Sample 5 (same Sign)"] = torch.tensor([
        -4.5, 0.01, 5.5938, -0.2227, 6.6562, -0.0588, -5, 0.0737, 
        -0.0693, -4.8, 0.0021, -0.0723, -0.0361, -0.015, -0.0266, -0.0049
    ])
    tensors["Real Sample 5 (opposite Sign)"] = torch.tensor([
        4.5, 0.01, 5.5938, -0.2227, 6.6562, -0.0588, 5, 0.0737, 
        -0.0693, 4.8, 0.0021, -0.0723, -0.0361, -0.015, -0.0266, -0.0049
    ])

    # 6. 基准组 (Baseline)
    torch.manual_seed(42)
    tensors["Random Normal Noise"] = torch.randn(16) * 10
    
    return tensors

def evaluate_tensor(name, t, quantizer, transform, device):
    # 将输入转为 [1, 16] 的形式并移至 device
    X = t.unsqueeze(0).to(device).bfloat16()
    
    # 1. 原始量化 MSE
    scales, zeros = quantizer.get_quantization_params(X)
    X_q_norot = quantizer(X, scales, zeros)
    mse_norot = (X_q_norot - X).pow(2).mean().item()
    
    # 2. 旋转量化 MSE
    X_rot = transform(X)
    scales_rot, zeros_rot = quantizer.get_quantization_params(X_rot)
    X_q_rot_inner = quantizer(X_rot, scales_rot, zeros_rot)
    
    # 为了对比，我们要看旋转后的“中间形态” X_rot 
    # 以及反旋转回来后的量化结果 X_q_rot
    X_q_rot = transform(X_q_rot_inner, inv_t=True)
    mse_rot = (X_q_rot - X).pow(2).mean().item()
    
    delta_L = mse_rot - mse_norot
    
    return mse_norot, mse_rot, delta_L, X.squeeze(0).float().cpu().numpy(), X_rot.squeeze(0).detach().float().cpu().numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: fast_hadamard_transform usually requires CUDA. This script might fail on CPU.")
    
    # 初始化量化器和变换 (NVFP4, E4M3, Group Size 16)
    quantizer = Quantizer(
        bits=4, 
        symmetric=True, 
        format="nvfp", 
        granularity="group", 
        group_size=16, 
        scale_precision="e4m3"
    )
    
    transform = build_transform("hadamard", group_size=16, device=device, dtype=torch.bfloat16)
    
    tensors = create_synthetic_tensors()
    
    print("=" * 100)
    print("Synthetic Ablation Test: Impact of Activation Shape on Hadamard Rotation")
    print("=" * 100)
    
    for name, t in tensors.items():
        orig_mse, rot_mse, delta, v_orig, v_rot = evaluate_tensor(name, t, quantizer, transform, device)
        
        # 标记变好还是变坏
        status = "❌ 恶化" if delta > 0 else "✅ 改善"
        if abs(delta) < 1e-5:
            status = "➖ 极小"
            
        print(f"Profile: {name}")
        print(f"  MSE: {orig_mse:.4f} (Orig) -> {rot_mse:.4f} (Rot) | Delta L: {delta:.4f} {status}")
        print(f"  Orig Values: {[float(f'{x:.2f}') for x in v_orig]}")
        print(f"  Rot  Values: {[float(f'{x:.2f}') for x in v_rot]}")
        print("-" * 100)

if __name__ == "__main__":
    main()
