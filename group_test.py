import os, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantization.quantizer import Quantizer
from src.transforms.transforms import build_transform
import matplotlib.pyplot as plt

def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_idx = 31
    
    print("Loading model and computing activations...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model.eval()
    
    text = "Detailed shape analysis of LLM activations. " * 300
    calib_data = tokenizer(text, return_tensors="pt").to(device)
    
    layer = model.model.layers[layer_idx].mlp.down_proj
    cache = []
    def hook(m, i, o): cache.append(i[0].detach())
    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        model(**calib_data)
    handle.remove()
    
    X = cache[0].view(-1, cache[0].shape[-1]).float() # (300, 14336) on CUDA
    dim = X.shape[-1]
    
    # ---------------- MXFP4 (Had128) Global Scan ----------------
    print("\nScanning for MXFP4 (Had128) worst INCREASE block...")
    mx_q = Quantizer(bits=4, format="mxfp", granularity="group", group_size=32, symmetric=True, scale_precision="e8m0")
    t_128 = build_transform("hadamard", size=128, group_size=128).to(device)
    
    X_128 = X.view(-1, 128)
    sc_nr_mx, z_nr_mx = mx_q.get_quantization_params(X_128)
    q_nr_mx = mx_q(X_128, sc_nr_mx, z_nr_mx)
    mse_nr_mx = ((q_nr_mx - X_128)**2).mean(dim=1)
    
    X_128_rot = t_128(X_128)
    sc_r_mx, z_r_mx = mx_q.get_quantization_params(X_128_rot)
    q_r_mx = mx_q(X_128_rot, sc_r_mx, z_r_mx)
    mse_r_mx = ((q_r_mx - X_128_rot)**2).mean(dim=1)
    
    delta_mx = mse_r_mx - mse_nr_mx
    worst_idx_mx = delta_mx.argmax().item()
    
    mx_token = worst_idx_mx // (dim // 128)
    mx_block = worst_idx_mx % (dim // 128)
    
    print("\n" + "="*60)
    print(f"MXFP4 WORST BLOCK FOUND: Token {mx_token}, Block 128-ch {mx_block}")
    
    block_128_orig = X_128[worst_idx_mx]
    block_128_q_nr = q_nr_mx[worst_idx_mx]
    block_128_rot = X_128_rot[worst_idx_mx]
    block_128_q_r = q_r_mx[worst_idx_mx]
    
    mx_sub_deltas = []
    print("\nSub-group (32-ch) Breakdown for this MXFP4 Block:")
    for g in range(4):
        sub_orig = block_128_orig[g*32:(g+1)*32]
        sub_rot = block_128_rot[g*32:(g+1)*32]
        mse_n = ((block_128_q_nr[g*32:(g+1)*32] - sub_orig)**2).mean().item()
        mse_r = ((block_128_q_r[g*32:(g+1)*32] - sub_rot)**2).mean().item()
        d = mse_r - mse_n
        mx_sub_deltas.append(d)
        print(f"  Group {g}: NoRot={mse_n:.6f}, Rot={mse_r:.6f}, Delta={d:>9.6f}")
        
    worst_sub_idx = np.argmax(mx_sub_deltas)
    block_delta_sum = sum(mx_sub_deltas)
    contribution = (mx_sub_deltas[worst_sub_idx] / block_delta_sum) * 100 if block_delta_sum > 0 else 0
    print("-" * 40)
    print(f"Is it single-group driven? Max Sub-group is {worst_sub_idx}, contributing {contribution:.1f}% to the total block increase.")

    # ---------------- NVFP4 (Had16) Global Scan ----------------
    print("\nScanning for NVFP4 (Had16) best DECREASE block...")
    nv_q = Quantizer(bits=4, format="nvfp", granularity="group", group_size=16, symmetric=True, scale_precision="e4m3")
    nv_q._track_global_scale = False
    from src.quantization.quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX
    from src.quantization.quantizer import get_reciprocal
    act_max_val = X.abs().max().to(torch.float32).view(1)
    nv_q.global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(act_max_val)).to(device)

    t_16 = build_transform("hadamard", size=16, group_size=16).to(device)
    X_16 = X.view(-1, 16)
    
    sc_nr_nv, z_nr_nv = nv_q.get_quantization_params(X_16)
    q_nr_nv = nv_q(X_16, sc_nr_nv, z_nr_nv)
    mse_nr_nv = ((q_nr_nv - X_16)**2).mean(dim=1)
    
    X_16_rot = t_16(X_16)
    sc_r_nv, z_r_nv = nv_q.get_quantization_params(X_16_rot)
    q_r_nv = nv_q(X_16_rot, sc_r_nv, z_r_nv)
    mse_r_nv = ((q_r_nv - X_16_rot)**2).mean(dim=1)
    
    delta_nv = mse_r_nv - mse_nr_nv
    best_idx_nv = delta_nv.argmin().item()
    
    nv_token = best_idx_nv // (dim // 16)
    nv_block = best_idx_nv % (dim // 16)
    
    print("\n" + "="*60)
    print(f"NVFP4 BEST BLOCK FOUND: Token {nv_token}, Block 16-ch {nv_block}")
    
    block_16_orig = X_16[best_idx_nv]
    block_16_q_nr = q_nr_nv[best_idx_nv]
    block_16_rot = X_16_rot[best_idx_nv]
    block_16_q_r = q_r_nv[best_idx_nv]
    
    def print_top3(orig, q, title):
        print(f"\n{title}:")
        orig_np = orig if isinstance(orig, np.ndarray) else orig.cpu().numpy()
        q_np = q if isinstance(q, np.ndarray) else q.cpu().numpy()
        sorted_idx = np.argsort(np.abs(orig_np))[::-1]
        for i in range(3):
            idx = sorted_idx[i]
            print(f"  Index {idx:>3}: Orig = {orig_np[idx]:>9.4f} | Quant = {q_np[idx]:>9.4f} | AbsErr = {abs(orig_np[idx]-q_np[idx]):>9.4f}")
            
    print_top3(block_128_orig, block_128_q_nr, "Top 3 Absolute Values - MXFP4 NoRot (128-ch)")
    print_top3(block_128_rot, block_128_q_r, "Top 3 Absolute Values - MXFP4 Rot (128-ch)")
    print("-" * 40)
    print_top3(block_16_orig, block_16_q_nr, "Top 3 Absolute Values - NVFP4 NoRot (16-ch)")
    print_top3(block_16_rot, block_16_q_r, "Top 3 Absolute Values - NVFP4 Rot (16-ch)")

    # ---------------- Token-wise Channel Stats ----------------
    print("\n" + "="*60)
    print("Channel-wise Statistics over 300 Tokens for the selected blocks:")
    
    # MXFP4 Stats
    X_mx_block = X[:, mx_block*128 : (mx_block+1)*128]
    mx_mean_abs = X_mx_block.abs().mean(dim=0)
    mx_max = X_mx_block.max(dim=0).values
    mx_min = X_mx_block.min(dim=0).values
    
    print(f"\n[MXFP4 128-ch Block {mx_block}] Top 3 Channels by Mean Absolute Value:")
    sorted_mean_mx = torch.argsort(mx_mean_abs, descending=True)
    for i in range(3):
        idx = sorted_mean_mx[i].item()
        print(f"  Channel {idx:>3}: MeanAbs = {mx_mean_abs[idx]:>8.4f}")
        
    print(f"\n[MXFP4 128-ch Block {mx_block}] Top 3 Channels by True Maximum Value:")
    sorted_max_mx = torch.argsort(mx_max, descending=True)
    for i in range(3):
        idx = sorted_max_mx[i].item()
        print(f"  Channel {idx:>3}: Max = {mx_max[idx]:>9.4f}, Min = {mx_min[idx]:>9.4f}")

    print(f"\n[MXFP4 128-ch Block {mx_block}] Top 3 Channels by True Minimum Value (Most Negative):")
    sorted_min_mx = torch.argsort(mx_min, descending=False)
    for i in range(3):
        idx = sorted_min_mx[i].item()
        print(f"  Channel {idx:>3}: Min = {mx_min[idx]:>9.4f}, Max = {mx_max[idx]:>9.4f}")

    # NVFP4 Stats
    X_nv_block = X[:, nv_block*16 : (nv_block+1)*16]
    nv_mean_abs = X_nv_block.abs().mean(dim=0)
    nv_max = X_nv_block.max(dim=0).values
    nv_min = X_nv_block.min(dim=0).values
    
    print(f"\n[NVFP4 16-ch Block {nv_block}] Top 3 Channels by Mean Absolute Value:")
    sorted_mean_nv = torch.argsort(nv_mean_abs, descending=True)
    for i in range(3):
        idx = sorted_mean_nv[i].item()
        print(f"  Channel {idx:>3}: MeanAbs = {nv_mean_abs[idx]:>8.4f}")
        
    print(f"\n[NVFP4 16-ch Block {nv_block}] Top 3 Channels by True Maximum Value:")
    sorted_max_nv = torch.argsort(nv_max, descending=True)
    for i in range(3):
        idx = sorted_max_nv[i].item()
        print(f"  Channel {idx:>3}: Max = {nv_max[idx]:>9.4f}, Min = {nv_min[idx]:>9.4f}")

    print(f"\n[NVFP4 16-ch Block {nv_block}] Top 3 Channels by True Minimum Value (Most Negative):")
    sorted_min_nv = torch.argsort(nv_min, descending=False)
    for i in range(3):
        idx = sorted_min_nv[i].item()
        print(f"  Channel {idx:>3}: Min = {nv_min[idx]:>9.4f}, Max = {nv_max[idx]:>9.4f}")
    print("="*60 + "\n")
    
    print("\nGenerating combined 2x2 plot: combined_extreme_cases.png")
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    
    # 1. MXFP4 NoRot
    ax = axes[0, 0]
    x_128 = np.arange(128)
    ax.bar(x_128, block_128_orig.cpu().numpy(), alpha=0.5, color='skyblue', label='Original')
    ax.scatter(x_128, block_128_q_nr.cpu().numpy(), color='blue', s=15, zorder=5, label='Quantized')
    for g in range(1, 4):
        ax.axvline(g*32 - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f"MXFP4 NoRot (128-ch) - MSE: {mse_nr_mx[worst_idx_mx]:.6f}")
    ax.legend()

    # 2. MXFP4 Rot
    ax = axes[0, 1]
    ax.bar(x_128, block_128_rot.cpu().numpy(), alpha=0.5, color='salmon', label='Rotated')
    ax.scatter(x_128, block_128_q_r.cpu().numpy(), color='red', s=15, zorder=5, label='Quantized')
    for g in range(1, 4):
        ax.axvline(g*32 - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f"MXFP4 Rot (128-ch) - MSE: {mse_r_mx[worst_idx_mx]:.6f}")
    ax.legend()
    
    # 3. NVFP4 NoRot
    ax = axes[1, 0]
    x_16 = np.arange(16)
    ax.bar(x_16, block_16_orig.cpu().numpy(), alpha=0.5, color='lightgreen', label='Original')
    ax.scatter(x_16, block_16_q_nr.cpu().numpy(), color='darkgreen', s=30, zorder=5, label='Quantized')
    ax.set_title(f"NVFP4 NoRot (16-ch) - MSE: {mse_nr_nv[best_idx_nv]:.6f}")
    ax.set_xticks(np.arange(0, 16, 2))
    ax.legend()
    
    # 4. NVFP4 Rot
    ax = axes[1, 1]
    ax.bar(x_16, block_16_rot.cpu().numpy(), alpha=0.5, color='orange', label='Rotated')
    ax.scatter(x_16, block_16_q_r.cpu().numpy(), color='darkorange', s=30, zorder=5, label='Quantized')
    ax.set_title(f"NVFP4 Rot (16-ch) - MSE: {mse_r_nv[best_idx_nv]:.6f}")
    ax.set_xticks(np.arange(0, 16, 2))
    ax.legend()
    
    plt.suptitle("Micro Analysis of Extreme Quantization Effects\nTop: MXFP4 Worst Increase | Bottom: NVFP4 Best Decrease", fontweight='bold', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("combined_extreme_cases.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
