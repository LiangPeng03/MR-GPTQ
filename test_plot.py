import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def main():
    model_path = "meta-llama/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_idx = 31

    print(f"Loading {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model.eval()

    text = "The quick brown fox jumps over the lazy dog. Machine learning is transforming artificial intelligence research across the globe. Data structures and algorithms form the foundation of modern computer science and software engineering practices."
    calib = tokenizer(text, return_tensors="pt").to(device)

    layer = model.model.layers[layer_idx].mlp.down_proj
    cache = []
    def hook(m, i, o): cache.append(i[0].detach())
    handle = layer.register_forward_hook(hook)
    
    print("Running forward pass...")
    with torch.no_grad():
        model(**calib)
    handle.remove()

    X = cache[0].view(-1, cache[0].shape[-1]).float().cpu().numpy()
    num_tokens, dim = X.shape
    print(f"Captured activation shape: {X.shape}")

    fp4_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    group_size = 16
    n_groups = 16  # 16 groups (256 channels)
    N = group_size * n_groups

    def quantize_fp4(x, group_size=16):
        q_x = np.zeros_like(x)
        for i in range(len(x) // group_size):
            group = x[i*group_size:(i+1)*group_size]
            max_val = np.max(np.abs(group))
            if max_val == 0:
                continue
            S = max_val / 6.0
            scaled = np.abs(group) / S
            idx = np.abs(scaled[:, None] - fp4_vals).argmin(axis=1)
            q_group = fp4_vals[idx] * S * np.sign(group)
            q_x[i*group_size:(i+1)*group_size] = q_group
        mse = np.mean((x - q_x)**2)
        return q_x, mse

    num_samples = 3  
    print(f"Randomly selecting {num_samples} representative blocks (256 channels each)...")
    
    token_norms = np.linalg.norm(X, axis=1)
    valid_tokens = np.argsort(token_norms)[len(token_norms)//2:]
    
    samples = []
    np.random.seed(42)
    for _ in range(num_samples):
        tok_idx = np.random.choice(valid_tokens)
        st = np.random.randint(0, (dim - N) // group_size) * group_size
        slice_64 = X[tok_idx, st:st+N]
        samples.append((tok_idx, st, slice_64))

    print("Generating plot...")
    
    fig, axs = plt.subplots(3, num_samples, figsize=(8 * num_samples, 12))
    colors = plt.cm.tab20(np.linspace(0, 1, n_groups))

    total_mse_orig = 0
    total_mse_conc = 0
    total_mse_disp = 0

    for col_idx, (tok_idx, st, x_orig) in enumerate(samples):
        # Original
        q_orig, mse_orig = quantize_fp4(x_orig)
        total_mse_orig += mse_orig

        # Concentrated
        idx_concentrated = np.argsort(np.abs(x_orig))[::-1]
        x_concentrated = x_orig[idx_concentrated]
        q_concentrated, mse_concentrated = quantize_fp4(x_concentrated)
        total_mse_conc += mse_concentrated

        # Dispersed
        idx_sorted = np.argsort(np.abs(x_orig))[::-1]
        x_dispersed = np.zeros_like(x_orig)
        for i in range(n_groups):
            x_dispersed[i*group_size:(i+1)*group_size] = x_orig[idx_sorted[i::n_groups]]
        q_dispersed, mse_dispersed = quantize_fp4(x_dispersed)
        total_mse_disp += mse_dispersed

        methods = [
            ("Original Arrangement", x_orig, q_orig, mse_orig),
            ("Concentrated Outliers", x_concentrated, q_concentrated, mse_concentrated),
            ("Dispersed Outliers", x_dispersed, q_dispersed, mse_dispersed)
        ]

        for row_idx, (title, x_val, q_val, mse) in enumerate(methods):
            ax = axs[row_idx, col_idx] if num_samples > 1 else axs[row_idx]
            for i in range(n_groups):
                start = i * group_size
                end = start + group_size
                x_group = x_val[start:end]
                q_group = q_val[start:end]
                x_pos = np.arange(start, end)
                
                ax.bar(x_pos, x_group, color=colors[i], alpha=0.6)
                
                # Replace black cross with matching colored dot (with subtle black border)
                ax.scatter(x_pos, q_group, color=colors[i], edgecolors='black', linewidth=0.5, marker='o', zorder=3, s=15)
                
                for j in range(group_size):
                    ax.plot([x_pos[j], x_pos[j]], [x_group[j], q_group[j]], color='red', alpha=0.5, linewidth=0.5)
                    
            ax.set_title(f"{title}\nToken {tok_idx}, Ch {st}-{st+N-1} | MSE: {mse:.4f}", fontsize=11)
            ax.set_xlim(-1, N)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

    avg_orig = total_mse_orig / num_samples
    avg_conc = total_mse_conc / num_samples
    avg_disp = total_mse_disp / num_samples

    fig.suptitle(f"Source: LLAMA3 8B Layer 31 down_proj input (Sampled {num_samples} Blocks, 256 Channels/Block)\n"
                 f"Global Avg MSE - Original: {avg_orig:.4f} | Concentrated: {avg_conc:.4f} | Dispersed: {avg_disp:.4f}", 
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dispersion_real_experiment_16groups.png')
    plt.savefig(save_path, dpi=200)
    print(f"Image generated successfully at {save_path}")

if __name__ == "__main__":
    main()
