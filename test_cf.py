import torch
import gc
import copy
import argparse
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantization import rtn_quantization, gptq_quantization
from src.utils.data_utils import get_data
from src.utils.model_utils import InputCollector, ForwardInterrupt
from src.quantization.gptq import GPTQ, Quantizer

from src.quantization.quantizer import Quantizer, get_reciprocal
from src.quantization.quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX

def apply_real_nvfp4_quant(x, group_size=16):
    device = x.device
    nv_q = Quantizer(bits=4, format="nvfp", granularity="group", group_size=group_size, symmetric=True, scale_precision="e4m3")
    nv_q._track_global_scale = False
    act_max = x.abs().max().to(torch.float32).view(1)
    # Replicate the global scale logic used in test_outlier_defs
    nv_q.global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(act_max)).to(device)
    
    x_16 = x.contiguous().view(-1, group_size)
    sc, z = nv_q.get_quantization_params(x_16)
    q = nv_q(x_16, sc, z)
    return q.view(x.shape)

def to_device(v, device):
    if isinstance(v, torch.Tensor): return v.to(device)
    if isinstance(v, tuple): return tuple(to_device(x, device) for x in v)
    if isinstance(v, list): return [to_device(x, device) for x in v]
    if isinstance(v, dict): return {k: to_device(val, device) for k, val in v.items()}
    return v

def maybe_first(obj):
    if isinstance(obj, tuple): return obj[0]
    return obj

def get_combined_weight(block, name):
    if name == "qkv": w = torch.cat([block.self_attn.q_proj.weight, block.self_attn.k_proj.weight, block.self_attn.v_proj.weight], dim=0)
    elif name == "o": w = block.self_attn.o_proj.weight
    elif name == "gate_up": w = torch.cat([block.mlp.gate_proj.weight, block.mlp.up_proj.weight], dim=0)
    elif name == "down": w = block.mlp.down_proj.weight
    else: return None
    return w.float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--dataset_name_or_path", type=str, default="fineweb-edu")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--num_sequences", type=int, default=32)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--format", type=str, default="nvfp")
    args = parser.parse_args()

    device = "cuda"
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    print("Loading calibration data...")
    calib_data = get_data(args.dataset_name_or_path, tokenizer, args.sequence_length, args.num_sequences, 42)
    
    model.config.use_cache = False
    model.requires_grad_(False)
    
    blocks = model.model.layers
    
    # Push data to get inputs for layers
    print("Capturing layer inputs through embedding...")
    blocks[0] = InputCollector(blocks[0], cpu_offload=False)
    model.get_input_embeddings().to(device)
    blocks[0] = blocks[0].to(device)
    for sample in calib_data:
        try:
            with torch.no_grad(): model(sample.to(device))
        except ForwardInterrupt:
            pass
    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module.cpu()
    model.get_input_embeddings().cpu()
    
    target_layers = [0, 15, 31]
    
    # We will test these strategies
    strategies = [
        {"name": "Baseline", "args": {"channel_resort": "none", "transform_class": "identity", "stagger_lambda": "auto"}},
        # {"name": "Stagger_A", "args": {"channel_resort": "stagger", "transform_class": "identity", "stagger_lambda": "0.0"}},
        # {"name": "Stagger_W", "args": {"channel_resort": "stagger", "transform_class": "identity", "stagger_lambda": "W_only"}},
        # {"name": "Stagger_W_A_2.0", "args": {"channel_resort": "stagger", "transform_class": "identity", "stagger_lambda": "2.0"}},
        {"name": "KMeans_FP4", "args": {"channel_resort": "kmeans_fp4", "transform_class": "identity", "stagger_lambda": "0.0"}},
        {"name": "KMeans_FP4_W", "args": {"channel_resort": "kmeans_fp4_w", "transform_class": "identity", "stagger_lambda": "0.0"}},
    ]
    
    stats = {} # layer -> name -> strat -> metrics
    
    from src.quantization.gptq import gptq_quantization
    # We cannot easily re-run gptq_quantization block by block if we modify it.
    # So we will write a custom block loop here that mimics gptq.py block quantization.
    
    for block_idx, block in enumerate(blocks):
        block = block.to(device)
        
        if block_idx not in target_layers:
            # Pass forward without quantization to maintain pure FP16 inputs
            print(f"Skipping layer {block_idx} (FP16 Pass)...")
            for i in range(len(input_args)):
                with torch.no_grad():
                    args_on_dev = to_device(input_args[i], device)
                    kwargs_on_dev = to_device(input_kwargs[i], device)
                    out = block(*args_on_dev, **kwargs_on_dev)
                    out_hidden = maybe_first(out).cpu()
                    input_args[i] = (out_hidden,) + input_args[i][1:]
            block = block.cpu()
            continue
            
        print(f"\n{'='*50}")
        print(f"==== Testing Layer {block_idx} ====")
        print(f"{'='*50}")
        
        # 1. Get Golden Outputs (FP16)
        golden_outputs = []
        for i in range(len(input_args)):
            with torch.no_grad():
                args_on_dev = to_device(input_args[i], device)
                kwargs_on_dev = to_device(input_kwargs[i], device)
                out = block(*args_on_dev, **kwargs_on_dev)
                golden_outputs.append(maybe_first(out).detach().cpu())
                
        stats[block_idx] = {}
        
        # 2. Iterate Strategies
        for strat in strategies:
            strat_name = strat["name"]
            
            # Set up mock args for this strategy
            strat_args = copy.copy(args)
            strat_args.channel_resort = strat["args"]["channel_resort"]
            strat_args.stagger_lambda = strat["args"]["stagger_lambda"]
            
            block_copy = copy.deepcopy(block).to(device)
            
            # (a) Collect act_caches
            act_caches = {}
            def hook_factory(name):
                def _hook(_, inp, out):
                    if name not in act_caches: act_caches[name] = []
                    if len(act_caches[name]) < 8:
                        act_caches[name].append(inp[0].detach().float().view(-1, inp[0].shape[-1]))
                return _hook
                
            resort_hooks = []
            resort_hooks.append(block_copy.self_attn.q_proj.register_forward_hook(hook_factory("qkv")))
            resort_hooks.append(block_copy.self_attn.o_proj.register_forward_hook(hook_factory("o")))
            resort_hooks.append(block_copy.mlp.gate_proj.register_forward_hook(hook_factory("gate_up")))
            resort_hooks.append(block_copy.mlp.down_proj.register_forward_hook(hook_factory("down")))
            
            for i in range(len(input_args)):
                with torch.no_grad():
                    args_on_dev = to_device(input_args[i], device)
                    kwargs_on_dev = to_device(input_kwargs[i], device)
                    block_copy(*args_on_dev, **kwargs_on_dev)
            for h in resort_hooks: h.remove()
            
            # (b) Apply Resort & Diagnostics
            for name, caches in list(act_caches.items()):
                if name not in stats[block_idx]: stats[block_idx][name] = {}
                
                X = torch.cat(caches, dim=0).to(device)
                threshold_A = torch.quantile(X.abs(), 0.9375, dim=1, keepdim=True)
                is_outlier_A = (X.abs() > threshold_A).float()
                
                W = get_combined_weight(block_copy, name).to(device)
                threshold_W = torch.quantile(W.abs(), 0.9375, dim=1, keepdim=True)
                is_outlier_W = (W.abs() > threshold_W).float()
                
                N_tokens = X.shape[0]
                M_out = W.shape[0]
                
                n_features, dim = is_outlier_A.shape
                n_groups = max(1, dim // 16)
                
                if strat_args.channel_resort in ["stagger", "stagger_swap"]:
                    if strat_args.stagger_lambda == "auto":
                        lambda_W = math.sqrt(N_tokens / M_out) if M_out > 0 else 1.0
                        joint_profile = torch.cat([is_outlier_A, is_outlier_W * lambda_W], dim=0)
                    elif strat_args.stagger_lambda == "W_only":
                        joint_profile = is_outlier_W
                    else:
                        lambda_W = float(strat_args.stagger_lambda)
                        joint_profile = torch.cat([is_outlier_A, is_outlier_W * lambda_W], dim=0)
                        
                    ch_freq = joint_profile.mean(dim=0)
                    
                    # Compute Perm
                    sorted_channels = torch.argsort(ch_freq, descending=True).tolist()
                    opt_groups = [[] for _ in range(n_groups)]
                    group_profiles = torch.zeros((n_groups, joint_profile.shape[0]), device=device, dtype=torch.float32)
                    group_sizes = torch.zeros(n_groups, device=device, dtype=torch.long)
                    for ch in sorted_channels:
                        c_profile = joint_profile[:, ch]
                        penalties = torch.mv(group_profiles, c_profile)
                        penalties[group_sizes >= 16] = float('inf')
                        best_group = penalties.argmin().item()
                        opt_groups[best_group].append(ch)
                        group_profiles[best_group] += c_profile
                        group_sizes[best_group] += 1
                        
                    if strat_args.channel_resort == "stagger_swap":
                        # Phase 2: MSE-Guided Pairwise Swap (Batched GPU version)
                        nv_q_fast = Quantizer(bits=4, format="nvfp", granularity="group", group_size=16, symmetric=True, scale_precision="e4m3")
                        global_act_max = X.abs().max().to(torch.float32).view(1)
                        nv_q_fast.global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(global_act_max)).to(X.device)
                        nv_q_fast._track_global_scale = False
                        
                        # Build grouped tensor for fast batch operations
                        flat_perm = []
                        for g in opt_groups: flat_perm.extend(g)
                        perm_t = torch.tensor(flat_perm, device=device, dtype=torch.long)
                        X_grouped = X[:, perm_t].view(N_tokens, n_groups, 16)  # (N, n_g, 16)
                        
                        # Initial quantize all groups at once
                        x_flat = X_grouped.reshape(-1, 16)
                        sc, z = nv_q_fast.get_quantization_params(x_flat)
                        xq_flat = nv_q_fast(x_flat, sc, z)
                        xq_grouped = xq_flat.view(N_tokens, n_groups, 16)
                        per_ch_err = ((xq_grouped - X_grouped) ** 2).sum(dim=0)  # (n_g, 16)
                        per_group_err = per_ch_err.sum(dim=1)  # (n_g,)
                        
                        num_iters = 30
                        for it in range(num_iters):
                            worst_g = per_group_err.argmax().item()
                            worst_ch = per_ch_err[worst_g].argmax().item()
                            
                            worst_group_base = X_grouped[:, worst_g, :].clone()  # (N, 16)
                            worst_ch_data = worst_group_base[:, worst_ch]         # (N,)
                            old_worst_err = per_group_err[worst_g].item()
                            
                            best_improvement = 0.0
                            best_swap = None
                            
                            for g_j in range(n_groups):
                                if g_j == worst_g: continue
                                
                                other_group = X_grouped[:, g_j, :]  # (N, 16)
                                old_other_err = per_group_err[g_j].item()
                                old_pair_err = old_worst_err + old_other_err
                                
                                # --- Batch eval: 16 modified worst groups ---
                                mod_w = worst_group_base.unsqueeze(0).expand(16, -1, -1).clone()  # (16, N, 16)
                                mod_w[:, :, worst_ch] = other_group.T  # (16, N)
                                mw_flat = mod_w.reshape(-1, 16)
                                sc_w, z_w = nv_q_fast.get_quantization_params(mw_flat)
                                mwq = nv_q_fast(mw_flat, sc_w, z_w).view(16, N_tokens, 16)
                                new_worst_errs = ((mwq - mod_w) ** 2).sum(dim=(1, 2))  # (16,)
                                
                                # --- Batch eval: 16 modified other groups ---
                                mod_o = other_group.unsqueeze(0).expand(16, -1, -1).clone()  # (16, N, 16)
                                for c_j in range(16):
                                    mod_o[c_j, :, c_j] = worst_ch_data
                                mo_flat = mod_o.reshape(-1, 16)
                                sc_o, z_o = nv_q_fast.get_quantization_params(mo_flat)
                                moq = nv_q_fast(mo_flat, sc_o, z_o).view(16, N_tokens, 16)
                                new_other_errs = ((moq - mod_o) ** 2).sum(dim=(1, 2))  # (16,)
                                
                                # Best swap among these 16 candidates
                                improvements = old_pair_err - (new_worst_errs + new_other_errs)
                                best_c_j = improvements.argmax().item()
                                imp_val = improvements[best_c_j].item()
                                
                                if imp_val > best_improvement:
                                    best_improvement = imp_val
                                    best_swap = (worst_g, worst_ch, g_j, best_c_j,
                                                 new_worst_errs[best_c_j].item(), new_other_errs[best_c_j].item())
                            
                            if best_improvement > 0 and best_swap is not None:
                                w_g, w_c, o_g, o_c, _, _ = best_swap
                                # Execute swap in X_grouped
                                temp_data = X_grouped[:, w_g, w_c].clone()
                                X_grouped[:, w_g, w_c] = X_grouped[:, o_g, o_c]
                                X_grouped[:, o_g, o_c] = temp_data
                                # Execute swap in opt_groups
                                temp_ch = opt_groups[w_g][w_c]
                                opt_groups[w_g][w_c] = opt_groups[o_g][o_c]
                                opt_groups[o_g][o_c] = temp_ch
                                # Update errors for both affected groups
                                for g_upd in [w_g, o_g]:
                                    g_data = X_grouped[:, g_upd, :]
                                    sc_u, z_u = nv_q_fast.get_quantization_params(g_data)
                                    gq = nv_q_fast(g_data, sc_u, z_u)
                                    per_ch_err[g_upd] = ((gq - g_data) ** 2).sum(dim=0)
                                    per_group_err[g_upd] = per_ch_err[g_upd].sum()
                            else:
                                break
                        
                    perm = []
                    for g in opt_groups: perm.extend(g)
                elif strat_args.channel_resort in ["kmeans_fp4", "kmeans_fp4_w"]:
                    # ============================================================
                    # NVFP4-Aware K-Means++ Channel Grouping
                    # ============================================================
                    if strat_args.channel_resort == "kmeans_fp4":
                        target_mat = X
                    else:
                        target_mat = W
                    
                    # Subsample tokens/rows to drastically speed up the distance calculations
                    # 4096 samples are statistically more than enough to capture the magnitude patterns
                    max_samples = 4096
                    if target_mat.shape[0] > max_samples:
                        # Use a deterministic subset for reproducibility
                        subset_idx = torch.linspace(0, target_mat.shape[0] - 1, max_samples, dtype=torch.long, device=device)
                        X_abs_T = target_mat[subset_idx].abs().T.float()  # (dim, max_samples)
                    else:
                        X_abs_T = target_mat.abs().T.float()  # (dim, N_items)
                    
                    # NVFP4 E2M1 grid multipliers
                    grid_mults = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device, dtype=torch.float32)
                    
                    def compute_loss_distances(all_ch_abs, ref_ch_abs, chunk_size=256):
                        """Batch compute loss distance of all channels to one reference channel with chunking."""
                        ref = ref_ch_abs.unsqueeze(0)  # (1, N_tokens)
                        n_ch = all_ch_abs.shape[0]
                        losses = torch.zeros(n_ch, device=device, dtype=torch.float32)
                        
                        for i in range(0, n_ch, chunk_size):
                            chunk = all_ch_abs[i:i+chunk_size]
                            p_max = torch.maximum(chunk, ref)  # (chunk_size, N_tokens)
                            p_min = torch.minimum(chunk, ref)  # (chunk_size, N_tokens)
                            scale = (p_max / 6.0).clamp(min=1e-10)  # (chunk_size, N_tokens)
                            grid = scale.unsqueeze(-1) * grid_mults  # (chunk_size, N_tokens, 8)
                            diff = (p_min.unsqueeze(-1) - grid).abs()  # (chunk_size, N_tokens, 8)
                            min_q = grid.gather(-1, diff.argmin(-1, keepdim=True)).squeeze(-1)
                            losses[i:i+chunk_size] = ((p_min - min_q) ** 2).sum(dim=1)
                            
                        return losses
                    
                    # --- Phase 1: K-Means++ Seed Selection ---
                    ch_sums = X_abs_T.sum(dim=1)  # (dim,)
                    accumulated_dist = torch.zeros(dim, device=device, dtype=torch.float32)
                    seed_order = []
                    
                    # First seed: channel with largest absolute sum
                    first_seed = ch_sums.argmax().item()
                    seed_order.append(first_seed)
                    accumulated_dist[first_seed] = -float('inf')
                    
                    # Compute distances to first seed
                    dists = compute_loss_distances(X_abs_T, X_abs_T[first_seed])
                    accumulated_dist += dists
                    accumulated_dist[first_seed] = -float('inf')  # re-set after addition
                    
                    for k in range(1, n_groups):
                        new_seed = accumulated_dist.argmax().item()
                        seed_order.append(new_seed)
                        accumulated_dist[new_seed] = -float('inf')
                        # Update accumulated distances
                        dists = compute_loss_distances(X_abs_T, X_abs_T[new_seed])
                        accumulated_dist += dists
                        # Already-selected seeds stay at -inf
                    
                    # --- Phase 2: Greedy Filling ---
                    opt_groups = [[s] for s in seed_order]
                    group_maxes = X_abs_T[seed_order].clone()  # (n_groups, N_tokens)
                    group_sizes = torch.ones(n_groups, device=device, dtype=torch.long)
                    
                    # Remaining channels sorted by absolute sum (descending)
                    seed_set = set(seed_order)
                    remaining = [c for c in range(dim) if c not in seed_set]
                    remaining_sums = ch_sums[remaining]
                    sorted_order = torch.argsort(remaining_sums, descending=True)
                    remaining_sorted = [remaining[i] for i in sorted_order.tolist()]
                    
                    for c in remaining_sorted:
                        c_abs = X_abs_T[c]  # (N_tokens,)
                        losses = torch.zeros(n_groups, device=device, dtype=torch.float32)
                        ref = c_abs.unsqueeze(0)  # (1, N_tokens)
                        
                        # Chunking over groups to avoid OOM
                        for i in range(0, n_groups, 256):
                            g_chunk = group_maxes[i:i+256]
                            p_max = torch.maximum(ref, g_chunk)
                            p_min = torch.minimum(ref, g_chunk)
                            scale = (p_max / 6.0).clamp(min=1e-10)
                            grid = scale.unsqueeze(-1) * grid_mults
                            diff = (p_min.unsqueeze(-1) - grid).abs()
                            min_q = grid.gather(-1, diff.argmin(-1, keepdim=True)).squeeze(-1)
                            losses[i:i+256] = ((p_min - min_q) ** 2).sum(dim=1)
                        
                        # Exclude full groups
                        losses[group_sizes >= 16] = float('inf')
                        best_g = losses.argmin().item()
                        
                        opt_groups[best_g].append(c)
                        group_maxes[best_g] = torch.maximum(group_maxes[best_g], c_abs)
                        group_sizes[best_g] += 1
                    
                    perm = []
                    for g in opt_groups: perm.extend(g)
                else:
                    # Baseline: no permutation
                    perm = list(range(dim))
                    
                perm_tensor = torch.tensor(perm, device=device, dtype=torch.long)
                
                # Apply perm to original masks
                permuted_A = is_outlier_A[:, perm_tensor]
                permuted_W = is_outlier_W[:, perm_tensor]
                grouped_A = permuted_A.view(N_tokens, n_groups, 16)
                grouped_W = permuted_W.view(M_out, n_groups, 16)
                outliers_per_group_A = grouped_A.sum(dim=-1).flatten()
                outliers_per_group_W = grouped_W.sum(dim=-1).flatten()
                
                # Compute Output Loss (MSE)
                with torch.no_grad():
                    # FP16 outputs
                    Y_fp16 = X @ W.T
                    
                    # Apply Permutation mathematically
                    X_perm = X[:, perm_tensor]
                    W_perm = W[:, perm_tensor]
                    
                    # Quantize with authentic NVFP4
                    X_q = apply_real_nvfp4_quant(X_perm, group_size=16)
                    W_q = apply_real_nvfp4_quant(W_perm, group_size=16)
                    
                    # Quantized outputs
                    Y_q = X_q @ W_q.T
                    mse_loss = torch.nn.functional.mse_loss(Y_fp16, Y_q).item()
                    
                    # Only W quantized
                    Y_q_W = X_perm @ W_q.T
                    mse_loss_W = torch.nn.functional.mse_loss(Y_fp16, Y_q_W).item()
                    
                    # Only A quantized
                    Y_q_A = X_q @ W_perm.T
                    mse_loss_A = torch.nn.functional.mse_loss(Y_fp16, Y_q_A).item()
                
                # Save metrics
                metrics = {
                    "A_ge2": (outliers_per_group_A >= 2).float().mean().item() * 100,
                    "A_ge3": (outliers_per_group_A >= 3).float().mean().item() * 100,
                    "W_ge2": (outliers_per_group_W >= 2).float().mean().item() * 100,
                    "W_ge3": (outliers_per_group_W >= 3).float().mean().item() * 100,
                    "MSE": mse_loss,
                    "MSE_W": mse_loss_W,
                    "MSE_A": mse_loss_A
                }
                stats[block_idx][name][strat_name] = metrics
                
                del X, threshold_A, is_outlier_A, W, threshold_W, is_outlier_W
                del permuted_A, permuted_W, grouped_A, grouped_W, perm_tensor
                del X_perm, W_perm, X_q, W_q, Y_fp16, Y_q, Y_q_W, Y_q_A
                if strat_args.channel_resort in ["stagger", "stagger_swap"]:
                    del joint_profile, group_profiles, group_sizes
                if strat_args.channel_resort in ["kmeans_fp4", "kmeans_fp4_w"]:
                    del X_abs_T, group_maxes, group_sizes
                torch.cuda.empty_cache()
            
            del block_copy
            del act_caches
            gc.collect()
            torch.cuda.empty_cache()
            
        # 3. Finally, pass the FP16 inputs through the original FP16 block for the next layers
        print(f"Advancing FP16 state to next layer...")
        for i in range(len(input_args)):
            with torch.no_grad():
                args_on_dev = to_device(input_args[i], device)
                kwargs_on_dev = to_device(input_kwargs[i], device)
                out = block(*args_on_dev, **kwargs_on_dev)
                out_hidden = maybe_first(out).cpu()
                input_args[i] = (out_hidden,) + input_args[i][1:]
        block = block.cpu()
        torch.cuda.empty_cache()
                
    print("\nAll target layers analyzed successfully.")
    
    # Print formatted markdown table
    print("\n" + "="*140)
    print(f"{'Layer':<5} | {'Matrix':<7} | {'Strategy':<16} | {'A>=2 %':<7} | {'A>=3 %':<7} | {'W>=2 %':<7} | {'W>=3 %':<7} | {'MSE(Joint)':<17} | {'MSE(W-only)':<17} | {'MSE(A-only)':<17}")
    print("-" * 140)
    for l_idx in target_layers:
        if l_idx not in stats: continue
        for mat in ["qkv", "o", "gate_up", "down"]:
            if mat not in stats[l_idx]: continue
            
            # Baseline metrics
            base = stats[l_idx][mat]["Baseline"]
            print(f"L{l_idx:<4} | {mat:<7} | {'Baseline':<16} | {base['A_ge2']:>6.2f}% | {base['A_ge3']:>6.2f}% | {base['W_ge2']:>6.2f}% | {base['W_ge3']:>6.2f}% | {base['MSE']:.3e}        | {base['MSE_W']:.3e}        | {base['MSE_A']:.3e}        ")
            
            # Stagger metrics (diff vs baseline)
            for strat in strategies:
                if strat["name"] == "Baseline": continue
                if strat["name"] not in stats[l_idx][mat]: continue
                s = stats[l_idx][mat][strat["name"]]
                
                a2_d = s['A_ge2'] - base['A_ge2']
                a3_d = s['A_ge3'] - base['A_ge3']
                w2_d = s['W_ge2'] - base['W_ge2']
                w3_d = s['W_ge3'] - base['W_ge3']
                mse_pct = (s['MSE'] - base['MSE']) / (base['MSE'] + 1e-12) * 100
                mse_w_pct = (s['MSE_W'] - base['MSE_W']) / (base['MSE_W'] + 1e-12) * 100
                mse_a_pct = (s['MSE_A'] - base['MSE_A']) / (base['MSE_A'] + 1e-12) * 100
                
                print(f"{'':<5} | {'':<7} | {strat['name']:<16} | {a2_d:>+6.2f}% | {a3_d:>+6.2f}% | {w2_d:>+6.2f}% | {w3_d:>+6.2f}% | {s['MSE']:.3e} ({mse_pct:>+5.1f}%) | {s['MSE_W']:.3e} ({mse_w_pct:>+5.1f}%) | {s['MSE_A']:.3e} ({mse_a_pct:>+5.1f}%)")
            print("-" * 140)

if __name__ == "__main__":
    main()
