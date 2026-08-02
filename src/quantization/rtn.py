import gc
import re
import argparse
from typing import List

import torch
from transformers import AutoModelForCausalLM

from .qlinear import QLinear
from .quant_ops import pack_fp4_to_uint8, cast_scales_to_eXmY, ScalePrecision, FP8_E4M3_MAX, FP4_E2M1_MAX
from .quantizer import get_reciprocal

from ..utils.common_utils import clear_device_cache, to, maybe_first_element
from ..utils.model_utils import InputCollector, ForwardInterrupt, get_attention_layer, get_mlp_layer
from ..transforms.transforms import build_transform, get_transform_matrix


def rtn_quantization(
    model: AutoModelForCausalLM, 
    calibration_data: List[torch.Tensor],
    args: argparse.Namespace, 
    device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    print("RTN quantization...")
    orig_dtype = model.config.torch_dtype if args.dtype == "auto" else args.dtype
    act_offload_device = "cpu" if args.cpu_offload_activations else device
    # We need calibration to track global scales for E4M3 OR to track activation MSE OR for channel resorting
    need_calibration = (args.scale_precision == ScalePrecision.E4M3) or (args.show_act_mse and args.a_bits < 16) or getattr(args, "channel_resort", False)
    # State dict with quantized weights, scales and hadamards
    quantized_state_dict = {}
    non_quantized_state_dict = {}
    # Get transformer blocks
    blocks = model.model.layers
    # Define common transform kwargs
    transform_kwargs = dict(device=device, group_size=args.hadamard_group_size)
    # Init quantizer kwargs
    weight_quantizer_kwargs = None
    if args.w_bits < 16:
        weight_quantizer_kwargs = dict(
            bits=args.w_bits, 
            symmetric=True, 
            format=args.format,
            granularity=args.w_granularity,
            observer=args.w_observer, 
            group_size=args.w_group_size,
            scale_precision=args.scale_precision,
        )
    act_quantizer_kwargs = None
    if args.a_bits < 16:
        act_quantizer_kwargs = dict(
            bits=args.a_bits, 
            symmetric=True, 
            format=args.format,
            granularity=args.a_granularity,
            observer=args.a_observer, 
            group_size=args.a_group_size,
            scale_precision=args.scale_precision,
        )

    if need_calibration:
        blocks = model.model.layers
        blocks[0] = InputCollector(blocks[0], cpu_offload=args.cpu_offload_activations)
        if args.cpu_offload_modules:
            model.get_input_embeddings().to(device)
            blocks[0] = blocks[0].to(device)

        for sample in calibration_data:
            try:
                with torch.no_grad():
                    model(sample.to(device=device))
            except ForwardInterrupt:
                pass
            
        input_args = blocks[0].input_args
        input_kwargs = blocks[0].input_kwargs
        blocks[0] = blocks[0].module

        if args.cpu_offload_modules:
            model.get_input_embeddings().cpu()

    # Iterate over transformer blocks
    for block_idx, block in enumerate(blocks):
        print(f"Processing block {block_idx}...")
        if args.cpu_offload_modules:
            block.to(device)

        # === 0. Resonance-Aware Channel Reordering & Global Scale Lock ===
        resort_perms = {}
        do_channel_resort = getattr(args, "channel_resort", "none") != "none"
        do_channel_rescale = getattr(args, "channel_rescale", "none") != "none"
        do_lock_global_scale = args.scale_precision in ["e4m3", ScalePrecision.E4M3] and getattr(args, "lock_global_scale", False)
        
        if do_lock_global_scale:
            print(f"  [DEBUG] lock_global_scale is ACTIVE for block {block_idx}!")
            
        fp16_act_global_max = {}
        need_act_stats = do_channel_resort or do_channel_rescale or do_lock_global_scale
        if need_act_stats and need_calibration:
            act_caches = {}
            resort_hooks = []
            lock_hooks = []
            
            if do_lock_global_scale:
                def lock_hook_factory(layer_name):
                    def _hook(_, inp, out):
                        max_val = inp[0].detach().float().abs().max().item()
                        if layer_name not in fp16_act_global_max:
                            fp16_act_global_max[layer_name] = max_val
                        else:
                            fp16_act_global_max[layer_name] = max(fp16_act_global_max[layer_name], max_val)
                    return _hook
                
                for layer_name, layer in block.named_modules():
                    if isinstance(layer, torch.nn.Linear) or type(layer).__name__ == "QLinear":
                        lock_hooks.append(layer.register_forward_hook(lock_hook_factory(layer_name)))
            
            if do_channel_resort or do_channel_rescale:
                import numpy as np
                print(f"  Computing channel resort/scale stats (Resort: {args.channel_resort}, Rescale: {args.channel_rescale})...")
                def hook_factory(name):
                    def _hook(_, inp, out):
                        if name not in act_caches:
                            act_caches[name] = []
                        # Collect up to 16 sequences to estimate P95 without OOM
                        if len(act_caches[name]) < 16:
                            act_caches[name].append(inp[0].detach().cpu().float().abs().view(-1, inp[0].shape[-1]))
                    return _hook

                resort_hooks.append(block.self_attn.q_proj.register_forward_hook(hook_factory("qkv")))
                resort_hooks.append(block.self_attn.o_proj.register_forward_hook(hook_factory("o")))
                resort_hooks.append(block.mlp.gate_proj.register_forward_hook(hook_factory("gate_up")))
                resort_hooks.append(block.mlp.down_proj.register_forward_hook(hook_factory("down")))

            device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                ikw = inp_kwargs.copy()
                ikw["use_cache"] = False
                if "past_key_value" in ikw: ikw["past_key_value"] = None
                if "output_attentions" in ikw: ikw["output_attentions"] = False
                with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                    block(*to(inp_args, device=device), **to(ikw, device=device))
            
            for h in resort_hooks: h.remove()
            if do_lock_global_scale:
                for h in lock_hooks: h.remove()
            
            def get_combined_weight(block, name):
                if name == "qkv":
                    w = torch.cat([block.self_attn.q_proj.weight, block.self_attn.k_proj.weight, block.self_attn.v_proj.weight], dim=0)
                elif name == "o":
                    w = block.self_attn.o_proj.weight
                elif name == "gate_up":
                    w = torch.cat([block.mlp.gate_proj.weight, block.mlp.up_proj.weight], dim=0)
                elif name == "down":
                    w = block.mlp.down_proj.weight
                else:
                    return None
                return w.float()

            act_means = {}
            for name, caches in list(act_caches.items()):
                X = torch.cat(caches, dim=0) # (N_tokens, Dim)
                if args.channel_resort == "stagger":
                    import math
                    threshold_A = torch.quantile(X.abs().float(), 0.9375, dim=1, keepdim=True)
                    is_outlier_A = (X.abs() > threshold_A).float().to(device)
                    
                    W = get_combined_weight(block, name).to(device)
                    threshold_W = torch.quantile(W.abs(), 0.9375, dim=1, keepdim=True)
                    is_outlier_W = (W.abs() > threshold_W).float()
                    
                    N_tokens = X.shape[0]
                    M_out = W.shape[0]
                    if getattr(args, "stagger_lambda", "auto") == "auto":
                        lambda_W = math.sqrt(N_tokens / M_out) if M_out > 0 else 1.0
                    else:
                        lambda_W = float(args.stagger_lambda)
                        
                    joint_profile = torch.cat([is_outlier_A, is_outlier_W * lambda_W], dim=0)
                    ch_freq = joint_profile.mean(dim=0)
                    
                    act_means[name] = {"mask": joint_profile, "freq": ch_freq}
                elif args.channel_resort in ["P95", "minmax"]:
                    act_means[name] = torch.quantile(X.float(), 0.95, dim=0).to(device)
                elif args.channel_resort in ["channel_cluster", "kmeans_fp4", "kmeans_fp4_w", "kmeans_fp4_top3"]:
                    max_samples = 4096
                    if X.shape[0] > max_samples:
                        subset_idx = torch.linspace(0, X.shape[0] - 1, max_samples, dtype=torch.long, device=X.device)
                        X_sub = X[subset_idx].float().to(device)
                    else:
                        X_sub = X.float().to(device)
                    act_means[name] = X_sub
                elif do_channel_rescale and not do_channel_resort:
                    # GICS-only mode: still need X_sub for GICS
                    max_samples = 4096
                    if X.shape[0] > max_samples:
                        subset_idx = torch.linspace(0, X.shape[0] - 1, max_samples, dtype=torch.long, device=X.device)
                        X_sub = X[subset_idx].float().to(device)
                    else:
                        X_sub = X.float().to(device)
                    act_means[name] = X_sub
                else: # "mean" or "R_val"
                    act_means[name] = X.float().mean(dim=0).to(device)
                del X
                del act_caches[name]

            def compute_resonance_perm(act_mean_abs, group_size=16, target_R=0.2):
                N = act_mean_abs.shape[0]
                vals = act_mean_abs.cpu().numpy()
                sorted_pos = np.argsort(-vals)  # descending
                available = np.ones(N, dtype=bool)
                perm = []
                for _ in range(N // group_size):
                    avail_sorted = sorted_pos[available[sorted_pos]]
                    if len(avail_sorted) < group_size:
                        break
                    top1 = avail_sorted[0]
                    available[top1] = False
                    avail_sorted = sorted_pos[available[sorted_pos]]
                    target_val = target_R * vals[top1]
                    candidates = avail_sorted[vals[avail_sorted] >= target_val]
                    if len(candidates) > 0:
                        top2 = candidates[-1]
                        available[top2] = False
                        avail_sorted = sorted_pos[available[sorted_pos]]
                        fill = avail_sorted[-(group_size - 2):]
                        available[fill] = False
                        perm.extend([top1, top2] + fill.tolist())
                    else:
                        fill = avail_sorted[-(group_size - 1):]
                        available[fill] = False
                        perm.extend([top1] + fill.tolist())
                return torch.tensor(perm, device=act_mean_abs.device, dtype=torch.long)
                
            def compute_gridsort_perm(act_stat, group_size=16):
                FP4_GRID = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=act_stat.device)
                N = act_stat.shape[0]
                vals = act_stat.float()
                available = torch.ones(N, dtype=torch.bool, device=act_stat.device)
                perm = []
                
                for _ in range(N // group_size):
                    avail_mask = available.clone()
                    avail_vals = vals.clone()
                    avail_vals[~avail_mask] = -1
                    anchor = avail_vals.argmax().item()
                    
                    scale = (vals[anchor] / 6.0) + 1e-12
                    available[anchor] = False
                    group = [anchor]
                    
                    remaining_indices = available.nonzero(as_tuple=True)[0]
                    if len(remaining_indices) < group_size - 1:
                        if len(remaining_indices) > 0:
                            group.extend(remaining_indices.tolist())
                            available[remaining_indices] = False
                        perm.extend(group)
                        break
                        
                    remaining_vals = vals[remaining_indices]
                    
                    normalized = remaining_vals / scale
                    distances = (normalized.unsqueeze(1) - FP4_GRID.unsqueeze(0)).abs()
                    min_grid_errors = distances.min(dim=1).values
                    
                    gas_scores = -min_grid_errors + remaining_vals * 1e-6
                    
                    n_needed = group_size - 1
                    _, top_indices = gas_scores.topk(n_needed)
                    selected = remaining_indices[top_indices].tolist()
                    
                    for s in selected:
                        available[s] = False
                    group.extend(selected)
                    perm.extend(group)
                
                return torch.tensor(perm, device=act_stat.device, dtype=torch.long)
                
            def compute_truncated_minmax_perm(act_stat, group_size=16, outlier_ratio=0.01):
                N = act_stat.shape[0]
                sorted_idx = torch.argsort(act_stat, descending=True)
                
                n_outliers = int(N * outlier_ratio)
                head, tail = 0, N - 1
                perm = []
                
                for _ in range(n_outliers):
                    if head >= tail: break
                    group = [sorted_idx[head].item()]
                    head += 1
                    for _ in range(group_size - 1):
                        if head > tail: break
                        group.append(sorted_idx[tail].item())
                        tail -= 1
                    perm.extend(group)
                    
                remaining = [sorted_idx[i].item() for i in range(head, tail + 1)]
                remaining = sorted(remaining)
                
                perm.extend(remaining)
                return torch.tensor(perm, device=act_stat.device, dtype=torch.long)

            def get_combined_weight_norm(block, name):
                if name == "qkv":
                    w = torch.cat([block.self_attn.q_proj.weight, block.self_attn.k_proj.weight, block.self_attn.v_proj.weight], dim=0)
                elif name == "o":
                    w = block.self_attn.o_proj.weight
                elif name == "gate_up":
                    w = torch.cat([block.mlp.gate_proj.weight, block.mlp.up_proj.weight], dim=0)
                elif name == "down":
                    w = block.mlp.down_proj.weight
                else:
                    return None
                return w.float().norm(p=2, dim=0).to(device)

            def compute_nvfp4_scale_first_perm(act_stat, w_norm, group_size=16):
                w_norm = w_norm.to(act_stat.device)
                N = act_stat.shape[0]
                n_groups = max(1, N // group_size)
                score = act_stat * w_norm
                score_sorted_idx = torch.argsort(score, descending=True)
                anchors = score_sorted_idx[:n_groups].tolist()
                groups = [[a] for a in anchors]
                group_lens = torch.ones(n_groups, dtype=torch.long, device=act_stat.device)
                FP4_GRID = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=act_stat.device)
                remaining_mask = torch.ones(N, dtype=torch.bool, device=act_stat.device)
                remaining_mask[anchors] = False
                remaining_idx = remaining_mask.nonzero(as_tuple=True)[0]
                remaining_score = score[remaining_idx]
                rem_sorted_idx = remaining_idx[torch.argsort(remaining_score, descending=True)]
                group_scales = torch.tensor([act_stat[a] / 6.0 + 1e-12 for a in anchors], device=act_stat.device)
                group_w_sums = torch.tensor([w_norm[a] for a in anchors], device=act_stat.device)
                for ch_idx in rem_sorted_idx.tolist():
                    val = act_stat[ch_idx]
                    w = w_norm[ch_idx]
                    new_scales = torch.maximum(group_scales, torch.tensor(val / 6.0 + 1e-12, device=act_stat.device))
                    normalized = val / new_scales
                    distances = (normalized.unsqueeze(1) - FP4_GRID.unsqueeze(0)).abs()
                    min_errs = distances.min(dim=1).values * new_scales
                    damage = min_errs * w
                    scale_diff = new_scales - group_scales
                    damage += scale_diff * group_w_sums
                    damage[group_lens >= group_size] = float('inf')
                    best_g_idx = damage.argmin().item()
                    groups[best_g_idx].append(ch_idx)
                    group_lens[best_g_idx] += 1
                    group_scales[best_g_idx] = new_scales[best_g_idx]
                    group_w_sums[best_g_idx] += w
                perm = []
                for g in groups:
                    perm.extend(g)
                return torch.tensor(perm, device=act_stat.device, dtype=torch.long)

            def compute_mxfp4_vip_l1_perm(act_stat, w_norm, block_size=128):
                w_norm = w_norm.to(act_stat.device)
                N = act_stat.shape[0]
                n_blocks = max(1, N // block_size)
                score = act_stat * w_norm
                score_sorted_idx = torch.argsort(score, descending=True)
                vips = score_sorted_idx[:n_blocks].tolist()
                blocks = [[vip] for vip in vips]
                block_lens = torch.ones(n_blocks, dtype=torch.long, device=act_stat.device)
                block_l1 = torch.tensor([act_stat[vip] for vip in vips], device=act_stat.device)
                remaining_mask = torch.ones(N, dtype=torch.bool, device=act_stat.device)
                remaining_mask[vips] = False
                remaining_idx = remaining_mask.nonzero(as_tuple=True)[0]
                remaining_p95 = act_stat[remaining_idx]
                rem_sorted_idx = remaining_idx[torch.argsort(remaining_p95, descending=True)]
                for ch_idx in rem_sorted_idx.tolist():
                    val = act_stat[ch_idx]
                    valid_l1 = block_l1.clone()
                    valid_l1[block_lens >= block_size] = float('inf')
                    min_block_idx = valid_l1.argmin().item()
                    blocks[min_block_idx].append(ch_idx)
                    block_lens[min_block_idx] += 1
                    block_l1[min_block_idx] += val
                perm = []
                for b in blocks:
                    perm.extend(b)
                return torch.tensor(perm, device=act_stat.device, dtype=torch.long)

            def compute_co_occurrence_staggered_perm(is_outlier_mask, ch_freq, group_size=16):
                sorted_channels_list = torch.argsort(ch_freq, descending=True).tolist()
                n_tokens, dim = is_outlier_mask.shape
                n_groups = max(1, dim // group_size)
                
                opt_groups = [[] for _ in range(n_groups)]
                group_profiles = torch.zeros((n_groups, n_tokens), device=is_outlier_mask.device, dtype=torch.float32)
                group_sizes = torch.zeros(n_groups, device=is_outlier_mask.device, dtype=torch.long)
                
                for ch in sorted_channels_list:
                    c_profile = is_outlier_mask[:, ch].float()
                    penalties = torch.mv(group_profiles, c_profile)
                    penalties[group_sizes >= group_size] = float('inf')
                    
                    best_group = penalties.argmin().item()
                    opt_groups[best_group].append(ch)
                    group_profiles[best_group] += c_profile
                    group_sizes[best_group] += 1
                    
                perm = []
                for g in opt_groups: perm.extend(g)
                return torch.tensor(perm, device=is_outlier_mask.device, dtype=torch.long)

            def compute_kmeans_fp4_perm(X_abs_T, group_size=16, channel_weights=None, act_alpha=0.0):
                dim = X_abs_T.shape[0]
                n_groups = dim // group_size
                device = X_abs_T.device
                grid_mults = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device, dtype=torch.float32)
                
                def compute_loss_distances(all_ch_abs, ref_ch_abs, chunk_size=256, c_weights=None):
                    ref = ref_ch_abs.unsqueeze(0)
                    n_ch = all_ch_abs.shape[0]
                    losses = torch.zeros(n_ch, device=device, dtype=torch.float32)
                    for i in range(0, n_ch, chunk_size):
                        chunk = all_ch_abs[i:i+chunk_size]
                        p_max = torch.maximum(chunk, ref)
                        p_min = torch.minimum(chunk, ref)
                        scale = (p_max / 6.0).clamp(min=1e-10)
                        grid = scale.unsqueeze(-1) * grid_mults
                        diff = (p_min.unsqueeze(-1) - grid).abs()
                        min_q = grid.gather(-1, diff.argmin(-1, keepdim=True)).squeeze(-1)
                        mse = (p_min - min_q) ** 2
                        if act_alpha > 0.0:
                            act_weight = p_min.abs() ** act_alpha
                            mse = mse * act_weight
                        mse = mse.sum(dim=1)
                        if c_weights is not None:
                            mse = mse * c_weights[i:i+chunk_size]
                        losses[i:i+chunk_size] = mse
                    return losses

                if act_alpha > 0.0:
                    ch_sums = (X_abs_T ** (1.0 + act_alpha)).sum(dim=1)
                else:
                    ch_sums = X_abs_T.sum(dim=1)
                
                if channel_weights is not None:
                    ch_sums = ch_sums * channel_weights
                accumulated_dist = torch.zeros(dim, device=device, dtype=torch.float32)
                seed_order = []
                first_seed = ch_sums.argmax().item()
                seed_order.append(first_seed)
                accumulated_dist[first_seed] = -float('inf')
                dists = compute_loss_distances(X_abs_T, X_abs_T[first_seed], c_weights=channel_weights)
                accumulated_dist += dists
                accumulated_dist[first_seed] = -float('inf')
                
                for k in range(1, n_groups):
                    new_seed = accumulated_dist.argmax().item()
                    seed_order.append(new_seed)
                    accumulated_dist[new_seed] = -float('inf')
                    dists = compute_loss_distances(X_abs_T, X_abs_T[new_seed], c_weights=channel_weights)
                    accumulated_dist += dists

                opt_groups = [[s] for s in seed_order]
                group_maxes = X_abs_T[seed_order].clone()
                is_assigned = torch.zeros(dim, dtype=torch.bool, device=device)
                group_sizes = torch.ones(n_groups, device=device, dtype=torch.long)
                seed_set = set(seed_order)
                remaining = [c for c in range(dim) if c not in seed_set]
                remaining_sums = ch_sums[remaining]
                sorted_order = torch.argsort(remaining_sums, descending=True)
                remaining_sorted = [remaining[i] for i in sorted_order.tolist()]

                for c in remaining_sorted:
                    c_abs = X_abs_T[c]
                    losses = torch.zeros(n_groups, device=device, dtype=torch.float32)
                    ref = c_abs.unsqueeze(0)
                    for i in range(0, n_groups, 256):
                        g_chunk = group_maxes[i:i+256]
                        p_max = torch.maximum(ref, g_chunk)
                        p_min = torch.minimum(ref, g_chunk)
                        scale = (p_max / 6.0).clamp(min=1e-10)
                        grid = scale.unsqueeze(-1) * grid_mults
                        diff = (p_min.unsqueeze(-1) - grid).abs()
                        min_q = grid.gather(-1, diff.argmin(-1, keepdim=True)).squeeze(-1)
                        losses[i:i+256] = ((p_min - min_q) ** 2).sum(dim=1)
                    losses[group_sizes >= group_size] = float('inf')
                    best_g = losses.argmin().item()
                    opt_groups[best_g].append(c)
                    group_maxes[best_g] = torch.maximum(group_maxes[best_g], c_abs)
                    group_sizes[best_g] += 1
                
                perm = []
                for g in opt_groups: perm.extend(g)
                return torch.tensor(perm, device=device, dtype=torch.long)

            for name, mean_val in act_means.items():
                w_norm = get_combined_weight_norm(block, name)
                
                if args.channel_resort == "stagger":
                    mask = mean_val["mask"]
                    freq = mean_val["freq"]
                    N = freq.shape[0]
                    if w_norm is None:
                        w_norm = torch.ones(N, device=freq.device)
                elif args.channel_resort in ["channel_cluster", "kmeans_fp4", "kmeans_fp4_w", "kmeans_fp4_top3"]:
                    N = mean_val.shape[1]
                    if w_norm is None:
                        w_norm = torch.ones(N, device=mean_val.device)
                else:
                    abs_vals = mean_val.abs()
                    N = abs_vals.shape[0]
                    if w_norm is None:
                        w_norm = torch.ones_like(abs_vals)
                
                quant_group_size = args.a_group_size if args.a_group_size else 16
                
                if args.channel_resort in ["mean", "P95"]:
                    # GridSort Strategy
                    if "o" in name:
                        head_dim = model.config.hidden_size // model.config.num_attention_heads
                        h_group_size = head_dim
                    else:
                        h_group_size = -1 # Global matrix for other projections
                        
                    if h_group_size <= 0:
                        p = compute_gridsort_perm(abs_vals, group_size=quant_group_size)
                    else:
                        p = torch.zeros(N, device=abs_vals.device, dtype=torch.long)
                        for i in range(0, N, h_group_size):
                            block_vals = abs_vals[i : i + h_group_size]
                            block_perm = compute_gridsort_perm(block_vals, group_size=quant_group_size)
                            p[i : i + h_group_size] = block_perm + i
                            
                    resort_perms[name] = p
                    print(f"    [{name:8}] GridSort completed.")
                elif args.channel_resort == "minmax":
                    # Dual-Track Pairing Strategy for NVFP4 and MXFP4
                    if args.format == "mxfp":
                        gs = args.hadamard_group_size if args.hadamard_group_size > 0 else 128
                        if "o" in name:
                            head_dim = model.config.hidden_size // model.config.num_attention_heads
                            h_group_size = head_dim
                        else:
                            h_group_size = -1
                            
                        if h_group_size <= 0:
                            p = compute_mxfp4_vip_l1_perm(abs_vals, w_norm, block_size=gs)
                        else:
                            p = torch.zeros(N, device=abs_vals.device, dtype=torch.long)
                            for i in range(0, N, h_group_size):
                                block_vals = abs_vals[i : i + h_group_size]
                                block_w_norm = w_norm[i : i + h_group_size]
                                block_gs = min(gs, h_group_size)
                                block_perm = compute_mxfp4_vip_l1_perm(block_vals, block_w_norm, block_size=block_gs)
                                p[i : i + h_group_size] = block_perm + i
                        print(f"    [{name:8}] VIP+L1 平滑 (gs={gs}) completed.")
                    else:
                        gs = quant_group_size
                        if "o" in name:
                            head_dim = model.config.hidden_size // model.config.num_attention_heads
                            h_group_size = head_dim
                        else:
                            h_group_size = -1
                            
                        if h_group_size <= 0:
                            p = compute_nvfp4_scale_first_perm(abs_vals, w_norm, group_size=gs)
                        else:
                            p = torch.zeros(N, device=abs_vals.device, dtype=torch.long)
                            for i in range(0, N, h_group_size):
                                block_vals = abs_vals[i : i + h_group_size]
                                block_w_norm = w_norm[i : i + h_group_size]
                                block_gs = min(gs, h_group_size)
                                block_perm = compute_nvfp4_scale_first_perm(block_vals, block_w_norm, group_size=block_gs)
                                p[i : i + h_group_size] = block_perm + i
                        print(f"    [{name:8}] Scale-First Greedy Fill (gs={gs}) completed.")
                    resort_perms[name] = p
                elif args.channel_resort == "stagger":
                    if "o" in name:
                        head_dim = model.config.hidden_size // model.config.num_attention_heads
                        h_group_size = head_dim
                    else:
                        h_group_size = -1
                        
                    if h_group_size <= 0:
                        p = compute_co_occurrence_staggered_perm(mask, freq, group_size=quant_group_size)
                    else:
                        p = torch.zeros(N, device=freq.device, dtype=torch.long)
                        for i in range(0, N, h_group_size):
                            block_mask = mask[:, i : i + h_group_size]
                            block_freq = freq[i : i + h_group_size]
                            block_gs = min(quant_group_size, h_group_size)
                            block_perm = compute_co_occurrence_staggered_perm(block_mask, block_freq, group_size=block_gs)
                            p[i : i + h_group_size] = block_perm + i
                    resort_perms[name] = p
                    print(f"    [{name:8}] Staggered Reordering (P93.75) completed.")
                elif args.channel_resort in ["channel_cluster", "kmeans_fp4", "kmeans_fp4_w", "kmeans_fp4_top3"]:
                    W = get_combined_weight(block, name).to(device)
                    if args.channel_resort in ["channel_cluster", "kmeans_fp4", "kmeans_fp4_top3"]:
                        target_mat = mean_val # X_sub
                    else:
                        target_mat = W
                        # Subsample target_mat (W) if using kmeans_fp4_w
                        max_samples = 4096
                        if target_mat.shape[0] > max_samples:
                            subset_idx = torch.linspace(0, target_mat.shape[0] - 1, max_samples, dtype=torch.long, device=target_mat.device)
                            target_mat = target_mat[subset_idx]
                            
                    X_abs_T = target_mat.abs().T.float()
                    
                    alpha = getattr(args, "kmeans_alpha", 0.0)
                    if args.channel_resort in ["channel_cluster", "kmeans_fp4", "kmeans_fp4_top3"] and alpha > 0.0:
                        W_norm = torch.norm(W, p=2, dim=0) # Shape: (In_Features,)
                        X_norm = torch.norm(mean_val, p=2, dim=0) # Act Norm
                        # Use W_norm * X_norm to align with true output contribution
                        # channel_weights = (W_norm * X_norm) ** alpha
                        channel_weights = (W_norm) ** alpha
                    else:
                        channel_weights = None
                        
                    act_alpha = getattr(args, "kmeans_act_alpha", 0.0)
                    
                    if name == "down" and getattr(args, "kmeans_block_size", 0) != -1:
                        if getattr(args, "kmeans_block_size", 0) > 0:
                            h_group_size = args.kmeans_block_size
                        else:
                            h_group_size = model.config.hidden_size // model.config.num_attention_heads
                    elif name == "o":
                        head_dim = model.config.hidden_size // model.config.num_attention_heads
                        h_group_size = head_dim
                    else:
                        h_group_size = -1
                        
                    if h_group_size <= 0:
                        p = compute_kmeans_fp4_perm(X_abs_T, group_size=quant_group_size, channel_weights=channel_weights, act_alpha=act_alpha)
                    else:
                        p = torch.zeros(N, device=X_abs_T.device, dtype=torch.long)
                        for i in range(0, N, h_group_size):
                            block_X_abs_T = X_abs_T[i : i + h_group_size]
                            block_channel_weights = channel_weights[i : i + h_group_size] if channel_weights is not None else None
                            block_gs = min(quant_group_size, h_group_size)
                            block_perm = compute_kmeans_fp4_perm(block_X_abs_T, group_size=block_gs, channel_weights=block_channel_weights, act_alpha=act_alpha)
                            p[i : i + h_group_size] = block_perm + i
                    print(f"    [{name:8}] KMeans FP4 grouping (target={args.channel_resort}, gs={quant_group_size}) completed.")
                    resort_perms[name] = p
                else:
                    # Old Resonance R_val Strategy
                    h_group_size = args.hadamard_group_size
                    if "o" in name:
                        head_dim = model.config.hidden_size // model.config.num_attention_heads
                        if h_group_size <= 0 or h_group_size > head_dim:
                            h_group_size = head_dim
                            
                    if h_group_size <= 0:
                        p = compute_resonance_perm(abs_vals, group_size=16)
                    else:
                        p = torch.zeros(N, device=abs_vals.device, dtype=torch.long)
                        for i in range(0, N, h_group_size):
                            block_vals = abs_vals[i : i + h_group_size]
                            block_perm = compute_resonance_perm(block_vals, group_size=16)
                            p[i : i + h_group_size] = block_perm + i
                    resort_perms[name] = p
                    
                    grouped = abs_vals[p].view(-1, 16)
                    top2_vals = torch.topk(grouped, k=2, dim=-1).values
                    R_vals = top2_vals[:, 1] / (top2_vals[:, 0] + 1e-9)
                    print(f"    [{name:8}] Mean R after resort: {R_vals.mean():.4f}")
                    
                if args.channel_rescale == "gics":
                    from .awq import run_channel_rescale
                    target_mat = mean_val # X_sub
                    W = get_combined_weight(block, name).to(device)
                    # Use channel permutation if available, otherwise use identity order
                    if name in resort_perms:
                        p_gics = resort_perms[name]
                        X_perm = target_mat[:, p_gics]
                        W_perm = W[:, p_gics]
                    else:
                        X_perm = target_mat
                        W_perm = W
                    
                    top_k = getattr(args, "gics_top_k", 5)
                    num_rounds = getattr(args, "channel_rescale_rounds", 3)
                    print(f"    [{name:8}] Running Coordinate Descent Channel Scale Search (top_k={top_k}, num_rounds={num_rounds})...")
                    S_top3 = run_channel_rescale(X_perm, W_perm, group_size=quant_group_size, top_k=top_k, num_rounds=num_rounds, weight_mse_ratio=3.0)
                    
                    if not hasattr(block, "gics_scales"):
                        block.gics_scales = {}
                    block.gics_scales[name] = S_top3

        # === 0.5 Run GICS independently when channel_resort is off ===
        if do_channel_rescale and not do_channel_resort:
            from .awq import run_channel_rescale
            quant_group_size = args.a_group_size if args.a_group_size else 16
            for name, mean_val in act_means.items():
                W = get_combined_weight(block, name).to(device)
                X_perm = mean_val  # no permutation, identity order
                W_perm = W
                
                top_k = getattr(args, "gics_top_k", 5)
                num_rounds = getattr(args, "channel_rescale_rounds", 3)
                print(f"    [{name:8}] Running Channel Rescale (standalone) Coordinate Descent Channel Scale Search (top_k={top_k}, num_rounds={num_rounds})...")
                S_gics = run_channel_rescale(X_perm, W_perm, group_size=quant_group_size, top_k=top_k, num_rounds=num_rounds, weight_mse_ratio=3.0)
                
                if not hasattr(block, "gics_scales"):
                    block.gics_scales = {}
                block.gics_scales[name] = S_gics

        # 1. Init transforms
        qkv_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        o_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        gate_up_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        down_in_transform = build_transform(args.transform_class, size=model.config.intermediate_size, **transform_kwargs)     

        has_resort_or_rescale = resort_perms or (hasattr(block, "gics_scales") and block.gics_scales)
        if has_resort_or_rescale:
            from ..transforms.transforms import CompositeTransform, PermutationTransform, DiagonalScaleTransform
            
            def build_composite_with_gics(name, base_transform):
                transforms_list = []
                if name in resort_perms:
                    transforms_list.append(PermutationTransform(resort_perms[name]))
                if hasattr(block, "gics_scales") and name in block.gics_scales:
                    transforms_list.append(DiagonalScaleTransform(block.gics_scales[name]))
                transforms_list.append(base_transform)
                return CompositeTransform(transforms_list) if len(transforms_list) > 1 else base_transform

            if "qkv" in resort_perms or (hasattr(block, "gics_scales") and "qkv" in block.gics_scales):
                qkv_in_transform = build_composite_with_gics("qkv", qkv_in_transform)
            if "gate_up" in resort_perms or (hasattr(block, "gics_scales") and "gate_up" in block.gics_scales):
                gate_up_in_transform = build_composite_with_gics("gate_up", gate_up_in_transform)
            # O: in-place permutation only for MHA (skip GQA to avoid cross-head corruption)
            if "o" in resort_perms:
                o_perm = resort_perms["o"]
                if hasattr(block, "gics_scales") and "o" in block.gics_scales:
                    # In-place DiagonalScaleTransform applies to W via inverse, but wait:
                    # DiagonalScaleTransform modifies activation as X * S, so it modifies Weight as W * S^-1
                    # So we should apply this explicitly here if we do in-place for O.
                    # But wait, o_in_transform is NOT fully in-place for activations! o_in_transform is PASSED to get_attention_layer
                    # Actually, for "o" and "down", if we do in-place weight permutation, the transform is STILL passed to quantized_attn!
                    # Wait! In original code, o_in_transform is untouched by PermutationTransform because we do in-place weight data moving.
                    # But if we have gics_scales, it MUST be applied to activations dynamically!
                    # So we MUST add DiagonalScaleTransform to o_in_transform!
                    o_in_transform = build_composite_with_gics("o", o_in_transform)
                    # For weights, we only do the permutation in-place, the scale will be handled by o_in_transform mathematically via fix_parametrization
                if block.self_attn.v_proj.weight.shape[0] == block.self_attn.o_proj.weight.shape[1]:
                    block.self_attn.o_proj.weight.data = block.self_attn.o_proj.weight.data[:, o_perm]
                    block.self_attn.v_proj.weight.data = block.self_attn.v_proj.weight.data[o_perm, :]
            # Down: in-place weight permutation
            if "down" in resort_perms:
                down_perm = resort_perms["down"]
                if hasattr(block, "gics_scales") and "down" in block.gics_scales:
                    down_in_transform = build_composite_with_gics("down", down_in_transform)
                block.mlp.down_proj.weight.data = block.mlp.down_proj.weight.data[:, down_perm]
                block.mlp.gate_proj.weight.data = block.mlp.gate_proj.weight.data[down_perm, :]
                block.mlp.up_proj.weight.data = block.mlp.up_proj.weight.data[down_perm, :]

        # NOTE: Do NOT pre-rotate weights here. fix_parametrization() below handles it.
        # Pre-rotating would cause double rotation since H(H(W)) = W.

        # 2. Replace blocks with quantized versions
        quantized_attn = get_attention_layer(model.config)(
            model.config,
            layer_idx=block_idx,
            weight_quantizer_kwargs=weight_quantizer_kwargs,
            act_quantizer_kwargs=act_quantizer_kwargs,
            qkv_in_transform=qkv_in_transform,
            o_in_transform=o_in_transform
        ).to(device=device, dtype=orig_dtype)
        quantized_mlp = get_mlp_layer(model.config)(
            model.config,
            weight_quantizer_kwargs=weight_quantizer_kwargs,
            act_quantizer_kwargs=act_quantizer_kwargs,
            gate_up_in_transform=gate_up_in_transform,
            down_in_transform=down_in_transform
        ).to(device=device, dtype=orig_dtype)

        quantized_attn.load_state_dict(block.self_attn.state_dict(), strict=False)
        quantized_mlp.load_state_dict(block.mlp.state_dict(), strict=False)
        block.self_attn = quantized_attn
        block.mlp = quantized_mlp
        block = block.to(device=device, dtype=orig_dtype)  

        # 3. Global scale collection for NVFP
        if args.scale_precision in ["e4m3", ScalePrecision.E4M3]:
            for layer_name, layer in block.named_modules():
                if isinstance(layer, QLinear):
                    # Weights are unrotated at this point, so this naturally captures the unrotated weight global scale.
                    if layer.weight_quantizer is not None:
                        # Compute global_scale directly from max abs (avoids expensive get_quantization_params MSE search)
                        max_val = layer.weight.abs().max().to(torch.float32).view(1)
                        layer.weight_quantizer.global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(max_val)
                        layer.weight_quantizer._track_global_scale = False
                    
                    if layer.act_quantizer is not None:
                        if do_lock_global_scale and layer_name in fp16_act_global_max:
                            act_max_val = torch.tensor([fp16_act_global_max[layer_name]], dtype=torch.float32)
                            act_global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(act_max_val)
                            layer.act_quantizer.global_scale = act_global_scale.to(layer.weight.device)
                            layer.act_quantizer._track_global_scale = False
                            print(f"  [DEBUG] Locked {layer_name} act_scale to {layer.act_quantizer.global_scale.item():.4f} (max_val={fp16_act_global_max[layer_name]:.4f})")

            if args.fuse_global_scale:
                if getattr(quantized_attn.q_proj, "weight_quantizer", None) is not None:
                    qkv_global_scale = min(
                        quantized_attn.q_proj.weight_quantizer.global_scale,
                        quantized_attn.k_proj.weight_quantizer.global_scale,
                        quantized_attn.v_proj.weight_quantizer.global_scale,
                    )
                    quantized_attn.q_proj.weight_quantizer.global_scale = qkv_global_scale
                    quantized_attn.k_proj.weight_quantizer.global_scale = qkv_global_scale
                    quantized_attn.v_proj.weight_quantizer.global_scale = qkv_global_scale
                
                if getattr(quantized_mlp.gate_proj, "weight_quantizer", None) is not None:
                    # gate_up fusion
                    gate_up_global_scale = min(
                        quantized_mlp.gate_proj.weight_quantizer.global_scale,
                        quantized_mlp.up_proj.weight_quantizer.global_scale
                    )
                    quantized_mlp.gate_proj.weight_quantizer.global_scale = gate_up_global_scale
                    quantized_mlp.up_proj.weight_quantizer.global_scale = gate_up_global_scale

        # For calibration: keep _train_mode=True so transforms (e.g. PermutationTransform from 
        # channel_resort) are correctly applied to weights during forward pass.
        # Temporarily remove weight_quantizer to skip W4 quantization during calibration,
        # so activation scales are calibrated on FP16 (transformed) weights.
        # Without this, F.linear(perm(x), W_unrotated) produces wrong activation distributions
        # when channel_resort is active, causing severe accuracy degradation.
        saved_weight_quantizers = {}
        for layer_name, layer in block.named_modules():
            if isinstance(layer, QLinear):
                saved_weight_quantizers[layer_name] = layer.weight_quantizer
                layer.weight_quantizer = None

        # Calibrate activations (if needed)
        if need_calibration:
            device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                    block(*to(inp_args, device=device), **to(inp_kwargs, device=device))

        # Restore weight quantizers and disable train_mode for fix_parametrization
        for layer_name, layer in block.named_modules():
            if isinstance(layer, QLinear):
                layer.weight_quantizer = saved_weight_quantizers.get(layer_name)
                layer._train_mode = False

        
        if args.export_quantized_model:
            for layer_name, layer in block.named_modules():
                if isinstance(layer, QLinear):
                    with torch.no_grad():
                        # Weights are already rotated by fix_parametrization, use directly
                        weight = layer.weight
                        scales, zeros = layer.weight_quantizer.get_quantization_params(weight)
                        qweight = layer.weight_quantizer.quantize(weight, scales, zeros)

                    weight_global_scale = layer.weight_quantizer.global_scale.to(scales.device)
                    act_global_scale = layer.act_quantizer.global_scale

                    # Stop tracking global scale
                    layer.weight_quantizer._track_global_scale = False
                    layer.act_quantizer._track_global_scale = False

                    transform_matrix = get_transform_matrix(args.transform_class, args.hadamard_group_size, device, orig_dtype).cpu()

                    if args.export_quantized_model == "realquant":
                        quantized_state_dict[f"model.layers.{block_idx}.{layer_name}"] = {
                            "qweight": pack_fp4_to_uint8(qweight).cpu(),
                            "scales": cast_scales_to_eXmY(scales * weight_global_scale, args.scale_precision).cpu(),
                            "forward_hadamard_matrix": transform_matrix,
                            "backward_hadamard_matrix": transform_matrix.clone(),
                            "weight_global_scale": weight_global_scale.clone(),
                            "act_global_scale": act_global_scale.clone()
                        }
                    # pseudoquant
                    else:
                        # Get dequantized weight
                        dqweight = layer.weight_quantizer(weight, scales, zeros)
                        quantized_state_dict[f"model.layers.{block_idx}.{layer_name}"] = {
                            "dqweight": dqweight.cpu(),
                            "forward_hadamard_matrix": transform_matrix,
                            "backward_hadamard_matrix": transform_matrix.clone(),
                            "weight_global_scale": weight_global_scale.clone(),
                            "act_global_scale": act_global_scale.clone()
                        }  
        

        # Offload calibration inputs to CPU to free GPU memory for fix_parametrization
        for i in range(len(input_args)):
            if isinstance(input_args[i], torch.Tensor):
                input_args[i] = input_args[i].cpu()
        for kw in input_kwargs:
            for k in kw:
                if isinstance(kw[k], torch.Tensor):
                    kw[k] = kw[k].cpu()
        torch.cuda.empty_cache()

        # 3. Fix model parametrization
        quantized_attn.fix_parametrization()
        quantized_mlp.fix_parametrization()
        # 4. Fix transforms and remove parametrizations
        qkv_in_transform.remove_parametrizations()
        o_in_transform.remove_parametrizations()
        gate_up_in_transform.remove_parametrizations()
        down_in_transform.remove_parametrizations() 

        if need_calibration:
            # Enable activation MSE tracking
            if args.show_act_mse:
                for layer_name, layer in block.named_modules():
                    if isinstance(layer, QLinear):
                        layer.track_act_mse = True

            device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                    out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                out = maybe_first_element(out).to(act_offload_device)
                # change only first input argument
                if len(inp_args) > 0:
                    inp_args[0].data = out
                elif "hidden_states" in inp_kwargs:
                    inp_kwargs["hidden_states"] = out
                else:
                    raise ValueError("Unsupported block input format.")

            # Print activation MSE
            if args.show_act_mse:
                for layer_name, layer in block.named_modules():
                    if isinstance(layer, QLinear) and hasattr(layer, 'act_mse_sum') and layer.act_mse_count > 0:
                        act_rel_mse = layer.act_mse_sum / layer.act_mse_count
                        print(f"[{layer_name:16}]: Activation Rel MSE: {act_rel_mse:.2e}")
                        layer.track_act_mse = False

        if args.cpu_offload_modules:
            block.cpu()

        clear_device_cache(garbage_collection=True)

    # Free the large input caches before evaluation
    del input_args
    del input_kwargs
    clear_device_cache(garbage_collection=True)

    return quantized_state_dict, non_quantized_state_dict
