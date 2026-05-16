import gc
import re
import argparse
from typing import List

import torch
from transformers import AutoModelForCausalLM

from .qlinear import QLinear
from .quant_ops import pack_fp4_to_uint8, cast_scales_to_eXmY, ScalePrecision

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

        # === 0. Resonance-Aware Channel Reordering ===
        resort_perms = {}
        if getattr(args, "channel_resort", "none") != "none" and need_calibration:
            import numpy as np
            print(f"  Computing resonance-aware channel permutations (Strategy: {args.channel_resort})...")
            act_caches = {}
            def hook_factory(name):
                def _hook(_, inp, out):
                    if name not in act_caches:
                        act_caches[name] = []
                    # Collect up to 16 sequences to estimate P95 without OOM
                    if len(act_caches[name]) < 16:
                        act_caches[name].append(inp[0].detach().cpu().float().abs().view(-1, inp[0].shape[-1]))
                return _hook

            hooks = []
            hooks.append(block.self_attn.q_proj.register_forward_hook(hook_factory("qkv")))
            hooks.append(block.self_attn.o_proj.register_forward_hook(hook_factory("o")))
            hooks.append(block.mlp.gate_proj.register_forward_hook(hook_factory("gate_up")))
            hooks.append(block.mlp.down_proj.register_forward_hook(hook_factory("down")))

            device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                ikw = inp_kwargs.copy()
                ikw["use_cache"] = False
                if "past_key_value" in ikw: ikw["past_key_value"] = None
                if "output_attentions" in ikw: ikw["output_attentions"] = False
                with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                    block(*to(inp_args, device=device), **to(ikw, device=device))
            for h in hooks: h.remove()
            
            act_means = {}
            for name, caches in list(act_caches.items()):
                X = torch.cat(caches, dim=0) # (N_tokens, Dim)
                if args.channel_resort == "P95":
                    act_means[name] = torch.quantile(X, 0.95, dim=0).to(device)
                else: # "mean" or "R_val"
                    act_means[name] = X.mean(dim=0).to(device)
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

            for name, mean_val in act_means.items():
                abs_vals = mean_val.abs()
                N = abs_vals.shape[0]
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

        # 1. Init transforms
        qkv_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        o_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        gate_up_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        down_in_transform = build_transform(args.transform_class, size=model.config.intermediate_size, **transform_kwargs)     

        if resort_perms:
            from ..transforms.transforms import CompositeTransform, PermutationTransform
            if "qkv" in resort_perms:
                qkv_in_transform = CompositeTransform([PermutationTransform(resort_perms["qkv"]), qkv_in_transform])
            if "gate_up" in resort_perms:
                gate_up_in_transform = CompositeTransform([PermutationTransform(resort_perms["gate_up"]), gate_up_in_transform])
            # O: in-place permutation only for MHA (skip GQA to avoid cross-head corruption)
            if "o" in resort_perms:
                o_perm = resort_perms["o"]
                if block.self_attn.v_proj.weight.shape[0] == block.self_attn.o_proj.weight.shape[1]:
                    block.self_attn.o_proj.weight.data = block.self_attn.o_proj.weight.data[:, o_perm]
                    block.self_attn.v_proj.weight.data = block.self_attn.v_proj.weight.data[o_perm, :]
            # Down: in-place weight permutation
            if "down" in resort_perms:
                down_perm = resort_perms["down"]
                block.mlp.down_proj.weight.data = block.mlp.down_proj.weight.data[:, down_perm]
                block.mlp.gate_proj.weight.data = block.mlp.gate_proj.weight.data[down_perm, :]
                block.mlp.up_proj.weight.data = block.mlp.up_proj.weight.data[down_perm, :]

        # NOTE: Do NOT pre-rotate weights here. fix_parametrization() below handles it.
        # Pre-rotating would cause double rotation since H(H(W)) = W.

        # 2. Replace blocks with quantized versions
        quantized_attn = get_attention_layer(model.config)(
            model.config, layer_idx=block_idx,
            weight_quantizer_kwargs=weight_quantizer_kwargs,
            act_quantizer_kwargs=act_quantizer_kwargs,
            qkv_in_transform=qkv_in_transform, o_in_transform=o_in_transform
        )
        quantized_mlp = get_mlp_layer(model.config)(
            model.config,
            weight_quantizer_kwargs=weight_quantizer_kwargs,
            act_quantizer_kwargs=act_quantizer_kwargs,
            gate_up_in_transform=gate_up_in_transform, down_in_transform=down_in_transform
        )

        quantized_attn.load_state_dict(block.self_attn.state_dict(), strict=False)
        quantized_mlp.load_state_dict(block.mlp.state_dict(), strict=False)
        block.self_attn = quantized_attn
        block.mlp = quantized_mlp
        block = block.to(device=device, dtype=orig_dtype)  

        # 3. Global scale collection for NVFP
        if weight_quantizer_kwargs and args.scale_precision == ScalePrecision.E4M3:
            for layer_name, layer in block.named_modules():
                if isinstance(layer, QLinear):
                    # Weights are already rotated above
                    layer.weight_quantizer.get_quantization_params(layer.weight)
                    layer.weight_quantizer._track_global_scale = False

            if args.fuse_global_scale:
                qkv_global_scale = min(
                    quantized_attn.q_proj.weight_quantizer.global_scale,
                    quantized_attn.k_proj.weight_quantizer.global_scale,
                    quantized_attn.v_proj.weight_quantizer.global_scale,
                )
                quantized_attn.q_proj.weight_quantizer.global_scale = qkv_global_scale
                quantized_attn.k_proj.weight_quantizer.global_scale = qkv_global_scale
                quantized_attn.v_proj.weight_quantizer.global_scale = qkv_global_scale
                # gate_up fusion
                gate_up_global_scale = min(
                    quantized_mlp.gate_proj.weight_quantizer.global_scale,
                    quantized_mlp.up_proj.weight_quantizer.global_scale
                )
                quantized_mlp.gate_proj.weight_quantizer.global_scale = gate_up_global_scale
                quantized_mlp.up_proj.weight_quantizer.global_scale = gate_up_global_scale

        # Calibrate activations (if needed)
        if need_calibration:
            device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                    block(*to(inp_args, device=device), **to(inp_kwargs, device=device))

        
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

    clear_device_cache(garbage_collection=True)

    return quantized_state_dict, non_quantized_state_dict
