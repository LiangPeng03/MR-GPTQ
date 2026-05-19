import re
import gc
import math
import argparse
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import AutoModelForCausalLM

from .qlinear import QLinear
from .quantizer import Quantizer
from .quant_args import QuantizationOrder
from .quant_ops import pack_fp4_to_uint8, cast_scales_to_eXmY, ScalePrecision
from .accumulate_hessian import accumulate_hessian
from ..transforms.transforms import build_transform, get_transform_matrix
from ..utils.linalg_utils import inv_sym
from ..utils.common_utils import clear_device_cache, to, maybe_first_element
from ..utils.model_utils import InputCollector, ForwardInterrupt, get_attention_layer, get_mlp_layer, get_number_of_rows_and_cols

try:
    import wandb
except ImportError:
    wandb = None


def get_relative_mse_error(q: torch.Tensor, w: torch.Tensor, H: torch.Tensor):
    delta = q - w
    return (delta).mm(H).mul(delta).mean() / (w.mm(H).mul(w).mean() + 1e-6)


class GPTQ:

    def __init__(
        self,
        layer: nn.Module,
        quantizer: Quantizer,
        quantization_order: str = "default",
        block_size: int = 128,
        rel_damp: float = 1e-2,
        export_quantized_model: str = "",
    ):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = get_number_of_rows_and_cols(layer)
        # Quantization properties
        self.quantizer = quantizer
        self.quantization_order = QuantizationOrder(quantization_order)
        self.block_size = block_size
        self.rel_damp = rel_damp
        # Backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.H = None
        self.num_samples = 0
        # Whether to apply real quantization
        self.export_quantized_model = export_quantized_model

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "OBC supports only linear and convolutional layers."

    # preparatory methods
    @torch.no_grad()
    def update(self, input: torch.Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            input: batch of layer inputs
        """
        # get batch size
        batch_size = input.shape[0]
        # init hessian
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        # input reshaping
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # output size (batch_size, channels * \prod kernel_size, num_patches)
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
        # cast input to float32 before addition
        input = input.float()
        # rescale and update matrix
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.mul_(beta)
        input.mul_(math.sqrt(alpha))
        accumulate_hessian(self.H, input)
        self.num_samples += batch_size

    def reset(self) -> None:
        self.W = self.layer.weight
        self.H = None
        self.num_samples = 0
        clear_device_cache()

    @torch.no_grad()
    def quantization_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        # 1) Hessian preparation
        assert self.H is not None, "One has to process at least one sample of calibration data to run pruning"
        # 2) Weight preparation
        # copy weight, flatten and convert to float
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
        # flag pre step as completed
        self.pre_step_completed = True

    @torch.no_grad()
    def step(self) -> torch.Tensor | Optional[torch.Tensor] | torch.Tensor:
        """
        Quantize the weight matrix using GPTQ
        """
        # 1) Define constants and chunk
        d_col, block_size, device, dtype = self.d_col, self.block_size, self.W_device, self.W_dtype
        # 2) Get quantization group size
        quantizer_group_size = self.quantizer.group_size
        group_size = quantizer_group_size or d_col
        num_groups = d_col // group_size

        # Init quantized weight
        qweight = None
        if self.export_quantized_model:
            qweight = torch.empty(self.W.shape, device=device, dtype=dtype)
        # Get scales and zeros 
        scales, zeros = self.quantizer.get_quantization_params(self.W) 
        # Dirty hack for GPTQ quantization
        self.quantizer.group_size = None
        # Get permutation
        if self.quantization_order == QuantizationOrder.ACTIVATION:
            perm = torch.argsort(self.H.diag(), descending=True)
            group_idx = torch.arange(num_groups, device=device).repeat_interleave(group_size)[perm]
        else:
            perm = torch.arange(d_col, device=device)
        perm_inv = torch.argsort(perm)
        # Permute Hessian prior to inversion
        self.H = self.H[perm][:, perm]
        # Get weight
        w = self.W[:, perm]
        # Get Hessian inverse   
        H_inv_cho = self._get_hessian_inverse()
        # Quantize
        for c1 in range(0, d_col, block_size):
            c2 = min(c1 + block_size, d_col)
            ncols = c2 - c1
            w_blk = w[:, c1:c2].clone()  
            errs = torch.zeros_like(w_blk)
            H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]
            # 2) Iterate over block
            for i in range(ncols):
                # Get weight column, corresponding Hessian diagonal and group_id
                w_ci = w_blk[:, i]
                d = H_inv_cho_blk[i, i]
                if self.quantization_order == QuantizationOrder.ACTIVATION:
                    g_idx = group_idx[c1 + i]
                else:
                    g_idx = (c1 + i) // group_size    
                # Quantize weight column
                if self.export_quantized_model:
                    q = self.quantizer.quantize(w_ci, scales[:, g_idx], zeros[:, g_idx])
                    w_q = self.quantizer.dequantize(q, scales[:, g_idx], zeros[:, g_idx])
                    qweight[:, c1 + i] = q
                else:
                    w_q = self.quantizer(w_ci, scales[:, g_idx], zeros[:, g_idx])
                w[:, c1 + i] = w_q
                # Update subsequent weight
                err = (w_ci - w_q) / d
                w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                errs[:, i] = err
            # 3) Update the weights after block
            w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

        # Invert permutation
        w = w[:, perm_inv].contiguous()
        if qweight is not None:
            qweight = qweight[:, perm_inv].contiguous()
        self.H = self.H[perm_inv][:, perm_inv]
        # Restore quantizer group size
        self.quantizer.group_size = quantizer_group_size
        
        return w.to(dtype), qweight, scales
    
    @torch.no_grad()
    def _get_hessian_inverse(self):
        w = self.W
        # Get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        H = self.H
        # mask rows with zero input channels
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        # Hessian regularization
        damp = self.rel_damp * torch.diag(self.H).mean()
        self.H[range(self.d_col), range(self.d_col)] += damp
        # invert
        try:
            H = inv_sym(H)
            H_inv_cho = torch.linalg.cholesky(H, upper=True)
        except:
            H_inv_cho = torch.eye(self.d_col, device=H.device, dtype=torch.float32)
        return H_inv_cho

    def quantize(self) -> torch.Tensor | Optional[torch.Tensor] | torch.Tensor:
        self.quantization_pre_step()
        return self.step()


def gptq_quantization(
    model: AutoModelForCausalLM, 
    calibration_data: List[torch.Tensor],
    args: argparse.Namespace, 
    device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    print("GPTQ quantization...")
    orig_dtype = model.config.torch_dtype if args.dtype == "auto" else args.dtype
    act_offload_device = "cpu" if args.cpu_offload_activations else device
    # State dict with quantized weights, scales and hadamards
    quantized_state_dict = {}
    non_quantized_state_dict = {}
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
            scale_precision=args.scale_precision
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
            scale_precision=args.scale_precision
        )

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
            if args.cpu_offload_activations:
                sample = sample.to(device="cpu")
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
        if getattr(args, "channel_resort", "none") != "none":
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

            resort_hooks = []
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
            
            act_means = {}
            for name, caches in list(act_caches.items()):
                X = torch.cat(caches, dim=0) # (N_tokens, Dim)
                if args.channel_resort in ["P95", "minmax"]:
                    act_means[name] = torch.quantile(X.float(), 0.95, dim=0).to(device)
                else: # "mean" or "R_val"
                    act_means[name] = X.float().mean(dim=0).to(device)
                del X
                del act_caches[name]

            def compute_resonance_perm(act_mean_abs, group_size=16, target_R=0.2):
                N = act_mean_abs.shape[0]
                vals = act_mean_abs.cpu().numpy()
                sorted_pos = np.argsort(-vals)
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

            def compute_hessian_quarantine_perm(h_diag, block_size=128):
                return torch.argsort(h_diag, descending=True).to(device=h_diag.device, dtype=torch.long)

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
                            p = compute_truncated_minmax_perm(abs_vals, group_size=gs, outlier_ratio=args.outlier_ratio)
                        else:
                            p = torch.zeros(N, device=abs_vals.device, dtype=torch.long)
                            for i in range(0, N, h_group_size):
                                block_vals = abs_vals[i : i + h_group_size]
                                block_gs = min(gs, h_group_size)
                                block_perm = compute_truncated_minmax_perm(block_vals, group_size=block_gs, outlier_ratio=args.outlier_ratio)
                                p[i : i + h_group_size] = block_perm + i
                        print(f"    [{name:8}] Truncated MinMax Pairing (gs={gs}, ratio={args.outlier_ratio}) completed.")
                    else:
                        gs = quant_group_size
                        if "o" in name:
                            head_dim = model.config.hidden_size // model.config.num_attention_heads
                            h_group_size = head_dim
                        else:
                            h_group_size = -1
                            
                        if h_group_size <= 0:
                            p = compute_truncated_minmax_perm(abs_vals, group_size=gs, outlier_ratio=args.outlier_ratio)
                        else:
                            p = torch.zeros(N, device=abs_vals.device, dtype=torch.long)
                            for i in range(0, N, h_group_size):
                                block_vals = abs_vals[i : i + h_group_size]
                                block_gs = min(gs, h_group_size)
                                block_perm = compute_truncated_minmax_perm(block_vals, group_size=block_gs, outlier_ratio=args.outlier_ratio)
                                p[i : i + h_group_size] = block_perm + i
                        print(f"    [{name:8}] Truncated MinMax Pairing (gs={gs}, ratio={args.outlier_ratio}) completed.")
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
            if "down" in resort_perms:
                down_perm = resort_perms["down"]
                block.mlp.down_proj.weight.data = block.mlp.down_proj.weight.data[:, down_perm]
                block.mlp.gate_proj.weight.data = block.mlp.gate_proj.weight.data[down_perm, :]
                block.mlp.up_proj.weight.data = block.mlp.up_proj.weight.data[down_perm, :]

        # 2. Replace blocks with quantized versions
        quantized_attn = get_attention_layer(model.config)(
            model.config,
            layer_idx=block_idx,
            act_quantizer_kwargs=act_quantizer_kwargs,
            qkv_in_transform=qkv_in_transform,
            o_in_transform=o_in_transform
        )
        quantized_mlp = get_mlp_layer(model.config)(
            model.config,
            act_quantizer_kwargs=act_quantizer_kwargs,
            gate_up_in_transform=gate_up_in_transform,
            down_in_transform=down_in_transform
        )

        quantized_attn.load_state_dict(block.self_attn.state_dict(), strict=False)
        quantized_mlp.load_state_dict(block.mlp.state_dict(), strict=False)

        block.self_attn = quantized_attn
        block.mlp = quantized_mlp

        # Move to original device and dtype
        block = block.to(device=device, dtype=orig_dtype)
        # Toggle off gradients for all parameters
        block.requires_grad_(False)

        # 3. Fix transforms and remove parametrizations
        qkv_in_transform.remove_parametrizations()
        o_in_transform.remove_parametrizations()
        gate_up_in_transform.remove_parametrizations()
        down_in_transform.remove_parametrizations() 

        # --- NEW: Transform all weights BEFORE calibration ---
        # This ensures the model forward pass (if any) uses pre-transformed weights, 
        # and more importantly, that GPTQ handles are initialized with rotated weights.
        block.self_attn.q_proj.weight.data = qkv_in_transform(block.self_attn.q_proj.weight, inv_t=True)
        block.self_attn.k_proj.weight.data = qkv_in_transform(block.self_attn.k_proj.weight, inv_t=True)
        block.self_attn.v_proj.weight.data = qkv_in_transform(block.self_attn.v_proj.weight, inv_t=True)
        block.self_attn.o_proj.weight.data = o_in_transform(block.self_attn.o_proj.weight, inv_t=True)
        block.mlp.gate_proj.weight.data = gate_up_in_transform(block.mlp.gate_proj.weight, inv_t=True)
        block.mlp.up_proj.weight.data = gate_up_in_transform(block.mlp.up_proj.weight, inv_t=True)
        block.mlp.down_proj.weight.data = down_in_transform(block.mlp.down_proj.weight, inv_t=True)

        # 4. Create GPTQ handles and hooks
        gptq_handles = {}
        hooks = {}
        for layer_name, layer in block.named_modules():
            if isinstance(layer, QLinear):
                # Create GPTQ handle
                gptq_handles[layer_name] = GPTQ(
                    layer, 
                    Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None, 
                    quantization_order=args.quantization_order, 
                    rel_damp=args.rel_damp,
                    export_quantized_model=args.export_quantized_model
                )
                
                # Update weight reference in handle (since it was just rotated)
                gptq_handles[layer_name].W = layer.weight.data.clone()

                # Get weight global scale
                if args.scale_precision == ScalePrecision.E4M3:
                    # Weights are already rotated, just compute scales
                    gptq_handles[layer_name].quantizer.get_quantization_params(layer.weight)
                    gptq_handles[layer_name].quantizer._track_global_scale = False

                # Attach hook. The activation inp[0] is ALREADY transformed by LlamaAttention/LlamaMLP.
                def update_handle_hook(name):
                    def _hook(_, inp, out):
                        x = inp[0]
                        gptq_handles[name].update(x)
                    return _hook
                
                hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))

        # Fuse global scales
        if args.fuse_global_scale and args.scale_precision == ScalePrecision.E4M3:
            # qkv fusion
            qkv_global_scale = min(
                gptq_handles["self_attn.q_proj"].quantizer.global_scale,
                gptq_handles["self_attn.k_proj"].quantizer.global_scale,
                gptq_handles["self_attn.v_proj"].quantizer.global_scale,
            )
            gptq_handles["self_attn.q_proj"].quantizer.global_scale = qkv_global_scale
            gptq_handles["self_attn.k_proj"].quantizer.global_scale = qkv_global_scale
            gptq_handles["self_attn.v_proj"].quantizer.global_scale = qkv_global_scale
            # gate_up fusion
            gate_up_global_scale = min(
                gptq_handles["mlp.gate_proj"].quantizer.global_scale,
                gptq_handles["mlp.up_proj"].quantizer.global_scale
            )
            gptq_handles["mlp.gate_proj"].quantizer.global_scale = gate_up_global_scale
            gptq_handles["mlp.up_proj"].quantizer.global_scale = gate_up_global_scale

        # Set train_mode to False BEFORE calibration
        # so QLinear doesn't dynamically transform weights we already pre-rotated
        for layer_name, layer in block.named_modules():
            if isinstance(layer, QLinear):
                layer._train_mode = False

        # 5. Process calibration data (Collects Hessian aligned with rotated weights)
        device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            inp_kwargs["use_cache"] = False
            if "past_key_value" in inp_kwargs: inp_kwargs["past_key_value"] = None
            if "output_attentions" in inp_kwargs: inp_kwargs["output_attentions"] = False
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
        
        # Remove hooks
        for hook in hooks.values(): hook.remove()

        # Freeze act scales AFTER calibration
        for layer_name, layer in block.named_modules():
            if isinstance(layer, QLinear):
                if layer.act_quantizer:
                    layer.act_quantizer._track_global_scale = False

        # 7. Run GPTQ quantization (Already aligned)
        for layer_name, gptq_handle in gptq_handles.items():
            dequantized_qweight, qweight, scales = gptq_handle.quantize()
            orig_weight = gptq_handle.layer.weight
            with torch.no_grad():
                relative_mse_error = get_relative_mse_error(dequantized_qweight.float(), orig_weight.float(), gptq_handle.H)
            print(f"[{layer_name:16}]: Relative MSE error: {relative_mse_error.item():.2e}")
            if args.log_wandb:
                wandb.log({f"gptq/{layer_name}_relative_mse": relative_mse_error.item()})
            gptq_handle.layer.weight.data = dequantized_qweight
            
            # Update quantized state dict (if needed)
            if args.export_quantized_model:
                weight_global_scale = gptq_handle.quantizer.global_scale.to(scales.device)
                act_global_scale = gptq_handle.layer.act_quantizer.global_scale

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
                    quantized_state_dict[f"model.layers.{block_idx}.{layer_name}"] = {
                        "dqweight": dequantized_qweight.cpu(),
                        "forward_hadamard_matrix": transform_matrix,
                        "backward_hadamard_matrix": transform_matrix.clone(),
                        "weight_global_scale": weight_global_scale.clone(),
                        "act_global_scale": act_global_scale.clone()
                    }

        # Enable activation MSE tracking
        if args.show_act_mse:
            for layer_name, layer in block.named_modules():
                if isinstance(layer, QLinear):
                    layer.track_act_mse = True

        # 8. Update activations
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
                    if args.log_wandb:
                        wandb.log({f"gptq/{layer_name}_act_relative_mse": act_rel_mse})
                    layer.track_act_mse = False


        if args.cpu_offload_modules:
            block = block.cpu()

        # 8. Clean-up
        del gptq_handles
        del hooks
        clear_device_cache(garbage_collection=True)

    clear_device_cache(garbage_collection=True)

    return quantized_state_dict, non_quantized_state_dict
