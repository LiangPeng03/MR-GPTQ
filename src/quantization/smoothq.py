"""
SmoothQuant: Smooth Activation-Weight Quantization via Per-Channel Scaling.

Based on the SmoothQuant paper (https://arxiv.org/abs/2211.10438):
  s_j = max(|X|_j)^α / max(|W|_j)^(1-α)
  X_new = X / s_j,  W_new = W * s_j

The returned scales (1/s_j) are designed to be consumed by DiagonalScaleTransform,
which applies x * scale to activations and w / scale = w * s_j to weights (via inv_t).
This ensures numeric equivalence: Y = X_new @ W_new^T = X @ W^T.

IMPORTANT: SmoothQuant MUST execute before channel_resort and channel_rescale,
since those rely on activation statistics that should reflect smoothed weights.
"""

from __future__ import annotations

import torch

from ..utils.common_utils import to


def _get_combined_weight(block, name: str) -> torch.Tensor | None:
    """Get combined weight tensor for the given projection group name."""
    if name == "qkv":
        w = torch.cat([
            block.self_attn.q_proj.weight,
            block.self_attn.k_proj.weight,
            block.self_attn.v_proj.weight,
        ], dim=0)
    elif name == "o":
        w = block.self_attn.o_proj.weight
    elif name == "gate_up":
        w = torch.cat([
            block.mlp.gate_proj.weight,
            block.mlp.up_proj.weight,
        ], dim=0)
    elif name == "down":
        w = block.mlp.down_proj.weight
    else:
        return None
    return w.float()


def apply_smoothquant(
    block,
    input_args: list,
    input_kwargs: list,
    args,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Compute SmoothQuant per-channel scales for a transformer block.

    Runs a calibration forward pass through the block to collect input activation
    statistics, then computes smoothing factors using the SmoothQuant formula.

    Args:
        block: The transformer block (with .self_attn and .mlp submodules).
        input_args: List of input positional args from calibration data.
        input_kwargs: List of input keyword args from calibration data.
        args: Global argparse namespace (must have smoothquant_alpha).
        device: Target device.

    Returns:
        A dict mapping projection group names ("qkv", "o", "gate_up", "down")
        to activation scale tensors (1/s_j), suitable for DiagonalScaleTransform.
        Returns empty dict if smoothquant_alpha is None.
    """
    alpha = getattr(args, "smoothquant_alpha", None)
    if alpha is None:
        return {}

    print(f"  [SmoothQuant] Computing per-channel smooth scales (alpha={alpha})...")

    # ---- 1. Collect per-channel input activation max via forward hooks ----
    act_caches: dict[str, list[torch.Tensor]] = {}

    def _hook_factory(name: str):
        def _hook(_, inp, _out):
            if name not in act_caches:
                act_caches[name] = []
            # Collect up to 16 sequences to limit memory
            if len(act_caches[name]) < 16:
                act_caches[name].append(inp[0].detach().float().abs().cpu())
        return _hook

    hooks = [
        block.self_attn.q_proj.register_forward_hook(_hook_factory("qkv")),
        block.self_attn.o_proj.register_forward_hook(_hook_factory("o")),
        block.mlp.gate_proj.register_forward_hook(_hook_factory("gate_up")),
        block.mlp.down_proj.register_forward_hook(_hook_factory("down")),
    ]

    device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    for inp_args, inp_kwargs in zip(input_args, input_kwargs):
        ikw = inp_kwargs.copy()
        ikw["use_cache"] = False
        if "past_key_value" in ikw:
            ikw["past_key_value"] = None
        if "output_attentions" in ikw:
            ikw["output_attentions"] = False
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
            block(*to(inp_args, device=device), **to(ikw, device=device))

    for h in hooks:
        h.remove()

    # ---- 2. Compute SmoothQuant scales ----
    smoothquant_scales: dict[str, torch.Tensor] = {}

    for name, caches in act_caches.items():
        # Per-channel max of input activations
        # caches[i] has shape (seq_len, hidden_dim) varying per sample
        X_max_per_sample = []
        for c in caches:
            # Reduce all dims except the last (channel) dim
            reduce_dims = tuple(range(c.ndim - 1))
            X_max_per_sample.append(c.amax(dim=reduce_dims))
        X_all = torch.stack(X_max_per_sample, dim=0)  # (n_samples, hidden_dim)
        X_max = X_all.amax(dim=0).to(device)  # (hidden_dim,)

        # Per-channel max of weight magnitude (along input dimension, i.e. dim=0)
        w = _get_combined_weight(block, name)
        if w is None:
            continue
        W_max = w.abs().amax(dim=0).to(device)  # (hidden_dim,)

        # SmoothQuant formula: s_j = max(|X|_j)^α / max(|W|_j)^(1-α)
        eps = 1e-8
        X_max_c = X_max.clamp(min=eps)
        W_max_c = W_max.clamp(min=eps)
        s = (X_max_c ** alpha) / (W_max_c ** (1.0 - alpha))
        s = s.clamp(min=0.01, max=100.0)

        # Store activation scale = 1/s_j for DiagonalScaleTransform
        #   Activation forward:  x * (1/s_j) = x / s_j  ✓
        #   Weight inv_t:        w / (1/s_j) = w * s_j  ✓
        smoothquant_scales[name] = 1.0 / s

        print(
            f"    [{name:8}] SmoothQuant s: mean={s.mean():.3f}, "
            f"min={s.min():.3f}, max={s.max():.3f}"
        )

    # Clean up
    del act_caches
    torch.cuda.empty_cache()

    return smoothquant_scales
