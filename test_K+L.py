"""Mechanism analysis for DACC and BITR.

The script compares the original layout, simple controls, production DACC,
and DACC followed by production BITR on calibration activations from layers
0, 15, and 31.  It reports activation MSE, MDR/HDR occupancy, and both the
relative and absolute MDR/HDR contributions.  The absolute regional MSE is
normalized by the number of all elements, so it is directly comparable with
the global MSE and is additive across regions.  It does not quantize the model
or run perplexity evaluation.
"""

import argparse
import copy
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.quantization.gics import optimize_channel_scales_coordinate_descent
from src.quantization.quant_ops import FP4_E2M1_MAX, FP8_E4M3_MAX, cast_to_fp4
from src.quantization.quantizer import get_reciprocal
from src.utils.data_utils import get_data


class ForwardInterrupt(Exception):
    pass


class InputCollector(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.input_args = []
        self.input_kwargs = []

    def forward(self, *input_args, **input_kwargs):
        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)
        raise ForwardInterrupt


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(to_device(x, device) for x in value)
    if isinstance(value, list):
        return [to_device(x, device) for x in value]
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def maybe_first(value):
    return value[0] if isinstance(value, tuple) else value


def get_combined_weight(block, name):
    if name == "qkv":
        weight = torch.cat([block.self_attn.q_proj.weight,
                            block.self_attn.k_proj.weight,
                            block.self_attn.v_proj.weight], dim=0)
    elif name == "o":
        weight = block.self_attn.o_proj.weight
    elif name == "gate_up":
        weight = torch.cat([block.mlp.gate_proj.weight,
                            block.mlp.up_proj.weight], dim=0)
    elif name == "down":
        weight = block.mlp.down_proj.weight
    else:
        raise ValueError(name)
    return weight.detach().float()


def compute_global_scale(x):
    tensor_max = x.abs().max().to(torch.float32).view(1)
    return (FP8_E4M3_MAX * FP4_E2M1_MAX *
            get_reciprocal(tensor_max)).to(x.device)


def scale_to_e4m3(raw_scale, global_scale):
    return ((raw_scale * global_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn).to(torch.float32)
            .mul(get_reciprocal(global_scale)))


def proxy_distances(all_channels, reference, channel_weights=None,
                    chunk_size=256):
    """Pairwise proxy used by the production cumulative-distance DACC."""
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6],
                        device=all_channels.device, dtype=torch.float32)
    result = torch.zeros(all_channels.shape[0], device=all_channels.device)
    reference = reference.unsqueeze(0)
    for start in range(0, all_channels.shape[0], chunk_size):
        chunk = all_channels[start:start + chunk_size]
        p_max = torch.maximum(chunk, reference)
        p_min = torch.minimum(chunk, reference)
        scale = (p_max / 6.0).clamp(min=1e-10)
        candidates = scale.unsqueeze(-1) * grid
        nearest = candidates.gather(
            -1, (p_min.unsqueeze(-1) - candidates).abs().argmin(
                -1, keepdim=True)).squeeze(-1)
        distance = ((p_min - nearest) ** 2).sum(dim=1)
        if channel_weights is not None:
            distance = distance * channel_weights[start:start + chunk.size(0)]
        result[start:start + chunk.size(0)] = distance
    return result


def compute_dacc_perm(x_abs_t, group_size=16, channel_weights=None):
    """Production DACC: cumulative seed selection followed by greedy fill."""
    channels = x_abs_t.shape[0]
    if channels % group_size:
        raise ValueError(f"channels={channels} is not divisible by {group_size}")
    n_groups = channels // group_size
    if channel_weights is None:
        channel_weights = torch.ones(channels, device=x_abs_t.device)
    importance = x_abs_t.sum(dim=1) * channel_weights

    first = importance.argmax().item()
    seeds = [first]
    cumulative = proxy_distances(x_abs_t, x_abs_t[first], channel_weights)
    cumulative[first] = -float("inf")
    while len(seeds) < n_groups:
        seed = cumulative.argmax().item()
        seeds.append(seed)
        cumulative[seed] = -float("inf")
        cumulative += proxy_distances(x_abs_t, x_abs_t[seed], channel_weights)
        cumulative[seed] = -float("inf")

    groups = [[seed] for seed in seeds]
    envelopes = x_abs_t[seeds].clone()
    sizes = torch.ones(n_groups, device=x_abs_t.device, dtype=torch.long)
    chosen = set(seeds)
    remaining = [c for c in range(channels) if c not in chosen]
    remaining.sort(key=lambda c: float(importance[c]), reverse=True)
    for channel in remaining:
        distances = proxy_distances(envelopes, x_abs_t[channel])
        distances[sizes >= group_size] = float("inf")
        group = distances.argmin().item()
        groups[group].append(channel)
        envelopes[group] = torch.maximum(envelopes[group], x_abs_t[channel])
        sizes[group] += 1
    return torch.tensor([c for group in groups for c in group],
                        device=x_abs_t.device, dtype=torch.long)


def compute_outlier_stagger_perm(x, weight, group_size=16, quantile=0.9375):
    """Greedily separate channels with co-occurring activation/weight outliers."""
    channels = x.shape[1]
    if channels % group_size:
        raise ValueError(
            f"channels={channels} is not divisible by {group_size}"
        )

    threshold_a = torch.quantile(
        x.abs().float(), quantile, dim=1, keepdim=True
    )
    activation_mask = (x.abs() > threshold_a).float()

    threshold_w = torch.quantile(
        weight.abs().float(), quantile, dim=1, keepdim=True
    )
    weight_mask = (weight.abs() > threshold_w).float()

    # Balance activation-token and weight-row contributions, matching the
    # staggering implementation used by the quantizer.
    weight_factor = math.sqrt(x.shape[0] / max(weight.shape[0], 1))
    joint_mask = torch.cat(
        [activation_mask, weight_mask * weight_factor], dim=0
    )
    frequency = joint_mask.mean(dim=0)
    channel_order = torch.argsort(frequency, descending=True).tolist()

    num_groups = channels // group_size
    groups = [[] for _ in range(num_groups)]
    group_profiles = torch.zeros(
        num_groups, joint_mask.shape[0], device=x.device,
        dtype=torch.float32,
    )
    group_sizes = torch.zeros(
        num_groups, device=x.device, dtype=torch.long
    )

    for channel in channel_order:
        profile = joint_mask[:, channel]
        penalties = torch.mv(group_profiles, profile)
        penalties[group_sizes >= group_size] = float("inf")
        group = penalties.argmin().item()
        groups[group].append(channel)
        group_profiles[group] += profile
        group_sizes[group] += 1

    return torch.tensor(
        [channel for group in groups for channel in group],
        device=x.device, dtype=torch.long,
    )


def compute_joint_second_moment_perm(x, weight, alpha=0.5):
    """PermuQuant-style joint second-moment ordering control."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], but got {alpha}")

    activation_moment = x.float().pow(2).mean(dim=0)
    weight_moment = weight.float().pow(2).mean(dim=0)
    eps = torch.finfo(torch.float32).tiny
    score = (
        alpha * torch.log(activation_moment.clamp_min(eps))
        + (1.0 - alpha) * torch.log(weight_moment.clamp_min(eps))
    )
    return torch.argsort(score, descending=True)


def build_layout_permutation(
    x, weight, name, kind, group_size, model,
    stagger_quantile=0.9375, permu_alpha=0.5,
):
    """Build a control or DACC permutation in the same scopes as production."""
    channels = x.shape[1]
    if name in {"o", "down"}:
        # The production quantizer keeps attention heads/intermediate chunks
        # independent for these two projections (default block scope=head_dim).
        scope = model.config.hidden_size // model.config.num_attention_heads
    else:
        scope = channels
    if scope % group_size:
        raise ValueError(f"scope={scope} is not divisible by B={group_size}")

    result = []
    x_abs_t = x.abs().T.float()
    for start in range(0, channels, scope):
        stop = min(start + scope, channels)
        local_values = x[:, start:stop].float()
        local_weight = weight[:, start:stop].float()
        local_x = x_abs_t[start:stop]
        if kind == "dacc":
            local_weights = local_weight.pow(2).sum(dim=0)
            local_perm = compute_dacc_perm(local_x, group_size, local_weights)
        elif kind == "outlier_stagger":
            local_perm = compute_outlier_stagger_perm(
                local_values, local_weight, group_size, stagger_quantile
            )
        elif kind == "joint_second_moment":
            local_perm = compute_joint_second_moment_perm(
                local_values, local_weight, permu_alpha
            )
        elif kind == "magnitude":
            local_perm = torch.argsort(local_x.sum(dim=1), descending=True)
        elif kind == "random":
            local_perm = torch.randperm(stop - start, device=x.device)
        elif kind == "identity":
            local_perm = torch.arange(stop - start, device=x.device)
        else:
            raise ValueError(kind)
        result.append(local_perm + start)
    return torch.cat(result)


def activation_stats(x, permutation, group_size=16, channel_scale=None):
    """Quantize one layout and return comparable original-unit statistics."""
    x_layout = x[:, permutation].float().contiguous()
    if channel_scale is None:
        channel_scale = torch.ones(x_layout.shape[1], device=x.device)
    channel_scale = channel_scale.to(x.device).float()
    x_work = x_layout * channel_scale.unsqueeze(0)
    groups = x_work.view(-1, group_size)
    abs_groups = groups.abs()
    block_max = abs_groups.amax(dim=1, keepdim=True)
    raw_scale = (block_max / FP4_E2M1_MAX).clamp(min=1e-10)
    raw_scale = torch.where(block_max == 0, torch.ones_like(raw_scale), raw_scale)
    global_scale = compute_global_scale(x_work)
    block_scale = scale_to_e4m3(raw_scale, global_scale)
    q = cast_to_fp4(groups / block_scale)
    scale_groups = channel_scale.unsqueeze(0).expand(
        x_layout.shape[0], -1).contiguous().view(-1, group_size)
    reconstructed = (q * block_scale) / scale_groups
    original = groups / scale_groups
    error = (original - reconstructed) ** 2

    # # u=6|z|/block_max is the normalized E2M1 coordinate used in the paper.
    # u = 6.0 * abs_groups / block_max.clamp(min=1e-10)
    # hdr = (u > 4.5) & (u < 5.5)
    # mdr = (((u > 2.25) & (u < 2.75)) |
    #        ((u > 3.25) & (u < 3.75)) |
    #        ((u > 4.25) & (u <= 4.5)) |
    #        ((u >= 5.5) & (u < 5.75)))
    # mdr_hdr = mdr | hdr

    # Use the actual effective block scale after E4M3 projection.
    # This corresponds to |z_i| = |x_i| / (s_T * s_B).
    u = (groups / block_scale).abs()

    valid = u <= FP4_E2M1_MAX

    hdr = valid & (u > 4.5) & (u < 5.5)

    mdr = valid & (
        ((u > 2.25) & (u < 2.75)) |
        ((u > 3.25) & (u < 3.75)) |
        ((u > 4.25) & (u <= 4.5)) |
        ((u >= 5.5) & (u < 5.75))
    )

    ldr = valid & ~(mdr | hdr)
    overflow = u > FP4_E2M1_MAX
    mdr_hdr = mdr | hdr

    total = error.sum().clamp(min=1e-20).item()
    # Absolute regional MSE contribution in the original value units.  The
    # denominator is the number of all values (not the number of values in
    # the region), making this quantity comparable to ``mse`` and additive
    # across disjoint regions.  This avoids the misleading effect where a
    # lower total MSE can make a region's percentage contribution increase.
    num_values = error.numel()
    mdr_hdr_abs_mse = error[mdr_hdr].sum().item() / max(num_values, 1)
    hdr_abs_mse = error[hdr].sum().item() / max(num_values, 1)
    return {
        "mse": float(error.mean().item()),
        "ldr_pct": float(ldr.float().mean().item() * 100),
        "mdr_pct": float(mdr.float().mean().item() * 100),
        "hdr_pct": float(hdr.float().mean().item() * 100),
        "overflow_pct": float(overflow.float().mean().item() * 100),
        "mdr_hdr_mse_pct": float(error[mdr_hdr].sum().item() / total * 100),
        "hdr_mse_pct": float(error[hdr].sum().item() / total * 100),
        "mdr_hdr_mse_abs": float(mdr_hdr_abs_mse),
        "hdr_mse_abs": float(hdr_abs_mse),
    }


def capture_inputs(model, calibration_data, device):
    blocks = model.model.layers
    blocks[0] = InputCollector(blocks[0]).to(device)
    model.get_input_embeddings().to(device)
    for sample in calibration_data:
        try:
            with torch.no_grad():
                model(sample.to(device))
        except ForwardInterrupt:
            pass
    input_args, input_kwargs = blocks[0].input_args, blocks[0].input_kwargs
    blocks[0] = blocks[0].module.cpu()
    model.get_input_embeddings().cpu()
    return input_args, input_kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--dataset_name_or_path", default="c4")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--num_sequences", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_random_permutations", type=int, default=5,
        help="Number of independent random layouts used for the Random control.",
    )
    parser.add_argument(
        "--stagger_quantile", type=float, default=0.9375,
        help="Per-row quantile used to identify outliers for staggering.",
    )
    parser.add_argument(
        "--permu_alpha", type=float, default=0.5,
        help="Activation/weight balance for joint second-moment ordering.",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("dacc_mechanism_results.json"))
    parser.add_argument("--skip_bitr", action="store_true")
    args = parser.parse_args()
    if args.num_random_permutations < 1:
        parser.error("--num_random_permutations must be at least 1")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    calibration_data = get_data(
        args.dataset_name_or_path, tokenizer, args.sequence_length,
        args.num_sequences, seed=args.seed)
    model.config.use_cache = False
    model.requires_grad_(False)
    input_args, input_kwargs = capture_inputs(model, calibration_data, device)

    target_layers = {0, 15, 31}
    matrix_names = ["qkv", "o", "gate_up", "down"]
    records = []
    blocks = model.model.layers

    for layer_idx, block in enumerate(blocks):
        block = block.to(device)
        if layer_idx not in target_layers:
            for i in range(len(input_args)):
                with torch.no_grad():
                    out = block(*to_device(input_args[i], device),
                                **to_device(input_kwargs[i], device))
                input_args[i] = (maybe_first(out).cpu(),) + input_args[i][1:]
            block.cpu()
            continue

        print(f"Processing layer {layer_idx}...")
        block_copy = copy.deepcopy(block).to(device)
        caches = {}

        def hook_factory(name):
            def hook(_, inputs, __):
                caches.setdefault(name, []).append(
                    inputs[0].detach().float().view(-1, inputs[0].shape[-1]))
            return hook

        hooks = [
            block_copy.self_attn.q_proj.register_forward_hook(hook_factory("qkv")),
            block_copy.self_attn.o_proj.register_forward_hook(hook_factory("o")),
            block_copy.mlp.gate_proj.register_forward_hook(hook_factory("gate_up")),
            block_copy.mlp.down_proj.register_forward_hook(hook_factory("down")),
        ]
        for i in range(len(input_args)):
            with torch.no_grad():
                block_copy(*to_device(input_args[i], device),
                           **to_device(input_kwargs[i], device))
        for hook in hooks:
            hook.remove()
        del block_copy
        torch.cuda.empty_cache()

        for name in matrix_names:
            if name not in caches:
                continue
            x_all = torch.cat(caches[name], dim=0).to(device)
            if x_all.shape[0] > args.max_tokens:
                indices = torch.linspace(
                    0, x_all.shape[0] - 1, args.max_tokens,
                    dtype=torch.long, device=device)
                x = x_all[indices]
            else:
                x = x_all
            channels = x.shape[1]
            if channels % args.group_size:
                raise ValueError(f"{name} has {channels} channels, not divisible by B")
            weight = get_combined_weight(block, name).to(device)
            # Production DACC uses squared input-column weight norms as its
            # weight-aware importance when alpha=2.
            x_abs_t = x.abs().T.float()
            identity = build_layout_permutation(
                x, weight, name, "identity", args.group_size, model)
            random_perms = [build_layout_permutation(
                x, weight, name, "random", args.group_size, model)
                for _ in range(args.num_random_permutations)]
            magnitude = build_layout_permutation(
                x, weight, name, "magnitude", args.group_size, model)
            outlier_stagger = build_layout_permutation(
                x, weight, name, "outlier_stagger", args.group_size, model,
                stagger_quantile=args.stagger_quantile)
            joint_second_moment = build_layout_permutation(
                x, weight, name, "joint_second_moment", args.group_size, model,
                permu_alpha=args.permu_alpha)
            dacc = build_layout_permutation(
                x, weight, name, "dacc", args.group_size, model)
            layouts = {
                "Original": (identity, None),
                "Magnitude": (magnitude, None),
                "Outlier-Stagger": (outlier_stagger, None),
                "Joint-2nd-Moment": (joint_second_moment, None),
                "DACC": (dacc, None),
            }
            for random_idx, random_perm in enumerate(random_perms, start=1):
                layouts[f"Random-{random_idx}"] = (random_perm, None)
            if not args.skip_bitr:
                print(f"  {name}: running DACC+BITR coordinate search")
                scales = optimize_channel_scales_coordinate_descent(
                    x[:, dacc].float(), weight[:, dacc].float(),
                    weight_mse_ratio=1.0, group_size=args.group_size,
                    top_k=5, num_rounds=3)
                layouts["DACC+BITR"] = (dacc, scales)
            for method, (perm, scale) in layouts.items():
                records.append({"layer": layer_idx, "matrix": name,
                                "method": method,
                                **activation_stats(x, perm, args.group_size, scale)})
            del x_all, x, x_abs_t, weight, random_perms
            torch.cuda.empty_cache()

        del caches
        gc.collect()
        for i in range(len(input_args)):
            with torch.no_grad():
                out = block(*to_device(input_args[i], device),
                            **to_device(input_kwargs[i], device))
            input_args[i] = (maybe_first(out).cpu(),) + input_args[i][1:]
        block.cpu()
        torch.cuda.empty_cache()

    methods = [
        "Original", "Random", "Magnitude", "Outlier-Stagger",
        "Joint-2nd-Moment", "DACC",
    ]
    if not args.skip_bitr:
        methods.append("DACC+BITR")
    summary = {}
    summary_std = {}
    keys = ["mse", "ldr_pct", "mdr_pct", "hdr_pct", "overflow_pct",
        "mdr_hdr_mse_pct", "hdr_mse_pct", "mdr_hdr_mse_abs",
        "hdr_mse_abs"]
    for method in methods:
        if method == "Random":
            # Treat each Random-i layout as one complete replicate, then
            # compute mean/std across the requested independent replicates.
            random_runs = []
            for random_idx in range(1, args.num_random_permutations + 1):
                run_rows = [r for r in records
                            if r["method"] == f"Random-{random_idx}"]
                random_runs.append({
                    key: float(np.mean([r[key] for r in run_rows]))
                    for key in keys
                })
            summary[method] = {
                key: float(np.mean([run[key] for run in random_runs]))
                for key in keys
            }
            summary_std[method] = {
                key: float(np.std([run[key] for run in random_runs], ddof=1))
                if len(random_runs) > 1 else 0.0
                for key in keys
            }
            continue
        else:
            rows = [r for r in records if r["method"] == method]
        summary[method] = {key: float(np.mean([r[key] for r in rows]))
                           for key in keys}
    output = {
        "config": {"model": args.model_name_or_path,
                   "dataset": args.dataset_name_or_path,
                   "seed": args.seed, "layers": sorted(target_layers),
                   "max_tokens_per_matrix": args.max_tokens,
                   "group_size": args.group_size,
                   "stagger_quantile": args.stagger_quantile,
                   "permu_alpha": args.permu_alpha,
                   "bitr": not args.skip_bitr},
        "records": records, "summary": summary,
        "summary_std": summary_std,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.output}")
    print("method,mse,mdr_pct,hdr_pct,mdr_hdr_mse_pct,hdr_mse_pct,"
          "mdr_hdr_mse_abs,hdr_mse_abs")
    for method in methods:
        row = summary[method]
        print(f"{method},{row['mse']:.6e},{row['mdr_pct']:.3f},"
              f"{row['hdr_pct']:.3f},{row['mdr_hdr_mse_pct']:.3f},"
              f"{row['hdr_mse_pct']:.3f},{row['mdr_hdr_mse_abs']:.6e},"
              f"{row['hdr_mse_abs']:.6e}")

    random_std = summary_std.get("Random", {})
    print("Random mean+/-std (across independent permutations),"
          "mse,mdr_hdr_mse_abs,hdr_mse_abs")
    row = summary["Random"]
    print(f"Random,{row['mse']:.6e}+/-{random_std.get('mse', 0.0):.6e},"
          f"{row['mdr_hdr_mse_abs']:.6e}+/-"
          f"{random_std.get('mdr_hdr_mse_abs', 0.0):.6e},"
          f"{row['hdr_mse_abs']:.6e}+/-"
          f"{random_std.get('hdr_mse_abs', 0.0):.6e}")

    # A compact copy-paste view in the units used by the paper's analysis
    # table (10^{-4}); the JSON file retains the unscaled values above.
    print("method,mdr_hdr_mse_x1e4,hdr_mse_x1e4")
    for method in methods:
        row = summary[method]
        print(f"{method},{row['mdr_hdr_mse_abs'] * 1e4:.4f},"
              f"{row['hdr_mse_abs'] * 1e4:.4f}")
    print("Random mean+/-std x1e4,mdr_hdr_mse,hdr_mse")
    random_row = summary["Random"]
    print(f"Random,{random_row['mdr_hdr_mse_abs'] * 1e4:.4f}+/-"
          f"{random_std.get('mdr_hdr_mse_abs', 0.0) * 1e4:.4f},"
          f"{random_row['hdr_mse_abs'] * 1e4:.4f}+/-"
          f"{random_std.get('hdr_mse_abs', 0.0) * 1e4:.4f}")


if __name__ == "__main__":
    main()
