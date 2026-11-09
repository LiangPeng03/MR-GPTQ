"""Cross-model activation and linear-output MSE for DACC and DACC+BITR.

This experiment is the compact cross-model companion to ``test_K+L.py``.
It evaluates production DACC and DACC followed by BITR. Activation MSE uses
the same NVFP4 reconstruction as ``test_K+L.py``. Linear-output MSE compares
``X @ W.T`` with the product of the quantized activation and weight; weights
use NADA's NVFP4 MSE_N observer and E4M3 block scales. This local matrix
metric does not include GPTQ error compensation or full-model propagation.
By default, the script uses the first, middle, and last transformer blocks.
For a 32-layer model this is
exactly layers 0, 15, and 31, matching the representative-layer protocol used
by ``test_K+L.py``.
"""

import argparse
import copy
import csv
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.quantization.gics import optimize_channel_scales_coordinate_descent
from src.quantization.quant_ops import FP4_E2M1_MAX, FP8_E4M3_MAX, cast_to_fp4
from src.quantization.quantizer import Quantizer, get_reciprocal
from src.utils.data_utils import get_data


DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen3-1.7B",
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen3-8B",
]


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
        return tuple(to_device(item, device) for item in value)
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def maybe_first(value):
    return value[0] if isinstance(value, tuple) else value


def get_combined_weight(block, name):
    if name == "qkv":
        weight = torch.cat(
            [
                block.self_attn.q_proj.weight,
                block.self_attn.k_proj.weight,
                block.self_attn.v_proj.weight,
            ],
            dim=0,
        )
    elif name == "o":
        weight = block.self_attn.o_proj.weight
    elif name == "gate_up":
        weight = torch.cat(
            [block.mlp.gate_proj.weight, block.mlp.up_proj.weight], dim=0
        )
    elif name == "down":
        weight = block.mlp.down_proj.weight
    else:
        raise ValueError(f"Unsupported matrix name: {name}")
    return weight.detach().float()


def compute_global_scale(x):
    tensor_max = x.abs().max().to(torch.float32).view(1)
    return (
        FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(tensor_max)
    ).to(x.device)


def scale_to_e4m3(raw_scale, global_scale):
    return (
        (raw_scale * global_scale)
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        .mul(get_reciprocal(global_scale))
    )


def proxy_distances(all_channels, reference, channel_weights=None, chunk_size=256):
    """Pairwise proxy used by the production cumulative-distance DACC."""
    grid = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6],
        device=all_channels.device,
        dtype=torch.float32,
    )
    result = torch.zeros(all_channels.shape[0], device=all_channels.device)
    reference = reference.unsqueeze(0)
    for start in range(0, all_channels.shape[0], chunk_size):
        chunk = all_channels[start : start + chunk_size]
        p_max = torch.maximum(chunk, reference)
        p_min = torch.minimum(chunk, reference)
        scale = (p_max / 6.0).clamp(min=1e-10)
        candidates = scale.unsqueeze(-1) * grid
        nearest = candidates.gather(
            -1,
            (p_min.unsqueeze(-1) - candidates)
            .abs()
            .argmin(-1, keepdim=True),
        ).squeeze(-1)
        distance = ((p_min - nearest) ** 2).sum(dim=1)
        if channel_weights is not None:
            distance = distance * channel_weights[start : start + chunk.size(0)]
        result[start : start + chunk.size(0)] = distance
    return result


def compute_dacc_perm(x_abs_t, group_size=16, channel_weights=None):
    """Production DACC: cumulative seed selection followed by greedy fill."""
    channels = x_abs_t.shape[0]
    if channels % group_size:
        raise ValueError(f"channels={channels} is not divisible by {group_size}")
    num_groups = channels // group_size
    if channel_weights is None:
        channel_weights = torch.ones(channels, device=x_abs_t.device)
    importance = x_abs_t.sum(dim=1) * channel_weights

    first = importance.argmax().item()
    seeds = [first]
    cumulative = proxy_distances(x_abs_t, x_abs_t[first], channel_weights)
    cumulative[first] = -float("inf")
    while len(seeds) < num_groups:
        seed = cumulative.argmax().item()
        seeds.append(seed)
        cumulative[seed] = -float("inf")
        cumulative += proxy_distances(
            x_abs_t, x_abs_t[seed], channel_weights
        )
        cumulative[seed] = -float("inf")

    groups = [[seed] for seed in seeds]
    envelopes = x_abs_t[seeds].clone()
    sizes = torch.ones(num_groups, device=x_abs_t.device, dtype=torch.long)
    chosen = set(seeds)
    remaining = [channel for channel in range(channels) if channel not in chosen]
    remaining.sort(key=lambda channel: float(importance[channel]), reverse=True)
    for channel in remaining:
        distances = proxy_distances(envelopes, x_abs_t[channel])
        distances[sizes >= group_size] = float("inf")
        group = distances.argmin().item()
        groups[group].append(channel)
        envelopes[group] = torch.maximum(envelopes[group], x_abs_t[channel])
        sizes[group] += 1
    return torch.tensor(
        [channel for group in groups for channel in group],
        device=x_abs_t.device,
        dtype=torch.long,
    )


def build_dacc_permutation(x, weight, name, group_size, model):
    """Build DACC in the same projection scopes as the production quantizer."""
    channels = x.shape[1]
    if name in {"o", "down"}:
        scope = model.config.hidden_size // model.config.num_attention_heads
    else:
        scope = channels
    if scope % group_size:
        raise ValueError(f"scope={scope} is not divisible by B={group_size}")

    result = []
    x_abs_t = x.abs().T.float()
    for start in range(0, channels, scope):
        stop = min(start + scope, channels)
        local_x = x_abs_t[start:stop]
        local_weight = weight[:, start:stop].float()
        channel_weights = local_weight.pow(2).sum(dim=0)
        local_perm = compute_dacc_perm(local_x, group_size, channel_weights)
        result.append(local_perm + start)
    return torch.cat(result)


def quantize_activations(x, permutation, group_size=16, channel_scale=None):
    """Return layout-order X, scaled quantized X, and original-unit act MSE."""
    x_layout = x[:, permutation].float().contiguous()
    if channel_scale is None:
        channel_scale = torch.ones(x_layout.shape[1], device=x.device)
    channel_scale = channel_scale.to(x.device).float()
    x_work = x_layout * channel_scale.unsqueeze(0)
    groups = x_work.view(-1, group_size)
    block_max = groups.abs().amax(dim=1, keepdim=True)
    raw_scale = (block_max / FP4_E2M1_MAX).clamp(min=1e-10)
    raw_scale = torch.where(
        block_max == 0, torch.ones_like(raw_scale), raw_scale
    )
    global_scale = compute_global_scale(x_work)
    block_scale = scale_to_e4m3(raw_scale, global_scale)
    quantized = cast_to_fp4(groups / block_scale)
    scale_groups = (
        channel_scale.unsqueeze(0)
        .expand(x_layout.shape[0], -1)
        .contiguous()
        .view(-1, group_size)
    )
    quantized_work = (quantized * block_scale).view_as(x_layout)
    reconstructed = quantized_work.view(-1, group_size) / scale_groups
    original = groups / scale_groups
    act_mse = float(((original - reconstructed) ** 2).mean().item())
    return x_layout, quantized_work, act_mse


def quantize_weights(weight_work, group_size):
    """Use NADA's weight observer for a local NVFP4 matrix-output probe."""
    quantizer = Quantizer(
        bits=4,
        format="nvfp",
        granularity="group",
        observer="mse_n",
        group_size=group_size,
        scale_precision="e4m3",
    )
    scales, zeros = quantizer.get_quantization_params(weight_work)
    return quantizer(weight_work, scales, zeros)


@torch.no_grad()
def evaluate_mse(x, weight, permutation, group_size, channel_scale=None,
                 output_chunk_size=128):
    """Measure activation and W4A4 linear-output MSE against FP32 XW.T."""
    x_layout, quantized_work, act_mse = quantize_activations(
        x, permutation, group_size, channel_scale
    )
    if channel_scale is None:
        weight_work = weight[:, permutation].float().contiguous()
    else:
        weight_work = (
            weight[:, permutation].float()
            / channel_scale.to(weight.device).float().unsqueeze(0)
        ).contiguous()
    quantized_weight = quantize_weights(weight_work, group_size)

    squared_error = 0.0
    output_elements = 0
    for start in range(0, x_layout.shape[0], output_chunk_size):
        stop = start + output_chunk_size
        reference = torch.nn.functional.linear(x[start:stop].float(), weight)
        reconstructed = torch.nn.functional.linear(
            quantized_work[start:stop], quantized_weight
        )
        squared_error += float((reference - reconstructed).square().sum().item())
        output_elements += reference.numel()
    output_mse = squared_error / output_elements
    return act_mse, output_mse


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
    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module.cpu()
    model.get_input_embeddings().cpu()
    torch.cuda.empty_cache()
    return input_args, input_kwargs


def representative_layers(num_layers):
    """First/middle/last layers; 32 layers maps to 0, 15, and 31."""
    if num_layers < 1:
        raise ValueError("The model has no transformer layers")
    return sorted({0, (num_layers - 1) // 2, num_layers - 1})


def parse_layers(value, num_layers):
    if value is None:
        return representative_layers(num_layers)
    layers = sorted({int(item.strip()) for item in value.split(",")})
    invalid = [layer for layer in layers if not 0 <= layer < num_layers]
    if invalid:
        raise ValueError(
            f"Invalid layer indices {invalid}; model has {num_layers} layers"
        )
    if not layers:
        raise ValueError("--layers must contain at least one layer index")
    return layers


def model_label(model_name_or_path):
    normalized = model_name_or_path.rstrip("/\\")
    basename = re.split(r"[/\\]", normalized)[-1]
    return re.sub(r"[^a-z0-9._-]+", "_", basename.lower()).strip("_")


def run_model(args, model_name_or_path, output_path, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 78)
    print(f"Loading {model_name_or_path} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    calibration_data = get_data(
        args.dataset_name_or_path,
        tokenizer,
        args.sequence_length,
        args.num_sequences,
        seed=args.seed,
    )
    model.config.use_cache = False
    model.requires_grad_(False)

    blocks = model.model.layers
    target_layers = parse_layers(args.layers, len(blocks))
    target_layer_set = set(target_layers)
    print(f"Representative layers: {target_layers}")
    input_args, input_kwargs = capture_inputs(model, calibration_data, device)

    matrix_names = ["qkv", "o", "gate_up", "down"]
    records = []

    for layer_idx, block in enumerate(blocks):
        block = block.to(device)
        if layer_idx not in target_layer_set:
            for sample_idx in range(len(input_args)):
                with torch.no_grad():
                    output = block(
                        *to_device(input_args[sample_idx], device),
                        **to_device(input_kwargs[sample_idx], device),
                    )
                input_args[sample_idx] = (
                    maybe_first(output).cpu(),
                ) + input_args[sample_idx][1:]
            block.cpu()
            torch.cuda.empty_cache()
            continue

        print(f"Processing layer {layer_idx}...")
        block_copy = copy.deepcopy(block).to(device)
        caches = {}

        def hook_factory(name):
            def hook(_, inputs, __):
                values = inputs[0].detach().float()
                caches.setdefault(name, []).append(
                    values.view(-1, values.shape[-1])
                )

            return hook

        hooks = [
            block_copy.self_attn.q_proj.register_forward_hook(
                hook_factory("qkv")
            ),
            block_copy.self_attn.o_proj.register_forward_hook(
                hook_factory("o")
            ),
            block_copy.mlp.gate_proj.register_forward_hook(
                hook_factory("gate_up")
            ),
            block_copy.mlp.down_proj.register_forward_hook(
                hook_factory("down")
            ),
        ]
        for sample_idx in range(len(input_args)):
            with torch.no_grad():
                block_copy(
                    *to_device(input_args[sample_idx], device),
                    **to_device(input_kwargs[sample_idx], device),
                )
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
                    0,
                    x_all.shape[0] - 1,
                    args.max_tokens,
                    dtype=torch.long,
                    device=device,
                )
                x = x_all[indices]
            else:
                x = x_all
            channels = x.shape[1]
            if channels % args.group_size:
                raise ValueError(
                    f"{name} has {channels} channels, not divisible by "
                    f"group size {args.group_size}"
                )

            weight = get_combined_weight(block, name).to(device)
            dacc = build_dacc_permutation(
                x, weight, name, args.group_size, model
            )
            dacc_act_mse, dacc_output_mse = evaluate_mse(
                x, weight, dacc, args.group_size,
                output_chunk_size=args.output_chunk_size,
            )
            print(f"  {name}: running DACC+BITR coordinate search")
            scales = optimize_channel_scales_coordinate_descent(
                x[:, dacc].float(),
                weight[:, dacc].float(),
                weight_mse_ratio=args.bitr_weight_mse_ratio,
                group_size=args.group_size,
                top_k=args.bitr_top_k,
                num_rounds=args.bitr_num_rounds,
            )
            bitr_act_mse, bitr_output_mse = evaluate_mse(
                x, weight, dacc, args.group_size, scales,
                output_chunk_size=args.output_chunk_size,
            )
            records.extend(
                [
                    {
                        "layer": layer_idx,
                        "matrix": name,
                        "method": "DACC",
                        "activation_mse": dacc_act_mse,
                        "output_mse": dacc_output_mse,
                    },
                    {
                        "layer": layer_idx,
                        "matrix": name,
                        "method": "DACC+BITR",
                        "activation_mse": bitr_act_mse,
                        "output_mse": bitr_output_mse,
                    },
                ]
            )
            print(
                f"  {name}: activation MSE DACC={dacc_act_mse:.6e}, "
                f"DACC+BITR={bitr_act_mse:.6e}; "
                f"output MSE DACC={dacc_output_mse:.6e}, "
                f"DACC+BITR={bitr_output_mse:.6e}"
            )
            del x_all, x, weight, dacc, scales
            torch.cuda.empty_cache()

        del caches
        gc.collect()
        for sample_idx in range(len(input_args)):
            with torch.no_grad():
                output = block(
                    *to_device(input_args[sample_idx], device),
                    **to_device(input_kwargs[sample_idx], device),
                )
            input_args[sample_idx] = (
                maybe_first(output).cpu(),
            ) + input_args[sample_idx][1:]
        block.cpu()
        torch.cuda.empty_cache()

    summary = {}
    for method in ("DACC", "DACC+BITR"):
        rows = [row for row in records if row["method"] == method]
        if not rows:
            raise RuntimeError(f"No records were collected for {method}")
        summary[method] = {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("activation_mse", "output_mse")
        }

    reductions = {}
    for metric in ("activation_mse", "output_mse"):
        dacc_value = summary["DACC"][metric]
        bitr_value = summary["DACC+BITR"][metric]
        reductions[metric] = (
            (dacc_value - bitr_value) / dacc_value * 100.0
            if dacc_value else 0.0
        )
    output = {
        "config": {
            "model": model_name_or_path,
            "dataset": args.dataset_name_or_path,
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "num_sequences": args.num_sequences,
            "layers": target_layers,
            "matrices": matrix_names,
            "max_tokens_per_matrix": args.max_tokens,
            "group_size": args.group_size,
            "bitr_top_k": args.bitr_top_k,
            "bitr_num_rounds": args.bitr_num_rounds,
            "bitr_weight_mse_ratio": args.bitr_weight_mse_ratio,
            "weight_observer": "mse_n",
            "output_mse_reference": "FP32 X @ W.T",
            "output_mse_quantization": "local NVFP4 W4A4, no GPTQ compensation",
            "output_chunk_size": args.output_chunk_size,
        },
        "records": records,
        "summary": summary,
        "reduction_pct": reductions,
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    print(f"Saved {output_path}")
    print("method,activation_mse,output_mse")
    for method in ("DACC", "DACC+BITR"):
        print(
            f"{method},{summary[method]['activation_mse']:.8e},"
            f"{summary[method]['output_mse']:.8e}"
        )
    print(
        f"BITR relative MSE reduction: activation="
        f"{reductions['activation_mse']:.3f}%, "
        f"output={reductions['output_mse']:.3f}%"
    )
    print(
        "[RESULT_SUMMARY] "
        + json.dumps(
            {
                "model": model_name_or_path,
                "dacc_activation_mse": summary["DACC"]["activation_mse"],
                "dacc_bitr_activation_mse": summary["DACC+BITR"]["activation_mse"],
                "dacc_output_mse": summary["DACC"]["output_mse"],
                "dacc_bitr_output_mse": summary["DACC+BITR"]["output_mse"],
                "activation_reduction_pct": reductions["activation_mse"],
                "output_reduction_pct": reductions["output_mse"],
            }
        )
    )
    del input_args, input_kwargs, calibration_data, tokenizer, blocks, model
    if "block" in locals():
        del block
    gc.collect()
    torch.cuda.empty_cache()
    return output


def write_summary_csv(path, results):
    with path.open("w", newline="") as handle:
        fieldnames = [
            "model",
            "layers",
            "dacc_activation_mse",
            "dacc_bitr_activation_mse",
            "activation_reduction_pct",
            "dacc_output_mse",
            "dacc_bitr_output_mse",
            "output_reduction_pct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result["config"]["model"],
                    "layers": ";".join(
                        str(layer) for layer in result["config"]["layers"]
                    ),
                    "dacc_activation_mse": result["summary"]["DACC"]["activation_mse"],
                    "dacc_bitr_activation_mse": result["summary"]["DACC+BITR"]["activation_mse"],
                    "activation_reduction_pct": result["reduction_pct"]["activation_mse"],
                    "dacc_output_mse": result["summary"]["DACC"]["output_mse"],
                    "dacc_bitr_output_mse": result["summary"]["DACC+BITR"]["output_mse"],
                    "output_reduction_pct": result["reduction_pct"]["output_mse"],
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        action="append",
        dest="models",
        help=(
            "Model ID or local path. Repeat this option to run multiple models. "
            "When omitted, the four main-result 1B-8B models are run."
        ),
    )
    parser.add_argument("--dataset_name_or_path", default="c4")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--num_sequences", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices. Default: first,middle,last.",
    )
    parser.add_argument("--bitr_top_k", type=int, default=5)
    parser.add_argument("--bitr_num_rounds", type=int, default=3)
    parser.add_argument("--bitr_weight_mse_ratio", type=float, default=1.0)
    parser.add_argument("--output_chunk_size", type=int, default=128)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("dacc_bitr_mse_results")
    )
    args = parser.parse_args()

    if args.group_size < 1:
        parser.error("--group_size must be positive")
    if args.max_tokens < 1:
        parser.error("--max_tokens must be positive")
    if args.bitr_top_k < 1:
        parser.error("--bitr_top_k must be positive")
    if args.bitr_num_rounds < 1:
        parser.error("--bitr_num_rounds must be positive")
    if args.output_chunk_size < 1:
        parser.error("--output_chunk_size must be positive")
    if not torch.cuda.is_available():
        parser.error("This experiment requires a CUDA GPU")

    models = args.models or DEFAULT_MODELS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name_or_path in models:
        output_path = args.output_dir / f"{model_label(model_name_or_path)}.json"
        results.append(
            run_model(args, model_name_or_path, output_path, device="cuda")
        )

    summary_path = args.output_dir / "summary.csv"
    write_summary_csv(summary_path, results)
    print("=" * 78)
    print(f"All {len(results)} model(s) completed. Summary: {summary_path}")
    print(
        "model,dacc_activation_mse,dacc_bitr_activation_mse,"
        "activation_reduction_pct,dacc_output_mse,dacc_bitr_output_mse,"
        "output_reduction_pct"
    )
    for result in results:
        print(
            f"{result['config']['model']},"
            f"{result['summary']['DACC']['activation_mse']:.8e},"
            f"{result['summary']['DACC+BITR']['activation_mse']:.8e},"
            f"{result['reduction_pct']['activation_mse']:.3f},"
            f"{result['summary']['DACC']['output_mse']:.8e},"
            f"{result['summary']['DACC+BITR']['output_mse']:.8e},"
            f"{result['reduction_pct']['output_mse']:.3f}"
        )


if __name__ == "__main__":
    main()
