#!/usr/bin/env python3
"""Compare one exported FPQuant linear layer with its decoded BF16 reference."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.model_executor.layers.quantization.fp_quant import quantized_forward


FP4_GRID = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_BITPACKING_PERM = [15, 14, 13, 12, 11, 10, 9, 8,
                       0, 1, 2, 3, 4, 5, 6, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--layer",
        default="model.layers.0.self_attn.o_proj",
    )
    parser.add_argument(
        "--source-layers",
        nargs="+",
        default=None,
        help=(
            "Optional checkpoint layers to concatenate along the output "
            "dimension, in vLLM fused-layer order (for example q/k/v or "
            "gate/up). The first layer supplies the shared input transform."
        ),
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 19, 128])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    magnitude = x.abs()
    rounded = torch.where(
        magnitude > 5.0,
        6.0,
        torch.where(
            magnitude >= 3.5,
            4.0,
            torch.where(
                magnitude >= 1.75,
                torch.round(magnitude),
                torch.round(magnitude * 2.0) * 0.5,
            ),
        ),
    )
    return rounded * sign


def quantize_activation(
    x: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    shape = x.shape
    grouped = x.reshape(x.shape[0], x.shape[1] // 16, 16)
    abs_x = grouped.abs()
    scale = abs_x.amax(dim=-1, keepdim=True) / 6
    safe_scale = scale.clone()
    safe_scale[safe_scale == 0] = 1
    grid = cast_to_fp4(abs_x / safe_scale).abs()
    numerator = (abs_x * grid).sum(dim=-1, keepdim=True)
    denominator = (grid * grid).sum(dim=-1, keepdim=True)
    denominator_zero = denominator == 0
    safe_denominator = denominator.clone()
    safe_denominator[denominator_zero] = 1
    scale = torch.where(
        denominator_zero,
        scale,
        numerator / safe_denominator,
    )
    scale = (
        (scale * global_scale)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        .div(global_scale)
        .to(x.dtype)
    )
    scale[scale == 0] = 1
    return (cast_to_fp4(grouped / scale) * scale).reshape(shape)


def main() -> None:
    args = parse_args()
    model = Path(args.model)
    index = json.loads(
        (model / "model.safetensors.index.json").read_text()
    )["weight_map"]

    def load(key: str) -> torch.Tensor:
        with safe_open(model / index[key], framework="pt", device="cpu") as reader:
            return reader.get_tensor(key).cuda()

    source_layers = args.source_layers or [args.layer]
    prefix = source_layers[0]
    qweight = torch.cat(
        [load(layer + ".qweight") for layer in source_layers], dim=0
    )
    weight_scales = torch.cat(
        [load(layer + ".scales") for layer in source_layers], dim=0
    )

    weight_global_scales = [
        load(layer + ".weight_global_scale").float()
        for layer in source_layers
    ]
    act_global_scales = [
        load(layer + ".act_global_scale").float()
        for layer in source_layers
    ]
    if not all(
        torch.equal(weight_global_scales[0], scale)
        for scale in weight_global_scales[1:]
    ):
        raise ValueError("Fused source layers have different weight global scales")
    if not all(
        torch.equal(act_global_scales[0], scale)
        for scale in act_global_scales[1:]
    ):
        raise ValueError("Fused source layers have different activation global scales")

    weight_global_scale = weight_global_scales[0]
    act_global_scale = act_global_scales[0]
    input_perm = load(prefix + ".input_perm").long()
    input_rescale = load(prefix + ".input_rescale").to(torch.bfloat16)

    output_features = qweight.shape[0]
    input_features = qweight.shape[1] * 2
    codebook = torch.empty(16, dtype=torch.float32, device="cuda")
    codebook[torch.tensor(FP4_BITPACKING_PERM, device="cuda")] = torch.tensor(
        FP4_GRID, dtype=torch.float32, device="cuda"
    )
    decoded_weight = codebook[
        torch.stack((qweight & 0xF, qweight >> 4), dim=-1)
        .reshape(output_features, input_features)
        .long()
    ]
    decoded_weight_scales = (
        weight_scales.view(torch.float8_e4m3fn).float() / weight_global_scale
    )
    decoded_weight = (
        decoded_weight
        * decoded_weight_scales.repeat_interleave(16, dim=1)
    ).to(torch.bfloat16)
    identity = torch.eye(16, dtype=torch.bfloat16, device="cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    print("LAYER:", args.layer)
    print("SOURCE_LAYERS:", source_layers)
    print("QWEIGHT_SHAPE:", tuple(qweight.shape))
    print("WEIGHT_SCALES_SHAPE:", tuple(weight_scales.shape))
    print("WEIGHT_GLOBAL_SCALE:", weight_global_scale.item())
    print("ACT_GLOBAL_SCALE:", act_global_scale.item())

    for rows in args.rows:
        x = torch.randn(
            rows,
            input_features,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        x = x.index_select(-1, input_perm) * input_rescale
        reference_x = quantize_activation(x, act_global_scale)
        reference = F.linear(reference_x, decoded_weight).float()
        actual = quantized_forward(
            x,
            qweight,
            weight_scales,
            weight_global_scale,
            act_global_scale,
            None,
            identity,
            "lss",
            "nvfp4",
        ).float()
        error = actual - reference
        denominator = reference.square().mean().clamp_min(1e-30)
        cosine = F.cosine_similarity(
            actual.reshape(1, -1), reference.reshape(1, -1)
        ).item()
        print(
            f"LINEAR_PARITY rows={rows} "
            f"max_abs={error.abs().max().item():.8g} "
            f"mean_abs={error.abs().mean().item():.8g} "
            f"relative_mse={(error.square().mean() / denominator).item():.8g} "
            f"cosine={cosine:.10f} "
            f"actual_mean_abs={actual.abs().mean().item():.8g} "
            f"reference_mean_abs={reference.abs().mean().item():.8g}",
            flush=True,
        )
    print("FP4_LINEAR_PARITY_DIAGNOSTIC_COMPLETE")


if __name__ == "__main__":
    main()
