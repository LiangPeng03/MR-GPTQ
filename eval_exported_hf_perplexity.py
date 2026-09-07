#!/usr/bin/env python3
"""Evaluate the mathematical contents of an exported NVFP4 checkpoint in HF."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics.perplexity import compute_perplexity
from src.quantization.quant_ops import FP4_BITPACKING_PERM, FP4_GRID, cast_to_fp4
from src.utils.data_utils import get_c4_eval, get_wikitext2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--exported-model", required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--dataset", choices=("wikitext2", "c4", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--capture-dir", default=None)
    parser.add_argument(
        "--capture-layers",
        type=int,
        nargs="+",
        default=[0, 1, 3, 7, 15, 23, 31, 35],
    )
    return parser.parse_args()


def make_codebook(device: torch.device) -> torch.Tensor:
    codebook = torch.empty(16, dtype=torch.float32, device=device)
    codebook[torch.tensor(FP4_BITPACKING_PERM, device=device)] = torch.tensor(
        FP4_GRID, dtype=torch.float32, device=device
    )
    return codebook


def activation_quantize(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    method: str,
) -> torch.Tensor:
    shape = x.shape
    grouped = x.reshape(*shape[:-1], shape[-1] // 16, 16)
    abs_x = grouped.abs()
    scale = abs_x.amax(dim=-1, keepdim=True) / 6
    if method == "lss":
        safe_scale = scale.clone()
        safe_scale[safe_scale == 0] = 1
        grid = cast_to_fp4(abs_x / safe_scale).abs()
        numerator = (abs_x * grid).sum(dim=-1, keepdim=True)
        denominator = (grid * grid).sum(dim=-1, keepdim=True)
        denominator_zero = denominator == 0
        safe_denominator = denominator.clone()
        safe_denominator[denominator_zero] = 1
        lss_scale = numerator / safe_denominator
        scale = torch.where(denominator_zero, scale, lss_scale)
    elif method != "abs_max":
        raise ValueError(f"Unsupported activation method: {method}")

    scale = (
        (scale * global_scale)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        .div(global_scale)
        .to(x.dtype)
    )
    scale[scale == 0] = 1
    quantized = cast_to_fp4(grouped / scale)
    return (quantized * scale).reshape(shape)


def transform_owner(name: str) -> str:
    if name.endswith(("self_attn.k_proj", "self_attn.v_proj")):
        return name.rsplit(".", 1)[0] + ".q_proj"
    if name.endswith("mlp.up_proj"):
        return name.rsplit(".", 1)[0] + ".gate_proj"
    return name


def main() -> None:
    args = parse_args()
    exported = Path(args.exported_model)
    config = json.loads((exported / "config.json").read_text())
    quant_config = config["quantization_config"]
    method = quant_config["forward_method"]
    if quant_config["forward_dtype"] != "nvfp4":
        raise ValueError("Only NVFP4 checkpoints are supported")
    if quant_config["hadamard_group_size"] != 16:
        raise ValueError("This diagnostic expects group size 16")
    if not quant_config.get("lss_identity_transform", False) and method == "lss":
        raise ValueError("LSS diagnostic currently supports identity transforms only")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    codebook = make_codebook(device)

    if args.capture_dir:
        capture_dir = Path(args.capture_dir)
        capture_dir.mkdir(parents=True, exist_ok=True)

        def save_once(name, value):
            path = capture_dir / f"{name}.pt"
            if not path.exists():
                if isinstance(value, (tuple, list)):
                    value = value[0]
                torch.save(value.detach().to("cpu"), path)

        model.model.embed_tokens.register_forward_hook(
            lambda _module, _inputs, output: save_once("embedding", output)
        )
        for layer_idx in args.capture_layers:
            model.model.layers[layer_idx].register_forward_hook(
                lambda _module, _inputs, output, layer_idx=layer_idx:
                    save_once(f"layer_{layer_idx:02d}", output)
            )

        layer0 = model.model.layers[0]
        layer0.input_layernorm.register_forward_hook(
            lambda _module, _inputs, output: save_once("l0_input_norm", output)
        )
        layer0.self_attn.q_proj.register_forward_hook(
            lambda _module, _inputs, output: save_once("l0_q", output)
        )
        layer0.self_attn.k_proj.register_forward_hook(
            lambda _module, _inputs, output: save_once("l0_k", output)
        )
        layer0.self_attn.v_proj.register_forward_hook(
            lambda _module, _inputs, output: save_once("l0_v", output)
        )
        layer0.self_attn.q_norm.register_forward_hook(
            lambda _module, _inputs, output: save_once("l0_q_norm", output)
        )
        layer0.self_attn.k_norm.register_forward_hook(
            lambda _module, _inputs, output: save_once("l0_k_norm", output)
        )

        def save_hf_rope(_module, _inputs, output):
            save_once("rope_cos", output[0])
            save_once("rope_sin", output[1])

        model.model.rotary_emb.register_forward_hook(save_hf_rope)
        layer0.self_attn.o_proj.register_forward_pre_hook(
            lambda _module, inputs: save_once("l0_attn_out", inputs[0])
        )
        layer0.self_attn.o_proj.register_forward_hook(
            lambda _module, _inputs, output: save_once("l0_o_proj", output)
        )
        layer0.post_attention_layernorm.register_forward_hook(
            lambda _module, _inputs, output:
                save_once("l0_post_attn_norm", output)
        )
        layer0.mlp.gate_proj.register_forward_hook(
            lambda _module, _inputs, output: save_once("l0_gate", output)
        )
        layer0.mlp.up_proj.register_forward_hook(
            lambda _module, _inputs, output: save_once("l0_up", output)
        )
        layer0.mlp.down_proj.register_forward_pre_hook(
            lambda _module, inputs: save_once("l0_mlp_act", inputs[0])
        )
        layer0.mlp.down_proj.register_forward_hook(
            lambda _module, _inputs, output: save_once("l0_down_proj", output)
        )

    index = json.loads(
        (exported / "model.safetensors.index.json").read_text()
    )["weight_map"]
    modules = dict(model.named_modules())
    qweight_keys = sorted(key for key in index if key.endswith(".qweight"))

    with ExitStack() as stack:
        readers = {
            shard: stack.enter_context(
                safe_open(exported / shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(index.values()))
        }

        def tensor(key: str) -> torch.Tensor:
            return readers[index[key]].get_tensor(key)

        hooks = []
        for number, qweight_key in enumerate(qweight_keys, start=1):
            name = qweight_key.removesuffix(".qweight")
            module = modules[name]
            qweight = tensor(qweight_key).to(device)
            encoded_scales = tensor(name + ".scales").to(device)
            weight_global_scale = tensor(name + ".weight_global_scale").to(
                device, dtype=torch.float32
            )
            fp4 = codebook[
                torch.stack((qweight & 0xF, qweight >> 4), dim=-1)
                .reshape(module.weight.shape)
                .long()
            ]
            scales = (
                encoded_scales.view(torch.float8_e4m3fn).to(torch.float32)
                / weight_global_scale
            )
            module.weight.data = (
                fp4 * scales.repeat_interleave(16, dim=1)
            ).to(torch.bfloat16)

            act_global_scale = tensor(name + ".act_global_scale").to(
                device, dtype=torch.float32
            )
            owner = transform_owner(name)
            perm_key = owner + ".input_perm"
            scale_key = owner + ".input_rescale"
            input_perm = tensor(perm_key).to(device) if perm_key in index else None
            input_rescale = (
                tensor(scale_key).to(device, dtype=torch.bfloat16)
                if scale_key in index
                else None
            )

            def pre_hook(
                _module,
                inputs,
                *,
                act_global_scale=act_global_scale,
                input_perm=input_perm,
                input_rescale=input_rescale,
            ):
                x = inputs[0]
                if input_perm is not None:
                    x = x.index_select(-1, input_perm)
                if input_rescale is not None:
                    x = x * input_rescale
                return (activation_quantize(x, act_global_scale, method), *inputs[1:])

            hooks.append(module.register_forward_pre_hook(pre_hook))
            print(f"HF_EXPORT_LOAD {number}/{len(qweight_keys)} {name}", flush=True)

    def limited(data):
        return data if args.limit is None else data[: args.limit]

    if args.dataset in ("wikitext2", "both"):
        data = limited(get_wikitext2(tokenizer, args.sequence_length))
        print(
            f"HF_EXPORTED_WIKITEXT2_PERPLEXITY: {compute_perplexity(model, data):.4f}",
            flush=True,
        )
    if args.dataset in ("c4", "both"):
        data = limited(get_c4_eval(tokenizer, args.sequence_length))
        print(
            f"HF_EXPORTED_C4_PERPLEXITY: {compute_perplexity(model, data):.4f}",
            flush=True,
        )
    print("HF_EXPORTED_PERPLEXITY_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
