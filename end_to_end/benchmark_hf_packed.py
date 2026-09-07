#!/usr/bin/env python3
"""HF-transformers throughput for the exported NVFP4-LSS checkpoint,
loaded PACKED (qweight/scales kept quantized, dequantize-inside-GEMM via
QuTLASS matmul_nvf4_bf16_tn) to match the 2026-08-14 four_over_six_rtn_hf
baseline's loading strategy and memory footprint.

Two arms:
  --arm lss           : Ours - packed weights from the exported checkpoint,
                        activation side = perm/rescale + LSS quantization
                        (vLLM kernel), GEMM = QuTLASS NVFP4.
  --arm four_over_six : TRUE 4/6 baseline - weight side: per-group search
                        over {absmax/6, absmax/4} minimizing MSE (offline,
                        quantizer.py FOUR_OVER_SIX logic); activation side:
                        the same 2-candidate MSE search at RUNTIME.  Same
                        QuTLASS GEMM.  This is the honest naive-framework
                        cost of 4/6 (the 8/14 baseline omitted the
                        activation-side search and used absmax instead).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent  # repo root (this script lives in end_to_end/)
sys.path.insert(0, str(ROOT))

# src/quantization/__init__.py pulls fast_hadamard_transform (not installed,
# never called on these code paths) - stub it before importing helpers.
import types

if "fast_hadamard_transform" not in sys.modules:
    _stub = types.ModuleType("fast_hadamard_transform")

    def _not_available(*_a, **_k):
        raise RuntimeError("hadamard_transform invoked - unexpected path")

    _stub.hadamard_transform = _not_available
    sys.modules["fast_hadamard_transform"] = _stub

# NOTE: do NOT import the pip `qutlass` package in this process.  vLLM's
# _C.abi3.so embeds its own qutlass build and registers the same
# TORCH_LIBRARY namespace (`_qutlass_C`); importing both crashes the process.
# All kernels are used through the `torch.ops.fp_quant.*` registrations in
# vllm.model_executor.layers.quantization.fp_quant instead.
import vllm.model_executor.layers.quantization.fp_quant  # registers torch.ops.fp_quant.*
from vllm._custom_ops import nvfp4_lss_quant_permute_scale

FPQ = torch.ops.vllm

from benchmark_e2e import PROMPT_SEED, NvmlMonitor, summarize
from eval_exported_hf_perplexity import transform_owner
from src.quantization.quant_ops import cast_to_fp4, pack_fp4_to_uint8

BASE_MODEL = (
    "/home/pengliang/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
EXPORTED_MODEL = (
    "/home/pengliang/Desktop/MR-GPTQ/e2e_models/"
    "qwen3_8b_ours_lss_validated_20260904_211031/ours"
)
PAD_TOKEN_ID = 151643
FP8_E4M3_MAX = 448.0


def four_over_six_search(
    x: torch.Tensor, group_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """True 4/6 scale search (src/quantization/quantizer.py:178-199).

    Args: x [..., G, 16] grouped tensor.  Returns (grid_values, scales):
    RAW E2M1 grid values in [-6, 6] (NOT dequantized - multiply by scales
    to reconstruct) and the per-group winning scale ({absmax/6, absmax/4}
    by per-group MSE)."""
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    s6 = (abs_max / 6.0).clamp(min=1e-30)
    s4 = (abs_max / 4.0).clamp(min=1e-30)
    recon6 = cast_to_fp4(x / s6) * s6
    recon4 = cast_to_fp4(x / s4) * s4
    err6 = (x - recon6).pow(2).sum(dim=-1, keepdim=True)
    err4 = (x - recon4).pow(2).sum(dim=-1, keepdim=True)
    scales = torch.where(err4 < err6, s4, s6)
    grid_values = cast_to_fp4(x / scales)
    return grid_values, scales.squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("lss", "four_over_six"), default="lss")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--exported-model", default=EXPORTED_MODEL)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nvml-interval-ms", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def fp4_codebook(device: torch.device) -> torch.Tensor:
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return torch.tensor(values + [-v for v in values], device=device)


class PackedLSSLinear(nn.Module):
    """Ours: packed NVFP4 weights + perm/rescale/LSS activation quantization."""

    def __init__(self, qweight, wscales_u8, w_gs, a_gs, perm, rescale, bias):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("wscales_u8", wscales_u8)
        self.register_buffer("w_gs", w_gs)
        self.register_buffer("a_gs", a_gs)
        self.register_buffer("perm", perm)
        self.register_buffer("rescale", rescale)
        self.out_features, in_half = qweight.shape
        self.in_features = in_half * 2
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        q, s = nvfp4_lss_quant_permute_scale(
            x_flat, self.perm, self.rescale, self.a_gs
        )
        alpha = 1.0 / (self.a_gs * self.w_gs)
        y = FPQ.matmul_nvf4_bf16(
            q,
            self.qweight,
            s,
            self.wscales_u8.view(torch.float8_e4m3fn),
            alpha,
        )
        y = y.unflatten(0, shape[:-1])
        if self.bias is not None:
            y = y + self.bias
        return y


class PackedFourOverSixLinear(nn.Module):
    """TRUE 4/6: weight-side 4/6 MSE search offline (here in __init__),
    activation-side 4/6 MSE search at runtime, same QuTLASS NVFP4 GEMM.
    Identity transform; global scales = 1 (per-group scales stored directly
    in E4M3, clamped to 448 like cast_scales_to_eXmY)."""

    def __init__(self, weight: torch.Tensor, bias, device: torch.device):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        w = weight.to(device).flatten(end_dim=-2).contiguous()
        w_groups = w.view(self.out_features, self.in_features // 16, 16)
        vals, scales = four_over_six_search(w_groups)
        # Pack with the same convention as the exported checkpoints.
        qweight = pack_fp4_to_uint8(vals.view(self.out_features, self.in_features))
        wscales = scales.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        self.register_buffer("qweight", qweight)
        self.register_buffer("wscales", wscales)
        self.register_buffer("one", torch.ones(1, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features).contiguous()
        m = x_flat.shape[0]
        x_groups = x_flat.view(m, self.in_features // 16, 16)
        vals, scales = four_over_six_search(x_groups)
        q = pack_fp4_to_uint8(vals.view(m, self.in_features))
        # Match the padded scale layout produced by the LSS kernel
        # (rows rounded to 128, groups rounded to 4) for the shared GEMM.
        padded_m = (m + 127) // 128 * 128
        padded_g = (self.in_features // 16 + 3) // 4 * 4
        s_padded = x_flat.new_zeros((padded_m, padded_g), dtype=torch.float32)
        s_padded[:m, : self.in_features // 16] = scales
        s_fp8 = s_padded.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        y = FPQ.matmul_nvf4_bf16(q, self.qweight, s_fp8, self.wscales, self.one)
        y = y.unflatten(0, shape[:-1])
        if self.bias is not None:
            y = y + self.bias
        return y


def load_packed_model(args, device: torch.device):
    if args.arm == "lss":
        index = json.loads(
            (Path(args.exported_model) / "model.safetensors.index.json").read_text()
        )["weight_map"]
        readers = {
            shard: safe_open(
                Path(args.exported_model) / shard, framework="pt", device="cpu"
            )
            for shard in sorted(set(index.values()))
        }

        def tensor(key: str) -> torch.Tensor:
            return readers[index[key]].get_tensor(key)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    model.config.use_cache = True

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name.endswith("lm_head"):
            continue
        bias = module.bias.data.to(device) if module.bias is not None else None
        if args.arm == "lss":
            owner = transform_owner(name)
            perm = tensor(owner + ".input_perm").to(device)
            rescale = tensor(owner + ".input_rescale").to(device, dtype=torch.bfloat16)
            new = PackedLSSLinear(
                tensor(name + ".qweight").to(device),
                tensor(name + ".scales").to(device),
                tensor(name + ".weight_global_scale").to(device, dtype=torch.float32),
                tensor(name + ".act_global_scale").to(device, dtype=torch.float32),
                perm,
                rescale,
                bias,
            )
        else:
            new = PackedFourOverSixLinear(module.weight.data, bias, device)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        parent._modules[name.rsplit(".", 1)[-1]] = new
        replaced += 1
    torch.cuda.empty_cache()
    print(f"replaced {replaced} linears (arm={args.arm})", flush=True)
    return model


def fixed_token_ids(tokenizer, token_count: int, variant: int) -> list[int]:
    seed_ids = tokenizer.encode(PROMPT_SEED, add_special_tokens=False)
    ids = (seed_ids * ((token_count + len(seed_ids) - 1) // len(seed_ids)))[
        :token_count
    ]
    if variant and len(seed_ids) > 1:
        ids[-1] = seed_ids[variant % len(seed_ids)]
    return ids


def numeric_self_check(model, device: torch.device, arm: str) -> None:
    """Packed-GEMM output must match the validated BF16-decode reference."""
    linear = next(
        m for n, m in model.named_modules()
        if n.endswith("layers.0.self_attn.q_proj")
    )
    torch.manual_seed(0)
    x = torch.randn(8, linear.in_features, device=device, dtype=torch.bfloat16)

    # Reference: decode weights (validated path) and activations to BF16.
    codebook = fp4_codebook(device)
    if arm == "four_over_six":
        fp4_w = codebook[
            torch.stack((linear.qweight & 0xF, linear.qweight >> 4), dim=-1)
            .reshape(linear.out_features, linear.in_features)
            .long()
        ]
        w_deq = (
            fp4_w
            * linear.wscales.to(torch.float32).repeat_interleave(16, dim=1)
        ).to(torch.bfloat16)
        x_groups = x.view(x.shape[0], linear.in_features // 16, 16)
        vals, scales = four_over_six_search(x_groups)
        # The GEMM consumes E4M3-rounded scales; mirror that here.
        s_e4m3 = (
            scales.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
            .to(torch.float32)
        )
        x_deq = (
            vals.view(x.shape)
            * s_e4m3.repeat_interleave(16, dim=1)
        ).to(torch.bfloat16)
        y_ref = x_deq @ w_deq.T
        with torch.no_grad():
            y_gem = linear(x).reshape(y_ref.shape)
        rel = (y_gem.float() - y_ref.float()).norm() / y_ref.float().norm()
        print(
            f"SELF_CHECK rel_err={rel.item():.6f} {'PASS' if rel < 0.02 else 'FAIL'}",
            flush=True,
        )
        if rel >= 0.02:
            raise RuntimeError("packed GEMM disagrees with 4/6 reference")
        return
    fp4_w = codebook[
        torch.stack((linear.qweight & 0xF, linear.qweight >> 4), dim=-1)
        .reshape(linear.out_features, linear.in_features)
        .long()
    ]
    w_deq = (
        fp4_w
        * (
            linear.wscales_u8.view(torch.float8_e4m3fn).to(torch.float32)
            / linear.w_gs
        ).repeat_interleave(16, dim=1)
    ).to(torch.bfloat16)
    q, s = nvfp4_lss_quant_permute_scale(x, linear.perm, linear.rescale, linear.a_gs)
    fp4_x = codebook[
        torch.stack((q & 0xF, q >> 4), dim=-1).reshape(x.shape).long()
    ]
    x_deq = (
        fp4_x
        * (
            s[: x.shape[0]].view(torch.float8_e4m3fn).to(torch.float32) / linear.a_gs
        ).repeat_interleave(16, dim=1)
    ).to(torch.bfloat16)
    y_ref = x_deq @ w_deq.T

    with torch.no_grad():
        y_gem = linear(x).reshape(y_ref.shape)

    rel = (y_gem.float() - y_ref.float()).norm() / y_ref.float().norm()
    print(f"SELF_CHECK rel_err={rel.item():.6f} {'PASS' if rel < 0.02 else 'FAIL'}", flush=True)
    if rel >= 0.02:
        raise RuntimeError("packed GEMM disagrees with BF16-decode reference")


def generate_batch(model, input_ids, attention_mask, max_new: int):
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        min_new_tokens=max_new,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=PAD_TOKEN_ID,
    )


def main() -> int:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    monitor = NvmlMonitor(args.gpu, args.nvml_interval_ms / 1000.0)
    idle = monitor.sample_once()
    monitor.start()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = load_packed_model(args, device)
    load_seconds = time.perf_counter() - load_started
    load_nvml = monitor.stop()
    post_load = monitor.sample_once()

    if args.arm == "lss":
        numeric_self_check(model, device, "lss")
    else:
        numeric_self_check(model, device, "four_over_six")

    smoke_ids = torch.tensor(
        [fixed_token_ids(tokenizer, 32, 0)], device=device, dtype=torch.long
    )
    with torch.no_grad():
        smoke_out = generate_batch(model, smoke_ids, None, 8)
    print(f"SMOKE generated {smoke_out.shape[1] - smoke_ids.shape[1]} tokens", flush=True)

    raw_runs, summaries = [], []
    for batch_size in args.batch_sizes:
        ids = torch.tensor(
            [fixed_token_ids(tokenizer, args.input_tokens, i) for i in range(batch_size)],
            device=device,
            dtype=torch.long,
        )
        mask = torch.ones_like(ids)
        expected = batch_size * args.output_tokens

        for w in range(args.warmups):
            with torch.no_grad():
                out = generate_batch(model, ids, mask, args.output_tokens)
            got = (out.shape[1] - args.input_tokens) * batch_size
            if got != expected:
                raise RuntimeError(f"warmup generated {got}, expected {expected}")
            print(f"WARMUP batch={batch_size} run={w + 1}/{args.warmups}", flush=True)

        batch_runs = []
        for r in range(args.repeats):
            monitor.start()
            started = time.perf_counter()
            with torch.no_grad():
                out = generate_batch(model, ids, mask, args.output_tokens)
            elapsed = time.perf_counter() - started
            nvml = monitor.stop()
            got = (out.shape[1] - args.input_tokens) * batch_size
            if got != expected:
                raise RuntimeError(f"generated {got}, expected {expected}")
            run = {
                "batch_size": batch_size,
                "repeat": r + 1,
                "output_tokens": got,
                "elapsed_seconds": elapsed,
                "output_tokens_per_second": got / elapsed,
                **nvml,
            }
            raw_runs.append(run)
            batch_runs.append(run)
            print(
                f"MEASURE batch={batch_size} run={r + 1}/{args.repeats} "
                f"tps={run['output_tokens_per_second']:.3f}",
                flush=True,
            )

        summary = {"batch_size": batch_size, "warmups": args.warmups,
                   "repeats": args.repeats,
                   "input_tokens_per_request": args.input_tokens,
                   "output_tokens_per_request": args.output_tokens}
        summary.update(summarize([x["output_tokens_per_second"] for x in batch_runs],
                                 "output_tokens_per_second"))
        summary.update(summarize([x["peak_memory_mib"] for x in batch_runs],
                                 "peak_memory_mib"))
        summaries.append(summary)

    result = {
        "method": f"{args.arm}_hf_packed",
        "runtime": "transformers+qutlass",
        "settings": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "batch_sizes": args.batch_sizes,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "temperature": 0,
            "ignore_eos_equivalent": "min_new_tokens=max_new_tokens",
            "nvml_interval_ms": args.nvml_interval_ms,
        },
        "environment": {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "raw_runs": raw_runs,
        "summaries": summaries,
        "load_seconds": load_seconds,
        "load_nvml": load_nvml,
        "idle_memory_mib": idle.memory_used_mib,
        "post_load_memory_mib": post_load.memory_used_mib,
    }
    output_dir = args.output_dir or (
        ROOT / "benchmark_results"
        / f"hf_packed_{args.arm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )
    monitor.close()
    print(f"RESULT_DIR={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
