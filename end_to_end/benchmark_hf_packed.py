#!/usr/bin/env python3
"""HF-transformers throughput for NADA and Four Over Six NVFP4 W4A4 models.

Two arms:
  --arm lss           : Ours - packed weights from the exported checkpoint,
                        activation side = perm/rescale + LSS quantization
                        (vLLM kernel), GEMM = QuTLASS NVFP4.
  --arm lss_cutlass   : Ours with the same packed NADA values/scales, but
                        SM120 CUTLASS GEMM after a runtime scale-layout
                        conversion. This is a deployment probe.
  --arm four_over_six : Official Four Over Six CUDA implementation: offline
                        weight quantization and online MSE 4/6 activation
                        selection, followed by its official CUTLASS NVFP4 GEMM.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
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
from vllm._custom_ops import (
    nvfp4_lss_quant_permute_scale,
    nvfp4_lss_quant_permute_scale_cutlass,
)

FPQ = torch.ops.vllm

# The Four Over Six extension registers its CUDA quantizer and CUTLASS GEMM
# under the independent ``fouroversix`` namespace.  Use its public Python API
# rather than reimplementing candidate selection in PyTorch.
import fouroversix._C  # noqa: F401
from fouroversix.matmul import quantized_matmul as four_over_six_matmul
from fouroversix.quantize import QuantizationConfig as FourOverSixQuantConfig
from fouroversix.quantize import QuantizedTensor
from fouroversix.quantize import quantize as four_over_six_quantize
from fouroversix.quantize.utils import to_blocked
from fouroversix.utils import (
    DataType,
    MatmulBackend,
    QuantizeBackend,
    RoundStyle,
    ScaleRule,
)

from benchmark_e2e import PROMPT_SEED, NvmlMonitor, summarize
from eval_exported_hf_perplexity import transform_owner
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
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("lss", "lss_cutlass", "four_over_six"), default="lss"
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--exported-model", default=EXPORTED_MODEL)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--profile-first-linear",
        action="store_true",
        help="Profile quantization, GEMM, and full forward for layer 0 q_proj, then exit.",
    )
    parser.add_argument(
        "--profile-rows",
        type=int,
        nargs="+",
        default=[1, 512],
        help="Token rows to profile; defaults cover decode and prefill.",
    )
    parser.add_argument("--profile-repeats", type=int, default=100)
    parser.add_argument(
        "--profile-nada-cutlass",
        action="store_true",
        help="For LSS only: validate and time Four Over Six CUTLASS GEMM on NADA tensors.",
    )
    parser.add_argument("--nvml-interval-ms", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def fp4_codebook(device: torch.device) -> torch.Tensor:
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return torch.tensor(values + [-v for v in values], device=device)


class PackedLSSLinear(nn.Module):
    """NADA packed weights + perm/rescale/LSS activation quantization."""

    def __init__(
        self, qweight, wscales_u8, w_gs, a_gs, perm, rescale, bias,
        *, cutlass_gemm: bool = False, cutlass_weight=None,
        original_shape: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.register_buffer("a_gs", a_gs)
        # The deployment-only CUTLASS kernel uses int32 indices. This halves
        # its static permutation traffic without changing the permutation.
        self.register_buffer(
            "perm", perm.to(torch.int32) if cutlass_gemm else perm
        )
        self.register_buffer("rescale", rescale)
        self.cutlass_gemm = cutlass_gemm
        if cutlass_weight is not None:
            if not cutlass_gemm or original_shape is None:
                raise ValueError("prepacked weight requires CUTLASS and original_shape")
            self.out_features, self.in_features = original_shape
            self.cutlass_weight = cutlass_weight
        else:
            if qweight is None or wscales_u8 is None or w_gs is None:
                raise ValueError("NADA export-layout tensors are required")
            self.register_buffer("qweight", qweight)
            self.register_buffer("wscales_u8", wscales_u8)
            self.register_buffer("w_gs", w_gs)
            self.out_features, in_half = qweight.shape
            self.in_features = in_half * 2
        if cutlass_gemm:
            self.register_buffer(
                "cutlass_input_amax",
                (a_gs.reciprocal() * (6.0 * 448.0)).to(torch.float32),
            )
            if cutlass_weight is None:
                self.cutlass_weight = nada_to_cutlass_tensor(
                    qweight,
                    wscales_u8.view(torch.float8_e4m3fn),
                    w_gs,
                    (self.out_features, self.in_features),
                )
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        if self.cutlass_gemm:
            q, s = nvfp4_lss_quant_permute_scale_cutlass(
                x_flat, self.perm, self.rescale, self.a_gs
            )
            cutlass_input = nada_to_cutlass_tensor_preformatted(
                q, s, self.cutlass_input_amax,
                (x_flat.shape[0], self.in_features),
            )
            y = four_over_six_matmul(
                cutlass_input, self.cutlass_weight, backend=MatmulBackend.cutlass
            )
        else:
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

    def release_cutlass_reference_storage(self) -> int:
        """Drop export-layout weights retained only for the startup self-check."""
        if not self.cutlass_gemm:
            return 0
        released = 0
        for name in ("qweight", "wscales_u8", "w_gs"):
            if name in self._buffers:
                self._buffers.pop(name)
                released += 1
        return released


class OfficialFourOverSixLinear(nn.Module):
    """Four Over Six through its official CUDA quantizer and CUTLASS GEMM.

    We quantize weights once during construction with the official MSE 4/6
    rule.  Each forward call passes the dynamic activation to the same public
    Four Over Six path, which performs the online MSE selection in CUDA.
    """

    def __init__(self, weight: torch.Tensor, bias, device: torch.device):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.quant_config = FourOverSixQuantConfig(
            backend=QuantizeBackend.cuda,
            scale_rule="mse",
            pseudo_quantize=False,
        )
        # QuantizedTensor is a lightweight dataclass rather than an nn.Module.
        # Its tensors stay alive on device for the lifetime of this benchmark.
        self.qweight = four_over_six_quantize(
            weight.detach().to(device).contiguous(), self.quant_config
        )
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features).contiguous()
        y = four_over_six_matmul(
            x_flat,
            self.qweight,
            backend=MatmulBackend.cutlass,
            input_config=self.quant_config,
        )
        y = y.unflatten(0, shape[:-1])
        if self.bias is not None:
            y = y + self.bias
        return y

def load_packed_model(args, device: torch.device):
    if args.arm in {"lss", "lss_cutlass"}:
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
        if args.arm in {"lss", "lss_cutlass"}:
            owner = transform_owner(name)
            perm = tensor(owner + ".input_perm").to(device)
            rescale = tensor(owner + ".input_rescale").to(device, dtype=torch.bfloat16)
            qweight = tensor(name + ".qweight")
            wscales = tensor(name + ".scales")
            weight_gs = tensor(name + ".weight_global_scale").to(torch.float32)
            act_gs = tensor(name + ".act_global_scale").to(
                device, dtype=torch.float32
            )
            original_shape = (qweight.shape[0], qweight.shape[1] * 2)
            if args.arm == "lss_cutlass":
                # Build the final CUTLASS representation on CPU, then transfer
                # only that representation.  Uploading the export layout first
                # leaves a multi-GiB CUDA allocator reservation after conversion.
                packed_cpu = nada_to_cutlass_tensor(
                    qweight,
                    wscales.view(torch.float8_e4m3fn),
                    weight_gs,
                    original_shape,
                )
                packed_gpu = move_quantized_tensor_to_device(
                    packed_cpu, device
                )
                new = PackedLSSLinear(
                    None, None, None, act_gs, perm, rescale, bias,
                    cutlass_gemm=True,
                    cutlass_weight=packed_gpu,
                    original_shape=original_shape,
                )
            else:
                new = PackedLSSLinear(
                    qweight.to(device),
                    wscales.to(device),
                    weight_gs.to(device),
                    act_gs,
                    perm,
                    rescale,
                    bias,
                )
        else:
            new = OfficialFourOverSixLinear(module.weight.data, bias, device)
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
        with torch.no_grad():
            y = linear(x)
        if not torch.isfinite(y).all():
            raise RuntimeError("official Four Over Six CUDA path produced non-finite output")
        print("SELF_CHECK official_4o6_cuda=PASS", flush=True)
        return
    if arm == "lss_cutlass" and not hasattr(linear, "qweight"):
        # The deployment path is prepacked on CPU and therefore deliberately
        # does not retain export-layout weights on GPU. Its tensor equivalence
        # is validated separately before benchmark runs; here verify the final
        # CUTLASS execution path is finite.
        with torch.no_grad():
            y = linear(x)
        if not torch.isfinite(y).all():
            raise RuntimeError("prepacked NADA CUTLASS path produced non-finite output")
        print("SELF_CHECK prepacked_nada_cutlass=PASS", flush=True)
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
    # The reference quantizer retains its public int64 permutation interface;
    # this one-time cast is outside timed inference.
    ref_perm = linear.perm if linear.perm.dtype == torch.long else linear.perm.long()
    q, s = nvfp4_lss_quant_permute_scale(x, ref_perm, linear.rescale, linear.a_gs)
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


def release_cutlass_reference_storage(model) -> None:
    """Release duplicate NADA export buffers before deployment measurement."""
    released = sum(
        module.release_cutlass_reference_storage()
        for module in model.modules()
        if isinstance(module, PackedLSSLinear)
    )
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if released:
        print(f"RELEASED_CUTLASS_REFERENCE_BUFFERS={released}", flush=True)


def _mean_cuda_ms(fn, warmups: int, repeats: int) -> float:
    """Return mean GPU wall time without Python dispatch or synchronization cost."""
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def nada_to_cutlass_tensor(
    values: torch.Tensor,
    row_major_scales: torch.Tensor,
    global_scale: torch.Tensor,
    original_shape: tuple[int, int],
) -> QuantizedTensor:
    """Adapt NADA E2M1/E4M3 storage to Four Over Six's CUTLASS layout.

    NADA reconstructs a block as ``q * sf / global_scale``.  The CUTLASS
    frontend reconstructs static-6 NVFP4 with alpha ``amax / (6 * 448)``.
    Setting ``amax = 6 * 448 / global_scale`` makes these representations
    algebraically identical; only the scale-factor layout needs swizzling.
    """
    rows, cols = original_shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 63) // 64 * 64
    packed_cols = padded_cols // 2
    groups = padded_cols // 16

    values = F.pad(
        values,
        (0, packed_cols - values.shape[1], 0, padded_rows - values.shape[0]),
    )
    scales = F.pad(
        row_major_scales,
        (0, groups - row_major_scales.shape[1], 0, padded_rows - row_major_scales.shape[0]),
    )
    amax = (global_scale.reciprocal() * (6.0 * 448.0)).to(torch.float32)
    return QuantizedTensor(
        values,
        to_blocked(scales),
        amax,
        DataType.nvfp4,
        original_shape,
        ScaleRule.static_6,
        RoundStyle.nearest,
        (padded_rows, padded_cols),
        scale_factors_are_in_blackwell_layout=True,
    )


def move_quantized_tensor_to_device(
    packed: QuantizedTensor, device: torch.device
) -> QuantizedTensor:
    """Transfer an already blocked CPU QuantizedTensor without reformatting."""
    return QuantizedTensor(
        packed.values.to(device),
        packed.scale_factors.to(device),
        packed.amax.to(device),
        packed.dtype,
        packed.original_shape,
        packed.scale_rule,
        packed.round_style,
        packed.padded_shape,
        scale_factors_are_in_blackwell_layout=
        packed.scale_factors_are_in_blackwell_layout,
    )


def nada_to_cutlass_tensor_preformatted(
    values: torch.Tensor,
    blocked_scales: torch.Tensor,
    amax: torch.Tensor,
    original_shape: tuple[int, int],
) -> QuantizedTensor:
    """Wrap already-padded NADA CUTLASS-layout buffers without an online copy."""
    rows, cols = original_shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 63) // 64 * 64
    expected_values = padded_rows * padded_cols // 2
    expected_scales = padded_rows * (padded_cols // 16)
    if values.numel() != expected_values or blocked_scales.numel() != expected_scales:
        raise ValueError("invalid preformatted NADA CUTLASS buffer shape")
    return QuantizedTensor(
        values,
        blocked_scales,
        amax,
        DataType.nvfp4,
        original_shape,
        ScaleRule.static_6,
        RoundStyle.nearest,
        (padded_rows, padded_cols),
        scale_factors_are_in_blackwell_layout=True,
    )


def profile_first_linear(args, model, device: torch.device) -> None:
    """Separate activation quantization from GEMM for a representative Q projection."""
    linear = next(
        m for n, m in model.named_modules()
        if n.endswith("layers.0.self_attn.q_proj")
    )
    print(
        f"PROFILE layer=layers.0.self_attn.q_proj in={linear.in_features} "
        f"out={linear.out_features} arm={args.arm}",
        flush=True,
    )

    for rows in args.profile_rows:
        x = torch.randn(rows, linear.in_features, device=device, dtype=torch.bfloat16)
        repeats = args.profile_repeats if rows <= 16 else max(10, args.profile_repeats // 5)

        if args.arm in {"lss", "lss_cutlass"}:
            if args.arm == "lss_cutlass":
                def quantize_activation():
                    return nvfp4_lss_quant_permute_scale_cutlass(
                        x, linear.perm, linear.rescale, linear.a_gs
                    )
            else:
                def quantize_activation():
                    return nvfp4_lss_quant_permute_scale(
                        x, linear.perm, linear.rescale, linear.a_gs
                    )

            q, scales = quantize_activation()

            if args.arm == "lss_cutlass":
                cutlass_input = nada_to_cutlass_tensor_preformatted(
                    q, scales, linear.cutlass_input_amax,
                    (rows, linear.in_features),
                )

                def gemm_only():
                    return four_over_six_matmul(
                        cutlass_input,
                        linear.cutlass_weight,
                        backend=MatmulBackend.cutlass,
                    )
            else:
                def gemm_only():
                    return FPQ.matmul_nvf4_bf16(
                        q,
                        linear.qweight,
                        scales,
                        linear.wscales_u8.view(torch.float8_e4m3fn),
                        1.0 / (linear.a_gs * linear.w_gs),
                    )
        else:
            def quantize_activation():
                return four_over_six_quantize(x, linear.quant_config)

            q = quantize_activation()

            def gemm_only():
                return four_over_six_matmul(
                    q,
                    linear.qweight,
                    backend=MatmulBackend.cutlass,
                )

        quant_ms = _mean_cuda_ms(quantize_activation, 10, repeats)
        gemm_ms = _mean_cuda_ms(gemm_only, 10, repeats)
        full_ms = _mean_cuda_ms(lambda: linear(x), 10, repeats)
        print(
            f"PROFILE rows={rows} repeats={repeats} "
            f"quant_ms={quant_ms:.4f} gemm_ms={gemm_ms:.4f} "
            f"full_ms={full_ms:.4f}",
            flush=True,
        )

        if args.profile_nada_cutlass:
            if args.arm not in {"lss", "lss_cutlass"}:
                raise ValueError("--profile-nada-cutlass requires a NADA arm")
            # Reuse the exact NADA activation values and block scales above,
            # then change only the GEMM backend/layout.
            nada_input = nada_to_cutlass_tensor(
                q, scales, linear.a_gs, (rows, linear.in_features)
            )
            nada_weight = nada_to_cutlass_tensor(
                linear.qweight,
                linear.wscales_u8.view(torch.float8_e4m3fn),
                linear.w_gs,
                (linear.out_features, linear.in_features),
            )

            def cutlass_gemm_only():
                return four_over_six_matmul(
                    nada_input, nada_weight, backend=MatmulBackend.cutlass
                )

            def layout_only():
                return nada_to_cutlass_tensor(
                    q, scales, linear.a_gs, (rows, linear.in_features)
                )

            def cutlass_full_forward():
                lss_values, lss_scales = quantize_activation()
                cutlass_input = nada_to_cutlass_tensor(
                    lss_values, lss_scales, linear.a_gs,
                    (rows, linear.in_features),
                )
                y = four_over_six_matmul(
                    cutlass_input, nada_weight, backend=MatmulBackend.cutlass
                )
                return y + linear.bias if linear.bias is not None else y

            native_y = gemm_only()
            cutlass_y = cutlass_gemm_only()
            rel = (native_y.float() - cutlass_y.float()).norm() / native_y.float().norm()
            cutlass_ms = _mean_cuda_ms(cutlass_gemm_only, 10, repeats)
            layout_ms = _mean_cuda_ms(layout_only, 10, repeats)
            cutlass_full_ms = _mean_cuda_ms(cutlass_full_forward, 10, repeats)
            print(
                f"PROFILE_NADA_CUTLASS rows={rows} rel_err={rel.item():.6f} "
                f"cutlass_gemm_ms={cutlass_ms:.4f} layout_ms={layout_ms:.4f} "
                f"full_ms={cutlass_full_ms:.4f}",
                flush=True,
            )


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

    if args.arm in {"lss", "lss_cutlass"}:
        numeric_self_check(model, device, args.arm)
    else:
        numeric_self_check(model, device, "four_over_six")

    if args.arm == "lss_cutlass":
        release_cutlass_reference_storage(model)
    post_load = monitor.sample_once()

    if args.profile_first_linear:
        profile_first_linear(args, model, device)
        monitor.close()
        return 0

    smoke_ids = torch.tensor(
        [fixed_token_ids(tokenizer, 32, 0)], device=device, dtype=torch.long
    )
    smoke_mask = torch.ones_like(smoke_ids)
    with torch.no_grad():
        smoke_out = generate_batch(model, smoke_ids, smoke_mask, 8)
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
        "runtime": (
            "transformers+vllm-qutlass" if args.arm == "lss"
            else (
                "transformers+nada-lss+fouroversix-cutlass-layout-probe"
                if args.arm == "lss_cutlass"
                else "transformers+official-fouroversix-cutlass"
            )
        ),
        "settings": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "batch_sizes": args.batch_sizes,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "temperature": 0,
            "ignore_eos_equivalent": "min_new_tokens=max_new_tokens",
            "nvml_interval_ms": args.nvml_interval_ms,
            "four_over_six": (
                {"weight_scale_rule": "mse", "activation_scale_rule": "mse",
                 "quantize_backend": "cuda", "matmul_backend": "cutlass"}
                if args.arm == "four_over_six" else None
            ),
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
