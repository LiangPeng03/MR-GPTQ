import json
import os
import sys
import time

if len(sys.argv) > 1 and sys.argv[1] == "quant-compare":
    from bench_quant_ops import main
    main(sys.argv[2:])
    raise SystemExit(0)
if len(sys.argv) > 1 and sys.argv[1] == "nada-breakdown":
    from bench_nada_breakdown import main
    main(sys.argv[2:])
    raise SystemExit(0)

import torch
from safetensors import safe_open
from vllm._custom_ops import (
    nvfp4_lss_quant_identity,
    nvfp4_lss_quant_permute_scale,
    permute_and_scale_cols,
)


def validate_and_benchmark_nada_fast():
    """Compare actual packed outputs and CUDA Graph latency against legacy LSS."""
    from vllm._custom_ops import nvfp4_lss_quant_permute_scale_cutlass
    from vllm.model_executor.layers.quantization import fp_quant
    from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked

    def graph_time(function):
        for _ in range(5):
            function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = function()
        for _ in range(10):
            graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 100
        del graph, result
        return elapsed

    torch.manual_seed(0)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("vLLM:", fp_quant.__file__, flush=True)
    for channels in (2048, 4096, 14336):
        perm = torch.randperm(channels, device="cuda")
        perm32 = perm.to(torch.int32)
        rescale = torch.linspace(0.5, 1.5, channels, device="cuda").bfloat16()
        a_gs = torch.tensor([64.0], device="cuda")
        w_gs = torch.tensor([64.0], device="cuda")
        alpha = 1.0 / (a_gs * w_gs)
        weight = torch.randn(256, channels, device="cuda", dtype=torch.bfloat16)
        qw, ws = nvfp4_lss_quant_identity(weight, w_gs)
        ws = ws[:256, :channels // 16].view(torch.uint8).contiguous()
        blocked_ws = to_blocked(ws).view(torch.float8_e4m3fn).reshape(-1, channels // 16)
        bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        hadamard = torch.eye(16, device="cuda", dtype=torch.bfloat16)
        for rows in (1, 16, 128, 512):
            x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
            if rows == 16:
                x[:4] = 0
                x[4:8] *= 1e4
                x[8:12] *= 1e-4
            old_q, old_s = nvfp4_lss_quant_permute_scale(x, perm, rescale, a_gs)
            new_q, new_s = nvfp4_lss_quant_permute_scale_cutlass(x, perm32, rescale, a_gs)
            row = torch.arange(rows, device="cuda")[:, None]
            group = torch.arange(channels // 16, device="cuda")[None, :]
            positions = (
                ((row // 128) * (channels // 64) + group // 4) * 512
                + (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4
            )
            assert torch.equal(old_q, new_q[:rows]), (channels, rows, "FP4 codes")
            assert torch.equal(
                old_s[:rows, :channels // 16].view(torch.uint8),
                new_s.view(torch.uint8).flatten()[positions],
            ), (channels, rows, "E4M3 scales")

            def legacy():
                return fp_quant.quantized_forward(
                    x, qw, ws, w_gs, a_gs, bias,
                    hadamard,
                    "lss", "nvfp4", perm, rescale,
                )

            def fast():
                return fp_quant.nada_fast_forward(
                    x, qw, blocked_ws, alpha, perm32, rescale, a_gs, bias,
                )

            expected, actual = legacy(), fast()
            assert torch.equal(expected, actual), (channels, rows, "GEMM output")
            old_ms, new_ms = graph_time(legacy), graph_time(fast)
            print(
                f"NADA_FAST_OK K={channels} M={rows} "
                f"legacy_ms={old_ms:.6f} fast_ms={new_ms:.6f} "
                f"speedup={old_ms / new_ms:.3f}x",
                flush=True,
            )
    print("ALL NADA FAST CHECKS PASSED", flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "nada-fast":
    validate_and_benchmark_nada_fast()
    raise SystemExit(0)


def validate_nada_checkpoint(model_dir):
    """Compare real fused QKV/gate-up parameters at their calibrated scales."""
    from vllm._custom_ops import nvfp4_lss_quant_permute_scale_cutlass
    from vllm.model_executor.layers.quantization import fp_quant
    from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked
    from pathlib import Path

    root = Path(model_dir)
    mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    def load(key):
        with safe_open(root / mapping[key], framework="pt", device="cpu") as shard:
            return shard.get_tensor(key).cuda().contiguous()

    config = json.loads((root / "config.json").read_text())
    torch.manual_seed(0)
    for layer in sorted({0, (config["num_hidden_layers"] - 1) // 2,
                         config["num_hidden_layers"] - 1}):
        for name, modules in (
            ("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
            ("o", ("self_attn.o_proj",)),
            ("gate_up", ("mlp.gate_proj", "mlp.up_proj")),
            ("down", ("mlp.down_proj",)),
        ):
            prefixes = [f"model.layers.{layer}.{module}" for module in modules]
            perm = load(prefixes[0] + ".input_perm")
            rescale = load(prefixes[0] + ".input_rescale")
            a_gs = load(prefixes[0] + ".act_global_scale")
            w_gs = load(prefixes[0] + ".weight_global_scale")
            for prefix in prefixes[1:]:
                for suffix, reference in (
                    ("input_perm", perm), ("input_rescale", rescale),
                    ("act_global_scale", a_gs), ("weight_global_scale", w_gs),
                ):
                    # Some exports store shared transforms only on q_proj/gate_proj.
                    if prefix + "." + suffix not in mapping:
                        if suffix in {"input_perm", "input_rescale"}:
                            continue
                        raise KeyError(prefix + "." + suffix)
                    assert torch.equal(load(prefix + "." + suffix), reference), (
                        layer, name, "fused parameter mismatch", suffix
                    )
            weight = torch.cat([load(prefix + ".qweight") for prefix in prefixes])
            scales = torch.cat([load(prefix + ".scales") for prefix in prefixes])
            channels = weight.shape[1] * 2
            blocked = to_blocked(scales).view(torch.float8_e4m3fn).reshape(-1, channels // 16)
            perm32 = perm.int()
            alpha = 1.0 / (a_gs * w_gs)
            hadamard = torch.eye(16, device="cuda", dtype=torch.bfloat16)
            for rows in (1, 2, 8, 16, 64, 128, 512):
                x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
                old_q, old_s = nvfp4_lss_quant_permute_scale(x, perm, rescale, a_gs)
                new_q, new_s = nvfp4_lss_quant_permute_scale_cutlass(x, perm32, rescale, a_gs)
                r = torch.arange(rows, device="cuda")[:, None]
                g = torch.arange(channels // 16, device="cuda")[None, :]
                positions = (
                    ((r // 128) * (channels // 64) + g // 4) * 512
                    + (r % 32) * 16 + ((r % 128) // 32) * 4 + g % 4
                )
                assert torch.equal(old_q, new_q[:rows]), (layer, name, rows, "FP4 codes")
                assert torch.equal(
                    old_s[:rows, :channels // 16].view(torch.uint8),
                    new_s.view(torch.uint8).flatten()[positions],
                ), (layer, name, rows, "FP8 scales")
                expected = fp_quant.quantized_forward(
                    x, weight, scales, w_gs, a_gs, None, hadamard,
                    "lss", "nvfp4", perm, rescale,
                )
                actual = fp_quant.nada_fast_forward(
                    x, weight, blocked, alpha, perm32, rescale, a_gs, None,
                )
                equal = torch.equal(expected, actual)
                maximum = (expected.float() - actual.float()).abs().max().item()
                print(
                    f"REAL_NADA layer={layer} group={name} M={rows} "
                    f"N={weight.shape[0]} K={channels} equal={equal} max_abs={maximum}",
                    flush=True,
                )
                assert equal, (layer, name, rows, "GEMM output")
    print("ALL REAL CHECKPOINT LAYER CHECKS PASSED", flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "nada-real":
    validate_nada_checkpoint(sys.argv[2])
    raise SystemExit(0)

model_dir = sys.argv[1]
index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
key = next(k for k in index["weight_map"] if k.endswith("input_perm"))
scale_key = key.replace("input_perm", "input_rescale")
shard = os.path.join(model_dir, index["weight_map"][key])

with safe_open(shard, framework="pt", device="cpu") as f:
    perm = f.get_tensor(key).cuda()
    scale = f.get_tensor(scale_key).cuda()

print("test_key:", key)
print("channels:", perm.numel(), "scale_dtype:", scale.dtype)

def timed(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters

for rows in (1, 16, 128):
    x = torch.randn(rows, perm.numel(), device="cuda", dtype=torch.bfloat16)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    x_ref = permute_and_scale_cols(x, perm, scale)
    q_ref, s_ref = nvfp4_lss_quant_identity(x_ref, global_scale)
    q_fused, s_fused = nvfp4_lss_quant_permute_scale(
        x, perm, scale, global_scale
    )
    torch.cuda.synchronize()

    print(
        f"rows={rows}: "
        f"fp4_codes_equal={torch.equal(q_ref, q_fused)}, "
        f"fp8_scales_equal={torch.equal(s_ref, s_fused)}, "
        f"scale_max_abs_diff={(s_ref.float() - s_fused.float()).abs().max().item():.8g}"
    )

    old_ms = timed(
        lambda: nvfp4_lss_quant_identity(
            permute_and_scale_cols(x, perm, scale), global_scale
        )
    )
    fused_ms = timed(
        lambda: nvfp4_lss_quant_permute_scale(
            x, perm, scale, global_scale
        )
    )
    lss_ms = timed(lambda: nvfp4_lss_quant_identity(x, global_scale))

    print(
        f"rows={rows}: old={old_ms:.4f} ms, "
        f"fused_A={fused_ms:.4f} ms, "
        f"lss_only={lss_ms:.4f} ms"
    )

x = torch.randn(16, perm.numel(), device="cuda", dtype=torch.bfloat16)
global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(20):
        nvfp4_lss_quant_permute_scale(x, perm, scale, global_scale)
    torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=15
))
