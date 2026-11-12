"""Packed NVFP4 operator benchmark. No models/checkpoints or environment edits.

Scope: frozen inverse global scale, BF16 input -> packed E2M1 and CUTLASS-layout
E4M3 scales. 4/6 and ScaleSearch are project-adapted Triton implementations
under this frozen-global interface, NOT their official backend benchmarks. ABSO uses
the existing native CUDA identity quantizer; no permutation/rescaling/GEMM.
"""
import argparse
import csv
import gc
import hashlib
import json
import platform
import random
import statistics
from datetime import datetime
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", nargs="+", type=int, default=[1, 8, 16, 128, 2048])
    p.add_argument("--widths", nargs="+", type=int, default=[4096, 12288, 14336])
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Optional subset of method names; e.g. ABSO_Triton_fused_3round")
    return p


def reference(x, gamma, importance, kind):
    import torch
    from vllm.model_executor.layers.quantization.utils.scalesweep import reference_quantize
    if kind in ("sweep_mse", "sweep_wmse"):
        return reference_quantize(x, gamma, importance if kind == "sweep_wmse" else None)[:2]
    values = x.float().reshape(-1, 16)
    table = torch.arange(127, device=x.device, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    base = (values.abs().amax(-1) * gamma / 6).clamp(0, 448)
    if kind == "search":
        code = base.to(torch.float8_e4m3fn).view(torch.uint8).long()
        raw_unclipped = code[:, None] + torch.arange(-2, 7, device=x.device)
        raw = raw_unclipped.clamp(1, 126)
        valid = (raw_unclipped >= 1) & (raw_unclipped <= 126)
    else:
        maximum = values.abs().amax(-1)[:, None]
        raw_scale = maximum / torch.tensor([6., 4.], device=x.device)
        raw_scale = torch.where(maximum > 0, raw_scale, torch.ones_like(raw_scale))
        raw = (raw_scale * gamma).clamp(0, 448).to(torch.float8_e4m3fn).view(torch.uint8).long()
        fallback = gamma.clamp(0, 448).to(torch.float8_e4m3fn).view(torch.uint8).long()
        raw = torch.where(raw == 0, fallback, raw)
        valid = torch.ones_like(raw, dtype=torch.bool)
    scale = table[raw] / gamma
    normalized = values[:, :, None] / scale[:, None, :]
    a = normalized.abs()
    q = torch.where(a > 5, 6, torch.where(a >= 3.5, 4,
                    torch.where(a >= 1.75, a.round(), (a * 2).round() * .5))) * normalized.sign()
    error = (q * scale[:, None, :] - values[:, :, None]).square().mean(1)
    winner = error.masked_fill(~valid, float("inf")).argmin(-1)
    group = torch.arange(values.shape[0], device=x.device)
    selected = q[group, :, winner].reshape_as(x)
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
    codes = torch.bucketize(selected.abs().contiguous(), grid).to(torch.uint8)
    codes |= (selected < 0).to(torch.uint8) * 8
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    scales = torch.zeros(((x.shape[0] + 127) // 128 * 128, (x.shape[1] // 16 + 3) // 4 * 4),
                         device=x.device, dtype=torch.float8_e4m3fn)
    scales.view(torch.uint8)[:x.shape[0], :x.shape[1] // 16] = raw[group, winner].reshape(x.shape[0], -1).byte()
    return packed, scales


def abso_reference(x, gamma, rounds=1):
    import torch
    # BF16 boundaries match the existing ABSO CUDA operator.
    grouped = x.reshape(x.shape[0], -1, 16)
    a = grouped.abs()
    initial = (a.amax(-1, keepdim=True) / 6).to(x.dtype)
    raw = initial
    for _ in range(rounds):
        safe = torch.where(raw == 0, torch.ones_like(raw), raw)
        normalized = (a / safe).to(x.dtype)
        z = normalized.float()
        g = torch.where(z > 5, 6, torch.where(z >= 3.5, 4,
                        torch.where(z >= 1.75, z.round(), (z * 2).round() * .5))).to(x.dtype)
        numerator = (a * g).sum(-1, keepdim=True)
        denominator = (g * g).sum(-1, keepdim=True)
        raw = torch.where(denominator > 0,
                          numerator / torch.where(denominator == 0, 1, denominator),
                          initial if rounds == 1 else torch.ones_like(initial))
    fp8 = (raw.float() * gamma).clamp(0, 448).to(torch.float8_e4m3fn)
    physical = (fp8.float() / gamma).to(x.dtype)
    physical = torch.where(physical == 0, torch.ones_like(physical), physical)
    normalized = (grouped / physical).to(x.dtype).float()
    a = normalized.abs()
    q = torch.where(a > 5, 6, torch.where(a >= 3.5, 4,
                    torch.where(a >= 1.75, a.round(), (a * 2).round() * .5))) * normalized.sign()
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
    # Native encode_fp4 uses the pre-rounding input sign, including negative
    # nonzero inputs that round to zero (valid E2M1 negative-zero nibble 0x8).
    # Testing q < 0 loses that sign because rounded negative zero compares equal
    # to zero. Match this native byte convention without changing the kernel.
    codes = torch.bucketize(q.abs().contiguous(), grid).byte() | (normalized < 0).byte() * 8
    codes = codes.reshape_as(x)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    padded = torch.zeros(((x.shape[0] + 127) // 128 * 128, (x.shape[1] // 16 + 3) // 4 * 4), device=x.device, dtype=torch.float8_e4m3fn)
    padded.view(torch.uint8)[:x.shape[0], :x.shape[1] // 16] = fp8.squeeze(-1).view(torch.uint8)
    return packed, padded


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.rows) < 1 or any(k <= 0 or k % 64 for k in args.widths):
        raise ValueError("Rows must be positive; K must be divisible by 64")
    if args.warmups < 1 or args.iterations < 1 or args.repeats < 2:
        raise ValueError("Need warmups>=1, iterations>=1, repeats>=2")
    import torch
    from vllm._custom_ops import nvfp4_lss_quant_identity
    from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked
    from vllm.model_executor.layers.quantization.utils import scalesweep as impl
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    gamma = torch.tensor([64.], device="cuda", dtype=torch.float32)
    names = ["ABSO_Triton_fused", "ABSO_Triton_fused_3round", "ABSO_CUDA_identity", "4o6_MSE_Triton_adapted", "ScaleSearch_Triton_adapted",
             "ScaleSweep_MSE_Triton", "ScaleSweep_WMSE_Triton"]
    kinds = dict(zip(names, ["abso_fused", "abso_3round", "abso", "4o6", "search", "sweep_mse", "sweep_wmse"]))
    if args.methods is not None:
        if len(set(args.methods)) != len(args.methods) or any(n not in kinds for n in args.methods):
            raise ValueError("Unknown or duplicate method name")
        names = args.methods

    def make(kind, x, importance):
        def call():
            if kind == "abso":
                q, s = nvfp4_lss_quant_identity(x, gamma)
            elif kind == "abso_fused":
                q, s = impl.abso_fused_quantize(x, gamma)
            elif kind == "abso_3round":
                q, s = impl.abso_fused_quantize(x, gamma, rounds=3)
            else:
                q, s = impl.scalesweep_quantize(x, gamma, importance if kind == "sweep_wmse" else None,
                                               kind == "sweep_wmse", mode={"search": 1, "4o6": 2}.get(kind, 0))
            # Shared logical-M packed values and CUTLASS scale layout. Include
            # conversion time; no GEMM or full-model inference is measured.
            return q, to_blocked(s).view(torch.float8_e4m3fn)
        return call

    # Small independent-reference checks; excluded from timing and memory.
    for m in (1, 16, 129):
        x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
        x[:, :16] = 0
        x[:, 16:32] *= 10000
        x[:, 32:48] *= .0001
        importance = torch.exp(torch.linspace(-3, 3, 256, device="cuda"))
        for name in names:
            kind = kinds[name]
            actual = make(kind, x, importance)()
            expected_q, expected_s = abso_reference(x, gamma, 3 if kind == "abso_3round" else 1) if kind in ("abso", "abso_fused", "abso_3round") else reference(x, gamma, importance, kind)
            if not torch.equal(actual[0], expected_q):
                mismatch = actual[0] != expected_q
                def canonical_zero_bytes(value):
                    low, high = value & 15, value >> 4
                    return torch.where(low == 8, 0, low) | (torch.where(high == 8, 0, high) << 4)
                print(f"PACK_MISMATCH method={name} M={m} bytes={mismatch.sum().item()} "
                      f"equal_after_zero_canonicalization={torch.equal(canonical_zero_bytes(actual[0]), canonical_zero_bytes(expected_q))}", flush=True)
                for row, column in mismatch.nonzero()[:4].tolist():
                    print(f"  row={row} byte={column} actual={actual[0][row,column].item()} "
                          f"expected={expected_q[row,column].item()}", flush=True)
            assert torch.equal(actual[0], expected_q), (name, m, "packed FP4")
            assert torch.equal(actual[1].view(torch.uint8), to_blocked(expected_s).view(torch.uint8)), (name, m, "FP8 scales/layout")
        print(f"QUANT_OP_CHECK_OK M={m} K=256", flush=True)
    # Also compare the fused and unfused implementations at every requested
    # benchmark shape, including saturation, underflow and zero groups.
    for m in args.rows:
        for k in args.widths:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            x[:, :16] = 0
            x[:, 16:32] *= 10000
            x[:, 32:48] *= .0001
            old = nvfp4_lss_quant_identity(x, gamma)
            new = impl.abso_fused_quantize(x, gamma)
            assert torch.equal(old[0], new[0]), (m, k, "fused/native FP4 bytes")
            assert torch.equal(old[1].view(torch.uint8), new[1].view(torch.uint8)), (m, k, "fused/native FP8 bytes")
            torch.cuda.synchronize()
            check_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(check_graph):
                captured = impl.abso_fused_quantize(x, gamma)
            check_graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(old[0], captured[0]), (m, k, "fused graph FP4 bytes")
            assert torch.equal(old[1].view(torch.uint8), captured[1].view(torch.uint8)), (m, k, "fused graph FP8 bytes")
            print(f"ABSO_FUSION_BITEXACT_OK M={m} K={k} eager/graph", flush=True)
            del x, old, new, captured, check_graph
            if "abso_3round" in [kinds[name] for name in names]:
                x3 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                x3[:, :16] = 0
                x3[:, 16:32] *= 10000
                x3[:, 32:48] *= .0001
                expected_q3, expected_s3 = abso_reference(x3, gamma, rounds=3)
                actual3 = impl.abso_fused_quantize(x3, gamma, rounds=3)
                assert torch.equal(actual3[0], expected_q3), (m, k, "3-round FP4 bytes")
                assert torch.equal(actual3[1].view(torch.uint8), expected_s3.view(torch.uint8)), (m, k, "3-round FP8 bytes")
                torch.cuda.synchronize()
                graph3 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph3):
                    captured3 = impl.abso_fused_quantize(x3, gamma, rounds=3)
                graph3.replay()
                torch.cuda.synchronize()
                assert torch.equal(captured3[0], expected_q3), (m, k, "3-round graph FP4 bytes")
                assert torch.equal(captured3[1].view(torch.uint8), expected_s3.view(torch.uint8)), (m, k, "3-round graph FP8 bytes")
                print(f"ABSO_3ROUND_BITEXACT_OK M={m} K={k} eager/graph", flush=True)
                del x3, expected_q3, expected_s3, actual3, captured3, graph3
    print("ALL QUANT OP CHECKS PASSED", flush=True)
    if args.check_only:
        return
    # Keep timed inputs independent of how many preflight cases were run.
    torch.manual_seed(args.seed)
    output = args.output_dir or Path("benchmark_results") / datetime.now().strftime("quant_ops_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(dict(
        settings=vars(args), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        cuda=torch.version.cuda, host=platform.node(), implementation=impl.__file__,
        implementation_sha256=hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(),
        global_scale=64, scope="frozen-global packed quantization + CUTLASS scale layout; no GEMM",
        four_over_six="project-adapted Triton; frozen-global interface; not official backend benchmark",
        scalesearch="project-adapted Triton; frozen-global interface; not official backend benchmark",
        abso="fused Triton and unfused native CUDA, checked byte-identical; no DACC/BITR",
        abso_iterations="1 or 3 LS updates, code reassignment between updates; one E4M3 projection after final update; matches LSS_3ROUND calibration semantics",
        input="Gaussian + 1% amplified outliers; hot repeated buffer",
        memory="separate per-mode allocator measurement: eager one call; graph capture + initial replay; peaks include output, exclude input/importance; reserved delta reported separately; not register/shared-memory usage"), indent=2, default=str))
    rows = []
    modes = ["eager", "graph"] if args.mode == "both" else [args.mode]
    rng = random.Random(args.seed)
    for m in args.rows:
        for k in args.widths:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            x *= torch.where(torch.rand_like(x) < .01, 10., 1.)
            importance = torch.exp(torch.linspace(-3, 3, k, device="cuda")).contiguous()
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                call = make(kinds[name], x, importance)
                for _ in range(args.warmups):
                    result = call()
                    del result
                torch.cuda.synchronize()
                for mode in modes:
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    baseline = torch.cuda.memory_allocated()
                    reserved_baseline = torch.cuda.memory_reserved()
                    torch.cuda.reset_peak_memory_stats()
                    graph = None
                    if mode == "graph":
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            graph_output = call()
                        graph.replay()
                        torch.cuda.synchronize()
                        live_output = sum(t.numel() * t.element_size() for t in graph_output)
                        timed = graph.replay
                    else:
                        result = call()
                        torch.cuda.synchronize()
                        live_output = sum(t.numel() * t.element_size() for t in result)
                    peak = torch.cuda.max_memory_allocated() - baseline
                    reserved_extra = torch.cuda.max_memory_reserved() - reserved_baseline
                    retained = torch.cuda.memory_allocated() - baseline
                    if mode == "eager":
                        del result
                        timed = call
                    samples = []
                    for _ in range(args.repeats):
                        for _ in range(5):
                            timed()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.iterations):
                            timed()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 1000 / args.iterations)
                    record = dict(method=name, mode=mode, M=m, K=k,
                                  median_us=statistics.median(samples), min_us=min(samples),
                                  max_us=max(samples), stdev_us=statistics.stdev(samples),
                                  input_bytes=x.numel() * x.element_size(), output_bytes=live_output,
                                  peak_extra_bytes=peak, workspace_estimate_bytes=max(0, peak-live_output),
                                  reserved_extra_bytes=reserved_extra,
                                  retained_after_setup_bytes=retained,
                                  memory_scope="capture_and_initial_replay" if mode=="graph" else "single_eager_call",
                                  persistent_importance_bytes=importance.numel()*4 if kinds[name]=="sweep_wmse" else 0)
                    rows.append(record)
                    print(json.dumps(record), flush=True)
                    with (output / "summary.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(record))
                        writer.writeheader()
                        writer.writerows(rows)
                    if graph is not None:
                        del timed, graph_output, graph
            del x, importance
    print("RESULT_DIR:", output.resolve(), flush=True)


if __name__ == "__main__":
    main()
