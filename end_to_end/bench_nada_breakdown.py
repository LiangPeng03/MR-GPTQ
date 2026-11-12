"""NADA inference cost attribution from actual checkpoint parameters.

Split-stage percentages normalize isolated CUDA-Graph stage measurements.
They are NOT an attribution of instructions inside the fused kernel.
Actual fused quantization/GEMM and whole-linear latency are measured separately.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import statistics
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=None,
                   help="Actual NADA checkpoint; otherwise use NADA_MODEL or the validated decode-graph manifest.")
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--rows", nargs="+", type=int, default=[1, 8, 16, 128, 2048])
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--check-only", action="store_true")
    return p


def resolve_model(explicit):
    if explicit is not None:
        return explicit.resolve()
    if os.environ.get("NADA_MODEL"):
        return Path(os.environ["NADA_MODEL"]).resolve()
    candidates = set()
    for file in Path("benchmark_results/nada_decode_graph_j7wF4U").rglob("*.json"):
        try:
            data = json.loads(file.read_text())
            if not isinstance(data, dict):
                continue
            values = list((data.get("models") or {}).values()) + [data.get("model_path")]
            for value in values:
                if not isinstance(value, str):
                    continue
                p = Path(value).resolve()
                try:
                    config = json.loads((p / "config.json").read_text())
                    q = config.get("quantization_config", {})
                    if (config.get("model_type") == "qwen3" and config.get("hidden_size") == 4096
                            and config.get("num_hidden_layers") == 36 and q.get("forward_method") == "lss"
                            and q.get("forward_dtype") == "nvfp4" and q.get("lss_identity_transform")
                            and not q.get("pseudoquantization")):
                        candidates.add(p)
                except (OSError, ValueError):
                    pass
        except (OSError, ValueError):
            pass
    if len(candidates) == 1:
        return candidates.pop()
    fallback = Path("/home/pengliang/Desktop/MR-GPTQ/e2e_models/qwen3_8b_ours_lss_validated_20260904_211031/ours")
    if not candidates and (fallback / "config.json").is_file():
        return fallback
    raise RuntimeError("Cannot uniquely find the validated Qwen3-8B NADA checkpoint; "
                       "set NADA_MODEL or --model. Candidates: " + repr(sorted(map(str, candidates))))


def shares(times):
    total = sum(times.values())
    if total <= 0:
        raise ValueError("Nonpositive timing denominator")
    return {key: value / total * 100 for key, value in times.items()}


def main(argv=None):
    args = parser().parse_args(argv)
    if (not args.rows or min(args.rows) < 1 or args.warmups < 1
            or args.iterations < 1 or args.repeats < 2):
        raise ValueError("Invalid shape/timing parameters")
    args.model = resolve_model(args.model)
    print("CHECKPOINT:", args.model, flush=True)
    import torch
    from safetensors import safe_open
    from vllm._custom_ops import permute_cols, nvfp4_lss_quant_permute_scale_cutlass, cutlass_scaled_fp4_mm
    from vllm.model_executor.layers.quantization.utils.scalesweep import abso_fused_quantize
    from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked
    from vllm.model_executor.layers.quantization.fp_quant import nada_fast_forward

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    config = json.loads((args.model / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if (config.get("model_type") != "qwen3" or config.get("hidden_size") != 4096
            or config.get("num_hidden_layers") != 36
            or quant.get("forward_method") != "lss" or quant.get("forward_dtype") != "nvfp4"
            or not quant.get("lss_identity_transform") or quant.get("pseudoquantization")):
        raise ValueError("Expected actual Qwen3-8B NADA real-NVFP4 identity/LSS checkpoint")
    mapping = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    layers = args.layers if args.layers is not None else [0, 17, 35]
    if any(layer < 0 or layer >= 36 for layer in layers):
        raise ValueError("Invalid layer index")
    output = args.output_dir or Path("benchmark_results/nada_breakdown")
    if not args.check_only:
        output.mkdir(parents=True, exist_ok=False)
    groups = dict(qkv=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                  o=("self_attn.o_proj",), gate_up=("mlp.gate_proj", "mlp.up_proj"),
                  down=("mlp.down_proj",))

    def load(key):
        with safe_open(args.model / mapping[key], framework="pt", device="cpu") as shard:
            return shard.get_tensor(key).cuda().contiguous()

    def timing(fn):
        for _ in range(args.warmups):
            result = fn()
            del result
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn()
        for _ in range(5):
            graph.replay()
        measurements = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            measurements.append(start.elapsed_time(end) * 1000 / args.iterations)
        del graph, captured
        return statistics.median(measurements)

    records, raw, hashes = [], [], []
    def save():
        if args.check_only:
            return
        for file, values in (("breakdown.csv", records), ("timings.csv", raw)):
            if values:
                with (output / file).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                    writer.writeheader()
                    writer.writerows(values)

    for layer in layers:
        for group, suffixes in groups.items():
            prefixes = [f"model.layers.{layer}.{suffix}" for suffix in suffixes]
            perm = load(prefixes[0] + ".input_perm").int()
            scale = load(prefixes[0] + ".input_rescale").bfloat16()
            ag, wg = load(prefixes[0] + ".act_global_scale"), load(prefixes[0] + ".weight_global_scale")
            for prefix in prefixes[1:]:
                for suffix, reference in (("input_perm", perm), ("input_rescale", scale),
                                          ("act_global_scale", ag), ("weight_global_scale", wg)):
                    if prefix + "." + suffix not in mapping and suffix in {"input_perm", "input_rescale"}:
                        continue
                    assert torch.equal(load(prefix + "." + suffix).to(reference.dtype), reference), (prefix, suffix)
            weight = torch.cat([load(prefix + ".qweight") for prefix in prefixes])
            ws = torch.cat([load(prefix + ".scales") for prefix in prefixes])
            n, k = weight.shape[0], weight.shape[1] * 2
            if not torch.equal(perm.sort().values, torch.arange(k, device="cuda", dtype=torch.int32)):
                raise ValueError("Invalid checkpoint permutation")
            if not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError("Invalid checkpoint rescale")
            blocked_ws = to_blocked(ws.view(torch.float8_e4m3fn)).view(torch.float8_e4m3fn).reshape(-1, k // 16)
            alpha = 1.0 / (wg * ag)
            bias_keys = [prefix + ".bias" for prefix in prefixes]
            if any(key in mapping for key in bias_keys):
                if not all(key in mapping for key in bias_keys):
                    raise ValueError("Partially missing fused bias")
                bias = torch.cat([load(key) for key in bias_keys])
            else:
                bias = None
            digest = hashlib.sha256()
            for value in (perm, scale, ag, wg):
                digest.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
            hashes.append(dict(layer=layer, group=group, transform_scale_sha256=digest.hexdigest()))

            for m in args.rows:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                xp = permute_cols(x, perm)
                xs = xp * scale
                q, sf = abso_fused_quantize(xs, ag)
                blocked_sf = to_blocked(sf).view(torch.float8_e4m3fn)
                fq, fs = nvfp4_lss_quant_permute_scale_cutlass(x, perm, scale, ag)
                assert torch.equal(q, fq[:m]), (layer, group, m, "split/fused packed values")
                r = torch.arange(m, device="cuda")[:, None]
                g = torch.arange(k // 16, device="cuda")[None, :]
                positions = ((r // 128) * (k // 64) + g // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + g % 4
                assert torch.equal(sf[:m, :k // 16].view(torch.uint8), fs.view(torch.uint8).flatten()[positions]), (layer, group, m, "split/fused scales")

                def mm(qv, scales):
                    return cutlass_scaled_fp4_mm(qv[:m], weight, scales.view(-1, k // 16),
                                                blocked_ws, alpha, torch.bfloat16)
                def split_chain():
                    value = permute_cols(x, perm) * scale
                    values, scales = abso_fused_quantize(value, ag)
                    return values, to_blocked(scales).view(torch.float8_e4m3fn)
                def fused_quant():
                    return nvfp4_lss_quant_permute_scale_cutlass(x, perm, scale, ag)
                def fused_chain():
                    return nada_fast_forward(x, weight, blocked_ws, alpha, perm, scale, ag, bias)

                split_y = mm(q, blocked_sf)
                if bias is not None:
                    split_y = split_y + bias
                expected = fused_chain()
                assert torch.equal(split_y, expected), (layer, group, m, "linear output")
                torch.cuda.synchronize()
                check_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(check_graph):
                    check_output = fused_chain()
                check_graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(check_output, expected), (layer, group, m, "graph output")
                del check_graph, check_output, expected, split_y
                print(f"BREAKDOWN_CHECK_OK layer={layer} group={group} M={m} N={n} K={k}", flush=True)
                if not args.check_only:
                    split_times = dict(DACC_permutation=timing(lambda: permute_cols(x, perm)),
                                       BITR_rescale=timing(lambda: xp * scale),
                                       ABSO_quant_pack=timing(lambda: abso_fused_quantize(xs, ag)),
                                       scale_layout=timing(lambda: to_blocked(sf)))
                    production = dict(fused_DACC_BITR_ABSO=timing(fused_quant),
                                      GEMM=timing(lambda: mm(fq, fs)))
                    if bias is not None:
                        y = mm(fq, fs)
                        production["bias"] = timing(lambda: y + bias)
                        del y
                    split_total = timing(split_chain)
                    linear_total = timing(fused_chain)
                    for view, times, whole in (("split_isolated_quant_stages", split_times, split_total),
                                               ("deployment_isolated_components", production, linear_total)):
                        percentages = shares(times)
                        for stage, us in times.items():
                            records.append(dict(layer=layer, group=group, M=m, N=n, K=k,
                                                view=view, stage=stage, median_us=us,
                                                share_of_isolated_sum_pct=percentages[stage],
                                                isolated_sum_us=sum(times.values()), whole_chain_us=whole,
                                                ratio_to_whole_chain_pct=us / whole * 100))
                    raw.append(dict(layer=layer, group=group, M=m, N=n, K=k,
                                    split_quant_chain_us=split_total, fused_quant_us=production["fused_DACC_BITR_ABSO"],
                                    GEMM_us=production["GEMM"], fused_linear_us=linear_total,
                                    split_isolated_sum_us=sum(split_times.values()),
                                    deployment_isolated_sum_us=sum(production.values())))
                    save()
                    print(json.dumps(raw[-1]), flush=True)
                del x, xp, xs, q, sf, blocked_sf, fq, fs
            del weight, ws, blocked_ws, perm, scale, ag, wg, alpha, bias
            gc.collect()
            torch.cuda.empty_cache()
    if not args.check_only:
        (output / "manifest.json").write_text(json.dumps(dict(
            model=str(args.model.resolve()), config=config, settings=vars(args),
            host=platform.node(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
            input="seeded synthetic BF16 activation; REAL checkpoint transforms and packed weights",
            denominator="stage percentages use isolated stage sum, NOT fused-kernel attribution or full-model share",
            whole_chain="measured separately; isolated component sums need not equal whole-chain latency",
            memory="not measured", parameters=hashes), indent=2, default=str))
    print("ALL NADA BREAKDOWN CHECKS PASSED", flush=True)
    if not args.check_only:
        print("RESULT_DIR:", output.resolve(), flush=True)


if __name__ == "__main__":
    main()
