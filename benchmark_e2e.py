#!/usr/bin/env python3
"""End-to-end vLLM throughput and GPU-memory benchmark.

The controller launches one fresh Python process per model so CUDA state, the
vLLM engine, and the KV cache cannot leak from one method into another.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODELS = (
    "rtn=/home/pengliang/Desktop/MR-GPTQ/e2e_models/Qwen3-8B/rtn_fixed",
    "gptq=/home/pengliang/Desktop/MR-GPTQ/e2e_models/Qwen3-8B/gptq",
    "mr_gptq=/home/pengliang/Desktop/MR-GPTQ/e2e_models/Qwen3-8B/mr_gptq",
    "ours=/home/pengliang/Desktop/MR-GPTQ/e2e_models/Qwen3-8B/ours",
)
DEFAULT_TOKENIZER = (
    "/home/pengliang/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
PROMPT_SEED = (
    "Artificial intelligence systems execute matrix operations to transform "
    "input tokens into contextual representations. Quantization reduces model "
    "memory and computation while attempting to preserve prediction quality. "
)


@dataclass
class NvmlSnapshot:
    memory_used_mib: float
    gpu_util_percent: float
    memory_util_percent: float
    power_watts: float


class NvmlMonitor:
    def __init__(self, physical_gpu_index: int, interval_s: float) -> None:
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError(
                "NVML sampling requires nvidia-ml-py (import name: pynvml)."
            ) from exc
        self.nvml = pynvml
        self.interval_s = interval_s
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_index)
        self.samples: list[NvmlSnapshot] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> NvmlSnapshot:
        memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
        try:
            power_watts = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except self.nvml.NVMLError:
            power_watts = float("nan")
        return NvmlSnapshot(
            memory_used_mib=memory.used / (1024**2),
            gpu_util_percent=float(util.gpu),
            memory_util_percent=float(util.memory),
            power_watts=power_watts,
        )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("NVML monitor is already running")
        self.samples = [self.sample_once()]
        self._stop_event.clear()

        def collect() -> None:
            while not self._stop_event.wait(self.interval_s):
                self.samples.append(self.sample_once())

        self._thread = threading.Thread(target=collect, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        if self._thread is None:
            raise RuntimeError("NVML monitor is not running")
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 4))
        self.samples.append(self.sample_once())
        self._thread = None
        power = [x.power_watts for x in self.samples if x.power_watts == x.power_watts]
        return {
            "peak_memory_mib": max(x.memory_used_mib for x in self.samples),
            "mean_gpu_util_percent": statistics.fmean(
                x.gpu_util_percent for x in self.samples
            ),
            "peak_gpu_util_percent": max(x.gpu_util_percent for x in self.samples),
            "mean_memory_util_percent": statistics.fmean(
                x.memory_util_percent for x in self.samples
            ),
            "mean_power_watts": statistics.fmean(power) if power else float("nan"),
            "peak_power_watts": max(power) if power else float("nan"),
            "nvml_sample_count": len(self.samples),
        }

    def close(self) -> None:
        self.nvml.nvmlShutdown()


def parse_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, received: {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, received: {value!r}")
    return name, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_mapping,
        metavar="NAME=PATH",
        help="Repeat for each model. Defaults to RTN/GPTQ/MR-GPTQ/Ours.",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--tokenizer-map",
        action="append",
        type=parse_mapping,
        default=[],
        metavar="NAME=PATH",
        help="Optional per-model tokenizer override.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--fp16-gpu-memory-utilization", type=float, default=None,
                        help="Optional GPU memory utilization used only for fp16.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--nvml-interval-ms", type=float, default=20.0)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable-cudagraph",
        action="store_true",
        help="Allow vLLM compilation/CUDA Graph instead of enforce_eager mode.",
    )
    parser.add_argument(
        "--reference-method",
        default="gptq",
        help="Method used for the relative comparison CSV (default: gptq).",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print child commands without loading models.",
    )
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_method", help=argparse.SUPPRESS)
    parser.add_argument("--_model_path", help=argparse.SUPPRESS)
    parser.add_argument("--_result_path", type=Path, help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.input_tokens <= 0 or args.output_tokens <= 0:
        raise ValueError("input/output token counts must be positive")
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be >= 0 and repeats must be >= 1")
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    if (
        args.fp16_gpu_memory_utilization is not None
        and not 0 < args.fp16_gpu_memory_utilization <= 1
    ):
        raise ValueError("fp16-gpu-memory-utilization must be in (0, 1]")
    if args.max_num_batched_tokens < args.input_tokens:
        raise ValueError("max-num-batched-tokens must be >= input-tokens")


def child_command(
    args: argparse.Namespace,
    method: str,
    model_path: str,
    tokenizer: str,
    result_path: Path,
) -> list[str]:
    gpu_memory_utilization = (
        args.fp16_gpu_memory_utilization
        if method == "fp16" and args.fp16_gpu_memory_utilization is not None
        else args.gpu_memory_utilization
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        "--_method",
        method,
        "--_model_path",
        model_path,
        "--_result_path",
        str(result_path),
        "--tokenizer",
        tokenizer,
        "--gpu",
        str(args.gpu),
        "--dtype",
        args.dtype,
        "--input-tokens",
        str(args.input_tokens),
        "--output-tokens",
        str(args.output_tokens),
        "--batch-sizes",
        *map(str, args.batch_sizes),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--nvml-interval-ms",
        str(args.nvml_interval_ms),
        "--seed",
        str(args.seed),
    ]
    if args.skip_smoke:
        command.append("--skip-smoke")
    if args.enable_cudagraph:
        command.append("--enable-cudagraph")
    return command


def controller_main(args: argparse.Namespace) -> int:
    models = args.model or [parse_mapping(x) for x in DEFAULT_MODELS]
    tokenizer_overrides = dict(args.tokenizer_map)
    names = [name for name, _ in models]
    if len(names) != len(set(names)):
        raise ValueError("model names must be unique")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or Path("benchmark_results") / stamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "controller_python": sys.executable,
        "models": dict(models),
        "tokenizer": args.tokenizer,
        "tokenizer_overrides": tokenizer_overrides,
        "settings": {
            key: value
            for key, value in vars(args).items()
            if not key.startswith("_") and key not in {"model", "tokenizer_map", "output_dir"}
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n"
    )

    failures: list[str] = []
    completed: list[dict[str, Any]] = []
    for index, (method, model_path) in enumerate(models, start=1):
        tokenizer = tokenizer_overrides.get(method, args.tokenizer)
        result_path = output_dir / f"{method}.json"
        command = child_command(args, method, model_path, tokenizer, result_path)
        print(f"\n[{index}/{len(models)}] method={method} model={model_path}", flush=True)
        print("command:", " ".join(command), flush=True)
        if args.dry_run:
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        completed_process = subprocess.run(command, env=env, check=False)
        if completed_process.returncode != 0 or not result_path.is_file():
            failures.append(method)
            print(f"FAILED: {method} exit_code={completed_process.returncode}", flush=True)
        else:
            completed.append(json.loads(result_path.read_text()))
            print(f"COMPLETED: {method}", flush=True)
        if index != len(models) and args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)

    if not args.dry_run:
        write_combined_outputs(output_dir, completed, args.reference_method)
    print(f"\nRESULT_DIR={output_dir}")
    if failures:
        print("FAILED_METHODS=" + ",".join(failures), file=sys.stderr)
        return 1
    return 0


def exact_token_prompt(tokenizer: Any, token_count: int, variant: int) -> dict[str, Any]:
    seed_ids = tokenizer.encode(PROMPT_SEED, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("Tokenizer produced an empty seed prompt")
    ids = (seed_ids * ((token_count + len(seed_ids) - 1) // len(seed_ids)))[:token_count]
    # Keep the length fixed while giving requests distinct final tokens.
    if variant and len(seed_ids) > 1:
        ids[-1] = seed_ids[variant % len(seed_ids)]
    return {"prompt_token_ids": ids}


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"median_{prefix}": statistics.median(values),
        f"mean_{prefix}": statistics.fmean(values),
        f"std_{prefix}": statistics.stdev(values) if len(values) > 1 else 0.0,
        f"min_{prefix}": min(values),
        f"max_{prefix}": max(values),
    }


def child_main(args: argparse.Namespace) -> int:
    if not args._method or not args._model_path or not args._result_path:
        raise ValueError("internal child arguments are incomplete")

    from importlib.metadata import PackageNotFoundError, version

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    try:
        vllm_version = version("vllm")
    except PackageNotFoundError:
        vllm_version = "unknown"

    monitor = NvmlMonitor(args.gpu, args.nvml_interval_ms / 1000.0)
    idle = monitor.sample_once()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    max_model_len = args.input_tokens + args.output_tokens

    monitor.start()
    load_started = time.perf_counter()
    llm = LLM(
        model=args._model_path,
        tokenizer=args.tokenizer,
        dtype=args.dtype,
        max_model_len=max_model_len,
        max_num_seqs=max(args.batch_sizes),
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.enable_cudagraph,
        enable_prefix_caching=False,
        disable_log_stats=True,
        seed=args.seed,
    )
    load_seconds = time.perf_counter() - load_started
    load_nvml = monitor.stop()
    post_load = monitor.sample_once()

    smoke_output = None
    if not args.skip_smoke:
        smoke = llm.generate(
            ["What is 2 + 2? Answer briefly."],
            SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, seed=args.seed),
            use_tqdm=False,
        )
        smoke_output = smoke[0].outputs[0].text
        print(f"SMOKE_OUTPUT[{args._method}]={smoke_output!r}", flush=True)

    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.output_tokens,
        min_tokens=args.output_tokens,
        ignore_eos=True,
        seed=args.seed,
    )
    raw_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for batch_size in args.batch_sizes:
        prompts = [exact_token_prompt(tokenizer, args.input_tokens, i) for i in range(batch_size)]
        for warmup_index in range(args.warmups):
            output = llm.generate(prompts, sampling, use_tqdm=False)
            generated = sum(len(item.outputs[0].token_ids) for item in output)
            expected = batch_size * args.output_tokens
            if generated != expected:
                raise RuntimeError(
                    f"Warm-up generated {generated} tokens, expected {expected}"
                )
            print(
                f"WARMUP method={args._method} batch={batch_size} "
                f"run={warmup_index + 1}/{args.warmups}",
                flush=True,
            )

        batch_runs: list[dict[str, Any]] = []
        for repeat_index in range(args.repeats):
            monitor.start()
            started = time.perf_counter()
            output = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            nvml = monitor.stop()
            generated = sum(len(item.outputs[0].token_ids) for item in output)
            expected = batch_size * args.output_tokens
            if generated != expected:
                raise RuntimeError(f"Generated {generated} tokens, expected {expected}")
            run = {
                "method": args._method,
                "batch_size": batch_size,
                "repeat": repeat_index + 1,
                "input_tokens": batch_size * args.input_tokens,
                "output_tokens": generated,
                "elapsed_seconds": elapsed,
                "output_tokens_per_second": generated / elapsed,
                "requests_per_second": batch_size / elapsed,
                **nvml,
                "peak_memory_delta_from_idle_mib": (
                    nvml["peak_memory_mib"] - idle.memory_used_mib
                ),
            }
            raw_runs.append(run)
            batch_runs.append(run)
            print(
                f"MEASURE method={args._method} batch={batch_size} "
                f"run={repeat_index + 1}/{args.repeats} "
                f"tps={run['output_tokens_per_second']:.3f} "
                f"peak_mib={run['peak_memory_mib']:.1f}",
                flush=True,
            )

        summary: dict[str, Any] = {
            "method": args._method,
            "batch_size": batch_size,
            "model_path": args._model_path,
            "load_seconds": load_seconds,
            "idle_memory_mib": idle.memory_used_mib,
            "post_load_memory_mib": post_load.memory_used_mib,
            "load_peak_memory_mib": load_nvml["peak_memory_mib"],
            "load_peak_memory_delta_from_idle_mib": (
                load_nvml["peak_memory_mib"] - idle.memory_used_mib
            ),
            "input_tokens_per_request": args.input_tokens,
            "output_tokens_per_request": args.output_tokens,
            "warmups": args.warmups,
            "repeats": args.repeats,
        }
        summary.update(
            summarize(
                [x["output_tokens_per_second"] for x in batch_runs],
                "output_tokens_per_second",
            )
        )
        summary.update(summarize([x["elapsed_seconds"] for x in batch_runs], "seconds"))
        summary.update(
            summarize([x["peak_memory_mib"] for x in batch_runs], "peak_memory_mib")
        )
        summary.update(
            summarize(
                [x["peak_memory_delta_from_idle_mib"] for x in batch_runs],
                "peak_memory_delta_from_idle_mib",
            )
        )
        summary["mean_gpu_util_percent"] = statistics.fmean(
            x["mean_gpu_util_percent"] for x in batch_runs
        )
        summary["mean_power_watts"] = statistics.fmean(
            x["mean_power_watts"] for x in batch_runs
        )
        summaries.append(summary)

    result = {
        "method": args._method,
        "model_path": args._model_path,
        "tokenizer": args.tokenizer,
        "smoke_output": smoke_output,
        "load_seconds": load_seconds,
        "load_nvml": load_nvml,
        "idle_nvml": asdict(idle),
        "post_load_nvml": asdict(post_load),
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm_version,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "settings": {
            "dtype": args.dtype,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "batch_sizes": args.batch_sizes,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "temperature": 0,
            "ignore_eos": True,
            "prefix_cache": False,
            "enforce_eager": not args.enable_cudagraph,
            "cudagraph_enabled": args.enable_cudagraph,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "nvml_interval_ms": args.nvml_interval_ms,
        },
        "raw_runs": raw_runs,
        "summaries": summaries,
    }
    args._result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )
    monitor.close()
    print(f"MODEL_RESULT={args._result_path}", flush=True)
    return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_combined_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    reference_method: str,
) -> None:
    raw = [run for result in results for run in result["raw_runs"]]
    summaries = [row for result in results for row in result["summaries"]]
    write_csv(output_dir / "raw_runs.csv", raw)
    write_csv(output_dir / "summary.csv", summaries)

    references = {
        row["batch_size"]: row
        for row in summaries
        if row["method"] == reference_method
    }
    comparisons: list[dict[str, Any]] = []
    for row in summaries:
        reference = references.get(row["batch_size"])
        if reference is None:
            continue
        throughput_ratio = (
            row["median_output_tokens_per_second"]
            / reference["median_output_tokens_per_second"]
        )
        comparisons.append(
            {
                "method": row["method"],
                "reference_method": reference_method,
                "batch_size": row["batch_size"],
                "median_output_tokens_per_second": row[
                    "median_output_tokens_per_second"
                ],
                "reference_median_output_tokens_per_second": reference[
                    "median_output_tokens_per_second"
                ],
                "throughput_ratio_vs_reference": throughput_ratio,
                "throughput_gap_percent_vs_reference": (throughput_ratio - 1.0) * 100.0,
                "median_peak_memory_mib": row["median_peak_memory_mib"],
                "reference_median_peak_memory_mib": reference[
                    "median_peak_memory_mib"
                ],
                "peak_memory_difference_mib": (
                    row["median_peak_memory_mib"]
                    - reference["median_peak_memory_mib"]
                ),
            }
        )
    write_csv(output_dir / "comparison_vs_reference.csv", comparisons)
    (output_dir / "all_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    return child_main(args) if args._child else controller_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
