#!/usr/bin/env python3
"""Simple end-to-end vLLM benchmark for MR-GPTQ models.

The model is loaded once, each batch size is warmed up, and only subsequent
generation calls are timed.  Throughput is reported as generated tokens per
second aggregated over the whole batch.  GPU memory is sampled through NVML
because vLLM executes the model in a worker process.

First-token latency is intentionally not reported here: the offline
``LLM.generate`` API returns a completed request rather than a token stream.
Use a streaming/API-server benchmark for TTFT.
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


DEFAULT_MODEL = (
    "/home/pengliang/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)


class GPUMemorySampler:
    """Poll absolute GPU memory usage in a background thread."""

    def __init__(self, device_index: int, interval_s: float = 0.05) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_gb = 0.0
        self._current_gb = 0.0
        self._lock = threading.Lock()
        self._nvml = None
        self._handle = None

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"[MEMORY] NVML unavailable: {exc}")

    @property
    def peak_gb(self) -> float:
        with self._lock:
            return self._peak_gb

    @property
    def current_gb(self) -> float:
        with self._lock:
            return self._current_gb

    def reset_peak(self) -> None:
        with self._lock:
            self._peak_gb = self._current_gb

    def _sample_once(self) -> None:
        if self._nvml is None or self._handle is None:
            return
        info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        used_gb = info.used / (1024**3)
        with self._lock:
            self._current_gb = used_gb
            self._peak_gb = max(self._peak_gb, used_gb)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if self._nvml is None:
            return
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sample_once()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM end-to-end generation throughput by batch size."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL),
        help="Local Hugging Face model directory.",
    )
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("TOKENIZER_MODEL"),
        help="Optional tokenizer directory. Useful when an exported model has an incompatible tokenizer_config.json.",
    )
    parser.add_argument(
        "--batch_sizes",
        default="1,2,4,8,16",
        help="Comma-separated batch sizes (default: 1,2,4,8,16).",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Explain the main idea clearly and concisely, including the key "
            "assumptions, reasoning steps, and practical implications. "
        ),
        help="Text repeated to construct each fixed-length input.",
    )
    parser.add_argument("--input_tokens", type=int, default=512)
    parser.add_argument("--output_tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("auto", "bfloat16", "float16"),
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable CUDA graphs; useful for debugging, not final speed results.",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=int(os.environ.get("GPU_INDEX", "0")),
        help="Physical NVML GPU index used for memory sampling.",
    )
    parser.add_argument(
        "--output_csv",
        default="e2e_benchmark_results.csv",
        help="Summary CSV path.",
    )
    parser.add_argument("--model_label", default="model")
    parser.add_argument(
        "--append_csv",
        action="store_true",
        help="Append rows to an existing CSV (used by the multi-model launcher).",
    )
    return parser.parse_args()


def build_fixed_prompt(tokenizer: Any, seed_text: str, target_tokens: int) -> str:
    """Build a deterministic prompt whose re-tokenized length is target_tokens."""
    if target_tokens <= 0:
        raise ValueError("--input_tokens must be positive")
    text = (seed_text.strip() + " ") * (target_tokens + 1)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:target_tokens]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

    # Decode/encode is not guaranteed to preserve the token count for every
    # tokenizer. Add or trim a neutral suffix until the requested length holds.
    for _ in range(32):
        actual = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if actual == target_tokens:
            return prompt
        if actual < target_tokens:
            prompt += " test" * (target_tokens - actual)
        else:
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:target_tokens]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
    raise RuntimeError("Could not construct an exact fixed-length prompt")


def build_prompts(tokenizer: Any, seed_text: str, batch_size: int, target_tokens: int) -> list[str]:
    # Requests differ by index but have exactly the same token length. This
    # avoids accidental cross-request prefix reuse while preserving fairness.
    return [
        build_fixed_prompt(tokenizer, f"Request {i:04d}. {seed_text}", target_tokens)
        for i in range(batch_size)
    ]


def generated_tokens(outputs: list[Any]) -> int:
    return sum(len(output.outputs[0].token_ids) for output in outputs)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {model_path / 'config.json'}")

    batch_sizes = [int(value.strip()) for value in args.batch_sizes.split(",")]
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("--batch_sizes must contain positive integers")
    if args.input_tokens <= 0 or args.output_tokens <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("token counts/repeats must be positive and warmup non-negative")

    import torch
    from vllm import LLM, SamplingParams

    print(f"MODEL={model_path}")
    if args.tokenizer:
        print(f"TOKENIZER={args.tokenizer}")
    print(f"BATCH_SIZES={batch_sizes}")
    print(
        "CONFIG="
        f"dtype={args.dtype}, input_tokens={args.input_tokens}, "
        f"output_tokens={args.output_tokens}, "
        f"warmup={args.warmup}, repeats={args.repeats}, "
        f"max_model_len={args.max_model_len}, "
        f"gpu_memory_utilization={args.gpu_memory_utilization}, "
        f"enforce_eager={args.enforce_eager}"
    )
    print(f"TORCH={torch.__version__}")
    print(f"GPU={torch.cuda.get_device_name(0)}")

    memory = GPUMemorySampler(args.gpu_index)
    memory.start()
    init_start = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tokenizer=args.tokenizer,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
    )
    init_seconds = time.perf_counter() - init_start
    init_peak_gb = memory.peak_gb
    print(f"MODEL_INIT_SEC={init_seconds:.4f}")
    print(f"MODEL_INIT_PEAK_MEMORY_GB={init_peak_gb:.3f}")

    tokenizer = llm.get_tokenizer()
    fixed_prompt = build_fixed_prompt(tokenizer, args.prompt, args.input_tokens)
    verified_input_tokens = len(tokenizer(fixed_prompt, add_special_tokens=False)["input_ids"])
    print(f"VERIFIED_INPUT_TOKENS={verified_input_tokens}")
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )

    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        prompts = build_prompts(tokenizer, args.prompt, batch_size, args.input_tokens)
        input_tokens = verified_input_tokens

        for warmup_index in range(args.warmup):
            llm.generate(prompts, params)
            print(f"BATCH={batch_size} WARMUP={warmup_index + 1} OK")

        speeds: list[float] = []
        elapsed_values: list[float] = []
        token_values: list[int] = []
        memory.reset_peak()
        for repeat_index in range(args.repeats):
            start = time.perf_counter()
            outputs = llm.generate(prompts, params)
            elapsed = time.perf_counter() - start
            tokens = generated_tokens(outputs)
            speed = tokens / elapsed
            speeds.append(speed)
            elapsed_values.append(elapsed)
            token_values.append(tokens)
            print(
                f"BATCH={batch_size} REPEAT={repeat_index + 1} "
                f"TOKENS={tokens} TIME_SEC={elapsed:.4f} "
                f"TOKENS_PER_SEC={speed:.2f}"
            )

        row = {
            "model_label": args.model_label,
            "model": str(model_path),
            "dtype": args.dtype,
            "batch_size": batch_size,
            "input_tokens": input_tokens,
            "requested_output_tokens": args.output_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "median_time_sec": median(elapsed_values),
            "median_generated_tokens": median(token_values),
            "median_tokens_per_sec": median(speeds),
            "mean_tokens_per_sec": mean(speeds),
            "std_tokens_per_sec": pstdev(speeds) if len(speeds) > 1 else 0.0,
            "peak_memory_gb": memory.peak_gb,
        }
        rows.append(row)
        print(
            f"BATCH={batch_size} MEDIAN_TOKENS_PER_SEC={row['median_tokens_per_sec']:.2f} "
            f"PEAK_MEMORY_GB={row['peak_memory_gb']:.3f}"
        )

    fieldnames = list(rows[0].keys()) if rows else []
    output_path = Path(args.output_csv)
    append = args.append_csv and output_path.exists() and output_path.stat().st_size > 0
    with open(output_path, "a" if append else "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        writer.writerows(rows)
    print(f"RESULT_CSV={Path(args.output_csv).resolve()}")
    print("NOTE=TTFT is not measured by offline LLM.generate; use streaming for first-token latency.")
    memory.stop()


if __name__ == "__main__":
    main()
