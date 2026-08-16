#!/usr/bin/env bash
# Reproducible Qwen3-8B end-to-end benchmark launcher.
#
# Default methods: RTN, GPTQ, MR-GPTQ, and Ours. Optional NAME=PATH arguments
# replace the defaults, which also makes Four-over-Six/ArcQuant easy to add:
#   bash run_e2e_qwen3_8b.sh \
#     four_over_six=/path/to/model \
#     arcquant=/path/to/model

set -eo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/vptq/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_PY="${BENCHMARK_PY:-$SCRIPT_DIR/benchmark_e2e.py}"
MODEL_ROOT="${MODEL_ROOT:-$SCRIPT_DIR/e2e_models/Qwen3-8B}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/benchmark_results/$RUN_TAG}"

INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
WARMUPS="${WARMUPS:-2}"
REPEATS="${REPEATS:-5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
NVML_INTERVAL_MS="${NVML_INTERVAL_MS:-20}"
DTYPE="${DTYPE:-bfloat16}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
ENABLE_CUDAGRAPH="${ENABLE_CUDAGRAPH:-0}"

if (( $# > 0 )); then
    model_specs=("$@")
else
    model_specs=(
        "rtn=$MODEL_ROOT/rtn_fixed"
        "gptq=$MODEL_ROOT/gptq"
        "mr_gptq=$MODEL_ROOT/mr_gptq"
        "ours=$MODEL_ROOT/ours"
    )
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$BENCHMARK_PY" ]]; then
    echo "ERROR: benchmark program is missing: $BENCHMARK_PY" >&2
    exit 1
fi
if [[ ! -f "$TOKENIZER_MODEL/tokenizer_config.json" ]]; then
    echo "ERROR: tokenizer directory is invalid: $TOKENIZER_MODEL" >&2
    exit 1
fi

model_args=()
for spec in "${model_specs[@]}"; do
    if [[ "$spec" != *=* ]]; then
        echo "ERROR: model must use NAME=PATH syntax: $spec" >&2
        exit 1
    fi
    method="${spec%%=*}"
    model_path="${spec#*=}"
    if [[ ! -f "$model_path/config.json" ]]; then
        echo "ERROR: missing model config for $method: $model_path/config.json" >&2
        exit 1
    fi
    if ! find -L "$model_path" -maxdepth 1 -type f -name '*.safetensors' \
        -print -quit 2>/dev/null | grep -q .; then
        echo "ERROR: no safetensors weights found for $method: $model_path" >&2
        exit 1
    fi
    model_args+=(--model "$method=$model_path")
done

export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

echo "PYTHON_BIN=$PYTHON_BIN"
echo "GPU_ID=$GPU_ID"
echo "TOKENIZER_MODEL=$TOKENIZER_MODEL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
printf 'MODEL_SPEC=%s\n' "${model_specs[@]}"

"$PYTHON_BIN" -m py_compile "$BENCHMARK_PY"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - <<'PY'
import torch
import vllm
import pynvml

print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("cuda:", torch.version.cuda)
print("visible_gpu_count:", torch.cuda.device_count())
PY

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "E2E_BENCHMARK_PREFLIGHT_OK"
    exit 0
fi

cudagraph_args=()
if [[ "$ENABLE_CUDAGRAPH" == "1" ]]; then
    cudagraph_args+=(--enable-cudagraph)
fi

"$PYTHON_BIN" "$BENCHMARK_PY" \
    "${model_args[@]}" \
    --tokenizer "$TOKENIZER_MODEL" \
    --gpu "$GPU_ID" \
    --dtype "$DTYPE" \
    --input-tokens "$INPUT_TOKENS" \
    --output-tokens "$OUTPUT_TOKENS" \
    --batch-sizes 1 2 4 8 16 \
    --warmups "$WARMUPS" \
    --repeats "$REPEATS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --nvml-interval-ms "$NVML_INTERVAL_MS" \
    "${cudagraph_args[@]}" \
    --output-dir "$OUTPUT_DIR"

echo "BENCHMARK_COMPLETE=$OUTPUT_DIR"
