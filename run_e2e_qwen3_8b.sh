#!/usr/bin/env bash
# Generic end-to-end benchmark launcher for locally exported Hugging Face models.
#
# Usage:
#   bash run_e2e_qwen3_8b.sh /path/to/model_a /path/to/model_b
#
# Or scan every immediate child model directory under MODEL_ROOT:
#   MODEL_ROOT=/path/to/exported_models bash run_e2e_qwen3_8b.sh
#
# A valid model directory must contain config.json and at least one
# *.safetensors file. All models are measured in fresh Python processes and
# appended to one result CSV.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/vptq/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_PY="${BENCHMARK_PY:-$SCRIPT_DIR/benchmark_e2e.py}"
MODEL_ROOT="${MODEL_ROOT:-$SCRIPT_DIR/e2e_models/Qwen3-8B}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULT_CSV="${RESULT_CSV:-$SCRIPT_DIR/e2e_results_${RUN_TAG}.csv}"

BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16}"
INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
WARMUP="${WARMUP:-2}"
REPEATS="${REPEATS:-5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false

is_model_dir() {
    local directory="$1"
    [[ -f "$directory/config.json" ]] || return 1
    # Hugging Face cache snapshots commonly store both config and weights as
    # symlinks into the blobs directory; -L is required to recognize them.
    find -L "$directory" -maxdepth 1 -type f -name '*.safetensors' \
        -print -quit 2>/dev/null | grep -q .
}

model_paths=()
if (( $# > 0 )); then
    for model in "$@"; do
        model_paths+=("$(cd "$model" 2>/dev/null && pwd || printf '%s' "$model")")
    done
else
    if [[ ! -d "$MODEL_ROOT" ]]; then
        echo "ERROR: MODEL_ROOT does not exist: $MODEL_ROOT" >&2
        exit 1
    fi

    while IFS= read -r -d '' directory; do
        if is_model_dir "$directory"; then
            model_paths+=("$directory")
        fi
    done < <(find "$MODEL_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    # MODEL_ROOT itself may also be a single exported model.
    if (( ${#model_paths[@]} == 0 )) && is_model_dir "$MODEL_ROOT"; then
        model_paths+=("$MODEL_ROOT")
    fi
fi

if (( ${#model_paths[@]} == 0 )); then
    echo "ERROR: no model directories were found." >&2
    echo "Expected config.json and one or more *.safetensors files in each directory." >&2
    exit 1
fi

for model in "${model_paths[@]}"; do
    if ! is_model_dir "$model"; then
        echo "ERROR: invalid model directory: $model" >&2
        echo "It must contain config.json and at least one *.safetensors file." >&2
        exit 1
    fi
done

echo "GPU_ID=$GPU_ID"
echo "RESULT_CSV=$RESULT_CSV"
echo "MODEL_COUNT=${#model_paths[@]}"
printf 'MODEL=%s\n' "${model_paths[@]}"

for model in "${model_paths[@]}"; do
    label="$(basename "$model")"
    echo "===== $label: $model ====="

    tokenizer_args=()
    if [[ -n "$TOKENIZER_MODEL" ]]; then
        tokenizer_args+=(--tokenizer "$TOKENIZER_MODEL")
    fi

    "$PYTHON_BIN" "$BENCHMARK_PY" \
        --model "$model" \
        --model_label "$label" \
        "${tokenizer_args[@]}" \
        --batch_sizes "$BATCH_SIZES" \
        --input_tokens "$INPUT_TOKENS" \
        --output_tokens "$OUTPUT_TOKENS" \
        --warmup "$WARMUP" \
        --repeats "$REPEATS" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --dtype "$DTYPE" \
        --gpu_index "$GPU_ID" \
        --output_csv "$RESULT_CSV" \
        --append_csv
done

echo "All benchmarks completed: $RESULT_CSV"
