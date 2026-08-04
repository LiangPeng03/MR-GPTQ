#!/bin/bash
# BNV MDS downstream evaluation
#
# Run one model per GPU and collect the five original MDS task columns.
# Only GSM8K and MBPP are executed; MMLU, ARC-Easy, and Lambada are emitted
# as NA by design.
#
# Usage:
#   bash bnv_mds.sh                 # dual-GPU dispatch on GPUs 0 and 1
#   bash bnv_mds.sh 0               # single-GPU run
#   bash bnv_mds.sh 0,1             # explicit dual-GPU run
#   LIMIT=8 bash bnv_mds.sh 0,1    # smoke test; omit LIMIT for full tasks
#
# Optional environment variables:
#   PYTHON_BIN=/path/to/python
#   MODELS="Qwen/Qwen3-8B meta-llama/Meta-Llama-3-8B"
#   LOG_FILE=eval_summary_mds1.log

set -o pipefail

GPU_IDS="${1:-0,1}"
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

LOG_FILE="${LOG_FILE:-eval_summary_mds1.log}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/awq/bin/python}"

# Six-model list used by the repository's comparison scripts.
if [[ -n "${MODELS:-}" ]]; then
    read -ra MODEL_ARRAY <<< "$MODELS"
else
    MODEL_ARRAY=(
        "HuggingFaceTB/SmolLM2-135M"
        "Qwen/Qwen3-0.6B"
        "meta-llama/Meta-Llama-3-8B"
        "Qwen/Qwen3-8B"
        "meta-llama/Llama-3.2-1B"
        "Qwen/Qwen3-1.7B"
    )
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"

if [[ -n "${LIMIT:-}" && "${LIMIT}" != "0" ]]; then
    EVAL_LIMIT_ARGS=(--limit "$LIMIT")
else
    EVAL_LIMIT_ARGS=()
fi

# Parse the final [RESULT_SUMMARY] JSON. Only gsm8k and mbpp are present in
# the run; the other three original MDS columns intentionally remain NA.
parse_results() {
    "$PYTHON_BIN" -c '
import json, sys

try:
    data = json.loads(sys.stdin.read().strip())
except Exception:
    data = {}

results = data.get("results", {})

def metric(task):
    value = results.get(task)
    if not isinstance(value, dict):
        return value
    for target in ("pass@1", "exact_match", "acc_norm", "acc", "f1"):
        for key, item in value.items():
            if target in key:
                return item
    return None

def fmt(value):
    return "NA" if value is None else f"{value:.6f}"

print("|".join([
    fmt(metric("gsm8k")),
    "NA",                         # mmlu: intentionally not run
    "NA",                         # arc_easy: intentionally not run
    fmt(metric("mbpp")),
    "NA",                         # lambada_openai: intentionally not run
]))
'
}

{
    echo ""
    echo "=========================================================="
    echo "BNV MDS Started at: $(date) on GPUs: $GPU_IDS"
    echo "Models: ${MODEL_ARRAY[*]}"
    if ((${#EVAL_LIMIT_ARGS[@]} > 0)); then
        echo "Evaluation limit: $LIMIT per task"
    else
        echo "Evaluation limit: full GSM8K and MBPP"
    fi
    printf "%-30s | %-4s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
        "Model" "GPU" "GSM8K" "MMLU" "ARC-E" "MBPP" "LAMBADA" "Status" "Time"
    echo "----------------------------------------------------------------------------------------------------"
} >> "$LOG_FILE"

TASKS=("${MODEL_ARRAY[@]}")
echo "Dispatching ${#TASKS[@]} model evaluations across $NUM_GPUS GPU(s)."

for ((i = 0; i < ${#TASKS[@]}; i++)); do
    MODEL="${TASKS[$i]}"
    GPU_ID="${GPUS[$((i % NUM_GPUS))]}"

    echo "----------------------------------------------------------"
    echo "Dispatching: $MODEL -> GPU $GPU_ID"

    (
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        START_TIME=$(date +%s)
        SAFE_MODEL=$(printf '%s' "$MODEL" | tr '/ :' '___')
        TMP_LOG="tmp_eval_mds_${i}_${GPU_ID}.out"

        "$PYTHON_BIN" model_quant.py \
            --model_name_or_path="$MODEL" \
            --format=nvfp \
            --w_bits=4 \
            --a_bits=4 \
            --seed=0 \
            --w_group_size=16 \
            --a_group_size=16 \
            --transform_class=identity \
            --w_observer=mse_n \
            --a_observer=lss \
            --quantization_order=activation \
            --hadamard_group_size=16 \
            --dataset_name_or_path=c4 \
            --num_sequences=128 \
            --rel_damp=0.01 \
            --sequence_length=2048 \
            --dtype=bfloat16 \
            --show_act_mse \
            --gptq \
            --channel_resort=kmeans_fp4_top3 \
            --channel_rescale=gics \
            --kmeans_alpha=2 \
            --fuse_global_scale \
            --eval_openllm \
            --lm_eval_tasks gsm8k mbpp \
            "${EVAL_LIMIT_ARGS[@]}" \
            > "$TMP_LOG" 2>&1
        EXIT_CODE=$?

        END_TIME=$(date +%s)
        ELAPSED_SEC=$((END_TIME - START_TIME))
        if ((ELAPSED_SEC >= 3600)); then
            ELAPSED=$(printf "%dh%02dm%02ds" $((ELAPSED_SEC / 3600)) $(((ELAPSED_SEC % 3600) / 60)) $((ELAPSED_SEC % 60)))
        elif ((ELAPSED_SEC >= 60)); then
            ELAPSED=$(printf "%dm%02ds" $((ELAPSED_SEC / 60)) $((ELAPSED_SEC % 60)))
        else
            ELAPSED="${ELAPSED_SEC}s"
        fi

        if ((EXIT_CODE != 0)); then
            # Keep the temporary output in the repository and print it. No
            # failed_logs/ directory or copied failure artifact is created.
            GSM8K="NA"
            MMLU="NA"
            ARC_E="NA"
            MBPP="NA"
            LAMBADA="NA"
            STATUS="FAILED($EXIT_CODE)"
            echo "!!! FAILED: $MODEL on GPU $GPU_ID; output=$TMP_LOG"
            cat "$TMP_LOG"
        else
            SUMMARY_LINE=$(grep '\[RESULT_SUMMARY\]' "$TMP_LOG" | tail -n 1 | sed 's/^.*\[RESULT_SUMMARY\] //')
            if [[ -z "$SUMMARY_LINE" ]]; then
                GSM8K="NA"
                MMLU="NA"
                ARC_E="NA"
                MBPP="NA"
                LAMBADA="NA"
                STATUS="NO_SUMMARY"
                echo "!!! NO RESULT_SUMMARY: $MODEL on GPU $GPU_ID; output=$TMP_LOG"
                cat "$TMP_LOG"
            else
                IFS='|' read -r GSM8K MMLU ARC_E MBPP LAMBADA <<< "$(printf '%s' "$SUMMARY_LINE" | parse_results)"
                STATUS="OK"
                rm -f "$TMP_LOG"
            fi
        fi

        (
            # Lock the existing log file itself; do not create a separate
            # .lock/CSV/log directory artifact.
            flock -x 200
            printf "%-30s | %-4s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
                "$MODEL" "$GPU_ID" "$GSM8K" "$MMLU" "$ARC_E" "$MBPP" "$LAMBADA" "$STATUS" "$ELAPSED" >> "$LOG_FILE"
        ) 200>>"$LOG_FILE"

        echo "Result: $MODEL | GPU $GPU_ID | GSM8K=$GSM8K | MMLU=$MMLU | ARC-E=$ARC_E | MBPP=$MBPP | LAMBADA=$LAMBADA | status=$STATUS | time=$ELAPSED"
    ) &

    # Keep at most one full model evaluation per GPU in flight.
    if (( (i + 1) % NUM_GPUS == 0 )); then
        wait
    fi
done

wait

{
    echo "----------------------------------------------------------------------------------------------------"
    echo "All BNV MDS tasks finished at: $(date)"
} >> "$LOG_FILE"

echo "Finished. Summary: $LOG_FILE"
