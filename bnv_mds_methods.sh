#!/bin/bash
# BNV MDS baseline-method queue: RTN, GPTQ, and MR-GPTQ.
#
# Each method evaluates the same six models on GSM8K and MBPP.  Methods run
# sequentially, while models within one method are dispatched across GPUs.
# This is safe to start on a server while bnv_mds.sh is still running:
# set WAIT_FOR_LOG=eval_summary_mds1.log to wait for its completion marker.
#
# Usage:
#   bash bnv_mds_methods.sh 0,1 gptq
#   bash bnv_mds_methods.sh 0,1 rtn,mr-gptq
#   WAIT_FOR_LOG=eval_summary_mds1.log bash bnv_mds_methods.sh 0,1 rtn,mr-gptq
#   LIMIT=8 bash bnv_mds_methods.sh 0,1 rtn,gptq,mr-gptq
#
# Optional environment variables:
#   PYTHON_BIN=/path/to/python
#   MODELS="Qwen/Qwen3-8B meta-llama/Meta-Llama-3-8B"
#   LOG_PREFIX=eval_summary_mds
#   WAIT_FOR_LOG=/shared/path/eval_summary_mds1.log
#   WAIT_MARKER="All BNV MDS tasks finished"
#   WAIT_POLL_SECONDS=60

set -o pipefail

GPU_IDS="${1:-0,1}"
METHODS_SPEC="${METHODS:-${2:-rtn,gptq,mr-gptq}}"
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}
if (( NUM_GPUS < 1 )); then
    echo "ERROR: no GPU ids supplied" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/awq/bin/python}"
LOG_PREFIX="${LOG_PREFIX:-eval_summary_mds}"
WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_MARKER="${WAIT_MARKER:-All BNV MDS tasks finished}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

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

normalize_method() {
    local raw
    raw=$(printf '%s' "$1" | tr '[:upper:]_' '[:lower:]-')
    case "$raw" in
        rtn)
            printf 'rtn'
            ;;
        gptq)
            printf 'gptq'
            ;;
        mr-gptq|mrgptq|mr-gptqz|mrgptqz)
            printf 'mr-gptq'
            ;;
        *)
            echo "ERROR: unsupported method '$1' (use rtn, gptq, or mr-gptq)" >&2
            return 2
            ;;
    esac
}

METHOD_ARRAY=()
IFS=',' read -ra RAW_METHODS <<< "$METHODS_SPEC"
for raw_method in "${RAW_METHODS[@]}"; do
    [[ -z "$raw_method" ]] && continue
    METHOD_ARRAY+=("$(normalize_method "$raw_method")") || exit 2
done
if ((${#METHOD_ARRAY[@]} == 0)); then
    echo "ERROR: method queue is empty" >&2
    exit 2
fi

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
    "NA",  # mmlu is intentionally not run
    "NA",  # arc-easy is intentionally not run
    fmt(metric("mbpp")),
    "NA",  # lambada is intentionally not run
]))
'
}

wait_for_previous_run() {
    [[ -z "$WAIT_FOR_LOG" ]] && return 0

    local start_bytes=0
    if [[ -f "$WAIT_FOR_LOG" ]]; then
        start_bytes=$(wc -c < "$WAIT_FOR_LOG")
    fi

    echo "Waiting for completion marker '$WAIT_MARKER' appended to $WAIT_FOR_LOG"
    while true; do
        if [[ -f "$WAIT_FOR_LOG" ]]; then
            if tail -c +$((start_bytes + 1)) "$WAIT_FOR_LOG" 2>/dev/null | grep -Fq "$WAIT_MARKER"; then
                echo "Previous run completed; starting queued methods."
                return 0
            fi
        fi
        sleep "$WAIT_POLL_SECONDS"
    done
}

method_config() {
    local method="$1"
    METHOD_GPTQ_ARGS=()
    case "$method" in
        rtn)
            METHOD_TRANSFORM="identity"
            METHOD_W_OBSERVER="mse"
            METHOD_A_OBSERVER="minmax"
            ;;
        gptq)
            METHOD_TRANSFORM="identity"
            METHOD_W_OBSERVER="mse"
            METHOD_A_OBSERVER="minmax"
            METHOD_GPTQ_ARGS=(--gptq)
            ;;
        mr-gptq)
            METHOD_TRANSFORM="hadamard"
            METHOD_W_OBSERVER="mse"
            METHOD_A_OBSERVER="minmax"
            METHOD_GPTQ_ARGS=(--gptq)
            ;;
    esac
}

run_method() {
    local method="$1"
    local log_file="${LOG_PREFIX}_${method}.log"
    method_config "$method"

    {
        echo ""
        echo "=========================================================="
        echo "BNV MDS method=$method started at: $(date) on GPUs: $GPU_IDS"
        echo "Models: ${MODEL_ARRAY[*]}"
        echo "Config: transform=$METHOD_TRANSFORM w_observer=$METHOD_W_OBSERVER a_observer=$METHOD_A_OBSERVER gptq=${METHOD_GPTQ_ARGS[*]:-off}"
        if ((${#EVAL_LIMIT_ARGS[@]} > 0)); then
            echo "Evaluation limit: $LIMIT per task"
        else
            echo "Evaluation limit: full GSM8K and MBPP"
        fi
        printf "%-30s | %-4s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
            "Model" "GPU" "GSM8K" "MMLU" "ARC-E" "MBPP" "LAMBADA" "Status" "Time"
        echo "----------------------------------------------------------------------------------------------------"
    } >> "$log_file"

    echo "Starting method=$method on GPUs $GPU_IDS; log=$log_file"
    for ((i = 0; i < ${#MODEL_ARRAY[@]}; i++)); do
        MODEL="${MODEL_ARRAY[$i]}"
        GPU_ID="${GPUS[$((i % NUM_GPUS))]}"
        echo "Dispatching: method=$method model=$MODEL -> GPU $GPU_ID"

        (
            export CUDA_VISIBLE_DEVICES="$GPU_ID"
            START_TIME=$(date +%s)
            TMP_LOG="tmp_eval_mds_${method}_${i}_${GPU_ID}.out"

            "$PYTHON_BIN" model_quant.py \
                --model_name_or_path="$MODEL" \
                --format=nvfp \
                --w_bits=4 \
                --a_bits=4 \
                --seed=0 \
                --w_group_size=16 \
                --a_group_size=16 \
                --transform_class="$METHOD_TRANSFORM" \
                --w_observer="$METHOD_W_OBSERVER" \
                --a_observer="$METHOD_A_OBSERVER" \
                --quantization_order=activation \
                --hadamard_group_size=16 \
                --dataset_name_or_path=c4 \
                --num_sequences=128 \
                --rel_damp=0.01 \
                --sequence_length=2048 \
                --dtype=bfloat16 \
                --show_act_mse \
                "${METHOD_GPTQ_ARGS[@]}" \
                --channel_resort=none \
                --channel_rescale=none \
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
                GSM8K="NA"; MMLU="NA"; ARC_E="NA"; MBPP="NA"; LAMBADA="NA"
                STATUS="FAILED($EXIT_CODE)"
                echo "!!! FAILED: method=$method model=$MODEL GPU=$GPU_ID; output=$TMP_LOG"
                cat "$TMP_LOG"
            else
                SUMMARY_LINE=$(grep '\[RESULT_SUMMARY\]' "$TMP_LOG" | tail -n 1 | sed 's/^.*\[RESULT_SUMMARY\] //')
                if [[ -z "$SUMMARY_LINE" ]]; then
                    GSM8K="NA"; MMLU="NA"; ARC_E="NA"; MBPP="NA"; LAMBADA="NA"
                    STATUS="NO_SUMMARY"
                    echo "!!! NO RESULT_SUMMARY: method=$method model=$MODEL GPU=$GPU_ID; output=$TMP_LOG"
                    cat "$TMP_LOG"
                else
                    IFS='|' read -r GSM8K MMLU ARC_E MBPP LAMBADA <<< "$(printf '%s' "$SUMMARY_LINE" | parse_results)"
                    STATUS="OK"
                    rm -f "$TMP_LOG"
                fi
            fi

            (
                flock -x 200
                printf "%-30s | %-4s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
                    "$MODEL" "$GPU_ID" "$GSM8K" "$MMLU" "$ARC_E" "$MBPP" "$LAMBADA" "$STATUS" "$ELAPSED" >> "$log_file"
            ) 200>>"$log_file"
            echo "Result: method=$method model=$MODEL GPU=$GPU_ID GSM8K=$GSM8K MBPP=$MBPP status=$STATUS time=$ELAPSED"
        ) &

        if (( (i + 1) % NUM_GPUS == 0 )); then
            wait
        fi
    done
    wait

    {
        echo "----------------------------------------------------------------------------------------------------"
        echo "BNV MDS method=$method finished at: $(date)"
    } >> "$log_file"
    echo "Finished method=$method; log=$log_file"
}

wait_for_previous_run
for method in "${METHOD_ARRAY[@]}"; do
    run_method "$method"
done

echo "All queued methods finished: ${METHOD_ARRAY[*]}"
