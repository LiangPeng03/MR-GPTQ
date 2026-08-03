#!/bin/bash
# BNV PPL Ablation Experiment Script (Instance 2)
# Supports multi-GPU dispatch (like compare.sh) and two outer grid-search loops
# Usage: bash bnv_ppl1.sh [GPU_IDS]
# Examples:
#   bash bnv_ppl1.sh 0        # single GPU
#   bash bnv_ppl1.sh 0,1      # dual GPU
#
# NOTE: This is a parallel instance of bnv_ppl.sh, intended to run on a second
# server that shares the same storage. Key differences from bnv_ppl.sh:
#   - Uses bnv_ppl1.log instead of bnv_ppl.log (avoids log conflict)
#   - Uses tmp_bnv_ppl1_* instead of tmp_bnv_ppl_* for temp files
#   - All other settings are identical to bnv_ppl.sh

# --- 配置区 ---
GPU_IDS=${1:-0,1}
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

LOG_FILE="bnv_ppl1.log"
PYTHON_BIN="$HOME/.conda/envs/awq/bin/python"

# --- 模型列表 ---
MODELS=(
    # "HuggingFaceTB/SmolLM2-135M"
    # "Qwen/Qwen3-0.6B"
    "meta-llama/Llama-3.2-1B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-8B"
    "meta-llama/Meta-Llama-3-8B"
    
)

# ============================================================
# Grid Search Parameter 1: kmeans_alpha (0 1 2 3 4)
# ============================================================
GRID_PARAM1_NAME="sample"
# GRID_PARAM1_VALUES=(0 1 2 2.4 3 4)
# GRID_PARAM1_VALUES=(0.5 1.5 2.5 3.5)
# GRID_PARAM1_VALUES=(four_over_six lss_3round)
GRID_PARAM1_VALUES=(128 256 64)
# GRID_PARAM1_VALUES=(3 5 10 16)
GRID_PARAM1_FLAG=""   # empty: injected inline in command body (compare.sh style)

# ============================================================
# TODO: Grid Search Parameter 2 — replace placeholder arrays
# ============================================================
GRID_PARAM2_NAME="lenth"          # e.g. "gics_top_k"
GRID_PARAM2_VALUES=(2048 1024 )                   # e.g. (3 5 7)
GRID_PARAM2_FLAG=""         # CLI flag
# ============================================================

# --- 环境变量 ---
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"

# --- Inline Python helper to parse [RESULT_SUMMARY] ---
parse_results() {
python3 -c "
import sys, json
text = sys.stdin.read()
line = text.strip()
try:
    d = json.loads(line)
except:
    d = {}
def get_metric(task):
    return d.get('results', {}).get(task)

def fmt(v, p):
    if v is None:
        return 'N/A'
    return f'{v:.{p}f}'

wiki = d.get('wikitext2_ppl')
c4   = d.get('c4_ppl')
piqa = get_metric('piqa')
arc  = get_metric('arc_challenge')
hella = get_metric('hellaswag')
wino = get_metric('winogrande')
boolq = get_metric('boolq')

print(f'{fmt(wiki,3)} | {fmt(c4,3)} | {fmt(piqa,4)} | {fmt(arc,4)} | {fmt(hella,4)} | {fmt(wino,4)} | {fmt(boolq,4)}')
"
}

# --- Initialize log file ---
echo "" >> $LOG_FILE
echo "==========================================================" >> $LOG_FILE
echo "BNV PPL Ablation (Instance 2) Started at: $(date) on GPUs: $GPU_IDS" >> $LOG_FILE
echo "Grid Param 1: $GRID_PARAM1_NAME = ${GRID_PARAM1_VALUES[*]}" >> $LOG_FILE
echo "Grid Param 2: $GRID_PARAM2_NAME = ${GRID_PARAM2_VALUES[*]}" >> $LOG_FILE
printf "%-30s | %-16s | %-16s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-10s\n" \
    "Model" "$GRID_PARAM1_NAME" "$GRID_PARAM2_NAME" "W-PPL" "C4-PPL" "PIQA" "ARC-C" "Hella" "WINO" "Time(s)" >> $LOG_FILE
echo "-----------------------------------------------------------------------------------------------------" >> $LOG_FILE

# --- Determine grid loop bounds ---
NUM_PARAM1=${#GRID_PARAM1_VALUES[@]}
NUM_PARAM2=${#GRID_PARAM2_VALUES[@]}

# Default to single pass if arrays are empty
if [ $NUM_PARAM1 -eq 0 ]; then
    GRID_PARAM1_VALUES=("_none_")
    NUM_PARAM1=1
    GRID_PARAM1_FLAG=""
    GRID_PARAM1_NAME="N/A"
fi
if [ $NUM_PARAM2 -eq 0 ]; then
    GRID_PARAM2_VALUES=("_none_")
    NUM_PARAM2=1
    GRID_PARAM2_FLAG=""
    GRID_PARAM2_NAME="N/A"
fi

# --- Build task list: MODEL x GRID_PARAM1 x GRID_PARAM2 ---
TASKS=()
for MODEL in "${MODELS[@]}"; do
    for VAL1 in "${GRID_PARAM1_VALUES[@]}"; do
        for VAL2 in "${GRID_PARAM2_VALUES[@]}"; do
            TASKS+=("$MODEL|$VAL1|$VAL2")
        done
    done
done

echo "Total tasks to dispatch: ${#TASKS[@]}"
echo ""

# --- Concurrent dispatch ---
for (( i=0; i<${#TASKS[@]}; i++ )); do
    TASK="${TASKS[$i]}"
    IFS='|' read -r MODEL VAL1 VAL2 <<< "$TASK"

    GPU_IDX=$((i % NUM_GPUS))
    GPU_ID=${GPUS[$GPU_IDX]}

    # Build parameter suffix
    PARAM1_SUFFIX=""
    PARAM2_SUFFIX=""
    if [ "$GRID_PARAM1_FLAG" != "" ] && [ "$VAL1" != "_none_" ]; then
        PARAM1_SUFFIX=" $GRID_PARAM1_FLAG $VAL1"
    fi
    if [ "$GRID_PARAM2_FLAG" != "" ] && [ "$VAL2" != "_none_" ]; then
        PARAM2_SUFFIX=" $GRID_PARAM2_FLAG $VAL2"
    fi

    echo "----------------------------------------------------------"
    echo "Dispatching: $MODEL | $GRID_PARAM1_NAME=$VAL1 $GRID_PARAM2_NAME=$VAL2 -> GPU $GPU_ID"

    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID

        tmp="tmp_bnv_ppl1_${i}_$$.out"
        START_TIME=$(date +%s)
        $PYTHON_BIN model_quant.py \
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
            --num_sequences="$VAL1" \
            --rel_damp=0.01 \
            --sequence_length="$VAL2" \
            --dtype=bfloat16 \
            --show_act_mse \
            --gptq \
            --channel_resort=channel_cluster \
            --channel_rescale=gics \
            --kmeans_alpha=2 \
            --fuse_global_scale \
            --eval_perplexity \
            $PARAM1_SUFFIX \
            $PARAM2_SUFFIX \
            > "$tmp" 2>&1
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        ELAPSED_SEC=$((END_TIME - START_TIME))
        # Convert to human-readable: 0h 0m 0s style
        ELAPSED_H=$((ELAPSED_SEC / 3600))
        ELAPSED_M=$(((ELAPSED_SEC % 3600) / 60))
        ELAPSED_S=$((ELAPSED_SEC % 60))
        if [ $ELAPSED_H -gt 0 ]; then
            ELAPSED=$(printf "%dh%02dm%02ds" $ELAPSED_H $ELAPSED_M $ELAPSED_S)
        elif [ $ELAPSED_M -gt 0 ]; then
            ELAPSED=$(printf "%dm%02ds" $ELAPSED_M $ELAPSED_S)
        else
            ELAPSED="${ELAPSED_S}s"
        fi

        if [ $EXIT_CODE -ne 0 ]; then
            mkdir -p failed_logs
            SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
            FAILED_LOG="failed_logs/${SAFE_MODEL}_${VAL1}_${VAL2}_$(date +%Y%m%d_%H%M%S).log"
            cp "$tmp" "$FAILED_LOG"
            echo "!!! TASK FAILED: $MODEL | $VAL1 | $VAL2 | GPU $GPU_ID | exit_code=$EXIT_CODE"
            echo "!!! Error log saved to: $FAILED_LOG"
            echo "!!! --- BEGIN ERROR OUTPUT ---"
            cat "$tmp"
            echo "!!! --- END ERROR OUTPUT ---"
            # Also record failure in main log
            (
                flock -x 200
                printf "%-30s | %-16s | %-16s | FAILED (exit=%d) | %s | %-10s\n" "$MODEL" "$VAL1" "$VAL2" "$EXIT_CODE" "see:$FAILED_LOG" "$ELAPSED" >> $LOG_FILE
            ) 200> "${LOG_FILE}.lock"
            rm -f "$tmp"
            exit 1
        fi

        # Parse results from JSON summary
        SUMMARY_LINE=$(grep "\[RESULT_SUMMARY\]" "$tmp" | sed 's/\[RESULT_SUMMARY\] //')
        if [ -z "$SUMMARY_LINE" ]; then
            # Fallback: grep PPL from stdout
            WIKI_PPL=$(grep -oP 'Wikitext-2 perplexity: \K[0-9.]+' "$tmp")
            C4_PPL=$(grep -oP 'C4 perplexity: \K[0-9.]+' "$tmp")
            [ -z "$WIKI_PPL" ] && WIKI_PPL="N/A"
            [ -z "$C4_PPL" ] && C4_PPL="N/A"
            summary="$WIKI_PPL | $C4_PPL | N/A | N/A | N/A | N/A | N/A"
        else
            summary=$(echo "$SUMMARY_LINE" | parse_results)
        fi
        echo "Result ($GRID_PARAM1_NAME=$VAL1 $GRID_PARAM2_NAME=$VAL2 on GPU $GPU_ID): $summary | ${ELAPSED}s"

        # Write to log with flock to avoid concurrent conflicts
        (
            flock -x 200
            printf "%-30s | %-16s | %-16s | %s | %-10s\n" "$MODEL" "$VAL1" "$VAL2" "$summary" "${ELAPSED}" >> $LOG_FILE
        ) 200> "${LOG_FILE}.lock"

        rm -f "$tmp"
    ) &

    # Full-load control: wait when dispatched tasks reach NUM_GPUS
    if (( (i + 1) % NUM_GPUS == 0 )); then
        wait
    fi
done

# Wait for remaining background jobs
wait

echo "==========================================================" >> $LOG_FILE
echo "" >> $LOG_FILE
echo "All Tasks Finished."
