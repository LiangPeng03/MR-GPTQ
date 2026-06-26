#!/bin/bash
# GPTQ (identity) vs MR-GPTQ (hadamard) comparison script
# Matches four-over-six evaluation: W4A4 NVFP4, all 7 downstream tasks, all 5 models
# Usage: bash compare.sh [GPU_IDS]
# Examples:
#   bash compare.sh 0,1      # dual GPU (default)
#   bash compare.sh 1,2      # dual GPU (other cards)

# --- 配置区 ---
GPU_IDS=${1:-0,1}
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

LOG_FILE="compare_gptq_mr-gptq.log"
PYTHON_BIN="$HOME/.conda/envs/awq/bin/python"

MODELS=(
    "HuggingFaceTB/SmolLM2-135M"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-8B"
)

# identity = GPTQ (no Hadamard rotation)
# hadamard = MR-GPTQ (with Hadamard rotation)
TRANSFORMS=(
    "identity"
    "hadamard"
)

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
echo "GPTQ vs MR-GPTQ Comparison Started at: $(date) on GPUs: $GPU_IDS" >> $LOG_FILE
printf "%-30s | %-12s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" \
    "Model" "Transform" "W-PPL" "C4-PPL" "PIQA" "ARC-C" "Hella" "WINO" "BoolQ" >> $LOG_FILE
echo "----------------------------------------------------------" >> $LOG_FILE

# --- Build task list ---
TASKS=()
for MODEL in "${MODELS[@]}"; do
    for TRANSFORM in "${TRANSFORMS[@]}"; do
        TASKS+=("$MODEL|$TRANSFORM")
    done
done

# --- Concurrent dispatch ---
for (( i=0; i<${#TASKS[@]}; i++ )); do
    TASK="${TASKS[$i]}"
    IFS='|' read -r MODEL TRANSFORM <<< "$TASK"

    GPU_IDX=$((i % NUM_GPUS))
    GPU_ID=${GPUS[$GPU_IDX]}

    echo "----------------------------------------------------------"
    echo "Dispatching: $MODEL | Transform: $TRANSFORM -> GPU $GPU_ID"

    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID

        # Safety: install required transformers for Qwen3 if not already present
        # (uncomment if qwen3 models fail to load)
        # pip install transformers>=4.51.0 -q

        tmp="tmp_eval_${TRANSFORM}_${GPU_ID}_$$.out"
        $PYTHON_BIN model_quant.py \
            --model_name_or_path="$MODEL" \
            --format=nvfp \
            --w_bits=4 \
            --a_bits=4 \
            --seed=0 \
            --w_group_size=16 \
            --a_group_size=16 \
            --transform_class="$TRANSFORM" \
            --w_observer=mse \
            --quantization_order=activation \
            --hadamard_group_size=16 \
            --dataset_name_or_path=fineweb-edu \
            --num_sequences=128 \
            --rel_damp=0.01 \
            --sequence_length=2048 \
            --dtype=bfloat16 \
            --gptq \
            --fuse_global_scale \
            --eval_perplexity \
            --eval_openllm \
            --lm_eval_tasks piqa winogrande boolq hellaswag arc_challenge \
            > "$tmp" 2>&1

        if [ $? -ne 0 ]; then
            echo "$MODEL | $TRANSFORM | FAILED (GPU $GPU_ID)"
            cat "$tmp"
            rm -f "$tmp"
            exit 1
        fi

        # Parse results from JSON summary
        SUMMARY_LINE=$(grep "\[RESULT_SUMMARY\]" "$tmp" | sed 's/\[RESULT_SUMMARY\] //')
        summary=$(echo "$SUMMARY_LINE" | parse_results)
        echo "Result ($TRANSFORM on GPU $GPU_ID): $summary"

        # Write to log with flock to avoid concurrent conflicts
        (
            flock -x 200
            printf "%-30s | %-12s | %s\n" "$MODEL" "$TRANSFORM" "$summary" >> $LOG_FILE
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
