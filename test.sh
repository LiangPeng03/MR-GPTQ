#!/bin/bash

# --- 配置区 ---
# 支持从命令行参数获取 GPU_ID，例如: "0", "1", "0,1"
GPU_IDS=${1:-0}
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

LOG_FILE="eval_summary_block_size.log"
PYTHON_BIN="$HOME/.conda/envs/awq/bin/python"

# 待测试模型列表 (当前仅跑小模型)
MODELS=(
    "HuggingFaceTB/SmolLM2-135M"
    "Qwen/Qwen3-0.6B"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "Qwen/Qwen3-8B"
)

# 待测试的 Block Size 列表
# BLOCK_SIZES=(64 128 256 512 1024)
BLOCK_SIZES=(0.5 1.0 2.0)

# 屏蔽冗余 Warning 和日志
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"

# 初始化日志文件
echo "" >> $LOG_FILE
echo "==========================================================" >> $LOG_FILE
echo "Batch Run Started at: $(date) on GPUs: $GPU_IDS" >> $LOG_FILE
printf "%-30s | %-10s | %-8s | %-8s | %-8s | %-8s | %-8s\n" "Model" "BlockSize" "W-PPL" "C4-PPL" "PIQA" "ARC-C" "WINO" >> $LOG_FILE
echo "----------------------------------------------------------" >> $LOG_FILE

# 准备任务列表
TASKS=()
for MODEL in "${MODELS[@]}"; do
    for BS in "${BLOCK_SIZES[@]}"; do
        TASKS+=("$MODEL|$BS")
    done
done

# 并发调度逻辑
for (( i=0; i<${#TASKS[@]}; i++ )); do
    TASK="${TASKS[$i]}"
    IFS='|' read -r MODEL BS <<< "$TASK"
    
    GPU_IDX=$((i % NUM_GPUS))
    GPU_ID=${GPUS[$GPU_IDX]}
    
    echo "----------------------------------------------------------"
    echo "Dispatching: $MODEL | Block Size: $BS -> GPU $GPU_ID"
    
    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        
        CMD="$PYTHON_BIN model_quant.py \
            --model_name_or_path=$MODEL \
            --format=nvfp \
            --w_bits=4 \
            --a_bits=4 \
            --seed=0 \
            --w_group_size=16 \
            --a_group_size=16 \
            --transform_class=identity \
            --channel_resort=kmeans_fp4 \
            --kmeans_block_size=-1 \
            --kmeans_alpha=0 \
            --kmeans_act_alpha=$BS \
            --w_observer=mse \
            --a_observer=lss \
            --quantization_order=activation \
            --dataset_name_or_path=fineweb-edu \
            --num_sequences=128 \
            --rel_damp=0.01 \
            --sequence_length=2048 \
            --dtype=bfloat16 \
            --gptq \
            --fuse_global_scale \
            --eval_perplexity \
            --eval_openllm \
            --lm_eval_tasks piqa winogrande"

        tmp_out="tmp_eval_bs_${BS}_${GPU_ID}_$$.out"
        # 不再把 CMD 写入 LOG，避免并发混乱，只在出问题时或者屏幕上看
        $CMD > $tmp_out 2>&1

        if [ $? -ne 0 ]; then
            echo "$MODEL | BlockSize: $BS | FAILED (GPU $GPU_ID)" >> $LOG_FILE
            cat $tmp_out
            rm $tmp_out
            exit 1
        fi

        SUMMARY_LINE=$(grep "\[RESULT_SUMMARY\]" $tmp_out | sed 's/\[RESULT_SUMMARY\] //')
        
        WIKI_PPL=$(echo "$SUMMARY_LINE" | python3 -c "import sys, json; v=json.load(sys.stdin).get('wikitext2_ppl'); print(f'{v:.3f}' if isinstance(v, (int, float)) else 'N/A')")
        C4_PPL=$(echo "$SUMMARY_LINE" | python3 -c "import sys, json; v=json.load(sys.stdin).get('c4_ppl'); print(f'{v:.3f}' if isinstance(v, (int, float)) else 'N/A')")
        PIQA=$(echo "$SUMMARY_LINE" | python3 -c "import sys, json; d=json.load(sys.stdin).get('results', {}); v=d.get('piqa'); print(f'{v:.5f}' if isinstance(v, (int, float)) else 'N/A')")
        ARC=$(echo "$SUMMARY_LINE" | python3 -c "import sys, json; d=json.load(sys.stdin).get('results', {}); v=d.get('arc_challenge'); print(f'{v:.5f}' if isinstance(v, (int, float)) else 'N/A')")
        WINO=$(echo "$SUMMARY_LINE" | python3 -c "import sys, json; d=json.load(sys.stdin).get('results', {}); v=d.get('winogrande'); print(f'{v:.5f}' if isinstance(v, (int, float)) else 'N/A')")

        echo "Result ($BS on GPU $GPU_ID): Wiki-PPL: $WIKI_PPL, PIQA: $PIQA"
        
        # 为了避免并发写入日志冲突，使用 flock 锁定
        (
            flock -x 200
            printf "%-30s | %-10s | %-8s | %-8s | %-8s | %-8s | %-8s\n" "$MODEL" "$BS" "$WIKI_PPL" "$C4_PPL" "$PIQA" "$ARC" "$WINO" >> $LOG_FILE
        ) 200> "${LOG_FILE}.lock"
        
        rm $tmp_out
    ) &
    
    # 满载控制：如果当前启动的任务数达到 GPU 数量，就等待这一批结束
    if (( (i + 1) % NUM_GPUS == 0 )); then
        wait
    fi
done

# 等待所有剩余的后台任务完成
wait

echo "==========================================================" >> $LOG_FILE
echo "" >> $LOG_FILE
echo "All Tasks Finished."
