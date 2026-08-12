#!/bin/bash

# --- 配置区 ---
GPU_ID=0
LOG_FILE="eval_summary_gptq_4000.log"
PYTHON_BIN="$HOME/.conda/envs/awq/bin/python"

# 待测试模型列表
MODELS=(
    # "HuggingFaceTB/SmolLM2-135M"
    # "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-8B"
    "meta-llama/Meta-Llama-3-8B"
    # "meta-llama/Llama-2-7b-hf"
    # "meta-llama/Llama-3.2-1B"
    # "Qwen/Qwen3-1.7B"
)

# 屏蔽冗余 Warning 和日志
export CUDA_VISIBLE_DEVICES=$GPU_ID
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"

# 初始化日志文件
echo "" >> $LOG_FILE
echo "==========================================================" >> $LOG_FILE
echo "Batch Run Started at: $(date)" >> $LOG_FILE
printf "%-30s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" "Model" "W-PPL" "C4-PPL" "PIQA" "ARC-C" "WINO" "BOOLQ" "HELLA" >> $LOG_FILE
echo "----------------------------------------------------------------------------------------" >> $LOG_FILE

# --- 循环运行 ---
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo "Processing: $MODEL"
    
    # 构造命令
    CMD="$PYTHON_BIN model_quant.py \
        --model_name_or_path=$MODEL \
        --format=nvfp \
        --w_bits=4 \
        --a_bits=4 \
        --seed=0 \
        --w_group_size=16 \
        --a_group_size=16 \
        --transform_class=identity \
        --w_observer=mse \
        --a_observer=minmax \
        --quantization_order=activation \
        --hadamard_group_size=16 \
        --dataset_name_or_path=c4 \
        --num_sequences=128 \
        --rel_damp=0.01 \
        --sequence_length=2048 \
        --dtype=bfloat16 \
        --show_act_mse \
        --gptq \
        --channel_resort=none \
        --channel_rescale=none \
        --kmeans_block_size -1 \
        --kmeans_alpha 2 \
        --fuse_global_scale \
        --eval_openllm \
        --lm_eval_tasks hellaswag arc_challenge "

    # 运行并静默非核心输出
    tmp_out="tmp_eval2.out"
    echo "Launch Command: $CMD" >> $LOG_FILE
    $CMD > $tmp_out 2>&1

    # 如果运行失败，记录错误
    if [ $? -ne 0 ]; then
        echo "$MODEL | FAILED (Check tmp_eval.out)" >> $LOG_FILE
        cat $tmp_out # 在屏幕显示错误信息
        continue
    fi

    # PPL: 从 stdout 直接 grep
    WIKI_PPL=$(grep -oP 'Wikitext-2 perplexity: \K[0-9.]+' $tmp_out)
    C4_PPL=$(grep -oP 'C4 perplexity: \K[0-9.]+' $tmp_out)
    [ -z "$WIKI_PPL" ] && WIKI_PPL="N/A"
    [ -z "$C4_PPL" ] && C4_PPL="N/A"

    # 下游任务: 有 SUMMARY 就解析，没有就 NA
    SUMMARY_LINE=$(grep "\[RESULT_SUMMARY\]" $tmp_out | sed 's/\[RESULT_SUMMARY\] //')
    if [ -n "$SUMMARY_LINE" ]; then
        read PIQA ARC WINO BOOLQ HELLA <<< "$(echo "$SUMMARY_LINE" | $PYTHON_BIN -c "
import sys, json
d = json.load(sys.stdin)['results']
print(d.get('piqa','NA'), d.get('arc_challenge','NA'), d.get('winogrande','NA'), d.get('boolq','NA'), d.get('hellaswag','NA'))
")"
    else
        PIQA="NA"
        ARC="NA"
        WINO="NA"
        BOOLQ="NA"
        HELLA="NA"
    fi

    # 打印到屏幕 (只显示 PPL)
    echo "Result: Wiki-PPL: $WIKI_PPL, C4-PPL: $C4_PPL"

    # 格式化写入日志
    printf "%-30s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" "$MODEL" "$WIKI_PPL" "$C4_PPL" "$PIQA" "$ARC" "$WINO" "$BOOLQ" "$HELLA" >> $LOG_FILE
    
    rm $tmp_out
done

echo "==========================================================" >> $LOG_FILE
echo "" >> $LOG_FILE
echo "All Tasks Finished."
