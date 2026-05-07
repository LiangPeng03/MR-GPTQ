#!/bin/bash

# --- 配置区 ---
GPU_ID=1
LOG_FILE="eval_summary_mx.log"
PYTHON_BIN="$HOME/.conda/envs/awq/bin/python"

# 待测试模型列表
MODELS=(
    # "HuggingFaceTB/SmolLM2-135M"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-8B"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Llama-2-7b-hf"
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
printf "%-30s | %-8s | %-8s | %-8s | %-8s | %-8s\n" "Model" "W-PPL" "C4-PPL" "PIQA" "ARC-C" "WINO" >> $LOG_FILE
echo "----------------------------------------------------------" >> $LOG_FILE

# --- 循环运行 ---
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo "Processing: $MODEL"
    
    # 构造命令
    CMD="$PYTHON_BIN model_quant.py \
        --model_name_or_path=$MODEL \
        --format=mxfp \
        --w_bits=4 \
        --a_bits=4 \
        --w_group_size=32 \
        --a_group_size=32 \
        --transform_class=identity \
        --w_observer=minmax \
        --quantization_order=activation \
        --hadamard_group_size=128 \
        --dataset_name_or_path=fineweb-edu \
        --num_sequences=128 \
        --gptq \
        --rel_damp=0.01 \
        --sequence_length=2048 \
        --dtype=bfloat16 \
        --fuse_global_scale \
        --eval_perplexity \
        --eval_openllm \
        --lm_eval_tasks piqa winogrande"

    # 运行并静默非核心输出
    tmp_out="tmp_mx_eval.out"
    echo "Launch Command: $CMD" >> $LOG_FILE
    $CMD > $tmp_out 2>&1

    # 如果运行失败，记录错误
    if [ $? -ne 0 ]; then
        echo "$MODEL | FAILED (Check tmp_mx_eval.out)" >> $LOG_FILE
        cat $tmp_out # 在屏幕显示错误信息
        continue
    fi

    # 从 JSON 摘要中提取数值
    SUMMARY_LINE=$(grep "\[RESULT_SUMMARY\]" $tmp_out | sed 's/\[RESULT_SUMMARY\] //')
    
    # 使用 Python 快速解析 JSON 字段并格式化数值
    WIKI_PPL=$(echo "$SUMMARY_LINE" | python3 -c "import sys, json; v=json.load(sys.stdin).get('wikitext2_ppl'); print(f'{v:.3f}' if isinstance(v, (int, float)) else 'N/A')")
    C4_PPL=$(echo "$SUMMARY_LINE" | python3 -c "import sys, json; v=json.load(sys.stdin).get('c4_ppl'); print(f'{v:.3f}' if isinstance(v, (int, float)) else 'N/A')")
    
    PIQA=$(echo "$SUMMARY_LINE" | python3 -c "import sys, json; d=json.load(sys.stdin).get('results', {}); v=d.get('piqa'); print(f'{v:.5f}' if isinstance(v, (int, float)) else 'N/A')")
    ARC=$(echo "$SUMMARY_LINE" | python3 -c "import sys, json; d=json.load(sys.stdin).get('results', {}); v=d.get('arc_challenge'); print(f'{v:.5f}' if isinstance(v, (int, float)) else 'N/A')")
    WINO=$(echo "$SUMMARY_LINE" | python3 -c "import sys, json; d=json.load(sys.stdin).get('results', {}); v=d.get('winogrande'); print(f'{v:.5f}' if isinstance(v, (int, float)) else 'N/A')")

    # 打印到屏幕
    echo "Result: Wiki-PPL: $WIKI_PPL, PIQA: $PIQA"
    
    # 格式化写入日志
    printf "%-30s | %-8s | %-8s | %-8s | %-8s | %-8s\n" "$MODEL" "$WIKI_PPL" "$C4_PPL" "$PIQA" "$ARC" "$WINO" >> $LOG_FILE
    
    rm $tmp_out
done

echo "==========================================================" >> $LOG_FILE
echo "" >> $LOG_FILE
echo "All Tasks Finished."
