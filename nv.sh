#!/bin/bash

gpu_id=1
export CUDA_VISIBLE_DEVICES=$gpu_id

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

MODEL1="HuggingFaceTB/SmolLM2-135M"
MODEL2="meta-llama/Llama-2-7b-hf"
MODEL3="meta-llama/Meta-Llama-3-8B"
MODEL4="Qwen/Qwen3-0.6B"
MODEL5="Qwen/Qwen3-8B"


$HOME/.conda/envs/awq/bin/python model_quant.py \
    --model_name_or_path=${MODEL1} \
    --format=nvfp \
    --w_bits=4 \
    --a_bits=4 \
    --seed=0 \
    --w_group_size=16 \
    --a_group_size=16 \
    --transform_class=identity \
    --w_observer=mse \
    --a_observer=lss \
    --quantization_order=activation \
    --hadamard_group_size=16 \
    --dataset_name_or_path=c4 \
    --num_sequences=128 \
    --rel_damp=0.01 \
    --sequence_length=2048 \
    --dtype=bfloat16 \
    --fuse_global_scale \
    --gptq \
    --channel_resort=kmeans_fp4_top3 \
    --kmeans_block_size -1 \
    --kmeans_alpha 2 \
    --kmeans_act_alpha 0 \
    --show_act_mse \
    --eval_perplexity \
    # --eval_openllm \
    # --lm_eval_tasks piqa winogrande boolq hellaswag arc_challenge \
    # --lm_eval_tasks piqa winogrande \
    # --lm_eval_tasks piqa arc_challenge winogrande \

