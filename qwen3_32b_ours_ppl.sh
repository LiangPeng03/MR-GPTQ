#!/bin/bash
# Qwen3-32B Ours (channel_cluster + GICS) W4A4: perplexity only.

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1

MODEL="/home/pengliang/models/Qwen3-32B"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/qwen3_32b_ours_ppl.log"
exec > >(tee -a "$LOG_FILE") 2>&1

$HOME/.conda/envs/awq/bin/python -u model_quant.py \
    --model_name_or_path="$MODEL" \
    --format=nvfp \
    --w_bits=4 \
    --a_bits=4 \
    --seed=42 \
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
    --fuse_global_scale \
    --gptq \
    --channel_resort=channel_cluster \
    --channel_rescale=gics \
    --kmeans_block_size=-1 \
    --kmeans_alpha=2 \
    --kmeans_act_alpha=0 \
    --cpu_offload_modules \
    --eval_perplexity
