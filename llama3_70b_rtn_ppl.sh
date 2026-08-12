#!/bin/bash
# Llama-3-70B RTN W4A4, PPL only.

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1

MODEL="/home/pengliang/models/Meta-Llama-3-70B"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/llama3_70b_rtn_ppl.log"
exec > >(tee -a "$LOG_FILE") 2>&1

$HOME/.conda/envs/awq/bin/python -u model_quant.py \
    --model_name_or_path="$MODEL" \
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
    --fuse_global_scale \
    --channel_resort=none \
    --channel_rescale=none \
    --cpu_offload_modules \
    --cpu_offload_activations \
    --cpu_offload_eval \
    --eval_perplexity
