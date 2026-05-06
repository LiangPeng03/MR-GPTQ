#!/bin/bash

# MR-GPTQ Quantization Script for SmolLM2-135M
# Formats: NVFP4 and MXFP4
# This script is adapted from the optimal hyperparameters described in the MR-GPTQ paper.
gpu_id=1
export CUDA_VISIBLE_DEVICES=$gpu_id

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

MODEL="HuggingFaceTB/SmolLM2-135M"
MODEL_ID="SmolLM2-135M"

# Shared optimal parameters for both NVFP4 and MXFP4 (MR-GPTQ)
NUM_SEQUENCES=128
W_BITS=4
A_BITS=4
W_OBSERVER="mse"
QUANTIZATION_ORDER="default"
TRANSFORM_CLASS="hadamard"
HADAMARD_GROUP_SIZE=64   # SmolLM2-135M hidden_size=576, must divide evenly (576/64=9)
DATASET="fineweb-edu" # Standard calibration dataset

# Note: model_quant.py automatically enforces:
# NVFP: w_group_size=16, a_group_size=16, scale_precision=e4m3
# MXFP: w_group_size=32, a_group_size=32, scale_precision=e8m0

echo "========================================="
echo "Starting NVFP4 Quantization for $MODEL"
echo "========================================="

python model_quant.py \
    --model_name_or_path=${MODEL} \
    --format=nvfp \
    --w_bits=${W_BITS} \
    --a_bits=${A_BITS} \
    --transform_class=${TRANSFORM_CLASS} \
    --w_observer=${W_OBSERVER} \
    --quantization_order=${QUANTIZATION_ORDER} \
    --gptq \
    --hadamard_group_size=${HADAMARD_GROUP_SIZE} \
    --dataset_name_or_path=${DATASET} \
    --num_sequences=${NUM_SEQUENCES} \
    --sequence_length=2048 \
    --dtype=bfloat16 \
    --eval_perplexity \
    --fuse_global_scale \
    --amp \
    --eval_openllm \
    --lm_eval_tasks piqa arc_challenge hellaswag winogrande \
    --lm_eval_batch_size 1

echo "========================================="
echo "Starting MXFP4 Quantization for $MODEL"
echo "========================================="

python model_quant.py \
    --model_name_or_path=${MODEL} \
    --format=mxfp \
    --w_bits=${W_BITS} \
    --a_bits=${A_BITS} \
    --transform_class=${TRANSFORM_CLASS} \
    --w_observer=${W_OBSERVER} \
    --quantization_order=${QUANTIZATION_ORDER} \
    --gptq \
    --hadamard_group_size=${HADAMARD_GROUP_SIZE} \
    --dataset_name_or_path=${DATASET} \
    --num_sequences=${NUM_SEQUENCES} \
    --sequence_length=2048 \
    --dtype=bfloat16 \
    --eval_perplexity \
    --fuse_global_scale \
    --amp \
    --eval_openllm \
    --lm_eval_tasks piqa arc_challenge hellaswag winogrande \
    --lm_eval_batch_size 1

echo "========================================="
echo "Quantization finished!"
echo "========================================="
