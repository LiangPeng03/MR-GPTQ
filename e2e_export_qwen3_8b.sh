#!/usr/bin/env bash
# Export one real NVFP4 W4A4 Qwen3-8B model for the end-to-end benchmark.
#
# Run one method at a time:
#   bash e2e_export_qwen3_8b.sh rtn
#   bash e2e_export_qwen3_8b.sh gptq
#   bash e2e_export_qwen3_8b.sh mr_gptq
#   bash e2e_export_qwen3_8b.sh ours
#
# The existing nv.sh is intentionally left unchanged: it is an evaluation
# script, while this script only exports realquant models.

set -euo pipefail

METHOD="${1:-}"
if [[ -z "$METHOD" ]]; then
    echo "Usage: $0 {rtn|gptq|mr_gptq|ours}" >&2
    exit 2
fi

case "$METHOD" in
    rtn|gptq|mr_gptq|ours) ;;
    *)
        echo "Unknown method '$METHOD'. Use rtn, gptq, mr_gptq, or ours." >&2
        exit 2
        ;;
esac

# Set MODEL_PATH explicitly on the server if the cache is not in the default
# location. The Hub name also works when the model is already in HF cache.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/awq/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-$PWD/e2e_models/Qwen3-8B}"
SAVE_PATH="$SAVE_ROOT/$METHOD"
LOG_FILE="$SAVE_ROOT/${METHOD}.log"

mkdir -p "$SAVE_PATH"
exec > >(tee -a "$LOG_FILE") 2>&1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

COMMON_ARGS=(
    --model_name_or_path "$MODEL_PATH"
    --format nvfp
    --w_bits 4
    --a_bits 4
    --seed 0
    --w_group_size 16
    --a_group_size 16
    --quantization_order activation
    --hadamard_group_size 16
    --dataset_name_or_path c4
    --num_sequences 128
    --rel_damp 0.01
    --sequence_length 2048
    --dtype bfloat16
    --fuse_global_scale
    --export_quantized_model realquant
    --save_path "$SAVE_PATH"
)

case "$METHOD" in
    rtn)
        METHOD_ARGS=(
            --transform_class identity
            --w_observer mse
            --a_observer minmax
            --channel_resort none
            --channel_rescale none
        )
        ;;
    gptq)
        METHOD_ARGS=(
            --transform_class identity
            --w_observer mse
            --a_observer minmax
            --channel_resort none
            --channel_rescale none
            --gptq
        )
        ;;
    mr_gptq)
        METHOD_ARGS=(
            --transform_class hadamard
            --w_observer mse
            --a_observer minmax
            --channel_resort none
            --channel_rescale none
            --gptq
        )
        ;;
    ours)
        # Keep the current innovation configuration from nv.sh.
        METHOD_ARGS=(
            --transform_class identity
            --w_observer mse_n
            --a_observer lss
            --channel_resort channel_cluster
            --channel_seed_strategy max_sum
            --channel_rescale gics
            --gics_top_k 5
            --channel_rescale_rounds 3
            --kmeans_block_size -1
            --kmeans_alpha 2
            --kmeans_act_alpha 0
            --gptq
        )
        ;;
esac

echo "method=$METHOD"
echo "model=$MODEL_PATH"
echo "save_path=$SAVE_PATH"
echo "python=$PYTHON_BIN"
echo "Starting realquant export at $(date)"

"$PYTHON_BIN" model_quant.py "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}"

echo "Finished realquant export at $(date)"
echo "Exported model: $SAVE_PATH"
