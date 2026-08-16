#!/usr/bin/env bash
# Run the complete Qwen3-8B end-to-end suite sequentially on one GPU:
# RTN, GPTQ, MR-GPTQ, Ours (NVFP4 BF16), then the original FP16 model.

set -eo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/vptq/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANT_LAUNCHER="$SCRIPT_DIR/run_e2e_qwen3_8b.sh"
BENCHMARK_PY="$SCRIPT_DIR/benchmark_e2e.py"
MODEL_ROOT="${MODEL_ROOT:-$SCRIPT_DIR/e2e_models/Qwen3-8B}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
FP16_MODEL="${FP16_MODEL:-$TOKENIZER_MODEL}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-qwen3_8b_all_e2e_$(date +%Y%m%d_%H%M%S)}"
MASTER_DIR="${MASTER_DIR:-$SCRIPT_DIR/benchmark_results/$RUN_TAG}"
QUANT_DIR="$MASTER_DIR/quant_nvfp4"
FP16_DIR="$MASTER_DIR/fp16"

INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
WARMUPS="${WARMUPS:-2}"
REPEATS="${REPEATS:-5}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
NVML_INTERVAL_MS="${NVML_INTERVAL_MS:-20}"
QUANT_GPU_MEMORY_UTILIZATION="${QUANT_GPU_MEMORY_UTILIZATION:-0.30}"
# FP16 weights alone need substantially more memory than the NVFP4 exports.
FP16_GPU_MEMORY_UTILIZATION="${FP16_GPU_MEMORY_UTILIZATION:-0.70}"
INTER_STAGE_COOLDOWN_SECONDS="${INTER_STAGE_COOLDOWN_SECONDS:-8}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi
for path in "$QUANT_LAUNCHER" "$BENCHMARK_PY"; do
    [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done
[[ ! -e "$MASTER_DIR" ]] || {
    echo "ERROR: result directory already exists: $MASTER_DIR" >&2
    exit 1
}

for spec in \
    "fp16=$FP16_MODEL" \
    "rtn=$MODEL_ROOT/rtn_fixed" \
    "gptq=$MODEL_ROOT/gptq" \
    "mr_gptq=$MODEL_ROOT/mr_gptq" \
    "ours=$MODEL_ROOT/ours"; do
    name="${spec%%=*}"
    path="${spec#*=}"
    [[ -f "$path/config.json" ]] || {
        echo "ERROR: missing config for $name: $path/config.json" >&2
        exit 1
    }
    find -L "$path" -maxdepth 1 -type f -name '*.safetensors' \
        -print -quit 2>/dev/null | grep -q . || {
        echo "ERROR: missing safetensors for $name: $path" >&2
        exit 1
    }
done
[[ -f "$TOKENIZER_MODEL/tokenizer_config.json" ]] || {
    echo "ERROR: invalid tokenizer: $TOKENIZER_MODEL" >&2
    exit 1
}

mkdir -p "$MASTER_DIR"
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

echo "MASTER_DIR=$MASTER_DIR"
echo "GPU_ID=$GPU_ID"
echo "CONFIG=input=${INPUT_TOKENS},output=${OUTPUT_TOKENS},warmups=${WARMUPS},repeats=${REPEATS},cudagraph=True"

echo "===== QUANTIZED NVFP4 (RTN/GPTQ/MR-GPTQ/Ours) ====="
OUTPUT_DIR="$QUANT_DIR" \
ENABLE_CUDAGRAPH=1 \
GPU_ID="$GPU_ID" \
GPU_MEMORY_UTILIZATION="$QUANT_GPU_MEMORY_UTILIZATION" \
INPUT_TOKENS="$INPUT_TOKENS" \
OUTPUT_TOKENS="$OUTPUT_TOKENS" \
WARMUPS="$WARMUPS" \
REPEATS="$REPEATS" \
MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
NVML_INTERVAL_MS="$NVML_INTERVAL_MS" \
TOKENIZER_MODEL="$TOKENIZER_MODEL" \
RUN_TAG="${RUN_TAG}_quant" \
bash "$QUANT_LAUNCHER"

echo "Waiting ${INTER_STAGE_COOLDOWN_SECONDS}s for vLLM worker cleanup before FP16..."
sleep "$INTER_STAGE_COOLDOWN_SECONDS"

echo "===== ORIGINAL FP16 ====="
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$BENCHMARK_PY" \
    --model "fp16=$FP16_MODEL" \
    --tokenizer "$TOKENIZER_MODEL" \
    --gpu "$GPU_ID" \
    --dtype float16 \
    --input-tokens "$INPUT_TOKENS" \
    --output-tokens "$OUTPUT_TOKENS" \
    --batch-sizes 1 2 4 8 16 \
    --warmups "$WARMUPS" \
    --repeats "$REPEATS" \
    --gpu-memory-utilization "$FP16_GPU_MEMORY_UTILIZATION" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --nvml-interval-ms "$NVML_INTERVAL_MS" \
    --enable-cudagraph \
    --reference-method fp16 \
    --output-dir "$FP16_DIR"

# The FP16 run needs a higher KV-cache reservation to fit its weights, so its
# absolute NVML peak is a deployment measurement, not a direct weight-memory
# comparison with the NVFP4 run. Throughput settings are otherwise identical.
awk 'FNR == 1 && NR != 1 {next} {print}' \
    "$QUANT_DIR/summary.csv" "$FP16_DIR/summary.csv" \
    > "$MASTER_DIR/all_summary.csv"

echo "ALL_SUMMARY=$MASTER_DIR/all_summary.csv"
echo "QUANT_COMPARISON=$QUANT_DIR/comparison_vs_reference.csv"
echo "FP16_SUMMARY=$FP16_DIR/summary.csv"
echo "ALL_E2E_BENCHMARK_COMPLETE=$MASTER_DIR"
