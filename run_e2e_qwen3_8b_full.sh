#!/usr/bin/env bash
# Launch the final Qwen3-8B E2E benchmark in a fixed, reproducible order:
# FP16 -> RTN -> GPTQ -> MR-GPTQ -> Ours.
#
# It starts one background job on one GPU. Every method is measured with
# input/output lengths 512/128, batch sizes 1/2/4/8/16, two warm-ups, and
# five measured repetitions. CUDA Graph is enabled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/vptq/bin/python}"
GPU_ID="${GPU_ID:-0}"

# Override this when benchmarking a later re-export.
SAVE_ROOT="${SAVE_ROOT:-$SCRIPT_DIR/e2e_models/Qwen3-8B_reexport_20260820_224623}"
FP16_MODEL="${FP16_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OURS_MODEL="${OURS_MODEL:-$SCRIPT_DIR/e2e_models/qwen3_8b_ours_lss_validated_20260904_211031/ours}"

RUN_TAG="${RUN_TAG:-qwen3_8b_reexport_final_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$SCRIPT_DIR/benchmark_results/$RUN_TAG"
LOG="$SCRIPT_DIR/benchmark_${RUN_TAG}.log"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
FP16_GPU_MEMORY_UTILIZATION="${FP16_GPU_MEMORY_UTILIZATION:-0.80}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: refusing to overwrite result directory: $OUTPUT_DIR" >&2
    exit 2
fi

for method in rtn gptq mr_gptq; do
    model_path="$SAVE_ROOT/$method"
    if [[ ! -f "$model_path/config.json" ||
          ! -f "$model_path/model.safetensors.index.json" ]]; then
        echo "ERROR: invalid checkpoint for $method: $model_path" >&2
        exit 1
    fi
done
if [[ ! -f "$OURS_MODEL/config.json" ||
      ! -f "$OURS_MODEL/model.safetensors.index.json" ]]; then
    echo "ERROR: invalid checkpoint for ours: $OURS_MODEL" >&2
    exit 1
fi
if [[ ! -f "$FP16_MODEL/config.json" ]]; then
    echo "ERROR: invalid FP16 checkpoint: $FP16_MODEL" >&2
    exit 1
fi

"$PYTHON_BIN" - "$OURS_MODEL/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)

quantization_config = config.get("quantization_config", {})
assert quantization_config.get("forward_method") == "lss", quantization_config

rope_parameters = config.get("rope_parameters", {})
rope_theta = config.get("rope_theta", rope_parameters.get("rope_theta"))
assert rope_theta == 1_000_000, (rope_theta, rope_parameters)

print("OURS_FORWARD_METHOD=lss")
print(f"OURS_ROPE_THETA={rope_theta}")
print("OURS_DEPLOYMENT_PREFLIGHT_OK")
PY

echo "== GPU status before launch =="
nvidia-smi
echo "RUN_TAG=$RUN_TAG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG=$LOG"
echo "ORDER=fp16,rtn,gptq,mr_gptq,ours"
echo "SETTINGS=input=512 output=128 batches=1,2,4,8,16 warmups=2 repeats=5"
echo "MEMORY_UTILIZATION=quantized:$GPU_MEMORY_UTILIZATION fp16:$FP16_GPU_MEMORY_UTILIZATION"
echo "OURS_MODEL=$OURS_MODEL"

nohup env \
    -u VLLM_FPQUANT_TORCH_REFERENCE \
    -u VLLM_FPQUANT_TORCH_REFERENCE_SPLIT \
    VLLM_USE_V1=1 \
    ENABLE_CUDAGRAPH=1 \
    GPU_ID="$GPU_ID" \
    RUN_TAG="$RUN_TAG" \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    FP16_GPU_MEMORY_UTILIZATION="$FP16_GPU_MEMORY_UTILIZATION" \
    bash "$SCRIPT_DIR/run_e2e_qwen3_8b.sh" \
        "fp16=$FP16_MODEL" \
        "rtn=$SAVE_ROOT/rtn" \
        "gptq=$SAVE_ROOT/gptq" \
        "mr_gptq=$SAVE_ROOT/mr_gptq" \
        "ours=$OURS_MODEL" \
    >"$LOG" 2>&1 &

BENCH_PID=$!
echo "BENCH_PID=$BENCH_PID"
echo "Monitor: tail -40 $LOG"
echo "Results: $OUTPUT_DIR/all_summary.csv"
