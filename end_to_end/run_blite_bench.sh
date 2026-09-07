#!/bin/bash
# Full e2e vLLM benchmark through benchmark_e2e.py (repo root).
# Usage: bash end_to_end/run_blite_bench.sh   (from anywhere)
REPO=/home/pengliang/Desktop/MR-GPTQ
cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES=0
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
"$HOME/.conda/envs/vptq/bin/python" benchmark_e2e.py \
  --model ours_lss_blite=/home/pengliang/Desktop/MR-GPTQ/e2e_models/qwen3_8b_ours_lss_validated_20260904_211031/ours \
  --batch-sizes 1 2 4 8 16 \
  --enable-cudagraph \
  --output-dir "$REPO/benchmark_results/blite_e2e" \
  > /tmp/blite_bench.log 2>&1
echo "bench exit: $?"
