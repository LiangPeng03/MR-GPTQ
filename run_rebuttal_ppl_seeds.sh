#!/usr/bin/env bash
# Run from this checkout with the existing MR-GPTQ environment.
# Usage: bash run_rebuttal_ppl_seeds.sh {preflight|run} {small|8b} {rtn|gptq|mr-gptq|nada}
#        GPU_ID=0 bash run_rebuttal_ppl_seeds.sh queue
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
action=${1:-preflight}
group=${2:-small}
method=${3:-gptq}
python_bin=${PYTHON_BIN:-"$HOME/.conda/envs/awq/bin/python"}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

if [[ "$action" == queue ]]; then
    launcher="$PWD/run_rebuttal_ppl_seeds.sh"
    export OUTPUT_ROOT
    OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/rebuttal_runs"}
    bash "$launcher" preflight small mr-gptq
    bash "$launcher" preflight 8b mr-gptq
    mkdir -p -- "$OUTPUT_ROOT"
    export CALIBRATION_TOKEN_DIR
    CALIBRATION_TOKEN_DIR=$(mktemp -d "$OUTPUT_ROOT/calibration_c4_128x2048_XXXXXX")
    echo "Shared calibration tokens: $CALIBRATION_TOKEN_DIR"
    failed=0
    for job in 'small gptq' 'small mr-gptq' '8b nada' '8b gptq' '8b mr-gptq' 'small rtn' '8b rtn'; do
        read -r queue_group queue_method <<< "$job"
        bash "$launcher" run "$queue_group" "$queue_method" || failed=1
    done
    exit "$failed"
fi

case "$group" in
    small)
        models=(HuggingFaceTB/SmolLM2-135M Qwen/Qwen3-0.6B meta-llama/Llama-3.2-1B Qwen/Qwen3-1.7B)
        seeds=(42 50 100 163)
        ;;
    8b)
        models=(meta-llama/Meta-Llama-3-8B Qwen/Qwen3-8B)
        seeds=(0 42 50 100 163)
        ;;
    *) echo "Unknown model group: $group" >&2; exit 2 ;;
esac

transform=identity
weight_observer=mse
act_observer=minmax
resort=none
rescale=none
gptq=()
case "$method" in
    rtn) ;;
    gptq) gptq=(--gptq) ;;
    mr-gptq) transform=hadamard; gptq=(--gptq) ;;
    nada)
        if [[ "$group" == small ]]; then
            echo "Small-model NADA results are already in Appendix M." >&2
            exit 2
        fi
        weight_observer=mse_n
        act_observer=lss
        resort=channel_cluster
        rescale=gics
        gptq=(--gptq)
        ;;
    *) echo "Unknown method: $method" >&2; exit 2 ;;
esac

output_root=${OUTPUT_ROOT:-"$PWD/rebuttal_runs"}

case "$action" in
    preflight)
        [[ -x "$python_bin" ]]
        [[ -d "$output_root" && -w "$output_root" || ! -e "$output_root" && -w "$(dirname -- "$output_root")" ]]
        printf 'host=%s\ncommit=%s\nmethod=%s\ngroup=%s\n' "$(hostname)" "$(git rev-parse --short HEAD)" "$method" "$group"
        nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
        "$python_bin" model_quant.py --help >/dev/null
        "$python_bin" - "$method" "${models[@]}" <<'PY'
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

method, *models = sys.argv[1:]
assert torch.cuda.is_available(), "CUDA is unavailable for the selected GPU"
print("visible GPU:", torch.cuda.get_device_name(0))
if method == "mr-gptq":
    from fast_hadamard_transform import hadamard_transform

    hadamard_transform(torch.ones((1, 16), device="cuda", dtype=torch.bfloat16))
for model in models:
    snapshot = snapshot_download(model, local_files_only=True)
    assert any(Path(snapshot).glob("*.safetensors")), snapshot
    AutoConfig.from_pretrained(model, local_files_only=True)
    AutoTokenizer.from_pretrained(model, local_files_only=True)
    print("model cache OK:", model)
for split, shard in (
    ("train", "en/c4-train.00000-of-01024.json.gz"),
    ("validation", "en/c4-validation.00000-of-00008.json.gz"),
):
    dataset = load_dataset("allenai/c4", data_files={split: shard}, split=split)
    print("C4", split, len(dataset))
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
print("WikiText-2 test", len(wiki))
PY
        echo "PREFLIGHT OK"
        exit 0
        ;;
    run) ;;
    *) echo "Expected preflight or run" >&2; exit 2 ;;
esac

mkdir -p -- "$output_root"
run_dir=$(mktemp -d "$output_root/$(date -u +%Y%m%dT%H%M%SZ)_${group}_${method}_XXXXXX")
git rev-parse HEAD > "$run_dir/commit.txt"
printf 'method\tmodel\tseed\twikitext2_ppl\tc4_ppl\tstatus\tlog\n' > "$run_dir/summary.tsv"
printf 'PID=%s GPU=%s results=%s\n' "$$" "$CUDA_VISIBLE_DEVICES" "$run_dir"
failed=0
for seed in "${seeds[@]}"; do
    for model in "${models[@]}"; do
        stem="${model##*/}_seed${seed}"
        log="$run_dir/$stem.log"
        command=(
            "$python_bin" -u model_quant.py
            --model_name_or_path "$model" --format nvfp --w_bits 4 --a_bits 4
            --seed "$seed" --w_group_size 16 --a_group_size 16
            --transform_class "$transform" --w_observer "$weight_observer" --a_observer "$act_observer"
            --quantization_order activation --hadamard_group_size 16
            --dataset_name_or_path c4 --num_sequences 128 --rel_damp 0.01
            --sequence_length 2048 --dtype bfloat16 --fuse_global_scale
            --channel_resort "$resort" --channel_rescale "$rescale"
            --kmeans_block_size -1 --kmeans_alpha 2 --eval_perplexity "${gptq[@]}"
        )
        if [[ -n "${CALIBRATION_TOKEN_DIR:-}" ]]; then
            command+=(--save_calibration_tokens "$CALIBRATION_TOKEN_DIR/$stem.pt")
        fi
        printf '%q ' "${command[@]}" > "$run_dir/$stem.cmd"
        printf '\n' >> "$run_dir/$stem.cmd"
        printf 'START %s %s seed=%s %s\n' "$method" "$model" "$seed" "$log"
        if "${command[@]}" > "$log" 2>&1 && scores=$("$python_bin" - "$log" <<'PY'
import math
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
scores = []
for label in ("Wikitext-2 perplexity:", "C4 perplexity:"):
    matches = re.findall(rf"^{re.escape(label)}\s+(\S+)", log, re.MULTILINE)
    if len(matches) != 1:
        raise SystemExit(f"Expected one {label!r} line, found {len(matches)}")
    scores.append(float(matches[0]))
if not all(isinstance(score, (int, float)) and math.isfinite(score) and score > 0 for score in scores):
    raise SystemExit(f"Invalid PPL: {scores!r}")
print(*scores, sep="\t")
PY
        ); then
            status=OK
        else
            status=FAILED
            scores=$'NA\tNA'
            failed=1
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$method" "$model" "$seed" "$scores" "$status" "$log" |
            tee -a "$run_dir/summary.tsv"
    done
done
exit "$failed"
