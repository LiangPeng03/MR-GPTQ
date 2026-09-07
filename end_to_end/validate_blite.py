"""Two-phase bit-exactness validation for the B-lite kernel change.

Phase 1 (old _C.abi3.so still installed): compute reference outputs and
torch.save them to /tmp/blite_ref.pt.
Phase 2 (new _C.abi3.so installed): recompute and compare with torch.equal.

Usage: python validate_blite.py <phase1|phase2>
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from vllm._custom_ops import (
    nvfp4_lss_quant_identity,
    nvfp4_lss_quant_permute_scale,
    permute_and_scale_cols,
)

MODEL_DIR = (
    "/home/pengliang/Desktop/MR-GPTQ/e2e_models/"
    "qwen3_8b_ours_lss_validated_20260904_211031/ours"
)
REF_PATH = "/tmp/blite_ref.pt"

phase = sys.argv[1]
torch.manual_seed(0)

index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
key = next(k for k in index["weight_map"] if k.endswith("input_perm"))
scale_key = key.replace("input_perm", "input_rescale")
gs_key = key.replace(".input_perm", ".act_global_scale")
shard = os.path.join(MODEL_DIR, index["weight_map"][key])
with safe_open(shard, framework="pt", device="cpu") as f:
    perm = f.get_tensor(key).cuda()
    scale = f.get_tensor(scale_key).cuda()
    act_gs = f.get_tensor(gs_key).cuda()  # real global scale

print(f"key: {key}, channels: {perm.numel()}, act_global_scale: {act_gs.item():.6g}")

n = perm.numel()


def make_inputs():
    tensors = {}
    for rows in (1, 16, 128):
        tensors[f"rand_{rows}"] = torch.randn(rows, n, device="cuda", dtype=torch.bfloat16)
    # Edge cases (single tensor, 128 rows):
    e = torch.randn(128, n, device="cuda", dtype=torch.bfloat16)
    e[:16] = 0.0  # all-zero groups
    e[16:32] *= 1e4  # huge -> E4M3 clamp at 448
    e[32:48] *= 1e-4  # tiny -> near-denormal scales
    tensors["edge"] = e
    return tensors


results = {}
for gs_name, gs in (("gs_real", act_gs.clone()), ("gs_one", torch.ones(1, device="cuda"))):
    for name, x in make_inputs().items():
        # identity path
        q_id, s_id = nvfp4_lss_quant_identity(x, gs)
        # permute/scale path
        q_ps, s_ps = nvfp4_lss_quant_permute_scale(x, perm, scale, gs)
        # cross-check: permute_and_scale_cols + identity (invariant of A step)
        x_t = permute_and_scale_cols(x, perm, scale)
        q_x, s_x = nvfp4_lss_quant_identity(x_t, gs)
        results[f"{gs_name}/{name}/identity_q"] = q_id.cpu()
        results[f"{gs_name}/{name}/identity_s"] = s_id.cpu()
        results[f"{gs_name}/{name}/ps_q"] = q_ps.cpu()
        results[f"{gs_name}/{name}/ps_s"] = s_ps.cpu()
        results[f"{gs_name}/{name}/cross_ok"] = torch.tensor(
            torch.equal(q_x, q_ps) and torch.equal(s_x, s_ps)
        )

if phase == "phase1":
    torch.save(results, REF_PATH)
    print(f"saved {len(results)} tensors to {REF_PATH}")
else:
    ref = torch.load(REF_PATH)
    assert set(ref) == set(results), "key mismatch"
    all_ok = True
    for k, v in ref.items():
        if not torch.equal(v, results[k]):
            all_ok = False
            if v.dtype.is_floating_point:
                print(f"MISMATCH {k}: max_abs_diff={(v.float()-results[k].float()).abs().max().item():.8g}")
            else:
                nd = (v != results[k]).sum().item()
                print(f"MISMATCH {k}: {nd}/{v.numel()} elements differ")
    print("ALL_EQUAL:", all_ok)
