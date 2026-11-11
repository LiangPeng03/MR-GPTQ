"""Two-phase bit-exactness validation for the B-lite kernel change.

Phase 1 (old _C.abi3.so still installed): compute reference outputs and
torch.save them to /tmp/blite_ref.pt.
Phase 2 (new _C.abi3.so installed): recompute and compare with torch.equal.

Usage: python validate_blite.py <phase1|phase2|scalesweep|shared-prepack>
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


@torch.no_grad()
def validate_shared_prepack():
    """Bit-exact legacy/shared-cache comparison, including decode CUDA Graphs.

    Tests the real loader/apply dispatch, not a different quantization reference.
    No checkpoint is read or written and no CUDA extension rebuild is needed.
    """
    from vllm._custom_ops import fusedQuantizeNv
    from vllm.model_executor.layers.quantization.fp_quant import (
        FPQuantConfig, FPQuantLinearMethod,
    )
    from vllm.model_executor.layers.quantization import fp_quant

    keys = ("VLLM_FPQUANT_SHARED_PREPACK", "VLLM_FPQUANT_NADA_FAST",
            "VLLM_FPQUANT_TORCH_REFERENCE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["VLLM_FPQUANT_NADA_FAST"] = "0"
        os.environ["VLLM_FPQUANT_TORCH_REFERENCE"] = "0"
        torch.manual_seed(0)
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print("FPQuant source:", fp_quant.__file__, flush=True)
        h = torch.ones(1, 1, device="cuda", dtype=torch.bfloat16)
        h2 = torch.tensor([[1., 1.], [1., -1.]], device="cuda",
                          dtype=torch.bfloat16)
        for _ in range(4):
            h = torch.kron(h, h2)
        h = (h / 4).contiguous()
        identity = torch.eye(16, device="cuda", dtype=torch.bfloat16)
        modes = (("rtn/gptq", "abs_max", identity),
                 ("mr_gptq", "abs_max", h),
                 ("nada_common", "lss", identity),
                 ("scalesweep_mse", "scalesweep_mse", identity),
                 ("scalesweep_wmse", "scalesweep_wmse", identity))
        for width in (2048, 4096, 6144, 12288, 14336):
            a_gs = torch.tensor([64.], device="cuda")
            w_gs = torch.tensor([32.], device="cuda")
            weight = torch.randn(128, width, device="cuda", dtype=torch.bfloat16)
            qw, ws = fusedQuantizeNv(weight, identity, w_gs)
            ws = ws[:128, :width // 16].contiguous().view(torch.uint8)
            for name, mode, transform in modes:
                layer = torch.nn.Module()
                for key, value in {
                    "qweight": qw, "scales": ws,
                    "weight_global_scale": w_gs, "act_global_scale": a_gs,
                    "forward_hadamard_matrix": transform,
                    "input_perm": torch.randperm(width, device="cuda"),
                    "input_rescale": torch.linspace(.5, 1.5, width, device="cuda").bfloat16(),
                }.items():
                    layer.register_buffer(key, value)
                layer.input_transform_enabled = mode == "lss"
                if mode == "scalesweep_wmse":
                    layer.register_buffer("act_importance", torch.linspace(
                        .1, 3., width, device="cuda"))
                    layer.act_importance._scalesweep_loaded = True
                method = FPQuantLinearMethod(FPQuantConfig(
                    hadamard_group_size=16, forward_dtype="nvfp4",
                    forward_method=mode, lss_identity_transform=mode == "lss",
                    scalesweep_identity_transform=mode.startswith("scalesweep_"),
                ))
                os.environ["VLLM_FPQUANT_SHARED_PREPACK"] = "0"
                method.process_weights_after_loading(layer)
                assert not layer._fpquant_prepack_enabled
                os.environ["VLLM_FPQUANT_SHARED_PREPACK"] = "1"
                method.process_weights_after_loading(layer)
                assert layer._fpquant_prepack_enabled
                assert not layer._nada_fast_enabled
                bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
                for rows in (1, 2, 4, 8, 16, 128, 512):
                    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                    layer._fpquant_prepack_enabled = False
                    expected = method.apply(layer, x, bias)
                    layer._fpquant_prepack_enabled = True
                    actual = method.apply(layer, x, bias)
                    assert torch.equal(expected, actual), (name, width, rows, "eager")
                    # Warm custom ops/Triton before graph capture.
                    for _ in range(3):
                        method.apply(layer, x, bias)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = method.apply(layer, x, bias)
                    for _ in range(2):
                        graph.replay()
                        torch.cuda.synchronize()
                        assert torch.equal(expected, captured), (name, width, rows, "graph")
                    del graph, captured, expected, actual
                print(f"SHARED_PREPACK_OK method={name} K={width} "
                      "M=1,2,4,8,16,128,512 eager/graph bit-exact", flush=True)
                del layer, method
        print("ALL SHARED PREPACK CHECKS PASSED", flush=True)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if len(sys.argv) > 1 and sys.argv[1] == "shared-prepack":
    validate_shared_prepack()
    raise SystemExit(0)

if len(sys.argv) > 1 and sys.argv[1] == "scalesweep":
    from vllm.model_executor.layers.quantization import fp_quant
    from vllm.model_executor.layers.quantization.utils.scalesweep import validate_runtime

    validate_runtime()
    raise SystemExit(0)

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
