import json
import os
import sys
import time

import torch
from safetensors import safe_open
from vllm._custom_ops import (
    nvfp4_lss_quant_identity,
    nvfp4_lss_quant_permute_scale,
    permute_and_scale_cols,
)

model_dir = sys.argv[1]
index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
key = next(k for k in index["weight_map"] if k.endswith("input_perm"))
scale_key = key.replace("input_perm", "input_rescale")
shard = os.path.join(model_dir, index["weight_map"][key])

with safe_open(shard, framework="pt", device="cpu") as f:
    perm = f.get_tensor(key).cuda()
    scale = f.get_tensor(scale_key).cuda()

print("test_key:", key)
print("channels:", perm.numel(), "scale_dtype:", scale.dtype)

def timed(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters

for rows in (1, 16, 128):
    x = torch.randn(rows, perm.numel(), device="cuda", dtype=torch.bfloat16)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    x_ref = permute_and_scale_cols(x, perm, scale)
    q_ref, s_ref = nvfp4_lss_quant_identity(x_ref, global_scale)
    q_fused, s_fused = nvfp4_lss_quant_permute_scale(
        x, perm, scale, global_scale
    )
    torch.cuda.synchronize()

    print(
        f"rows={rows}: "
        f"fp4_codes_equal={torch.equal(q_ref, q_fused)}, "
        f"fp8_scales_equal={torch.equal(s_ref, s_fused)}, "
        f"scale_max_abs_diff={(s_ref.float() - s_fused.float()).abs().max().item():.8g}"
    )

    old_ms = timed(
        lambda: nvfp4_lss_quant_identity(
            permute_and_scale_cols(x, perm, scale), global_scale
        )
    )
    fused_ms = timed(
        lambda: nvfp4_lss_quant_permute_scale(
            x, perm, scale, global_scale
        )
    )
    lss_ms = timed(lambda: nvfp4_lss_quant_identity(x, global_scale))

    print(
        f"rows={rows}: old={old_ms:.4f} ms, "
        f"fused_A={fused_ms:.4f} ms, "
        f"lss_only={lss_ms:.4f} ms"
    )

x = torch.randn(16, perm.numel(), device="cuda", dtype=torch.bfloat16)
global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(20):
        nvfp4_lss_quant_permute_scale(x, perm, scale, global_scale)
    torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=15
))
