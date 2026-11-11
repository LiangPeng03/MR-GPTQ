"""Opt-in diagnostics; no checkpoint exports or changes to default NADA."""
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .qlinear import QLinear
from .quantizer import Quantizer


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def decode_codes(packed, device):
    """Decode E2M1 nibbles; retain the exact packed bytes as the invariant."""
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                          -0., -.5, -1, -1.5, -2, -3, -4, -6], device=device)
    packed = packed.to(device)
    indices = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    return table[indices].flatten(-2)


@torch.no_grad()
def refit_layer(layer, max_groups=8):
    """One sparse coordinate sweep of E4M3 scales; freeze FP4/global scale.

    Two tokens per original calibration sequence were captured. First 3/4
    sequences fit the update; the last 1/4 accept/reject the whole linear layer.
    These are internal update-validation samples, not unseen GPTQ samples.
    """
    meta = layer._nada_diag
    device, dtype = layer.weight.device, layer.weight.dtype
    codes = decode_codes(meta["codes"], device)
    scales = meta["scales"].to(device).clone()
    gamma = meta["global_scale"].to(device)
    original = layer.weight.detach().clone()
    if not torch.isfinite(gamma).all() or not (gamma > 0).all():
        raise ValueError("Invalid frozen weight global scale")
    reconstructed = (codes.reshape(*scales.shape, 16) * scales[..., None]).flatten(-2).to(dtype)
    if not torch.equal(original, reconstructed):
        raise RuntimeError("Captured FP4 codes/scales do not reconstruct the GPTQ weights exactly")

    chunks = meta["samples"]
    split = min(len(chunks) - 1, max(1, 3 * len(chunks) // 4))
    x_fit = torch.cat(chunks[:split]).to(device)
    x_val = torch.cat(chunks[split:]).to(device)
    aq = layer.act_quantizer
    if aq is None or aq.observer.value != "lss" or aq._track_global_scale:
        raise RuntimeError("Post-GPTQ refit requires original frozen ABSO/LSS")

    def inputs(x):
        s, z = aq.get_quantization_params(x)
        return aq(x, s, z).float()

    xq_fit, xq_val = inputs(x_fit), inputs(x_val)
    reference = meta["reference_weight"].to(device).float()
    fit_residual = F.linear(x_fit.float(), reference) - F.linear(xq_fit, original.float())
    val_residual = F.linear(x_val.float(), reference) - F.linear(xq_val, original.float())
    before_fit = fit_residual.square().mean().item()
    before_val = val_residual.square().mean().item()
    importance = xq_fit.square().mean(0)
    priority = ((original.float() - reference).square() * importance).reshape(
        original.shape[0], -1, 16).sum((0, 2))
    groups = priority.topk(min(max_groups, priority.numel())).indices.tolist()
    table = torch.arange(1, 127, device=device, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    changed = 0
    for group in groups:
        sl = slice(group * 16, (group + 1) * 16)
        old_block = layer.weight[:, sl].detach().clone()
        code_block = codes[:, sl]
        old_fit = F.linear(xq_fit[:, sl], old_block.float())
        basis = F.linear(xq_fit[:, sl], code_block)
        target = fit_residual + old_fit
        denominator = basis.square().sum(0)
        proposal = torch.where(denominator > 0,
                               (basis * target).sum(0) / denominator.clamp_min(1e-30),
                               scales[:, group])
        raw = (proposal * gamma).clamp(table[0], table[-1])
        upper = torch.searchsorted(table, raw.contiguous()).clamp(0, 125)
        lower = (torch.searchsorted(table, raw.contiguous(), right=True) - 1).clamp(0, 125)
        candidates = (scales[:, group], table[lower] / gamma, table[upper] / gamma)
        best_loss = fit_residual.square().sum(0)
        best_block, best_scale = old_block.clone(), scales[:, group].clone()
        for candidate in candidates[1:]:
            block = (code_block * candidate[:, None]).to(dtype)
            delta = F.linear(xq_fit[:, sl], block.float()) - old_fit
            loss = (fit_residual - delta).square().sum(0)
            improve = loss < best_loss
            best_loss = torch.where(improve, loss, best_loss)
            best_block[improve] = block[improve]
            best_scale = torch.where(improve, candidate, best_scale)
        difference = best_block.float() - old_block.float()
        fit_residual -= F.linear(xq_fit[:, sl], difference)
        val_residual -= F.linear(xq_val[:, sl], difference)
        changed += int((best_scale != scales[:, group]).sum().item())
        layer.weight[:, sl] = best_block
        scales[:, group] = best_scale

    # Verify the full output objectives rather than relying on accumulated deltas.
    after_fit = (F.linear(x_fit.float(), reference) - F.linear(xq_fit, layer.weight.float())).square().mean().item()
    after_val = (F.linear(x_val.float(), reference) - F.linear(xq_val, layer.weight.float())).square().mean().item()
    accepted = (all(math.isfinite(v) for v in (after_fit, after_val)) and changed > 0
                and after_fit <= before_fit + max(1e-9, before_fit * 1e-6)
                and after_val < before_val - max(1e-9, before_val * 1e-6))
    if accepted:
        meta["scales"] = scales.cpu()
    else:
        layer.weight.copy_(original)
    return dict(fit_before=before_fit, fit_proposed=after_fit,
                validation_before=before_val, validation_proposed=after_val,
                accepted=accepted, changed_scales=changed if accepted else 0,
                groups_tried=len(groups), fit_sequences=split,
                validation_sequences=len(chunks) - split)


@torch.no_grad()
def run_diagnostics(model, tokenizer, args):
    from ..metrics.perplexity import compute_perplexity
    from ..utils.data_utils import get_wikitext2, get_c4_eval

    output = Path(args.nada_diagnostics)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.tsv").exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    linears = [(name, module) for name, module in model.named_modules() if isinstance(module, QLinear)]
    if not linears or any(not hasattr(layer, "_nada_diag") for _, layer in linears):
        raise RuntimeError("Missing post-GPTQ diagnostic tensors")
    originals = {name: layer.act_quantizer for name, layer in linears}
    before_state = tensor_hash(model.state_dict().items())
    before_codes = tensor_hash((name, layer._nada_diag["codes"]) for name, layer in linears)
    global_hash = tensor_hash((name, aq.global_scale) for name, aq in originals.items())
    datasets = {"wikitext2": get_wikitext2(tokenizer, args.sequence_length),
                "c4": get_c4_eval(tokenizer, args.sequence_length)}
    rows, updates = [], []
    with (output / "summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "wikitext2_ppl", "c4_ppl",
                                "model_state_sha256", "fp4_codes_sha256", "activation_global_sha256"], delimiter="\t")
        writer.writeheader()
        handle.flush()
        try:
            for mode in ("NADA_WMSE_W_ABSO", "FIXED_W_A16", "FIXED_W_A_SCALESWEEP_WMSE", "POST_GPTQ_SCALE_ABSO"):
                for name, layer in linears:
                    layer.act_quantizer = originals[name]
                if mode == "FIXED_W_A16":
                    for _, layer in linears:
                        layer.act_quantizer = None
                elif mode == "FIXED_W_A_SCALESWEEP_WMSE":
                    shared = {}
                    for name, layer in linears:
                        old = originals[name]
                        if id(old) not in shared:
                            new = Quantizer(bits=4, format="nvfp", granularity="group", group_size=16,
                                            scale_precision="e4m3", observer="scalesweep_wmse",
                                            global_scale_max=448)
                            new.global_scale = old.global_scale.clone()
                            new._track_global_scale = False
                            new.importance = layer._nada_diag_importance.to(layer.weight.device)
                            shared[id(old)] = new
                        layer.act_quantizer = shared[id(old)]
                elif mode == "POST_GPTQ_SCALE_ABSO":
                    for name, layer in linears:
                        result = refit_layer(layer, args.nada_refit_groups)
                        updates.append(dict(layer=name, **result))
                        print("[POST_GPTQ_REFIT]", name, json.dumps(result), flush=True)
                        (output / "scale_updates.json").write_text(json.dumps(updates, indent=2))

                state_hash = tensor_hash(model.state_dict().items())
                codes_hash = tensor_hash((name, layer._nada_diag["codes"]) for name, layer in linears)
                assert codes_hash == before_codes, "FP4 codes changed"
                if mode != "POST_GPTQ_SCALE_ABSO":
                    assert state_hash == before_state, "Fixed-weight diagnostic changed model tensors"
                for name, layer in linears:
                    if layer.act_quantizer is not None:
                        assert torch.equal(layer.act_quantizer.global_scale, originals[name].global_scale)
                row = dict(method=mode, model_state_sha256=state_hash,
                           fp4_codes_sha256=codes_hash, activation_global_sha256=global_hash)
                print("START DIAGNOSTIC", mode, flush=True)
                for dataset, data in datasets.items():
                    row[dataset + "_ppl"] = compute_perplexity(model, data, batch_size=1)
                assert tensor_hash((name, aq.global_scale) for name, aq in originals.items()) == global_hash, "Original activation globals changed"
                for name, layer in linears:
                    if layer.act_quantizer is not None:
                        assert torch.equal(layer.act_quantizer.global_scale, originals[name].global_scale)
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print("[DIAGNOSTIC_RESULT]", json.dumps(row), flush=True)
        finally:
            for name, layer in linears:
                layer.act_quantizer = originals[name]
    (output / "results.json").write_text(json.dumps(dict(
        reference="nada-wmse-w, not original NADA", seed=args.seed,
        calibration_sequences=args.num_sequences, sequence_length=args.sequence_length,
        selection="calibration only; sparse one-sweep scale refit",
        refit_groups_per_layer=args.nada_refit_groups, results=rows, updates=updates), indent=2))
    print("DIAGNOSTICS DONE:", output / "summary.tsv", flush=True)


@torch.no_grad()
def validate_diagnostics(device="cuda"):
    """Small real-operator check before loading/quantizing a model."""
    from .gptq import GPTQ
    from .quant_ops import pack_fp4_to_uint8

    torch.manual_seed(0)
    kwargs = dict(bits=4, format="nvfp", granularity="group", group_size=16,
                  scale_precision="e4m3", observer="scalesweep_wmse", global_scale_max=448)
    w = torch.randn(16, 32, device=device, dtype=torch.bfloat16)
    x = torch.randn(32, 32, device=device, dtype=torch.bfloat16)
    hessian = x.float().T @ x.float() + torch.eye(32, device=device)
    outputs = []
    for export in ("", "realquant"):
        layer = torch.nn.Linear(32, 16, bias=False, device=device, dtype=torch.bfloat16)
        layer.weight.copy_(w)
        quantizer = Quantizer(**kwargs)
        quantizer.global_scale = torch.tensor([32.], device=device)
        quantizer._track_global_scale = False
        handle = GPTQ(layer, quantizer, quantization_order="activation", export_quantized_model=export)
        handle.H = hessian.clone()
        outputs.append(handle.quantize())
    assert torch.equal(outputs[0][0], outputs[1][0]), "Code capture changed GPTQ weights"
    assert torch.equal(outputs[0][2], outputs[1][2]), "Code capture changed GPTQ scales"
    quantized, codes, scales = outputs[1]
    packed = pack_fp4_to_uint8(codes).cpu()
    assert torch.equal(decode_codes(packed, device), codes.float())
    reconstructed = (codes.reshape(16, 2, 16).float() * scales[..., None]).flatten(-2).to(w.dtype)
    assert torch.equal(reconstructed, quantized), "FP4 scale reconstruction mismatch"

    aq = Quantizer(bits=4, format="nvfp", granularity="group", group_size=16,
                   scale_precision="e4m3", observer="lss")
    aq.global_scale = torch.tensor([64.], device=device)
    aq._track_global_scale = False
    layer = QLinear(32, 16, bias=False, device=device, dtype=w.dtype, act_quantizer=aq)
    layer.weight.copy_(quantized)
    layer._train_mode = False
    layer._nada_diag = dict(codes=packed, scales=scales.cpu(),
                            global_scale=torch.tensor([32.]), reference_weight=w.cpu(),
                            samples=[sample.cpu() for sample in x.reshape(8, 4, 32)])
    before_codes = tensor_hash([("codes", packed)])
    result = refit_layer(layer, max_groups=2)
    assert all(math.isfinite(result[key]) for key in
               ("fit_before", "fit_proposed", "validation_before", "validation_proposed"))
    assert tensor_hash([("codes", layer._nada_diag["codes"])]) == before_codes
    if result["accepted"]:
        assert result["validation_proposed"] < result["validation_before"]
    else:
        assert torch.equal(layer.weight, quantized), "Rejected refit did not restore weights"
    final = (decode_codes(packed, device).reshape(16, 2, 16)
             * layer._nada_diag["scales"].to(device)[..., None]).flatten(-2).to(w.dtype)
    assert torch.equal(final, layer.weight), "Updated scales do not reconstruct the same-code weights"

    sweep = Quantizer(**kwargs)
    sweep.global_scale = aq.global_scale.clone()
    sweep._track_global_scale = False
    sweep.importance = w.float().square().mean(0)
    s, z = sweep.get_quantization_params(x)
    assert torch.isfinite(sweep(x, s, z)).all()
    assert torch.equal(sweep.global_scale, aq.global_scale)
    print("DIAGNOSTIC PREFLIGHT: code capture/packing/refit/rollback/frozen globals OK", flush=True)
    print("REFIT TEST:", json.dumps(result), flush=True)
