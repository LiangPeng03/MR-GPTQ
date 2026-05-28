import torch
from typing import List, Optional

def run_awq_search(
    X: torch.Tensor,
    weights: List[torch.Tensor],
    weight_quantizer=None,
    act_quantizer=None,
    steps: int = 20,
    X_max: Optional[torch.Tensor] = None,
    use_gajs: bool = False,
) -> torch.Tensor:
    """
    Finds the optimal AWQ diagonal scale `s` for a given input activation `X` 
    and a list of weights (e.g. [q_proj, k_proj, v_proj] that share the same X).
    
    If use_gajs is True, it uses the Grid-Aligned Joint Scaling (Adam) method.
    Otherwise, it uses the baseline 1D grid search.
    """
    device = X.device
    in_features = X.shape[-1]
    
    if len(X.shape) == 3:
        X = X.reshape(-1, in_features)
    
    # Fast path if no quantizers
    if weight_quantizer is None and act_quantizer is None:
        return torch.ones(in_features, device=device, dtype=X.dtype)
        
    if X_max is None:
        X_max = X.abs().max(dim=0).values.clamp(min=1e-4)
        
    # To save VRAM/compute, sample subset if X is large
    n_samples = X.shape[0]
    if n_samples > 2048:
        # Use random permutation for the subset
        idx = torch.randperm(n_samples, device=device)[:2048]
        X_sub = X[idx].detach()
    else:
        X_sub = X.detach()
        
    # Precompute FP16 outputs for MSE comparison
    with torch.no_grad():
        out_origs = [torch.matmul(X_sub, w.t()).detach() for w in weights]
        
    if use_gajs:
        # ---------------------------------------------------------
        # PLAN A (Strict Bound Grid Search): Exact E_max Limits
        # ---------------------------------------------------------
        C = in_features
        s_c_min = torch.zeros(C, device=device, dtype=torch.float32)
        
        # 1. Compute s_c_min from weights (prevent weight group scale inflation)
        for w in weights:
            if weight_quantizer is not None:
                q_scales, _ = weight_quantizer.get_quantization_params(w)
                E_max = (q_scales.detach() * weight_quantizer.q_max).to(torch.float32)
                
                if weight_quantizer.group_size:
                    dim = w.ndim - 1 if weight_quantizer.dim == -1 else weight_quantizer.dim
                    num_groups = w.shape[dim] // weight_quantizer.group_size
                    shape = list(w.shape)
                    shape = shape[:dim] + [num_groups, weight_quantizer.group_size] + shape[dim+1:]
                    w_grouped = w.detach().abs().to(torch.float32).view(shape)
                    E_max = E_max.unsqueeze(dim + 1)
                    ratio = w_grouped / E_max.clamp(min=1e-8)
                    
                    dims_to_reduce = [i for i in range(ratio.ndim) if i != dim and i != dim+1]
                    if dims_to_reduce:
                        ratio_c = ratio.amax(dim=dims_to_reduce)
                    else:
                        ratio_c = ratio
                    ratio_c = ratio_c.reshape(C)
                else:
                    dim = w.ndim - 1 if weight_quantizer.dim == -1 else weight_quantizer.dim
                    ratio = w.detach().abs().to(torch.float32) / E_max.clamp(min=1e-8)
                    dims_to_reduce = [i for i in range(ratio.ndim) if i != dim]
                    if dims_to_reduce:
                        ratio_c = ratio.amax(dim=dims_to_reduce)
                    else:
                        ratio_c = ratio
                    ratio_c = ratio_c.reshape(C)
            else:
                w_abs = w.detach().abs().to(torch.float32)
                ratio_c = (w_abs / w_abs.max().clamp(min=1e-8)).max(dim=0).values.reshape(C)
                
            s_c_min = torch.max(s_c_min, ratio_c)
            
        s_c_min = s_c_min.clamp(min=0.01, max=1.0)
        
        # 2. Compute s_c_max from activations (prevent activation group scale inflation)
        if act_quantizer is not None:
            q_scales, _ = act_quantizer.get_quantization_params(X_sub)
            E_max = (q_scales.detach() * act_quantizer.q_max).to(torch.float32)
            
            if act_quantizer.group_size:
                dim = X_sub.ndim - 1 if act_quantizer.dim == -1 else act_quantizer.dim
                num_groups = X_sub.shape[dim] // act_quantizer.group_size
                shape = list(X_sub.shape)
                shape = shape[:dim] + [num_groups, act_quantizer.group_size] + shape[dim+1:]
                X_grouped = X_sub.detach().abs().to(torch.float32).view(shape)
                E_max = E_max.unsqueeze(dim + 1)
                ratio = E_max.clamp(min=1e-8) / X_grouped.clamp(min=1e-8)
                
                dims_to_reduce = [i for i in range(ratio.ndim) if i != dim and i != dim+1]
                if dims_to_reduce:
                    s_c_max = ratio.amin(dim=dims_to_reduce)
                else:
                    s_c_max = ratio
                s_c_max = s_c_max.reshape(C)
            else:
                dim = X_sub.ndim - 1 if act_quantizer.dim == -1 else act_quantizer.dim
                ratio = E_max.clamp(min=1e-8) / X_sub.detach().abs().to(torch.float32).clamp(min=1e-8)
                dims_to_reduce = [i for i in range(ratio.ndim) if i != dim]
                if dims_to_reduce:
                    s_c_max = ratio.amin(dim=dims_to_reduce)
                else:
                    s_c_max = ratio
                s_c_max = s_c_max.reshape(C)
        else:
            X_abs = X_sub.detach().abs().to(torch.float32)
            X_token_max = X_abs.max(dim=1, keepdim=True).values.clamp(min=1e-8)
            ratio = X_token_max / X_abs.clamp(min=1e-8)
            s_c_max = ratio.min(dim=0).values.reshape(C)
            
        s_c_max = s_c_max.clamp(min=1.0, max=100.0)
        
        best_error = float('inf')
        best_scale = torch.ones(in_features, device=device, dtype=X.dtype)
        best_alpha = -1.0
        
        search_steps = 20 if steps <= 0 else steps
        
        with torch.no_grad():
            # ---- Compute Baseline Loss (s=1.0) ----
            s_base = torch.ones(in_features, device=device, dtype=X.dtype)
            X_scaled_base = X_sub * s_base
            if act_quantizer is not None:
                q_scales, q_zeros = act_quantizer.get_quantization_params(X_scaled_base.unsqueeze(0))
                X_q_base = act_quantizer(X_scaled_base.unsqueeze(0), q_scales, q_zeros).squeeze(0)
            else:
                X_q_base = X_scaled_base
            
            baseline_err = 0.0
            for w, out_orig in zip(weights, out_origs):
                w_scaled_base = w.detach() / s_base.unsqueeze(0)
                if weight_quantizer is not None:
                    w_q_scales, w_q_zeros = weight_quantizer.get_quantization_params(w_scaled_base)
                    w_q_base = weight_quantizer(w_scaled_base, w_q_scales, w_q_zeros)
                else:
                    w_q_base = w_scaled_base
                out_q_base = torch.matmul(X_q_base, w_q_base.t())
                baseline_err += (out_q_base - out_orig).pow(2).mean().item()
                
            best_error = baseline_err
            # ---------------------------------------
            
            for i in range(search_steps + 1):
                alpha = i / search_steps
                
                # Interpolate exactly between safe extremes
                # alpha=0 -> max activation compression (s <= 1)
                # alpha=1 -> max weight compression (s >= 1)
                s = (s_c_min ** (1.0 - alpha)) * (s_c_max ** alpha)
                s = s.to(X.dtype)
                
                # NO mean normalization here! Normalization would break strict boundaries.
                
                # Apply scaling
                X_scaled = X_sub * s
                
                # Quantize X
                if act_quantizer is not None:
                    q_scales, q_zeros = act_quantizer.get_quantization_params(X_scaled.unsqueeze(0))
                    X_q = act_quantizer(X_scaled.unsqueeze(0), q_scales, q_zeros).squeeze(0)
                else:
                    X_q = X_scaled
                    
                total_err = 0.0
                
                for w, out_orig in zip(weights, out_origs):
                    # Scale weight
                    w_scaled = w.detach() / s.unsqueeze(0)
                    
                    # Quantize W
                    if weight_quantizer is not None:
                        w_q_scales, w_q_zeros = weight_quantizer.get_quantization_params(w_scaled)
                        w_q = weight_quantizer(w_scaled, w_q_scales, w_q_zeros)
                    else:
                        w_q = w_scaled
                        
                    # Joint Output
                    out_q = torch.matmul(X_q, w_q.t())
                    
                    # MSE Loss
                    err = (out_q - out_orig).pow(2).mean().item()
                    total_err += err
                    
                # Debug print for first and last step
                if i == 0 or i == search_steps:
                    print(f"    [Strict Grid Step {i}] Loss: {total_err:.4f}, alpha: {alpha:.2f}, s_max: {s.max().item():.4f}, s_min: {s.min().item():.4f}")
                    
                if total_err < best_error:
                    best_error = total_err
                    best_scale = s.clone()
                    best_alpha = alpha
                    
            print(f"      => Baseline Loss: {baseline_err:.4f} -> GAJS Min Loss: {best_error:.4f} (Selected alpha: {best_alpha:.2f})")
                    
        return best_scale
        
    else:
        # ---------------------------------------------------------
        # BASELINE: 1D Grid Search (Original AWQ)
        # ---------------------------------------------------------
        if steps == 0:
            return torch.ones(in_features, device=device, dtype=X.dtype)
            
        best_error = float('inf')
        best_scale = torch.ones_like(X_max)
        
        with torch.no_grad():
            for i in range(steps + 1):
                alpha = i / steps
                s = X_max ** alpha
                
                s = s / s.mean()
                s = torch.clamp(s, min=0.1, max=10.0)
                
                X_scaled = X_sub * s
                
                if act_quantizer is not None:
                    q_scales, q_zeros = act_quantizer.get_quantization_params(X_scaled.unsqueeze(0))
                    X_q = act_quantizer(X_scaled.unsqueeze(0), q_scales, q_zeros).squeeze(0)
                else:
                    X_q = X_scaled
                    
                total_err = 0.0
                
                for w, out_orig in zip(weights, out_origs):
                    w_scaled = w / s.unsqueeze(0)
                    
                    if weight_quantizer is not None:
                        w_q_scales, w_q_zeros = weight_quantizer.get_quantization_params(w_scaled)
                        w_q = weight_quantizer(w_scaled, w_q_scales, w_q_zeros)
                    else:
                        w_q = w_scaled
                        
                    out_q = torch.matmul(X_q, w_q.t())
                    err = (out_q - out_orig).pow(2).mean().item()
                    total_err += err
                    
                if total_err < best_error:
                    best_error = total_err
                    best_scale = s.clone()
                    
            return best_scale
