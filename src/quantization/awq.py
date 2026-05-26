import torch
from typing import List, Optional

@torch.no_grad()
def run_awq_search(
    X: torch.Tensor,
    weights: List[torch.Tensor],
    weight_quantizer=None,
    act_quantizer=None,
    steps: int = 20,
    X_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Finds the optimal AWQ diagonal scale `s` for a given input activation `X` 
    and a list of weights (e.g. [q_proj, k_proj, v_proj] that share the same X).
    
    Returns:
        s (torch.Tensor): Optimal scaling vector of shape (in_features,)
    """
    device = X.device
    in_features = X.shape[-1]
    
    if len(X.shape) == 3:
        X = X.reshape(-1, in_features)
    
    # Fast path if no quantizers or no steps
    if steps == 0 or (weight_quantizer is None and act_quantizer is None):
        return torch.ones(in_features, device=device, dtype=X.dtype)
        
    if X_max is None:
        X_max = X.abs().max(dim=0).values.clamp(min=1e-4)
        
    # We will search alpha in [0, 1]
    best_error = float('inf')
    best_scale = torch.ones_like(X_max)
    
    # To save VRAM/compute, sample subset if X is large
    n_samples = X.shape[0]
    if n_samples > 2048:
        idx = torch.randperm(n_samples, device=device)[:2048]
        X_sub = X[idx]
    else:
        X_sub = X
        
    # Precompute FP16 outputs for MSE
    # Y_orig = sum( || X * W_i^T ||_2^2 ) is not needed, we just compare MSE directly.
    # Actually, we can sum the MSE across all weights.
    out_origs = [torch.matmul(X_sub, w.t()) for w in weights]
    
    for i in range(steps + 1):
        alpha = i / steps
        s = X_max ** alpha
        
        # Normalize s to have mean 1 (keeps overall magnitude similar)
        s = s / s.mean()
        
        # Limit scaling factor to 10x (prevent excessive magnification)
        s = torch.clamp(s, min=0.1, max=10.0)
        
        # Apply scaling
        X_scaled = X_sub * s
        
        # Quantize X
        if act_quantizer is not None:
            # act_quantizer expects (batch, seq, dim) or similar. Usually (1, num_tokens, dim)
            q_scales, q_zeros = act_quantizer.get_quantization_params(X_scaled.unsqueeze(0))
            X_q = act_quantizer(X_scaled.unsqueeze(0), q_scales, q_zeros).squeeze(0)
        else:
            X_q = X_scaled
            
        total_err = 0.0
        
        for w, out_orig in zip(weights, out_origs):
            # Scale weight
            w_scaled = w / s.unsqueeze(0)
            
            # Quantize W
            if weight_quantizer is not None:
                w_q_scales, w_q_zeros = weight_quantizer.get_quantization_params(w_scaled)
                w_q = weight_quantizer(w_scaled, w_q_scales, w_q_zeros)
            else:
                w_q = w_scaled
                
            out_q = torch.matmul(X_q, w_q.t())
            
            # MSE
            err = (out_q - out_orig).pow(2).mean().item()
            total_err += err
            
        if total_err < best_error:
            best_error = total_err
            best_scale = s.clone()
            
    return best_scale
