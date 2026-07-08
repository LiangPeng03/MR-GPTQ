import torch
from src.quantization.quantizer import get_reciprocal
from src.quantization.quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX

def compute_global_scale(x):
    act_max = x.abs().max().to(torch.float32).view(1)
    return (FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(act_max)).to(x.device)

def scale_to_e4m3(raw_scale, global_scale):
    return (raw_scale * global_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX) \
        .to(torch.float8_e4m3fn) \
        .to(torch.float32) \
        .mul(get_reciprocal(global_scale))

def optimize_channel_scales_coordinate_descent(X, W, weight_mse_ratio=3.0, group_size=16, top_k=5, num_rounds=3, nonlinear_beta=0.0):
    """
    Coordinate Descent Joint Tree Search for NVFP4 Channel Scaling.
    Uses Trace Trick to minimize output MSE directly, with optional non-linear weighting.
    """
    device = X.device
    in_features = X.shape[1]
    num_groups = in_features // group_size
    T = X.shape[0]
    OutDim = W.shape[0]
    
    S_g = torch.ones((num_groups, group_size), device=device) # [num_groups, 16]
    
    # 21 个候选缩放值 (0.5 到 1.5 之间，步长 0.05，包含 1.0)
    candidates = torch.linspace(0.5, 1.5, steps=21, device=device)
    num_cands = candidates.shape[0]
    
    X_g = X.view(T, num_groups, group_size)
    W_g = W.view(OutDim, num_groups, group_size)
    
    gs_w = compute_global_scale(W)
    
    def quantize_nvfp4(tensor, gs=None):
        vmax = tensor.abs().amax(dim=-1, keepdim=True)
        raw_scale = vmax.mul_(1/6.0).clamp_(min=1e-10)
        scale = scale_to_e4m3(raw_scale, gs) if gs is not None else raw_scale
        
        normalized = tensor / scale
        abs_norm = normalized.abs()
        q_abs = torch.where(abs_norm > 5.0, 6.0,
                  torch.where(abs_norm >= 3.5, 4.0,
                  torch.where(abs_norm >= 1.75, torch.round(abs_norm),
                                         torch.round(abs_norm * 2.0) * 0.5)))
        return q_abs.copysign(normalized) * scale

    # 非线性输出加权
    if nonlinear_beta > 0.0:
        H = torch.einsum('t i, t j -> i j', X.float(), X.float())
        W_float = W.float()
        W_H = torch.einsum('o i, i j -> o j', W_float, H)
        w = torch.einsum('o j, o j -> o', W_float, W_H) # shape [OutDim]
        s = (w + 1e-12) ** (nonlinear_beta / 4.0)
    else:
        s = torch.ones(OutDim, device=device)
        
    s_view = s.view(OutDim, 1, 1) # 方便广播到 W_g 维度
    
    # 依然保留敏感度选取 top-k (使用简单的 norm2_X * norm2_W)
    norm2_X = (X_g.float() ** 2).sum(dim=0) # [num_groups, 16]
    norm2_W = (W_g.float() ** 2).sum(dim=0) # [num_groups, 16]
    sensitivity_c = norm2_X * norm2_W # [num_groups, 16]
    topk_vals, topk_idx = torch.topk(sensitivity_c, k=top_k, dim=-1) # [num_groups, top_k]
    
    target_elements = 200_000_000 # 峰值显存控制
    max_dim = max(T, OutDim)
    chunk_G = max(1, target_elements // (max_dim * 16 * num_cands))
    
    for g_start in range(0, num_groups, chunk_G):
        g_end = min(g_start + chunk_G, num_groups)
        G_curr = g_end - g_start
        
        # 使用原始有符号的 X_g 和 W_g 进行 Trace Trick 计算
        C = X_g[:, g_start:g_end, :].float() # [T, G_curr, 16]
        D_unscaled = W_g[:, g_start:g_end, :].float() # [OutDim, G_curr, 16]
        D = D_unscaled * s_view
        
        topk_idx_chunk = topk_idx[g_start:g_end] # [G_curr, top_k]
        
        # 预计算当前 chunk 的常量 Trace 矩阵
        CC = torch.einsum('t g i, t g j -> g i j', C, C)
        DD = torch.einsum('o g i, o g j -> g i j', D, D)
        term3 = (CC * DD).sum(dim=(-1, -2)) # [G_curr]
        
        # 本地维护最优的 Scale 矩阵
        local_S = torch.ones((G_curr, group_size), device=device)
        
        for r in range(num_rounds):
            for k in range(top_k):
                temp_S = local_S.unsqueeze(0).expand(num_cands, G_curr, group_size).clone()
                
                target_ch_idx = topk_idx_chunk[:, k] # [G_curr]
                c_expanded = candidates.unsqueeze(1).expand(num_cands, G_curr)
                idx_expanded = target_ch_idx.unsqueeze(0).unsqueeze(-1).expand(num_cands, G_curr, 1)
                temp_S.scatter_(2, idx_expanded, c_expanded.unsqueeze(-1))
                
                X_temp = C.unsqueeze(0) * temp_S.unsqueeze(1) # [21, T, G_curr, 16]
                W_temp = D_unscaled.unsqueeze(0) / temp_S.unsqueeze(1) # [21, OutDim, G_curr, 16]
                
                A = quantize_nvfp4(X_temp).float() # X_q: [21, T, G_curr, 16]
                W_q = quantize_nvfp4(W_temp, gs=gs_w).float() # W_q
                B = W_q * s_view.unsqueeze(0) # [21, OutDim, G_curr, 16]
                
                # Trace Trick 运算
                AA = torch.einsum('v t g i, v t g j -> v g i j', A, A)
                BB = torch.einsum('v o g i, v o g j -> v g i j', B, B)
                AC = torch.einsum('v t g i, t g j -> v g i j', A, C)
                DB = torch.einsum('o g i, v o g j -> v g i j', D, B)
                
                term1 = (AA * BB).sum(dim=(-1, -2)) # [21, G_curr]
                term2 = -2.0 * (AC * DB.transpose(-1, -2)).sum(dim=(-1, -2)) # [21, G_curr]
                
                score = term1 + term2 + term3.unsqueeze(0) # [21, G_curr]
                
                # 取得 21 个候选中的最低分数及其索引
                min_score, min_idx = score.min(dim=0) # [G_curr]
                
                # 直接更新为当前找到的最优解
                best_c = candidates[min_idx] # [G_curr]
                local_S.scatter_(1, target_ch_idx.unsqueeze(1), best_c.unsqueeze(1))
                
        S_g[g_start:g_end] = local_S
        
    S_out = S_g.view(-1)
    
    # 打印统计信息
    mean_s = S_out.mean().item()
    min_s = S_out.min().item()
    max_s = S_out.max().item()
    pct_high = (S_out > 1.2).float().mean().item() * 100
    pct_low = (S_out < 0.8).float().mean().item() * 100
    print(f"      [Coord Descent Trace] S_g Dist: Mean={mean_s:.3f}, Min={min_s:.3f}, Max={max_s:.3f} | >1.2: {pct_high:.1f}%, <0.8: {pct_low:.1f}%")
    
    return S_out
