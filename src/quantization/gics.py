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

def optimize_channel_scales_coordinate_descent(X, W, weight_mse_ratio=3.0, group_size=16, top_k=5, num_rounds=3):
    """
    Coordinate Descent Joint Tree Search for NVFP4 Channel Scaling.
    Targets Top-K channels and optimizes them iteratively for num_rounds.
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
    
    def calc_mse_positive_inplace(mat_temp, gs=None):
        vmax = mat_temp.amax(dim=-1, keepdim=True)
        raw_scale = vmax.mul_(1/6.0).clamp_(min=1e-10)
        scale = scale_to_e4m3(raw_scale, gs) if gs is not None else raw_scale
        
        mat_temp.div_(scale)
        quantized = torch.where(mat_temp > 5.0, 6.0,
                      torch.where(mat_temp >= 3.5, 4.0,
                      torch.where(mat_temp >= 1.75, torch.round(mat_temp),
                                             torch.round(mat_temp * 2.0) * 0.5)))
        mat_temp.sub_(quantized).pow_(2).mul_(scale.pow(2))
        return mat_temp.sum(dim=(-3, -1))
        
    abs_X_g = X_g.abs()
    abs_W_g = W_g.abs()
    
    base_act_mse = calc_mse_positive_inplace(abs_X_g.clone()) # [num_groups]
    base_w_mse = calc_mse_positive_inplace(abs_W_g.clone(), gs=gs_w) # [num_groups]
    
    norm2_X = (X_g.float() ** 2).sum(dim=0) # [num_groups, 16]
    norm2_W = (W_g.float() ** 2).sum(dim=0) # [num_groups, 16]
    
    max_X_c = abs_X_g.amax(dim=0) # [num_groups, 16]
    median_X_c = abs_X_g.median(dim=0).values # [num_groups, 16]
    spikiness = max_X_c / (median_X_c + 1e-12) # [num_groups, 16]
    
    sensitivity_c = norm2_X * norm2_W #* spikiness # [num_groups, 16]
    topk_vals, topk_idx = torch.topk(sensitivity_c, k=top_k, dim=-1) # [num_groups, top_k]
    
    target_elements = 400_000_000 # 峰值显存控制
    max_dim = max(T, OutDim)
    chunk_G = max(1, target_elements // (max_dim * 16 * num_cands))
    
    for g_start in range(0, num_groups, chunk_G):
        g_end = min(g_start + chunk_G, num_groups)
        G_curr = g_end - g_start
        
        X_g_chunk = abs_X_g[:, g_start:g_end, :] # [T, G_curr, 16]
        W_g_chunk = abs_W_g[:, g_start:g_end, :] # [OutDim, G_curr, 16]
        topk_idx_chunk = topk_idx[g_start:g_end] # [G_curr, top_k]
        
        b_act = base_act_mse[g_start:g_end].unsqueeze(0) + 1e-12
        b_w = base_w_mse[g_start:g_end].unsqueeze(0) + 1e-12
        
        # 本地维护最优的 Scale 矩阵
        local_S = torch.ones((G_curr, group_size), device=device)
        
        for r in range(num_rounds):
            for k in range(top_k):
                # temp_S 继承当前的最佳 scale [num_cands, G_curr, 16]
                temp_S = local_S.unsqueeze(0).expand(num_cands, G_curr, group_size).clone()
                
                # 针对第 k 个目标通道，替换为 21 个候选值
                target_ch_idx = topk_idx_chunk[:, k] # [G_curr]
                c_expanded = candidates.unsqueeze(1).expand(num_cands, G_curr)
                idx_expanded = target_ch_idx.unsqueeze(0).unsqueeze(-1).expand(num_cands, G_curr, 1)
                temp_S.scatter_(2, idx_expanded, c_expanded.unsqueeze(-1))
                
                X_temp = X_g_chunk.unsqueeze(0) * temp_S.unsqueeze(1) # [21, T, G_curr, 16]
                W_temp = W_g_chunk.unsqueeze(0) / temp_S.unsqueeze(1) # [21, OutDim, G_curr, 16]
                
                act_mse = calc_mse_positive_inplace(X_temp) # [21, G_curr]
                w_mse = calc_mse_positive_inplace(W_temp, gs=gs_w) # [21, G_curr]
                
                score = (act_mse / b_act) + weight_mse_ratio * (w_mse / b_w) # [21, G_curr]
                
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
    print(f"      [Coord Descent Search] S_g Dist: Mean={mean_s:.3f}, Min={min_s:.3f}, Max={max_s:.3f} | >1.2: {pct_high:.1f}%, <0.8: {pct_low:.1f}%")
    
    return S_out
