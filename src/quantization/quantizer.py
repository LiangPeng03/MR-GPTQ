from typing import Tuple, Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

from .quant_args import QuantizationFormat, QuantizationGranularity, QuantizationObserver, ScalePrecision
from .quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX, FP4_SCALE, get_quantization_fns, get_quantization_range, cast_to_eBm0
from ..helpers import split_dim


if triton is not None:
    @triton.jit
    def _scalesweep_round_even(value):
        integer = tl.floor(value)
        fraction = value - integer
        increment = (fraction > 0.5) | (
            (fraction == 0.5) & ((integer.to(tl.int32) & 1) != 0)
        )
        return integer + increment.to(tl.float32)

    @triton.jit
    def _decode_positive_e4m3(raw):
        exponent = raw >> 3
        mantissa = raw & 7
        bits = ((exponent - 7 + 127) << 23) | (mantissa << 20)
        return tl.where(exponent > 0, bits.to(tl.float32, bitcast=True),
                        mantissa.to(tl.float32) / 512.0)

    @triton.jit
    def _scalesweep_scale_kernel(
        X, Importance, InverseGlobal, Result, groups, column_groups,
        LOWER: tl.constexpr, WEIGHTED: tl.constexpr, BLOCKS: tl.constexpr,
        UPPER: tl.constexpr = 7, SCALESEARCH: tl.constexpr = False,
    ):
        rows = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
        channels = tl.arange(0, 16)
        candidates = tl.arange(0, 16)
        values = tl.load(
            X + rows[:, None] * 16 + channels[None, :],
            mask=rows[:, None] < groups, other=0,
        ).to(tl.float32)
        inverse = tl.load(InverseGlobal).to(tl.float32)
        base = tl.minimum(tl.max(tl.abs(values), axis=1) * inverse / 6.0, 448.0)
        # Positive E4M3 floor, including subnormals, without searchsorted.
        bits = base.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) - 127 + 7
        mantissa = (bits & 0x7fffff) >> 20
        normal_raw = (exponent << 3) + mantissa
        subnormal_raw = tl.floor(base * 512.0).to(tl.int32)
        base_raw = tl.where(exponent > 0, normal_raw, subnormal_raw)
        base_raw = tl.minimum(tl.maximum(base_raw, 0), 126)
        if SCALESEARCH:
            next_raw = tl.minimum(base_raw + 1, 126)
            down = _decode_positive_e4m3(base_raw)
            up = _decode_positive_e4m3(next_raw)
            choose_up = ((up - base) < (base - down)) | (
                ((up - base) == (base - down)) & ((next_raw & 1) == 0)
            )
            base_raw = tl.where(choose_up, next_raw, base_raw)
        offsets = candidates + LOWER
        candidate_raw = base_raw[:, None] + offsets[None, :]
        raw = tl.minimum(tl.maximum(candidate_raw, 1), 126)
        scale_exponent = raw >> 3
        scale_mantissa = raw & 7
        fp32_bits = ((scale_exponent - 7 + 127) << 23) | (scale_mantissa << 20)
        decoded = tl.where(
            scale_exponent > 0, fp32_bits.to(tl.float32, bitcast=True),
            scale_mantissa.to(tl.float32) / 512.0,
        )
        scales = decoded / inverse
        normalized = values[:, :, None] / scales[:, None, :]
        magnitude = tl.abs(normalized)
        code = tl.where(
            magnitude > 5.0, 6.0,
            tl.where(magnitude >= 3.5, 4.0,
                     tl.where(magnitude >= 1.75, _scalesweep_round_even(magnitude),
                              _scalesweep_round_even(magnitude * 2.0) * 0.5)),
        )
        code = tl.where(normalized < 0, -code, code)
        difference = code * scales[:, None, :] - values[:, :, None]
        error = difference * difference
        if WEIGHTED:
            importance = tl.load(
                Importance + (rows[:, None] % column_groups) * 16 + channels[None, :],
                mask=rows[:, None] < groups, other=0,
            ).to(tl.float32)
            error = error * importance[:, :, None]
        losses = tl.sum(error, axis=1) / 16.0
        valid = offsets[None, :] <= UPPER
        if SCALESEARCH:
            valid = valid & (candidate_raw >= 1) & (candidate_raw <= 126)
        losses = tl.where(valid, losses, float("inf"))
        minimum = tl.min(losses, axis=1)
        winner = tl.min(
            tl.where(losses == minimum[:, None], candidates[None, :], 16), axis=1
        )
        selected = tl.sum(tl.where(candidates[None, :] == winner[:, None], scales, 0.0), axis=1)
        tl.store(Result + rows, selected, mask=rows < groups)

# Utility function for inversion.
def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


class Quantizer:

    def __init__(
        self, 
        bits: int, 
        symmetric: bool = True,
        format: str = "int",
        granularity: str = "channel",
        observer: str = "minmax",
        dim: int = -1,
        group_size: Optional[int] = None,
        scale_precision: str = "fp16",
        scale_min_clip: Optional[float] = None,
        global_scale_max: Optional[float] = None,
    ):
        # Sanity checks
        if format in ["fp", "nvfp", "mxfp"]:
            assert symmetric, "Only symmetric quantization is supported for floating point formats."

        if granularity == "group":
            assert group_size is not None, "Group size must be specified when granularity is 'group'."
        else:
            assert group_size is None, "Group size must be None when granularity is not 'group'."

        self.bits = bits
        self.symmetric = symmetric
        self.format = QuantizationFormat(format)
        self.granularity = QuantizationGranularity(granularity)
        self.observer = QuantizationObserver(observer)
        self.scale_precision = ScalePrecision(scale_precision)
        self.dim = dim
        self.group_size = group_size
        self.scale_min_clip = scale_min_clip
        self.is_scalesweep = self.observer in {
            QuantizationObserver.SCALESWEEP_MSE, QuantizationObserver.SCALESWEEP_WMSE,
            QuantizationObserver.SCALESEARCH,
        }
        self.importance = None
        self._scalesweep_checked_importance = None
        self._scalesweep_checked_global = None
        self._wlss_importance_cache = None
        self.global_scale_max = (
            256 if self.observer in {QuantizationObserver.SCALESWEEP_MSE,
                                    QuantizationObserver.SCALESWEEP_WMSE}
            else FP8_E4M3_MAX
        )
        if global_scale_max is not None:
            if (self.format != QuantizationFormat.NVFP
                    or not 0 < float(global_scale_max) <= FP8_E4M3_MAX):
                raise ValueError("global_scale_max must be in (0, 448] for NVFP4.")
            self.global_scale_max = float(global_scale_max)
        if self.observer == QuantizationObserver.LSS_WMSE and (
            self.format != QuantizationFormat.NVFP or self.bits != 4
            or not self.symmetric or self.group_size != 16 or self.dim != -1
            or self.granularity != QuantizationGranularity.GROUP
            or self.scale_precision != ScalePrecision.E4M3
        ):
            raise ValueError("Weighted LSS requires symmetric NVFP4 group-16/E4M3.")
        if self.is_scalesweep and (
            self.format != QuantizationFormat.NVFP or self.bits != 4
            or not self.symmetric or self.group_size != 16 or self.dim != -1
            or self.granularity != QuantizationGranularity.GROUP
            or self.scale_precision != ScalePrecision.E4M3
        ):
            raise ValueError("ScaleSweep requires symmetric NVFP4 group-16 with E4M3 scales.")

        self.quant_fn, self.dequant_fn, self.quant_dequant_fn = get_quantization_fns(
            format=self.format,
            bits=self.bits,
        )

        self.q_min, self.q_max = get_quantization_range(
            format=self.format,
            bits=self.bits,
            symmetric=self.symmetric,
        )
        
        # Global scale is 3 for MXFP quantization
        if self.format == QuantizationFormat.MXFP:
            self.global_scale = torch.tensor([3.0], dtype=torch.float32)
        else:
            self.global_scale = torch.tensor([float("inf")], dtype=torch.float32)
        # Scale tracking is needed only for E4M3 scale quantization
        self._track_global_scale = (self.scale_precision == ScalePrecision.E4M3)

    def _reshape_before_quantization(
        self, 
        x: torch.Tensor, 
        scales: Optional[torch.Tensor] = None,
        zeros: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.group_size:
            dim = x.ndim - 1 if self.dim == -1 else self.dim
            num_groups = x.shape[dim] // self.group_size
            x = split_dim(x, num_groups, dim)
            if scales is not None:
                scales = scales.unsqueeze(dim + 1)
            if zeros is not None:
                zeros = zeros.unsqueeze(dim + 1)
        return x, scales, zeros

    def get_quantization_params(
        self, 
        x: torch.Tensor,
        # MSE observer quantization params
        scale_search_iters: int = 100,
        max_scale_shrink_factor: float = 0.80,
        error_norm: float = 2.4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get scale and zero point for an input tensor.
        """
        if self.is_scalesweep:
            return self._scalesweep_params(x)
        dim = x.ndim - 1 if self.dim == -1 else self.dim
        if self.granularity == QuantizationGranularity.GROUP:
            reduce_dim = dim + 1
        elif self.granularity == QuantizationGranularity.CHANNEL:
            reduce_dim = dim
        else:
            reduce_dim = None
        x, _, _ = self._reshape_before_quantization(x)

        x_min = x.amin(dim=reduce_dim, keepdim=True)
        x_max = x.amax(dim=reduce_dim, keepdim=True)

        if self.symmetric:
            scales = 2 * torch.maximum(-x_min, x_max) / (self.q_max - self.q_min)
            zeros =  torch.zeros_like(x_min)
        else:
            scales = (x_max - x_min) / (self.q_max - self.q_min)
            zeros = -(x_min / scales).round()

        if self.observer == QuantizationObserver.MSE:
            # MSE observer: scale_shrink_factor searches from 1.0 to 0.2
            init_scales = scales.clone() 
            best_quantization_error = torch.full(x.shape[:-1], float("inf"), device=x.device, dtype=x.dtype)

            for i in range(scale_search_iters):
                scale_shrink_factor = 1.0 - i * 0.8 / scale_search_iters
                candidate_scales = scale_shrink_factor * init_scales
                candidate_zeros = torch.zeros_like(x_min) if self.symmetric else -(x_min / candidate_scales).round() 
                q = self.quant_fn(x, candidate_scales, candidate_zeros, self.q_min, self.q_max)
                x_reconstructed = self.dequant_fn(q, candidate_scales, candidate_zeros)
                quantization_error = (x - x_reconstructed).abs_().pow_(error_norm).sum(dim=-1)

                if (quantization_error < best_quantization_error).any():
                    improved_ids = torch.where(quantization_error < best_quantization_error)
                    best_quantization_error[improved_ids] = quantization_error[improved_ids]
                    scales[improved_ids] = candidate_scales[improved_ids]
                    if not self.symmetric:
                        zeros[improved_ids] = candidate_zeros[improved_ids]

        elif self.observer == QuantizationObserver.MSE_N:
            # MSE_N observer: scale_shrink_factor searches from 1.1 to 0.5
            init_scales = scales.clone() 
            best_quantization_error = torch.full(x.shape[:-1], float("inf"), device=x.device, dtype=x.dtype)

            for i in range(scale_search_iters):
                scale_shrink_factor = 1.1 - i * 0.6 / scale_search_iters
                candidate_scales = scale_shrink_factor * init_scales
                candidate_zeros = torch.zeros_like(x_min) if self.symmetric else -(x_min / candidate_scales).round() 
                q = self.quant_fn(x, candidate_scales, candidate_zeros, self.q_min, self.q_max)
                x_reconstructed = self.dequant_fn(q, candidate_scales, candidate_zeros)
                quantization_error = (x - x_reconstructed).abs_().pow_(error_norm).sum(dim=-1)

                if (quantization_error < best_quantization_error).any():
                    improved_ids = torch.where(quantization_error < best_quantization_error)
                    best_quantization_error[improved_ids] = quantization_error[improved_ids]
                    scales[improved_ids] = candidate_scales[improved_ids]
                    if not self.symmetric:
                        zeros[improved_ids] = candidate_zeros[improved_ids]

        elif self.observer in {QuantizationObserver.LSS, QuantizationObserver.LSS_WMSE}:
            if not self.symmetric:
                raise NotImplementedError("LSS observer only supports symmetric quantization.")
            abs_x = x.abs()
            s_0 = scales.clone()
            s_0[s_0 == 0] = 1.0
            
            # Get grid assignments (g_i)
            # Use quant_fn with scale s_0. This returns cast_to_fp4(abs_x / s_0) for NVFP4.
            g = self.quant_fn(abs_x, s_0, zeros, self.q_min, self.q_max)
            g = g.abs() # Ensure positive grid values
            
            if self.observer == QuantizationObserver.LSS_WMSE:
                width = x.shape[-2] * self.group_size
                importance = self._wlss_importance(width, x.device)
                # Preserve the original LSS BF16 product/sum boundaries. With
                # uniform importance this is exactly the existing LSS formula.
                num = ((abs_x * g).float() * importance).sum(
                    dim=reduce_dim, keepdim=True).to(x.dtype)
                den = ((g * g).float() * importance).sum(
                    dim=reduce_dim, keepdim=True).to(x.dtype)
            else:
                num = (abs_x * g).sum(dim=reduce_dim, keepdim=True)
                den = (g * g).sum(dim=reduce_dim, keepdim=True)
            
            den_zero = (den == 0)
            den[den_zero] = 1.0
            
            lss_scales = num / den
            scales = torch.where(den_zero, scales, lss_scales)

        elif self.observer == QuantizationObserver.FOUR_OVER_SIX:
            if not self.symmetric:
                raise NotImplementedError("Four-over-six observer only supports symmetric quantization.")
            abs_max = x.abs().amax(dim=reduce_dim, keepdim=True)
            # Scale candidate 1: /6 (standard)
            s_6 = abs_max / (self.q_max - self.q_min) * 2
            s_6[s_6 == 0] = 1.0
            # Scale candidate 2: /4 (four over six)
            s_4 = abs_max / 4.0
            s_4[s_4 == 0] = 1.0

            # Quantize-dequantize with both scales
            q_6 = self.quant_fn(x, s_6, zeros, self.q_min, self.q_max)
            x_recon_6 = self.dequant_fn(q_6, s_6, zeros)
            q_4 = self.quant_fn(x, s_4, zeros, self.q_min, self.q_max)
            x_recon_4 = self.dequant_fn(q_4, s_4, zeros)

            # Per-group MSE comparison
            err_6 = (x - x_recon_6).pow(2).sum(dim=-1)
            err_4 = (x - x_recon_4).pow(2).sum(dim=-1)
            select_4 = (err_4 < err_6).unsqueeze(-1)
            scales = torch.where(select_4, s_4, s_6)

        elif self.observer == QuantizationObserver.LSS_3ROUND:
            if not self.symmetric:
                raise NotImplementedError("LSS_3round observer only supports symmetric quantization.")
            abs_x = x.abs()
            s_0 = scales.clone()
            s_0[s_0 == 0] = 1.0

            # Round 1: initial LSS
            g = self.quant_fn(abs_x, s_0, zeros, self.q_min, self.q_max)
            g = g.abs()
            num = (abs_x * g).sum(dim=reduce_dim, keepdim=True)
            den = (g * g).sum(dim=reduce_dim, keepdim=True)
            den_zero = (den == 0)
            den[den_zero] = 1.0
            s_1 = num / den
            s_1[den_zero] = 1.0

            # Round 2
            g_2 = self.quant_fn(abs_x, s_1, zeros, self.q_min, self.q_max)
            g_2 = g_2.abs()
            num_2 = (abs_x * g_2).sum(dim=reduce_dim, keepdim=True)
            den_2 = (g_2 * g_2).sum(dim=reduce_dim, keepdim=True)
            den_2_zero = (den_2 == 0)
            den_2[den_2_zero] = 1.0
            s_2 = num_2 / den_2
            s_2[den_2_zero] = 1.0

            # Round 3
            g_3 = self.quant_fn(abs_x, s_2, zeros, self.q_min, self.q_max)
            g_3 = g_3.abs()
            num_3 = (abs_x * g_3).sum(dim=reduce_dim, keepdim=True)
            den_3 = (g_3 * g_3).sum(dim=reduce_dim, keepdim=True)
            den_3_zero = (den_3 == 0)
            den_3[den_3_zero] = 1.0
            s_3 = num_3 / den_3
            s_3[den_3_zero] = 1.0

            scales = s_3

        # Reshape back
        if self.group_size:
            x = x.flatten(dim, dim + 1)
            scales = scales.squeeze(dim + 1)
            if zeros is not None:
                zeros = zeros.squeeze(dim + 1)

        if self.scale_precision == ScalePrecision.E4M3:
            with torch.no_grad():
                if self._track_global_scale:
                    current_global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(x.abs().max().to(torch.float32).view(1))
                    if not current_global_scale:
                        raise ValueError(f"Current global scale is not finite: {current_global_scale}\n")
                    # Update global scale using min of current and computed scale
                    self.global_scale = torch.minimum(self.global_scale.to(x.device), current_global_scale)
                    
                    if not self.global_scale.isfinite():
                        raise ValueError(f"Global scale is not finite: {self.global_scale}\n")
                    
                # Clamp, convert to fp8, convert back, and rescale in one chain
                scales = (scales * self.global_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX) \
                    .to(torch.float8_e4m3fn) \
                    .to(torch.float32) \
                    .mul(get_reciprocal(self.global_scale)) \
                    .to(x.dtype)
        
        elif self.scale_precision == ScalePrecision.E8M0:
            # Inspired by quantize_tseng (see https://github.com/IST-DASLab/Quartet/blob/main/notebooks/benchmark_mxfp4.ipynb)
            # NOTE (in quartet x.abs().max() is defined as a scale insteaf of x.abs().max() / q_max )
            scales = cast_to_eBm0(FP4_E2M1_MAX * scales, ebits=8, emax=2) / FP4_SCALE

        # Set scales to 1 if zero
        scales[scales == 0] = 1

        if scales.isnan().any():
            raise ValueError(f"Scales are not finite.")
      
        return scales, zeros

    def _wlss_importance(self, width, device):
        """Frozen weight-column second moments in the transformed coordinates."""
        if self.importance is None:
            raise ValueError("Weighted LSS requires calibrated channel importance.")
        cached = self._wlss_importance_cache
        if (cached is not None and cached[0] is self.importance
                and cached[1] == device and cached[2].numel() == width):
            return cached[2]
        weights = self.importance.detach().to(device=device, dtype=torch.float32)
        if weights.numel() != width:
            raise ValueError("Weighted LSS importance must match the input width.")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Weighted LSS importance must be finite and nonnegative.")
        weights = weights.reshape(-1, self.group_size)
        maximum = weights.amax(-1, keepdim=True)
        # Groupwise normalization cancels in the LS ratio and bounds products.
        # A zero-importance group has no preferred scale: fall back to LSS.
        weights = torch.where(
            maximum > 0, weights / maximum.clamp_min(torch.finfo(torch.float32).tiny),
            torch.ones_like(weights),
        ).contiguous()
        self._wlss_importance_cache = (self.importance, device, weights)
        return weights

    @torch.no_grad()
    def _scalesweep_params(self, x):
        """ScaleSweep or ScaleSearch FP8-code search with bounded storage.

        ScaleSearch follows Algorithm 1 and Section 4.1 of arXiv:2605.12464.
        E4M3 code 127 is NaN in PyTorch and is excluded from finite candidates.
        """
        weighted = self.observer == QuantizationObserver.SCALESWEEP_WMSE
        scalesearch = self.observer == QuantizationObserver.SCALESEARCH
        if weighted and self.importance is None:
            raise ValueError("ScaleSweep-WMSE requires calibrated channel importance.")
        if x.shape[-1] % 16:
            raise ValueError("ScaleSweep input width must be divisible by 16.")
        if self._track_global_scale:
            maximum = x.detach().abs().amax().float().reshape(1)
            current = torch.where(maximum > 0, self.global_scale_max * FP4_E2M1_MAX / maximum,
                                  torch.ones_like(maximum))
            self.global_scale = torch.minimum(self.global_scale.to(x.device), current)
        inverse_global = self.global_scale.to(device=x.device, dtype=torch.float32)
        if self._scalesweep_checked_global is not self.global_scale:
            if not torch.isfinite(inverse_global).all() or not (inverse_global > 0).all():
                raise ValueError("ScaleSweep global scale must be finite and positive.")
            self._scalesweep_checked_global = self.global_scale
        groups = x.detach().reshape(-1, 16).float().contiguous()
        result = torch.empty(groups.shape[0], device=x.device, dtype=torch.float32)
        importance = None
        if weighted:
            importance = self.importance.to(device=x.device, dtype=torch.float32)
            if importance.numel() != x.shape[-1]:
                raise ValueError("ScaleSweep importance must have one entry per input channel.")
            importance = importance.reshape(-1, 16).contiguous()
            if self._scalesweep_checked_importance is not self.importance:
                if not torch.isfinite(importance).all() or (importance < 0).any():
                    raise ValueError("ScaleSweep importance must be finite and nonnegative.")
                self._scalesweep_checked_importance = self.importance
        lower = -2 if scalesearch else (-8 if weighted else -3)
        upper = 6 if scalesearch else 7
        if triton is not None and groups.is_cuda:
            _scalesweep_scale_kernel[(triton.cdiv(groups.shape[0], 16),)](
                groups, importance if weighted else groups, inverse_global, result,
                groups.shape[0], x.shape[-1] // 16,
                LOWER=lower, WEIGHTED=weighted, BLOCKS=16,
                UPPER=upper, SCALESEARCH=scalesearch,
                num_warps=4, enable_fp_fusion=False,
            )
            scales = result.reshape(*x.shape[:-1], x.shape[-1] // 16)
            return scales, torch.zeros_like(scales)
        table = torch.arange(127, device=x.device, dtype=torch.uint8)
        table = table.view(torch.float8_e4m3fn).float()
        for start in range(0, groups.shape[0], 65536):
            values = groups[start:start + 65536]
            base = values.abs().amax(-1) * inverse_global / FP4_E2M1_MAX
            base = base.clamp(0, FP8_E4M3_MAX)
            unrounded = base
            base = (torch.searchsorted(table, base.contiguous(), right=True) - 1).clamp(0, 126)
            if scalesearch:
                next_raw = (base + 1).clamp(0, 126)
                down, up = table[base], table[next_raw]
                choose_up = ((up - unrounded) < (unrounded - down)) | (
                    ((up - unrounded) == (unrounded - down)) & (next_raw % 2 == 0)
                )
                base = torch.where(choose_up, next_raw, base)
            best_error = torch.full_like(base, float("inf"), dtype=torch.float32)
            best_scale = torch.empty_like(best_error)
            weights = None
            if importance is not None:
                indices = torch.arange(start, start + len(values), device=x.device)
                weights = importance[indices % importance.shape[0]]
            for offset in range(lower, upper + 1):
                raw = (base + offset).clamp(1, 126)
                scale = (table[raw] / inverse_global).unsqueeze(-1)
                reconstructed = self.dequant_fn(
                    self.quant_fn(values, scale, None, self.q_min, self.q_max),
                    scale, None,
                )
                error = (reconstructed - values).square()
                if weights is not None:
                    error = error * weights
                error = error.mean(-1)
                if scalesearch:
                    valid = ((base + offset) >= 1) & ((base + offset) <= 126)
                    error = torch.where(valid, error, torch.full_like(error, float("inf")))
                improved = error < best_error
                best_error = torch.where(improved, error, best_error)
                best_scale = torch.where(improved, scale.squeeze(-1), best_scale)
            result[start:start + len(values)] = best_scale
        scales = result.reshape(*x.shape[:-1], x.shape[-1] // 16)
        return scales, torch.zeros_like(scales)
        
    def quantize(self, x: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = x.shape
        q = self.quant_fn(
            *self._reshape_before_quantization(x, scales, zeros), 
            self.q_min, 
            self.q_max
        ).reshape(original_shape)
        return q

    def dequantize(self, q: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = q.shape
        return self.dequant_fn(
            *self._reshape_before_quantization(q, scales, zeros), 
        ).reshape(original_shape)
    
    def __call__(self, x: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = x.shape
        q = self.quant_dequant_fn(
            *self._reshape_before_quantization(x, scales, zeros), 
            self.q_min, 
            self.q_max
        ).reshape(original_shape)
        return q.to(x.dtype) if self.is_scalesweep else q


@torch.no_grad()
def validate_nada_wmse(device="cuda"):
    """Small numerical preflight; does not load/export models or change files."""
    from .qlinear import set_scalesweep_activation_importance

    torch.manual_seed(0)
    kwargs = dict(bits=4, format="nvfp", granularity="group", group_size=16,
                  scale_precision="e4m3")

    def frozen(observer, where=device, **extra):
        q = Quantizer(observer=observer, **kwargs, **extra)
        q.global_scale = torch.tensor([64.], device=where)
        q._track_global_scale = False
        return q

    x = torch.randn(2, 7, 64, device=device, dtype=torch.bfloat16)
    x[..., :16] = 0
    ordinary, weighted = frozen("lss"), frozen("lss_wmse")
    expected, _ = ordinary.get_quantization_params(x)
    for constant in (0., 1., 10.):
        weighted.importance = torch.full((64,), constant, device=device)
        actual, _ = weighted.get_quantization_params(x)
        assert torch.equal(expected, actual), f"Uniform/zero importance mismatch: {constant}"
    weighted.importance = torch.exp(torch.linspace(-5, 5, 64, device=device))
    actual, _ = weighted.get_quantization_params(x)
    assert not torch.equal(expected, actual), "Weighted LSS importance has no effect"
    grouped = x.abs().reshape(2, 7, 4, 16)
    initial = grouped.amax(-1, keepdim=True) / 6
    initial = torch.where(initial == 0, torch.ones_like(initial), initial)
    grid = ordinary.quant_fn(grouped, initial, None, -6, 6).abs()
    weights = weighted.importance.reshape(4, 16)
    weights = weights / weights.amax(-1, keepdim=True)
    numerator = ((grouped * grid).float() * weights).sum(-1, keepdim=True).to(x.dtype)
    denominator = ((grid * grid).float() * weights).sum(-1, keepdim=True).to(x.dtype)
    reference = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30),
                            grouped.amax(-1, keepdim=True) / 6).squeeze(-1)
    reference = ((reference * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
                 .float() / 64).to(x.dtype)
    reference[reference == 0] = 1
    assert torch.equal(actual, reference), "Weighted LS closed-form mismatch"
    for bad in (None, torch.ones(63, device=device),
                -torch.ones(64, device=device),
                torch.full((64,), float("nan"), device=device)):
        weighted.importance = bad
        try:
            weighted.get_quantization_params(x)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid weighted LSS importance was accepted")
    print("NADA weighted LSS: formula/uniform/zero/importance checks OK", flush=True)

    layers = []
    perm = torch.randperm(64, device=device)
    rescale = torch.linspace(.5, 1.5, 64, device=device).bfloat16()
    for rows in (32, 16, 16):
        layer = torch.nn.Linear(64, rows, bias=False, device=device, dtype=torch.bfloat16)
        layer.weight.data = (layer.weight.data[:, perm] / rescale).contiguous()
        layer.act_quantizer = frozen("lss_wmse")
        layers.append(layer)
    set_scalesweep_activation_importance(layers)
    target = torch.cat([layer.weight.float() for layer in layers]).square().mean(0)
    for layer in layers:
        assert torch.allclose(layer.act_quantizer.importance, target, rtol=1e-6, atol=1e-8)
    assert layers[0].act_quantizer.importance is layers[1].act_quantizer.importance
    print("NADA transformed/fused QKV importance: OK", flush=True)

    base_search = Quantizer(observer="scalesweep_wmse", **kwargs)
    search = frozen("scalesweep_wmse", global_scale_max=448)
    assert base_search.global_scale_max == 256 and search.global_scale_max == 448
    values = torch.randn(8, 64, device=device, dtype=torch.float32)
    search.importance = torch.exp(torch.linspace(-2, 2, 64, device=device))
    scales, _ = search.get_quantization_params(values)
    cpu_reference = frozen("scalesweep_wmse", where="cpu", global_scale_max=448)
    cpu_reference.importance = search.importance.cpu()
    expected, _ = cpu_reference.get_quantization_params(values.cpu())
    assert torch.equal(scales.cpu(), expected), "Weight WMSE CUDA/CPU search mismatch"
    print("NADA weight WMSE: CUDA/reference OK; original global bound 448 preserved", flush=True)
    print("ALL NADA WMSE CHECKS PASSED", flush=True)
