from typing import Tuple, Optional

import torch

from .quant_args import QuantizationFormat, QuantizationGranularity, QuantizationObserver, ScalePrecision
from .quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX, FP4_SCALE, get_quantization_fns, get_quantization_range, cast_to_eBm0
from ..helpers import split_dim

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
        scale_min_clip: Optional[float] = None
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

        elif self.observer == QuantizationObserver.LSS:
            if not self.symmetric:
                raise NotImplementedError("LSS observer only supports symmetric quantization.")
            abs_x = x.abs()
            s_0 = scales.clone()
            s_0[s_0 == 0] = 1.0
            
            # Get grid assignments (g_i)
            # Use quant_fn with scale s_0. This returns cast_to_fp4(abs_x / s_0) for NVFP4.
            g = self.quant_fn(abs_x, s_0, zeros, self.q_min, self.q_max)
            g = g.abs() # Ensure positive grid values
            
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
        return q
