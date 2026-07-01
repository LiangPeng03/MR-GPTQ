import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import Quantizer
from ..transforms.transforms import BaseTransform


class QLinear(nn.Linear):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        weight_quantizer: Quantizer = None,
        act_quantizer: Quantizer = None,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.weight_quantizer = weight_quantizer
        self.act_quantizer = act_quantizer
        self._train_mode = True

    def forward(
        self, 
        x: torch.Tensor, 
        in_transform: BaseTransform = None, 
        out_transform: BaseTransform = None
    ) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if self._train_mode:
            if in_transform:
                weight = in_transform(weight, inv_t=True, dim=-1)
            if out_transform:
                weight = out_transform(weight, inv_t=True, dim=0)
                if bias is not None:
                    bias = out_transform(bias, inv_t=True, dim=0)

            if self.weight_quantizer is not None:
                w_scales, w_zeros = self.weight_quantizer.get_quantization_params(weight)
                weight = self.weight_quantizer(weight, w_scales, w_zeros)

        if self.act_quantizer is not None:
            a_scales, a_zeros = self.act_quantizer.get_quantization_params(x)
            x_q = self.act_quantizer(x, a_scales, a_zeros)
            
            if hasattr(self, 'track_act_mse') and self.track_act_mse:
                if getattr(self, 'act_mse_sum', None) is None:
                    self.act_mse_sum = 0.0
                    self.act_mse_count = 0
                
                # Calculate relative MSE of the activation tensor
                rel_mse = (x_q - x).pow(2).mean().item() / (x.pow(2).mean().item() + 1e-6)
                self.act_mse_sum += rel_mse
                self.act_mse_count += 1
            
            x = x_q

        return F.linear(x, weight, bias)

    def fix_parametrization(
        self, 
        in_transform: BaseTransform = None, 
        out_transform: BaseTransform = None
    ) -> None:
        weight = self.weight
        bias = self.bias

        if in_transform:
            weight = in_transform(weight, inv_t=True, dim=-1)
        if out_transform:
            weight = out_transform(weight, inv_t=True, dim=0)
            if bias is not None:
                bias = out_transform(bias, inv_t=True, dim=0)

        if self.weight_quantizer is not None:
            # MinMax is fast on GPU, but MSE searches can cause OOM on large matrices.
            # Process in chunks to keep memory usage low while staying on GPU.
            orig_track = self.weight_quantizer._track_global_scale
            if orig_track and hasattr(self.weight_quantizer, "scale_precision") and self.weight_quantizer.scale_precision.value == "e4m3":
                from .quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX
                from .quantizer import get_reciprocal
                max_val = weight.abs().max().to(torch.float32).view(1)
                current_global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(max_val)
                self.weight_quantizer.global_scale = torch.minimum(self.weight_quantizer.global_scale.to(weight.device), current_global_scale)
                self.weight_quantizer._track_global_scale = False
                
            chunk_size = 256
            w_scales_list = []
            w_zeros_list = []
            for i in range(0, weight.shape[0], chunk_size):
                chunk = weight[i:i+chunk_size]
                s, z = self.weight_quantizer.get_quantization_params(chunk)
                w_scales_list.append(s)
                if z is not None:
                    w_zeros_list.append(z)
            w_scales = torch.cat(w_scales_list, dim=0)
            w_zeros = torch.cat(w_zeros_list, dim=0) if w_zeros_list else None
            
            weight = self.weight_quantizer(weight, w_scales, w_zeros)
            self.weight_quantizer._track_global_scale = False

        if self.act_quantizer is not None:
            self.act_quantizer._track_global_scale = False

        self.weight.data = weight
        if bias is not None:
            self.bias.data = bias

        self._train_mode = False
