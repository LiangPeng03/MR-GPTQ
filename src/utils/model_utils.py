import numpy as np
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import AutoConfig

from .common_utils import to
from .llama_utils import QuantizedLlamaMLP, QuantizedLlamaAttention
try:
    from .qwen3_utils import QuantizedQwen3MLP, QuantizedQwen3Attention
except (ImportError, ModuleNotFoundError):
    QuantizedQwen3MLP = None
    QuantizedQwen3Attention = None

### Calibration utils and modules

LINEAR_LAYERS = (nn.Linear, _ConvNd)


class ForwardInterrupt(Exception):
    pass


class InputCollector(nn.Module):

    def __init__(self, module: nn.Module, cpu_offload: bool = False):
        super().__init__()
        self.module = module
        self.cpu_offload = cpu_offload
        self.input_args = []
        self.input_kwargs = []

    def forward(self, *input_args, **input_kwargs):
        """
        Assumes that the wrapped module has a single
        input that can reside in inputs or input_kwargs.
        """
        if self.cpu_offload:
            input_args = to(input_args, device="cpu")
            input_kwargs = to(input_kwargs, device="cpu")
        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)
        raise ForwardInterrupt
    
def get_number_of_rows_and_cols(layer):
    return layer.weight.shape[0], np.prod(layer.weight.shape[1:])

def get_mlp_layer(config: AutoConfig):
    if config.model_type == "llama":
        return QuantizedLlamaMLP
    elif config.model_type == "qwen3":
        if QuantizedQwen3MLP is None:
            raise ImportError("Qwen3 dependencies (transformers.models.qwen3) are missing in your environment.")
        return QuantizedQwen3MLP
    else:
        raise ValueError(f"Model type {config.model_type} not supported")

def get_attention_layer(config: AutoConfig):
    if config.model_type == "llama":
        return QuantizedLlamaAttention
    elif config.model_type == "qwen3":
        if QuantizedQwen3Attention is None:
            raise ImportError("Qwen3 dependencies (transformers.models.qwen3) are missing in your environment.")
        return QuantizedQwen3Attention
    else:
        raise ValueError(f"Model type {config.model_type} not supported")
