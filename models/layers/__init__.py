import torch.nn as nn

from .bitlinear import BitLinear

__all__ = ["BitLinear", "Linear", "LayerNorm", "RMSNorm"]

LAYERS_REGISTRY = {
    "bitlinear": BitLinear,
    "linear": nn.Linear,
    "layernorm": nn.LayerNorm,
    "rmsnorm": nn.RMSNorm,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "identity": nn.Identity,
}
