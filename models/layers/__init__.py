import torch.nn as nn

from .bitlinear import BitLinear
from .ternaryllm import TLinear
__all__ = ["BitLinear", "TLinear", "Linear", "LayerNorm", "RMSNorm"]

LAYERS_REGISTRY = {
    "bitlinear": BitLinear,
    "tlinear": TLinear,
    "linear": nn.Linear,
    "layernorm": nn.LayerNorm,
    "rmsnorm": nn.RMSNorm,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "identity": nn.Identity,
}
