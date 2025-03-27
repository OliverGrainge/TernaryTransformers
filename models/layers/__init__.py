import torch.nn as nn

from .bitlinear import BitLinear
from .ternaryllm import TLinear, TLinearChannel, TLinearGroup
from .trilm import TriLinear

__all__ = [
    "BitLinear",
    "TLinear",
    "TLinearChannel",
    "TLinearGroup",
    "TriLinear",
    "Linear",
    "LayerNorm",
    "RMSNorm",
]

LAYERS_REGISTRY = {
    "bitlinear": BitLinear,
    "tlinear": TLinear,
    "tlinear_channel": TLinearChannel,
    "tlinear_group": TLinearGroup,
    "trilinear": TriLinear,
    "linear": nn.Linear,
    "layernorm": nn.LayerNorm,
    "rmsnorm": nn.RMSNorm,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "identity": nn.Identity,
}

LINEAR_REGISTRY = {
    "bitlinear": BitLinear,
    "tlinear": TLinear,
    "tlinear_channel": TLinearChannel,
    "tlinear_group": TLinearGroup,
    "trilinear": TriLinear,
    "linear": nn.Linear,
}
