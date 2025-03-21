import torch.nn as nn

from .bitlinear import BitLinear
from .ternaryllm import TLinear
from .trilm import TriLinear
__all__ = ["BitLinear", "TLinear", "TriLinear", "Linear", "LayerNorm", "RMSNorm"]

LAYERS_REGISTRY = {
    "bitlinear": BitLinear,
    "tlinear": TLinear,
    "trilinear": TriLinear,
    "linear": nn.Linear,
    "layernorm": nn.LayerNorm,
    "rmsnorm": nn.RMSNorm,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "identity": nn.Identity,
}
