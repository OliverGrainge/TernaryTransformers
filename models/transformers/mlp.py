from typing import Tuple, Type, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.blocks import ViTAttention, ViTFeedForward
from models.layers import LAYERS_REGISTRY


class MLP(nn.Module):
    def __init__(self, in_dim: int, mlp_dim: int, out_dim: int=10.0, dropout: float = 0.0, linear_layer: str = "Linear", activation_layer: str = "RELU", norm_layer: str = "LayerNorm"):
        super().__init__()
        linear_layer = LAYERS_REGISTRY[linear_layer.lower()]
        activation_layer = LAYERS_REGISTRY[activation_layer.lower()]
        norm_layer = LAYERS_REGISTRY[norm_layer.lower()]
        self.norm = norm_layer(in_dim)
        self.linear1 = linear_layer(in_dim, mlp_dim)
        self.activation = activation_layer()
        self.linear2 = linear_layer(mlp_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


