from typing import Tuple, Type, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.blocks import ViTAttention, ViTFeedForward
from models.layers import LAYERS_REGISTRY
from config import BackboneConfig


class MLP(nn.Module):
    def __init__(
        self,
        backbone_config: BackboneConfig,
    ):
        super().__init__()
        linear_layer = LAYERS_REGISTRY[backbone_config.feedforward_linear_layer]
        activation_layer = LAYERS_REGISTRY[backbone_config.feedforward_activation_layer]
        norm_layer = LAYERS_REGISTRY[backbone_config.feedforward_norm_layer]

        self.norm = norm_layer(backbone_config.in_dim)
        layers = []

        # First layer
        layers.append(linear_layer(backbone_config.in_dim, backbone_config.dim))
        layers.append(activation_layer())
        layers.append(nn.Dropout(backbone_config.dropout))

        # Middle layers
        for _ in range(backbone_config.depth - 2):
            layers.append(linear_layer(backbone_config.dim, backbone_config.dim))
            layers.append(activation_layer())
            layers.append(nn.Dropout(backbone_config.dropout))

        # Final layer
        layers.append(linear_layer(backbone_config.dim, backbone_config.out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.layers(x)
        return x
