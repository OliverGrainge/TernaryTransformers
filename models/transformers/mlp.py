from typing import Tuple, Type, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.blocks import ViTAttention, ViTFeedForward
from models.layers import LAYERS_REGISTRY
from config import ModelConfig


class MLP(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
    ):
        super().__init__()
        linear_layer = LAYERS_REGISTRY[model_config.mlp_linear_layer.lower()]
        activation_layer = LAYERS_REGISTRY[model_config.mlp_activation_layer.lower()]
        norm_layer = LAYERS_REGISTRY[model_config.mlp_norm_layer.lower()]

        self.norm = norm_layer(model_config.mlp_in_dim)
        layers = []

        # First layer
        layers.append(linear_layer(model_config.mlp_in_dim, model_config.mlp_dim))
        layers.append(activation_layer())
        layers.append(nn.Dropout(model_config.mlp_dropout))

        # Middle layers
        for _ in range(model_config.mlp_depth - 1):
            layers.append(linear_layer(model_config.mlp_dim, model_config.mlp_dim))
            layers.append(activation_layer())
            layers.append(nn.Dropout(model_config.mlp_dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.norm(x)
        x = self.layers(x)
        return x
