from typing import Tuple, Type, Union

import torch
import torch.nn as nn

from models.layers import LAYERS_REGISTRY


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float = 0.1,
        linear_layer: str = "Linear",
        activation_layer: str = "RELU",
        norm_layer: str = "LayerNorm",
    ):
        """
        Args:
            in_dim (int): Input dimension
            hidden_dim (int): Hidden dimension for middle layers
            depth (int): Number of linear layers
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            linear_layer (str, optional): Type of linear layer. Defaults to "Linear".
            activation_layer (str, optional): Type of activation function. Defaults to "RELU".
            norm_layer (str, optional): Type of normalization layer. Defaults to "LayerNorm".
        """
        super().__init__()
        
        # Get layer types from registry
        linear = LAYERS_REGISTRY[linear_layer.lower()]
        activation = LAYERS_REGISTRY[activation_layer.lower()]
        norm = LAYERS_REGISTRY[norm_layer.lower()]

        self.norm = norm(in_dim)
        layers = []

        # First layer
        layers.append(linear(in_dim, hidden_dim))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))

        # Middle layers
        for _ in range(depth - 1):
            layers.append(linear(hidden_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.layers(x)
        return x
