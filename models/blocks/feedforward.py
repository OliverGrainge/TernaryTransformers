from typing import Type

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        linear_layer: Type[nn.Linear] = nn.Linear,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        activation_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            norm_layer(dim),
            linear_layer(dim, hidden_dim),
            activation_layer(),
            nn.Dropout(dropout),
            linear_layer(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
