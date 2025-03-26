import torch
import torch.nn as nn
from config import HeadConfig


class ImageClassificationHead(nn.Module):
    def __init__(
        self,
        head_config: HeadConfig,
    ):
        super().__init__()

        # Build MLP layers
        layers = []
        in_features = head_config.in_dim

        for _ in range(head_config.depth - 1):
            layers.extend(
                [nn.Linear(in_features, head_config.dim), nn.GELU(), nn.Dropout(head_config.dropout)]
            )
            in_features = head_config.dim

        layers.extend(
            [
                nn.Linear(in_features, head_config.dim),
                nn.Dropout(head_config.dropout),
                nn.Linear(head_config.dim, head_config.out_dim),
            ]
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0]
        x = self.mlp(x)
        return x
