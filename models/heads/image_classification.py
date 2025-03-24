import torch
import torch.nn as nn


class ImageClassificationHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dim: int,
        mlp_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Build MLP layers
        layers = []
        in_features = dim

        for _ in range(num_layers - 1):
            layers.extend(
                [nn.Linear(in_features, mlp_dim), nn.GELU(), nn.Dropout(dropout)]
            )
            in_features = mlp_dim

        layers.extend(
            [
                nn.Linear(in_features, mlp_dim),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, num_classes),
            ]
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0]
        x = self.mlp(x)
        return x
