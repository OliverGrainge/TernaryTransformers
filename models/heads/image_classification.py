import torch
import torch.nn as nn

from config import ModelConfig


class ImageClassificationHead(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
    ):
        super().__init__()
        self.model_config = model_config
        # Build MLP layers
        layers = []
        in_features = model_config.head_in_dim

        for _ in range(model_config.head_depth - 1):
            layers.extend(
                [
                    nn.Linear(in_features, model_config.head_dim),
                    nn.GELU(),
                    nn.Dropout(model_config.head_dropout),
                ]
            )
            in_features = model_config.head_dim

        layers.extend(
            [
                nn.Linear(in_features, model_config.head_dim),
                nn.Dropout(model_config.head_dropout),
                nn.Linear(model_config.head_dim, model_config.head_out_dim),
            ]
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_config.backbone_type.lower() != "mlp":
            x = x[:, 0]
        x = self.mlp(x)
        return x
