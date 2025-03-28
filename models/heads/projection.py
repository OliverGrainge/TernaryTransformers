import torch.nn as nn

from config import Config
from models.layers import LAYERS_REGISTRY


class ProjectionHead(nn.Module):
    def __init__(self, model_config: Config):
        super().__init__()
        self.linear_layer = LAYERS_REGISTRY[model_config.head_linear_layer.lower()](
            model_config.head_in_dim, model_config.head_out_dim
        )

    def forward(self, x):
        return self.linear_layer(x)
