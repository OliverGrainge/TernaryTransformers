import torch.nn as nn
from models.layers import LAYERS_REGISTRY
from config import ModelConfig

class ProjectionHead(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.linear_layer = LAYERS_REGISTRY[model_config.head_linear_layer.lower()](model_config.head_in_dim, model_config.head_out_dim)

    def forward(self, x):
        return self.linear_layer(x)
