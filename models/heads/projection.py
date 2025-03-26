import torch.nn as nn
from models.layers import LAYERS_REGISTRY
from config import HeadConfig

class ProjectionHead(nn.Module):
    def __init__(self, head_config: HeadConfig):
        super().__init__()
        self.linear_layer = LAYERS_REGISTRY[head_config.linear_layer.lower()](head_config.in_dim, head_config.out_dim)

    def forward(self, x):
        return self.linear_layer(x)
