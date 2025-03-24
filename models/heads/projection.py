import torch.nn as nn
from models.layers import LAYERS_REGISTRY


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, linear_layer="Linear"):
        super().__init__()
        self.linear_layer = LAYERS_REGISTRY[linear_layer.lower()](in_dim, out_dim)

    def forward(self, x):
        return self.linear_layer(x)
