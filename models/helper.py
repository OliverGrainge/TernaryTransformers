import inspect

import torch
import torch.nn as nn

from config import Config
from models.heads import HEADS_REGISTRY
from models.transformers import TRANSFORMERS_REGISTRY


class Model(nn.Module):
    def __init__(self, transformer, head):
        super().__init__()
        self.transformer = transformer
        self.head = head

    def forward(self, x, **kwargs):
        transformer_output = self.transformer(x, **kwargs)
        return self.head(transformer_output)


def create_model(
    model_config: Config,
):
    transformer_cls = TRANSFORMERS_REGISTRY[model_config.backbone_type.lower()]
    head_cls = HEADS_REGISTRY[model_config.head_type.lower()]

    # Create model components with all parameters
    transformer = transformer_cls(model_config)
    head = head_cls(model_config)
    model = Model(transformer, head)
    return model
