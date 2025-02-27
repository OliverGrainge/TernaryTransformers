import torch.nn as nn

from models.heads import HEADS_REGISTRY
from models.transformers import TRANSFORMERS_REGISTRY


def create_model(
    backbone: str, head: str, backbone_kwargs: dict = {}, head_kwargs: dict = {}
):
    transformer = TRANSFORMERS_REGISTRY[backbone.lower()](**backbone_kwargs)
    head = HEADS_REGISTRY[head.lower()](**head_kwargs)
    model = nn.Sequential(transformer, head)
    return model
