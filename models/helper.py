from models.transformers import TRANSFORMERS_REGISTRY
from models.heads import HEADS_REGISTRY
import torch.nn as nn


def create_model(backbone: str, head: str, transformer_kwargs: dict={},  head_kwargs: dict={}):
    transformer = TRANSFORMERS_REGISTRY[backbone.lower()](**transformer_kwargs)
    head = HEADS_REGISTRY[head.lower()](**head_kwargs)
    model = nn.Sequential(transformer, head)
    return model


