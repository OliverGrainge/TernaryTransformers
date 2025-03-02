from .vit import ViT, MiniViT, ViTSmall
from .mlp import MLP
from .bert import Bert

__all__ = ["ViT", "MiniViT", "ViTSmall", "MLP", "Bert"]

TRANSFORMERS_REGISTRY = {
    "vit": ViT,
    "minivit": MiniViT, 
    "vitsmall": ViTSmall, 
    "mlp": MLP,
    "bert": Bert,
}
