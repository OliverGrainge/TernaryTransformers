from .vit import ViT, MiniViT, ViTSmall
from .mlp import MLP
from .bert import Bert
from .gpt import CausalTransformer

__all__ = ["ViT", "MiniViT", "ViTSmall", "MLP", "Bert", "CausalTransformer"]

TRANSFORMERS_REGISTRY = {
    "vit": ViT,
    "minivit": MiniViT,
    "vitsmall": ViTSmall,
    "mlp": MLP,
    "bert": Bert,
    "causaltransformer": CausalTransformer,
}
