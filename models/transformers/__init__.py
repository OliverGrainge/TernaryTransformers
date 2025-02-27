from .vit import ViT
from .mlp import MLP

__all__ = ["ViT"]

TRANSFORMERS_REGISTRY = {
    "vit": ViT,
    "mlp": MLP,
}
