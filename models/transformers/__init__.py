from .vit import ViT
from .mlp import MLP
from .bert import Bert
from .gpt import CausalTransformer, GPT

__all__ = ["ViT", "MiniViT", "ViTSmall", "MLP", "Bert", "CausalTransformer"]

TRANSFORMERS_REGISTRY = {
    "vit": ViT,
    "mlp": MLP,
    "bert": Bert,
    "causaltransformer": CausalTransformer,
    "gpt": GPT,
}
