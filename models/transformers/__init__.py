from .bert import Bert
from .gpt import GPT, CausalTransformer
from .mlp import MLP
from .vit import ViT

__all__ = ["ViT", "MiniViT", "ViTSmall", "MLP", "Bert", "CausalTransformer"]

TRANSFORMERS_REGISTRY = {
    "vit": ViT,
    "mlp": MLP,
    "bert": Bert,
    "causaltransformer": CausalTransformer,
    "gpt": GPT,
}
