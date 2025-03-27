from .attention import TransformerAttention
from .vitattention import ViTAttention
from .feedforward import FeedForward

__all__ = ["ViTAttention", "FeedForward", "TransformerAttention"]

BLOCKS_REGISTRY = {
    "vitattention": ViTAttention,
    "feedforward": FeedForward,
    "attention": TransformerAttention,
}
