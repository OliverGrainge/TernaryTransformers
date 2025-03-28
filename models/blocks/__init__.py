from .attention import TransformerAttention
from .feedforward import FeedForward
from .vitattention import ViTAttention

__all__ = ["ViTAttention", "FeedForward", "TransformerAttention"]

BLOCKS_REGISTRY = {
    "vitattention": ViTAttention,
    "feedforward": FeedForward,
    "attention": TransformerAttention,
}
