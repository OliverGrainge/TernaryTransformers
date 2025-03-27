from .attention import TransformerAttention
from .vitattention import ViTAttention
from .vitfeedforward import ViTFeedForward

__all__ = ["ViTAttention", "ViTFeedForward", "TransformerAttention"]

BLOCKS_REGISTRY = {
    "vitattention": ViTAttention,
    "vitfeedforward": ViTFeedForward,
    "attention": TransformerAttention,
}
