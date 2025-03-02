from .vitattention import ViTAttention
from .vitfeedforward import ViTFeedForward
from .attention import TransformerAttention

__all__ = ["ViTAttention", "ViTFeedForward", "TransformerAttention"]

BLOCKS_REGISTRY = {
    "vitattention": ViTAttention,
    "vitfeedforward": ViTFeedForward,
    "attention": TransformerAttention,
}
