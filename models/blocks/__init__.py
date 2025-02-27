from .vitattention import ViTAttention
from .vitfeedforward import ViTFeedForward

__all__ = ["ViTAttention", "ViTFeedForward"]

BLOCKS_REGISTRY = {
    "vitattention": ViTAttention,
    "vitfeedforward": ViTFeedForward,
}
