from .image_classification import ImageClassificationHead
import torch.nn as nn

__all__ = ["ImageClassificationHead"]

HEADS_REGISTRY = {
    "imageclassificationhead": ImageClassificationHead,
    "none": nn.Identity
}
