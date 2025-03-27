import torch.nn as nn

from .image_classification import ImageClassificationHead
from .mlm import MLMHead
from .projection import ProjectionHead

__all__ = ["ImageClassificationHead", "ProjectionHead", "MLMHead"]

HEADS_REGISTRY = {
    "imageclassificationhead": ImageClassificationHead,
    "mlmhead": MLMHead,
    "projectionhead": ProjectionHead,
    "none": nn.Identity,
}
