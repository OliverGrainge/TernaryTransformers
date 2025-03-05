from .image_classification import ImageClassificationHead
from .mlm import MLMHead
from .projection import ProjectionHead

import torch.nn as nn

__all__ = ["ImageClassificationHead", "ProjectionHead", "MLMHead"]

HEADS_REGISTRY = {
    "imageclassificationhead": ImageClassificationHead,
    "mlmhead": MLMHead,
    "projectionhead": ProjectionHead,
    "none": nn.Identity,
}
