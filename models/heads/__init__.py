from .image_classification import ImageClassificationHead
from .mlm import MLMHead
import torch.nn as nn

__all__ = ["ImageClassificationHead"]

HEADS_REGISTRY = {
    "imageclassificationhead": ImageClassificationHead,
    "mlmhead": MLMHead,
    "none": nn.Identity,
    
}
