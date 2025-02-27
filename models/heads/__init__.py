from .image_classification import ImageClassificationHead

__all__ = ["ImageClassificationHead"]

HEADS_REGISTRY = {
    "imageclassificationhead": ImageClassificationHead,
}