from .cifar10 import CIFAR10DataModule
from .mnist import MNISTDataModule

__all__ = ["BaseDataModule", "CIFAR10DataModule", "MNISTDataModule"]


ALL_IMAGE_CLASSIFICATION_DATALOADERS = {
    "cifar10": CIFAR10DataModule,
    "mnist": MNISTDataModule,
}
