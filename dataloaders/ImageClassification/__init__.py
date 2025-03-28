from .cifar10 import CIFAR10DataModule
from .mnist import MNISTDataModule
from config import CIFAR10DataConfig, MNISTDataConfig
__all__ = ["BaseDataModule", "CIFAR10DataModule", "MNISTDataModule"]


ALL_IMAGE_CLASSIFICATION_DATALOADERS = {
    "cifar10": (CIFAR10DataModule, CIFAR10DataConfig),
    "mnist": (MNISTDataModule, MNISTDataConfig),
}
