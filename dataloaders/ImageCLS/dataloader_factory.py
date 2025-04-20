from .cifar10 import CIFAR10DataModule
from .tinyimagenet import TinyImageNetHFDataModule
from .imagenet import ImageNetDataModule
from pytorch_lightning import LightningDataModule


def ImageCLSDataModule(dataset_name: str="cifar10", *args, **kwargs) -> LightningDataModule: 
    if "cifar10" in dataset_name.lower(): 
        return CIFAR10DataModule(*args, **kwargs)
    elif "tinyimagenet" in dataset_name.lower(): 
        return TinyImageNetHFDataModule(*args, **kwargs)
    elif "imagenet" in dataset_name.lower(): 
        return ImageNetDataModule(*args, **kwargs)
    else: 
        raise ValueError(f"Dataset {dataset_name} not supported")
