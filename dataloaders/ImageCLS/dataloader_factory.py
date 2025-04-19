from .cifar10 import CIFAR10DataModule
from pytorch_lightning import LightningDataModule


def ImageCLSDataModule(dataset_name: str="cifar10", *args, **kwargs) -> LightningDataModule: 
    if "cifar10" in dataset_name.lower(): 
        return CIFAR10DataModule(*args, **kwargs)
    else: 
        raise ValueError(f"Dataset {dataset_name} not supported")
