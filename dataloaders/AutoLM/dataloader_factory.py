from .shakespeare import ShakespeareDataModule
from pytorch_lightning import LightningDataModule


def AutoLMDataModule(dataset_name: str="shakespeare", *args, **kwargs) -> LightningDataModule: 
    if "shakespeare" in dataset_name.lower(): 
        return ShakespeareDataModule(*args, **kwargs)
    else: 
        raise ValueError(f"Dataset {dataset_name} not supported")