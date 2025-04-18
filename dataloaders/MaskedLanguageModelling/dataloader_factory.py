from .wikitext import Wikitext2DataModule
from pytorch_lightning import LightningDataModule


def MLMDataModule(dataset_name: str="wikitext2", *args, **kwargs) -> LightningDataModule: 
    if "wikitext2" in dataset_name.lower(): 
        return Wikitext2DataModule(*args, **kwargs)
    else: 
        raise ValueError(f"Dataset {dataset_name} not supported")
