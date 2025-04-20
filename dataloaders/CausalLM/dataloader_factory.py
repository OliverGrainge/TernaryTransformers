from .shakespeare import ShakespeareDataModule
from .wikitext2 import WikiText2DataModule
from pytorch_lightning import LightningDataModule


def CausalLMDataModule(dataset_name: str="shakespeare", *args, **kwargs) -> LightningDataModule: 
    if "shakespeare" in dataset_name.lower(): 
        return ShakespeareDataModule(*args, **kwargs)
    elif "wikitext2" in dataset_name.lower(): 
        return WikiText2DataModule(*args, **kwargs)
    else: 
        raise ValueError(f"Dataset {dataset_name} not supported")