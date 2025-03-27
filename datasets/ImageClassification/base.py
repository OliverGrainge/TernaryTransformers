import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(pl.LightningDataModule):
    """
    Base class for PyTorch Lightning DataModules that handles common functionality.
    Subclasses should implement prepare_data(), setup() and the dataset creation methods.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        transform = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform

        # These will be populated in setup()
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """
        Download data if needed. This method is called only from a single process.
        Should not set state here (use setup instead)
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data, called on every GPU separately
        """
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
