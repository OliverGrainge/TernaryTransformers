import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from config import DataConfig


class BaseDataModule(pl.LightningDataModule):
    """
    Base class for PyTorch Lightning DataModules that handles common functionality.
    Subclasses should implement prepare_data(), setup() and the dataset creation methods.
    """

    def __init__(
        self,
        data_config: DataConfig,
        transform=None,
    ):
        super().__init__()
        self._validate_config(data_config)
        self.data_config = data_config
        self.transform = transform
        # These will be populated in setup()
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _validate_config(self, data_config: DataConfig) -> None:
        """Validate that the data_config contains all required parameters for image classification."""
        required_params = ["data_dir", "batch_size", "num_workers", "pin_memory"]

        missing_params = [
            param for param in required_params if not hasattr(data_config, param)
        ]
        if missing_params:
            raise ValueError(
                f"Missing required parameters in data_config: {', '.join(missing_params)}"
            )

        # Validate data directory exists
        os.makedirs(data_config.data_dir, exist_ok=True)

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
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
        )
