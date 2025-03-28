import os
from typing import Any, Dict, Optional

from torchvision import datasets, transforms

from config import DataConfig

from .base import BaseDataModule


class MNISTDataModule(BaseDataModule):
    def __init__(
        self,
        data_config: DataConfig,
    ) -> None:
        super().__init__(data_config)
        if self.transform is None:
            self.transform = self._get_default_transform()
        self.save_configs()

    @classmethod
    def _get_default_transform(cls) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.1307,),
                    std=(0.3081,),
                ),
            ]
        )

    def save_configs(self) -> None:
        hparams: Dict[str, Any] = {
            **{f"data_{k}": v for k, v in self.data_config.__dict__.items()},
        }
        self.save_hyperparameters(hparams)

    def prepare_data(self) -> None:
        # Download MNIST data if not already downloaded
        for train in (True, False):
            datasets.MNIST(self.data_config.data_dir, train=train, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = datasets.MNIST(
                self.data_config.data_dir, train=True, transform=self.transform
            )
            self.val_dataset = datasets.MNIST(
                self.data_config.data_dir, train=False, transform=self.transform
            )

        if stage in ("test", None):
            self.test_dataset = datasets.MNIST(
                self.data_config.data_dir, train=False, transform=self.transform
            )
