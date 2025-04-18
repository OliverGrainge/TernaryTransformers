import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional

class CIFAR10DataModule(pl.LightningDataModule):
    """
    A standalone LightningDataModule for CIFAR-10.
    All constructor arguments are exposed to the LightningCLI.

    Args:
        data_dir: path to store/download the dataset
        batch_size: number of samples per batch
        num_workers: processes for data loading
        pin_memory: whether to pin memory in DataLoader
        image_size: target resolution (pixels) for H and W
    """
    # Default normalization stats for CIFAR-10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        # Automatically saves args to self.hparams for LightningCLI
        self.save_hyperparameters()

        # Build default transform: resize -> to tensor -> normalize
        self.transform = transforms.Compose([
            transforms.Resize(self.hparams.image_size),
            transforms.ToTensor(),
            transforms.Normalize(type(self).mean, type(self).std),
        ])


    def prepare_data(self) -> None:
        """
        Download CIFAR-10 train and test splits if not already present.
        """
        datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for different stages (fit, test).
        """
        if stage in ("fit", None):
            self.train_dataset = datasets.CIFAR10(
                self.hparams.data_dir,
                train=True,
                transform=self.transform,
            )
            self.val_dataset = datasets.CIFAR10(
                self.hparams.data_dir,
                train=False,
                transform=self.transform,
            )
        if stage in ("test", None):
            self.test_dataset = datasets.CIFAR10(
                self.hparams.data_dir,
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )