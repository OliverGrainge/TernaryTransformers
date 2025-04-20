import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from typing import Optional


class ImageNetDataset:
    """
    A wrapper to apply torchvision transforms to HuggingFace ImageNet-1k examples.
    Each example is a dict: {'image': PIL.Image, 'label': int}
    """
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        img = example["image"].convert("RGB")
        img = self.transform(img)
        label = example["label"]
        return img, label


class ImageNetDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the HuggingFace 'ILSVRC/imagenet-1k' dataset.

    Args:
        data_dir: directory to cache/download the HF dataset
        batch_size: number of samples per batch
        num_workers: number of workers for DataLoader
        pin_memory: whether to pin memory in DataLoader
        image_size: target resolution (pixels) for H and W
    """
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_dir: str = "./data",  # cache directory for HF cache
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = False,
        image_size: int = 224,  # typical ImageNet input size
    ):
        super().__init__()
        # save all hyperparameters including data_dir
        self.save_hyperparameters()

        # Training transforms with augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.hparams.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(type(self).mean, type(self).std),
        ])

        # Validation/Test transforms without augmentation
        self.val_transform = transforms.Compose([
            transforms.Resize(256),               # shorter side to 256px
            transforms.CenterCrop(self.hparams.image_size),
            transforms.ToTensor(),
            transforms.Normalize(type(self).mean, type(self).std),
        ])

    def prepare_data(self):
        """
        Download/cache the HF dataset splits.
        """
        # Only download once
        load_dataset(
            "ILSVRC/imagenet-1k", split="train", cache_dir=self.hparams.data_dir
        )
        load_dataset(
            "ILSVRC/imagenet-1k", split="validation", cache_dir=self.hparams.data_dir
        )
        load_dataset(
            "ILSVRC/imagenet-1k", split="test", cache_dir=self.hparams.data_dir
        )

    def setup(self, stage: Optional[str] = None):
        """
        Load and wrap HF datasets for different stages.
        """
        if stage in ("fit", None):
            train_hf = load_dataset(
                "ILSVRC/imagenet-1k", split="train", cache_dir=self.hparams.data_dir
            )
            val_hf = load_dataset(
                "ILSVRC/imagenet-1k", split="validation", cache_dir=self.hparams.data_dir
            )
            self.train_dataset = HFImageNetDataset(train_hf, self.train_transform)
            self.val_dataset   = HFImageNetDataset(val_hf,   self.val_transform)

        if stage in ("test", None):
            test_hf = load_dataset(
                "ILSVRC/imagenet-1k", split="test", cache_dir=self.hparams.data_dir
            )
            self.test_dataset = ImageNetDataset(test_hf, self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
