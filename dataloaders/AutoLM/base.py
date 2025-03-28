from pathlib import Path
from typing import Literal, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os 

from config import Config
from typing import Dict, Any

class AutoregressiveLMDataModule(pl.LightningDataModule):
    def __init__(self, data_config: Config):
        """Base class for autoregressive language modeling data modules.

        Args:
            data_config: Configuration for data loading
        """
        super().__init__()
        self._validate_config(data_config)
        self.data_config = data_config
        

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.save_configs()

    def save_configs(self) -> None:
        hparams: Dict[str, Any] = {
            **{f"data_{k}": v for k, v in self.data_config.__dict__.items()},
        }
        self.save_hyperparameters(hparams)

    def _validate_config(self, data_config: Config) -> None:
        """Validate that the data_config contains all required parameters for autoregressive LM."""
        required_params = [
            "data_dir",
            "tokenizer_name",
            "batch_size",
            "num_workers",
            "context_length",
        ]
        os.makedirs(data_config.data_dir, exist_ok=True)
        missing_params = [
            param for param in required_params if not hasattr(data_config, param)
        ]
        if missing_params:
            raise ValueError(
                f"Missing required parameters in data_config: {', '.join(missing_params)}"
            )

        # Validate data directory exists or can be created
        Path(data_config.data_dir).mkdir(parents=True, exist_ok=True)

        # Validate context length
        if data_config.context_length <= 0:
            raise ValueError("Context length must be positive")

    def _load_data(self) -> str:
        """Load or download dataset. Must be implemented by child classes."""
        raise NotImplementedError

    def _split_data(
        self, text: str, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> dict[str, str]:
        """Split data into train/val/test sets."""
        if not 0 < train_ratio + val_ratio < 1:
            raise ValueError("Invalid split ratios")

        n = len(text)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return {
            "train": text[:train_end],
            "val": text[train_end:val_end],
            "test": text[val_end:],
        }

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(split="train")
            self.val_dataset = self._create_dataset(split="val")
        if stage == "test":
            self.test_dataset = self._create_dataset(split="test")

    def _create_dataset(self, split: Literal["train", "val", "test"]):
        """Create dataset for a specific split. Must be implemented by child classes."""
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
        )

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
