import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import requests
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Config

from .base import AutoregressiveLMDataModule


class CharacterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_config: Config,
        tokenizer: AutoTokenizer,
        context_length: int = 64,
        split: Literal["train", "val", "test"] = "train",
    ):
        """Shakespeare character-level dataset.

        Args:
            data_config: Configuration for data loading
            tokenizer: Tokenizer for text processing
            context_length: Length of context window
            split: Which data split to use
        """
        self.data_dir = Path(data_config.data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Load and split data
        text = self._load_shakespeare_data()
        split_data = self._split_data(
            text
        )
        data = split_data[split]

        # Tokenize the data
        encodings = tokenizer(data, truncation=False, return_tensors="pt")
        self.data = encodings["input_ids"].squeeze()
        self.context_length = context_length
        self.vocab_size = tokenizer.vocab_size

    def _load_shakespeare_data(self) -> str:
        """Load or download Shakespeare dataset."""
        shakespeare_path = self.data_dir / "shakespeare.txt"
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

        if shakespeare_path.exists():
            return shakespeare_path.read_text()

        try:
            response = requests.get(url)
            response.raise_for_status()
            text = response.text
            shakespeare_path.write_text(text)
            return text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download Shakespeare dataset: {e}")

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

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


class ShakespeareDataModule(AutoregressiveLMDataModule):
    def __init__(self, data_config: Config):
        """Initialize Shakespeare data module.

        Args:
            data_config: Configuration for data loading
        """
        super().__init__(data_config)

    def _load_data(self) -> str:
        """Load or download Shakespeare dataset."""
        shakespeare_path = Path(self.data_config.data_dir) / "shakespeare.txt"
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

        if shakespeare_path.exists():
            return shakespeare_path.read_text()

        try:
            response = requests.get(url)
            response.raise_for_status()
            text = response.text
            shakespeare_path.write_text(text)
            return text
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download Shakespeare dataset: {e}")

    def _create_dataset(self, split: Literal["train", "val", "test"]):
        """Create Shakespeare dataset for a specific split."""
        return CharacterDataset(
            self.data_config,
            self.tokenizer,
            context_length=self.data_config.context_length,
            split=split,
        )
