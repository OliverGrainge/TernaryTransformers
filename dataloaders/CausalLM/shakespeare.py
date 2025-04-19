import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import requests
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CharacterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str = "./data/shakespeare",
        tokenizer: AutoTokenizer = None,
        context_length: int = 64,
        split: Literal["train", "val", "test"] = "train",
    ):
        """Shakespeare character-level dataset.

        Args:
            data_dir: Directory for data storage
            tokenizer: Tokenizer for text processing
            context_length: Length of context window
            split: Which data split to use
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Load and split data
        text = self._load_shakespeare_data()
        split_data = self._split_data(text)
        data = split_data[split]

        # Tokenize the data
        tokenizer.model_max_length = int(1e12)
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


class ShakespeareDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/shakespeare",
        context_length: int = 64,
        batch_size: int = 32,
        num_workers: int = 6,
        tokenizer_name: str = "gpt2",
        vocab_size: int = 1024
    ):
        """Initialize Shakespeare data module.

        Args:
            data_dir: Directory for data storage
            context_length: Length of context window
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            tokenizer_name: Name of the pretrained tokenizer to use
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.vocab_size = vocab_size


        # Initialize tokenizer
        if tokenizer_name == "bpe" and not os.path.exists(os.path.join(self.data_dir, "shakespeare-tokenizer.json")): 
            tokenizer = Tokenizer(models.BPE())
            tokenizer.normalizer = normalizers.NFD()
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
            tokenizer.train([os.path.join(self.data_dir, "shakespeare.txt")], trainer=trainer)
            tokenizer.save(os.path.join(self.data_dir, "shakespeare-tokenizer.json"))

        elif tokenizer_name == "bpe" and os.path.exists(os.path.join(self.data_dir, "shakespeare-tokenizer.json")): 
            tokenizer = Tokenizer.from_file("shakespeare-tokenizer.json")
        else: 
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # These will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """Download data if needed. This method is called only from a single process."""
        # Create dataset temporarily to trigger download if needed
        CharacterDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            context_length=self.context_length,
            split="train"
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation and testing."""
        if stage in ("fit", None):
            self.train_dataset = CharacterDataset(
                data_dir=self.data_dir,
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                split="train"
            )
            self.val_dataset = CharacterDataset(
                data_dir=self.data_dir,
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                split="val"
            )

        if stage in ("test", None):
            self.test_dataset = CharacterDataset(
                data_dir=self.data_dir,
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                split="test"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
