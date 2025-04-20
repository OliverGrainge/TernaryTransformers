import os
from pathlib import Path
from typing import Literal, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WikiText2Dataset(Dataset):
    def __init__(
        self,
        split: Literal["train", "validation", "test"],
        tokenizer: AutoTokenizer,
        context_length: int = 64,
        data_dir: str = "./data/wikitext2",
    ):
        """
        WikiText-2 causal LM dataset, with cache in `data_dir`.
        """
        # ensure cache directory exists
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # load the raw wikitext-2 data (cached to data_dir)
        ds = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=split,
            cache_dir=str(self.data_dir),
        )
        # join all non-null lines into one long text
        lines = [ex["text"] for ex in ds if ex["text"] is not None]
        full_text = "\n\n".join(lines)

        # one‑shot tokenize
        tokenizer.model_max_length = int(1e12)
        enc = tokenizer(full_text, return_tensors="pt", truncation=False)
        self.ids = enc["input_ids"].squeeze(0)  # shape (N,)

        self.context_length = context_length
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return self.ids.size(0) - self.context_length

    def __getitem__(self, idx: int):
        chunk = self.ids[idx : idx + self.context_length + 1]
        return chunk[:-1], chunk[1:]


class WikiText2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/wikitext2",
        tokenizer_name: str = "gpt2",
        context_length: int = 64,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        DataModule for WikiText-2 (raw), with data cached in `data_dir`.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer_name = tokenizer_name
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_data(self):
        # ensure cache dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # download dataset + tokenizer into cache
        load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            cache_dir=str(self.data_dir),
        )
        AutoTokenizer.from_pretrained(self.tokenizer_name, cache_dir=str(self.data_dir))

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = WikiText2Dataset(
                split="train",
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                data_dir=str(self.data_dir),
            )
            self.val_dataset = WikiText2Dataset(
                split="validation",
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                data_dir=str(self.data_dir),
            )
        if stage in ("test", None):
            self.test_dataset = WikiText2Dataset(
                split="test",
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                data_dir=str(self.data_dir),
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
