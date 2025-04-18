import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

class Wikitext2DataModule(pl.LightningDataModule):
    """
    LightningDataModule for masked‐language‐modeling on Wikitext‑2.
    All args are primitives so LightningCLI can pick them up automatically.
    """

    def __init__(
        self,
        # where to cache wikitext2
        data_dir: str = "./data/wikitext2",
        # tokenizer & collator
        tokenizer_name: str = "bert-base-uncased",
        mlm_probability: float = 0.15,
        max_length: int = 512,
        # loader config
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = True,
    ):
        super().__init__()

        # fixed to wikitext‑2
        self.dataset_name = "wikitext"
        self.dataset_config = "wikitext-2-raw-v1"

        # user‐configurable
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.mlm_probability = mlm_probability
        self.max_length = max_length

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # placeholders
        self.tokenizer = None
        self.collator = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        # download wikitext‑2
        load_dataset(
            self.dataset_name,
            self.dataset_config,
            cache_dir=self.data_dir,
        )

    def setup(self, stage: Optional[str] = None):
        # initialize tokenizer + collator once
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=self.mlm_probability,
            )

        splits = []
        if stage in (None, "fit"):
            splits += ["train", "validation"]
        if stage in (None, "test"):
            splits += ["test"]

        for split in splits:
            ds = load_dataset(
                self.dataset_name,
                self.dataset_config,
                cache_dir=self.data_dir,
                split=split,
            ).map(
                self._tokenize,
                batched=True,
                remove_columns=["text"],
            )

            key = "val" if split == "validation" else split
            setattr(self, f"{key}_dataset", ds)

    def _tokenize(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )
