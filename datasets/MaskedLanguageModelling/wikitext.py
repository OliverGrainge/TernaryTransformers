import datasets
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from typing import Optional
from config import DataConfig


class WikiTextMLMDataModule(pl.LightningDataModule):
    def __init__(self, data_config: DataConfig):
        super().__init__()
        self.data_config = data_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.data_config.tokenizer_name)
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.train_config.mlm_probability,
        )

    def prepare_data(self):
        load_dataset(
            "wikitext", "wikitext-2-raw-v1", cache_dir=self.data_config.data_dir
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                cache_dir=self.data_config.data_dir,
                split="train",
            )
            self.val_dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                cache_dir=self.data_config.data_dir,
                split="test",
            )
        elif stage == "test":
            self.test_dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                cache_dir=self.data_config.data_dir,
                split="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            collate_fn=self.collator,
        )
