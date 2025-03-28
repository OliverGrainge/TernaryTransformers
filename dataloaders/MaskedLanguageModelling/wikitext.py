import os
from typing import Optional

from datasets import load_dataset

from config import DataConfig

from .base import BaseMLMDataModule


class WikiTextMLMDataModule(BaseMLMDataModule):
    def __init__(self, data_config: DataConfig):
        super().__init__(data_config)
        self.dataset_name = "wikitext"
        self.dataset_config = "wikitext-2-raw-v1"

    def prepare_data(self):
        os.makedirs(self.data_config.data_dir, exist_ok=True)
        load_dataset(
            self.dataset_name, self.dataset_config, cache_dir=self.data_config.data_dir
        )

    def setup(self, stage: Optional[str] = None):
        stage = stage or "fit"
        splits = self.split_mapping.get(stage, [])

        for split in splits:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                cache_dir=self.data_config.data_dir,
                split=split,
            )

            dataset = dataset.map(
                self._tokenize_dataset, remove_columns=["text"], batched=True
            )

            # Use validation instead of val to match base class
            split_name = "validation" if split == "validation" else split
            setattr(self, f"{split_name}_dataset", dataset)
