from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from config import DataConfig


class BaseMLMDataModule(pl.LightningDataModule):
    def __init__(self, data_config: DataConfig):
        super().__init__()
        self._validate_config(data_config)
        self.data_config = data_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.data_config.tokenizer_name)
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.data_config.mlm_probability,
        )
        self.split_mapping = {"fit": ["train", "validation"], "test": ["test"]}

    def _validate_config(self, data_config: DataConfig) -> None:
        """Validate that the data_config contains all required parameters."""
        required_params = [
            "tokenizer_name",
            "batch_size",
            "num_workers",
            "pin_memory",
            "mlm_probability",
        ]

        missing_params = [
            param for param in required_params if not hasattr(data_config, param)
        ]
        if missing_params:
            raise ValueError(
                f"Missing required parameters in data_config: {', '.join(missing_params)}"
            )

    def _tokenize_dataset(self, examples):
        """Helper method to tokenize text consistently."""
        return self.tokenizer(
            examples["text"], truncation=True, max_length=512, padding="max_length"
        )

    def _get_dataloader(self, dataset):
        """Helper method to create DataLoader with consistent configuration."""
        return DataLoader(
            dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            collate_fn=self.collator,
        )

    def prepare_data(self):
        """Abstract method to be implemented by child classes"""
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        """Abstract method to be implemented by child classes"""
        raise NotImplementedError

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._get_dataloader(self.validation_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset)
