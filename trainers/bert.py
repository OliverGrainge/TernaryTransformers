from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from config import DataConfig, ModelConfig, TrainConfig, parse_configs
# Suppose you have a create_model function that returns (model, backbone_kwargs, head_kwargs)
# from your codebase
from models.helper import create_model


class BertMLMTrainer(pl.LightningModule):
    """
    A single PyTorch Lightning class that:
      - Uses a 'backbone' + 'head' via create_model(...)
      - Downloads and preprocesses WikiText-2
      - Trains via Masked Language Modeling
      - Provides train/val DataLoaders
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        data_config: DataConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config

        self.model = create_model(model_config)

        self.experiment_name = self.get_experiment_name()

        self.tokenizer = AutoTokenizer.from_pretrained(self.train_config.tokenizer_name)
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.train_config.mlm_probability,
        )
        self.train_dataset = None
        self.val_dataset = None

        self.loss_fn = nn.CrossEntropyLoss()

    def get_experiment_name(self):
        return f"Backbone[{self.model_config.backbone_type}]-LayerType[{self.model_config.feedforward_linear_layer}]-Activation[{self.model_config.feedforward_activation_layer}]"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass that matches the BERT model's interface.
        For MLM, computes loss only on masked positions where labels != -100.
        """
        # Get prediction logits from model
        logits = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # [batch_size, sequence_length, vocab_size]

        # If no labels provided, return logits
        if labels is None:
            return logits

        # Create mask for non-ignored positions (where labels != -100)
        masked_positions = labels != -100  # [batch_size, sequence_length]

        # Get logits and labels for masked positions only
        masked_logits = logits[masked_positions]  # [num_masked_tokens, vocab_size]
        masked_labels = labels[masked_positions]  # [num_masked_tokens]

        # Compute loss only on masked tokens
        loss = self.loss_fn(
            masked_logits,  # [num_masked_tokens, vocab_size]
            masked_labels,  # [num_masked_tokens]
        )

        return loss

    # -----------------
    # Lightning hooks
    # -----------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.train_config.learning_rate)

    def training_step(self, batch, batch_idx):
        """
        Each batch is a dict from the DataCollator:
          - input_ids
          - attention_mask
          - labels
          ... possibly token_type_ids
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]  # original IDs for computing MLM loss
        attention_mask = batch.get("attention_mask", None)
        token_type_ids = batch.get("token_type_ids", None)

        loss = self.forward(input_ids, attention_mask, token_type_ids, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        token_type_ids = batch.get("token_type_ids", None)

        loss = self.forward(input_ids, attention_mask, token_type_ids, labels)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def prepare_data(self):
        """
        Called once. Download the dataset here (only if not present).
        """
        load_dataset(
            "wikitext", "wikitext-2-raw-v1", cache_dir=self.data_config.data_dir
        )
        AutoTokenizer.from_pretrained(self.train_config.tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        """
        Called on each GPU/process. Create the dataset splits, tokenize, etc.
        """
        if stage == "fit" or stage is None:
            raw_datasets = load_dataset(
                "wikitext", "wikitext-2-raw-v1", cache_dir=self.data_config.data_dir
            )

            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.model_config.max_seq_len,
                    return_special_tokens_mask=True,
                )

            train_ds = raw_datasets["train"]
            val_ds = raw_datasets["validation"]

            # Tokenize
            train_ds = train_ds.map(
                tokenize_function, batched=True, remove_columns=["text"]
            )
            val_ds = val_ds.map(
                tokenize_function, batched=True, remove_columns=["text"]
            )

            # (Optionally) shorten for debugging
            if self.train_config.total_train_samples:
                train_ds = train_ds.select(
                    range(min(self.train_config.total_train_samples, len(train_ds)))
                )
            if self.train_config.total_val_samples:
                val_ds = val_ds.select(
                    range(min(self.train_config.total_val_samples, len(val_ds)))
                )

            # Convert to PyTorch
            train_ds.set_format(type="torch")
            val_ds.set_format(type="torch")

            self.train_dataset = train_ds
            self.val_dataset = val_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            collate_fn=self.collator,
        )
