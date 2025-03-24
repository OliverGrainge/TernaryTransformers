import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from typing import Optional

# Suppose you have a create_model function that returns (model, backbone_kwargs, head_kwargs)
# from your codebase
from models.helper import create_model


class WikiText2BertMLMTrainer(pl.LightningModule):
    """
    A single PyTorch Lightning class that:
      - Uses a 'backbone' + 'head' via create_model(...)
      - Downloads and preprocesses WikiText-2
      - Trains via Masked Language Modeling
      - Provides train/val DataLoaders
    """

    def __init__(
        self,
        # Model factory arguments
        backbone: str = "Bert",
        head: str = "mlmhead",
        backbone_kwargs: dict = {
            "vocab_size": 30522,
            "max_seq_len": 512,
            "dim": 768,
            "depth": 12,
            "heads": 12,
            "mlp_dim": 3072,
            "dim_head": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "num_segments": 2,
            "attention_norm_layer": "LayerNorm",
            "feedforward_norm_layer": "LayerNorm",
            "attention_activation_layer": "GELU",
            "feedforward_activation_layer": "GELU",
            "attention_linear_layer": "Linear",
            "feedforward_linear_layer": "Linear",
        },
        head_kwargs: dict = {"dim": 768, "vocab_size": 30522},
        # Data args
        tokenizer_name: str = "bert-base-uncased",
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        mlm_probability: float = 0.15,
        batch_size: int = 16,
        num_workers: int = 0,
        # Optimizer args
        learning_rate: float = 1e-4,
        # Others you might want to track
        max_seq_len: int = 128,
        total_train_samples: int = 20_000,  # For smaller debug runs
        total_val_samples: int = 1_000,
        **extra_kwargs,
    ):
        """
        Args:
            backbone, head: Names used in `create_model(...)`.
            backbone_kwargs, head_kwargs: Dicts with the config for the backbone/head.
            tokenizer_name: HF tokenizer.
            dataset_name, dataset_config: HF dataset info (like "wikitext", "wikitext-2-raw-v1").
            mlm_probability: Fraction of tokens to mask for MLM.
            batch_size, num_workers: Dataloader config.
            learning_rate: LR for Adam.
            max_seq_len: Maximum sequence length for tokenization.
            total_train_samples, total_val_samples: Subsample for quick debug, if needed.
            extra_kwargs: Catch any leftover parameters you might want to store.
        """
        super().__init__()

        # Default to empty dicts if none provided
        backbone_kwargs = backbone_kwargs or {}
        head_kwargs = head_kwargs or {}

        # 1) Create your model (backbone + head).
        #    This likely returns something like an nn.Module that does MLM.
        #    (In practice, you'd define your BERT backbone as 'backbone'
        #     and your classification / MLM head as 'head'.)
        self.model, used_backbone_kwargs, used_head_kwargs = create_model(
            backbone=backbone,
            head=head,
            backbone_kwargs=backbone_kwargs,
            head_kwargs=head_kwargs,
        )

        # 2) Flatten all hyperparameters for logging (like in CIFAR10Trainer).
        hparams_dict = {
            "backbone": backbone,
            "head": head,
            "tokenizer_name": tokenizer_name,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "mlm_probability": mlm_probability,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "learning_rate": learning_rate,
            "max_seq_len": max_seq_len,
            "total_train_samples": total_train_samples,
            "total_val_samples": total_val_samples,
            **{f"backbone_{k}": v for k, v in used_backbone_kwargs.items()},
            **{f"head_{k}": v for k, v in used_head_kwargs.items()},
            **extra_kwargs,
        }

        self.experiment_name = self.experiment_name(hparams_dict)
        self.save_hyperparameters(hparams_dict)

        self.tokenizer_name = tokenizer_name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.mlm_probability = mlm_probability
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len
        self.total_train_samples = total_train_samples
        self.total_val_samples = total_val_samples

        # 3) Hugging Face Tokenizer & Data Collator
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_probability
        )

        # 4) For convenience, keep references to datasets
        self.train_dataset = None
        self.val_dataset = None

        # 5) Define your loss function (if your model doesn't internally handle it)
        self.loss_fn = nn.CrossEntropyLoss()

    def experiment_name(self, hparams):
        return f"Backbone[{hparams['backbone']}]-LayerType[{hparams['backbone_feedforward_linear_layer']}]-Activation[{hparams['backbone_feedforward_activation_layer']}]"

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
        load_dataset(self.dataset_name, self.dataset_config)
        AutoTokenizer.from_pretrained(self.tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        """
        Called on each GPU/process. Create the dataset splits, tokenize, etc.
        """
        if stage == "fit" or stage is None:
            raw_datasets = load_dataset(
                self.dataset_name, self.dataset_config, cache_dir="data"
            )

            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.max_seq_len,
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
            if self.total_train_samples:
                train_ds = train_ds.select(
                    range(min(self.total_train_samples, len(train_ds)))
                )
            if self.total_val_samples:
                val_ds = val_ds.select(range(min(self.total_val_samples, len(val_ds))))

            # Convert to PyTorch
            train_ds.set_format(type="torch")
            val_ds.set_format(type="torch")

            self.train_dataset = train_ds
            self.val_dataset = val_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
