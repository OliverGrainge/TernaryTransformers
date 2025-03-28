from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from config import DataConfig, ModelConfig, TrainConfig, parse_configs
from datasets import load_dataset

# Suppose you have a create_model function that returns (model, backbone_kwargs, head_kwargs)
# from your codebase
from models.helper import create_model


class MLMTrainer(pl.LightningModule):

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config

        self.model = create_model(model_config)
        self.experiment_name = self.get_experiment_name()

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
        logits = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        if labels is None:
            return logits

        masked_positions = labels != -100

        masked_logits = logits[masked_positions]
        masked_labels = labels[masked_positions]

        loss = self.loss_fn(
            masked_logits,
            masked_labels,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.train_config.learning_rate)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
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

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        token_type_ids = batch.get("token_type_ids", None)
        loss = self.forward(input_ids, attention_mask, token_type_ids, labels)
        self.log("test_loss", loss, prog_bar=True)
        return loss

   