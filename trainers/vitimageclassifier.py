from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.transformers import ViT
from torch.optim import Optimizer
from typing import Any


class ViTImageClassifierModule(pl.LightningModule):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        dim_head: int = 64,
        channels: int = 3,
        dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        attention_norm_layer: str = "LayerNorm",
        attention_activation_layer: str = "GELU",
        attention_linear_layer: str = "Linear",
        feedforward_norm_layer: str = "LayerNorm",
        feedforward_activation_layer: str = "GELU",
        feedforward_linear_layer: str = "Linear",
        ffn_dim: int = None,
        num_classes: int = 10,
    ):
        super().__init__()

        self.model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            channels=channels,
            dropout=dropout,
            embedding_dropout=embedding_dropout,
            attention_norm_layer=attention_norm_layer,
            attention_activation_layer=attention_activation_layer,
            attention_linear_layer=attention_linear_layer,
            feedforward_norm_layer=feedforward_norm_layer,
            feedforward_activation_layer=feedforward_activation_layer,
            feedforward_linear_layer=feedforward_linear_layer,
            ffn_dim=ffn_dim,
        )

        self.logits = nn.Linear(dim, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, x):
        x = self.model(x)
        logits = self.logits(x[:, 0])
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        classification_loss = self.loss_fn(logits, y)
        self.log("train_loss", classification_loss)
        return classification_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.long())
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.long())
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc)
        self.log("test_loss", loss)
    