from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import Config
from models.helper import create_model


class ImageClassificationTrainer(pl.LightningModule):
    def __init__(
        self,
        model_config: Config,
        train_config: Config,
    ):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config

        self.model = create_model(model_config)
        self.experiment_name = self.experiment_name(model_config)
        self.loss_fn = nn.CrossEntropyLoss()

        self.save_configs()

    def save_configs(self):
        hparams = {
            **{f"model_{k}": v for k, v in self.model_config.__dict__.items()},
            **{f"train_{k}": v for k, v in self.train_config.__dict__.items()},
            "experiment_name": self.experiment_name,
        }
        self.save_hyperparameters(hparams)

    def experiment_name(self, config: Config):
        try: 
            return f"Backbone[{config.backbone_type}]-LayerType[{config.feedforward_linear_layer}]-Activation[{config.feedforward_activation_layer}]"
        except: 
            return f"Backbone[{config.backbone_type}]-LayerType[{config.mlp_linear_layer}]-Activation[{config.mlp_activation_layer}]"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        classification_loss = self.loss_fn(logits, y)
        self.log("train_classification_loss", classification_loss)
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
        print("==============", logits.shape, y.shape)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.train_config.learning_rate
        )
