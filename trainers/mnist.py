import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional
import torch.nn as nn 
from models.helper import create_model
from config import ModelConfig, TrainConfig, DataConfig


class MNISTTrainer(pl.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig, 
        train_config: TrainConfig, 
        data_config: DataConfig
    ):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config

        self.model = create_model(model_config)
        self.experiment_name = self.experiment_name(model_config)
        self.loss_fn = nn.CrossEntropyLoss()

        self.save_configs()

    def save_configs(self):
        hparams = {
            **{f"model_{k}": v for k, v in self.model_config.__dict__.items()},
            **{f"train_{k}": v for k, v in self.train_config.__dict__.items()},
            **{f"data_{k}": v for k, v in self.data_config.__dict__.items()},
            'experiment_name': self.experiment_name
        }
        self.save_hyperparameters(hparams)

    def experiment_name(self, config: ModelConfig):
        return f"Backbone[{config.backbone_type}]-LayerType[{config.mlp_linear_layer}]-Activation[{config.mlp_activation_layer}]"

    def forward(self, x):
        # Flatten the input: [B, 1, 28, 28] -> [B, 784]
        x = x.view(x.size(0), -1)
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
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.train_config.learning_rate)

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )
        dataset = datasets.MNIST(self.data_config.data_dir, train=True, download=True, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
        )

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )
        dataset = datasets.MNIST(
            self.data_config.data_dir, train=False, download=True, transform=transform
        )
        return DataLoader(
            dataset, batch_size=self.train_config.batch_size, num_workers=self.train_config.num_workers
        )
