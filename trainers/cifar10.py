import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional
from models.helper import create_model
from config import ModelConfig, TrainConfig, DataConfig
import torch.nn as nn

class CIFAR10Trainer(pl.LightningModule):
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
        self.loss_fn = nn.CrossEntropyLoss()
        self.experiment_name = self.experiment_name(model_config)
        self.save_configs()

        
    def experiment_name(self, model_config: ModelConfig):
        return f"Backbone[{model_config.backbone_type}]-LayerType[{model_config.feedforward_linear_layer}]-Activation[{model_config.feedforward_activation_layer}]"

    def save_configs(self):
        hparams = {
            **{f"model_{k}": v for k, v in self.model_config.__dict__.items()},
            **{f"train_{k}": v for k, v in self.train_config.__dict__.items()},
            **{f"data_{k}": v for k, v in self.data_config.__dict__.items()},
            'experiment_name': self.experiment_name
        }
        self.save_hyperparameters(hparams)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        classification_loss = self.loss_fn(logits, y)
        self.log("train_classification_loss", classification_loss)
        return classification_loss
        return total_loss

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
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],  # CIFAR10 mean
                    std=[0.2470, 0.2435, 0.2616],  # CIFAR10 std
                ),
            ]
        )
        dataset = datasets.CIFAR10(
            self.data_config.data_dir, train=True, download=True, transform=transform
        )
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
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],  # CIFAR10 mean
                    std=[0.2470, 0.2435, 0.2616],  # CIFAR10 std
                ),
            ]
        )
        dataset = datasets.CIFAR10(
            self.data_config.data_dir, train=False, download=True, transform=transform
        )
        return DataLoader(
            dataset, batch_size=self.train_config.batch_size, num_workers=self.train_config.num_workers
        )
