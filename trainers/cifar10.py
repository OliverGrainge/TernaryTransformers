import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional
from models.helper import create_model


class CIFAR10Trainer(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "MiniViT",
        head: str = "ImageClassificationHead",
        backbone_kwargs: dict = {"image_size": 32},
        head_kwargs: dict = {"num_classes": 10, "dim": 128, "num_layers": 1},
        learning_rate: float = 1e-4,
        num_workers: int = 0,
        batch_size: int = 12,
        decay_weight: float = 0.1,  # Weight for the decay loss
        max_epochs: int = 10,  # Required for progress calculation
    ):
        super().__init__()

        self.model, backbone_kwargs, head_kwargs = create_model(
            backbone=backbone,
            head=head,
            backbone_kwargs=backbone_kwargs,
            head_kwargs=head_kwargs,
        )
        # Create a dictionary with all hyperparameters
        hparams = {
            "backbone": backbone,
            "head": head,
            "learning_rate": learning_rate,
            "num_workers": num_workers,
            "batch_size": batch_size,
            **{
                f"backbone_{k}": v for k, v in backbone_kwargs.items()
            },  # Flatten backbone_kwargs
            **{f"head_{k}": v for k, v in head_kwargs.items()},  # Flatten head_kwargs
            "decay_weight": decay_weight,
            "max_epochs": max_epochs,
        }
        print(hparams)
        self.experiment_name = self.experiment_name(hparams)
        self.save_hyperparameters(hparams)  # Save all hyperparameters at once

        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.decay_weight = decay_weight
        self.max_epochs = max_epochs

    def experiment_name(self, hparams):
        return f"Backbone[{hparams['backbone']}]-LayerType[{hparams['backbone_feedforward_linear_layer']}]-Activation[{hparams['backbone_feedforward_activation_layer']}]"

    def forward(self, x):
        # Input is already in correct shape: [B, 3, 32, 32]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Calculate current progress (0 to 1)
        current_progress = self.current_epoch / self.max_epochs
        self.model.set_progress(current_progress)

        x, y = batch
        logits = self(x)

        # Compute main loss
        classification_loss = self.loss_fn(logits, y)

        # Compute decay loss
        decay_loss = self.model.compute_decay(reduction="mean")

        # Combine losses

        total_loss = classification_loss + self.decay_weight * decay_loss

        # Log all components
        self.log("train_loss", total_loss)
        self.log("train_classification_loss", classification_loss)
        self.log("train_decay_loss", decay_loss)
        self.log("current_progress", current_progress)

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
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

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
            "data", train=True, download=True, transform=transform
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
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
            "data", train=False, download=True, transform=transform
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
