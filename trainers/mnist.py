import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional
from models.helper import create_model


class MNISTTrainer(pl.LightningModule):
    def __init__(
        self,
        backbone: str,
        head: str,
        backbone_kwargs: dict = {},
        head_kwargs: dict = {},
        learning_rate: float = 1e-4,
        num_workers: int = 0,
        batch_size: int = 12,
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
            'backbone': backbone,
            'head': head,
            'learning_rate': learning_rate,
            'num_workers': num_workers,
            'batch_size': batch_size,
            **{f"backbone_{k}": v for k, v in backbone_kwargs.items()},  # Flatten backbone_kwargs
            **{f"head_{k}": v for k, v in head_kwargs.items()}  # Flatten head_kwargs
        }
        self.save_hyperparameters(hparams)  # Save all hyperparameters at once
        
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # Flatten the input: [B, 1, 28, 28] -> [B, 784]
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

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
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )
        dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
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
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )
        dataset = datasets.MNIST(
            "data", train=False, download=True, transform=transform
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
