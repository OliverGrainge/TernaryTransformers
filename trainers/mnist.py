import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional

class MNISTTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-4, num_workers: int=0, batch_size: int=12):
        super().__init__()
        self.model = model
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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
