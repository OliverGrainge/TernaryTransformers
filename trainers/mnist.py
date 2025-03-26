import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional
from models.helper import create_model
from config import Config


class MNISTTrainer(pl.LightningModule):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()

        self.model = create_model(config.model_config)
        # Create a dictionary with all hyperparameters
        self.experiment_name = self.experiment_name(config)


    def experiment_name(self, config: Config):
        return f"Backbone[{config.model.backbone}]-LayerType[{config.model.feedforward_linear_layer}]-Activation[{config.model.feedforward_activation_layer}]"

    def forward(self, x):
        # Flatten the input: [B, 1, 28, 28] -> [B, 784]
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Calculate current progress (0 to 1)
        current_progress = self.current_epoch / self.max_epochs
        self.model.set_progress(current_progress)

        x, y = batch
        logits = self(x)

        # Compute main loss
        classification_loss = self.loss_fn(logits, y)

        # Combine losses


        # Log all components
        self.log("train_classification_loss", classification_loss)
        self.log("current_progress", current_progress)

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
        return torch.optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )
        dataset = datasets.MNIST(self.config.paths.data_dir, train=True, download=True, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
        )

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )
        dataset = datasets.MNIST(
            self.config.paths.data_dir, train=False, download=True, transform=transform
        )
        return DataLoader(
            dataset, batch_size=self.config.training.batch_size, num_workers=self.config.training.num_workers
        )
