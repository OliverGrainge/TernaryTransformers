import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import requests
from typing import Optional
from models.helper import create_model
from config import ModelConfig, TrainConfig, DataConfig, parse_configs
import os

class CharacterDataset(Dataset):
    def __init__(self, data_config: DataConfig, block_size=64, split="train"):
        # Ensure data directory exists
        data_dir = data_config.data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Download tiny shakespeare if not already present
        shakespeare_path = os.path.join(data_dir, "shakespeare.txt")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            with open(shakespeare_path, "r") as f:
                text = f.read()
        except FileNotFoundError:
            text = requests.get(url).text
            with open(shakespeare_path, "w") as f:
                f.write(text)

        # Create vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Train/val split
        n = len(text)
        train_data = text[: int(n * 0.9)]
        val_data = text[int(n * 0.9) :]

        # Select appropriate split
        data = train_data if split == "train" else val_data

        # Encode the data
        self.data = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


"""
        backbone: str = "CausalTransformer",
        head: str = "projection",
        backbone_kwargs: dict = {
            "vocab_size": 50257,  # Default GPT-2 vocabulary size
            "max_seq_len": 1024,
            "dim": 768,
            "depth": 12,
            "heads": 12,
            "mlp_dim": 3072,
            "dim_head": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "attention_norm_layer": "LayerNorm",
            "feedforward_norm_layer": "LayerNorm",
            "attention_activation_layer": "GELU",
            "feedforward_activation_layer": "GELU",
            "attention_linear_layer": "Linear",
            "feedforward_linear_layer": "Linear",
        },
        head_kwargs: dict = {"in_dim": 768, "out_dim": 50257},
        learning_rate: float = 1e-3,
        num_workers: int = 0,
        batch_size: int = 32,
        block_size: int = 64,
        max_epochs: int = 10,
"""
class TinyShakespeareTrainer(pl.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        data_config: DataConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.model = create_model(model_config)

        self.loss_fn = torch.nn.CrossEntropyLoss()
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
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.train_config.learning_rate)

    def train_dataloader(self):
        dataset = CharacterDataset(self.data_config, block_size=self.train_config.block_size, split="train")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        dataset = CharacterDataset(self.data_config, block_size=self.train_config.block_size, split="val")
        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
        )

    def generate(
        self
    ):
        self.eval()
        with torch.no_grad():
            dataset = CharacterDataset(self.data_config, block_size=self.train_config.block_size)
            x = torch.tensor([dataset.stoi[c] for c in self.train_config.start_text], dtype=torch.long)

            generated = list(self.train_config.start_text)
            for _ in range(self.train_config.max_tokens):
                # Take last block_size tokens
                x_cond = x[-self.train_config.block_size :]
                logits = self(x_cond.unsqueeze(0))[0]
                logits = logits[-1] / self.train_config.temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(dataset.itos[next_token.item()])
                x = torch.cat([x, next_token])

        return "".join(generated)
