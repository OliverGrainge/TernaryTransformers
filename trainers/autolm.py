import pytorch_lightning as pl
import torch

from config import DataConfig, ModelConfig, TrainConfig, parse_configs
from models.helper import create_model



class AutoLMTrainer(pl.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
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
            "experiment_name": self.experiment_name,
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
        return torch.optim.Adam(
            self.model.parameters(), lr=self.train_config.learning_rate
        )

  