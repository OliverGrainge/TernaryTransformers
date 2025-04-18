from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn


# Suppose you have a create_model function that returns (model, backbone_kwargs, head_kwargs)
# from your codebase
from models.transformers import Bert


class BertModule(pl.LightningModule):

    def __init__(
        self,
        vocab_size: int = 30522,
        context_length: int = 512,
        num_segments: int = 2,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        dim_head: int = None,
        ffn_dim: int = None,
        dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        attention_norm_layer: str = "LayerNorm",
        attention_activation_layer: str = "GELU",
        attention_linear_layer: str = "Linear",
        feedforward_norm_layer: str = "LayerNorm",
        feedforward_activation_layer: str = "GELU",
        feedforward_linear_layer: str = "Linear",

        optimizer: str = "AdamW",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        lr_scheduler: str = None,
        lr_t_max: int = None,  # Number of epochs/steps for cosine schedule
        lr_step_size: int = None,  # For StepLR (Number of epochs/steps between each lr update)
        lr_gamma: float = None,  # For StepLR (Multiplicative factor of learning rate decay)
        scheduler_interval: str = "step",  # "epoch" or "step"
        scheduler_frequency: int = 1,  # How often to step the scheduler
    ):
        super().__init__()

        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_t_max = lr_t_max
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

        self.model = Bert(
            vocab_size=vocab_size,
            context_length=context_length,
            num_segments=num_segments,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ffn_dim=ffn_dim,
            dropout=dropout,
            embedding_dropout=embedding_dropout,
            attention_norm_layer=attention_norm_layer,
            attention_activation_layer=attention_activation_layer,
            attention_linear_layer=attention_linear_layer,
            feedforward_norm_layer=feedforward_norm_layer,
            feedforward_activation_layer=feedforward_activation_layer,
            feedforward_linear_layer=feedforward_linear_layer,
        )

        self.logits = nn.Linear(dim, vocab_size)

        self.loss_fn = nn.CrossEntropyLoss()


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        logits = self.logits(x)

        if labels is None:
            return logits

        masked_positions = labels != -100

        masked_logits = logits[masked_positions]
        masked_labels = labels[masked_positions]

        loss = self.loss_fn(
            masked_logits,
            masked_labels,
        )

        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        token_type_ids = batch.get("token_type_ids", None)
        loss = self.forward(input_ids, attention_mask, token_type_ids, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        token_type_ids = batch.get("token_type_ids", None)
        loss = self.forward(input_ids, attention_mask, token_type_ids, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        token_type_ids = batch.get("token_type_ids", None)
        loss = self.forward(input_ids, attention_mask, token_type_ids, labels)
        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.optimizer_name)
        optimizer = optimizer_class(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        if self.lr_scheduler:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.lr_scheduler)
            # Different schedulers require different arguments
            if self.lr_scheduler == "StepLR":
                scheduler = scheduler_class(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
            elif self.lr_scheduler == "CosineAnnealingLR":
                scheduler = scheduler_class(
                    optimizer,
                    T_max=self.lr_t_max,  # Number of epochs/steps for one cosine cycle
                    eta_min=0,  # Minimum learning rate, defaults to 0
                    last_epoch=-1,  # The index of last epoch, defaults to -1
                    verbose=False  # If True, prints a message when lr is updated
                )
            else:
                raise ValueError(f"Scheduler {self.lr_scheduler} not implemented")
                
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.scheduler_interval,  # "step" or "epoch"
                    "frequency": self.scheduler_frequency,  # How often to step the scheduler
                    "monitor": "val_loss"  # For ReduceLROnPlateau
                }
            }
        
        return optimizer

   