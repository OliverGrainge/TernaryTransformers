import pytorch_lightning as pl
import torch

from models.transformers import GPT



class GPTCausalModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        dim_head: int = 64,
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

        # Save optimization related parameters
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_t_max = lr_t_max
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.vocab_size = vocab_size

        self.model = GPT(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
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

        self.loss_fn = torch.nn.CrossEntropyLoss()

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)
        perplexity = torch.exp(loss)
        self.log("test_perplexity", perplexity)
        return {"test_loss": loss, "test_perplexity": perplexity}

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

  