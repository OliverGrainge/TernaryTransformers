import pytorch_lightning as pl
import torch

from models.transformers import GPT



class GPTCausalModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int = 50257,
        context_length: int = 1024,
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
    ):
        super().__init__()

        self.model = GPT(
            vocab_size=vocab_size,
            context_length=context_length,
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

        self.save_hyperparameters()
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

  