import torch
import os
import sys
import wandb
from pytorch_lightning import Trainer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers import TinyShakespeareTrainer
import pytorch_lightning as pl

module = TinyShakespeareTrainer(
    backbone="CausalTransformer",
    head="projection",
    backbone_kwargs={
        "vocab_size": 65,  # This will be automatically adjusted by the dataset
        "max_seq_len": 64,
        "dim": 384,        # Smaller model than BERT example
        "depth": 6,
        "heads": 6,
        "mlp_dim": 1536,
        "dim_head": 64,
        "dropout": 0.1,
        "emb_dropout": 0.1,
        "attention_norm_layer": "LayerNorm",
        "feedforward_norm_layer": "LayerNorm",
        "attention_activation_layer": "GELU",
        "feedforward_activation_layer": "GELU",
        "attention_linear_layer": "Linear",
        "feedforward_linear_layer": "Linear"
    },
    head_kwargs={"in_dim": 384, "out_dim": 65},  # Will be automatically adjusted
    learning_rate=3e-4,
    batch_size=64,
    block_size=64,
    max_epochs=50
)

trainer = pl.Trainer(
    max_epochs=module.hparams.max_epochs,
    logger=pl.loggers.WandbLogger(
        project="TinyShakespeare", 
        name="shakespeare_transformer"
    ),
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints/shakespeare/",
            filename="shakespeare-{epoch}-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
    ],
)

trainer.fit(module)

# Test the model after training
model = module.eval()
sample_text = model.generate(
    start_text="O Romeo, Romeo, ",
    max_tokens=100,
    temperature=0.8
)
print("\nGenerated text:")
print(sample_text) 