import torch
import os
import sys
import wandb
from datasets import load_dataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers import WikiText2BertMLMTrainer
import pytorch_lightning as pl

# Load the dataset first
try:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
except Exception as e:
    print("Error loading dataset. Trying with force_download=True...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", force_download=True)

module = WikiText2BertMLMTrainer(
    backbone="Bert",
    head="MLMHead",
    backbone_kwargs={
        'vocab_size': 30522,
        'max_seq_len': 512,
        'dim': 256,
        'depth': 6,
        'heads': 8,
        'mlp_dim': 1024,
        'dim_head': 32,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'num_segments': 2,
        'attention_norm_layer': 'LayerNorm',
        'feedforward_norm_layer': 'LayerNorm',
        'attention_activation_layer': 'GELU',
        'feedforward_activation_layer': 'GELU',
        'attention_linear_layer': 'Linear',
        'feedforward_linear_layer': 'Linear',
    },
    head_kwargs={"dim": 256, "vocab_size": 30522},
    batch_size=128,
    dataset=dataset,
)


trainer = pl.Trainer(
    max_epochs=10,
    logger=pl.loggers.WandbLogger(
        project="WikiText2-MLM", name=module.experiment_name
    ),
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints/bert/",
            filename=f"{module.experiment_name}-{{epoch}}-{{val_loss:.2f}}",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
    ],
)


trainer.fit(module)
