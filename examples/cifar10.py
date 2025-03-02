import torch
import os
import sys
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers import CIFAR10Trainer
import pytorch_lightning as pl


module = CIFAR10Trainer(
    backbone="minivit",
    head="ImageClassificationHead",
    backbone_kwargs={
        "depth": 2,
        "heads": 4,
        "mlp_dim": 128 * 3,
        "dim": 128,
        "image_size": 32,  # CIFAR10 images are 32x32
        "patch_size": 4,
        "in_channels": 3,  # CIFAR10 has RGB images (3 channels)
        "dim_head": 64,
        "dropout": 0.1,
        "emb_dropout": 0,
        "embedding_norm": "LayerNorm",
        "embedding_linear": "Linear",
        "attention_linear_layer": "Linear",
        "attention_norm_layer": "LayerNorm",
        "feedforward_linear_layer": "Linear",
        "feedforward_norm_layer": "LayerNorm",
        "attention_activation_layer": "GELU",
        "feedforward_activation_layer": "GELU",
    },
    batch_size=64,
)


trainer = pl.Trainer(
    max_epochs=10,
    #accelerator="cpu",
    logger=pl.loggers.WandbLogger(
        project="cifar10-classification", name=module.experiment_name
    ),
)


trainer.fit(module)
