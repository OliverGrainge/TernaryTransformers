import torch
import os
import sys
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers import MNISTTrainer
import pytorch_lightning as pl


module = MNISTTrainer(
    backbone="mlp",
    head="none",
    backbone_kwargs={"in_dim": 784, "mlp_dim": 512, "out_dim": 10, "linear_layer": "BitLinear", "activation_layer": "RELU", "num_layers": 3, "norm_layer": "identity"},
    batch_size=128,
)


trainer = pl.Trainer(
    max_epochs=10,
    accelerator="cpu",
    logger=pl.loggers.WandbLogger(
        project="mnist-classification", name=module.experiment_name
    ),
)


trainer.fit(module)
