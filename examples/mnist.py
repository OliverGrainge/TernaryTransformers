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
    backbone_kwargs={"in_dim": 784, "mlp_dim": 512, "out_dim": 10, "linear_layer": "trilinear", "activation_layer": "RELU", "num_layers": 6, "norm_layer": "layernorm"},
    batch_size=64,
)


trainer = pl.Trainer(
    max_epochs=10,
    accelerator="cpu",
    logger=pl.loggers.WandbLogger(
        project="mnist-classification", name=module.experiment_name
    ),
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints/mnist/",
            filename=f"{module.experiment_name}-{{epoch}}-{{val_loss:.2f}}",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
    ],
    log_every_n_steps=5,
    val_check_interval=0.25
)


trainer.fit(module)
