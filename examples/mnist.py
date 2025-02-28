import torch
import os
import sys
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.helper import create_model
from trainers import MNISTTrainer
import pytorch_lightning as pl

model = create_model(
    backbone="mlp",
    head="none",
    backbone_kwargs={"in_dim": 784, "mlp_dim": 512, "out_dim": 10},
)

trainer = pl.Trainer(
    max_epochs=10,
    logger=pl.loggers.WandbLogger(project="mnist-classification", name="my-mnist-experiment")
)
module = MNISTTrainer(model, batch_size=128)



trainer.fit(module)

