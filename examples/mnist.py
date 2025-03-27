import argparse
import os
import sys

import torch
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl

from config import DataConfig, ModelConfig, TrainConfig, parse_configs
from trainers import MNISTTrainer


class MNISTModelConfig(ModelConfig):
    backbone_type: str = "mlp"
    mlp_in_dim: int = 784
    mlp_dim: int = 256
    mlp_depth: int = 3
    mlp_dropout: float = 0.0

    head_type: str = "ImageClassificationHead"
    head_in_dim: int = 256
    head_dim: int = 256
    head_output_dim: int = 10
    head_depth: int = 1
    head_dropout: float = 0.0


class MNISTTrainConfig(TrainConfig):
    project_name: str = "MNIST"
    batch_size: int = 64
    max_epochs: int = 5
    accelerator: str = "auto"
    log_steps: int = 10
    val_check_interval: int = 0.25
    num_workers: int = 8


class MNISTDataConfig(DataConfig):
    checkpoints_dir: str = os.path.join(DataConfig.checkpoints_dir, "mnist")


def main():
    model_config, train_config, data_config = parse_configs(
        MNISTModelConfig, MNISTTrainConfig, MNISTDataConfig
    )

    print(model_config)
    print(train_config)
    print(data_config)

    module = MNISTTrainer(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
    )

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        accelerator=train_config.accelerator,
        logger=pl.loggers.WandbLogger(
            project=train_config.project_name, name=module.experiment_name
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=MNISTDataConfig.checkpoints_dir,
                filename=f"{module.experiment_name}-{{epoch}}-{{val_loss:.2f}}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
        ],
        log_every_n_steps=train_config.log_steps,
        val_check_interval=train_config.val_check_interval,
    )

    trainer.fit(module)


if __name__ == "__main__":
    main()
