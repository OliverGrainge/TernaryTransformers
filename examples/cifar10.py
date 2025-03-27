import argparse
import os
import sys

import torch
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl

from config import DataConfig, ModelConfig, TrainConfig, parse_configs
from trainers import CIFAR10Trainer


class CIFAR10ModelConfig(ModelConfig):
    backbone_type = "ViT"
    transformer_heads = 4
    transformer_dim = 128
    transformer_ffn_dim = 384
    transformer_depth = 6
    transformer_dropout = 0.1
    transformer_dim_head = transformer_dim // transformer_heads
    image_in_channels = 1
    image_size = 32
    image_patch_size = 4
    embedding_dropout = 0.0

    embedding_norm_layer = "LayerNorm"
    embedding_linear_layer = "Linear"
    attention_linear_layer = "Linear"
    attention_norm_layer = "LayerNorm"
    feedforward_linear_layer = "Linear"
    feedforward_norm_layer = "LayerNorm"

    head_type: str = "ImageClassificationHead"
    head_in_dim = 128
    head_out_dim = 10
    head_dim = 128
    head_linear_layer = "Linear"
    head_depth = 1
    head_dropout = 0.0


class CIFAR10TrainConfig(TrainConfig):
    project_name = "CIFAR10"
    batch_size = 128
    max_epochs = 100
    learning_rate = 0.001


class CIFAR10DataConfig(DataConfig):
    checkpoints_dir: str = os.path.join(DataConfig.checkpoints_dir, "cifar10")


def main():
    model_config, train_config, data_config = parse_configs(
        CIFAR10ModelConfig, CIFAR10TrainConfig, CIFAR10DataConfig
    )

    print(model_config)

    module = CIFAR10Trainer(
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
                dirpath=CIFAR10DataConfig.checkpoints_dir,
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
