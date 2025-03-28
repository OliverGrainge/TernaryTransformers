import argparse
import os
import sys

import torch
from pytorch_lightning import Trainer

import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from config import (NanoGPTModelConfig, ShakespeareDataConfig,
                    ShakespeareTrainConfig, parse_configs)
from dataloaders.AutoLM import ShakespeareDataModule
from trainers import AutoLMTrainer


def main():
    model_config, train_config, data_config = parse_configs(
        NanoGPTModelConfig, ShakespeareTrainConfig, ShakespeareDataConfig
    )

    module = AutoLMTrainer(
        model_config=model_config,
        train_config=train_config,
    )

    data_module = ShakespeareDataModule(
        data_config=data_config,
    )

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        accelerator=train_config.accelerator,
        precision=train_config.precision,
        gradient_clip_val=train_config.gradient_clip_val,
        log_every_n_steps=train_config.log_every_n_steps,
        val_check_interval=train_config.val_check_interval,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        logger=pl.loggers.WandbLogger(
            project=train_config.project_name, name=module.experiment_name
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=data_config.checkpoints_dir,
                filename="shakespeare-{epoch}-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
        ],
    )

    trainer.fit(module, data_module)


if __name__ == "__main__":
    main()
