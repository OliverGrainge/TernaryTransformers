import torch
import os
import sys
import wandb
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers import MNISTTrainer
from config import Config, ModelConfig, TrainingConfig, DataConfig 
import pytorch_lightning as pl


def main():
    model_config = ModelConfig.from_parser()
    training_config = TrainingConfig.from_parser()
    data_config = DataConfig.from_parser()

    module = MNISTTrainer(
        model_config=model_config, 
        training_config=training_config, 
        data_config=data_config,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=pl.loggers.WandbLogger(
            project=args.project_name, name=module.experiment_name
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints/mnist/",
                filename=f"{module.experiment_name}-{{epoch}}-{{val_loss:.2f}}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
        ],
        log_every_n_steps=args.log_steps,
        val_check_interval=args.val_check_interval,
    )

    trainer.fit(module)


if __name__ == "__main__":
    main()
