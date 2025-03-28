import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from config import (MLPModelConfig, MNISTDataConfig, MNISTTrainConfig,
                    parse_configs)
from dataloaders.ImageClassification import MNISTDataModule
from trainers import ImageClassificationTrainer


def main():
    model_config, train_config, data_config = parse_configs(
        MLPModelConfig, MNISTTrainConfig, MNISTDataConfig
    )

    print(model_config)

    module = ImageClassificationTrainer(
        model_config=model_config,
        train_config=train_config,
    )

    data_module = MNISTDataModule(
        data_config=data_config,
    )

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        accelerator=train_config.accelerator,
        log_every_n_steps=train_config.log_every_n_steps,
        val_check_interval=train_config.val_check_interval,
        precision=train_config.precision,
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
    )

    trainer.fit(module, data_module)


if __name__ == "__main__":
    main()
