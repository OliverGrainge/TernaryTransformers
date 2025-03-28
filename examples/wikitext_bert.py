import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from config import (BertMLMTrainConfig, MiniBertModelConfig,
                    WikiTextMLMDataConfig, parse_configs)
from dataloaders.MaskedLanguageModelling import WikiTextMLMDataModule
from trainers import MLMTrainer


def main():
    model_config, train_config, data_config = parse_configs(
        MiniBertModelConfig,
        BertMLMTrainConfig,
        WikiTextMLMDataConfig,
    )

    # Load the dataset first

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    module = MLMTrainer(
        model_config=model_config,
        train_config=train_config,
    )

    data_module = WikiTextMLMDataModule(
        data_config=data_config,
    )

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        accelerator=train_config.accelerator,
        precision=train_config.precision,
        log_every_n_steps=train_config.log_every_n_steps,
        val_check_interval=train_config.val_check_interval,
        logger=pl.loggers.WandbLogger(
            project=train_config.project_name, name=module.experiment_name
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=data_config.checkpoints_dir,
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
