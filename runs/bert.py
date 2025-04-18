import multiprocessing as mp
import os
import sys

import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_float32_matmul_precision("high")

from dataloaders.MaskedLanguageModelling import MLMDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from trainers import BertModule

from runs.cli import TernaryCLI

mp.set_start_method("forkserver", force=True)


def main():
    cli = TernaryCLI(
        model_class=BertModule,
        datamodule_class=MLMDataModule,
        trainer_defaults={
            "max_epochs": 20,
            "accelerator": "auto",
            "precision": "16-mixed",
            "devices": 1,
            "log_every_n_steps": 10,
        },
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
