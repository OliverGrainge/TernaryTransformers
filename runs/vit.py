import multiprocessing as mp
import os
import sys
import torch
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_float32_matmul_precision('high')

from dataloaders.ImageClassification import ImageClassificationDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from trainers import ViTImageClassifierModule
mp.set_start_method("forkserver", force=True)
from pytorch_lightning.cli import LightningCLI
from runs.cli import TernaryCLI


def main():
    TernaryCLI(
        model_class=ViTImageClassifierModule,
        datamodule_class=ImageClassificationDataModule,
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
