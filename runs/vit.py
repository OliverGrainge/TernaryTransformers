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


def main():

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch}-{val_loss:.2f}",
        every_n_epochs=1,
        dirpath="./checkpoints/vit/",
    )

    LightningCLI(
        model_class=ViTImageClassifierModule,
        datamodule_class=ImageClassificationDataModule,
        trainer_defaults={
            "max_epochs": 20,
            "accelerator": "auto",
            "precision": "16-mixed",
            "devices": 1,
            "log_every_n_steps": 10,
            "callbacks": [checkpoint_callback],
        },
        
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
