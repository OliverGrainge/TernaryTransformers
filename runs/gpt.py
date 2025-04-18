import multiprocessing as mp
import os
import sys
import torch
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_float32_matmul_precision('high')

from dataloaders.AutoLM import AutoLMDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from trainers import GPTCausalModule
from pytorch_lightning.cli import LightningCLI
from runs.cli import TernaryCLI

# On macOS you can use 'forkserver' (safer than fork) or even force 'fork'
mp.set_start_method("forkserver", force=True)


def main():
    TernaryCLI(
        model_class=GPTCausalModule,
        datamodule_class=AutoLMDataModule,
        trainer_defaults={
            "max_epochs": 20,
            "accelerator": "auto",
            "precision": "bf16-mixed",
            "devices": 1,
            "log_every_n_steps": 10,
        },
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
