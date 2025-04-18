import os
import sys
import multiprocessing as mp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloaders.AutoLM.shakespeare import ShakespeareDataModule
from trainers import GPTCausalModule


# On macOS you can use 'forkserver' (safer than fork) or even force 'fork'
mp.set_start_method('forkserver', force=True)

def main():
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch}-{val_loss:.2f}",
        every_n_epochs=1,
        dirpath="./checkpoints/cifar10_classification/"
    )

    LightningCLI(
        model_class=GPTCausalModule,
        datamodule_class=ShakespeareDataModule,
        trainer_defaults={
            "max_epochs": 20,
            "accelerator": "auto",
            "precision": "bf16-mixed",
            "devices": 1,
            "log_every_n_steps": 10,
            "callbacks": [checkpoint_callback]
        },
        save_config_callback=None
    )

if __name__ == "__main__":
    main()
