from typing import Any

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI


class TernaryCLI(LightningCLI):
    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        extra_callbacks = [
            self._get(self.config_init, c)
            for c in self._parser(self.subcommand).callback_keys
        ]
        extra_callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="{epoch}-{val_loss:.2f}",
                dirpath=f"checkpoints/{self.config.fit.data.dataset_name}",
            )
        )
        trainer_config = {
            **self._get(self.config_init, "trainer", default={}),
            **kwargs,
        }
        return self._instantiate_trainer(trainer_config, extra_callbacks)
