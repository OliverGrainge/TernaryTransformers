from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Any 
from pytorch_lightning import Trainer

class TernaryCLI(LightningCLI):
    
    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        print("Instantiate trainer ================================================================")
        trainer = super().instantiate_trainer(**kwargs)
        trainer.callbacks.append(ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="my-checkpoint-{epoch}-{val_loss:.2f}",
        ))
        return trainer
    