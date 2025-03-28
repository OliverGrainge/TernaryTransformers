from .shakespeare import ShakespeareDataModule
from config import ShakespeareDataConfig

__all__ = ["ShakespeareDataModule"]


ALL_AUTOLM_DATALOADERS = {
    "shakespeare": (ShakespeareDataModule, ShakespeareDataConfig),
}
