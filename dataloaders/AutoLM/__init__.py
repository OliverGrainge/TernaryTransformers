from .shakespeare import ShakespeareDataModule

__all__ = ["ShakespeareDataModule"]


ALL_AUTOLM_DATALOADERS = {
    "shakespeare": ShakespeareDataModule,
}
