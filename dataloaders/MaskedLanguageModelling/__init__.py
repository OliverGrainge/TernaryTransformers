from .wikitext import WikiTextMLMDataModule
from config import WikiTextMLMDataConfig

__all__ = ["WikiTextMLMDataModule"]


ALL_MLM_DATALOADERS = {
    "wikitext": (WikiTextMLMDataModule, WikiTextMLMDataConfig),
}
