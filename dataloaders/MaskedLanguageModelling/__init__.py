from .wikitext import WikiTextMLMDataModule

__all__ = ["WikiTextMLMDataModule"]


ALL_MLM_DATALOADERS = {
    "wikitext": WikiTextMLMDataModule,
}
