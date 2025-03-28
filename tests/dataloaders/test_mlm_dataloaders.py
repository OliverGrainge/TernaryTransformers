import pytest 
from dataloaders.MaskedLanguageModelling import ALL_MLM_DATALOADERS
from config import DataConfig

@pytest.fixture
def data_config():
    return DataConfig(
        tokenizer_name="bert-base-uncased",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        mlm_probability=0.15,
    )


@pytest.mark.parametrize("dataloader_cls", ALL_MLM_DATALOADERS.values())
def test_mlm_dataloader_members(data_config, dataloader_cls):
    datamodule = dataloader_cls(data_config)
    assert hasattr(datamodule, "setup")
    assert hasattr(datamodule, "train_dataloader")
    assert hasattr(datamodule, "val_dataloader")
    assert hasattr(datamodule, "test_dataloader")


@pytest.mark.parametrize("dataloader_cls", ALL_MLM_DATALOADERS.values())
def test_mlm_dataloader_transform(data_config, dataloader_cls):
    datamodule = dataloader_cls(data_config)
    assert hasattr(datamodule, "save_configs")


@pytest.mark.parametrize("dataloader_cls", ALL_MLM_DATALOADERS.values())
def test_mlm_dataloader_transform(data_config, dataloader_cls):
    datamodule = dataloader_cls(data_config)
    assert hasattr(datamodule, "save_configs")

