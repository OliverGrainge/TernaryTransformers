from unittest.mock import patch, mock_open
import pytest
from pathlib import Path

from dataloaders.AutoLM import ALL_AUTOLM_DATALOADERS
from config import DataConfig
from transformers import AutoTokenizer

MOCK_TEXT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
"""

@pytest.fixture
def data_config():
    return DataConfig(
        tokenizer_name="gpt2",
        batch_size=4,
        num_workers=0,
        context_length=16
    )


@pytest.mark.parametrize("dataloader_cls", ALL_AUTOLM_DATALOADERS.values())
def test_shakespeare_dataloader(data_config, dataloader_cls):
    datamodule = dataloader_cls(data_config)


@pytest.mark.parametrize("dataloader_cls", ALL_AUTOLM_DATALOADERS.values())
def test_shakespeare_dataloader(data_config, dataloader_cls):
    datamodule = dataloader_cls(data_config)
    assert hasattr(datamodule, "setup")
    assert hasattr(datamodule, "train_dataloader")
    assert hasattr(datamodule, "val_dataloader")
    assert hasattr(datamodule, "test_dataloader")


@pytest.mark.parametrize("dataloader_cls", ALL_AUTOLM_DATALOADERS.values())
def test_shakespeare_dataloader(data_config, dataloader_cls):
    datamodule = dataloader_cls(data_config)
    assert hasattr(datamodule, "save_configs")






