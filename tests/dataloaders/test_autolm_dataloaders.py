from unittest.mock import patch, mock_open
import pytest
from pathlib import Path
import torch 

from dataloaders.AutoLM import ALL_AUTOLM_DATALOADERS
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

@pytest.mark.unit
@pytest.mark.parametrize("dataloader_tuple", ALL_AUTOLM_DATALOADERS.values())
def test_alm_dataloade_members(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple
    datamodule = dataloader_cls(config_cls())
    assert hasattr(datamodule, "setup")
    assert hasattr(datamodule, "train_dataloader")
    assert hasattr(datamodule, "val_dataloader")
    assert hasattr(datamodule, "test_dataloader")

@pytest.mark.unit
@pytest.mark.parametrize("dataloader_tuple", ALL_AUTOLM_DATALOADERS.values())
def test_alm_dataloader_configs(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple
    datamodule = dataloader_cls(config_cls())
    assert hasattr(datamodule, "save_configs")

@pytest.mark.unit
@pytest.mark.parametrize("dataloader_tuple", ALL_AUTOLM_DATALOADERS.values())
def test_alm_vocab_size(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple
    datamodule = dataloader_cls(config_cls())
    datamodule = dataloader_cls(config_cls())
    assert hasattr(datamodule, "vocab_size"), "Vocab size should be defined"
    assert isinstance(datamodule.vocab_size, int), "Vocab size should be an integer"
    assert datamodule.vocab_size > 0, "Vocab size should be greater than 0"

# --------------------------------- Integration Tests ---------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("dataloader_tuple", ALL_AUTOLM_DATALOADERS.values())
def test_alm_setup(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple
    config = config_cls(
        tokenizer_name="gpt2",
        batch_size=4,
        num_workers=0,
        context_length=16
    )
    datamodule = dataloader_cls(config)
    datamodule.setup("fit")
    assert hasattr(datamodule, "train_dataset"), "Train dataloader should be defined"
    assert hasattr(datamodule, "val_dataset"), "Val dataloader should be defined"
    assert isinstance(datamodule.train_dataset, Dataset), "Train dataset should be an instance of CharacterDataset"
    assert isinstance(datamodule.val_dataset, Dataset), "Val dataset should be an instance of CharacterDataset"
    datamodule.setup("test")
    assert hasattr(datamodule, "test_dataset"), "Test dataloader should not be defined"
    

@pytest.mark.integration
@pytest.mark.parametrize("dataloader_tuple", ALL_AUTOLM_DATALOADERS.values())
def test_alm_dataloader_instantiation(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple

    datamodule = dataloader_cls(config_cls())
    datamodule.setup("fit")
    datamodule.setup("test")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    assert isinstance(train_loader, DataLoader), "Train loader should be an instance of DataLoader"
    assert isinstance(val_loader, DataLoader), "Val loader should be an instance of DataLoader"
    assert isinstance(test_loader, DataLoader), "Test loader should be an instance of DataLoader"

@pytest.mark.integration
@pytest.mark.parametrize("dataloader_tuple", ALL_AUTOLM_DATALOADERS.values())
def test_alm_train_dataloader(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple
    datamodule = dataloader_cls(config_cls())
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        break 

    assert len(batch) == 2, "Batch should have 2 elements"
    assert isinstance(batch[0], torch.Tensor), "First element of batch should be a tensor"
    assert isinstance(batch[1], torch.Tensor), "Second element of batch should be a tensor"
    assert batch[0].shape == batch[1].shape, "First and second element of batch should have the same shape"
    assert batch[0].dtype == torch.long, "First element of batch should be a long tensor"
    assert batch[1].dtype == torch.long, "Second element of batch should be a long tensor"
    




