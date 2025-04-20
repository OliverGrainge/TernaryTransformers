from dataloaders.ImageCLS import ALL_IMAGE_CLASSIFICATION_DATALOADERS
import pytest
from torchvision import transforms
from torch.utils.data import DataLoader
import torch 

from config import CIFAR10DataConfig
@pytest.fixture
def data_config():
    return CIFAR10DataConfig()
    
@pytest.mark.unit
@pytest.mark.parametrize("dataloader_tuple", ALL_IMAGE_CLASSIFICATION_DATALOADERS.values())
def test_imcl_dataloader_members(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple
    datamodule = dataloader_cls(config_cls())
    assert hasattr(datamodule, "train_dataloader")
    assert hasattr(datamodule, "val_dataloader")
    assert hasattr(datamodule, "test_dataloader")

@pytest.mark.unit
@pytest.mark.parametrize("dataloader_tuple", ALL_IMAGE_CLASSIFICATION_DATALOADERS.values())
def test_imcl_dataloader_transform(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple
    datamodule = dataloader_cls(config_cls())
    assert hasattr(datamodule, "transform")
    assert isinstance(datamodule.transform, transforms.Compose)

@pytest.mark.unit
@pytest.mark.parametrize("dataloader_tuple", ALL_IMAGE_CLASSIFICATION_DATALOADERS.values())
def test_imcl_dataloader_save_configs(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple
    datamodule = dataloader_cls(config_cls())
    assert hasattr(datamodule, "save_configs")


# --------------------------------- Integration Tests ---------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("dataloader_tuple", ALL_IMAGE_CLASSIFICATION_DATALOADERS.values())
def test_imcl_setup(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple
    datamodule = dataloader_cls(config_cls())
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    assert isinstance(train_loader, DataLoader), "Train loader should be an instance of DataLoader"
    assert isinstance(val_loader, DataLoader), "Val loader should be an instance of DataLoader"
    assert isinstance(test_loader, DataLoader), "Test loader should be an instance of DataLoader"



@pytest.mark.integration
@pytest.mark.parametrize("dataloader_tuple", ALL_IMAGE_CLASSIFICATION_DATALOADERS.values())
def test_imcl_train_dataloader(dataloader_tuple):
    dataloader_cls, config_cls = dataloader_tuple
    datamodule = dataloader_cls(config_cls())
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        break

    assert len(batch) == 2, "Batch should have 2 elements"
    assert isinstance(batch[0], torch.Tensor), "First element of batch should be a tensor"
    assert isinstance(batch[1], torch.Tensor), "Second element of batch should be a tensor"
    assert batch[1].dtype == torch.long, "Second element of batch should be a long tensor"
    assert batch[0].shape[0] == batch[1].shape[0], "First and second element of batch should have the same batch size"


