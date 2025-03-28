from dataloaders.ImageClassification import ALL_IMAGE_CLASSIFICATION_DATALOADERS
import pytest
from torchvision import transforms

from config import DataConfig
@pytest.fixture
def data_config():
    return DataConfig(
        batch_size=4,
        num_workers=0,
    )

@pytest.mark.parametrize("dataloader_cls", ALL_IMAGE_CLASSIFICATION_DATALOADERS.values())
def test_cifar10_dataloader_members(data_config, dataloader_cls):
    datamodule = dataloader_cls(data_config)
    assert hasattr(datamodule, "train_dataloader")
    assert hasattr(datamodule, "val_dataloader")
    assert hasattr(datamodule, "test_dataloader")

@pytest.mark.parametrize("dataloader_cls", ALL_IMAGE_CLASSIFICATION_DATALOADERS.values())
def test_cifar10_dataloader_transform(data_config, dataloader_cls):
    datamodule = dataloader_cls(data_config)
    assert hasattr(datamodule, "transform")
    assert isinstance(datamodule.transform, transforms.Compose)


@pytest.mark.parametrize("dataloader_cls", ALL_IMAGE_CLASSIFICATION_DATALOADERS.values())
def test_cifar10_dataloader_transform(data_config, dataloader_cls):
    datamodule = dataloader_cls(data_config)
    assert hasattr(datamodule, "save_configs")