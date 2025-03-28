import pytest 
from dataloaders.MaskedLanguageModelling import ALL_MLM_DATALOADERS
from config import DataConfig
from torch.utils.data import DataLoader

# --------------------------------- Unit Tests ---------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("dataloader_info", ALL_MLM_DATALOADERS.values())
def test_mlm_dataloader_members(dataloader_info):
    
    dataloader_cls, config_cls = dataloader_info
    config = config_cls(tokenizer_name="bert-base-uncased")
    datamodule = dataloader_cls(config)
    assert hasattr(datamodule, "setup")
    assert hasattr(datamodule, "train_dataloader")
    assert hasattr(datamodule, "val_dataloader")
    assert hasattr(datamodule, "test_dataloader")

@pytest.mark.unit
@pytest.mark.parametrize("dataloader_info", ALL_MLM_DATALOADERS.values())
def test_mlm_dataloader_save_configs(dataloader_info):
    dataloader_cls, config_cls = dataloader_info
    config = config_cls(tokenizer_name="bert-base-uncased")
    datamodule = dataloader_cls(config)
    assert hasattr(datamodule, "save_configs")

# --------------------------------- Integration Tests ---------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("dataloader_info", ALL_MLM_DATALOADERS.values())
def test_mlm_dataloader_setup(dataloader_info):
    dataloader_cls, config_cls = dataloader_info
    config = config_cls(tokenizer_name="bert-base-uncased")
    datamodule = dataloader_cls(config)
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    assert isinstance(train_loader, DataLoader), "Train loader should be an instance of DataLoader"
    assert isinstance(val_loader, DataLoader), "Val loader should be an instance of DataLoader"
    assert isinstance(test_loader, DataLoader), "Test loader should be an instance of DataLoader"

@pytest.mark.integration
@pytest.mark.parametrize("dataloader_info", ALL_MLM_DATALOADERS.values())
def test_mlm_dataloader_train_dataloader(dataloader_info):
    dataloader_cls, config_cls = dataloader_info
    config = config_cls(tokenizer_name="bert-base-uncased")
    datamodule = dataloader_cls(config)
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        break

    assert batch["input_ids"].shape[0] == batch["labels"].shape[0], "Input ids and labels should have the same batch size"
    assert batch["input_ids"].shape[0] == batch["attention_mask"].shape[0], "Input ids and attention mask should have the same batch size"
    assert batch["input_ids"].shape[0] == batch["token_type_ids"].shape[0], "Input ids and token type ids should have the same batch size"
    
    
