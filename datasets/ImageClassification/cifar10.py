from typing import Optional
from torchvision import datasets, transforms
from .base import BaseDataModule

class CIFAR10DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        transform = None,
    ):
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],  # CIFAR10 RGB means
                    std=[0.2470, 0.2435, 0.2616]    # CIFAR10 RGB stds
                )
            ])
        super().__init__(data_dir, batch_size, num_workers, pin_memory, transform)
        
        

    def prepare_data(self):
        # Download CIFAR10 data if not already downloaded
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.CIFAR10(
                self.data_dir,
                train=True,
                transform=self.transform
            )
            
            self.val_dataset = datasets.CIFAR10(
                self.data_dir,
                train=False,
                transform=self.transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                self.data_dir,
                train=False,
                transform=self.transform
            )
