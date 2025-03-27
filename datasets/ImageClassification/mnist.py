from torchvision import datasets, transforms
from .base import BaseDataModule
from typing import Optional

class MNISTDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        transform = None,
    ):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,),  # MNIST mean and std
                transforms.Lambda(lambda x: x.view(-1))  # Flatten to [batch_size, 784]
            ])
        super().__init__(data_dir, batch_size, num_workers, pin_memory, transform)
        
    def prepare_data(self):
        # Download MNIST data if not already downloaded
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.MNIST(
                self.data_dir,
                train=True,
                transform=self.transform
            )
            
            # Using validation split from training data
            self.val_dataset = datasets.MNIST(
                self.data_dir,
                train=False,
                transform=self.transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = datasets.MNIST(
                self.data_dir,
                train=False,
                transform=self.transform
            )
