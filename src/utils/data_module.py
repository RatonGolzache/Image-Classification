import os
import pickle
from typing import Any, List, Tuple, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets

def _pil_from_array(array: np.ndarray):
    """Safe PIL conversion for multiprocessing."""
    from PIL import Image
    return Image.fromarray(array)

class CIFAR10Pickle(Dataset):
    """
    Loads CIFAR-10 from the official pickle batches (data_batch_1-5, test_batch).
    Expects: cifar-10-batches-py/ folder with data_batch_*.bin and batches.meta.
    """
    def __init__(self, root: str, train: bool = True, transform=None) -> None:
        self.root = os.path.join(root, "cifar-10-batches-py")
        self.transform = transform
        self.train = train

        if train:
            batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            batch_files = ["test_batch"]

        self.data = []
        self.labels = []

        for batch_file in batch_files:
            batch_path = os.path.join(self.root, batch_file)
            with open(batch_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            
            self.data.append(batch[b"data"])
            self.labels.extend(batch[b"labels"])

        # Ensure data is uint8 NHWC (H,W,C) format
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.data[idx]
        label = self.labels[idx]

        #Convert numpy array to PIL Image FIRST
        if self.transform:
            # Convert HWC uint8 numpy -> PIL Image -> apply transforms
            image = _pil_from_array(image)
            image = self.transform(image)
        else:
            # If no transform, convert to tensor manually
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label

class FashionMNISTDataset(Dataset):
    """Fashion-MNIST dataset wrapper for consistent interface."""
    def __init__(self, root: str, train: bool = True, transform=None):
        self.dataset = datasets.FashionMNIST(
            root=root, train=train, download=True, transform=transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]

class DataModule:
    """Unified data module for CIFAR-10 and Fashion-MNIST with enhanced augmentations."""
    """Augmentation done according to best practices from: https://medium.com/@BurtMcGurt/a-practical-guide-to-data-augmentation-in-pytorch-with-examples-and-visualizations-761ad5c2a903"""
    
    # CIFAR-10 normalization stats from the article
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    
    # Fashion-MNIST normalization stats
    FASHIONMNIST_MEAN = (0.2860,)
    FASHIONMNIST_STD = (0.3205,)

    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        batch_size: int = 128,
        num_workers: int = 4,
        use_augmentation: bool = True,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation

        assert self.dataset_name in ["cifar10", "fashionmnist"], \
            "dataset_name must be 'cifar10' or 'fashionmnist'"

    def _cifar10_train_transform(self) -> T.Compose:
        if self.use_augmentation:
            return T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4),
                T.ColorJitter(
                    brightness=0.3, 
                    contrast=0.3, 
                    saturation=0.3
                ),
                T.ToTensor(),
                T.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD),
            ])
        else:
            return T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD),
            ])

    def _cifar10_test_transform(self) -> T.Compose:
        return T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD),
        ])

    def _fashionmnist_train_transform(self) -> T.Compose:
        if self.use_augmentation:
            return T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(28, padding=4),
                T.ColorJitter(
                    brightness=0.3, 
                    contrast=0.3, 
                    saturation=0.3
                ),
                T.ToTensor(),
                T.Normalize(self.FASHIONMNIST_MEAN, self.FASHIONMNIST_STD),
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(self.FASHIONMNIST_MEAN, self.FASHIONMNIST_STD),
            ])

    def _fashionmnist_test_transform(self) -> T.Compose:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(self.FASHIONMNIST_MEAN, self.FASHIONMNIST_STD),
        ])

    def get_dataloaders(self):
        if self.dataset_name == "cifar10":
            train_dataset = CIFAR10Pickle(
                self.data_root, train=True, transform=self._cifar10_train_transform()
            )
            test_dataset = CIFAR10Pickle(
                self.data_root, train=False, transform=self._cifar10_test_transform()
            )
        else:  # fashionmnist - loads directly from PyTorch
            train_dataset = FashionMNISTDataset(
                self.data_root, train=True, transform=self._fashionmnist_train_transform()
            )
            test_dataset = FashionMNISTDataset(
                self.data_root, train=False, transform=self._fashionmnist_test_transform()
            )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=True
        )

        return train_loader, test_loader
