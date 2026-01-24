import os
import pickle
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

def _pil_from_array(array: np.ndarray):
    from PIL import Image
    return Image.fromarray(array)

class CIFAR10Pickle(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, use_pil_pipeline=False):
        self.root = os.path.join(root, "cifar-10-batches-py")
        self.transform = transform
        self.use_pil_pipeline = use_pil_pipeline
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

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.data[idx]
        label = self.labels[idx]

        if self.use_pil_pipeline:
            image = _pil_from_array(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.transform:
            image = self.transform(image)

        return image, label

class FashionMNISTDataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None):
        self.dataset = datasets.FashionMNIST(
            root=root, train=train, download=True, transform=transform
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]

class DataModule:
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    FASHIONMNIST_MEAN = (0.2860,)
    FASHIONMNIST_STD = (0.3205,)

    def __init__(self, dataset_name: str, data_root: str, batch_size=128, num_workers=0, use_augmentation=True):
        self.dataset_name = dataset_name.lower()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation
        assert self.dataset_name in ["cifar10", "fashionmnist"]

    def _get_normalize_transform(self):
        if self.dataset_name == "cifar10":
            return T.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD)
        else:
            return T.Normalize(self.FASHIONMNIST_MEAN, self.FASHIONMNIST_STD)

    def _cifar10_train_transform(self):
        normalize = self._get_normalize_transform()
        
        if self.use_augmentation:
            # PIL → aug → tensor → normalize
            return T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                T.ToTensor(),
                normalize
            ])
        else:
            # tensor → normalize
            return T.Compose([normalize])

    def _cifar10_test_transform(self):
        return self._get_normalize_transform()

    def _fashionmnist_train_transform(self):
        normalize = self._get_normalize_transform()
        
        if self.use_augmentation:
            # PIL → aug → tensor → normalize
            return T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(28, padding=4),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                T.ToTensor(),
                normalize
            ])
        else:
            # PIL → tensor → normalize
            return T.Compose([
                T.ToTensor(),
                normalize
            ])

    def _fashionmnist_test_transform(self):
        return T.Compose([
            T.ToTensor(),
            self._get_normalize_transform()
        ])

    def get_dataloaders(self):
        use_pil_pipeline = self.use_augmentation

        if self.dataset_name == "cifar10":
            train_dataset = CIFAR10Pickle(
                self.data_root, train=True,
                transform=self._cifar10_train_transform(),
                use_pil_pipeline=use_pil_pipeline
            )
            test_dataset = CIFAR10Pickle(
                self.data_root, train=False,
                transform=self._cifar10_test_transform(),
                use_pil_pipeline=False
            )
        else:
            train_dataset = FashionMNISTDataset(
                self.data_root, train=True,
                transform=self._fashionmnist_train_transform()
            )
            test_dataset = FashionMNISTDataset(
                self.data_root, train=False,
                transform=self._fashionmnist_test_transform()
            )

        # Safe DataLoader
        pin_mem = torch.cuda.is_available()
        workers = self.num_workers if pin_mem else 0

        kwargs = {
            'batch_size': self.batch_size,
            'num_workers': workers,
            'pin_memory': pin_mem,
            'persistent_workers': workers > 0,
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **kwargs)

        return train_loader, test_loader
