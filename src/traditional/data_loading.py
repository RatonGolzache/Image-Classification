import pickle
import os
import numpy as np

from torchvision import datasets

def unpickle(file_path):
    """Load a CIFAR batch file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f, encoding='bytes')
    

def load_cifar_dataset(data_dir):
    """
    Loads all CIFAR training and test data.
    """

    # ---------- Load training batches ----------
    X_train_list = []
    y_train_list = []

    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        batch = unpickle(batch_path)

        X_train_list.append(batch[b'data'])
        y_train_list.append(batch[b'labels'])

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)

    # ---------- Load test batch ----------
    test_batch = unpickle(os.path.join(data_dir, "test_batch"))
    X_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])

    # ---------- Load label names ----------
    meta = unpickle(os.path.join(data_dir, "batches.meta"))
    label_names = [name.decode("utf-8") for name in meta[b'label_names']]

    # ---------- Reshape images ----------
    # From (N, 3072) → (N, 3, 32, 32) → (N, 32, 32, 3)
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # ---------- Keep images as unit8 ----------
    X_train = X_train.astype(np.uint8)
    X_test  = X_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test, label_names

def load_fashion_mnist_dataset(data_dir="./data"):
    """
    Loads Fashion-MNIST using torchvision, returns it as uint8 RGB-like (3 channels)
    so you can reuse the exact same feature pipeline as CIFAR.
    """

    train = datasets.FashionMNIST(root=data_dir, train=True, download=True)
    test  = datasets.FashionMNIST(root=data_dir, train=False, download=True)

    X_train = train.data.numpy().astype(np.uint8)    
    y_train = train.targets.numpy()
    X_test  = test.data.numpy().astype(np.uint8)     
    y_test  = test.targets.numpy()

    # grayscale -> 3-channel 
    X_train = np.stack([X_train, X_train, X_train], axis=-1) 
    X_test  = np.stack([X_test,  X_test,  X_test],  axis=-1)

    label_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    return X_train, y_train, X_test, y_test, label_names