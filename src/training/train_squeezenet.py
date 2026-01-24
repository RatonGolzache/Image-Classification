import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_module import DataModule  # Updated import
from models.squeezenet import SqueezeNetCIFAR10
from models.squeezenet import SqueezeNetFashionMNIST

def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SqueezeNet on CIFAR-10 or Fashion-MNIST")
    parser.add_argument("--config", type=str, default="../../config/config.yml", 
                        help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    
    # Reproducibility
    seed = cfg.get("misc", {}).get("seed", 19)
    set_seed(seed)
    
    # Device setup
    device = get_device()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    
    # Checkpoint setup
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = cfg["training"].get("checkpoint_name", "model.pt")
    ckpt_path = ckpt_dir / checkpoint_name
    info_path = ckpt_dir / f"{checkpoint_name.rsplit('.', 1)[0]}.json"
    
    # Data loading
    dataset_cfg = cfg["dataset"]
    data_module = DataModule(
        dataset_name=dataset_cfg["name"],
        data_root=dataset_cfg["root"],
        batch_size=dataset_cfg["batch_size"],
        num_workers=dataset_cfg["num_workers"],
        use_augmentation=dataset_cfg.get("augment", True),
    )
    train_loader, test_loader = data_module.get_dataloaders()
    
    # Model setup
    model_cfg = cfg["model"]
    num_classes = model_cfg.get("num_classes", 10)
    dataset_name = dataset_cfg["name"].lower()
    
    if model_cfg["type"] == "squeezenet":
        if dataset_name == "cifar10":
            model = SqueezeNetCIFAR10(num_classes=num_classes, dropout=model_cfg.get("dropout", 0.5)).to(device)
        elif dataset_name == "fashionmnist":
            model = SqueezeNetFashionMNIST(num_classes=num_classes, dropout=model_cfg.get("dropout", 0.5)).to(device)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    
    # Training
    num_epochs = cfg["training"]["epochs"]
    start_time = time.time()
    history = []
    
    print(f"Training {dataset_cfg['name']} on {device_name} for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Epoch [{epoch:2d}/{num_epochs}] train_loss: {train_loss:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
        })
    
    # Final timing
    total_time_sec = time.time() - start_time
    
    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "history": history,
        "model_type": model_cfg["type"],
        "dataset_name": dataset_name,
    }, ckpt_path)
    
    # Save runtime info
    info = {
        "device_type": device.type,
        "device_name": device_name,
        "dataset": dataset_name,
        "total_time_seconds": total_time_sec,
        "epochs_trained": num_epochs,
    }
    
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nCheckpoint saved: {ckpt_path}")
    print(f"Training info:    {info_path}")
    print(f"Total time:       {total_time_sec:.2f}s ({total_time_sec/60:.1f}min)")
    print(f"Device:           {device_name}")

if __name__ == "__main__":
    main()
