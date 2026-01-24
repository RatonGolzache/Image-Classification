import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_module import DataModule
from models.squeezenet import SqueezeNetCIFAR10, SqueezeNetFashionMNIST
from models.resnet_transfer import ResNetTransfer


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


def freeze_backbone(model: nn.Module) -> None:
    """Freeze features/backbone parameters for transfer learning."""
    if hasattr(model, 'features'):
        for param in model.features.parameters():
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def get_trainable_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """Get fresh Adam optimizer with ONLY trainable params."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)


def print_model_stats(model: nn.Module, stage: str = "init", is_frozen: bool = False):
    """Print detailed model stats: total/trainable params, % frozen."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    frozen_pct = 100 * frozen_params / total_params if total_params > 0 else 0
    
    print(f"\n[{stage}] Model Stats:")
    print(f"  Total params:     {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"  Frozen params:    {frozen_params:,} ({frozen_pct:.1f}%)")
    print(f"  Status:           {'FROZEN BACKBONE' if is_frozen else 'FULLY TRAINABLE'}")
    
    # Features breakdown (for ResNet)
    if hasattr(model, 'features'):
        feat_params = sum(p.numel() for p in model.features.parameters())
        feat_frozen = sum(p.numel() for p in model.features.parameters() if not p.requires_grad)
        print(f"  Features (backbone): {feat_params:,} params ({100*feat_frozen/feat_params:.1f}% frozen)")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch with batch timing."""
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)
    
    print(f"  Starting epoch: {num_batches} batches, batch_size={dataloader.batch_size}")
    batch_start = time.time()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
        # Batch timing every 100 batches
        if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - batch_start
            batches_done = batch_idx + 1
            print(f"    Batches 1-{batches_done}/{num_batches}: {elapsed:.1f}s "
                  f"({elapsed/batches_done*1000:.0f}ms/batch)")
    
    epoch_time = time.time() - batch_start
    avg_loss = running_loss / len(dataloader.dataset)
    print(f"  Epoch complete: {epoch_time:.1f}s total ({epoch_time/60:.1f}min), "
          f"loss={avg_loss:.4f}")
    return avg_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SqueezeNet/ResNetTransfer on CIFAR-10 or Fashion-MNIST")
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
    train_loader, _ = data_module.get_dataloaders()  # Only need train_loader
    
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
        is_transfer = False
    elif model_cfg["type"] == "resnet_transfer":
        model = ResNetTransfer(num_classes=num_classes, dropout=model_cfg.get("dropout", 0.5)).to(device)
        is_transfer = True
    else:
        raise ValueError(f"Unsupported model type: {model_cfg['type']}")
    
    print_model_stats(model, "AFTER MODEL INIT", False)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Initial optimizer
    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"]["weight_decay"]
    optimizer = get_trainable_optimizer(model, lr, wd)  # Always trainable-only
    
    if is_transfer:
        freeze_backbone(model)
        optimizer = get_trainable_optimizer(model, lr, wd)  # Frozen optimizer
        print_model_stats(model, "AFTER FREEZE", True)

    
    # Training
    num_epochs = cfg["training"]["epochs"]
    unfreeze_epoch = num_epochs // 2 + 1
    start_time = time.time()
    history = []
    
    print(f"Training {dataset_cfg['name']} ({model_cfg['type']}) on {device_name} for {num_epochs} epochs...")
    if is_transfer:
        print(f"Transfer learning: Backbone frozen until epoch {unfreeze_epoch}")
    
    for epoch in range(1, num_epochs + 1):
        # Handle unfreezing for transfer models
        if is_transfer and epoch == unfreeze_epoch:
            print(f"Epoch {epoch}: Unfreezing backbone for fine-tuning")
            unfreeze_backbone(model)
            optimizer = get_trainable_optimizer(model, lr, wd)  

        print_model_stats(model, f"EPOCH {epoch}", epoch < unfreeze_epoch)
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Epoch [{epoch:2d}/{num_epochs}] train_loss: {train_loss:.4f} "
              f"(frozen: {is_transfer and epoch < unfreeze_epoch})")
        
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "backbone_frozen": is_transfer and epoch < unfreeze_epoch,
        })
    
    # Save checkpoint & info (unchanged)
    total_time_sec = time.time() - start_time
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "history": history,
        "model_type": model_cfg["type"],
        "dataset_name": dataset_name,
    }, ckpt_path)
    
    info = {
        "device_type": device.type,
        "device_name": device_name,
        "dataset": dataset_name,
        "model_type": model_cfg["type"],
        "total_time_seconds": total_time_sec,
        "epochs_trained": num_epochs,
        "unfreeze_epoch": unfreeze_epoch if is_transfer else None,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nCheckpoint saved: {ckpt_path}")
    print(f"Training info:    {info_path}")
    print(f"Total time:       {total_time_sec:.2f}s ({total_time_sec/60:.1f}min)")
    print(f"Device:           {device_name}")


if __name__ == "__main__":
    main()
