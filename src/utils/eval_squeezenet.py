import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import sys

# Fix import - use the new data_module
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_module import DataModule  # Updated import path

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_class_names(dataset_name: str) -> list:
    """Get class names for the dataset."""
    if dataset_name.lower() == "cifar10":
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif dataset_name.lower() == "fashionmnist":
        return [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    else:
        return [f"Class_{i}" for i in range(10)]  # fallback

def evaluate(model: nn.Module, dataloader, device: torch.device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    # Balanced accuracy over all classes
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Per-class accuracy
    num_classes = int(y_true.max() + 1)
    per_class_acc = []
    for c in range(num_classes):
        idx = (y_true == c)
        correct = (y_pred[idx] == c).sum()
        total = idx.sum()
        acc = float(correct) / float(total) if total > 0 else 0.0
        per_class_acc.append(acc)

    # Confusion matrix (rows=true, cols=pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return bal_acc, per_class_acc, cm

def save_results(results_path: Path, dataset_name: str, bal_acc: float, 
                per_class_acc: list, cm: np.ndarray, class_names: list):
    """Save evaluation results to text file with formatted confusion matrix."""
    
    with open(results_path, 'w') as f:
        f.write(f"Model Evaluation Results\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Per-class Accuracy:\n")
        for i, (class_name, acc) in enumerate(zip(class_names, per_class_acc)):
            f.write(f"  {class_name:<12}: {acc:.4f}\n")
        f.write("\n")
        
        # Formatted confusion matrix with labels
        f.write("Confusion Matrix (rows=true labels, columns=predicted labels):\n")
        f.write(" " * 15 + "".join(f"{name[:12]:>12}" for name in class_names) + "\n")
        f.write("-" * (15 + 12 * len(class_names)) + "\n")
        
        # Row labels and values
        for i, true_class in enumerate(class_names):
            row = f"{true_class:<12}: "
            for j in range(len(class_names)):
                row += f"{cm[i,j]:>12}"
            f.write(row + "\n")
        
        f.write("-" * (15 + 12 * len(class_names)) + "\n")
        f.write(f"{'Total':<12}: {y_pred.sum():>12} predictions\n")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SqueezeNet/CNN on test set and save results"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file used for training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt) saved by the training script",
    )
    args = parser.parse_args()

    print("Loading evaluation config...")
    cfg = load_config(args.config)

    print("Setting up device...")
    device = get_device()

    # Setup data and test loader
    dataset_cfg = cfg["dataset"]
    data_module = DataModule(
        dataset_name=dataset_cfg["name"],
        data_root=dataset_cfg["root"],
        batch_size=dataset_cfg["batch_size"],
        num_workers=dataset_cfg["num_workers"],
        use_augmentation=False,
    )
    _, test_loader = data_module.get_dataloaders()

    model_cfg = cfg["model"]
    num_classes = model_cfg.get("num_classes", 10)
    dataset_name = dataset_cfg["name"].lower()
    
    print(f"Loading {dataset_name} model...")
    if model_cfg["type"] == "squeezenet":
        if dataset_name == "cifar10":
            from models.squeezenet import SqueezeNetCIFAR10  
            model = SqueezeNetCIFAR10(
                num_classes=num_classes,
                dropout=model_cfg.get("dropout", 0.5),
            ).to(device)
        elif dataset_name == "fashionmnist":
            from models.squeezenet import SqueezeNetFashionMNIST
            model = SqueezeNetFashionMNIST(
                num_classes=num_classes,
                dropout=model_cfg.get("dropout", 0.5),
            ).to(device)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded successfully. Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Evaluate
    print("Evaluating model...")
    bal_acc, per_class_acc, cm = evaluate(model, test_loader, device)

    # Get class names and create eval folder
    class_names = get_class_names(dataset_name)
    project_root = Path(__file__).parent.parent
    eval_dir = project_root / "eval"
    eval_dir.mkdir(exist_ok=True)

    # Generate results filename
    ckpt_name = Path(args.checkpoint).stem
    results_filename = f"{ckpt_name}_eval_results.txt"
    results_path = eval_dir / results_filename

    # Save detailed results
    save_results(results_path, dataset_name.upper(), bal_acc, per_class_acc, cm, class_names)
    
    print(f"\nBalanced accuracy: {bal_acc:.4f}")
    print("Per-class accuracy:")
    for class_name, acc in zip(class_names, per_class_acc):
        print(f"  {class_name:<12}: {acc:.4f}")
    
    print(f"\nDetailed results saved to: {results_path}")
    print("Confusion matrix preview:")
    print(cm)

if __name__ == "__main__":
    main()
