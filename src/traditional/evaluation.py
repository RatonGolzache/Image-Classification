import os
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def per_class_accuracy(y_true, y_pred, label_names):
    """Compute per-class accuracy from predictions."""

    cm = confusion_matrix(y_true, y_pred)
    class_acc = {}

    for i, label in enumerate(label_names):
        denom = cm[i].sum()
        class_acc[label] = 0.0 if denom == 0 else (cm[i, i] / denom)

    return class_acc


def save_confusion_matrix(y_true, y_pred, label_names, out_path, title=None):
    """Save confusion matrix image to out_path."""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(xticks_rotation=45)

    if title is not None:
        plt.title(title)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
