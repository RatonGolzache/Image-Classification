import os
import time
import argparse
import yaml

from sklearn.model_selection import train_test_split

from .data_loading import load_cifar_dataset, load_fashion_mnist_dataset
from .pipelines import run_histogram_pipeline, run_sift_bovw_pipeline


# ----------------------------
# Helpers: subsampling 
# ----------------------------
def subsample_dataset(X, y, fraction: float, random_state: int):
    """
    Stratified subsample to keep class proportions.
    """
    if fraction >= 1.0:
        return X, y

    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=fraction,
        stratify=y,
        random_state=random_state
    )
    return X_sub, y_sub


# ----------------------------
# Helpers: TXT output 
# ----------------------------
def summarize_best_configs(results, score_key="accuracy"):
    lines = []
    lines.append("===== BEST CONFIGURATION SUMMARY =====\n")

    for pipeline_name, pipeline_results in results.items():
        lines.append(f"Pipeline: {pipeline_name}")

        # Best SVM
        best_svm_cfg, best_svm_score = None, -1.0
        for cfg_name, cfg_metrics in pipeline_results.get("svm_grid", {}).items():
            score = cfg_metrics.get(score_key, -1.0)
            if score > best_svm_score:
                best_svm_score = score
                best_svm_cfg = cfg_name

        if best_svm_cfg:
            lines.append(f"  Best SVM: {best_svm_cfg} | {score_key}={best_svm_score:.4f}")
        else:
            lines.append("  Best SVM: not found")

        # Best kNN
        best_knn_cfg, best_knn_score = None, -1.0
        for cfg_name, cfg_metrics in pipeline_results.get("knn_grid", {}).items():
            score = cfg_metrics.get(score_key, -1.0)
            if score > best_knn_score:
                best_knn_score = score
                best_knn_cfg = cfg_name

        if best_knn_cfg:
            lines.append(f"  Best k-NN: {best_knn_cfg} | {score_key}={best_knn_score:.4f}")
        else:
            lines.append("  Best k-NN: not found")

        lines.append("")

    lines.append("===== FULL RESULTS BELOW =====\n")
    return "\n".join(lines)


def write_results_txt(results: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summarize_best_configs(results))

        for pipeline_name, pipeline_results in results.items():
            f.write(f"\n=== {pipeline_name} ===\n")

            for block_name, block_data in pipeline_results.items():

                # simple values
                if isinstance(block_data, (float, int, str)):
                    f.write(f"\n  {block_name}: {block_data}\n")
                    continue

                # grids (dict)
                if isinstance(block_data, dict):
                    f.write(f"\n  {block_name}:\n")
                    for cfg_name, metrics in block_data.items():
                        f.write(f"    {cfg_name}:\n")

                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, dict):
                                f.write(f"      {metric_name}:\n")
                                for cls_name, cls_val in metric_value.items():
                                    f.write(f"        {cls_name}: {cls_val:.4f}\n")
                            else:
                                if isinstance(metric_value, (float, int)):
                                    f.write(f"      {metric_name}: {metric_value:.4f}\n")
                                else:
                                    f.write(f"      {metric_name}: {metric_value}\n")


# ----------------------------
# YAML Config
# ----------------------------
def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------
# Main runner (CIFAR + Fashion)
# ----------------------------
def run_dataset(dataset_name: str, X_train, y_train, X_test, y_test, label_names, cfg: dict, output_dir: str):
    random_state = int(cfg["global"].get("random_state", 19))
    use_fraction = float(cfg["global"].get("use_fraction", 1.0))
    vocab_size = int(cfg["global"].get("vocab_size", 100))

    # Grids
    svm_C_grid = tuple(cfg["grids"]["svm"]["C_grid"])
    svm_kernel_grid = tuple(cfg["grids"]["svm"]["kernel_grid"])
    svm_gamma_grid = tuple(cfg["grids"]["svm"]["gamma_grid"])

    knn_k_grid = tuple(cfg["grids"]["knn"]["k_grid"])
    knn_weights_grid = tuple(cfg["grids"]["knn"]["weights_grid"])

    # Subsample
    X_train, y_train = subsample_dataset(X_train, y_train, use_fraction, random_state)
    X_test, y_test = subsample_dataset(X_test, y_test, use_fraction, random_state)

    print("\n===========================================")
    print(f"DATASET: {dataset_name}")
    print(f"OUTPUT : {output_dir}")
    print(f"Train  : {len(X_train)}")
    print(f"Test   : {len(X_test)}")
    print("===========================================")

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Histogram
    results["Histogram"] = run_histogram_pipeline(
        X_train, y_train,
        X_test, y_test,
        label_names,
        output_dir=os.path.join(output_dir, "histogram"),
        hist_bins=(4, 4, 4),
        svm_C_grid=svm_C_grid,
        svm_kernel_grid=svm_kernel_grid,
        svm_gamma_grid=svm_gamma_grid,
        knn_k_grid=knn_k_grid,
        knn_weights_grid=knn_weights_grid,
    )

    # SIFT + BoVW
    results["SIFT+BoVW"] = run_sift_bovw_pipeline(
        X_train, y_train,
        X_test, y_test,
        label_names,
        output_dir=os.path.join(output_dir, "sift_bovw"),
        vocab_size=vocab_size,
        svm_C_grid=svm_C_grid,
        svm_kernel_grid=svm_kernel_grid,
        svm_gamma_grid=svm_gamma_grid,
        knn_k_grid=knn_k_grid,
        knn_weights_grid=knn_weights_grid,
    )

    # Save txt summary
    results_txt_path = os.path.join(output_dir, "results.txt")
    write_results_txt(results, results_txt_path)
    print(f"Saved results to: {results_txt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_traditional.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    start_total = time.time()

    # CIFAR-10
    cifar_cfg = cfg["datasets"]["cifar10"]
    if cifar_cfg.get("enabled", True):
        X_train, y_train, X_test, y_test, label_names = load_cifar_dataset(cifar_cfg["data_dir"])
        run_dataset(
            "cifar10",
            X_train, y_train,
            X_test, y_test,
            label_names,
            cfg=cfg,
            output_dir=cifar_cfg["output_dir"],
        )

    # Fashion-MNIST
    fashion_cfg = cfg["datasets"]["fashion_mnist"]
    if fashion_cfg.get("enabled", True):
        X_train, y_train, X_test, y_test, label_names = load_fashion_mnist_dataset(fashion_cfg["data_dir"])
        run_dataset(
            "fashion_mnist",
            X_train, y_train,
            X_test, y_test,
            label_names,
            cfg=cfg,
            output_dir=fashion_cfg["output_dir"],
        )

    print(f"\nALL DONE. Total runtime: {time.time() - start_total:.2f} seconds")


if __name__ == "__main__":
    main()
