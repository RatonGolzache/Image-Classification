import os
import time
import numpy as np

from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from .features import extract_color_histogram, extract_sift_bovw_features, extract_sift_descriptors, compute_bovw_histogram
from .evaluation import per_class_accuracy, save_confusion_matrix


def run_histogram_pipeline(
    X_train, y_train,
    X_test, y_test,
    label_names,
    output_dir,
    hist_bins=(4, 4, 4),
    svm_C_grid=(0.1, 1, 10),
    svm_kernel_grid=("linear", "rbf"),
    svm_gamma_grid=("scale", 0.01, 0.1),
    knn_k_grid=(3, 5, 11),
    knn_weights_grid=("uniform", "distance"),
):
    print("\n=== Color Histogram Pipeline ===")
    os.makedirs(output_dir, exist_ok=True)

    # Feature extraction
    print("\n--- Feature Extraction ---")
    start = time.time()
    X_hist_train = np.array([extract_color_histogram(img, bins=hist_bins) for img in X_train])
    X_hist_test = np.array([extract_color_histogram(img, bins=hist_bins) for img in X_test])
    feat_time = time.time() - start

    normalizer = Normalizer(norm="l2")
    X_hist_train = normalizer.fit_transform(X_hist_train)
    X_hist_test = normalizer.transform(X_hist_test)

    results = {"feature_time": float(feat_time), "svm_grid": {}, "knn_grid": {}}

    # ---- SVM grid ----
    print("\n--- Histogram + SVM ---")
    for kernel in svm_kernel_grid:
        for C in svm_C_grid:
            gamma_values = ["NA"] if kernel == "linear" else svm_gamma_grid

            for gamma in gamma_values:
                model_name = f"SVC_kernel={kernel}_C={C}_gamma={gamma}"

                start = time.time()
                svm = SVC(kernel=kernel, C=C) if kernel == "linear" else SVC(kernel=kernel, C=C, gamma=gamma)
                svm.fit(X_hist_train, y_train)
                train_time = time.time() - start

                start = time.time()
                y_pred = svm.predict(X_hist_test)
                test_time = time.time() - start

                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                per_class = per_class_accuracy(y_test, y_pred, label_names)

                cm_path = os.path.join(output_dir, f"Histogram_{model_name}_cm.png").replace(".", "_")
                save_confusion_matrix(y_test, y_pred, label_names, cm_path, title=f"Histogram | {model_name}")

                results["svm_grid"][model_name] = {
                    "accuracy": float(acc),
                    "balanced_accuracy": float(bal_acc),
                    "train_time": float(train_time),
                    "test_time": float(test_time),
                    "per_class_accuracy": per_class,
                }

    # ---- kNN grid ----
    print("\n--- Histogram + kNN ---")
    for k in knn_k_grid:
        for w in knn_weights_grid:
            model_name = f"kNN_k={k}_weights={w}"

            start = time.time()
            knn = KNeighborsClassifier(n_neighbors=k, weights=w)
            knn.fit(X_hist_train, y_train)
            train_time = time.time() - start

            start = time.time()
            y_pred = knn.predict(X_hist_test)
            test_time = time.time() - start

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            per_class = per_class_accuracy(y_test, y_pred, label_names)

            cm_path = os.path.join(output_dir, f"Histogram_{model_name}_cm.png").replace(".", "_")
            save_confusion_matrix(y_test, y_pred, label_names, cm_path, title=f"Histogram | {model_name}")

            results["knn_grid"][model_name] = {
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "train_time": float(train_time),
                "test_time": float(test_time),
                "per_class_accuracy": per_class,
            }

    return results


def run_sift_bovw_pipeline(
    X_train, y_train,
    X_test, y_test,
    label_names,
    output_dir,
    vocab_size=100,
    svm_C_grid=(0.1, 1, 10),
    svm_kernel_grid=("linear", "rbf"),
    svm_gamma_grid=("scale", 0.01, 0.1),
    knn_k_grid=(3, 5, 11),
    knn_weights_grid=("uniform", "distance"),
):
    print("\n=== SIFT + BoVW Pipeline ===")
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Feature Extraction ---")

    # Training features (includes kmeans fit)
    start = time.time()
    X_bovw_train, kmeans = extract_sift_bovw_features(X_train, vocab_size=vocab_size)
    feat_time_train = time.time() - start

    # Test features (reuse vocab)
    start = time.time()
    X_bovw_test = np.array([
        compute_bovw_histogram(extract_sift_descriptors(img), kmeans)
        for img in X_test
    ])
    feat_time_test = time.time() - start

    normalizer = Normalizer(norm="l2")
    X_bovw_train = normalizer.fit_transform(X_bovw_train)
    X_bovw_test = normalizer.transform(X_bovw_test)

    results = {
        "vocab_size": int(vocab_size),
        "feature_time_train": float(feat_time_train),
        "feature_time_test": float(feat_time_test),
        "feature_time_total": float(feat_time_train + feat_time_test),
        "svm_grid": {},
        "knn_grid": {},
    }

    # ---- SVM grid ----
    print("\n--- SIFT+BoVW + SVM ---")
    for kernel in svm_kernel_grid:
        for C in svm_C_grid:
            gamma_values = ["NA"] if kernel == "linear" else svm_gamma_grid

            for gamma in gamma_values:
                model_name = f"SVC_kernel={kernel}_C={C}_gamma={gamma}"

                start = time.time()
                svm = SVC(kernel=kernel, C=C) if kernel == "linear" else SVC(kernel=kernel, C=C, gamma=gamma)
                svm.fit(X_bovw_train, y_train)
                train_time = time.time() - start

                start = time.time()
                y_pred = svm.predict(X_bovw_test)
                test_time = time.time() - start

                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                per_class = per_class_accuracy(y_test, y_pred, label_names)

                cm_path = os.path.join(output_dir, f"SIFTBoVW_vocab{vocab_size}_{model_name}_cm.png").replace(".", "_")
                save_confusion_matrix(y_test, y_pred, label_names, cm_path, title=f"SIFT+BoVW | {model_name}")

                results["svm_grid"][model_name] = {
                    "accuracy": float(acc),
                    "balanced_accuracy": float(bal_acc),
                    "train_time": float(train_time),
                    "test_time": float(test_time),
                    "per_class_accuracy": per_class,
                }

    # ---- kNN grid ----
    print("\n--- SIFT+BoVW + kNN ---")
    for k in knn_k_grid:
        for w in knn_weights_grid:
            model_name = f"kNN_k={k}_weights={w}"

            start = time.time()
            knn = KNeighborsClassifier(n_neighbors=k, weights=w)
            knn.fit(X_bovw_train, y_train)
            train_time = time.time() - start

            start = time.time()
            y_pred = knn.predict(X_bovw_test)
            test_time = time.time() - start

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            per_class = per_class_accuracy(y_test, y_pred, label_names)

            cm_path = os.path.join(output_dir, f"SIFTBoVW_vocab{vocab_size}_{model_name}_cm.png").replace(".", "_")
            save_confusion_matrix(y_test, y_pred, label_names, cm_path, title=f"SIFT+BoVW | {model_name}")

            results["knn_grid"][model_name] = {
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "train_time": float(train_time),
                "test_time": float(test_time),
                "per_class_accuracy": per_class,
            }

    return results
