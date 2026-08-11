"""
Experiment 4 — Kernel and classification analysis.

Usage:
    python kernel_analysis.py <job_idx>
    python kernel_analysis.py --list

Grid:
    3 circuits × 4 qubit counts × 4 layer counts × 4 datasets × 5 seeds
    = 960 jobs.

Each job writes one CSV containing classification and kernel metrics.
"""

import csv
import itertools
import os
import sys
import time

import numpy as np
import pennylane as qml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from circuits import CIRCUITS, build_feature_map
from loader import load_dataset


OUTPUT_DIR = "kernel_results"
N_SAMPLES = 400
RANDOM_STATE_BASE = 42

QUBIT_COUNTS = [4, 8, 12, 16]
LAYER_COUNTS = [1, 2, 4, 8]
SEEDS = [0, 1, 2, 3, 4]


def build_job_list():
    return [
        {
            "circuit": circuit,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "dataset": dataset,
            "seed": seed,
        }
        for circuit, n_qubits, n_layers, dataset, seed in itertools.product(
            CIRCUITS, QUBIT_COUNTS, LAYER_COUNTS, DATASETS, SEEDS
        )
    ]


def build_kernel(circuit_name, n_qubits, n_layers):
    device = qml.device("lightning.qubit", wires=n_qubits)
    feature_map = build_feature_map(circuit_name, n_qubits, n_layers)

    @qml.qnode(device)
    def kernel_circuit(x1, x2):
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=range(n_qubits))

    def kernel_matrix(X1, X2):
        n1, n2 = len(X1), len(X2)
        kernel = np.zeros((n1, n2))

        symmetric = n1 == n2 and X1 is X2

        if symmetric:
            total = n1 * (n1 + 1) // 2
            done = 0

            for i in range(n1):
                kernel[i, i] = float(kernel_circuit(X1[i], X1[i])[0])
                done += 1

                for j in range(i + 1, n1):
                    value = float(kernel_circuit(X1[i], X1[j])[0])
                    kernel[i, j] = value
                    kernel[j, i] = value
                    done += 1

                _print_progress(done, total)
        else:
            total = n1 * n2

            for i, x1 in enumerate(X1):
                for j, x2 in enumerate(X2):
                    kernel[i, j] = float(kernel_circuit(x1, x2)[0])
                    _print_progress(i * n2 + j + 1, total)

        return kernel

    return kernel_matrix


def _print_progress(done, total):
    if total == 0:
        return

    percentage = int(100 * done / total)
    milestone = (percentage // 10) * 10

    if milestone > 0 and done == total:
        print(f"  kernel matrix: 100% ({done}/{total})", flush=True)
    elif milestone > 0 and done % max(1, total // 10) == 0:
        print(f"  kernel matrix: {milestone}% ({done}/{total})", flush=True)


def centered_kta(kernel, labels):
    labels_pm = np.where(labels == 0, -1.0, 1.0)

    column_mean = kernel.mean(axis=0)
    row_mean = kernel.mean(axis=1, keepdims=True)
    total_mean = kernel.mean()

    centered = kernel - column_mean - row_mean + total_mean
    target = np.outer(labels_pm, labels_pm)

    numerator = np.sum(centered * target)
    denominator = (
        np.linalg.norm(centered, "fro")
        * np.linalg.norm(target, "fro")
        + 1e-12
    )

    return float(numerator / denominator)


def kernel_spectrum_metrics(kernel):
    n = kernel.shape[0]
    off_diagonal = kernel[~np.eye(n, dtype=bool)]

    eigenvalues = np.linalg.eigvalsh(kernel)
    eigenvalues = np.clip(eigenvalues, 0, None)[::-1]

    first_moment = np.sum(eigenvalues)
    second_moment = np.sum(eigenvalues**2)

    effective_dimension = (
        first_moment**2 / (second_moment + 1e-12)
    )

    spectral_ratio = float(
        eigenvalues[0] / (eigenvalues[-1] + 1e-12)
    )

    normalized = kernel / (np.trace(kernel) / n + 1e-12)
    frobenius_to_identity = float(
        np.linalg.norm(normalized - np.eye(n)) / n
    )

    return {
        "off_diag_variance": float(np.var(off_diagonal)),
        "frobenius_to_id": frobenius_to_identity,
        "effective_dimension": float(effective_dimension),
        "spectral_ratio": spectral_ratio,
    }


def classification_metrics(y_true, y_pred):
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(
        balanced_accuracy_score(y_true, y_pred)
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_c0": float(precision[0]),
        "recall_c0": float(recall[0]),
        "f1_c0": float(f1[0]),
        "precision_c1": float(precision[1]),
        "recall_c1": float(recall[1]),
        "f1_c1": float(f1[1]),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def run_job(job):
    circuit = job["circuit"]
    n_qubits = job["n_qubits"]
    n_layers = job["n_layers"]
    dataset = job["dataset"]
    seed = job["seed"]

    print(
        f"\n{'=' * 70}\n"
        f"Job: circuit={circuit} | qubits={n_qubits} | "
        f"layers={n_layers} | dataset={dataset} | seed={seed}\n"
        f"{'=' * 70}",
        flush=True,
    )

    X, y = load_dataset(
        dataset,
        n_samples=400,
        n_qubits=n_qubits,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE_BASE + seed,
    )

    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kernel_fn = build_kernel(circuit, n_qubits, n_layers)

    start = time.time()

    print(
        f"  Calculating K_train ({len(X_train)} × {len(X_train)})...",
        flush=True,
    )
    K_train = kernel_fn(X_train, X_train)

    print(
        f"  Calculating K_test ({len(X_test)} × {len(X_train)})...",
        flush=True,
    )
    K_test = kernel_fn(X_test, X_train)

    classifier = SVC(kernel="precomputed", C=1.0)
    classifier.fit(K_train, y_train)

    train_pred = classifier.predict(K_train)
    test_pred = classifier.predict(K_test)

    train_metrics = classification_metrics(y_train, train_pred)
    test_metrics = classification_metrics(y_test, test_pred)

    kta = centered_kta(K_train, y_train)
    spectrum = kernel_spectrum_metrics(K_train)

    n_support_vectors = len(classifier.support_)
    support_vector_fraction = n_support_vectors / len(X_train)

    return {
        "circuit": circuit,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "dataset": dataset,
        "seed": seed,
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),

        "train_accuracy": round(train_metrics["accuracy"], 6),
        "train_balanced_accuracy": round(
            train_metrics["balanced_accuracy"], 6
        ),
        "train_precision_c0": round(train_metrics["precision_c0"], 6),
        "train_recall_c0": round(train_metrics["recall_c0"], 6),
        "train_f1_c0": round(train_metrics["f1_c0"], 6),
        "train_precision_c1": round(train_metrics["precision_c1"], 6),
        "train_recall_c1": round(train_metrics["recall_c1"], 6),
        "train_f1_c1": round(train_metrics["f1_c1"], 6),
        "train_TP": train_metrics["TP"],
        "train_TN": train_metrics["TN"],
        "train_FP": train_metrics["FP"],
        "train_FN": train_metrics["FN"],

        "test_accuracy": round(test_metrics["accuracy"], 6),
        "test_balanced_accuracy": round(
            test_metrics["balanced_accuracy"], 6
        ),
        "test_precision_c0": round(test_metrics["precision_c0"], 6),
        "test_recall_c0": round(test_metrics["recall_c0"], 6),
        "test_f1_c0": round(test_metrics["f1_c0"], 6),
        "test_precision_c1": round(test_metrics["precision_c1"], 6),
        "test_recall_c1": round(test_metrics["recall_c1"], 6),
        "test_f1_c1": round(test_metrics["f1_c1"], 6),
        "test_TP": test_metrics["TP"],
        "test_TN": test_metrics["TN"],
        "test_FP": test_metrics["FP"],
        "test_FN": test_metrics["FN"],

        "kta": round(kta, 6),
        "effective_dimension": round(
            spectrum["effective_dimension"], 4
        ),
        "off_diag_variance": round(
            spectrum["off_diag_variance"], 8
        ),
        "frobenius_to_id": round(
            spectrum["frobenius_to_id"], 6
        ),
        "spectral_ratio": round(
            spectrum["spectral_ratio"], 4
        ),

        "n_support_vectors": n_support_vectors,
        "support_vector_frac": round(
            support_vector_fraction, 6
        ),
        "elapsed_seconds": round(time.time() - start, 1),
    }


FIELDNAMES = [
    "circuit", "n_qubits", "n_layers", "dataset", "seed",
    "n_samples", "n_train", "n_test",

    "train_accuracy", "train_balanced_accuracy",
    "train_precision_c0", "train_recall_c0", "train_f1_c0",
    "train_precision_c1", "train_recall_c1", "train_f1_c1",
    "train_TP", "train_TN", "train_FP", "train_FN",

    "test_accuracy", "test_balanced_accuracy",
    "test_precision_c0", "test_recall_c0", "test_f1_c0",
    "test_precision_c1", "test_recall_c1", "test_f1_c1",
    "test_TP", "test_TN", "test_FP", "test_FN",

    "kta", "effective_dimension",
    "off_diag_variance", "frobenius_to_id", "spectral_ratio",
    "n_support_vectors", "support_vector_frac",
    "elapsed_seconds",
]


def save_result(result, job_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"job_{job_id}.csv")

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in FIELDNAMES})

    print(f"  Saved: {path}", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        jobs = build_job_list()

        for index, job in enumerate(jobs):
            print(
                f"{index:>4}  {job['circuit']:<18} "
                f"{job['n_qubits']:>2}q  {job['n_layers']:>2}L  "
                f"{job['dataset']:<12} seed={job['seed']}"
            )

        print(f"\nTotal jobs: {len(jobs)}")
        return

    if len(sys.argv) < 2:
        print("Usage: python kernel_analysis.py <job_idx>")
        print("       python kernel_analysis.py --list")
        sys.exit(1)

    job_index = int(sys.argv[1])
    jobs = build_job_list()

    if not 0 <= job_index < len(jobs):
        print(f"job_idx must be between 0 and {len(jobs) - 1}.")
        sys.exit(1)

    job = jobs[job_index]
    job_id = (
        f"{job_index:04d}_{job['circuit']}_"
        f"{job['n_qubits']}q_{job['n_layers']}l_"
        f"{job['dataset']}_s{job['seed']}"
    )

    result = run_job(job)
    save_result(result, job_id)


if __name__ == "__main__":
    main()
