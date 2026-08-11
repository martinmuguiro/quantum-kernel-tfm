"""Experiment 4 — Quantum embedding geometry analysis."""

import csv
import itertools
import os
import sys
import time

import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from circuits import CIRCUITS, build_feature_map
from loader import load_dataset


OUTPUT_DIR = "embedding_results"
N_SAMPLES = 400
RANDOM_STATE = 42
SVD_QUBIT_LIMIT = 8

QUBIT_COUNTS = [4, 8, 12, 16]
LAYER_COUNTS = [1, 2, 4, 8]


def build_job_list():
    return [
        {
            "circuit": circuit,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "dataset": dataset,
        }
        for circuit, n_qubits, n_layers, dataset in itertools.product(
            CIRCUITS, QUBIT_COUNTS, LAYER_COUNTS, DATASETS
        )
    ]


def build_state_evaluator(circuit_name, n_qubits, n_layers):
    device = qml.device("lightning.qubit", wires=n_qubits)
    feature_map = build_feature_map(circuit_name, n_qubits, n_layers)

    @qml.qnode(device)
    def state_circuit(x):
        feature_map(x)
        return qml.state()

    def get_states(X):
        dimension = 2**n_qubits
        states = np.empty((len(X), dimension), dtype=complex)

        for index, x in enumerate(X):
            states[index] = np.asarray(state_circuit(x))

            if (index + 1) % 50 == 0 or index + 1 == len(X):
                print(
                    f"  states: {index + 1}/{len(X)}",
                    flush=True,
                )

        return states

    return get_states


def state_entropy_metrics(states):
    probabilities = np.abs(states) ** 2
    probabilities /= (
        probabilities.sum(axis=1, keepdims=True) + 1e-15
    )

    log_probabilities = np.where(
        probabilities > 1e-15,
        np.log(probabilities),
        0.0,
    )

    entropy = -np.sum(
        probabilities * log_probabilities,
        axis=1,
    )
    participation_ratio = 1.0 / (
        np.sum(probabilities**2, axis=1) + 1e-15
    )

    dimension = states.shape[1]
    occupancy = np.mean(
        np.sum(probabilities > 0.01, axis=1) / dimension
    )

    return {
        "shannon_entropy_mean": float(np.mean(entropy)),
        "shannon_entropy_std": float(np.std(entropy)),
        "shannon_entropy_max": float(np.log(dimension)),
        "participation_ratio_mean": float(
            np.mean(participation_ratio)
        ),
        "participation_ratio_std": float(
            np.std(participation_ratio)
        ),
        "basis_occupancy_frac": float(occupancy),
    }


def centroid_distance(states, labels):
    magnitudes = np.abs(states)

    class_zero = magnitudes[labels == 0].mean(axis=0)
    class_one = magnitudes[labels == 1].mean(axis=0)

    return float(np.linalg.norm(class_zero - class_one))


def svd_metrics(states):
    _, singular_values, _ = np.linalg.svd(
        states, full_matrices=False
    )
    singular_values = singular_values[
        singular_values > 1e-12
    ]

    variance = singular_values**2
    total_variance = variance.sum()
    cumulative_variance = (
        np.cumsum(variance) / total_variance
    )

    return {
        "svd_rank_90pct": int(
            np.searchsorted(cumulative_variance, 0.90) + 1
        ),
        "svd_effective_rank": float(
            total_variance**2
            / (np.sum(variance**2) + 1e-15)
        ),
        "svd_top1_variance_frac": float(
            variance[0] / total_variance
        ),
        "svd_n_singular_values": int(len(singular_values)),
    }


def run_job(job):
    circuit = job["circuit"]
    n_qubits = job["n_qubits"]
    n_layers = job["n_layers"]
    dataset = job["dataset"]

    print(
        f"\n{'=' * 70}\n"
        f"Job: circuit={circuit} | qubits={n_qubits} | "
        f"layers={n_layers} | dataset={dataset}\n"
        f"{'=' * 70}",
        flush=True,
    )

    X, y = load_dataset(dataset, n_samples=N_SAMPLES)

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train = scaler.fit_transform(X_train)

    start = time.time()

    print(
        f"  Calculating states ({len(X_train)} points)...",
        flush=True,
    )

    states = build_state_evaluator(
        circuit, n_qubits, n_layers
    )(X_train)

    entropy = state_entropy_metrics(states)
    distance = centroid_distance(states, y_train)

    if n_qubits <= SVD_QUBIT_LIMIT:
        print("  Calculating SVD...", flush=True)
        svd = svd_metrics(states)
    else:
        svd = {
            "svd_rank_90pct": "",
            "svd_effective_rank": "",
            "svd_top1_variance_frac": "",
            "svd_n_singular_values": "",
        }

    return {
        "circuit": circuit,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "dataset": dataset,
        "n_samples": len(X),
        "n_train": len(X_train),

        "shannon_entropy_mean": round(
            entropy["shannon_entropy_mean"], 6
        ),
        "shannon_entropy_std": round(
            entropy["shannon_entropy_std"], 6
        ),
        "shannon_entropy_max": round(
            entropy["shannon_entropy_max"], 6
        ),
        "participation_ratio_mean": round(
            entropy["participation_ratio_mean"], 4
        ),
        "participation_ratio_std": round(
            entropy["participation_ratio_std"], 4
        ),
        "basis_occupancy_frac": round(
            entropy["basis_occupancy_frac"], 6
        ),

        "centroid_distance": round(distance, 6),

        "svd_rank_90pct": svd["svd_rank_90pct"],
        "svd_effective_rank": svd["svd_effective_rank"],
        "svd_top1_variance_frac": svd[
            "svd_top1_variance_frac"
        ],
        "svd_n_singular_values": svd[
            "svd_n_singular_values"
        ],

        "elapsed_seconds": round(time.time() - start, 1),
    }


FIELDNAMES = [
    "circuit", "n_qubits", "n_layers", "dataset",
    "n_samples", "n_train",
    "shannon_entropy_mean", "shannon_entropy_std",
    "shannon_entropy_max",
    "participation_ratio_mean", "participation_ratio_std",
    "basis_occupancy_frac",
    "centroid_distance",
    "svd_rank_90pct", "svd_effective_rank",
    "svd_top1_variance_frac", "svd_n_singular_values",
    "elapsed_seconds",
]


def save_result(result, job_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"job_{job_id}.csv")

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in FIELDNAMES})

    print(f"Saved: {path}", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        jobs = build_job_list()
        for index, job in enumerate(jobs):
            print(
                f"{index:>4}  {job['circuit']:<18} "
                f"{job['n_qubits']:>2}q  {job['n_layers']:>2}L  "
                f"{job['dataset']}"
            )
        print(f"\nTotal jobs: {len(jobs)}")
        return

    if len(sys.argv) < 2:
        print("Usage: python embedding_analysis.py <job_idx>")
        print("       python embedding_analysis.py --list")
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
        f"{job['dataset']}"
    )

    save_result(run_job(job), job_id)


if __name__ == "__main__":
    main()
