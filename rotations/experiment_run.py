"""
exp2_main.py -- Experiment 2: gate ablation study.

Evaluates two blocks of quantum kernel circuits across four datasets and
20 random seeds, appending every individual result to a single CSV.

Block 1 -- rotation axis ablation (no entanglement):
    9 combinations of data_axis x param_axis:
        R_data(x) + R_param(theta) + R_data(x)
    3 encoding-only circuits (no structural layer): R_X(x), R_Y(x), R_Z(x)
    Total: 12 configs.

Block 2 -- entanglement ablation (RY encoding fixed):
    no_ent, partial (CRZ on 0-1, 2-3), full (CRZ ring 0-1-2-3-0)
    RY(x) + [entanglement] + RY(x)
    Total: 3 configs.


Grand total: 15 configs (see ALL_CONFIGS) -> len(EXPERIMENTS) jobs
(encoding-only configs skip seeds > 0 and write a single row per dataset).


Output:
    results_exp2/exp2_results.csv  
"""

import csv
import fcntl
import os
import sys

import numpy as np
import pennylane as qml
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from loader import load_dataset
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_QUBITS = 4
N_SEEDS = 20
TEST_SIZE = 0.30
RANDOM_STATE = 42

OUTPUT_DIR = "results_exp2"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "exp2_results.csv")
CSV_FIELDS = [
    "block", "config_id", "config_label",
    "dataset", "seed",
    "acc_train", "acc_test",
]

DATASETS = ["moons", "circles", "xor", "checkerboard"]
AXES = ["X", "Y", "Z"]

N_SAMPLES = 400


# ---------------------------------------------------------------------------
# Circuit configurations
# ---------------------------------------------------------------------------

def _build_block1_configs():
    """9 data_axis/param_axis combinations + 3 encoding-only circuits."""
    configs = []
    for data_axis in AXES:
        for param_axis in AXES:
            configs.append({
                "block": 1,
                "config_id": len(configs) + 1,
                "config_label": f"enc_{data_axis}_param_{param_axis}",
                "data_axis": data_axis,
                "param_axis": param_axis,
                "encoding_only": False,
            })
    for axis in AXES:
        configs.append({
            "block": 1,
            "config_id": len(configs) + 1,
            "config_label": f"enc_{axis}_only",
            "data_axis": axis,
            "param_axis": None,
            "encoding_only": True,
        })
    return configs


BLOCK1_CONFIGS = _build_block1_configs()

BLOCK2_CONFIGS = [
    {"block": 2, "config_id": 13, "config_label": "ry_no_ent", "ent_level": "no_ent"},
    {"block": 2, "config_id": 14, "config_label": "ry_partial", "ent_level": "partial"},
    {"block": 2, "config_id": 15, "config_label": "ry_full", "ent_level": "full"},
]

ALL_CONFIGS = BLOCK1_CONFIGS + BLOCK2_CONFIGS 


def _build_experiments():
    """Flatten configs into a (config, dataset, seed) list.

    Encoding-only configs are evaluated once per dataset (seed fixed to 0);
    all other configs are evaluated once per (dataset, seed) pair.
    """
    experiments = []
    for cfg in ALL_CONFIGS:
        for dataset in DATASETS:
            if cfg.get("encoding_only", False):
                experiments.append((cfg, dataset, 0))
            else:
                for seed in range(N_SEEDS):
                    experiments.append((cfg, dataset, seed))
    return experiments


EXPERIMENTS = _build_experiments()

# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def apply_rotation(axis, angle, wire):
    """Apply a single-qubit rotation gate around X, Y, or Z."""
    if axis == "X":
        qml.RX(angle, wires=wire)
    elif axis == "Y":
        qml.RY(angle, wires=wire)
    elif axis == "Z":
        qml.RZ(angle, wires=wire)


def build_block1(x, params, data_axis, param_axis, encoding_only=False):
    """Block 1 circuit: R_data(x) + R_param(theta) + R_data(x).

    If encoding_only is True, stops after the first encoding layer.
    """
    for wire in range(N_QUBITS):
        apply_rotation(data_axis, x[wire % 2], wire)

    if encoding_only:
        return

    for wire in range(N_QUBITS):
        apply_rotation(param_axis, params[wire], wire)

    for wire in range(N_QUBITS):
        apply_rotation(data_axis, x[wire % 2], wire)


def build_block2(x, params, ent_level):
    """Block 2 circuit: RY(x) + [entanglement] + RY(x).

    params has shape (N_QUBITS,) and is indexed directly by the CRZ gates.
    """
    for wire in range(N_QUBITS):
        qml.RY(x[wire % 2], wires=wire)

    if ent_level == "partial":
        qml.CRZ(params[0], wires=[0, 1])
        qml.CRZ(params[2], wires=[2, 3])
    elif ent_level == "full":
        qml.CRZ(params[0], wires=[0, 1])
        qml.CRZ(params[1], wires=[1, 2])
        qml.CRZ(params[2], wires=[2, 3])
        qml.CRZ(params[3], wires=[3, 0])

    for wire in range(N_QUBITS):
        qml.RY(x[wire % 2], wires=wire)


def get_circuit_fn(cfg):
    """Return a callable build_fn(x, params) for the given config."""
    if cfg["block"] == 1:
        data_axis = cfg["data_axis"]
        param_axis = cfg["param_axis"]
        encoding_only = cfg.get("encoding_only", False)
        return lambda x, p: build_block1(x, p, data_axis, param_axis, encoding_only)
    ent_level = cfg["ent_level"]
    return lambda x, p: build_block2(x, p, ent_level)


# ---------------------------------------------------------------------------
# Kernel computation
# ---------------------------------------------------------------------------

def compute_kernels(x_train, x_test, build_fn, params):
    """Compute the train-train and test-train quantum kernel matrices."""
    dev = qml.device("lightning.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        build_fn(x1, params)
        qml.adjoint(build_fn)(x2, params)
        return qml.probs(wires=range(N_QUBITS))

    n_train = len(x_train)
    n_test = len(x_test)

    k_train = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(i, n_train):
            value = kernel_circuit(x_train[i], x_train[j])[0]
            k_train[i, j] = value
            k_train[j, i] = value

    k_test = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            k_test[i, j] = kernel_circuit(x_test[i], x_train[j])[0]

    return k_train, k_test


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_one(cfg, dataset_name, seed):
    """Run one (config, dataset, seed) experiment and return its result row."""
    X, y = load_dataset(
        dataset_name,
        n_samples=N_SAMPLES,
        n_qubits=N_QUBITS,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    if cfg.get("encoding_only", False):
        params = None
    else:
        rng = np.random.default_rng(seed)
        params = rng.uniform(-np.pi, np.pi, size=(N_QUBITS,))

    build_fn = get_circuit_fn(cfg)
    k_train, k_test = compute_kernels(x_train, x_test, build_fn, params)

    classifier = SVC(kernel="precomputed", C=1.0)
    classifier.fit(k_train, y_train)

    acc_train = classifier.score(k_train, y_train)
    acc_test = classifier.score(k_test, y_test)

    return {
        "block": cfg["block"],
        "config_id": cfg["config_id"],
        "config_label": cfg["config_label"],
        "dataset": dataset_name,
        "seed": seed,
        "acc_train": round(acc_train, 6),
        "acc_test": round(acc_test, 6),
    }


# ---------------------------------------------------------------------------
# CSV output (file-locked for parallel SLURM jobs)
# ---------------------------------------------------------------------------

def save_row(row):
    """Append one result row to the output CSV, guarded by a file lock."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lock_path = OUTPUT_CSV + ".lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            file_exists = os.path.isfile(OUTPUT_CSV)
            with open(OUTPUT_CSV, "a", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python exp2_main.py <job_index>  (1 to {len(EXPERIMENTS)})")
        sys.exit(1)

    job_idx = int(sys.argv[1]) - 1  # 1-indexed from SLURM

    if not 0 <= job_idx < len(EXPERIMENTS):
        raise ValueError(
            f"job_index must be between 1 and {len(EXPERIMENTS)}."
        )

    cfg, dataset, seed = EXPERIMENTS[job_idx]

    print(
        f"[Job {job_idx + 1}/{len(EXPERIMENTS)}]  "
        f"block={cfg['block']}  config={cfg['config_label']}  "
        f"dataset={dataset}  seed={seed}"
    )

    row = run_one(cfg, dataset, seed)
    save_row(row)

    print(f"  acc_train={row['acc_train']:.4f}  acc_test={row['acc_test']:.4f}  -> saved")


if __name__ == "__main__":
    main()
