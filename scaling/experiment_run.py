"""
exp3_main.py -- Experiment 3: qubit scaling study.

Evaluates three selected circuits across 2, 4, 8 and 16 qubits,
four datasets and 20 random seeds.

Selected circuits (best per block from Experiment 2):
    ry_partial    : RY(x) -> CRZ(0-1) -> CRZ(2-3) -> ... -> RY(x)  [Block 2 best]
    enc_Y_param_Z : RY(x) -> RZ(theta) -> RY(x)                     [Block 1b best]
    enc_X_only    : RX(x)                                           [Block 1a best]

enc_X_only has no structural parameters, so its circuit does not depend
on seed -- it is evaluated once per (n_qubits, dataset) instead of once
per (n_qubits, dataset, seed).

Total jobs: see len(EXPERIMENTS)
    ry_partial, enc_Y_param_Z : 2 circuits x 4 qubit counts x 4 datasets x 20 seeds = 640
    enc_X_only                : 1 circuit  x 4 qubit counts x 4 datasets x 1 seed   =  16
    Total                                                                          = 656

Datasets are loaded and pi-normalized via loader.load_dataset (shared
with Experiment 1), so rotation-gate encoding angles stay in [-pi, pi].

Usage (SLURM array 1..len(EXPERIMENTS)):
    python exp3_main.py <job_index>

Output:
    results_exp3/exp3_results.csv
"""

import csv
import fcntl
import os
import sys

import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from loader import load_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SEEDS = 20
TEST_SIZE = 0.30
RANDOM_STATE = 42

OUTPUT_DIR = "results_exp3"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "exp3_results.csv")
CSV_FIELDS = [
    "circuit", "n_qubits", "dataset", "seed",
    "acc_train", "acc_test",
]

QUBIT_COUNTS = [2, 4, 8, 16]
DATASETS = ["moons", "circles", "xor", "checkerboard"]
CIRCUITS = ["ry_partial", "enc_Y_param_Z", "enc_X_only"]
N_SAMPLES = 400


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def build_circuit(x, params, circuit, n_qubits):
    """Build one of the three selected circuits.

    Encoding convention: qubit i encodes x[i % 2].
    params: 1D array of structural angles, or None for encoding-only.
    """
    if circuit == "enc_X_only":
        # RX(x) -- single encoding gate, no structural params
        for wire in range(n_qubits):
            qml.RX(x[wire % 2], wires=wire)

    elif circuit == "enc_Y_param_Z":
        # RY(x) -> RZ(theta) -> RY(x)
        for wire in range(n_qubits):
            qml.RY(x[wire % 2], wires=wire)
        for wire in range(n_qubits):
            qml.RZ(params[wire], wires=wire)
        for wire in range(n_qubits):
            qml.RY(x[wire % 2], wires=wire)

    elif circuit == "ry_partial":
        # RY(x) -> CRZ(0-1) -> CRZ(2-3) -> ... -> RY(x)
        # For n_qubits > 4: independent pairs (0-1), (2-3), (4-5), ...
        for wire in range(n_qubits):
            qml.RY(x[wire % 2], wires=wire)
        for wire in range(0, n_qubits - 1, 2):
            qml.CRZ(params[wire], wires=[wire, wire + 1])
        for wire in range(n_qubits):
            qml.RY(x[wire % 2], wires=wire)

    else:
        raise ValueError(f"Unknown circuit: {circuit}")


def n_params(circuit, n_qubits):
    """Number of structural parameters needed (0 for encoding-only)."""
    if circuit == "enc_X_only":
        return 0
    if circuit == "enc_Y_param_Z":
        return n_qubits
    if circuit == "ry_partial":
        return n_qubits  # only even-indexed entries are used, see build_circuit
    raise ValueError(circuit)


def get_circuit_fn(circuit, n_qubits):
    """Return a callable build_fn(x, params) for the given circuit."""
    return lambda x, p: build_circuit(x, p, circuit, n_qubits)


# ---------------------------------------------------------------------------
# Experiment list (1-indexed for SLURM)
# ---------------------------------------------------------------------------

def _build_experiments():
    """Flatten (circuit, n_qubits, dataset, seed) combinations.

    Circuits with no structural parameters (n_params == 0) do not depend
    on seed, so they are evaluated once per (n_qubits, dataset) instead
    of once per (n_qubits, dataset, seed).
    """
    experiments = []
    for circuit in CIRCUITS:
        for n_qubits in QUBIT_COUNTS:
            for dataset in DATASETS:
                if n_params(circuit, n_qubits) == 0:
                    experiments.append({
                        "circuit": circuit,
                        "n_qubits": n_qubits,
                        "dataset": dataset,
                        "seed": 0,
                    })
                else:
                    for seed in range(N_SEEDS):
                        experiments.append({
                            "circuit": circuit,
                            "n_qubits": n_qubits,
                            "dataset": dataset,
                            "seed": seed,
                        })
    return experiments


EXPERIMENTS = _build_experiments()


# ---------------------------------------------------------------------------
# Kernel computation
# ---------------------------------------------------------------------------

def compute_kernels(x_train, x_test, build_fn, n_qubits, params):
    """Compute the train-train and test-train quantum kernel matrices."""
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        build_fn(x1, params)
        qml.adjoint(build_fn)(x2, params)
        return qml.probs(wires=range(n_qubits))

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

def run_one(exp):
    """Run one experiment dict and return its result row."""
    circuit = exp["circuit"]
    n_qubits = exp["n_qubits"]
    dataset = exp["dataset"]
    seed = exp["seed"]

    x, y = load_dataset(dataset, n_samples=N_SAMPLES)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    n_structural_params = n_params(circuit, n_qubits)
    if n_structural_params == 0:
        params = None
    else:
        rng = np.random.default_rng(seed)
        params = rng.uniform(-np.pi, np.pi, size=(n_structural_params,))

    build_fn = get_circuit_fn(circuit, n_qubits)
    k_train, k_test = compute_kernels(x_train, x_test, build_fn, n_qubits, params)

    classifier = SVC(kernel="precomputed", C=1.0)
    classifier.fit(k_train, y_train)

    return {
        "circuit": circuit,
        "n_qubits": n_qubits,
        "dataset": dataset,
        "seed": seed,
        "acc_train": round(float(classifier.score(k_train, y_train)), 6),
        "acc_test": round(float(classifier.score(k_test, y_test)), 6),
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
        print(f"Usage: python exp3_main.py <job_index>  (1 to {len(EXPERIMENTS)})")
        sys.exit(1)

    job_idx = int(sys.argv[1]) - 1  # 1-indexed from SLURM

    if not 0 <= job_idx < len(EXPERIMENTS):
        raise ValueError(
            f"job_index must be between 1 and {len(EXPERIMENTS)}."
        )

    exp = EXPERIMENTS[job_idx]

    print(
        f"[Job {job_idx + 1}/{len(EXPERIMENTS)}]  "
        f"circuit={exp['circuit']}  n_qubits={exp['n_qubits']}  "
        f"dataset={exp['dataset']}  seed={exp['seed']}"
    )

    row = run_one(exp)
    save_row(row)

    print(f"  acc_train={row['acc_train']:.4f}  acc_test={row['acc_test']:.4f}  -> saved")


if __name__ == "__main__":
    main()
