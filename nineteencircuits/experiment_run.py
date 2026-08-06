import fcntl
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from loader import load_dataset
from kernel import quantum_kernel, quantum_kernel_cross
from svm import train_and_evaluate_svm

from nineteencircuits.circuits import CIRCUITS


LAYER_VALUES = [1, 2, 3, 5]
SEEDS = range(20, 30)


def run_single_configuration(
    dataset,
    config,
    n_layers,
    n_samples,
    use_gpu,
    seeds,
):
    """
    Evaluate a circuit configuration over multiple random seeds.
    """

    x, y = load_dataset(dataset, n_samples)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
    )

    n_qubits = 4
    accuracies = []

    for seed in seeds:
        np.random.seed(seed)

        k_train = quantum_kernel(
            x_train,
            config,
            n_qubits,
            n_layers,
            gpu=use_gpu,
        )

        k_test = quantum_kernel_cross(
            x_test,
            x_train,
            config,
            n_qubits,
            n_layers,
            gpu=use_gpu,
        )

        accuracies.append(
            train_and_evaluate_svm(
                k_train,
                y_train,
                k_test,
                y_test,
            )
        )

    return accuracies


def append_results(
    dataset,
    circuit_id,
    config,
    n_layers,
    accuracies,
    use_gpu,
):
    """
    Append experiment results to the output CSV.
    """

    os.makedirs("results", exist_ok=True)

    suffix = "_gpu" if use_gpu else "_cpu"

    csv_path = f"results/{dataset}_results{suffix}.csv"
    lock_path = f"{csv_path}.lock"

    rows = [
        {
            "circuit_num": circuit_id,
            "config": config,
            "layers": n_layers,
            "seed": seed,
            "accuracy": accuracy,
        }
        for seed, accuracy in zip(SEEDS, accuracies)
    ]

    dataframe = pd.DataFrame(rows)

    with open(lock_path, "w") as lock_file:

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        try:
            file_exists = os.path.exists(csv_path)

            with open(csv_path, "a") as output:
                dataframe.to_csv(
                    output,
                    header=not file_exists,
                    index=False,
                )

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    print(
        f"[OK] circuit={circuit_id:02d} | "
        f"layers={n_layers} | "
        f"mean={np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}"
    )


def main():

    if len(sys.argv) < 3:
        print(
            "Usage: python run_experiments.py "
            "<dataset> <experiment_idx> [gpu] [n_samples]"
        )
        sys.exit(1)

    dataset = sys.argv[1]
    experiment_idx = int(sys.argv[2])

    use_gpu = (
        len(sys.argv) > 3
        and sys.argv[3].lower() in {"true", "1", "yes"}
    )

    n_samples = (
        int(sys.argv[4])
        if len(sys.argv) > 4
        else 200
    )

    total_experiments = len(CIRCUITS) * len(LAYER_VALUES)

    if not 1 <= experiment_idx <= total_experiments:
        raise ValueError(
            f"experiment_idx must be between "
            f"1 and {total_experiments}."
        )

    experiment = experiment_idx - 1

    circuit_index = experiment // len(LAYER_VALUES)
    layer_index = experiment % len(LAYER_VALUES)

    circuit = CIRCUITS[circuit_index]
    n_layers = LAYER_VALUES[layer_index]

    print(f"Dataset     : {dataset}")
    print(f"Experiment  : {experiment_idx}/{total_experiments}")
    print(f"Circuit     : {circuit_index + 1}")
    print(f"Layers      : {n_layers}")
    print(f"GPU         : {use_gpu}")
    print(f"Seeds       : {len(SEEDS)}")
    print()

    accuracies = run_single_configuration(
        dataset,
        circuit,
        n_layers,
        n_samples,
        use_gpu,
        SEEDS,
    )

    append_results(
        dataset,
        circuit_index + 1,
        circuit,
        n_layers,
        accuracies,
        use_gpu,
    )


if __name__ == "__main__":
    main()
