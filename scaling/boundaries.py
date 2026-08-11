```python
"""
exp3_boundaries.py -- Experiment 3: decision boundary plots.

Generates one decision-boundary plot for each combination of:

    - Dataset: checkerboard
    - Circuits: the three circuits selected in Experiment 3
    - Qubits: 2, 4, 8, 16

Total jobs: 3 circuits × 4 qubit counts = 12.

Usage:
    python exp3_boundaries.py <job_index>
    python exp3_boundaries.py --list

Job indices are 0-based.

Output:
    results_exp3/boundaries_checkerboard/
"""
    
import argparse
import os
import itertools

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from exp3_main import (
    CIRCUITS,
    RANDOM_STATE,
    TEST_SIZE,
    build_circuit,
    load_dataset,
    n_params,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET = "checkerboard"
DATASET_LABEL = "Checkerboard"

QUBIT_COUNTS = [2, 4, 8, 16]
BOUNDARY_SEED = 0
GRID_RESOLUTION = 60
GRID_PADDING = 0.35

OUTPUT_DIR = os.path.join(
    "results_exp3",
    "boundaries_checkerboard",
)

CIRCUIT_LABELS = {
    "ry_partial": r"$R_Y$ — partial entanglement",
    "enc_Y_param_Z": r"$R_Y(x)$–$R_Z(\theta)$–$R_Y(x)$",
    "enc_X_only": r"$R_X(x)$ only",
}


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

CLASS_0_COLOR = "#E84855"
CLASS_1_COLOR = "#2E86AB"
BOUNDARY_COLOR = "#1A1A2E"
REGION_ALPHA = 0.13


# ---------------------------------------------------------------------------
# Experiment jobs
# ---------------------------------------------------------------------------

def build_job_list():
    """Return all circuit × qubit-count combinations."""
    return [
        {
            "circuit": circuit,
            "n_qubits": n_qubits,
        }
        for circuit, n_qubits in itertools.product(
            CIRCUITS,
            QUBIT_COUNTS,
        )
    ]


# ---------------------------------------------------------------------------
# Quantum kernel
# ---------------------------------------------------------------------------

def build_kernel(circuit, n_qubits, params):
    """Build a callable that evaluates the quantum kernel."""
    device = qml.device("lightning.qubit", wires=n_qubits)

    def ansatz(x, structural_params):
        build_circuit(
            x,
            structural_params,
            circuit,
            n_qubits,
        )

    @qml.qnode(device)
    def kernel_circuit(x1, x2):
        ansatz(x1, params)
        qml.adjoint(ansatz)(x2, params)
        return qml.probs(wires=range(n_qubits))

    def kernel_matrix(X1, X2):
        n1, n2 = len(X1), len(X2)
        kernel = np.zeros((n1, n2))

        if n1 == n2 and X1 is X2:
            for i in range(n1):
                kernel[i, i] = float(
                    kernel_circuit(X1[i], X1[i])[0]
                )

                for j in range(i + 1, n1):
                    value = float(
                        kernel_circuit(X1[i], X1[j])[0]
                    )
                    kernel[i, j] = value
                    kernel[j, i] = value
        else:
            for i, x1 in enumerate(X1):
                for j, x2 in enumerate(X2):
                    kernel[i, j] = float(
                        kernel_circuit(x1, x2)[0]
                    )

        return kernel

    return kernel_matrix


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def plot_boundary(job):
    """Train the SVM and generate one decision-boundary figure."""
    circuit = job["circuit"]
    n_qubits = job["n_qubits"]

    print(
        f"Dataset: {DATASET_LABEL} | "
        f"Circuit: {CIRCUIT_LABELS[circuit]} | "
        f"{n_qubits} qubits",
        flush=True,
    )

    X, y = load_dataset(DATASET)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Use seed 0 to select a representative parameter set.
    n_parameters = n_params(circuit, n_qubits)

    if n_parameters == 0:
        params = None
    else:
        rng = np.random.default_rng(BOUNDARY_SEED)
        params = rng.uniform(
            -np.pi,
            np.pi,
            size=n_parameters,
        )

    kernel = build_kernel(
        circuit,
        n_qubits,
        params,
    )

    print("  Computing training kernel...", flush=True)
    K_train = kernel(X_train, X_train)

    print("  Computing test kernel...", flush=True)
    K_test = kernel(X_test, X_train)

    classifier = SVC(
        kernel="precomputed",
        C=1.0,
    )
    classifier.fit(K_train, y_train)

    train_accuracy = accuracy_score(
        y_train,
        classifier.predict(K_train),
    )
    test_accuracy = accuracy_score(
        y_test,
        classifier.predict(K_test),
    )

    print(
        f"  train={train_accuracy:.3f} | "
        f"test={test_accuracy:.3f} | "
        f"SV={len(classifier.support_)}/{len(X_train)}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Decision grid
    # -----------------------------------------------------------------------

    x_min = X_train[:, 0].min() - GRID_PADDING
    x_max = X_train[:, 0].max() + GRID_PADDING
    y_min = X_train[:, 1].min() - GRID_PADDING
    y_max = X_train[:, 1].max() + GRID_PADDING

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, GRID_RESOLUTION),
        np.linspace(y_min, y_max, GRID_RESOLUTION),
    )

    X_grid = np.c_[xx.ravel(), yy.ravel()]

    print(
        f"  Computing grid kernel "
        f"({len(X_grid)} × {len(X_train)})...",
        flush=True,
    )

    K_grid = kernel(X_grid, X_train)

    predictions = classifier.predict(K_grid)
    Z = predictions.reshape(xx.shape).astype(float)

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    ax.contourf(
        xx,
        yy,
        Z,
        levels=[-999, 0.5],
        colors=[CLASS_0_COLOR],
        alpha=REGION_ALPHA,
    )

    ax.contourf(
        xx,
        yy,
        Z,
        levels=[0.5, 999],
        colors=[CLASS_1_COLOR],
        alpha=REGION_ALPHA,
    )

    try:
        ax.contour(
            xx,
            yy,
            Z,
            levels=[0.5],
            colors=[BOUNDARY_COLOR],
            linewidths=1.8,
        )
    except ValueError:
        # No boundary exists when the classifier predicts a single class
        # throughout the plotted region.
        pass

    ax.scatter(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        c=CLASS_0_COLOR,
        s=38,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )

    ax.scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        c=CLASS_1_COLOR,
        s=38,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        f"{DATASET_LABEL} · "
        f"{CIRCUIT_LABELS[circuit]} · "
        f"{n_qubits} qubits",
        fontsize=13,
        pad=10,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLASS_1_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
            label="Class 1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLASS_0_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
            label="Class 0",
        ),
        Line2D(
            [0],
            [0],
            color=BOUNDARY_COLOR,
            linewidth=1.8,
            label="Decision boundary",
        ),
    ]

    legend = ax.legend(
        handles=legend_handles,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.97),
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        borderpad=0.4,
        handlelength=1.8,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#DDDDDD")
    legend.get_frame().set_linewidth(0.8)

    ax.text(
        0.97,
        0.04,
        f"train {train_accuracy:.2f}   "
        f"test {test_accuracy:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#DDDDDD",
            "alpha": 0.92,
        },
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"boundary_{DATASET}_{circuit}_{n_qubits}q.pdf",
    )

    fig.savefig(output_path)
    plt.close(fig)

    print(f"  Saved: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate an Experiment 3 decision-boundary plot."
    )

    parser.add_argument(
        "job_index",
        nargs="?",
        type=int,
        help="0-based job index.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available jobs.",
    )

    args = parser.parse_args()
    jobs = build_job_list()

    if args.list:
        print(f"{'IDX':>5}  {'CIRCUIT':<22} {'QUBITS':>6}")

        for index, job in enumerate(jobs):
            print(
                f"{index:>5}  "
                f"{job['circuit']:<22} "
                f"{job['n_qubits']:>6}"
            )

        print(f"\nTotal jobs: {len(jobs)}")
        return

    if args.job_index is None:
        parser.error(
            "Provide a job index or use --list."
        )

    if not 0 <= args.job_index < len(jobs):
        parser.error(
            f"job_index must be between 0 and {len(jobs) - 1}."
        )

    plot_boundary(jobs[args.job_index])


if __name__ == "__main__":
    main()
```
