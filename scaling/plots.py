"""
exp3_plots.py -- Experiment 3: qubit scaling plots.

Reads:
    results_exp3/exp3_results.csv

Generates:
    - One 2x2 plot per circuit with train/test accuracy.
    - Test-accuracy comparison of all circuits.
    - Train-accuracy comparison of all circuits.
    - Generalization-gap comparison.

Usage:
    python exp3_plots.py
    python exp3_plots.py --indir results_exp3
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = "results_exp3"

DATASETS = ["moons", "circles", "xor", "checkerboard"]
QUBITS = [2, 4, 8, 16]
CIRCUITS = ["ry_partial", "enc_Y_param_Z", "enc_X_only"]

DATASET_COLORS = {
    "moons": "#E66101",
    "circles": "#B2ABD2",
    "checkerboard": "#5E3C99",
    "xor": "#FDB863",
}

CIRCUIT_COLORS = {
    "ry_partial": "#D6604D",
    "enc_Y_param_Z": "#4393C3",
    "enc_X_only": "#92C5DE",
}

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
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Data loading and aggregation
# ---------------------------------------------------------------------------

def load_data(input_dir):
    """Load and validate Experiment 3 results."""
    path = os.path.join(input_dir, "exp3_results.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)

    df = df[df["circuit"].notna()]
    df = df[df["circuit"].astype(str) != "circuit"]
    df = df[df["n_qubits"].notna()]
    df = df[df["n_qubits"].astype(str) != "n_qubits"]

    df["n_qubits"] = df["n_qubits"].astype(int)

    print(
        f"Loaded {len(df)} rows, "
        f"{df['circuit'].nunique()} circuits, "
        f"{df['n_qubits'].nunique()} qubit counts."
    )

    return df


def aggregate_by_qubits(df, circuit, dataset):
    """Compute mean and standard deviation over seeds."""
    subset = df[
        (df["circuit"] == circuit)
        & (df["dataset"] == dataset)
    ]

    grouped = (
        subset.groupby("n_qubits")[["acc_test", "acc_train"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped.columns = [
        "n_qubits",
        "test_mean",
        "test_std",
        "train_mean",
        "train_std",
    ]

    grouped["test_std"] = grouped["test_std"].fillna(0)
    grouped["train_std"] = grouped["train_std"].fillna(0)

    return grouped.sort_values("n_qubits")


# ---------------------------------------------------------------------------
# Single-circuit plots
# ---------------------------------------------------------------------------

def plot_single_circuit(df, circuit, output_path):
    """Plot train and test accuracy for one circuit across datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    axes = axes.flatten()

    for index, dataset in enumerate(DATASETS):
        ax = axes[index]
        color = DATASET_COLORS[dataset]
        data = aggregate_by_qubits(df, circuit, dataset)

        if data.empty:
            ax.set_title(dataset.capitalize(), fontsize=11, weight="bold")
            continue

        x = data["n_qubits"].values

        ax.plot(
            x,
            data["test_mean"],
            color=color,
            linewidth=2.0,
            linestyle="-",
            label="Test",
            zorder=3,
        )
        ax.fill_between(
            x,
            data["test_mean"] - data["test_std"],
            data["test_mean"] + data["test_std"],
            color=color,
            alpha=0.18,
            zorder=2,
        )

        ax.plot(
            x,
            data["train_mean"],
            color=color,
            linewidth=1.5,
            linestyle="--",
            label="Train",
            zorder=3,
        )
        ax.fill_between(
            x,
            data["train_mean"] - data["train_std"],
            data["train_mean"] + data["train_std"],
            color=color,
            alpha=0.10,
            zorder=2,
        )

        ax.set_xticks(QUBITS)
        ax.set_xlabel("Number of qubits", labelpad=5)
        ax.set_ylabel("Accuracy", labelpad=5)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="#999", linewidth=0.8, linestyle=":")
        ax.set_title(dataset.capitalize(), fontsize=11, weight="bold", pad=6)
        ax.grid(True, linestyle=":", alpha=0.35, color="#aaa")

        if index == 0:
            ax.legend(
                frameon=False,
                fontsize=9,
                loc="lower left",
            )

    fig.suptitle(
        CIRCUIT_LABELS[circuit],
        fontsize=12,
        weight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Circuit comparison
# ---------------------------------------------------------------------------

def plot_comparison(df, accuracy_column, title_suffix, output_path):
    """Compare all circuits for either train or test accuracy."""
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    axes = axes.flatten()

    mean_column = (
        "test_mean" if accuracy_column == "acc_test" else "train_mean"
    )
    std_column = (
        "test_std" if accuracy_column == "acc_test" else "train_std"
    )

    for index, dataset in enumerate(DATASETS):
        ax = axes[index]

        for circuit in CIRCUITS:
            color = CIRCUIT_COLORS[circuit]
            data = aggregate_by_qubits(df, circuit, dataset)

            if data.empty:
                continue

            x = data["n_qubits"].values
            mean = data[mean_column].values
            std = data[std_column].values

            ax.plot(
                x,
                mean,
                color=color,
                linewidth=2.0,
                label=CIRCUIT_LABELS[circuit],
                zorder=3,
            )
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.15,
                zorder=2,
            )

        ax.set_xticks(QUBITS)
        ax.set_xlabel("Number of qubits", labelpad=5)
        ax.set_ylabel(f"{title_suffix} accuracy", labelpad=5)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="#999", linewidth=0.8, linestyle=":")
        ax.set_title(dataset.capitalize(), fontsize=11, weight="bold", pad=6)
        ax.grid(True, linestyle=":", alpha=0.35, color="#aaa")

        if index == 0:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color=CIRCUIT_COLORS[circuit],
                    linewidth=2.0,
                    label=CIRCUIT_LABELS[circuit],
                )
                for circuit in CIRCUITS
            ]
            ax.legend(
                handles=handles,
                frameon=False,
                fontsize=8,
                loc="lower left",
            )

    fig.suptitle(
        f"Qubit scaling — {title_suffix} accuracy",
        fontsize=12,
        weight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Generalization gap
# ---------------------------------------------------------------------------

def plot_gap(df, output_path):
    """Plot train-test accuracy gap for all circuits."""
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    axes = axes.flatten()

    for index, dataset in enumerate(DATASETS):
        ax = axes[index]

        for circuit in CIRCUITS:
            color = CIRCUIT_COLORS[circuit]

            subset = df[
                (df["circuit"] == circuit)
                & (df["dataset"] == dataset)
            ].copy()

            if subset.empty:
                continue

            subset["gap"] = subset["acc_train"] - subset["acc_test"]

            data = (
                subset.groupby("n_qubits")["gap"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("n_qubits")
            )

            data["std"] = data["std"].fillna(0)

            x = data["n_qubits"].values
            mean = data["mean"].values
            std = data["std"].values

            ax.plot(
                x,
                mean,
                color=color,
                linewidth=2.0,
                label=CIRCUIT_LABELS[circuit],
                zorder=3,
            )
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.15,
                zorder=2,
            )

        ax.set_xticks(QUBITS)
        ax.set_xlabel("Number of qubits", labelpad=5)
        ax.set_ylabel("Train − Test accuracy", labelpad=5)
        ax.axhline(0.0, color="#999", linewidth=0.8, linestyle=":")
        ax.set_title(dataset.capitalize(), fontsize=11, weight="bold", pad=6)
        ax.grid(True, linestyle=":", alpha=0.35, color="#aaa")

        if index == 0:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color=CIRCUIT_COLORS[circuit],
                    linewidth=2.0,
                    label=CIRCUIT_LABELS[circuit],
                )
                for circuit in CIRCUITS
            ]
            ax.legend(
                handles=handles,
                frameon=False,
                fontsize=8,
                loc="upper left",
            )

    fig.suptitle(
        "Qubit scaling — generalization gap (train − test)",
        fontsize=12,
        weight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing exp3_results.csv",
    )
    args = parser.parse_args()

    os.makedirs(args.indir, exist_ok=True)

    print(f"Loading data from: {args.indir}")
    df = load_data(args.indir)

    for circuit in CIRCUITS:
        plot_single_circuit(
            df,
            circuit,
            os.path.join(
                args.indir,
                f"plot_exp3_{circuit}.pdf",
            ),
        )

    plot_comparison(
        df,
        "acc_test",
        "Test",
        os.path.join(args.indir, "plot_exp3_comparison_test.pdf"),
    )

    plot_comparison(
        df,
        "acc_train",
        "Train",
        os.path.join(args.indir, "plot_exp3_comparison_train.pdf"),
    )

    plot_gap(
        df,
        os.path.join(args.indir, "plot_exp3_gap.pdf"),
    )

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
