"""Experiment 4 — Kernel result plots."""

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


CIRCUIT_COLORS = {
    "no_entanglement": "#2E86AB",
    "zz_linear": "#E84855",
    "zz_full": "#F4A261",
}

CIRCUIT_LABELS = {
    "no_entanglement": "No entanglement",
    "zz_linear": "ZZ linear",
    "zz_full": "ZZ full",
}

CIRCUITS = [
    "no_entanglement",
    "zz_linear",
    "zz_full",
]

DATASET_TITLES = {
    "moons": "Moons",
    "circles": "Circles",
    "xor": "XOR",
    "checkerboard": "Checkerboard",
}

DATASET_ORDER = ["moons", "circles", "xor", "checkerboard"]
LAYER_ORDER = [1, 2, 4, 8]

METRICS = {
    "test_accuracy": {
        "ylabel": "Test accuracy",
        "yscale": "linear",
        "yline": 0.5,
        "label": "Test accuracy",
    },
    "kta": {
        "ylabel": "Centered KTA",
        "yscale": "linear",
        "yline": 0.0,
        "label": "Kernel-Target Alignment",
    },
    "off_diag_variance": {
        "ylabel": "Off-diagonal variance",
        "yscale": "log",
        "yline": None,
        "label": "Off-diagonal variance",
    },
    "effective_dimension": {
        "ylabel": "Effective dimension",
        "yscale": "linear",
        "yline": None,
        "label": "Effective dimension",
    },
    "acc_gap": {
        "ylabel": "Train − test accuracy",
        "yscale": "linear",
        "yline": 0.0,
        "label": "Train–test gap",
    },
    "support_vector_frac": {
        "ylabel": "Support vector fraction",
        "yscale": "linear",
        "yline": 1.0,
        "label": "SV fraction",
    },
}


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def load_results(results_dir):
    paths = sorted(
        glob.glob(os.path.join(results_dir, "job_*.csv"))
    )

    if not paths:
        raise FileNotFoundError(
            f"No result CSVs found in {results_dir}"
        )

    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f"Warning: could not read {path}: {exc}")

    if not frames:
        raise RuntimeError("No readable result files found.")

    data = pd.concat(frames, ignore_index=True)

    numeric_columns = list(METRICS) + [
        "train_accuracy",
        "frobenius_to_id",
        "spectral_ratio",
        "n_qubits",
        "n_layers",
    ]

    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    data = data[data["circuit"].notna()]
    data = data[data["circuit"].astype(str) != "circuit"]

    if "acc_gap" not in data:
        data["acc_gap"] = (
            data["train_accuracy"]
            - data["test_accuracy"]
        )

    return data


def aggregate(data, metric):
    return (
        data.groupby(
            ["circuit", "n_qubits", "n_layers", "dataset"]
        )[metric]
        .mean()
        .reset_index()
    )


def circuit_handles():
    return [
        Line2D(
            [0],
            [0],
            color=CIRCUIT_COLORS[circuit],
            lw=2,
            label=CIRCUIT_LABELS[circuit],
        )
        for circuit in CIRCUITS
    ]


def add_legend(ax, **kwargs):
    return ax.legend(
        handles=circuit_handles(),
        fontsize=8.5,
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        borderpad=0.4,
        handlelength=1.8,
        **kwargs,
    )


def save_figure(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def draw_lines(ax, data, metric, config, qubits, legend=False):
    if config["yline"] is not None:
        ax.axhline(
            config["yline"],
            color="#bbb",
            lw=0.9,
            linestyle="--",
            alpha=0.7,
        )

    for circuit in CIRCUITS:
        subset = (
            data[data["circuit"] == circuit]
            .sort_values("n_qubits")
        )

        if subset.empty:
            continue

        ax.plot(
            subset["n_qubits"],
            subset[metric],
            color=CIRCUIT_COLORS[circuit],
            marker="o",
            lw=1.6,
            markersize=5,
            label=CIRCUIT_LABELS[circuit],
        )

    ax.set_xticks(qubits)
    ax.set_xticklabels([str(int(q)) for q in qubits])
    ax.set_yscale(config["yscale"])
    ax.grid(True, linestyle=":", alpha=0.3, color="#aaa")

    if legend:
        add_legend(ax, loc="best")


def plot_per_layer(data, metric, config, output_dir):
    aggregated = aggregate(data, metric)
    qubits = sorted(data["n_qubits"].dropna().unique())

    for layers in LAYER_ORDER:
        figure, axes = plt.subplots(
            2, 2, figsize=(8.5, 6.5)
        )

        for index, dataset in enumerate(DATASET_ORDER):
            ax = axes.flat[index]
            subset = aggregated[
                (aggregated["n_layers"] == layers)
                & (aggregated["dataset"] == dataset)
            ]

            draw_lines(
                ax,
                subset,
                metric,
                config,
                qubits,
                legend=index == 0,
            )

            ax.set_xlabel("Number of qubits")
            ax.set_ylabel(config["ylabel"])
            ax.set_title(
                DATASET_TITLES[dataset],
                fontweight="bold",
            )

        figure.suptitle(
            f"{config['label']} vs. qubits — L={layers}"
        )
        figure.tight_layout()

        save_figure(
            figure,
            os.path.join(
                output_dir,
                f"ker_A_{metric}_L{layers}.pdf",
            ),
        )


def plot_4x4(data, metric, config, output_dir):
    aggregated = aggregate(data, metric)
    qubits = sorted(data["n_qubits"].dropna().unique())

    figure, axes = plt.subplots(
        4, 4, figsize=(13, 10), sharex=True
    )

    for row, layers in enumerate(LAYER_ORDER):
        for column, dataset in enumerate(DATASET_ORDER):
            ax = axes[row, column]

            subset = aggregated[
                (aggregated["n_layers"] == layers)
                & (aggregated["dataset"] == dataset)
            ]

            draw_lines(
                ax,
                subset,
                metric,
                config,
                qubits,
                legend=row == 0 and column == 0,
            )

            if row == 0:
                ax.set_title(
                    DATASET_TITLES[dataset],
                    fontweight="bold",
                )

            if column == 0:
                ax.set_ylabel(
                    f"L={layers}\n\n{config['ylabel']}"
                )
            else:
                ax.set_ylabel("")

            if row == 3:
                ax.set_xlabel("Number of qubits")
            else:
                ax.set_xlabel("")

    figure.suptitle(
        f"{config['label']} vs. qubits — rows: layers, cols: datasets"
    )
    figure.tight_layout()

    save_figure(
        figure,
        os.path.join(
            output_dir,
            f"ker_B_{metric}_4x4.pdf",
        ),
    )


def plot_kta_vs_layers(data, output_path):
    grouped = (
        data.groupby(["circuit", "n_layers"])["kta"]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0)

    figure, ax = plt.subplots(figsize=(5.5, 4.2))

    ax.axhline(
        0.0, color="#bbb", lw=0.9,
        linestyle="--", alpha=0.7
    )

    for circuit in CIRCUITS:
        subset = grouped[
            grouped["circuit"] == circuit
        ].sort_values("n_layers")

        if subset.empty:
            continue

        color = CIRCUIT_COLORS[circuit]

        ax.plot(
            subset["n_layers"],
            subset["mean"],
            color=color,
            marker="o",
            lw=1.8,
            markersize=6,
            label=CIRCUIT_LABELS[circuit],
        )

        ax.fill_between(
            subset["n_layers"],
            subset["mean"] - subset["std"],
            subset["mean"] + subset["std"],
            color=color,
            alpha=0.12,
        )

    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Centered KTA")
    ax.set_xticks(sorted(data["n_layers"].unique()))
    ax.set_title(
        "KTA vs. layers\n(mean over qubits and datasets)"
    )
    ax.grid(True, linestyle=":", alpha=0.35, color="#aaa")
    ax.legend(fontsize=9)

    figure.tight_layout()
    save_figure(figure, output_path)


def plot_accuracy_vs_kta(data, output_path):
    grouped = (
        data.groupby(["circuit", "n_qubits"])
        [["test_accuracy", "kta"]]
        .mean()
        .reset_index()
    )

    figure, ax = plt.subplots(figsize=(5.5, 4.8))

    ax.axhline(
        0.5,
        color="#bbb",
        lw=0.9,
        linestyle="--",
        alpha=0.7,
    )

    for circuit in CIRCUITS:
        subset = grouped[
            grouped["circuit"] == circuit
        ]

        if subset.empty:
            continue

        color = CIRCUIT_COLORS[circuit]

        ax.scatter(
            subset["kta"],
            subset["test_accuracy"],
            color=color,
            s=65,
            edgecolors="white",
            linewidths=0.5,
            label=CIRCUIT_LABELS[circuit],
        )

        for _, row in subset.iterrows():
            ax.annotate(
                f"{int(row['n_qubits'])}q",
                (row["kta"], row["test_accuracy"]),
                fontsize=7.5,
                color=color,
                xytext=(4, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Centered KTA")
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        "Test accuracy vs. KTA\n"
        "(mean over seeds, layers and datasets)"
    )
    ax.grid(True, linestyle=":", alpha=0.35, color="#aaa")
    ax.legend(fontsize=9)

    figure.tight_layout()
    save_figure(figure, output_path)


def make_plots(results_dir):
    data = load_results(results_dir)

    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    for metric, config in METRICS.items():
        plot_per_layer(data, metric, config, output_dir)
        plot_4x4(data, metric, config, output_dir)

    plot_kta_vs_layers(
        data,
        os.path.join(output_dir, "ker_07_kta_vs_layers.pdf"),
    )
    plot_accuracy_vs_kta(
        data,
        os.path.join(output_dir, "ker_08_acc_vs_kta_scatter.pdf"),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plots.py <results_dir>")
        sys.exit(1)

    make_plots(sys.argv[1])
