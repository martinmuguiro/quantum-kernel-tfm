"""
exp2_plots_refined.py  —  Experiment 2: refined plots with per-dataset heatmaps
                           and per-dataset ranking grid.
================================================================================
Reads results_exp2/exp2_results.csv and generates:

  - Heatmaps (2x2 grid, one per dataset) for Block 1 (rotation axis ablation)
      → exp2_heatmap_test_per_dataset.pdf
      → exp2_heatmap_train_per_dataset.pdf

  - Global ranking: 2x2 grid (one subplot per dataset) with horizontal bars
    showing all configurations, colored by block.
      → exp2_ranking_per_dataset.pdf

  - Encoding-only comparison (Block 1)
      → exp2_encoding_only.pdf

  - Entanglement ablation (Block 2) 2x2 grid per dataset, test and train
      → exp2_entanglement_test.pdf
      → exp2_entanglement_train.pdf

  - Accuracy tables (CSV and PDF)
      → exp2_accuracy_table.csv
      → exp2_accuracy_table.pdf

Usage:
    python exp2_plots_refined.py                        # uses results_exp2/
    python exp2_plots_refined.py --indir my_results/    # custom input dir
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
# ============================================================
# STYLE (matches original)
# ============================================================
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

DATASET_COLORS = {
    "moons": "#E66101",
    "circles": "#B2ABD2",
    "checkerboard": "#5E3C99",
    "xor": "#FDB863",
}

DATASETS = ["moons", "circles", "xor", "checkerboard"]
AXES = ["X", "Y", "Z"]
ENT_LEVELS = [
    "no_ent",
    "partial",
    "full",
]

ENT_LABELS = {
    "no_ent": "No entanglement",
    "partial": "Partial",
    "full": "Full",
}
BLOCK_COLORS = {1: "#4393C3", 2: "#D6604D"}
ENC_ONLY_COLOR = "#92C5DE"

# ============================================================
# DATA LOADING
# ============================================================
def load_data(indir):
    path = os.path.join(indir, "exp2_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_csv(path)
    df = df[df["config_label"].notna()]
    df = df[df["config_label"].astype(str) != "config_label"]
    df["block"] = df["block"].astype(int)
    df["config_label"] = df["config_label"].astype(str)
    # Remove the old ry_* rows if present (they are not used)
    df = df[~df["config_label"].isin(["ry_no_ent", "ry_partial", "ry_full"])]
    print(f"Loaded {len(df)} valid rows from {path}")
    return df

def mean_std(df, group_cols, val_col):
    g = df.groupby(group_cols)[val_col].agg(["mean", "std"]).reset_index()
    g.columns = group_cols + ["mean", "std"]
    g["std"] = g["std"].fillna(0)
    return g

# ============================================================
# BLOCK 1: HEATMAP PER DATASET (2x2 grid, grayscale)
# ============================================================

def plot_heatmaps_per_dataset(df, acc_col, title_str, out_path):
    """
    2x2 heatmaps (dataset-wise), improved readability version.
    """

    b1 = df[(df["block"] == 1) & (~df["config_label"].str.endswith("_only"))]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.6))
    axes = axes.flatten()

    vmin, vmax = 0.5, 1.0
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        ds_df = b1[b1["dataset"] == ds]

        agg = ds_df.groupby("config_label")[acc_col].mean().reset_index()

        mat = np.full((3, 3), np.nan)

        for _, row in agg.iterrows():
            parts = row["config_label"].split("_")
            da = parts[1]
            pa = parts[3]
            r = AXES.index(da)
            c = AXES.index(pa)
            mat[r, c] = row[acc_col]

        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")

        # Axes labels
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"$R_{{{a}}}(\\theta)$" for a in AXES], fontsize=9)

        ax.set_yticks(range(3))
        ax.set_yticklabels([f"$R_{{{a}}}(x)$" for a in AXES], fontsize=9)

        ax.tick_params(length=0)

        if idx in [0, 2]:
            ax.set_ylabel("Encoding gate", labelpad=8)
        if idx in [2, 3]:
            ax.set_xlabel("Structural gate", labelpad=8)

        ax.set_title(ds.capitalize(), fontsize=11, pad=6)

        # ─────────────────────────────────────────────
        # Annotated values (robust + publication style)
        # ─────────────────────────────────────────────
        for r in range(3):
            for c in range(3):
                v = mat[r, c]
                if np.isnan(v):
                    continue

                rgba = cmap(norm(v))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]

                # Base text color (kept readable)
                text_color = "white" if luminance < 0.55 else "black"

                # Strong readability via stroke + subtle box
                ax.text(
                    c, r, f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="semibold",
                    color=text_color,
                    path_effects=[
                        pe.withStroke(linewidth=2.8, foreground="black" if text_color == "white" else "white")
                    ],
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.25
                    )
                )

    # ─────────────────────────────────────────────
    # Colorbar (clean + aligned)
    # ─────────────────────────────────────────────
    fig.subplots_adjust(right=0.88)

    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)

    cbar.set_label(f"Mean {acc_col.replace('_', ' ')}", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Title
    fig.suptitle(title_str, fontsize=13, y=0.98)

    # Layout tuning
    fig.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.08,
        top=0.92,
        wspace=0.25,
        hspace=0.28
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")

# ============================================================
# BLOCK 1: ENCODING-ONLY vs AXIS COMBOS (per dataset)
# ============================================================
def plot_encoding_only(df, out_path):
    enc_only = df[df["config_label"].str.endswith("_only")]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        eo_ds = enc_only[enc_only["dataset"] == ds]
        eo_agg = eo_ds.groupby("config_label")["acc_test"].agg(["mean", "std"]).reset_index()
        eo_agg["std"] = eo_agg["std"].fillna(0)

        labels = [f"$R_X(x)$ only", f"$R_Y(x)$ only", f"$R_Z(x)$ only"]
        means  = list(eo_agg["mean"])
        stds   = list(eo_agg["std"])
        colors = [ENC_ONLY_COLOR] * 3

        ax.bar(labels, means, yerr=stds, color=colors, capsize=4, width=0.55, alpha=0.85)
        ax.set_ylim(0.4, 1.05)
        ax.set_title(ds.capitalize(), fontsize=10, weight="normal")
        ax.set_ylabel("Test accuracy")
        ax.tick_params(axis="x", labelsize=9)
        ax.axhline(0.5, color="#999", lw=0.8, linestyle=":")

    fig.suptitle("Encoding‑only circuits", fontsize=12, weight="normal", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# ============================================================
# BLOCK 2: ENTANGLEMENT ABLATION (2x2 grid, test/train)
# ============================================================
def plot_entanglement(df, acc_col, title_suffix, out_path):
    b2 = df[df["block"] == 2]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    axes = axes.flatten()
    x = np.arange(len(ENT_LEVELS))
    width = 0.55

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        ds_df = b2[b2["dataset"] == ds]
        agg = mean_std(ds_df, ["config_label"], acc_col)

        means = []
        stds = []
        for el in ENT_LEVELS:
            lbl = f"ry_{el}"
            row = agg[agg["config_label"] == lbl]
            if len(row):
                means.append(row["mean"].iloc[0])
                stds.append(row["std"].iloc[0])
            else:
                means.append(np.nan)
                stds.append(0)

        ax.bar(x, means, yerr=stds, color=DATASET_COLORS[ds], capsize=4, width=width, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([ENT_LABELS[e] for e in ENT_LEVELS], fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0.4, 1.05)
        ax.set_title(ds.capitalize(), fontsize=10, weight="normal")
        ax.set_ylabel(f"{title_suffix} accuracy")
        ax.axhline(0.5, color="#999", lw=0.8, linestyle=":")

    fig.suptitle(f"Entanglement ablation — {title_suffix} accuracy",
                 fontsize=12, weight="normal", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# ============================================================
# GLOBAL RANKING: 2x2 grid, one per dataset, horizontal bars
# ============================================================
def plot_ranking_per_dataset(df, out_path):
    """
    For each dataset: horizontal bar chart of all configurations,
    sorted by test accuracy descending, colored by block.
    """
    # Prepare data: for each dataset, compute mean test accuracy per config
    # (over seeds)
    df_agg = df.groupby(["dataset", "config_label", "block"])["acc_test"].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        sub = df_agg[df_agg["dataset"] == ds].copy()
        sub = sub[sub["config_label"] != "ry_no_ent_clean"]
        # Sort descending for nicer plotting (top = best at top of bars)
        sub = sub.sort_values("acc_test", ascending=True)  # because barh uses bottom-to-top

        # Colors
        def row_color(row):
            if str(row["config_label"]).endswith("_only"):
                return ENC_ONLY_COLOR
            return BLOCK_COLORS[int(row["block"])]

        colors = [row_color(r) for _, r in sub.iterrows()]

        # Labels: clean them for display
        labels = [str(lbl).replace("_clean", "").replace("ry_", "") for lbl in sub["config_label"]]

        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub["acc_test"], color=colors, alpha=0.85, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0.4, 1.02)
        ax.set_xlabel("Test accuracy")
        ax.set_title(ds.capitalize(), fontsize=10, weight="normal")
        ax.axvline(0.5, color="#999", lw=0.8, linestyle=":")

        # Add value labels on bars
        for i, (_, row) in enumerate(sub.iterrows()):
            acc = row["acc_test"]
            if acc > 0.95:
                x_text = acc - 0.03
                ha = "right"
            else:
                x_text = acc + 0.01
                ha = "left"
            ax.text(x_text, i, f"{acc:.3f}", va="center", fontsize=6, ha=ha)

    # Common legend
    legend_patches = [
        mpatches.Patch(color=ENC_ONLY_COLOR, alpha=0.85, label="Encoding only (Block 1)"),
        mpatches.Patch(color=BLOCK_COLORS[1], alpha=0.85, label="Rotation axis combinations (Block 1)"),
        mpatches.Patch(color=BLOCK_COLORS[2], alpha=0.85, label="Entanglement levels (Block 2)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Global ranking of all configurations – per dataset", fontsize=12, weight="normal", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# ============================================================
# ACCURACY TABLE (CSV + PDF)
# ============================================================
def save_accuracy_table(df, out_csv, out_pdf):
    agg_test = df.groupby(["config_label", "block"])["acc_test"].agg(
        mean_acc_test="mean", std_acc_test="std").reset_index()
    agg_train = df.groupby(["config_label", "block"])["acc_train"].agg(
        mean_acc_train="mean", std_acc_train="std").reset_index()
    tbl = agg_test.merge(agg_train, on=["config_label", "block"])
    tbl["std_acc_test"] = tbl["std_acc_test"].fillna(0)
    tbl["std_acc_train"] = tbl["std_acc_train"].fillna(0)
    tbl = tbl.sort_values("mean_acc_test", ascending=False).reset_index(drop=True)
    for col in ["mean_acc_test", "std_acc_test", "mean_acc_train", "std_acc_train"]:
        tbl[col] = tbl[col].round(4)

    tbl.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # PDF table
    n_rows = len(tbl)
    fig_h = max(3.5, 0.38 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    ax.axis("off")
    col_labels = ["Config", "Block", "Test mean", "Test std", "Train mean", "Train std"]
    cell_data = [
        [row["config_label"], int(row["block"]),
         f"{row['mean_acc_test']:.4f}", f"{row['std_acc_test']:.4f}",
         f"{row['mean_acc_train']:.4f}", f"{row['std_acc_train']:.4f}"]
        for _, row in tbl.iterrows()
    ]
    table = ax.table(cellText=cell_data, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4393C3")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, n_rows + 1):
        fc = "#F0F4F8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(fc)
    ax.set_title("Experiment 2 – Accuracy summary (sorted by test accuracy)",
                 fontsize=10, weight="normal", pad=8)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved: {out_pdf}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="results_exp2",
                        help="Directory containing exp2_results.csv")
    args = parser.parse_args()

    outdir = args.indir
    os.makedirs(outdir, exist_ok=True)

    df = load_data(args.indir)

    # Block 1: heatmaps per dataset (grayscale 2x2)
    plot_heatmaps_per_dataset(df, "acc_test",
                              "Rotation axis ablation – test accuracy per dataset",
                              os.path.join(outdir, "exp2_heatmap_test_per_dataset.pdf"))
    plot_heatmaps_per_dataset(df, "acc_train",
                              "Rotation axis ablation – train accuracy per dataset",
                              os.path.join(outdir, "exp2_heatmap_train_per_dataset.pdf"))

    # Block 1: encoding-only comparison
    plot_encoding_only(df, os.path.join(outdir, "exp2_encoding_only.pdf"))

    # Block 2: entanglement
    plot_entanglement(df, "acc_test", "Test",
                      os.path.join(outdir, "exp2_entanglement_test.pdf"))
    plot_entanglement(df, "acc_train", "Train",
                      os.path.join(outdir, "exp2_entanglement_train.pdf"))

    # Global ranking per dataset (2x2 grid)
    plot_ranking_per_dataset(df, os.path.join(outdir, "exp2_ranking_per_dataset.pdf"))

    # Tables
    save_accuracy_table(df,
                        os.path.join(outdir, "exp2_accuracy_table.csv"),
                        os.path.join(outdir, "exp2_accuracy_table.pdf"))

    print("\nAll plots and tables saved successfully.")
