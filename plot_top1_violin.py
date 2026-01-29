#!/usr/bin/env python3
"""Plot violin plots of best vector logit diff across scales for each method."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULT_FILES = {
    "MELBO": {
        "survival-instinct": "results/eval_20260127_170043.json",
        "corrigible-neutral-HHH": "results/eval_20260127_170514.json",
    },
    "Power Iter": {
        "survival-instinct": "results/eval_20260127_170939.json",
        "corrigible-neutral-HHH": "results/eval_20260127_171412.json",
    },
    "Multi-PI": {
        "survival-instinct": "results/eval_20260127_171839.json",
        "corrigible-neutral-HHH": "results/eval_20260127_172307.json",
    },
}

# Best performing vector for each method (manually selected)
BEST_VECTOR_IDX = {
    "MELBO": 5,
    "Power Iter": 1,
    "Multi-PI": 0,
}


def load_best_vector_data(filepath: str, method: str) -> pd.DataFrame:
    """Load results and filter to best performing steering vector."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    best_idx = BEST_VECTOR_IDX[method]

    top1 = [
        r for r in results
        if r["vector_type"] == "steering" and r["vector_idx"] == best_idx
    ]

    df = pd.DataFrame(top1)
    df["method"] = method
    return df


def load_dataset(dataset: str) -> pd.DataFrame:
    """Load data from all methods for a dataset."""
    dfs = []
    for method, files in RESULT_FILES.items():
        filepath = files[dataset]
        if Path(filepath).exists():
            df = load_best_vector_data(filepath, method)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    """Create stacked violin plots for both datasets."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    datasets = ["survival-instinct", "corrigible-neutral-HHH"]

    for ax, dataset in zip(axes, datasets):
        combined = load_dataset(dataset)

        if combined.empty:
            continue

        sns.violinplot(
            data=combined,
            x="scale",
            y="survival_logit_diff",
            hue="method",
            ax=ax,
            inner="box",
            cut=0,
            bw_method=0.2,  # Fixed bandwidth for consistent KDE across groups
            density_norm="width",  # Normalize each violin to same width
        )

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Annotations
        ax.text(
            1.02, 0.75, "← Prefers\n   Survival",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="center", color="red", fontweight="bold",
        )
        ax.text(
            1.02, 0.25, "← Prefers\n   Corrigible",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="center", color="green", fontweight="bold",
        )

        # Arrows
        ax.annotate(
            "", xy=(1.01, 0.85), xytext=(1.01, 0.65),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
        )
        ax.annotate(
            "", xy=(1.01, 0.15), xytext=(1.01, 0.35),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "green", "lw": 2},
        )

        ax.set_xlabel("Steering Scale", fontsize=11)
        ax.set_ylabel("Survival Logit Diff", fontsize=11)
        ax.set_title(f"Dataset: {dataset}", fontsize=12)
        ax.legend(title="Method", loc="upper left")

    fig.suptitle(
        "Best Vector per Method (by avg |change in median|)\n"
        "Positive = prefers survival, Negative = prefers corrigible",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.90)

    output_path = "results/violin_best_vectors_stacked.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
