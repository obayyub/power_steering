#!/usr/bin/env python3
"""Generate violin plots for different vector selection criteria."""

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

# Different vector selections
VECTOR_SELECTIONS = {
    "best_survival": {
        "MELBO": 7,
        "Power Iter": 9,
        "Multi-PI": 0,
        "title": "Best on survival-instinct (by magnitude)",
    },
    "best_corrigible": {
        "MELBO": 4,
        "Power Iter": 5,
        "Multi-PI": 2,
        "title": "Best on corrigible-neutral-HHH (by magnitude)",
    },
    "best_both": {
        "MELBO": 9,
        "Power Iter": 9,
        "Multi-PI": 1,
        "title": "Best on Both (robust, by magnitude)",
    },
    "best_linear_survival": {
        "MELBO": 8,
        "Power Iter": 9,
        "Multi-PI": 1,
        "title": "Best on survival-instinct (linear: |slope|×R²)",
    },
    "best_linear_corrigible": {
        "MELBO": 5,
        "Power Iter": 5,
        "Multi-PI": 2,
        "title": "Best on corrigible-neutral-HHH (linear: |slope|×R²)",
    },
    "best_linear_both": {
        "MELBO": 5,
        "Power Iter": 5,
        "Multi-PI": 0,
        "title": "Best on Both (linear: |slope|×R²)",
    },
}


def load_vector_data(filepath: str, vec_idx: int, method: str) -> pd.DataFrame:
    """Load results for a specific vector index."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    results = [
        r for r in data["results"]
        if r["vector_type"] == "steering" and r["vector_idx"] == vec_idx
    ]

    df = pd.DataFrame(results)
    df["method"] = method
    return df


def plot_selection(selection_name: str, selection: dict, output_path: str):
    """Create violin plot for a vector selection."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    datasets = ["survival-instinct", "corrigible-neutral-HHH"]

    for ax, dataset in zip(axes, datasets):
        dfs = []
        for method in ["MELBO", "Power Iter", "Multi-PI"]:
            vec_idx = selection[method]
            filepath = RESULT_FILES[method][dataset]
            df = load_vector_data(filepath, vec_idx, f"{method} (v{vec_idx})")
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        sns.violinplot(
            data=combined,
            x="scale",
            y="survival_logit_diff",
            hue="method",
            ax=ax,
            inner="box",
            cut=0,
            bw_method=0.2,
            density_norm="width",
        )

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

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

        ax.set_xlabel("Steering Scale", fontsize=11)
        ax.set_ylabel("Survival Logit Diff", fontsize=11)
        ax.set_title(f"Dataset: {dataset}", fontsize=12)
        ax.legend(title="Method", loc="upper left", fontsize=9)

    fig.suptitle(
        f"{selection['title']}\n"
        f"MELBO v{selection['MELBO']}, Power Iter v{selection['Power Iter']}, Multi-PI v{selection['Multi-PI']}",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.90)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path("results")

    for name, selection in VECTOR_SELECTIONS.items():
        output_path = output_dir / f"violin_{name}.png"
        plot_selection(name, selection, str(output_path))


if __name__ == "__main__":
    main()
