#!/usr/bin/env python3
"""Generate violin plots for each individual vector."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULT_FILES = {
    "melbo": {
        "survival-instinct": "results/eval_20260127_170043.json",
        "corrigible-neutral-HHH": "results/eval_20260127_170514.json",
    },
    "power_iter": {
        "survival-instinct": "results/eval_20260127_170939.json",
        "corrigible-neutral-HHH": "results/eval_20260127_171412.json",
    },
    "multi_pi": {
        "survival-instinct": "results/eval_20260127_171839.json",
        "corrigible-neutral-HHH": "results/eval_20260127_172307.json",
    },
}


def load_vector_data(filepath: str, vec_idx: int) -> pd.DataFrame:
    """Load results for a specific vector index."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    results = [
        r for r in data["results"]
        if r["vector_type"] == "steering" and r["vector_idx"] == vec_idx
    ]

    return pd.DataFrame(results)


def plot_vector(method: str, vec_idx: int, output_dir: Path):
    """Create violin plot for a single vector."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    datasets = ["survival-instinct", "corrigible-neutral-HHH"]

    for ax, dataset in zip(axes, datasets):
        filepath = RESULT_FILES[method][dataset]
        df = load_vector_data(filepath, vec_idx)

        if df.empty:
            continue

        sns.violinplot(
            data=df,
            x="scale",
            y="survival_logit_diff",
            ax=ax,
            inner="box",
            cut=0,
            color="steelblue",
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

        ax.set_xlabel("Steering Scale", fontsize=11)
        ax.set_ylabel("Survival Logit Diff", fontsize=11)
        ax.set_title(f"Dataset: {dataset}", fontsize=12)

    fig.suptitle(
        f"{method} - Vector {vec_idx}",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.92)

    output_path = output_dir / f"vec_{vec_idx:02d}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    base_dir = Path("results/vector_violins")

    for method in RESULT_FILES:
        output_dir = base_dir / method
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {method}...")
        for vec_idx in range(12):
            plot_vector(method, vec_idx, output_dir)
            print(f"  vec_{vec_idx:02d}.png")

    print(f"\nDone! Plots saved to {base_dir}/")


if __name__ == "__main__":
    main()
