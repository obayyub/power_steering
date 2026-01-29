#!/usr/bin/env python3
"""
Generate violin plots for steering vector evaluation results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_results(results_file: str) -> tuple[str, dict]:
    """Load results from JSON file."""
    with open(results_file) as f:
        data = json.load(f)
    return data["model"], data["results"]


def compute_survival_logit_diff(results: dict) -> pd.DataFrame:
    """
    Compute survival-direction logit difference for each result.

    survival_logit_diff = logit_survival - logit_corrigible
    Positive = towards survival, Negative = towards corrigibility
    """
    rows = []

    for dataset_name, dataset_results in results.items():
        for r in dataset_results:
            # Determine which logit corresponds to survival vs corrigible
            if r["survival_letter"] == "A":
                survival_logit = r["logit_A"]
                corrigible_logit = r["logit_B"]
            else:
                survival_logit = r["logit_B"]
                corrigible_logit = r["logit_A"]

            survival_logit_diff = survival_logit - corrigible_logit

            rows.append({
                "dataset": dataset_name,
                "vector_type": r["vector_type"],
                "vector_idx": r["vector_idx"],
                "scale": r["scale"],
                "survival_logit_diff": survival_logit_diff,
                "chose_survival": r["chose_survival"],
            })

    return pd.DataFrame(rows)


def plot_violin(df: pd.DataFrame, model_name: str, output_file: str):
    """Create violin plot comparing methods across scales."""

    # Get unique datasets
    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]

    # Color palette for methods
    palette = {"melbo": "#1f77b4", "power_iter": "#ff7f0e", "random": "#2ca02c"}

    for ax, dataset in zip(axes, datasets):
        dataset_df = df[df["dataset"] == dataset]

        # Create violin plot
        sns.violinplot(
            data=dataset_df,
            x="scale",
            y="survival_logit_diff",
            hue="vector_type",
            ax=ax,
            palette=palette,
            inner="box",
            cut=0,
        )

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_xlabel("Scale")
        ax.set_ylabel("Survival Logit Diff")
        ax.legend(title="Method", loc="upper left")

    # Main title
    model_short = model_name.split("/")[-1]
    fig.suptitle(
        f"Survival-Direction Logit Difference ({model_short})\n"
        "(positive = towards survival, negative = towards corrigibility)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/ab_eval_20260126_214856.json",
                        help="Path to results JSON file")
    parser.add_argument("--output", default="results/survival_logit_violin_14B.png",
                        help="Output image path")
    args = parser.parse_args()

    print(f"Loading results from {args.results}...")
    model_name, results = load_results(args.results)

    print(f"Computing survival logit differences...")
    df = compute_survival_logit_diff(results)

    print(f"Generating violin plot...")
    plot_violin(df, model_name, args.output)

    # Print summary statistics
    print("\nSummary by method and scale:")
    summary = df.groupby(["dataset", "vector_type", "scale"]).agg({
        "survival_logit_diff": ["mean", "std"],
        "chose_survival": "mean"
    }).round(2)
    print(summary)


if __name__ == "__main__":
    main()
