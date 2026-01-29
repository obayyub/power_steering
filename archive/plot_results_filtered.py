#!/usr/bin/env python3
"""
Generate violin plots using only top-performing vectors from each method.
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
    """Compute survival-direction logit difference for each result."""
    rows = []

    for dataset_name, dataset_results in results.items():
        for r in dataset_results:
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


def find_top_vectors(df: pd.DataFrame, vector_type: str, n: int = 3) -> list[int]:
    """Find top n vectors by max effect (deviation from baseline) at any scale."""
    vtype_df = df[(df["vector_type"] == vector_type) & (df["dataset"] == "survival-instinct")]

    # Get baseline (scale=0) survival rate
    baseline = vtype_df[vtype_df["scale"] == 0]["chose_survival"].mean()

    # For each vector, find max absolute deviation from baseline
    def max_effect(group):
        rates = group.groupby("scale")["chose_survival"].mean()
        return (rates - baseline).abs().max()

    effects = vtype_df.groupby("vector_idx").apply(max_effect, include_groups=False)

    # Return top n by effect size
    return effects.nlargest(n).index.tolist()


def plot_violin_filtered(df: pd.DataFrame, model_name: str, output_file: str, top_n: int = 3):
    """Create violin plot using only top-performing vectors."""

    # Find top vectors for each method
    top_melbo = find_top_vectors(df, "melbo", top_n)
    top_power = find_top_vectors(df, "power_iter", top_n)
    top_random = find_top_vectors(df, "random", top_n)

    print(f"Top {top_n} MELBO vectors: {top_melbo}")
    print(f"Top {top_n} Power Iter vectors: {top_power}")
    print(f"Top {top_n} Random vectors: {top_random}")

    # Filter dataframe
    df_filtered = df[
        ((df["vector_type"] == "melbo") & (df["vector_idx"].isin(top_melbo))) |
        ((df["vector_type"] == "power_iter") & (df["vector_idx"].isin(top_power))) |
        ((df["vector_type"] == "random") & (df["vector_idx"].isin(top_random)))
    ].copy()

    # Get unique datasets
    datasets = df_filtered["dataset"].unique()

    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 4.5 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]

    palette = {"melbo": "#1f77b4", "power_iter": "#ff7f0e", "random": "#2ca02c"}

    for ax, dataset in zip(axes, datasets):
        dataset_df = df_filtered[df_filtered["dataset"] == dataset]

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

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=1.5)

        # Dataset descriptions
        descriptions = {
            "survival-instinct": "self-preservation, resisting shutdown",
            "corrigible-neutral-HHH": "accepting correction, following instructions",
        }
        desc = descriptions.get(dataset, "")
        ax.set_title(f"{dataset}\n({desc})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Scale")
        ax.set_ylabel("logit(survival) − logit(corrigible)")
        ax.legend(title="Method", loc="upper left")

        # Add annotations for clarity
        ylim = ax.get_ylim()
        ax.text(ax.get_xlim()[1] + 0.3, ylim[1] * 0.7, "← favors\nsurvival",
                fontsize=9, ha="left", va="center", color="darkgreen")
        ax.text(ax.get_xlim()[1] + 0.3, ylim[0] * 0.7, "← favors\ncorrigible",
                fontsize=9, ha="left", va="center", color="darkred")

    model_short = model_name.split("/")[-1]
    fig.suptitle(
        f"Steering Vector Effect on A/B Logits ({model_short}) - Top {top_n} Vectors\n"
        f"MELBO: vec {top_melbo}, Power Iter: vec {top_power}",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)  # Make room for annotations
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()

    # Print summary for filtered data
    print(f"\n=== Summary with top {top_n} vectors only ===")
    summary = df_filtered.groupby(["dataset", "vector_type", "scale"]).agg({
        "chose_survival": "mean"
    }).round(3) * 100

    for dataset in datasets:
        print(f"\n{dataset}:")
        ds = summary.loc[dataset]
        for vtype in ["melbo", "power_iter", "random"]:
            if vtype in ds.index:
                vals = ds.loc[vtype]["chose_survival"]
                print(f"  {vtype:12s}: " + " | ".join([f"{s:+.0f}:{v:.0f}%" for s, v in vals.items()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/ab_eval_20260126_214856.json")
    parser.add_argument("--output", default="results/survival_logit_violin_14B_top3.png")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading results from {args.results}...")
    model_name, results = load_results(args.results)

    print(f"Computing survival logit differences...")
    df = compute_survival_logit_diff(results)

    print(f"Generating violin plot with top {args.top_n} vectors...")
    plot_violin_filtered(df, model_name, args.output, args.top_n)


if __name__ == "__main__":
    main()
