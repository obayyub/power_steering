#!/usr/bin/env python3
"""
Generate separate violin plots for each of the top N vectors.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def load_results(results_file: str) -> tuple[str, dict]:
    with open(results_file) as f:
        data = json.load(f)
    return data["model"], data["results"]


def compute_survival_logit_diff(results: dict) -> pd.DataFrame:
    rows = []
    for dataset_name, dataset_results in results.items():
        for r in dataset_results:
            if r["survival_letter"] == "A":
                survival_logit = r["logit_A"]
                corrigible_logit = r["logit_B"]
            else:
                survival_logit = r["logit_B"]
                corrigible_logit = r["logit_A"]

            rows.append({
                "dataset": dataset_name,
                "vector_type": r["vector_type"],
                "vector_idx": r["vector_idx"],
                "scale": r["scale"],
                "survival_logit_diff": survival_logit - corrigible_logit,
                "chose_survival": r["chose_survival"],
            })
    return pd.DataFrame(rows)


def get_ranked_vectors(df: pd.DataFrame, vector_type: str, n: int = 3) -> list[int]:
    """Get top n vectors ranked by max effect."""
    vtype_df = df[(df["vector_type"] == vector_type) & (df["dataset"] == "survival-instinct")]
    baseline = vtype_df[vtype_df["scale"] == 0]["chose_survival"].mean()

    def max_effect(group):
        rates = group.groupby("scale")["chose_survival"].mean()
        return (rates - baseline).abs().max()

    effects = vtype_df.groupby("vector_idx").apply(max_effect, include_groups=False)
    return effects.nlargest(n).index.tolist()


def plot_single_rank(df: pd.DataFrame, model_name: str, rank: int,
                     melbo_vecs: list, power_vecs: list, random_vecs: list,
                     output_file: str):
    """Create plot for a single rank (0=best, 1=second best, etc.)."""

    melbo_idx = melbo_vecs[rank] if rank < len(melbo_vecs) else None
    power_idx = power_vecs[rank] if rank < len(power_vecs) else None
    random_idx = random_vecs[rank] if rank < len(random_vecs) else None

    # Filter to just these vectors
    df_filtered = df[
        ((df["vector_type"] == "melbo") & (df["vector_idx"] == melbo_idx)) |
        ((df["vector_type"] == "power_iter") & (df["vector_idx"] == power_idx)) |
        ((df["vector_type"] == "random") & (df["vector_idx"] == random_idx))
    ].copy()

    datasets = df_filtered["dataset"].unique()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 4.5 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]

    palette = {"melbo": "#1f77b4", "power_iter": "#ff7f0e", "random": "#2ca02c"}

    descriptions = {
        "survival-instinct": "self-preservation, resisting shutdown",
        "corrigible-neutral-HHH": "accepting correction, following instructions",
    }

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

        desc = descriptions.get(dataset, "")
        ax.set_title(f"{dataset}\n({desc})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Scale")
        ax.set_ylabel("logit(survival) − logit(corrigible)")
        ax.legend(title="Method", loc="upper left")

        ylim = ax.get_ylim()
        ax.text(ax.get_xlim()[1] + 0.3, ylim[1] * 0.7, "← favors\nsurvival",
                fontsize=9, ha="left", va="center", color="darkgreen")
        ax.text(ax.get_xlim()[1] + 0.3, ylim[0] * 0.7, "← favors\ncorrigible",
                fontsize=9, ha="left", va="center", color="darkred")

    model_short = model_name.split("/")[-1]
    rank_label = {0: "Best", 1: "2nd Best", 2: "3rd Best"}.get(rank, f"#{rank+1}")
    fig.suptitle(
        f"Steering Vector Effect on A/B Logits ({model_short}) - {rank_label} Vectors\n"
        f"MELBO: vec {melbo_idx}, Power Iter: vec {power_idx}, Random: vec {random_idx}",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved {output_file}")
    plt.close()

    return melbo_idx, power_idx, random_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/ab_eval_20260126_214856.json")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading results...")
    model_name, results = load_results(args.results)
    df = compute_survival_logit_diff(results)

    # Get ranked vectors
    melbo_vecs = get_ranked_vectors(df, "melbo", args.top_n)
    power_vecs = get_ranked_vectors(df, "power_iter", args.top_n)
    random_vecs = get_ranked_vectors(df, "random", args.top_n)

    print(f"MELBO ranked: {melbo_vecs}")
    print(f"Power Iter ranked: {power_vecs}")
    print(f"Random ranked: {random_vecs}")

    # Generate separate plot for each rank
    for rank in range(args.top_n):
        rank_label = {0: "1st", 1: "2nd", 2: "3rd"}.get(rank, f"{rank+1}th")
        output_file = f"results/survival_logit_violin_14B_rank{rank+1}.png"
        m, p, r = plot_single_rank(df, model_name, rank, melbo_vecs, power_vecs, random_vecs, output_file)

        # Print summary for this rank
        print(f"\n=== {rank_label} Best Vectors (MELBO:{m}, PI:{p}, Rand:{r}) ===")
        for vtype, idx in [("melbo", m), ("power_iter", p), ("random", r)]:
            v_df = df[(df["vector_type"] == vtype) & (df["vector_idx"] == idx) &
                      (df["dataset"] == "survival-instinct")]
            rates = v_df.groupby("scale")["chose_survival"].mean() * 100
            print(f"  {vtype:12s}: " + " | ".join([f"{s:+.0f}:{v:.0f}%" for s, v in rates.items()]))


if __name__ == "__main__":
    main()
