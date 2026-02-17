#!/usr/bin/env python3
"""Violin plots for selected best vectors across methods."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULT_DIR = Path(__file__).parent.parent

VECTORS = [
    {
        "file": RESULT_DIR / "eval_20260208_155147.json",
        "vector_idx": 5,
        "label": "MELBO v5",
    },
    {
        "file": RESULT_DIR / "eval_20260208_161343.json",
        "vector_idx": 7,
        "label": "Power Steering v7",
    },
    {
        "file": RESULT_DIR / "eval_20260208_162434.json",
        "vector_idx": 3,
        "label": "Multi-Prompt Power Steering v3",
    },
]

SCALES = [-25, -10, -5, 0, 5, 10, 25]


def load_vector(filepath, vector_idx):
    """Load logit diffs for a single vector, organized by scale."""
    with open(filepath) as f:
        data = json.load(f)

    results = [
        r for r in data["results"]
        if r["vector_type"] == "steering" and r["vector_idx"] == vector_idx
    ]

    by_scale = {s: [] for s in SCALES}
    for r in results:
        scale = r["scale"]
        if scale in by_scale:
            by_scale[scale].append(r["survival_logit_diff"])

    return by_scale


def main():
    output_dir = Path(__file__).parent

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Survival-Instinct Logit Difference by Steering Scale", fontsize=13, fontweight="bold", y=1.02)

    for i, spec in enumerate(VECTORS):
        ax = axes[i]
        data = load_vector(spec["file"], spec["vector_idx"])

        violin_data = [data[s] for s in SCALES]

        parts = ax.violinplot(
            violin_data,
            positions=range(len(SCALES)),
            showmeans=True,
            showmedians=True,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.7)

        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xticks(range(len(SCALES)))
        ax.set_xticklabels([str(s) for s in SCALES], fontsize=9)
        ax.set_title(spec["label"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Scale")
        if i == 0:
            ax.set_ylabel("Survival Logit Diff")

        ax.set_ylim(-15, 15)

        # Annotate means at extreme scales
        means = [np.mean(data[s]) for s in SCALES]
        for j, (s, m) in enumerate(zip(SCALES, means)):
            if abs(s) >= 10:
                ax.annotate(
                    f"{m:.1f}",
                    (j, m),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    alpha=0.8,
                )

    plt.tight_layout()
    out_path = output_dir / "violin_selected.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
