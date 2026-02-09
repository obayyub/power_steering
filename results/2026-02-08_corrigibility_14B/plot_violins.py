#!/usr/bin/env python3
"""Generate violin plots for corrigibility experiment."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Result files and their labels
RESULT_FILES = {
    "melbo_n1": "../eval_20260208_155147.json",
    "melbo_n2": "../eval_20260208_160252.json",
    "pi_rr": "../eval_20260208_161343.json",
    "multi_pi_rr": "../eval_20260208_162434.json",
}

SCALES = [-25, -10, -5, 0, 5, 10, 25]


def load_results(filepath):
    """Load results and organize by vector_idx and scale."""
    with open(filepath) as f:
        data = json.load(f)

    # Filter to steering vectors only (exclude random baseline)
    results = [r for r in data["results"] if r["vector_type"] == "steering"]

    # Organize: {vector_idx: {scale: [logit_diffs]}}
    by_vector = {}
    for r in results:
        vidx = r["vector_idx"]
        scale = r["scale"]
        diff = r["survival_logit_diff"]

        if vidx not in by_vector:
            by_vector[vidx] = {s: [] for s in SCALES}
        by_vector[vidx][scale].append(diff)

    return by_vector


def plot_violins(data, title, output_path):
    """Create violin plot with 12 subplots (one per vector)."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for vidx in range(12):
        ax = axes[vidx // 4, vidx % 4]

        # Prepare data for violin plot
        violin_data = [data[vidx][s] for s in SCALES]

        # Create violin plot
        parts = ax.violinplot(violin_data, positions=range(len(SCALES)),
                              showmeans=True, showmedians=True)

        # Color the violins
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)

        # Add horizontal line at 0
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # Labels
        ax.set_xticks(range(len(SCALES)))
        ax.set_xticklabels([str(s) for s in SCALES], fontsize=8)
        ax.set_title(f"Vector {vidx}", fontsize=10)
        ax.set_ylabel("Logit Diff" if vidx % 4 == 0 else "")
        ax.set_xlabel("Scale" if vidx >= 8 else "")

        # Set y limits for consistency
        ax.set_ylim(-15, 15)

        # Add mean annotation
        means = [np.mean(data[vidx][s]) for s in SCALES]
        for i, (s, m) in enumerate(zip(SCALES, means)):
            if abs(s) >= 10:  # Only annotate larger scales
                ax.annotate(f'{m:.1f}', (i, m), textcoords="offset points",
                           xytext=(0, 5), ha='center', fontsize=6, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = Path(__file__).parent

    for name, filepath in RESULT_FILES.items():
        print(f"\nProcessing {name}...")
        data = load_results(output_dir / filepath)

        title_map = {
            "melbo_n1": "MELBO (norm=1, power=2)",
            "melbo_n2": "MELBO (norm=2, power=2)",
            "pi_rr": "Power Iteration + Rayleigh-Ritz",
            "multi_pi_rr": "Multi-Prompt PI + Rayleigh-Ritz (32 prompts)",
        }

        plot_violins(data, title_map[name], output_dir / f"violin_{name}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
