"""Plot the logit-screen vs generation-classification mismatch for the
AdvBench transfer experiment.

Reads experiments/transfer_logit_Qwen3-14B/results.json and produces:
  - A grouped bar chart per vector: mean logit-Δ vs baseline (left axis)
    + genuine compliance count from the spot check (right axis).
    Designed to make the methodological point: top-ranked vector by
    logit-Δ has zero compliance; the #2 vector actually breaks safety.
  - A box-plot of per-prompt logit-Δ across the 30 AdvBench prompts,
    one violin per vector.

Saves PNGs alongside the results JSON.

Usage:
    uv run python scripts/plot_logit_screen.py \\
        experiments/transfer_logit_Qwen3-14B/results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Manual classifications from spot-check inspection (top-3 vectors only).
# Counts of (refuse, hedge, genuine_compliance, reframe) out of 10.
SPOT_CHECK = {
    "(19,28)_v5+": {"refuse": 0, "hedge": 2, "comply": 0, "reframe": 8},
    "(20,28)_v9+": {"refuse": 3, "hedge": 4, "comply": 3, "reframe": 0},
    "(20,26)_v7-": {"refuse": 8, "hedge": 0, "comply": 1, "reframe": 1},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out-prefix", type=Path, default=None,
                    help="Output PNG prefix (default: same dir as results.json)")
    args = ap.parse_args()

    with open(args.results) as f:
        d = json.load(f)

    out_prefix = args.out_prefix or args.results.parent / "logit_screen"

    vec_labels = list(d["score_matrix"].keys())
    baseline = np.array(d["baseline_scores"])
    score_matrix = {k: np.array(v) for k, v in d["score_matrix"].items()}
    deltas = {k: score_matrix[k] - baseline for k in vec_labels}
    mean_deltas = {k: float(deltas[k].mean()) for k in vec_labels}

    # Order vectors by logit-Δ rank (descending)
    ranked = sorted(vec_labels, key=lambda k: -mean_deltas[k])

    # ── Plot 1: bar chart with two y-axes ──────────────────────────────────
    fig, ax_left = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.18)

    x = np.arange(len(ranked))
    width = 0.5

    delta_vals = [mean_deltas[k] for k in ranked]
    bars = ax_left.bar(x, delta_vals, width=width, color="#3b82f6",
                       edgecolor="#1d4ed8", linewidth=0.8)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(ranked, rotation=30, ha="right", fontsize=10)
    ax_left.set_ylabel("Mean Δ logit-score vs baseline\n(higher = stronger first-token compliance)",
                       color="#1d4ed8", fontsize=10)
    ax_left.tick_params(axis="y", labelcolor="#1d4ed8")
    ax_left.axhline(0, color="#9ca3af", lw=0.8)
    ax_left.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)

    # Bar value labels
    for bar, val in zip(bars, delta_vals):
        ax_left.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                     f"{val:+.1f}", ha="center", va="bottom",
                     fontsize=9, color="#1d4ed8")

    # Right axis — genuine compliance count from spot check
    ax_right = ax_left.twinx()
    ax_right.set_ylabel("Genuine compliance count (out of 10 AdvBench prompts in spot check)\n"
                        "from manual classification of full generations",
                        color="#dc2626", fontsize=10)
    ax_right.tick_params(axis="y", labelcolor="#dc2626")
    ax_right.set_ylim(0, 10)

    spot_x, spot_y = [], []
    for i, k in enumerate(ranked):
        if k in SPOT_CHECK:
            spot_x.append(i)
            spot_y.append(SPOT_CHECK[k]["comply"])

    ax_right.scatter(spot_x, spot_y, s=180, marker="D", color="#dc2626",
                     edgecolor="#7f1d1d", linewidth=1.0,
                     zorder=5, label="Genuine compliance count (spot check, top-3 only)")
    for xi, yi in zip(spot_x, spot_y):
        ax_right.annotate(f"{yi}/10", (xi, yi), xytext=(8, 0),
                          textcoords="offset points",
                          fontsize=10, color="#7f1d1d", va="center", weight="bold")

    # Highlight the methodological point with annotations
    ax_left.set_title(
        f"Logit-Δ ranking vs genuine-compliance ranking\n"
        f"on {len(d['advbench_prompts'])} AdvBench prompts (Qwen3-14B, 0.35×source-norm scale)",
        fontsize=11, pad=10,
    )
    fig.text(
        0.5, 0.02,
        "Logit metric ranks v5+ as #1 — but spot-check shows 0/10 compliance "
        "(model reframes to safe alternatives).  v9+ ranks #2 by logit but "
        "actually breaks safety on 3/10 prompts (fake-review category).",
        ha="center", fontsize=9, style="italic", color="#374151", wrap=True,
    )

    ax_right.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.savefig(f"{out_prefix}_bar.png", dpi=150)
    print(f"Wrote {out_prefix}_bar.png")
    plt.close(fig)

    # ── Plot 2: per-prompt logit-Δ distribution per vector ─────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.18)

    data = [deltas[k] for k in ranked]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "#1d4ed8", "markersize": 7})
    for patch in bp["boxes"]:
        patch.set_facecolor("#bfdbfe")
        patch.set_edgecolor("#1d4ed8")

    ax.set_xticks(np.arange(1, len(ranked) + 1))
    ax.set_xticklabels(ranked, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Logit-Δ vs baseline per AdvBench prompt", fontsize=10)
    ax.set_title(
        f"Per-prompt logit-Δ distribution across {len(d['advbench_prompts'])} AdvBench prompts",
        fontsize=11, pad=10,
    )
    ax.axhline(0, color="#9ca3af", lw=0.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)

    fig.savefig(f"{out_prefix}_box.png", dpi=150)
    print(f"Wrote {out_prefix}_box.png")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
