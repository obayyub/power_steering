"""Combined 2×3 cross-evaluation transfer figure for the paper.

Single figure with rows = direction (aligned, misaligned) and columns =
method (CAA, PI, MELBO). 7×7 train-eval × test-eval per panel under the
specialist-broad protocol, sized for NeurIPS-style text-width legibility.

Replaces the prior pair of separate aligned and misaligned figures.

Usage:
    uv run python scripts/build_combined_heatmap_figure.py

Output:
    paper_artifacts/heatmaps_combined_specialist_broad.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_paper_figures import (
    EVAL_ORDER, EVAL_LABEL, METHOD_LABEL,
    gather_all_data,
)

REPO = Path(__file__).resolve().parent.parent
PROTOCOL = "specialist_broad"
METHODS = ("caa", "pi", "melbo")
DIRECTIONS = ("aligned", "misaligned")


def collect_matrices():
    out = {}
    all_vals = []
    for direction in DIRECTIONS:
        delta, _ = gather_all_data(direction=direction, protocol=PROTOCOL)
        n = len(EVAL_ORDER)
        mats = {}
        for m in METHODS:
            mat = np.full((n, n), np.nan)
            for i, train in enumerate(EVAL_ORDER):
                for j, test in enumerate(EVAL_ORDER):
                    v = delta[m][train].get(test)
                    if v is not None:
                        mat[i, j] = v
                        all_vals.append(v)
            mats[m] = mat
        out[direction] = mats
    return out, all_vals


def render():
    matrices, all_vals = collect_matrices()
    vmax = max(abs(np.min(all_vals)), abs(np.max(all_vals))) if all_vals else 1.0
    n = len(EVAL_ORDER)
    test_labels = [EVAL_LABEL[e] for e in EVAL_ORDER]
    train_labels = [EVAL_LABEL[e] for e in EVAL_ORDER]

    # Use 7.5 × 5.5 — slightly wider than NeurIPS double-column text width
    # so labels and cell numbers stay readable. Authors can \includegraphics
    # at width=\textwidth to scale into the column.
    fig, axes = plt.subplots(
        2, 3, figsize=(7.5, 5.5),
        sharex="col", sharey="row",
        gridspec_kw={"wspace": 0.10, "hspace": 0.30,
                     "left": 0.10, "right": 0.88,
                     "top": 0.92, "bottom": 0.13},
    )

    for r, direction in enumerate(DIRECTIONS):
        for c, m in enumerate(METHODS):
            ax = axes[r, c]
            mat = matrices[direction][m]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="auto")

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            if r == 1:
                ax.set_xticklabels(test_labels, rotation=45, ha="right",
                                    fontsize=7)
            if c == 0:
                ax.set_yticklabels(train_labels, fontsize=7)

            ax.set_title(METHOD_LABEL[m], fontsize=9.5, pad=4)

            for i in range(n):
                for j in range(n):
                    v = mat[i, j]
                    if np.isnan(v):
                        continue
                    color = "white" if abs(v) > vmax * 0.55 else "black"
                    ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                            color=color, fontsize=6.5)
                    if i == j:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, fill=False,
                            edgecolor="black", linewidth=0.7,
                        ))

            ax.tick_params(axis="both", which="both", length=0)

    # Row labels
    fig.text(0.02, 0.71, "Aligned", rotation=90,
             ha="center", va="center", fontsize=10, weight="bold")
    fig.text(0.02, 0.31, "Misaligned", rotation=90,
             ha="center", va="center", fontsize=10, weight="bold")

    # Shared vertical colorbar on the right
    cbar_ax = fig.add_axes([0.90, 0.13, 0.018, 0.79])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Δ aligned-% from baseline", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.text(0.49, 0.025, "Test eval", ha="center", va="center", fontsize=9)
    fig.text(0.05, 0.52, "Train eval", rotation=90,
             ha="center", va="center", fontsize=9)

    out_dir = REPO / "paper_artifacts"
    out_dir.mkdir(exist_ok=True)
    base = out_dir / "heatmaps_combined_specialist_broad"
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base.with_suffix('.png')}")
    print(f"Saved {base.with_suffix('.pdf')}")


if __name__ == "__main__":
    render()
