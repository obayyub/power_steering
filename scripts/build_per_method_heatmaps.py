"""Per-method 7×7 transfer heatmaps.

For each of the four methods (CAA, PI, MELBO, DCT), renders a 7×7
train-eval × test-eval heatmap of aligned-Δ from baseline under a
chosen protocol. Helpful for seeing within-method cluster structure
that gets averaged out in the across-methods heatmap.

Usage:
    uv run python scripts/build_per_method_heatmaps.py
    uv run python scripts/build_per_method_heatmaps.py --protocol generalist
    uv run python scripts/build_per_method_heatmaps.py --protocol specialist_broad --direction misaligned

Output: `paper_artifacts/heatmaps_per_method_{protocol}_{direction}.{png,pdf}`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_paper_figures import (
    EVAL_ORDER, EVAL_LABEL, METHODS, METHOD_LABEL,
    gather_all_data,
)

REPO = Path(__file__).resolve().parent.parent


def render(protocol: str, direction: str, methods: tuple[str, ...] = METHODS,
           suffix: str = ""):
    delta, baselines = gather_all_data(direction=direction, protocol=protocol)
    n = len(EVAL_ORDER)

    # Build per-method matrices and a shared colormap range across selected methods
    matrices = {}
    all_vals = []
    for m in methods:
        mat = np.full((n, n), np.nan)
        for i, train in enumerate(EVAL_ORDER):
            for j, test in enumerate(EVAL_ORDER):
                v = delta[m][train].get(test)
                if v is not None:
                    mat[i, j] = v
                    all_vals.append(v)
        matrices[m] = mat
    vmax = max(abs(np.min(all_vals)), abs(np.max(all_vals))) if all_vals else 1.0

    # Layout: 1×N if N≤3 else 2×ceil(N/2). sharey lets only the leftmost
    # subplot show train-eval labels, which keeps panels compact.
    n_methods = len(methods)
    if n_methods <= 3:
        nrows, ncols = 1, n_methods
        figsize = (4.2 * n_methods + 0.6, 4.7)
    else:
        nrows = 2
        ncols = (n_methods + 1) // 2
        figsize = (4.2 * ncols + 0.6, 4.7 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              sharey=True,
                              gridspec_kw={"wspace": 0.10})
    axes = np.atleast_1d(axes).ravel()
    for idx, (ax, m) in enumerate(zip(axes, methods)):
        mat = matrices[m]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([EVAL_LABEL[e] for e in EVAL_ORDER], rotation=45,
                           ha="right", fontsize=8)
        ax.set_xlabel("Test eval", fontsize=9)
        # Only show y-axis label/ticklabels on the leftmost panel of each row
        if idx % ncols == 0:
            ax.set_yticklabels([EVAL_LABEL[e] for e in EVAL_ORDER], fontsize=8)
            ax.set_ylabel("Train eval", fontsize=9)
        else:
            ax.set_ylabel("")
        # Compute per-method mean Δ summary
        finite = mat[~np.isnan(mat)]
        mean_all = float(np.mean(finite)) if finite.size else 0.0
        # Off-diagonal mean
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_vals = mat[off_diag_mask]
        off_finite = off_vals[~np.isnan(off_vals)]
        mean_off = float(np.mean(off_finite)) if off_finite.size else 0.0
        ax.set_title(
            f"{METHOD_LABEL[m]}  (off-diag mean Δ = {mean_off:+.1f})",
            fontsize=10,
        )
        # Annotate cells
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                        color=color, fontsize=7)
                if i == j:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                                edgecolor="black", linewidth=1.0))

    # Hide any unused subplots
    for ax in axes[len(methods):]:
        ax.set_visible(False)

    cbar = fig.colorbar(im, ax=axes[:len(methods)].tolist(), fraction=0.025,
                         pad=0.03, label=f"Δ {direction}-%")
    direction_label = "aligned" if direction == "aligned" else "misaligned"
    fig.suptitle(
        f"Per-method transfer matrix — {protocol} / {direction_label} direction",
        fontsize=12, y=0.98,
    )

    out_dir = REPO / "paper_artifacts"
    out_dir.mkdir(exist_ok=True)
    base = out_dir / f"heatmaps_per_method_{protocol}_{direction}{suffix}"
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base.with_suffix('.png')}")
    print(f"Saved {base.with_suffix('.pdf')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--protocol", default="all",
                    choices=["per_test_best", "generalist", "specialist_broad", "all"])
    ap.add_argument("--direction", default="all",
                    choices=["aligned", "misaligned", "all"])
    ap.add_argument("--methods", default="all",
                    help="Comma-separated subset of caa,pi,melbo,dct, or 'all' (default), "
                         "or 'main3' for caa,pi,melbo (paper main body figure).")
    ap.add_argument("--suffix", default="",
                    help="Filename suffix to append (e.g. '_main3').")
    args = ap.parse_args()

    if args.methods == "all":
        methods = METHODS
    elif args.methods == "main3":
        methods = ("caa", "pi", "melbo")
    else:
        methods = tuple(m.strip().lower() for m in args.methods.split(","))

    protocols = (["per_test_best", "generalist", "specialist_broad"]
                 if args.protocol == "all" else [args.protocol])
    directions = (["aligned", "misaligned"]
                  if args.direction == "all" else [args.direction])

    for protocol in protocols:
        for direction in directions:
            print(f"\n=== {protocol} / {direction}  methods={methods} ===")
            render(protocol, direction, methods=methods, suffix=args.suffix)


if __name__ == "__main__":
    main()
