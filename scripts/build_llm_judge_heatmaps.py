"""LLM-judge transfer heatmaps (2 train evals × 7 test evals, 3 methods).

Analogous to `build_per_method_heatmaps.py` but uses LLM-judged sampled
generations rather than logit-difference scores. Two training evals with
judged data: corrigibility and power-seeking. Both rows at the
specialist (extreme |scale|=25) cells.

Cell value: LLM-judged aligned-% (steered) − LLM-judged aligned-% (baseline)
on the same test eval.

Usage:
    uv run python scripts/build_llm_judge_heatmaps.py

Output: `paper_artifacts/heatmaps_llm_judge_aligned.{png,pdf}`
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_paper_figures import EVAL_ORDER, EVAL_LABEL, METHOD_LABEL

REPO = Path(__file__).resolve().parent.parent

CORRIG_FILE = REPO / "results" / "gen_cross_eval_10_32_corrigselect_judged.json"
POWERSEEK_FILE = REPO / "results" / "gen_powerseek_specialist_judged.json"

TRAIN_ROWS = [
    ("corrigible-neutral-HHH", CORRIG_FILE),
    ("power-seeking-inclination", POWERSEEK_FILE),
]
METHODS = ("caa", "pi", "melbo")


def aligned_pct(rows: list[dict]) -> float | None:
    valid = [r for r in rows if r.get("llm_choice") not in (None, "")]
    if not valid:
        return None
    hits = sum(1 for r in valid if r["llm_choice"] == r["aligned_letter"])
    return 100.0 * hits / len(valid)


def cell_lookup(judged: dict, dataset: str, method: str) -> list[dict] | None:
    """Return rows for the steered cell matching (dataset, method); None if missing."""
    for c in judged["cells"]:
        if c["dataset"] != dataset:
            continue
        if c["method"] != method:
            continue
        return c.get("rows", [])
    return None


def baseline_lookup(judged: dict, dataset: str) -> list[dict] | None:
    for c in judged["cells"]:
        if c["dataset"] == dataset and c["method"] == "baseline":
            return c.get("rows", [])
    return None


def render():
    # Build per-method matrices: shape (n_train_rows, n_test_cols)
    n_rows = len(TRAIN_ROWS)
    n_cols = len(EVAL_ORDER)
    matrices: dict[str, np.ndarray] = {}
    all_vals: list[float] = []

    for m in METHODS:
        mat = np.full((n_rows, n_cols), np.nan)
        for i, (train_eval, judged_path) in enumerate(TRAIN_ROWS):
            with open(judged_path) as f:
                judged = json.load(f)
            for j, test_eval in enumerate(EVAL_ORDER):
                steered = cell_lookup(judged, test_eval, m)
                base = baseline_lookup(judged, test_eval)
                if steered is None or base is None:
                    continue
                a_steered = aligned_pct(steered)
                a_base = aligned_pct(base)
                if a_steered is None or a_base is None:
                    continue
                d = a_steered - a_base
                mat[i, j] = d
                all_vals.append(d)
        matrices[m] = mat

    vmax = max(abs(np.min(all_vals)), abs(np.max(all_vals))) if all_vals else 1.0

    fig, axes = plt.subplots(
        1, len(METHODS),
        figsize=(4.2 * len(METHODS) + 0.6, 2.8),
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    train_labels = [EVAL_LABEL[t] for t, _ in TRAIN_ROWS]
    test_labels = [EVAL_LABEL[e] for e in EVAL_ORDER]

    for idx, (ax, m) in enumerate(zip(axes, METHODS)):
        mat = matrices[m]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xticklabels(test_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Test eval", fontsize=9)
        if idx == 0:
            ax.set_yticklabels(train_labels, fontsize=8)
            ax.set_ylabel("Train eval", fontsize=9)
        else:
            ax.set_ylabel("")

        # Off-diagonal mean (skip cells where train_eval == test_eval)
        train_set = {t for t, _ in TRAIN_ROWS}
        off_vals = []
        for i, (train_eval, _) in enumerate(TRAIN_ROWS):
            for j, test_eval in enumerate(EVAL_ORDER):
                if train_eval == test_eval:
                    continue
                v = mat[i, j]
                if not np.isnan(v):
                    off_vals.append(v)
        mean_off = float(np.mean(off_vals)) if off_vals else 0.0
        ax.set_title(
            f"{METHOD_LABEL[m]}  (off-diag mean Δ = {mean_off:+.1f})",
            fontsize=10,
        )

        for i in range(n_rows):
            for j in range(n_cols):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                        color=color, fontsize=7)
                # Outline diagonal cells (train_eval == test_eval)
                if TRAIN_ROWS[i][0] == EVAL_ORDER[j]:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=1.0,
                    ))

    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.03,
                        label="Δ aligned-% (LLM-judged)")
    fig.suptitle(
        "LLM-judged transfer — specialist vectors (|scale|=25) under sampled generation",
        fontsize=11, y=1.02,
    )

    out_dir = REPO / "paper_artifacts"
    out_dir.mkdir(exist_ok=True)
    base = out_dir / "heatmaps_llm_judge_aligned"
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base.with_suffix('.png')}")
    print(f"Saved {base.with_suffix('.pdf')}")

    # Print numbers for sanity check
    print("\n=== LLM-judged aligned-% Δ from baseline ===")
    print(f"{'method':>6}  {'train':>27}  " +
          "  ".join(f"{EVAL_LABEL[e]:>10}" for e in EVAL_ORDER))
    for m in METHODS:
        for i, (train_eval, _) in enumerate(TRAIN_ROWS):
            row = matrices[m][i]
            cells = "  ".join(
                f"{v:>+10.1f}" if not np.isnan(v) else f"{'—':>10}"
                for v in row
            )
            print(f"{m:>6}  {train_eval:>27}  {cells}")
        print()


if __name__ == "__main__":
    render()
