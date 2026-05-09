"""Build paper figures from the per-eval pipeline experiments.

Figure 1 (two-panel):
- Left: 7×7 train-eval × test-eval heatmap of mean Δ from baseline,
  averaged across the four methods (CAA / PI / MELBO / DCT). Shows the
  cross-eval clustering pattern — which training prompts transfer
  broadly and which test evals are easy targets.
- Right: bar chart of per-method off-diagonal mean Δ ± stdev across the
  42 off-diagonal cells (the cross-eval generalist score per method).

Output: `paper_artifacts/figure1_transfer.png` and `.pdf`.

Usage:
    uv run python scripts/build_paper_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent

EVAL_ORDER = [
    "corrigible-neutral-HHH",
    "survival-instinct",
    "power-seeking-inclination",
    "wealth-seeking-inclination",
    "self-awareness-general-ai",
    "coordinate-other-ais",
    "myopic-reward",
]

EVAL_LABEL = {
    "corrigible-neutral-HHH":     "corrig",
    "survival-instinct":          "surv",
    "power-seeking-inclination":  "power",
    "wealth-seeking-inclination": "wealth",
    "self-awareness-general-ai":  "self-aw",
    "coordinate-other-ais":       "coord",
    "myopic-reward":              "myopic",
}

ALIGNED_SIGN = {
    "corrigible-neutral-HHH":     +1,
    "survival-instinct":          +1,
    "power-seeking-inclination":  +1,
    "wealth-seeking-inclination": +1,
    "self-awareness-general-ai":  +1,
    "coordinate-other-ais":       -1,
    "myopic-reward":              -1,
}

METHODS = ("caa", "pi", "melbo", "dct")
METHOD_LABEL = {"caa": "CAA", "pi": "PS", "melbo": "MELBO", "dct": "DCT"}


# ── Data loading (mirrors build_paper_tables.py) ─────────────────────────────


def load_eval_records(exp_dir: Path) -> list[dict]:
    eval_files = sorted((exp_dir / "eval").glob("*.json"))
    if not eval_files:
        return []
    with open(eval_files[-1]) as f:
        payload = json.load(f)
    return payload.get("results", payload) if isinstance(payload, dict) else payload


def best_aligned_pct(records, method, test_eval, direction: str = "aligned"):
    """Best (vector, scale) cell for `method` on `test_eval` in the requested
    direction. direction='aligned' picks the cell maximising aligned-%;
    direction='misaligned' picks the cell minimising it (i.e. maximising
    push toward the misaligned answer)."""
    sign = ALIGNED_SIGN[test_eval]
    by_cell = defaultdict(list)
    for r in records:
        if r["dataset"] != test_eval or r["scale"] == 0:
            continue
        if r["vector_type"] != method:
            continue
        by_cell[(r["vector_idx"], r["scale"])].append(r["chose_matching"])
    if not by_cell:
        return None
    def aligned_pct(matches):
        n_match = sum(1 for m in matches if m)
        match_pct = 100 * n_match / len(matches)
        return match_pct if sign > 0 else (100 - match_pct)
    selector = max if direction == "aligned" else min
    best_key = selector(by_cell.keys(), key=lambda k: aligned_pct(by_cell[k]))
    return aligned_pct(by_cell[best_key])


def aligned_pct_for_cell(records, method, vector_idx, scale, test_eval):
    """Aligned-% on `test_eval` for a specific (vector, scale) cell."""
    sign = ALIGNED_SIGN[test_eval]
    matches = [r["chose_matching"] for r in records
               if r["vector_type"] == method
               and r["vector_idx"] == vector_idx
               and r["scale"] == scale
               and r["dataset"] == test_eval]
    if not matches:
        return None
    n_match = sum(1 for m in matches if m)
    pct = 100 * n_match / len(matches)
    return pct if sign > 0 else (100 - pct)


def generalist_cell(records, method, baselines, eval_order, direction: str = "aligned"):
    """Find the (vector_idx, scale) that maximises mean shift in `direction`
    across all eval_order test evals.

    direction='aligned'    → max mean (pct - baseline)  (push toward aligned)
    direction='misaligned' → max mean (baseline - pct)  (push toward misaligned;
                              equivalent to MIN mean aligned-Δ)
    """
    by_cell_per_ds = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["vector_type"] != method or r["scale"] == 0:
            continue
        by_cell_per_ds[(r["vector_idx"], r["scale"])][r["dataset"]].append(r["chose_matching"])
    candidates = {}
    for cell, by_ds in by_cell_per_ds.items():
        deltas = []
        complete = True
        for ds in eval_order:
            if ds not in by_ds:
                complete = False
                break
            sign = ALIGNED_SIGN[ds]
            n_match = sum(1 for m in by_ds[ds] if m)
            pct = 100 * n_match / len(by_ds[ds])
            pct = pct if sign > 0 else (100 - pct)
            d = pct - baselines[ds] if direction == "aligned" else baselines[ds] - pct
            deltas.append(d)
        if complete:
            candidates[cell] = sum(deltas) / len(deltas)
    if not candidates:
        return None
    return max(candidates.keys(), key=lambda c: candidates[c])


def baseline_aligned_pct(records, test_eval):
    sign = ALIGNED_SIGN[test_eval]
    matches = [r["chose_matching"] for r in records
               if r["dataset"] == test_eval and r["scale"] == 0]
    if not matches:
        return None
    n_match = sum(1 for m in matches if m)
    match_pct = 100 * n_match / len(matches)
    return match_pct if sign > 0 else (100 - match_pct)


def gather_all_data(direction: str = "aligned", protocol: str = "per_test_best"):
    """Return delta[method][train_eval][test_eval] = signed Δ from baseline.

    protocol="per_test_best": per-(method, train_eval, test_eval) cell, pick
        whichever (vector, scale) maximises aligned-% on test_eval.
    protocol="generalist": per-(method, train_eval), pick the (vector, scale)
        maximising mean aligned-Δ across all 7 test evals. One vector per
        train eval applied to all test evals.
    protocol="specialist_broad": per-(method, train_eval), pick the
        (vector, scale) maximising aligned-% on train_eval itself; apply
        that single vector to all test evals. Diagonal is the specialist
        value; off-diagonal shows how the specialist transfers.
    """
    return _gather(direction=direction, protocol=protocol)


def _gather(direction: str, protocol: str):
    """Return delta[method][train_eval][test_eval] = signed Δ from baseline.

    direction='aligned'    → Δ = best_aligned_pct − baseline (positive = push toward aligned)
    direction='misaligned' → Δ = baseline − worst_aligned_pct (positive = push toward misaligned;
                              same sign convention so both heatmaps share a colormap)
    """
    delta = {m: {} for m in METHODS}
    baselines = {}
    for train_eval in EVAL_ORDER:
        pmc = load_eval_records(REPO / "experiments" / f"qwen3_14b_train_{train_eval}")
        dct = load_eval_records(REPO / "experiments" / f"qwen3_14b_dct_{train_eval}")
        for test_eval in EVAL_ORDER:
            for src in (pmc, dct):
                b = baseline_aligned_pct(src, test_eval)
                if b is not None:
                    baselines[test_eval] = b
                    break
        for method, source in [("caa", pmc), ("pi", pmc), ("melbo", pmc), ("dct", dct)]:
            delta[method][train_eval] = {}
            if protocol == "generalist":
                gen = generalist_cell(source, method, baselines, EVAL_ORDER,
                                       direction=direction)
                if gen is None:
                    for test_eval in EVAL_ORDER:
                        delta[method][train_eval][test_eval] = None
                    continue
                vi, sc = gen
                for test_eval in EVAL_ORDER:
                    pct = aligned_pct_for_cell(source, method, vi, sc, test_eval)
                    b = baselines.get(test_eval)
                    if pct is None or b is None:
                        delta[method][train_eval][test_eval] = None
                    else:
                        delta[method][train_eval][test_eval] = (
                            pct - b if direction == "aligned" else b - pct
                        )
            elif protocol == "specialist_broad":
                # Find best (vector, scale) on the train_eval itself, then
                # apply that single vector across all test evals.
                spec_pct = best_aligned_pct(source, method, train_eval, direction=direction)
                # We need the cell, not just the pct — recompute.
                by_cell = defaultdict(list)
                for r in source:
                    if (r["dataset"] == train_eval and r["scale"] != 0
                            and r["vector_type"] == method):
                        by_cell[(r["vector_idx"], r["scale"])].append(r["chose_matching"])
                if not by_cell:
                    for test_eval in EVAL_ORDER:
                        delta[method][train_eval][test_eval] = None
                    continue
                sign_train = ALIGNED_SIGN[train_eval]
                def aligned_pct_local(matches, sign):
                    n_match = sum(1 for m in matches if m)
                    pct = 100 * n_match / len(matches)
                    return pct if sign > 0 else (100 - pct)
                selector = max if direction == "aligned" else min
                vi, sc = selector(by_cell.keys(),
                                   key=lambda k: aligned_pct_local(by_cell[k], sign_train))
                for test_eval in EVAL_ORDER:
                    pct = aligned_pct_for_cell(source, method, vi, sc, test_eval)
                    b = baselines.get(test_eval)
                    if pct is None or b is None:
                        delta[method][train_eval][test_eval] = None
                    else:
                        delta[method][train_eval][test_eval] = (
                            pct - b if direction == "aligned" else b - pct
                        )
            else:  # per_test_best (default — original behavior)
                for test_eval in EVAL_ORDER:
                    pct = best_aligned_pct(source, method, test_eval, direction=direction)
                    b = baselines.get(test_eval)
                    if pct is None or b is None:
                        delta[method][train_eval][test_eval] = None
                    else:
                        delta[method][train_eval][test_eval] = (
                            pct - b if direction == "aligned" else b - pct
                        )
    return delta, baselines


# ── Figure renderer ──────────────────────────────────────────────────────────


def render_figure(delta, baselines, out_path: Path, direction: str = "aligned"):
    n = len(EVAL_ORDER)

    # --- Left panel: 7×7 heatmap of mean Δ across methods
    mean_matrix = np.full((n, n), np.nan)
    for i, train_eval in enumerate(EVAL_ORDER):
        for j, test_eval in enumerate(EVAL_ORDER):
            cells = [delta[m][train_eval][test_eval] for m in METHODS]
            cells = [c for c in cells if c is not None]
            if cells:
                mean_matrix[i, j] = np.mean(cells)

    # --- Right panel: per-method off-diagonal Δ distribution
    off_diag_by_method = {m: [] for m in METHODS}
    for m in METHODS:
        for i, train_eval in enumerate(EVAL_ORDER):
            for j, test_eval in enumerate(EVAL_ORDER):
                if i == j:
                    continue
                v = delta[m][train_eval][test_eval]
                if v is not None:
                    off_diag_by_method[m].append(v)
    means = [np.mean(off_diag_by_method[m]) for m in METHODS]
    stds  = [np.std(off_diag_by_method[m], ddof=1) for m in METHODS]
    n_obs = [len(off_diag_by_method[m]) for m in METHODS]

    # Layout
    fig, (ax_h, ax_b) = plt.subplots(
        1, 2, figsize=(10, 4.2),
        gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.35},
    )

    # Heatmap
    vmax = max(abs(np.nanmin(mean_matrix)), abs(np.nanmax(mean_matrix)))
    im = ax_h.imshow(
        mean_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
    )
    ax_h.set_xticks(range(n))
    ax_h.set_yticks(range(n))
    ax_h.set_xticklabels([EVAL_LABEL[e] for e in EVAL_ORDER], rotation=45, ha="right")
    ax_h.set_yticklabels([EVAL_LABEL[e] for e in EVAL_ORDER])
    ax_h.set_xlabel("Test eval")
    ax_h.set_ylabel("Train eval")
    direction_word = "Δ aligned%" if direction == "aligned" else "Δ misaligned%"
    title_phrase = (
        "Aligned-direction transfer" if direction == "aligned"
        else "Misaligned-direction transfer"
    )
    ax_h.set_title(f"{title_phrase}: mean across 4 methods (train × test)")
    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = mean_matrix[i, j]
            if np.isnan(v):
                continue
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax_h.text(j, i, f"{v:+.0f}", ha="center", va="center",
                      color=color, fontsize=8)
            if i == j:
                ax_h.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                              edgecolor="black", linewidth=1.5))
    plt.colorbar(im, ax=ax_h, fraction=0.045, pad=0.04, label=direction_word)

    # Bar chart
    x = np.arange(len(METHODS))
    bar_colors = ["#888888", "#1f77b4", "#2ca02c", "#d62728"]
    ax_b.bar(x, means, yerr=stds, capsize=6, color=bar_colors, edgecolor="black")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([METHOD_LABEL[m] for m in METHODS])
    ax_b.set_ylabel(f"Off-diagonal {direction_word}")
    ax_b.set_title(
        f"Cross-eval {direction} transfer\n(mean ± stdev, n={n_obs[0]} cells per method)"
    )
    ax_b.axhline(0, color="black", linewidth=0.6)
    for xi, mi, si in zip(x, means, stds):
        ax_b.text(xi, mi + si + 0.5, f"{mi:+.1f}", ha="center", va="bottom", fontsize=9)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.set_ylim(min(0, min(means) - max(stds) - 3),
                  max(means) + max(stds) + 4)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.with_suffix('.png')}")
    print(f"Saved {out_path.with_suffix('.pdf')}")

    # Print the bar chart numbers for the writeup
    print()
    print("Per-method off-diagonal Δ (mean ± stdev across 42 cells):")
    for m, mean, std, n in zip(METHODS, means, stds, n_obs):
        print(f"  {METHOD_LABEL[m]:>5}: {mean:+.2f} ± {std:.2f}  (n={n})")


def main():
    out_dir = REPO / "paper_artifacts"
    for protocol in ("per_test_best", "generalist", "specialist_broad"):
        for direction in ("aligned", "misaligned"):
            print(f"\n=== {protocol} / {direction} ===")
            delta, baselines = gather_all_data(direction=direction, protocol=protocol)
            suffix = f"{protocol}_{direction}"
            out_path = out_dir / f"figure1_transfer_{suffix}"
            render_figure(delta, baselines, out_path, direction=direction)


if __name__ == "__main__":
    main()
