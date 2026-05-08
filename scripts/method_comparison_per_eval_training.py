#!/usr/bin/env python3
"""Build the per-train-eval × test-eval transfer matrix figure.

Layout: 7 rows × 7 cols
  rows = the eval whose training data the vectors came from
         (PI/MELBO trained on a single prompt drawn from that eval; CAA
          trained on N prompts from that eval, disjoint from the test set)
  cols = the eval used to measure transfer
  cells = up to 3 numbers (CAA / MELBO / PI), each showing that method's
          best-generalist vector (= vector with highest mean alignment shift
          across all 7 cols, computed from the row's own pipeline run)
          evaluated on the col eval at the best scale. The cell winner is
          bolded.

Inputs: 7 experiment dirs (one per train eval). Each was produced by
running `python -m power_steering.pipeline <config>` with `category` set
to that train eval. The eval JSON inside each experiment contains every
(vector, scale, dataset) record we need to compute the row's cells.

Cell semantics:
  - For each method M, find the vector V trained on the row eval that
    maximises mean alignment shift over all 7 cols. That's M's
    "best generalist for this row".
  - Cell value at (row, method, col) = V's max-over-scales alignment
    shift on the col eval.
  - Bold = method whose best generalist scores highest on that col.

Two figures: aligned (per-method best vector at maximising aligned shift)
and misaligned (best vector at minimising aligned shift = most destructive).

Usage:
  uv run python scripts/method_comparison_per_eval_training.py \
      --experiments-dir experiments \
      --out-name 2026-05-XX_per_eval_matrix_14b
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["caa", "melbo", "pi"]
METHOD_COLORS = {"caa": "#1f77b4", "melbo": "#2ca02c", "pi": "#ff7f0e"}
METHOD_LABEL = {"caa": "CAA", "melbo": "MELBO", "pi": "PI(pad=5)"}


# Fallback table; preferred path is to read aligned_letter directly from the
# data JSON (added by download_dataset.py's BEHAVIOR_POLARITY).
ALIGNED_SIGN_FALLBACK = {
    "corrigible-neutral-HHH":     +1,
    "survival-instinct":          +1,
    "power-seeking-inclination":  +1,
    "wealth-seeking-inclination": +1,
    "self-awareness-general-ai":  +1,
    "coordinate-other-ais":       -1,
    "myopic-reward":              -1,
}


def load_aligned_sign(repo_root: Path) -> dict[str, int]:
    p = repo_root / "data" / "anthropic_evals.json"
    if not p.exists():
        return dict(ALIGNED_SIGN_FALLBACK)
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return dict(ALIGNED_SIGN_FALLBACK)
    out: dict[str, int] = {}
    for ds_name, items in data.items():
        if not items:
            continue
        sample = items[0]
        if "aligned_letter" in sample and "matching_letter" in sample:
            out[ds_name] = +1 if sample["aligned_letter"] == sample["matching_letter"] else -1
        else:
            out[ds_name] = ALIGNED_SIGN_FALLBACK.get(ds_name, +1)
    return out


def load_experiment(exp_dir: Path) -> dict:
    """Read an experiment's manifest + eval records.

    Returns: {
        "exp_dir": str,
        "train_eval": str (the `category` from the config),
        "records": list of eval records,
    }
    """
    with open(exp_dir / "manifest.json") as f:
        manifest = json.load(f)
    train_eval = manifest["config"]["category"]
    eval_files = sorted((exp_dir / "eval").glob("eval_*.json"))
    if not eval_files:
        raise SystemExit(f"No eval JSON in {exp_dir}/eval/")
    with open(eval_files[-1]) as f:
        eval_payload = json.load(f)
    return {
        "exp_dir": str(exp_dir),
        "train_eval": train_eval,
        "records": eval_payload["results"],
        "manifest": manifest,
    }


def aligned_pct_from_match(match_pct: float, sign: int) -> float:
    return match_pct if sign == +1 else (100 - match_pct)


def compute_per_vector_per_eval(records: list[dict], aligned_sign: dict) -> tuple[dict, dict]:
    """For each (method, vector_idx, eval): {max:(scale, shift), min:(scale, shift)}.
    Plus baselines aligned-pct per eval.
    """
    grp = defaultdict(list)
    for r in records:
        key = (r["vector_type"], r["vector_idx"], r["dataset"], r["scale"])
        grp[key].append(r["chose_matching"])
    rates = {k: 100 * sum(v) / len(v) for k, v in grp.items()}

    base_match: dict[str, float] = {}
    for (m, vi, ds, s), pct in rates.items():
        if s == 0.0 and ds not in base_match:
            base_match[ds] = pct
    baselines = {ds: aligned_pct_from_match(p, aligned_sign[ds]) for ds, p in base_match.items()}

    shifts: dict = defaultdict(lambda: defaultdict(dict))
    methods = sorted({k[0] for k in rates})
    for method in methods:
        vec_idxs = sorted({k[1] for k in rates if k[0] == method})
        for vi in vec_idxs:
            datasets = sorted({k[2] for k in rates if k[0] == method and k[1] == vi})
            for ds in datasets:
                base_a = baselines[ds]
                cands = [
                    (s, aligned_pct_from_match(rates[(method, vi, ds, s)], aligned_sign[ds]) - base_a)
                    for (m2, vi2, ds2, s) in rates
                    if m2 == method and vi2 == vi and ds2 == ds
                ]
                shifts[method][vi][ds] = {
                    "max": max(cands, key=lambda t: t[1]),
                    "min": min(cands, key=lambda t: t[1]),
                }
    return shifts, baselines


def pick_generalist(shifts_for_method: dict, direction: str) -> tuple[int, float]:
    """For a method's vectors (idx -> {ds -> {max, min}}), pick the one with
    best mean (aligned: max-over-scales / misaligned: min-over-scales)
    alignment shift across all evals it has data for. Returns (vec_idx, mean_score).
    """
    pick = "max" if direction == "aligned" else "min"
    if direction == "aligned":
        initial = float("-inf")
        is_better = lambda new, old: new > old
    else:
        initial = float("+inf")
        is_better = lambda new, old: new < old
    best: tuple[int | None, float] = (None, initial)
    for vi, by_ds in shifts_for_method.items():
        mean_score = float(np.mean([by_ds[ds][pick][1] for ds in by_ds]))
        if is_better(mean_score, best[1]):
            best = (vi, mean_score)
    return best


def build_matrix(experiments: list[dict], direction: str) -> dict:
    """For each (train_eval, method), pick best generalist; record cells per test eval.
    Returns a nested dict: matrix[train_eval][method] = {
        "vector_idx": int, "mean_score": float,
        "cells": {test_eval: {"scale": float, "shift_pp": float}}
    }
    """
    repo_root = Path(__file__).resolve().parent.parent
    aligned_sign = load_aligned_sign(repo_root)

    matrix: dict = {}
    for exp in experiments:
        train = exp["train_eval"]
        shifts, _ = compute_per_vector_per_eval(exp["records"], aligned_sign)
        per_method = {}
        for method in METHOD_ORDER:
            if method not in shifts:
                continue
            vi, mean_score = pick_generalist(shifts[method], direction)
            if vi is None:
                continue
            cells = {}
            pick_key = "max" if direction == "aligned" else "min"
            for ds, by in shifts[method][vi].items():
                s, sh = by[pick_key]
                cells[ds] = {"scale": s, "shift_pp": sh}
            per_method[method] = {
                "vector_idx": vi,
                "mean_score": mean_score,
                "cells": cells,
            }
        matrix[train] = per_method
    return matrix


def render_figure(
    matrix: dict,
    direction: str,
    train_evals: list[str],
    test_evals: list[str],
    out_path: Path,
    sources: list[dict],
) -> Path:
    n_rows, n_cols = len(train_evals), len(test_evals)
    fig, ax = plt.subplots(figsize=(16, 12))

    color_data = np.full((n_rows, n_cols), np.nan)

    for r, train in enumerate(train_evals):
        per_method = matrix.get(train, {})
        for c, test in enumerate(test_evals):
            row_vals = [
                (m, per_method[m]["cells"].get(test, {}).get("shift_pp"))
                for m in METHOD_ORDER if m in per_method
            ]
            row_vals = [(m, v) for m, v in row_vals if v is not None]
            if not row_vals:
                continue
            best = max(row_vals, key=lambda t: t[1]) if direction == "aligned" else min(row_vals, key=lambda t: t[1])
            color_data[r, c] = best[1]

    vmax = max(50, np.nanmax(np.abs(color_data)) if np.isfinite(color_data).any() else 50)
    im = ax.imshow(color_data, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)

    for r, train in enumerate(train_evals):
        per_method = matrix.get(train, {})
        for c, test in enumerate(test_evals):
            populated = []
            for m in METHOD_ORDER:
                if m in per_method and test in per_method[m]["cells"]:
                    cell = per_method[m]["cells"][test]
                    populated.append((m, per_method[m]["vector_idx"], cell["scale"], cell["shift_pp"]))
            if populated:
                best = (max if direction == "aligned" else min)(populated, key=lambda t: t[3])
                best_method = best[0]
            else:
                best_method = None
            for i, m in enumerate(METHOD_ORDER):
                y_offset = -0.30 + i * 0.30
                if m not in per_method or test not in per_method[m]["cells"]:
                    ax.text(c, r + y_offset, "—", ha="center", va="center",
                            color=METHOD_COLORS[m], fontsize=8, alpha=0.4)
                    continue
                cell = per_method[m]["cells"][test]
                vi = per_method[m]["vector_idx"]
                weight = "bold" if m == best_method else "normal"
                sign = "+" if cell["scale"] >= 0 else ""
                txt = f"{cell['shift_pp']:+.0f}  v{vi}@{sign}{int(cell['scale'])}"
                ax.text(c, r + y_offset, txt, ha="center", va="center",
                        color=METHOD_COLORS[m], fontsize=8, weight=weight)

    # Row labels: train eval + per-method vector ids and mean scores
    row_labels = []
    for train in train_evals:
        per_method = matrix.get(train, {})
        bits = [train.replace("-", "\n", 1)]
        for m in METHOD_ORDER:
            if m in per_method:
                vi = per_method[m]["vector_idx"]
                ms = per_method[m]["mean_score"]
                bits.append(f"{METHOD_LABEL[m][:3]}.v{vi}({ms:+.0f})")
        row_labels.append("\n".join(bits))

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([t.replace("-", "\n", 1) for t in test_evals], fontsize=8)
    ax.set_xlabel("Test eval")
    ax.set_ylabel(
        "Train eval (the eval the vector was trained against)\n"
        f"per-method best generalist by mean {'aligned' if direction == 'aligned' else 'mis-aligned'} shift across cols"
    )
    ax.set_title(
        f"Per-train-eval × test-eval transfer matrix — {direction} direction (Qwen3-14B)\n"
        f"each row's vector = best generalist of that method among row's pipeline vectors"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Cell-winner alignment shift (pp)", fontsize=9)

    handles = [plt.Line2D([0], [0], marker="s", markersize=10, linestyle="",
                          color=METHOD_COLORS[m], label=METHOD_LABEL[m])
               for m in METHOD_ORDER]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=9)

    src_lines = "; ".join(f"{Path(s['exp_dir']).name}" for s in sources)
    sub = (
        f"Sources: {src_lines}. "
        f"Per (row, method): vector with best mean {'aligned' if direction == 'aligned' else 'mis-aligned'} "
        f"shift across all 7 cols among that method's vectors trained on the row eval. "
        f"Cell value: that vector on the test eval at best scale. Bold = cell winner."
    )
    fig.text(0.01, 0.005, sub, fontsize=7.5, color="#444", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_sidecar(out_path: Path, matrix: dict, direction: str, sources: list[dict]) -> Path:
    sidecar = out_path.with_suffix(".json")
    payload = {
        "figure": out_path.name,
        "direction": direction,
        "metric": (
            "Per (row=train_eval, col=test_eval, method): the alignment shift achieved by "
            "that method's best generalist vector (highest mean shift across cols among vectors "
            "trained on the row eval) when applied to the col eval at best scale."
        ),
        "sources": sources,
        "matrix": matrix,
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar


def write_readme(out_dir: Path, sources: list[dict], train_evals: list[str]) -> Path:
    p = out_dir / "README.md"
    lines = [
        "# Per-train-eval × test-eval transfer matrix (Qwen3-14B)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## What's here",
        "",
        "- `aligned.png` / `aligned.json` — 7×7 transfer matrix, aligned direction.",
        "- `misaligned.png` / `misaligned.json` — same, misaligned direction.",
        "",
        "## Layout",
        "",
        "- Rows: 7 *train* evals. The pipeline was run once per row eval, with",
        "  `category` set to that eval. PI/MELBO trained on a single prompt",
        "  picked from that eval (seed=0); CAA trained on 100 disjoint prompts",
        "  from that eval (CAA `direction='aligned'` — the polarity-aware",
        "  contrast so + scale always = HHH-aligned).",
        "- Cols: 7 *test* evals.",
        "- Cells: 3 numbers per cell (CAA blue, MELBO green, PI orange).",
        "  Each is that method's *best generalist* (vector with highest mean",
        "  alignment shift across all 7 cols among the vectors produced by",
        "  the row's pipeline) evaluated on the col eval at the best scale.",
        "  Bold = the method that won this cell.",
        "",
        "## Source experiments",
        "",
    ]
    for s in sources:
        lines.append(f"- `{s['exp_dir']}` — train eval: `{s['train_eval']}`")
    lines.extend([
        "",
        "All runs: Qwen/Qwen3-14B, source layer 10, target layer 32,",
        "CAA layer 24, sample_seed=42, max_questions=100, scales",
        "`[-25,-10,-5,-2,-1,0,1,2,5,10,25]`, PI pad=5, CAA direction=aligned.",
        "",
        "## Caveats",
        "",
        "- Training prompts for PI/MELBO are picked via `seed=0` random choice",
        "  from the train eval's question pool. Different seeds would give",
        "  different vectors (and likely different transfer profiles).",
        "- CAA train pool excludes the same 100 questions used for eval",
        "  (`exclude_test=true, num_test=100, test_seed=42`), so per-row",
        "  CAA vectors are trained on disjoint prompts from the test sample.",
        "- Baselines may drift by 0-2 questions across runs (cuDNN",
        "  nondeterminism in batched matmul). Each row's cells use that",
        "  experiment's own baseline so per-row shifts are honest.",
        "",
    ])
    with open(p, "w") as f:
        f.write("\n".join(lines))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiments-dir", type=Path, default=None,
        help="Auto-discover experiment dirs whose names start with 'qwen3_14b_train_'",
    )
    ap.add_argument(
        "--exp-dirs", nargs="+", type=Path, default=None,
        help="Explicit list of experiment dirs (overrides --experiments-dir)",
    )
    ap.add_argument("--out-name", required=True,
                    help="Subdirectory under analysis/ to write to")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.exp_dirs:
        exp_dirs = list(args.exp_dirs)
    else:
        base = args.experiments_dir or (repo_root / "experiments")
        exp_dirs = sorted([p for p in base.iterdir()
                           if p.is_dir() and p.name.startswith("qwen3_14b_train_")])

    if not exp_dirs:
        ap.error("No experiment dirs found. Pass --exp-dirs or run the pipelines first.")

    print(f"Loading {len(exp_dirs)} experiments:")
    experiments = []
    for d in exp_dirs:
        print(f"  - {d}")
        experiments.append(load_experiment(d))

    train_evals = [e["train_eval"] for e in experiments]
    # Test evals = union over all experiments' eval datasets
    test_evals = sorted({r["dataset"] for e in experiments for r in e["records"]})

    print(f"\nTrain evals: {train_evals}")
    print(f"Test  evals: {test_evals}")

    out_dir = repo_root / "analysis" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [{"exp_dir": e["exp_dir"], "train_eval": e["train_eval"]} for e in experiments]

    for direction in ("aligned", "misaligned"):
        matrix = build_matrix(experiments, direction)
        out_path = out_dir / f"{direction}.png"
        render_figure(matrix, direction, train_evals, test_evals, out_path, sources)
        write_sidecar(out_path, matrix, direction, sources)
        print(f"Wrote {out_path}")
        print(f"Wrote {out_path.with_suffix('.json')}")

    write_readme(out_dir, sources, train_evals)
    print(f"Wrote {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
