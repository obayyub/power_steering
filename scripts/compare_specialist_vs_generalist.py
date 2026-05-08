#!/usr/bin/env python3
"""Compare the per-train-eval generalist vs the best-possible specialist.

The current matrix figure picks, for each (train_eval, method), the vector
that has the highest mean alignment shift across all cols (the GENERALIST).
The diagonal of that matrix is "generalist's score on its home eval".

This script also computes the SPECIALIST diagonal — for each (train_eval,
method), the vector with the maximum alignment shift on the home eval
(ignoring transfer). The gap between specialist and generalist diagonals
shows how much you're giving up by picking a generalist.

Reads the experiment dirs directly (not the matrix JSON) so it can pick
the per-cell maximum across all vectors and scales.

Usage:
    uv run python scripts/compare_specialist_vs_generalist.py \
        --experiments-dir experiments
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHOD_ORDER = ["caa", "melbo", "pi"]
METHOD_LABEL = {"caa": "CAA", "melbo": "MELBO", "pi": "PI"}

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
    with open(p) as f:
        data = json.load(f)
    out: dict[str, int] = {}
    for ds_name, items in data.items():
        if items and "aligned_letter" in items[0] and "matching_letter" in items[0]:
            out[ds_name] = +1 if items[0]["aligned_letter"] == items[0]["matching_letter"] else -1
        else:
            out[ds_name] = ALIGNED_SIGN_FALLBACK.get(ds_name, +1)
    return out


def aligned_pct(match_pct: float, sign: int) -> float:
    return match_pct if sign == +1 else (100 - match_pct)


def load_experiment(exp_dir: Path) -> dict:
    with open(exp_dir / "manifest.json") as f:
        manifest = json.load(f)
    eval_files = sorted((exp_dir / "eval").glob("eval_*.json"))
    with open(eval_files[-1]) as f:
        payload = json.load(f)
    return {
        "exp_dir": str(exp_dir),
        "train_eval": manifest["config"]["category"],
        "records": payload["results"],
    }


def per_vector_per_eval(records: list[dict], aligned_sign: dict) -> tuple[dict, dict]:
    """Returns:
      shifts[method][vec_idx][eval] = (best_scale, best_aligned_shift)
      baselines[eval] = aligned_pct at scale=0
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
    baselines = {ds: aligned_pct(p, aligned_sign[ds]) for ds, p in base_match.items()}

    shifts: dict = defaultdict(lambda: defaultdict(dict))
    methods = sorted({k[0] for k in rates})
    for method in methods:
        vecs = sorted({k[1] for k in rates if k[0] == method})
        for vi in vecs:
            for ds in sorted({k[2] for k in rates if k[0] == method and k[1] == vi}):
                base_a = baselines[ds]
                cands = [
                    (s, aligned_pct(rates[(method, vi, ds, s)], aligned_sign[ds]) - base_a)
                    for (m2, vi2, ds2, s) in rates
                    if m2 == method and vi2 == vi and ds2 == ds
                ]
                shifts[method][vi][ds] = max(cands, key=lambda t: t[1])
    return shifts, baselines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", type=Path, default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    base = args.experiments_dir or (repo_root / "experiments")
    exp_dirs = sorted([p for p in base.iterdir()
                       if p.is_dir() and p.name.startswith("qwen3_14b_train_")])
    if not exp_dirs:
        ap.error("No qwen3_14b_train_* experiments found.")

    aligned_sign = load_aligned_sign(repo_root)

    print(f"\n{'train eval':<28} {'method':<6} "
          f"{'specialist v@scale':<18} {'spec diag':>9}  "
          f"{'generalist v@scale':<18} {'gen diag':>8}  "
          f"{'gap':>5}")
    print("-" * 110)

    summary_by_method: dict[str, list[tuple[float, float]]] = {m: [] for m in METHOD_ORDER}

    for exp_dir in exp_dirs:
        exp = load_experiment(exp_dir)
        train_eval = exp["train_eval"]
        shifts, _ = per_vector_per_eval(exp["records"], aligned_sign)

        for method in METHOD_ORDER:
            if method not in shifts:
                continue

            # SPECIALIST: vector with max shift on the home (train) eval
            spec_vi, (spec_scale, spec_diag) = max(
                ((vi, shifts[method][vi][train_eval]) for vi in shifts[method]
                 if train_eval in shifts[method][vi]),
                key=lambda t: t[1][1],
            )

            # GENERALIST: vector with max mean shift across all evals it has data for
            gen_vi, gen_mean = None, float("-inf")
            for vi, by_ds in shifts[method].items():
                m = float(np.mean([by_ds[ds][1] for ds in by_ds]))
                if m > gen_mean:
                    gen_vi, gen_mean = vi, m
            gen_scale, gen_diag = shifts[method][gen_vi][train_eval]

            gap = spec_diag - gen_diag
            summary_by_method[method].append((spec_diag, gen_diag))
            sign = "+" if spec_scale >= 0 else ""
            sign_g = "+" if gen_scale >= 0 else ""
            print(f"{train_eval:<28} {METHOD_LABEL[method]:<6} "
                  f"v{spec_vi}@{sign}{int(spec_scale):<14} {spec_diag:>+8.1f}  "
                  f"v{gen_vi}@{sign_g}{int(gen_scale):<14} {gen_diag:>+7.1f}  "
                  f"{gap:>+5.1f}")
        print("-" * 110)

    print("\nMETHOD SUMMARY (mean across 7 train evals):")
    print(f"{'method':<8} {'spec diag mean':>14} {'gen diag mean':>14} {'gap':>6}")
    for method in METHOD_ORDER:
        rows = summary_by_method[method]
        if not rows:
            continue
        s = np.mean([r[0] for r in rows])
        g = np.mean([r[1] for r in rows])
        print(f"{METHOD_LABEL[method]:<8} {s:>+13.1f}  {g:>+13.1f}  {s-g:>+5.1f}")
    print("\nReads as: spec diag = best specialist's score on home eval. "
          "gen diag = best generalist's score on home eval. "
          "gap = how much you give up by picking the generalist.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
