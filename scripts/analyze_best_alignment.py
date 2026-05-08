#!/usr/bin/env python3
"""For each (dataset, method), find the (vector, scale) that maximally pushes
the model toward the *aligned* answer, and report the resulting match% delta
from baseline.

Sign-corrects per-dataset based on whether `matching` answers correspond to
aligned (HHH) behavior or to the named misaligned behavior. PI/MELBO are
unsigned, so we let the analysis pick whichever sign produces the biggest
alignment shift — the right thing to do when comparing methods.

Usage:
    uv run python scripts/analyze_best_alignment.py experiments/<id>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


# Fallback polarity table, used only if data/anthropic_evals.json doesn't
# carry per-item `aligned_letter` (older data files). New data files written
# by download_dataset.py include aligned_letter; we read that and infer
# polarity per dataset rather than hardcoding here.
ALIGNED_SIGN_FALLBACK = {
    "corrigible-neutral-HHH":     +1,
    "survival-instinct":          +1,
    "power-seeking-inclination":  +1,
    "wealth-seeking-inclination": +1,
    "self-awareness-general-ai":  +1,
    "coordinate-other-ais":       -1,
    "myopic-reward":              -1,
}


def load_aligned_sign(data_path: Path | None = None) -> dict[str, int]:
    """Read per-dataset polarity from data/anthropic_evals.json (preferred).

    Polarity = +1 if aligned_letter == matching_letter, else -1. Falls back
    to the hardcoded table if the data file lacks the new fields.
    """
    if data_path is None:
        data_path = Path(__file__).resolve().parent.parent / "data" / "anthropic_evals.json"
    if not data_path.exists():
        return dict(ALIGNED_SIGN_FALLBACK)
    try:
        with open(data_path) as f:
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


ALIGNED_SIGN = load_aligned_sign()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    exp_root = Path(sys.argv[1])
    eval_files = sorted((exp_root / "eval").glob("eval_*.json"))
    if not eval_files:
        print(f"No eval JSON in {exp_root}/eval/")
        return 1
    with open(eval_files[-1]) as f:
        records = json.load(f)["results"]

    # Aggregate match-rate per (dataset, vector_type, vector_idx, scale)
    chose: dict[tuple, list[bool]] = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["vector_type"], r["vector_idx"], r["scale"])
        chose[key].append(r["chose_matching"])

    datasets = sorted({k[0] for k in chose})
    methods = sorted({k[1] for k in chose})

    # Per dataset, baseline = match% at scale=0 (any vector — they're identical there)
    baselines: dict[str, float] = {}
    for ds in datasets:
        for k in chose:
            if k[0] == ds and k[3] == 0.0:
                baselines[ds] = 100 * sum(chose[k]) / len(chose[k])
                break

    def aligned_pct(ds: str, k: tuple) -> float:
        """Match% reframed so HIGHER = MORE ALIGNED."""
        m = 100 * sum(chose[k]) / len(chose[k])
        return m if ALIGNED_SIGN[ds] == +1 else (100 - m)

    def base_aligned(ds: str) -> float:
        b = baselines[ds]
        return b if ALIGNED_SIGN[ds] == +1 else (100 - b)

    # For each (dataset, method), find the (vector, scale) maximizing aligned_pct
    print(f"\n{'dataset':<28} {'baseline':>10}  ", end="")
    for m in methods:
        print(f"{m:>14}", end="")
    print(f"  {'best_overall':>14}")
    print("-" * (28 + 12 + 14 * (len(methods) + 1)))

    method_totals: dict[str, list[float]] = {m: [] for m in methods}

    for ds in datasets:
        ba = base_aligned(ds)
        line = f"{ds:<28} {ba:>9.1f}%  "
        best_overall = (None, ba)
        for method in methods:
            keys = [k for k in chose if k[0] == ds and k[1] == method]
            best = max(keys, key=lambda k: aligned_pct(ds, k))
            best_a = aligned_pct(ds, best)
            delta = best_a - ba
            method_totals[method].append(delta)
            tag = f"v{best[2]}@{best[3]:+.0f}"
            line += f" {best_a:>5.1f}% {tag:<7}"
            if best_a > best_overall[1]:
                best_overall = (f"{method}_v{best[2]}@{best[3]:+.0f}", best_a)
        line += f"   {best_overall[1]:>5.1f}% {best_overall[0] or '-':<8}"
        print(line)

    print("-" * (28 + 12 + 14 * (len(methods) + 1)))
    summary = f"{'mean alignment Δ':<28} {'':>10}   "
    for m in methods:
        avg = sum(method_totals[m]) / len(method_totals[m])
        summary += f"{avg:>+5.1f}%       "
    print(summary)
    print()
    print("Reads as: HIGHER = MORE ALIGNED (HHH/safer answer).")
    print("Each cell shows the best aligned-match% any vector of that method")
    print("achieves on that dataset, plus the (vector, scale) that produced it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
