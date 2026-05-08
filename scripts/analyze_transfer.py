#!/usr/bin/env python3
"""Print a per-dataset transfer table for an experiment's eval results.

For each (dataset, vector) pair, shows matching% and mean matching_logit_diff
at the baseline (scale=0) and at the extreme positive/negative scales. The
"swing" column is the matching% delta between the most-positive and
most-negative scale, signed so + means positive scale increases matching.

Usage:
    uv run python scripts/analyze_transfer.py experiments/<id>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


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

    # Aggregate by (dataset, vector_type, vector_idx, scale)
    grp: dict[tuple, list[float]] = defaultdict(list)
    chose: dict[tuple, list[bool]] = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["vector_type"], r["vector_idx"], r["scale"])
        grp[key].append(r["matching_logit_diff"])
        chose[key].append(r["chose_matching"])

    datasets = sorted({k[0] for k in grp})
    vec_keys = sorted({(k[1], k[2]) for k in grp})
    scales = sorted({k[3] for k in grp})
    s_min, s_max = scales[0], scales[-1]
    s_zero = 0.0 if 0.0 in scales else None

    def pct(lst):
        return 100 * sum(lst) / len(lst)

    def mean(lst):
        return sum(lst) / len(lst)

    for ds in datasets:
        print(f"\n=== {ds} ===")
        header = f"{'vector':<10} {'base%':>6} {'base_d':>7}   "
        header += f"{f'{s_min:+}%':>7} {f'{s_min:+}_d':>8}   "
        header += f"{f'{s_max:+}%':>7} {f'{s_max:+}_d':>8}   "
        header += f"{'swing%':>7}"
        print(header)
        print("-" * len(header))
        for vt, vi in vec_keys:
            label = f"{vt}_v{vi}"
            base = chose.get((ds, vt, vi, 0.0))
            base_d = grp.get((ds, vt, vi, 0.0))
            lo = chose.get((ds, vt, vi, s_min))
            lo_d = grp.get((ds, vt, vi, s_min))
            hi = chose.get((ds, vt, vi, s_max))
            hi_d = grp.get((ds, vt, vi, s_max))
            if not (base and lo and hi):
                continue
            swing = pct(hi) - pct(lo)
            print(
                f"{label:<10} "
                f"{pct(base):>5.1f}% {mean(base_d):>+7.2f}   "
                f"{pct(lo):>6.1f}% {mean(lo_d):>+8.2f}   "
                f"{pct(hi):>6.1f}% {mean(hi_d):>+8.2f}   "
                f"{swing:>+6.1f}"
            )

    print(f"\n(scales: {scales})")
    print(f"swing% = match%(scale={s_max:+}) - match%(scale={s_min:+}); "
          f"+ means positive scale steers toward matching/HHH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
