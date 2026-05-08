#!/usr/bin/env python3
"""Mine the per-train-eval × test-eval matrix for interesting patterns.

Reads the sidecar JSON produced by `method_comparison_per_eval_training.py`
and reports:

  1. Diagonal vs off-diagonal stats per method (does training on X help X
     more than training on something else?).
  2. Method dominance per cell (where does CAA/MELBO/PI win, lose, tie?).
  3. Transfer clusters: which test evals get strong transfer from many
     training sources?
  4. The "selfish" train evals: which ones produce vectors that work great
     on themselves but generalize poorly?
  5. The "generous" train evals: vectors that are nearly as good elsewhere
     as on themselves.

Usage:
    uv run python scripts/analyze_per_eval_matrix.py \
        analysis/2026-05-03_per_eval_matrix_14b/aligned.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean


METHOD_LABEL = {"caa": "CAA", "melbo": "MELBO", "pi": "PI"}


def load_matrix(path: Path) -> tuple[dict, list[str], list[str], str]:
    with open(path) as f:
        payload = json.load(f)
    matrix = payload["matrix"]
    direction = payload["direction"]
    train_evals = sorted(matrix.keys())
    # Test evals = union of cells.keys() across all (train, method)
    test_evals: set[str] = set()
    for train in matrix:
        for method, info in matrix[train].items():
            test_evals.update(info["cells"].keys())
    return matrix, train_evals, sorted(test_evals), direction


def cell_value(matrix: dict, train: str, method: str, test: str) -> float | None:
    info = matrix.get(train, {}).get(method)
    if not info:
        return None
    cell = info["cells"].get(test)
    return cell["shift_pp"] if cell else None


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    matrix, train_evals, test_evals, direction = load_matrix(Path(sys.argv[1]))
    methods = ["caa", "melbo", "pi"]
    sign = +1 if direction == "aligned" else -1
    bigger = "more aligned" if direction == "aligned" else "more misaligned"

    print(f"\n=== Direction: {direction} ({bigger}) ===\n")

    # 1. Diagonal vs off-diagonal per method
    print("─── 1. Diagonal vs off-diagonal (per method) ───")
    print(f"{'method':<8} {'diag mean':>10} {'off-diag mean':>14} {'specialization Δ':>18}")
    for method in methods:
        diag, off = [], []
        for train in train_evals:
            for test in test_evals:
                v = cell_value(matrix, train, method, test)
                if v is None:
                    continue
                (diag if train == test else off).append(v)
        if not diag:
            continue
        d_m, o_m = mean(diag), mean(off)
        delta = (d_m - o_m) * sign  # positive = stronger on diagonal in the desired direction
        print(f"{METHOD_LABEL[method]:<8} {d_m:>+10.1f} {o_m:>+14.1f} {delta:>+17.1f}")
    print("\nReads as: do vectors trained on X work better on X than on others? "
          f"+ specialization Δ = yes ({bigger} on diag).\n")

    # 2. Method dominance per cell — count wins
    print("─── 2. Method wins per cell ───")
    wins = {m: 0 for m in methods}
    for train in train_evals:
        for test in test_evals:
            cells = []
            for m in methods:
                v = cell_value(matrix, train, m, test)
                if v is not None:
                    cells.append((m, v))
            if not cells:
                continue
            best = max(cells, key=lambda t: sign * t[1])
            wins[best[0]] += 1
    total = sum(wins.values())
    for m in methods:
        pct = 100 * wins[m] / total if total else 0
        print(f"  {METHOD_LABEL[m]:<8} {wins[m]:>3} / {total} cells ({pct:.0f}%)")
    print()

    # 3. Per-test-eval: which train-evals produce the best generalists?
    print("─── 3. Best train-eval source per test eval (best across methods) ───")
    print(f"{'test eval':<28} {'best train':<28} {'best method':<8} {'shift':>7}")
    for test in test_evals:
        best = (None, None, None, sign * float("-inf"))
        for train in train_evals:
            for m in methods:
                v = cell_value(matrix, train, m, test)
                if v is None:
                    continue
                if sign * v > sign * best[3]:
                    best = (train, m, m, v)
        if best[0]:
            train, m_label, m, v = best[0], METHOD_LABEL[best[1]], best[2], best[3]
            print(f"{test:<28} {train:<28} {m_label:<8} {v:>+6.1f}")
    print()

    # 4. Per-train-eval: mean off-diagonal effect (how generous)
    print("─── 4. 'Generosity' of each training source (mean off-diagonal across methods) ───")
    print(f"{'train eval':<28} {'mean off-diag (3 methods)':>26}")
    rows = []
    for train in train_evals:
        offs = []
        for test in test_evals:
            if train == test:
                continue
            for m in methods:
                v = cell_value(matrix, train, m, test)
                if v is not None:
                    offs.append(v)
        if offs:
            rows.append((train, mean(offs)))
    rows.sort(key=lambda r: sign * r[1], reverse=True)
    for train, m in rows:
        bar_chars = int(abs(m))
        bar = ("+" if m > 0 else "-") * min(bar_chars, 40)
        print(f"{train:<28} {m:>+10.1f}  {bar}")
    print(f"\nTop = best generalizing training source ({bigger} on average across other evals).\n")

    # 5. Per-test-eval: mean off-diagonal effect (how easy to push)
    print("─── 5. 'Pushability' of each test eval (mean shift from non-self trainings) ───")
    print(f"{'test eval':<28} {'mean off-diag (3 methods)':>26}")
    rows = []
    for test in test_evals:
        offs = []
        for train in train_evals:
            if train == test:
                continue
            for m in methods:
                v = cell_value(matrix, train, m, test)
                if v is not None:
                    offs.append(v)
        if offs:
            rows.append((test, mean(offs)))
    rows.sort(key=lambda r: sign * r[1], reverse=True)
    for test, m in rows:
        bar_chars = int(abs(m))
        bar = ("+" if m > 0 else "-") * min(bar_chars, 40)
        print(f"{test:<28} {m:>+10.1f}  {bar}")
    print(f"\nTop = test eval most {bigger} on average from foreign-trained vectors.\n")

    # 6. Vector-ID stability: which vector indices show up as generalists most often?
    print("─── 6. Most-frequently-picked generalist vector indices per method ───")
    from collections import Counter
    for method in methods:
        c = Counter()
        for train in train_evals:
            info = matrix.get(train, {}).get(method)
            if info:
                c[info["vector_idx"]] += 1
        if c:
            top = ", ".join(f"v{vi}({n})" for vi, n in c.most_common())
            print(f"  {METHOD_LABEL[method]:<8} {top}")
    print("\nReads as: across the 7 row pipelines, which vector_idx ranks first by mean alignment shift?")
    print("Convergence to a low rank (e.g. v0/v1) suggests that direction is principled across training prompts.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
