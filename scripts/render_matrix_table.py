#!/usr/bin/env python3
"""Render the per-train-eval × test-eval matrix as a markdown table.

Layout:
  one row per (train_eval, method) combination = 21 rows total
  one column per test eval = 7 columns
  cell value = best generalist's alignment shift on the test eval, plus
  the (vector_idx, scale) annotation. Per (train_eval, test_eval) the
  winning method's cell is bolded with **markdown**.

Reads both sidecar JSONs from the analysis dir. Writes:
  <analysis_dir>/aligned_table.md
  <analysis_dir>/misaligned_table.md

Usage:
  uv run python scripts/render_matrix_table.py \
      --analysis-dir analysis/2026-05-03_per_eval_matrix_14b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_ORDER = ["caa", "melbo", "pi"]
METHOD_LABEL = {"caa": "CAA", "melbo": "MELBO", "pi": "PI"}


def short(ds: str) -> str:
    """Compact dataset name for table headers."""
    return {
        "coordinate-other-ais": "coord-other",
        "corrigible-neutral-HHH": "corrigible",
        "myopic-reward": "myopic",
        "power-seeking-inclination": "power-seek",
        "self-awareness-general-ai": "self-aware",
        "survival-instinct": "survival",
        "wealth-seeking-inclination": "wealth-seek",
    }.get(ds, ds)


def fmt_cell(cell: dict | None, vector_idx: int | None = None) -> str:
    if cell is None:
        return "—"
    s = cell["scale"]
    sign = "+" if s >= 0 else ""
    if vector_idx is not None:
        return f"{cell['shift_pp']:+.0f} v{vector_idx}@{sign}{int(s)}"
    return f"{cell['shift_pp']:+.0f} @{sign}{int(s)}"


def render_table(matrix: dict, train_evals: list[str], test_evals: list[str], direction: str) -> str:
    """Build markdown table. Bold the winning method per (train, test)."""
    pick_max = direction == "aligned"
    lines: list[str] = []
    # Header
    header = "| Train | Method | " + " | ".join(short(t) for t in test_evals) + " |"
    align = "|---|---|" + "---:|" * len(test_evals)
    lines.append(header)
    lines.append(align)

    for train in train_evals:
        per_method = matrix.get(train, {})
        # First decide winners per cell across methods
        for ti, test in enumerate(test_evals):
            pass  # winners computed inline below

        # Render 3 rows
        for mi, method in enumerate(METHOD_ORDER):
            info = per_method.get(method)
            row_cells: list[str] = []
            train_cell = short(train) if mi == 0 else ""
            for test in test_evals:
                # Determine winner across all methods for this (train, test)
                cells_by_m = {}
                for m in METHOD_ORDER:
                    minfo = per_method.get(m)
                    if minfo and test in minfo["cells"]:
                        cells_by_m[m] = minfo["cells"][test]
                if not cells_by_m:
                    winner = None
                else:
                    chooser = max if pick_max else min
                    winner = chooser(cells_by_m.items(), key=lambda t: t[1]["shift_pp"])[0]

                if not info or test not in info["cells"]:
                    row_cells.append("—")
                    continue
                cell_str = fmt_cell(info["cells"][test], vector_idx=info.get("vector_idx"))
                if method == winner:
                    cell_str = f"**{cell_str}**"
                row_cells.append(cell_str)

            lines.append(f"| {train_cell} | {METHOD_LABEL[method]} | " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, required=True)
    args = ap.parse_args()

    for direction in ("aligned", "misaligned"):
        sidecar = args.analysis_dir / f"{direction}.json"
        with open(sidecar) as f:
            payload = json.load(f)
        matrix = payload["matrix"]
        train_evals = sorted(matrix.keys())
        test_evals = sorted({t for train in matrix for m in matrix[train].values() for t in m["cells"]})

        table = render_table(matrix, train_evals, test_evals, direction)

        out_path = args.analysis_dir / f"{direction}_table.md"
        sources_short = ", ".join(
            Path(s["exp_dir"]).name for s in payload.get("sources", [])
        )
        header = (
            f"# Per-train-eval × test-eval matrix — {direction} direction (Qwen3-14B)\n\n"
            f"Rows: train eval × method (CAA / MELBO / PI). Each method's row uses\n"
            f"that method's best generalist (vector with highest mean alignment\n"
            f"shift across all 7 cols among row-trained vectors). Cell value:\n"
            f"`shift v<idx>@<scale>`. **Bold** = method that wins this (train, test) cell.\n"
            f"\nSources: {sources_short}\n\n"
        )
        with open(out_path, "w") as f:
            f.write(header)
            f.write(table)
            f.write("\n")
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
