"""Build the two main-body tables for the paper:

Table 1 — Specialist (diagonal). For each training eval, the best aligned-%
that each method achieves when evaluated on the SAME eval it was trained on.
Small, headline-friendly.

Table 2 — Cross-eval transfer matrix (7×7 per method). For each training eval
(rows) × test eval (columns), the best aligned-% any vector of that method
achieves on that test eval. The full transfer picture.

Output:
- Console-printed markdown tables
- `paper_artifacts/table1_specialist.md`, `paper_artifacts/table2_matrix_<method>.md`

Usage:
    uv run python scripts/build_paper_tables.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Order to render evals in tables (consistent with previous session notes)
EVAL_ORDER = [
    "corrigible-neutral-HHH",
    "survival-instinct",
    "power-seeking-inclination",
    "wealth-seeking-inclination",
    "self-awareness-general-ai",
    "coordinate-other-ais",
    "myopic-reward",
]

# Short labels for table headers (full names are too wide)
EVAL_LABEL = {
    "corrigible-neutral-HHH":     "corrig",
    "survival-instinct":          "surv",
    "power-seeking-inclination":  "power",
    "wealth-seeking-inclination": "wealth",
    "self-awareness-general-ai":  "self-aw",
    "coordinate-other-ais":       "coord",
    "myopic-reward":              "myopic",
}

# Polarity per dataset: aligned-shift = sign × matching_logit_diff.
# 5 evals: aligned == matching. 2 evals (myopic, coord-other): aligned == NOT matching.
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


# ── Eval-record loading ──────────────────────────────────────────────────────


def load_eval_records(exp_dir: Path) -> list[dict]:
    """Read the latest eval JSON from an experiment dir and return the records."""
    eval_files = sorted((exp_dir / "eval").glob("*.json"))
    if not eval_files:
        return []
    with open(eval_files[-1]) as f:
        payload = json.load(f)
    return payload.get("results", payload) if isinstance(payload, dict) else payload


def best_aligned_pct(
    records: list[dict], method: str, test_eval: str,
    direction: str = "aligned",
) -> tuple[float | None, int | None, float | None]:
    """For a given (method, test_eval, direction), return (aligned%, v, s) of
    the cell that pushes hardest in `direction`.

    direction='aligned'    → max aligned% (push toward aligned answer)
    direction='misaligned' → min aligned% (push toward misaligned answer)
    """
    sign = ALIGNED_SIGN[test_eval]

    by_cell_match = defaultdict(list)
    for r in records:
        if r["dataset"] != test_eval or r["scale"] == 0:
            continue
        if r["vector_type"] != method:
            continue
        key = (r["vector_idx"], r["scale"])
        by_cell_match[key].append(r["chose_matching"])
    if not by_cell_match:
        return None, None, None

    def aligned_pct(matches):
        n = len(matches)
        n_match = sum(1 for m in matches if m)
        match_pct = 100 * n_match / n
        return match_pct if sign > 0 else (100 - match_pct)

    selector = max if direction == "aligned" else min
    best_key = selector(by_cell_match.keys(),
                        key=lambda k: aligned_pct(by_cell_match[k]))
    return aligned_pct(by_cell_match[best_key]), best_key[0], best_key[1]


def baseline_aligned_pct(records: list[dict], test_eval: str) -> float | None:
    """Read the unsteered (scale=0) baseline aligned-% for this eval."""
    sign = ALIGNED_SIGN[test_eval]
    matches = [r["chose_matching"] for r in records
               if r["dataset"] == test_eval and r["scale"] == 0]
    if not matches:
        return None
    n_match = sum(1 for m in matches if m)
    match_pct = 100 * n_match / len(matches)
    return match_pct if sign > 0 else (100 - match_pct)


# ── Per-train-eval data assembly ─────────────────────────────────────────────


def gather_all_data(direction: str = "aligned") -> dict:
    """Return nested dict: data[train_eval][method][test_eval] = (aligned%, v, s).

    `direction` selects per-cell best in either aligned or misaligned direction.
    Walks `experiments/qwen3_14b_train_<eval>/` (PI/MELBO/CAA) and
    `experiments/qwen3_14b_dct_<eval>/` (DCT), since DCT was trained in
    its own per-eval pipelines.
    """
    data: dict = {}
    baselines: dict = {}

    for train_eval in EVAL_ORDER:
        data[train_eval] = {m: {} for m in METHODS}

        # PI/MELBO/CAA from qwen3_14b_train_<eval>
        pmc_dir = REPO / "experiments" / f"qwen3_14b_train_{train_eval}"
        pmc_records = load_eval_records(pmc_dir)
        for method in ("caa", "pi", "melbo"):
            for test_eval in EVAL_ORDER:
                pct, v, s = best_aligned_pct(
                    pmc_records, method, test_eval, direction=direction,
                )
                data[train_eval][method][test_eval] = (pct, v, s)

        # DCT from qwen3_14b_dct_<eval>
        dct_dir = REPO / "experiments" / f"qwen3_14b_dct_{train_eval}"
        dct_records = load_eval_records(dct_dir)
        for test_eval in EVAL_ORDER:
            pct, v, s = best_aligned_pct(
                dct_records, "dct", test_eval, direction=direction,
            )
            data[train_eval]["dct"][test_eval] = (pct, v, s)

        # Baselines (use whichever record set has them)
        for test_eval in EVAL_ORDER:
            for src in (pmc_records, dct_records):
                b = baseline_aligned_pct(src, test_eval)
                if b is not None:
                    baselines[test_eval] = b
                    break

    return {"data": data, "baselines": baselines}


# ── Table renderers ─────────────────────────────────────────────────────────


def render_table_1_specialist(bundle: dict, direction: str = "aligned") -> str:
    """Specialist diagonal: best aligned-% per method on its own training eval.

    direction='aligned' shows the cell maximising aligned-%;
    direction='misaligned' shows the cell minimising aligned-% (i.e.
    maximally pushing toward the misaligned answer).
    """
    data, base = bundle["data"], bundle["baselines"]
    title_dir = "aligned" if direction == "aligned" else "misaligned"
    lines = []
    lines.append(f"# Table 1 ({title_dir}) — Specialist (in-domain) best {title_dir} steering")
    lines.append("")
    lines.append(f"Each method trained on the eval shown in the row, evaluated on "
                 f"the same eval. Best (vector, scale) cell per method in the "
                 f"{title_dir} direction. Numbers are model-aligned-% (lower = "
                 f"more pushed toward misaligned).")
    lines.append("")
    header = "| Eval | baseline | " + " | ".join(m.upper() for m in METHODS) + " |"
    sep    = "|---" + "|---:" * (1 + len(METHODS)) + "|"
    lines.append(header)
    lines.append(sep)
    for ev in EVAL_ORDER:
        b = base.get(ev)
        cells = [f"{b:.0f}" if b is not None else "?"]
        for m in METHODS:
            pct, vi, sc = data[ev][m].get(ev, (None, None, None))
            if pct is None:
                cells.append("-")
            else:
                cells.append(f"{pct:.0f}")
        lines.append(f"| {EVAL_LABEL[ev]:>8} | " + " | ".join(cells) + " |")

    # Mean row
    means = {}
    for m in METHODS:
        vals = [data[ev][m].get(ev, (None,))[0] for ev in EVAL_ORDER]
        vals = [v for v in vals if v is not None]
        means[m] = sum(vals) / len(vals) if vals else None
    base_mean = sum(base[ev] for ev in EVAL_ORDER) / len(EVAL_ORDER)
    lines.append(f"| **mean** | **{base_mean:.0f}** | "
                 + " | ".join(f"**{means[m]:.0f}**" if means[m] is not None else "-"
                              for m in METHODS) + " |")

    # Mean delta from baseline (direction-appropriate sign)
    delta_lines = []
    for m in METHODS:
        if means[m] is not None:
            d = means[m] - base_mean if direction == "aligned" else base_mean - means[m]
            delta_lines.append(f"**+{d:.1f}**")
        else:
            delta_lines.append("-")
    label = "Δ aligned" if direction == "aligned" else "Δ misaligned"
    lines.append(f"| {label} | — | " + " | ".join(delta_lines) + " |")
    return "\n".join(lines)


def render_table_2_combined(bundle: dict) -> str:
    """Single 7×7 transfer matrix with all 4 methods per cell (winner bold).

    Each cell shows four lines (CAA / PI / MELBO / DCT) using `<br>` for
    in-cell line breaks. The cell-winner is bolded. Diagonal cells are also
    annotated with `★` to mark the specialist case.
    """
    data = bundle["data"]
    base = bundle["baselines"]
    lines = []
    lines.append("# Table 2 — Combined cross-eval matrix (all 4 methods per cell)")
    lines.append("")
    lines.append("Each cell shows best aligned-% for CAA / PI / MELBO / DCT respectively, "
                 "rendered top-to-bottom. **Bold** = cell winner. `★` = diagonal "
                 "(specialist) cell.")
    lines.append("")

    cols = [EVAL_LABEL[e] for e in EVAL_ORDER]
    lines.append("| train \\\\ test | base | " + " | ".join(cols) + " |")
    lines.append("|---" + "|---:" * (1 + len(cols)) + "|")

    for train_eval in EVAL_ORDER:
        row = [f"**{EVAL_LABEL[train_eval]}**", f"{base[train_eval]:.0f}"]
        for test_eval in EVAL_ORDER:
            # Collect the 4 method values
            vals = {m: data[train_eval][m].get(test_eval, (None,))[0] for m in METHODS}
            valid = {m: v for m, v in vals.items() if v is not None}
            if not valid:
                row.append("-")
                continue
            winner = max(valid, key=lambda m: valid[m])
            star = "★" if test_eval == train_eval else ""
            cell_lines = []
            for m in METHODS:
                v = vals[m]
                if v is None:
                    cell_lines.append(f"{m.upper()[:3]} —")
                    continue
                label = m.upper()[:3]
                if m == winner:
                    cell_lines.append(f"**{label} {v:.0f}{star}**")
                else:
                    cell_lines.append(f"{label} {v:.0f}")
            row.append("<br>".join(cell_lines))
        lines.append("| " + " | ".join(row) + " |")

    # Tally cell-winners by method (off-diagonal only for "fair" generalist score)
    win_counts = defaultdict(int)
    win_diag = defaultdict(int)
    win_off_diag = defaultdict(int)
    tied_cells = 0
    for train_eval in EVAL_ORDER:
        for test_eval in EVAL_ORDER:
            vals = {m: data[train_eval][m].get(test_eval, (None,))[0] for m in METHODS}
            valid = {m: v for m, v in vals.items() if v is not None}
            if not valid:
                continue
            top = max(valid.values())
            winners = [m for m, v in valid.items() if v == top]
            if len(winners) > 1:
                tied_cells += 1
            for w in winners:
                win_counts[w] += 1 / len(winners)
                if test_eval == train_eval:
                    win_diag[w] += 1 / len(winners)
                else:
                    win_off_diag[w] += 1 / len(winners)
    lines.append("")
    lines.append("**Cell-winner counts** (ties split equally; 49 cells total, 7 diagonal):")
    lines.append("")
    lines.append("| Method | Total wins | Diagonal (specialist) | Off-diagonal (generalist) |")
    lines.append("|---|---:|---:|---:|")
    for m in METHODS:
        lines.append(f"| {m.upper()} | {win_counts[m]:.1f} | "
                     f"{win_diag[m]:.1f} / 7 | {win_off_diag[m]:.1f} / 42 |")
    if tied_cells:
        lines.append(f"\n_{tied_cells} cells had ties._")
    return "\n".join(lines)


def render_table_2_matrix(bundle: dict, method: str) -> str:
    """7×7 transfer matrix for one method: rows = train eval, cols = test eval."""
    data = bundle["data"]
    base = bundle["baselines"]
    lines = []
    lines.append(f"# Table 2{'-' + method.upper()} — Cross-eval matrix ({method.upper()})")
    lines.append("")
    lines.append(f"Each cell: best aligned-% achieved by a {method.upper()} vector "
                 "trained on the row's eval, evaluated on the column's eval.")
    lines.append("")

    # Header
    cols = [EVAL_LABEL[e] for e in EVAL_ORDER]
    lines.append("| train \\\\ test | baseline | " + " | ".join(cols) + " | row mean Δ |")
    lines.append("|---" + "|---:" * (1 + len(cols) + 1) + "|")

    for train_eval in EVAL_ORDER:
        cells = [f"{base[EVAL_ORDER[0]]:.0f}"]  # placeholder
        cells = []
        deltas = []
        for test_eval in EVAL_ORDER:
            pct, vi, sc = data[train_eval][method].get(test_eval, (None,))[:3]
            if pct is None:
                cells.append("-")
            else:
                # bold diagonal (specialist)
                if test_eval == train_eval:
                    cells.append(f"**{pct:.0f}**")
                else:
                    cells.append(f"{pct:.0f}")
                b = base.get(test_eval)
                if b is not None:
                    deltas.append(pct - b)
        b_train = base.get(train_eval, 0)
        row_mean_delta = sum(deltas) / len(deltas) if deltas else 0
        lines.append(f"| {EVAL_LABEL[train_eval]:>8} | {b_train:.0f} | "
                     + " | ".join(cells)
                     + f" | {row_mean_delta:+.1f} |")

    # Column-mean delta row (each test eval averaged across train evals)
    col_means = []
    for test_eval in EVAL_ORDER:
        ds = []
        for train_eval in EVAL_ORDER:
            pct = data[train_eval][method].get(test_eval, (None,))[0]
            b = base.get(test_eval)
            if pct is not None and b is not None:
                ds.append(pct - b)
        col_means.append(sum(ds)/len(ds) if ds else 0)
    lines.append("| col mean Δ | — | " + " | ".join(f"{m:+.1f}" for m in col_means)
                 + f" | {sum(col_means)/len(col_means):+.1f} |")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    out_dir = REPO / "paper_artifacts"
    out_dir.mkdir(exist_ok=True)

    # Table 1 — both directions
    for direction in ("aligned", "misaligned"):
        print(f"\n=== Table 1 ({direction}) ===")
        bundle = gather_all_data(direction=direction)
        t1 = render_table_1_specialist(bundle, direction=direction)
        print(t1)
        (out_dir / f"table1_specialist_{direction}.md").write_text(t1 + "\n")

    # Table 2 family uses aligned-direction (the headline framing)
    bundle_aligned = gather_all_data(direction="aligned")

    # Table 2 (combined) — single 7×7 with all 4 methods per cell, winner bold
    t2c = render_table_2_combined(bundle_aligned)
    (out_dir / "table2_combined.md").write_text(t2c + "\n")

    # Table 2 (per-method)
    for method in METHODS:
        t2 = render_table_2_matrix(bundle_aligned, method)
        (out_dir / f"table2_matrix_{method}.md").write_text(t2 + "\n")

    print()
    print(f"Wrote tables to {out_dir.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
