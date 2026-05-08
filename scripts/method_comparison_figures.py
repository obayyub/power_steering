#!/usr/bin/env python3
"""Render two method-comparison figures (positive / negative direction).

For each (method, dataset) cell:
  - Pulls every (vector, scale) record for that method on that dataset.
  - Picks the (vector, scale) that maximizes alignment shift (positive
    direction) or minimizes it (negative direction). Sign flip is
    implicit since the scale range covers ±values.
  - Plots three bars (CAA / MELBO / PI) per dataset; bar value is the
    alignment shift (in pp), with the (vector_idx, scale) annotated
    on the bar.

Sources:
  - PI vectors come from the experiments/<exp_pi>/eval/eval_*.json
    (run with pad=5).
  - MELBO + CAA vectors come from a different experiment dir (the prior
    full Phase D run); pass via --exp-melbo-caa.

Output:
  analysis/<name>/aligned.png    + sidecar JSON
  analysis/<name>/misaligned.png + sidecar JSON
  analysis/<name>/README.md      describing source experiments + caveats

Usage:
  uv run python scripts/method_comparison_figures.py \
      --exp-pi      experiments/20260503_005015_Qwen3-14B \
      --exp-melbo-caa experiments/20260502_190738_Qwen3-14B \
      --out-name    2026-05-03_method_comparison_14b
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


# Fallback polarity table used only if data/anthropic_evals.json doesn't
# carry per-item `aligned_letter`. New data files written by
# download_dataset.py include aligned_letter; we read that and infer the
# polarity per dataset at runtime instead of hardcoding here.
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

METHOD_ORDER = ["caa", "melbo", "pi"]
METHOD_COLORS = {"caa": "#1f77b4", "melbo": "#2ca02c", "pi": "#ff7f0e"}
METHOD_LABEL = {"caa": "CAA", "melbo": "MELBO", "pi": "PI (pad=5)"}


def load_records(exp_dir: Path) -> tuple[list[dict], dict]:
    eval_files = sorted((exp_dir / "eval").glob("eval_*.json"))
    if not eval_files:
        raise SystemExit(f"No eval JSON in {exp_dir}/eval/")
    with open(eval_files[-1]) as f:
        payload = json.load(f)
    return payload["results"], {
        "exp_dir": str(exp_dir),
        "eval_file": eval_files[-1].name,
        "model": payload.get("model"),
    }


def compute_per_dataset_baseline(records: list[dict]) -> dict[str, float]:
    """match% at scale=0 per dataset (any vector — they're identical at 0)."""
    out: dict[str, list[bool]] = defaultdict(list)
    seen: dict[str, tuple] = {}
    for r in records:
        if r["scale"] != 0.0:
            continue
        vkey = (r["vector_type"], r["vector_idx"])
        if r["dataset"] in seen and seen[r["dataset"]] != vkey:
            continue  # only count one vector's baseline; they're the same
        seen[r["dataset"]] = vkey
        out[r["dataset"]].append(r["chose_matching"])
    return {ds: 100 * sum(v) / len(v) for ds, v in out.items()}


def aligned_pct_from_match(ds: str, match_pct: float) -> float:
    return match_pct if ALIGNED_SIGN[ds] == +1 else (100 - match_pct)


def best_per_method_dataset(
    records: list[dict],
    direction: str,
) -> dict[tuple[str, str], dict]:
    """For each (method, dataset), find the (vector, scale) that maximizes
    (direction='aligned') or minimizes (direction='misaligned') the
    alignment shift relative to baseline.
    """
    # Group by (method, vector_idx, dataset, scale) -> match% rate
    grp: dict[tuple, list[bool]] = defaultdict(list)
    for r in records:
        key = (r["vector_type"], r["vector_idx"], r["dataset"], r["scale"])
        grp[key].append(r["chose_matching"])
    rates = {k: 100 * sum(v) / len(v) for k, v in grp.items()}

    baselines = compute_per_dataset_baseline(records)

    methods = sorted({k[0] for k in rates})
    datasets = sorted({k[2] for k in rates})

    chooser = max if direction == "aligned" else min

    out: dict[tuple[str, str], dict] = {}
    for method in methods:
        for ds in datasets:
            base_aligned = aligned_pct_from_match(ds, baselines[ds])
            cands = [
                (vi, s, aligned_pct_from_match(ds, rates[(method, vi, ds, s)]))
                for (m, vi, ds2, s) in rates
                if m == method and ds2 == ds
            ]
            best = chooser(cands, key=lambda t: t[2])
            vi, s, aligned = best
            shift = aligned - base_aligned
            out[(method, ds)] = {
                "vector_idx": vi,
                "scale": s,
                "aligned_pct": aligned,
                "baseline_aligned_pct": base_aligned,
                "shift_pp": shift,
            }
    return out


def render_figure(
    cells: dict[tuple[str, str], dict],
    direction: str,
    out_path: Path,
    title: str,
    sources: dict,
) -> Path:
    datasets = sorted({ds for _, ds in cells})
    n_ds = len(datasets)
    width = 0.25
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    for i, method in enumerate(METHOD_ORDER):
        offsets = (i - 1) * width
        vals = [cells[(method, ds)]["shift_pp"] for ds in datasets]
        bars = ax.bar(
            x + offsets, vals, width=width,
            color=METHOD_COLORS[method], label=METHOD_LABEL[method],
            edgecolor="black", linewidth=0.4,
        )
        for ds, b in zip(datasets, bars):
            cell = cells[(method, ds)]
            sign = "+" if cell["scale"] >= 0 else ""
            label = f"{cell['shift_pp']:+.0f}\nv{cell['vector_idx']} @ {sign}{cell['scale']:.0f}"
            y = b.get_height()
            va = "bottom" if y >= 0 else "top"
            offset = 1.2 if y >= 0 else -1.2
            ax.text(
                b.get_x() + b.get_width() / 2, y + offset, label,
                ha="center", va=va, fontsize=7,
            )

    ax.axhline(y=0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace("-", "\n", 1) for ds in datasets], fontsize=9)
    ax.set_ylabel("Alignment shift vs baseline (pp)\n+ = pushes toward HHH-aligned answer")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    sub = (
        f"PI from {Path(sources['pi']['exp_dir']).name} (pad=5); "
        f"MELBO+CAA from {Path(sources['melbo_caa']['exp_dir']).name}. "
        f"Each cell: best (vector, scale) for that method on that dataset; "
        f"sign of scale represents the chosen direction."
    )
    fig.text(0.01, 0.005, sub, fontsize=7.5, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_sidecar(out_path: Path, cells: dict, direction: str, sources: dict) -> Path:
    """Sidecar JSON with full reproduction info."""
    sidecar = out_path.with_suffix(".json")
    payload = {
        "figure": out_path.name,
        "direction": direction,
        "metric": "alignment_shift_pp (best aligned-match% over (vector, scale) - baseline aligned-match%)",
        "aligned_sign": ALIGNED_SIGN,
        "sources": sources,
        "cells": [
            {
                "method": method, "dataset": ds,
                **cell,
            }
            for (method, ds), cell in cells.items()
        ],
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar


def write_readme(
    out_dir: Path,
    aligned_cells: dict,
    misaligned_cells: dict,
    sources: dict,
) -> Path:
    p = out_dir / "README.md"
    lines = [
        f"# Method comparison — Qwen3-14B (CAA / MELBO / PI(pad=5))",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## What's here",
        "",
        "Two grouped-bar figures comparing the three methods on each of the 7",
        "Anthropic advanced-ai-risk evals.",
        "",
        "- **`aligned.png`** — for each (method, dataset), shows the best",
        "  achievable alignment shift across all of that method's vectors and",
        "  all scales. Sign of scale is shown so you can see which direction",
        "  each method was pushed in. + = HHH-aligned shift.",
        "- **`misaligned.png`** — same but picks the (vector, scale) that",
        "  maximally pushes the model AWAY from aligned (most negative shift).",
        "- **`aligned.json`**, **`misaligned.json`** — full per-cell data:",
        "  vector_idx, scale chosen, aligned-match% achieved, baseline,",
        "  alignment shift in pp, polarity table, source paths.",
        "",
        "## Sources",
        "",
        f"- **PI (pad=5) vectors**: `{sources['pi']['exp_dir']}` "
        f"(eval: `{sources['pi']['eval_file']}`)",
        f"- **MELBO + CAA vectors**: `{sources['melbo_caa']['exp_dir']}` "
        f"(eval: `{sources['melbo_caa']['eval_file']}`)",
        "",
        "Both runs: Qwen/Qwen3-14B, source layer 10, target layer 32,",
        "CAA layer 24, sample_seed=42, max_questions=100, scales",
        "`[-25,-10,-5,-2,-1,0,1,2,5,10,25]`. Same training prompt (seed=0",
        "selects index 197 of corrigible-neutral-HHH).",
        "",
        "Baselines drift by 0-2 questions across runs (cuDNN nondeterminism",
        "in batched matmul affecting argmax tiebreaks). Each cell uses its",
        "own source experiment's baseline so per-method shifts are honest.",
        "",
        "## How `aligned` is defined per dataset",
        "",
        "Anthropic's `answer_matching_behavior` field is HHH-aligned for some",
        "evals and the named-misaligned behavior for others. We sign-correct",
        "downstream so HIGHER ALWAYS = MORE ALIGNED:",
        "",
        "| Dataset | matching = aligned? |",
        "|---|---|",
    ]
    for ds, sign in ALIGNED_SIGN.items():
        lines.append(f"| {ds} | {'yes' if sign == +1 else 'no (flipped)'} |")
    lines.extend([
        "",
        "## Aligned-direction summary (positive figure)",
        "",
        "Per-dataset best across all methods:",
        "",
        "| Dataset | Winner | Shift (pp) | Vector @ scale |",
        "|---|---|---:|---|",
    ])
    datasets = sorted({ds for _, ds in aligned_cells})
    for ds in datasets:
        ranked = sorted(
            [(method, aligned_cells[(method, ds)]) for method in METHOD_ORDER],
            key=lambda t: t[1]["shift_pp"], reverse=True,
        )
        winner_m, winner = ranked[0]
        sign = "+" if winner["scale"] >= 0 else ""
        lines.append(
            f"| {ds} | {METHOD_LABEL[winner_m]} | "
            f"{winner['shift_pp']:+.1f} | "
            f"v{winner['vector_idx']} @ {sign}{winner['scale']:.0f} |"
        )
    lines.extend([
        "",
        "## Misaligned-direction summary (negative figure)",
        "",
        "Per-dataset most-destructive across all methods:",
        "",
        "| Dataset | Winner | Shift (pp) | Vector @ scale |",
        "|---|---|---:|---|",
    ])
    for ds in datasets:
        ranked = sorted(
            [(method, misaligned_cells[(method, ds)]) for method in METHOD_ORDER],
            key=lambda t: t[1]["shift_pp"],
        )
        winner_m, winner = ranked[0]
        sign = "+" if winner["scale"] >= 0 else ""
        lines.append(
            f"| {ds} | {METHOD_LABEL[winner_m]} | "
            f"{winner['shift_pp']:+.1f} | "
            f"v{winner['vector_idx']} @ {sign}{winner['scale']:.0f} |"
        )
    lines.extend([
        "",
        "## Generalist analysis",
        "",
        "Deferred — the question of how to define 'generalist' (mean across",
        "evals? median? worst-case? count of wins?) is open. Once decided,",
        "follow-up figure: top-K vectors by chosen generalist metric, shown",
        "across all 7 evals. See session note 2026-05-02.md for context.",
        "",
    ])
    with open(p, "w") as f:
        f.write("\n".join(lines))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-pi", required=True, type=Path)
    ap.add_argument("--exp-melbo-caa", required=True, type=Path)
    ap.add_argument("--out-name", required=True,
                    help="Subdirectory under analysis/ to write to")
    args = ap.parse_args()

    pi_records, pi_meta = load_records(args.exp_pi)
    other_records, other_meta = load_records(args.exp_melbo_caa)

    # Filter: PI from one source, MELBO+CAA from the other
    pi_only = [r for r in pi_records if r["vector_type"] == "pi"]
    melbo_caa = [r for r in other_records if r["vector_type"] in ("melbo", "caa")]

    methods_in_pi = sorted({r["vector_type"] for r in pi_only})
    methods_in_other = sorted({r["vector_type"] for r in melbo_caa})
    print(f"Loaded {len(pi_only)} PI records ({methods_in_pi}) from {args.exp_pi.name}")
    print(f"Loaded {len(melbo_caa)} MELBO+CAA records ({methods_in_other}) from {args.exp_melbo_caa.name}")

    # Combine for the per-method analysis (each method only has its own records)
    combined = pi_only + melbo_caa

    aligned_cells = best_per_method_dataset(combined, "aligned")
    misaligned_cells = best_per_method_dataset(combined, "misaligned")

    out_dir = Path(__file__).resolve().parent.parent / "analysis" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "pi": pi_meta,
        "melbo_caa": other_meta,
    }

    aligned_png = render_figure(
        aligned_cells, "aligned", out_dir / "aligned.png",
        title="Best aligned shift per method × dataset (Qwen3-14B)",
        sources=sources,
    )
    write_sidecar(aligned_png, aligned_cells, "aligned", sources)

    misaligned_png = render_figure(
        misaligned_cells, "misaligned", out_dir / "misaligned.png",
        title="Best MIS-aligned shift per method × dataset (Qwen3-14B)",
        sources=sources,
    )
    write_sidecar(misaligned_png, misaligned_cells, "misaligned", sources)

    readme = write_readme(out_dir, aligned_cells, misaligned_cells, sources)

    print(f"\nWrote:")
    print(f"  {aligned_png}")
    print(f"  {misaligned_png}")
    print(f"  {readme}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
