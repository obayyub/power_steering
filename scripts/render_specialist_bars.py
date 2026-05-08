#!/usr/bin/env python3
"""Per-eval specialist bar chart.

For each of the 7 evals and each method (CAA / MELBO / PI), find the vector
*trained on that eval* that maximizes (or minimizes) alignment shift *on
that eval* — i.e., the true home-eval specialist. Render as a grouped bar
chart: 7 dataset groups × 3 method bars.

Two figures: aligned and misaligned.

Usage:
    uv run python scripts/render_specialist_bars.py \
        --experiments-dir experiments \
        --analysis-dir   analysis/2026-05-03_per_eval_matrix_14b
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


def find_specialist(records: list[dict], train_eval: str, method: str,
                    aligned_sign: dict, direction: str) -> dict | None:
    """Best (vector, scale) of `method` evaluated on `train_eval` (home eval)."""
    rates = defaultdict(list)
    for r in records:
        if r["vector_type"] != method or r["dataset"] != train_eval:
            continue
        rates[(r["vector_idx"], r["scale"])].append(r["chose_matching"])
    if not rates:
        return None
    base_match = None
    for (vi, s), v in rates.items():
        if s == 0.0:
            base_match = 100 * sum(v) / len(v)
            break
    base_a = aligned_pct(base_match, aligned_sign[train_eval])

    cands = []
    for (vi, s), v in rates.items():
        pct = 100 * sum(v) / len(v)
        shift = aligned_pct(pct, aligned_sign[train_eval]) - base_a
        cands.append({"vector_idx": vi, "scale": s, "shift_pp": shift})
    chooser = max if direction == "aligned" else min
    return chooser(cands, key=lambda d: d["shift_pp"])


def render_figure(specialists: dict, evals: list[str], direction: str,
                  out_path: Path, sources: list[dict]) -> Path:
    n_ds = len(evals)
    width = 0.25
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    for i, method in enumerate(METHOD_ORDER):
        offsets = (i - 1) * width
        vals = [specialists.get((ds, method), {}).get("shift_pp", 0.0) for ds in evals]
        bars = ax.bar(
            x + offsets, vals, width=width,
            color=METHOD_COLORS[method], label=METHOD_LABEL[method],
            edgecolor="black", linewidth=0.4,
        )
        for ds, b in zip(evals, bars):
            d = specialists.get((ds, method))
            if d is None:
                continue
            sign = "+" if d["scale"] >= 0 else ""
            label = f"{d['shift_pp']:+.0f}\nv{d['vector_idx']}@{sign}{int(d['scale'])}"
            y = b.get_height()
            va = "bottom" if y >= 0 else "top"
            offset = 1.2 if y >= 0 else -1.2
            ax.text(b.get_x() + b.get_width() / 2, y + offset, label,
                    ha="center", va=va, fontsize=7)

    ax.axhline(y=0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace("-", "\n", 1) for ds in evals], fontsize=9)
    ax.set_ylabel("Specialist alignment shift on home eval (pp)\n+ = pushes toward HHH-aligned answer")
    ax.set_title(
        f"Per-eval SPECIALIST shift — {direction} direction (Qwen3-14B)\n"
        "each bar = best vector of that method TRAINED ON THAT EVAL, evaluated on its home eval"
    )
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    src_lines = ", ".join(Path(s["exp_dir"]).name for s in sources)
    sub = (
        f"Sources: {src_lines}. Each bar value = max-over-scales (or min) "
        f"alignment shift on the bar's eval, using vectors trained on that eval. "
        f"Annotation shows the vector_idx and scale that achieved it. "
        f"Compare to the per-train-eval × test-eval matrix's diagonals — those use the GENERALIST "
        f"(best mean shift across all evals), which can be a different vector than the specialist."
    )
    fig.text(0.01, 0.005, sub, fontsize=7.5, color="#444", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_sidecar(out_path: Path, specialists: dict, direction: str, sources: list[dict]) -> Path:
    sidecar = out_path.with_suffix(".json")
    payload = {
        "figure": out_path.name,
        "direction": direction,
        "metric": "Per (eval, method) SPECIALIST: vector trained on the eval, "
                  "max/min alignment shift on the eval at best scale.",
        "sources": sources,
        "cells": [
            {"eval": ds, "method": m, **info}
            for (ds, m), info in specialists.items()
        ],
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", type=Path, default=None)
    ap.add_argument("--analysis-dir", type=Path, required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    base = args.experiments_dir or (repo_root / "experiments")
    exp_dirs = sorted([p for p in base.iterdir()
                       if p.is_dir() and p.name.startswith("qwen3_14b_train_")])
    if not exp_dirs:
        ap.error("No qwen3_14b_train_* experiments found.")

    aligned_sign = load_aligned_sign(repo_root)
    experiments = [load_experiment(d) for d in exp_dirs]
    evals = sorted({e["train_eval"] for e in experiments})

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    sources = [{"exp_dir": e["exp_dir"], "train_eval": e["train_eval"]} for e in experiments]

    for direction in ("aligned", "misaligned"):
        specialists: dict[tuple[str, str], dict] = {}
        for exp in experiments:
            train = exp["train_eval"]
            for method in METHOD_ORDER:
                spec = find_specialist(exp["records"], train, method, aligned_sign, direction)
                if spec is not None:
                    specialists[(train, method)] = spec

        out_path = args.analysis_dir / f"specialists_{direction}.png"
        render_figure(specialists, evals, direction, out_path, sources)
        write_sidecar(out_path, specialists, direction, sources)
        print(f"Wrote {out_path}")
        print(f"Wrote {out_path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
