#!/usr/bin/env python3
"""Build the B-2 generalist transfer matrix figures.

Layout: 7 rows × 7 cols
  rows = top-7 generalists by mean alignment shift across all 7 evals
         (per method — different vector per (method, rank))
  cols = the 7 test evals
  cells = up to 3 numbers (CAA / MELBO / PI) showing that method's
          R-th-best-generalist alignment shift on the test eval, best
          (most aligned for aligned figure; most misaligned for the
          misaligned figure) bolded.

Two figures: `generalists_aligned.png` and `generalists_misaligned.png`.

Vector pool: PI from the pad=5 experiment, MELBO + CAA from the
yesterday's full run. Same model / sample_seed / scales.

Generalist score per vector:
  aligned    : mean over evals of (max_over_scales of alignment_shift)
  misaligned : mean over evals of (min_over_scales of alignment_shift)

The "max" / "min" inside the mean lets each vector use whichever scale
works best per eval. The mean across evals is the generalist criterion.

Each method has its own ranked list. CAA only has 1 vector so its rank-1
slot is filled and ranks 2-7 are blank.

Usage:
  uv run python scripts/method_comparison_generalists.py \
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


# Fallback polarity table; replaced at import time with values read from
# data/anthropic_evals.json's `aligned_letter` field when present.
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
METHOD_LABEL = {"caa": "CAA", "melbo": "MELBO", "pi": "PI(pad=5)"}


def load_records(exp_dir: Path) -> tuple[list[dict], dict]:
    eval_files = sorted((exp_dir / "eval").glob("eval_*.json"))
    if not eval_files:
        raise SystemExit(f"No eval JSON in {exp_dir}/eval/")
    with open(eval_files[-1]) as f:
        payload = json.load(f)
    return payload["results"], {"exp_dir": str(exp_dir),
                                 "eval_file": eval_files[-1].name,
                                 "model": payload.get("model")}


def aligned_pct_from_match(ds: str, match_pct: float) -> float:
    return match_pct if ALIGNED_SIGN[ds] == +1 else (100 - match_pct)


def build_per_vector_per_eval(records: list[dict]) -> dict:
    """Return:
      shifts[method][vec_idx][dataset] = {"max": (scale, shift), "min": (scale, shift)}
      baselines[dataset] = aligned_pct at scale=0
    """
    grp = defaultdict(list)
    for r in records:
        key = (r["vector_type"], r["vector_idx"], r["dataset"], r["scale"])
        grp[key].append(r["chose_matching"])
    rates = {k: 100 * sum(v) / len(v) for k, v in grp.items()}

    # baseline per dataset (any vector — they're identical at scale 0)
    base_match = {}
    for (m, vi, ds, s), pct in rates.items():
        if s == 0.0 and ds not in base_match:
            base_match[ds] = pct
    baselines = {ds: aligned_pct_from_match(ds, p) for ds, p in base_match.items()}

    shifts: dict = defaultdict(lambda: defaultdict(dict))
    methods = sorted({k[0] for k in rates})
    for method in methods:
        vecs = sorted({k[1] for k in rates if k[0] == method})
        for vi in vecs:
            datasets = sorted({k[2] for k in rates if k[0] == method and k[1] == vi})
            for ds in datasets:
                base_a = baselines[ds]
                cands = [
                    (s, aligned_pct_from_match(ds, rates[(method, vi, ds, s)]) - base_a)
                    for (m, vi2, ds2, s) in rates
                    if m == method and vi2 == vi and ds2 == ds
                ]
                hi = max(cands, key=lambda t: t[1])
                lo = min(cands, key=lambda t: t[1])
                shifts[method][vi][ds] = {"max": hi, "min": lo}
    return shifts, baselines


def rank_generalists(shifts: dict, direction: str, top_k: int = 7) -> dict:
    """For each method, rank vectors by mean alignment shift (max for aligned,
    min for misaligned) across all evals. Return method -> [(vec_idx, score), ...] desc/asc.
    """
    ranked: dict[str, list[tuple[int, float]]] = {}
    pick = "max" if direction == "aligned" else "min"
    for method, by_vec in shifts.items():
        scored = []
        for vi, by_ds in by_vec.items():
            mean_shift = float(np.mean([by_ds[ds][pick][1] for ds in by_ds]))
            scored.append((vi, mean_shift))
        scored.sort(key=lambda t: t[1], reverse=(direction == "aligned"))
        ranked[method] = scored[:top_k]
    return ranked


def render_figure(
    ranked: dict,
    shifts: dict,
    direction: str,
    out_path: Path,
    title: str,
    sources: dict,
    top_k: int = 7,
) -> Path:
    datasets = sorted({ds for vec_dict in shifts["caa"].values() for ds in vec_dict})
    pick_key = "max" if direction == "aligned" else "min"

    n_rows, n_cols = top_k, len(datasets)
    fig, ax = plt.subplots(figsize=(15, 11))

    # Color cell by the winner's value (most aligned for aligned figure;
    # most misaligned -> most negative for misaligned figure).
    color_data = np.full((n_rows, n_cols), np.nan)

    for r in range(n_rows):
        for c in range(n_cols):
            ds = datasets[c]
            vals = []
            for method in METHOD_ORDER:
                if r < len(ranked[method]):
                    vi, _ = ranked[method][r]
                    s, sh = shifts[method][vi][ds][pick_key]
                    vals.append((method, vi, s, sh))
            if not vals:
                continue
            best = max(vals, key=lambda t: t[3]) if direction == "aligned" else min(vals, key=lambda t: t[3])
            color_data[r, c] = best[3]

    vmax = max(50, np.nanmax(np.abs(color_data)))
    im = ax.imshow(color_data, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)

    # Annotate
    for r in range(n_rows):
        for c in range(n_cols):
            ds = datasets[c]
            row_vals = []
            for method in METHOD_ORDER:
                if r < len(ranked[method]):
                    vi, _ = ranked[method][r]
                    s, sh = shifts[method][vi][ds][pick_key]
                    row_vals.append((method, vi, s, sh))
                else:
                    row_vals.append((method, None, None, None))
            populated = [t for t in row_vals if t[3] is not None]
            best = max(populated, key=lambda t: t[3]) if direction == "aligned" else min(populated, key=lambda t: t[3])
            best_method = best[0]

            for i, (method, vi, s, sh) in enumerate(row_vals):
                y_offset = -0.30 + i * 0.30
                if vi is None:
                    ax.text(c, r + y_offset, "—",
                            ha="center", va="center",
                            color=METHOD_COLORS[method], fontsize=8, alpha=0.4)
                    continue
                weight = "bold" if method == best_method else "normal"
                sign = "+" if s >= 0 else ""
                txt = f"{sh:+.0f}  v{vi}@{sign}{int(s)}"
                ax.text(c, r + y_offset, txt,
                        ha="center", va="center",
                        color=METHOD_COLORS[method], fontsize=8, weight=weight)

    # Row labels: rank + per-method vector ids and mean scores
    row_labels = []
    for r in range(n_rows):
        bits = [f"#{r+1}"]
        for method in METHOD_ORDER:
            if r < len(ranked[method]):
                vi, mean_score = ranked[method][r]
                bits.append(f"{METHOD_LABEL[method][:3]}.v{vi} ({mean_score:+.0f})")
        row_labels.append("\n".join(bits))

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([d.replace("-", "\n", 1) for d in datasets], fontsize=8)
    ax.set_xlabel("Test eval")
    ax.set_ylabel(
        f"Generalist rank — by mean {'aligned' if direction == 'aligned' else 'mis-aligned'} shift across all 7 evals\n"
        "(per-method vector id and mean score in row label)"
    )
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Cell-winner alignment shift (pp)", fontsize=9)

    # Legend by method color
    handles = [plt.Line2D([0], [0], marker="s", markersize=10, linestyle="",
                          color=METHOD_COLORS[m], label=METHOD_LABEL[m])
               for m in METHOD_ORDER]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=9)

    sub = (
        f"Vector pool: PI(pad=5) from {Path(sources['pi']['exp_dir']).name}, "
        f"MELBO+CAA from {Path(sources['melbo_caa']['exp_dir']).name}. "
        f"Per-method ranking by mean {'max' if direction == 'aligned' else 'min'} alignment shift across all 7 evals. "
        f"Cell value: that method's R-th generalist on the test eval, scale picked best per cell. "
        f"Bold = winner across the (up to) 3 method values in the cell."
    )
    fig.text(0.01, 0.005, sub, fontsize=7.5, color="#444", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_sidecar(out_path: Path, ranked: dict, shifts: dict, direction: str, sources: dict) -> Path:
    sidecar = out_path.with_suffix(".json")
    pick_key = "max" if direction == "aligned" else "min"
    matrix = []
    for r in range(7):
        row = {"rank": r + 1, "cells_by_dataset": {}}
        for method in METHOD_ORDER:
            if r < len(ranked[method]):
                vi, mean_score = ranked[method][r]
                cells = {}
                for ds, by in shifts[method][vi].items():
                    s, sh = by[pick_key]
                    cells[ds] = {"scale": s, "shift_pp": sh}
                row.setdefault("vectors", {})[method] = {
                    "vector_idx": vi, "mean_score": mean_score,
                    "per_eval_shift": cells,
                }
        matrix.append(row)
    payload = {
        "figure": out_path.name,
        "direction": direction,
        "metric": (
            "Per row R, per method M: M's R-th generalist (ranked by mean "
            "{aligned: max-over-scales / misaligned: min-over-scales} alignment shift across all 7 evals). "
            "Cell value = that vector's max/min alignment shift on the test eval."
        ),
        "aligned_sign": ALIGNED_SIGN,
        "sources": sources,
        "matrix": matrix,
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-pi", required=True, type=Path)
    ap.add_argument("--exp-melbo-caa", required=True, type=Path)
    ap.add_argument("--out-name", required=True)
    args = ap.parse_args()

    pi_records, pi_meta = load_records(args.exp_pi)
    other_records, other_meta = load_records(args.exp_melbo_caa)

    pi_only = [r for r in pi_records if r["vector_type"] == "pi"]
    melbo_caa = [r for r in other_records if r["vector_type"] in ("melbo", "caa")]
    combined = pi_only + melbo_caa

    shifts, baselines = build_per_vector_per_eval(combined)

    out_dir = Path(__file__).resolve().parent.parent / "analysis" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {"pi": pi_meta, "melbo_caa": other_meta}

    for direction in ("aligned", "misaligned"):
        ranked = rank_generalists(shifts, direction, top_k=7)
        out_path = out_dir / f"generalists_{direction}.png"
        render_figure(
            ranked, shifts, direction, out_path,
            title=(f"Generalist transfer matrix — {direction} direction (Qwen3-14B)\n"
                   f"top-7 generalists per method × 7 test evals"),
            sources=sources,
        )
        write_sidecar(out_path, ranked, shifts, direction, sources)
        print(f"Wrote {out_path}")
        print(f"Wrote {out_path.with_suffix('.json')}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
