#!/usr/bin/env python3
"""Cosine similarity across the 7 train evals' per-method specialist vectors.

Tests whether the specialist directions converge across training prompts.
A high |cosine| within a method's matrix suggests the method discovers a
shared model-internal axis regardless of which eval it was trained on.

Per method: identify the home-eval specialist vector for each of the 7
train evals (12 candidates per method except CAA which has 1), load the
actual unit-normalized vector tensor from the experiment dir, compute the
7×7 pairwise cosine matrix, render a heatmap.

|cosine| reading: PI eigenvectors are sign-ambiguous (v and -v are both
valid); MELBO and CAA have a more meaningful signed direction. We render
SIGNED cosine in the cells, but the diagnostic for "same axis?" is
|cosine|, so the colorbar centers on 0 with a diverging palette and we
also report mean |cosine| per method below the figure.

Usage:
    uv run python scripts/cosine_specialists.py \
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
import torch


METHOD_ORDER = ["caa", "melbo", "pi"]
METHOD_LABEL = {"caa": "CAA", "melbo": "MELBO", "pi": "PI(pad=5)"}
METHOD_LAYER_NOTE = {
    "caa":   "layer 24, residual stream output",
    "melbo": "layer 10, down_proj output",
    "pi":    "layer 10, down_proj output",
}

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
        "exp_dir": exp_dir,
        "train_eval": manifest["config"]["category"],
        "records": payload["results"],
    }


def find_specialist_idx_and_scale(records: list[dict], train_eval: str, method: str,
                                  aligned_sign: dict) -> tuple[int, float] | None:
    """Best (vector_idx, scale) of `method` evaluated on `train_eval` (aligned direction).

    Returns the (vector_idx, scale) that maximizes alignment shift on the
    home eval. Scale's sign is meaningful for picking the canonical
    "aligned" sign of the vector.
    """
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
    best = max(
        ((vi, s, aligned_pct(100 * sum(v) / len(v), aligned_sign[train_eval]) - base_a)
         for (vi, s), v in rates.items()),
        key=lambda t: t[2],
    )
    return best[0], best[1]


def find_vector_file(exp_dir: Path, method: str) -> Path:
    candidates = sorted((exp_dir / "vectors").glob(f"{method}_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No {method}_*.pt in {exp_dir}/vectors/")
    return candidates[-1]


def load_specialist_vector(exp_dir: Path, method: str, vec_idx: int) -> torch.Tensor:
    """Load the specialist vector and unit-normalize it."""
    path = find_vector_file(exp_dir, method)
    data = torch.load(path, map_location="cpu", weights_only=True)
    vecs = data["vectors"] if isinstance(data, dict) else data
    v = vecs[vec_idx].float()
    return v / v.norm()


def render_figure(per_method_cos: dict, per_method_idx_scale: dict,
                  evals: list[str], out_path: Path, sources: list[dict]) -> Path:
    n = len(evals)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)

    short = lambda s: s.replace("-", "\n", 1)

    for ax, method in zip(axes, METHOD_ORDER):
        cos = per_method_cos[method]
        im = ax.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r")
        for i in range(n):
            for j in range(n):
                v = cos[i, j]
                color = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color=color, fontsize=8)
        # Annotation outside the cell (vector idx + sign of scale)
        labels = []
        for ev in evals:
            info = per_method_idx_scale[method].get(ev)
            if info is None:
                labels.append(short(ev))
            else:
                vi, s = info
                sign = "+" if s >= 0 else "−"
                labels.append(f"{short(ev)}\nv{vi}{sign}")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)

        mean_abs = float(np.mean(np.abs(cos[~np.eye(n, dtype=bool)])))
        ax.set_title(
            f"{METHOD_LABEL[method]}  |  mean |cos|(off-diag) = {mean_abs:.2f}\n"
            f"vectors at {METHOD_LAYER_NOTE[method]}",
            fontsize=10,
        )

    fig.suptitle(
        "Per-method cosine similarity across the 7 train evals' specialist vectors\n"
        "(specialist = best vector trained on that eval, on the home eval; sign of scale shown next to vector idx)",
        fontsize=12,
    )
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.10, shrink=0.5)
    cbar.set_label("Signed cosine similarity")

    src_lines = ", ".join(Path(s["exp_dir"]).name for s in sources)
    fig.text(0.01, 0.005,
             f"Sources: {src_lines}. Cosines computed on unit-normalized vectors. "
             f"For PI, sign is arbitrary (eigenvectors); use |cos| for 'same axis?' interpretation. "
             f"For CAA / MELBO, sign of cosine reflects whether the discovered direction is "
             f"the same or opposite across training prompts.",
             fontsize=7.5, color="#444", wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_sidecar(out_path: Path, per_method_cos: dict, per_method_idx_scale: dict,
                  evals: list[str], sources: list[dict]) -> Path:
    sidecar = out_path.with_suffix(".json")
    payload = {
        "figure": out_path.name,
        "metric": "Pairwise signed cosine similarity (unit-normalized vectors). "
                  "Per method, the 7 vectors are each method's home-eval specialist for the train eval.",
        "evals": evals,
        "specialists_per_method": {
            m: {ev: {"vector_idx": vi, "scale": s} for ev, (vi, s) in per_method_idx_scale[m].items()}
            for m in METHOD_ORDER
        },
        "cosine_matrices": {
            m: per_method_cos[m].tolist() for m in METHOD_ORDER
        },
        "sources": sources,
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
    evals = [e["train_eval"] for e in experiments]

    per_method_idx_scale: dict[str, dict[str, tuple[int, float]]] = {m: {} for m in METHOD_ORDER}
    per_method_vecs: dict[str, dict[str, torch.Tensor]] = {m: {} for m in METHOD_ORDER}

    for exp in experiments:
        train = exp["train_eval"]
        for method in METHOD_ORDER:
            spec = find_specialist_idx_and_scale(exp["records"], train, method, aligned_sign)
            if spec is None:
                continue
            vi, scale = spec
            try:
                v = load_specialist_vector(exp["exp_dir"], method, vi)
            except (FileNotFoundError, KeyError, IndexError) as e:
                print(f"  skip {method} {train}: {e}")
                continue
            per_method_idx_scale[method][train] = (vi, scale)
            per_method_vecs[method][train] = v

    per_method_cos: dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        n = len(evals)
        cos = np.zeros((n, n))
        for i, ei in enumerate(evals):
            for j, ej in enumerate(evals):
                vi = per_method_vecs[method].get(ei)
                vj = per_method_vecs[method].get(ej)
                if vi is None or vj is None:
                    cos[i, j] = np.nan
                    continue
                cos[i, j] = float(torch.dot(vi, vj))
        per_method_cos[method] = cos

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    sources = [{"exp_dir": str(e["exp_dir"]), "train_eval": e["train_eval"]} for e in experiments]
    out_path = args.analysis_dir / "cosine_specialists.png"
    render_figure(per_method_cos, per_method_idx_scale, evals, out_path, sources)
    write_sidecar(out_path, per_method_cos, per_method_idx_scale, evals, sources)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
