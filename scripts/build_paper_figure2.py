"""Figure 2 — Logit-difference vs sampled-generation aligned-% scatter.

For each (layer-pair, method, test-eval) cell where we have generation data,
plot logit-diff aligned-% (from the eval JSONs) against generation aligned-%
(from the gen JSONs). The diagonal y = x reference line shows where the
two metrics agree.

Two layer pairs:
- (10, 32) corrig-best: 4 methods × 7 datasets (results/gen_cross_eval_10_32_corrigselect.json)
- (18, 25) corrig-best: 3 methods × 7 datasets (results/gen_cross_eval_18_25.json)

Output: paper_artifacts/figure2_logit_vs_gen.{png,pdf}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent

ALIGNED_SIGN = {
    "corrigible-neutral-HHH":     +1,
    "survival-instinct":          +1,
    "power-seeking-inclination":  +1,
    "wealth-seeking-inclination": +1,
    "self-awareness-general-ai":  +1,
    "coordinate-other-ais":       -1,
    "myopic-reward":              -1,
}

METHOD_COLOR = {
    "caa":   "#888888",
    "pi":    "#1f77b4",
    "melbo": "#2ca02c",
    "dct":   "#d62728",
}
METHOD_LABEL = {"caa": "CAA", "pi": "PI", "melbo": "MELBO", "dct": "DCT"}


def load_eval_records(exp_dir: Path) -> list[dict]:
    eval_files = sorted((exp_dir / "eval").glob("*.json"))
    if not eval_files:
        return []
    with open(eval_files[-1]) as f:
        payload = json.load(f)
    return payload.get("results", payload) if isinstance(payload, dict) else payload


def cell_logit_aligned_pct(records, method, vector_idx, scale, dataset):
    """Compute aligned-% for a specific (method, vector_idx, scale, dataset)
    cell from eval records. Uses chose_matching with polarity flip."""
    sign = ALIGNED_SIGN[dataset]
    matches = [r["chose_matching"] for r in records
               if r["dataset"] == dataset
               and r["vector_type"] == method
               and r["vector_idx"] == vector_idx
               and r["scale"] == scale]
    if not matches:
        return None
    n_match = sum(1 for m in matches if m)
    pct = 100 * n_match / len(matches)
    return pct if sign > 0 else (100 - pct)


def collect_points(gen_path: Path, eval_dirs: dict[str, Path]) -> list[dict]:
    """For each non-baseline cell in gen_path, look up the matching logit
    aligned-% from the corresponding eval JSON. eval_dirs is a per-method
    map (so DCT can come from a different experiment than PI/MELBO/CAA)."""
    with open(gen_path) as f:
        gen = json.load(f)

    # Cache eval records per dir
    records_cache: dict[Path, list[dict]] = {}
    def get_records(d):
        if d not in records_cache:
            records_cache[d] = load_eval_records(d)
        return records_cache[d]

    points = []
    for c in gen["cells"]:
        if c["method"] == "baseline":
            continue
        method = c["method"]
        if method not in eval_dirs:
            continue
        records = get_records(eval_dirs[method])
        logit_pct = cell_logit_aligned_pct(
            records, method, c["vector_idx"], c["scale"], c["dataset"],
        )
        if logit_pct is None:
            continue
        points.append({
            "method": method,
            "dataset": c["dataset"],
            "vector_idx": c["vector_idx"],
            "scale": c["scale"],
            "logit": logit_pct,
            "gen": c["summary"]["aligned_pct"],
        })
    return points


def render(points_by_layer: dict[str, list[dict]], out_path: Path):
    panels = [(label, pts) for label, pts in points_by_layer.items() if len(pts) >= 8]
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.5),
                              sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (label, points) in zip(axes, panels):
        # y = x reference
        ax.plot([0, 100], [0, 100], color="black", linestyle="--", linewidth=0.8,
                alpha=0.4, zorder=1)
        # Per-method scatter
        seen_methods = set()
        for p in points:
            ax.scatter(
                p["logit"], p["gen"],
                color=METHOD_COLOR[p["method"]],
                edgecolor="black", linewidth=0.4,
                s=55, alpha=0.85, zorder=3,
                label=METHOD_LABEL[p["method"]] if p["method"] not in seen_methods else None,
            )
            seen_methods.add(p["method"])

        # Per-method best-fit line + correlation
        for method in ("caa", "pi", "melbo", "dct"):
            mp = [(p["logit"], p["gen"]) for p in points if p["method"] == method]
            if len(mp) < 3:
                continue
            xs, ys = zip(*mp)
            slope, intercept = np.polyfit(xs, ys, 1)
            x_range = np.linspace(min(xs), max(xs), 50)
            ax.plot(x_range, slope * x_range + intercept,
                    color=METHOD_COLOR[method], alpha=0.45, linewidth=1.0,
                    zorder=2)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Logit-difference aligned-%")
        ax.set_ylabel("Generation aligned-%")
        ax.set_title(f"Layer pair {label}")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    fig.suptitle("Logit-difference vs sampled-generation aligned-%\n"
                 "(corrigibility-trained vectors, 7 cross-eval test sets)",
                 y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.with_suffix('.png')}")
    print(f"Saved {out_path.with_suffix('.pdf')}")


def main():
    # (10, 32): PI/MELBO/CAA vectors live in qwen3_14b_train_corrigible-neutral-HHH;
    #          DCT vectors in qwen3_14b_dct_corrigible-neutral-HHH
    eval_dirs_10_32 = {
        "pi":    REPO / "experiments" / "qwen3_14b_train_corrigible-neutral-HHH",
        "melbo": REPO / "experiments" / "qwen3_14b_train_corrigible-neutral-HHH",
        "caa":   REPO / "experiments" / "qwen3_14b_train_corrigible-neutral-HHH",
        "dct":   REPO / "experiments" / "qwen3_14b_dct_corrigible-neutral-HHH",
    }
    pts_10_32 = collect_points(
        REPO / "results" / "gen_cross_eval_10_32_corrigselect.json",
        eval_dirs_10_32,
    )

    # (18, 25): all three live in the drill experiment dir; no CAA there.
    eval_dirs_18_25 = {
        "pi":    REPO / "experiments" / "drill_dct_corrigibility_18_25_Qwen3-14B",
        "melbo": REPO / "experiments" / "drill_dct_corrigibility_18_25_Qwen3-14B",
        "dct":   REPO / "experiments" / "drill_dct_corrigibility_18_25_Qwen3-14B",
    }
    pts_18_25 = collect_points(
        REPO / "results" / "gen_cross_eval_18_25.json",
        eval_dirs_18_25,
    )

    # Print correlation per (panel, method) for the writeup
    print("Correlation between logit-aligned% and gen-aligned%:")
    for label, points in [("(10, 32)", pts_10_32), ("(18, 25)", pts_18_25)]:
        print(f"\n  Layer pair {label}  (n={len(points)} cells):")
        for method in ("caa", "pi", "melbo", "dct"):
            mp = [(p["logit"], p["gen"]) for p in points if p["method"] == method]
            if len(mp) < 3:
                continue
            xs, ys = zip(*mp)
            r = np.corrcoef(xs, ys)[0, 1]
            slope, intercept = np.polyfit(xs, ys, 1)
            print(f"    {METHOD_LABEL[method]:>5}: r = {r:+.2f}  slope = {slope:.2f}")

    out = REPO / "paper_artifacts" / "figure2_logit_vs_gen"
    render({"(10, 32)": pts_10_32, "(18, 25)": pts_18_25}, out)


if __name__ == "__main__":
    main()
