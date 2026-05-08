#!/usr/bin/env python3
"""Three CPU-only checks for whether per-cell matrix entries are real signal
or degeneracy/noise artifacts.

Loads each per-train-eval experiment dir, extracts per-(vector, scale, eval)
match%, and classifies each cell of the matrix:

  1. Baseline-regression ceiling: the alignment shift achievable by the
     model collapsing to 50/50 random output. = `50 - baseline_aligned_pct`.
     If observed |shift| <= this ceiling, we cannot rule out pure noise
     from output degeneracy.

  2. Monotonicity: Spearman ρ between signed scale and alignment shift
     across the 11 scales. Real steering should produce a (mostly) monotonic
     response. |ρ| < 0.5 = suspicious.

  3. Logit collapse: ratio of |logit_diff| std at chosen-best scale vs at
     scale=0. If <0.5, the response distribution has narrowed dramatically
     (degeneracy signature). If >=0.7, the distribution is intact and shifted.

Prints per-cell verdicts and a summary classification.

Usage:
    uv run python scripts/check_signal_robustness.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


METHOD_ORDER = ["caa", "melbo", "pi"]
METHOD_LABEL = {"caa": "CAA", "melbo": "MELBO", "pi": "PI"}

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


def spearman(xs: list[float], ys: list[float]) -> float:
    """Simple Spearman implementation."""
    if len(xs) < 2:
        return 0.0
    x_rank = np.argsort(np.argsort(xs))
    y_rank = np.argsort(np.argsort(ys))
    n = len(xs)
    d2 = sum((x_rank[i] - y_rank[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


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


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    aligned_sign = load_aligned_sign(repo_root)

    exp_dirs = sorted([p for p in (repo_root / "experiments").iterdir()
                       if p.is_dir() and p.name.startswith("qwen3_14b_train_")])
    print(f"Loading {len(exp_dirs)} experiments...\n")
    experiments = [load_experiment(d) for d in exp_dirs]

    # Per-eval baseline aligned %
    base_match: dict[str, float] = {}
    all_records = [r for e in experiments for r in e["records"]]
    for r in all_records:
        if r["scale"] == 0.0 and r["dataset"] not in base_match:
            base_match[r["dataset"]] = None
    for ds in base_match:
        # Average baseline match% across all baselines (multiple from each experiment)
        vals = [r["chose_matching"] for r in all_records if r["scale"] == 0.0 and r["dataset"] == ds]
        base_match[ds] = 100 * sum(vals) / len(vals)
    baselines_aligned = {ds: aligned_pct(p, aligned_sign[ds]) for ds, p in base_match.items()}

    print("=== Baseline aligned% per eval (and regression-to-mean shift if model goes 50/50) ===\n")
    print(f"{'eval':<28} {'aligned baseline':>17} {'shift if random':>16}")
    for ds, ba in sorted(baselines_aligned.items()):
        ceiling = 50 - ba  # signed: positive = degeneracy aligns; negative = degeneracy misaligns
        print(f"{ds:<28} {ba:>16.1f}% {ceiling:>+15.1f}")
    print()

    print("=== Per-cell robustness ===\n")
    print(f"{'train':<22} {'method':<6} {'test':<22} "
          f"{'best shift':>10} {'scale':>5} {'ceil':>5} "
          f"{'spearman':>8} {'logit ratio':>11}  verdict")
    print("-" * 124)

    counts = defaultdict(int)
    suspicious_cells: list[dict] = []

    for exp in experiments:
        train = exp["train_eval"]
        # Group records: (method, vec, ds) -> list of (scale, [chose_matching], [logit_diff])
        per_vec_eval = defaultdict(lambda: defaultdict(lambda: {"chose": [], "ldiff": []}))
        for r in exp["records"]:
            key = (r["vector_type"], r["vector_idx"], r["dataset"])
            per_vec_eval[key][r["scale"]]["chose"].append(r["chose_matching"])
            per_vec_eval[key][r["scale"]]["ldiff"].append(r["matching_logit_diff"])

        # For each method, find best generalist (mean alignment shift across evals)
        for method in METHOD_ORDER:
            keys_for_method = [(m, vi, ds) for (m, vi, ds) in per_vec_eval if m == method]
            if not keys_for_method:
                continue
            vec_idxs = sorted({vi for (m, vi, ds) in keys_for_method})
            datasets = sorted({ds for (m, vi, ds) in keys_for_method})

            # Best generalist = vector with max mean (over evals) of max-over-scales aligned shift
            def vec_score(vi: int) -> float:
                shifts = []
                for ds in datasets:
                    if (method, vi, ds) not in per_vec_eval:
                        continue
                    base_a = baselines_aligned[ds]
                    cands = [
                        aligned_pct(100 * sum(per_vec_eval[(method, vi, ds)][s]["chose"]) /
                                    len(per_vec_eval[(method, vi, ds)][s]["chose"]),
                                    aligned_sign[ds]) - base_a
                        for s in per_vec_eval[(method, vi, ds)]
                    ]
                    shifts.append(max(cands))
                return float(np.mean(shifts)) if shifts else float("-inf")

            gen_vi = max(vec_idxs, key=vec_score)

            # Now check each test eval's cell for this generalist
            for ds in datasets:
                if (method, gen_vi, ds) not in per_vec_eval:
                    continue
                base_a = baselines_aligned[ds]
                ceiling = 50 - base_a  # signed
                scale_data = per_vec_eval[(method, gen_vi, ds)]
                scales = sorted(scale_data.keys())
                # Per-scale aligned shift
                shifts = []
                for s in scales:
                    pct = 100 * sum(scale_data[s]["chose"]) / len(scale_data[s]["chose"])
                    shifts.append(aligned_pct(pct, aligned_sign[ds]) - base_a)
                # Best (matches matrix definition)
                best_idx = int(np.argmax(shifts))
                best_scale = scales[best_idx]
                best_shift = shifts[best_idx]
                # Logit collapse: std at best scale vs std at scale=0
                ldiff_0 = scale_data[0.0]["ldiff"]
                ldiff_best = scale_data[best_scale]["ldiff"]
                std0 = stdev(ldiff_0) if len(ldiff_0) > 1 else 0.0
                stdb = stdev(ldiff_best) if len(ldiff_best) > 1 else 0.0
                logit_ratio = stdb / std0 if std0 > 0 else 0.0
                # Spearman ρ between signed scale and shift
                rho = spearman(scales, shifts)

                # Verdict
                # 1) Below noise: same sign as ceiling AND magnitude <= 1.5 * |ceiling|
                same_sign_as_ceiling = (best_shift > 0) == (ceiling > 0)
                noise_dominant = same_sign_as_ceiling and abs(best_shift) <= 1.5 * abs(ceiling)
                # 2) Non-monotonic: |rho| < 0.5
                weak_monotone = abs(rho) < 0.5
                # 3) Logit collapse: ratio < 0.5
                collapsed = logit_ratio < 0.5

                tags = []
                if abs(best_shift) < 5:
                    tags.append("tiny")
                if noise_dominant and abs(best_shift) >= 5:
                    tags.append("ceiling")
                if weak_monotone:
                    tags.append("non-mono")
                if collapsed:
                    tags.append("collapsed")

                if not tags:
                    verdict = "ROBUST"
                else:
                    verdict = "SUSPECT (" + ",".join(tags) + ")"
                counts[verdict.split(" ")[0]] += 1

                if verdict.startswith("SUSPECT"):
                    suspicious_cells.append({
                        "train": train, "method": method, "test": ds,
                        "vec_idx": gen_vi, "best_shift": best_shift,
                        "scale": best_scale, "ceiling": ceiling,
                        "spearman": rho, "logit_ratio": logit_ratio,
                        "verdict": verdict,
                    })

                print(f"{train:<22} {METHOD_LABEL[method]:<6} {ds:<22} "
                      f"{best_shift:>+9.1f} {best_scale:>+5.0f} {ceiling:>+5.1f} "
                      f"{rho:>+8.2f} {logit_ratio:>10.2f}  {verdict}")

    print()
    print("=== Summary ===")
    total = sum(counts.values())
    for verdict, n in sorted(counts.items()):
        pct = 100 * n / total
        print(f"  {verdict:<10} {n:>3} / {total} ({pct:.0f}%)")
    print()
    print(f"Of {total} (train, test, method) cells:")
    print(f"  ROBUST  = monotonic, above noise floor, no logit collapse")
    print(f"  SUSPECT = at least one of: tiny (<5pp), ceiling (within noise),")
    print(f"            non-monotonic (|spearman|<0.5), or collapsed (logit std halved)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
