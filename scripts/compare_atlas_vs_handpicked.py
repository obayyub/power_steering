"""Compare PI/MELBO trained at atlas-best (s, t) vs at hand-picked (10, 32).

Reads paired eval JSONs:
  - Baseline (hand-picked layer pair):
        experiments/qwen3_14b_train_<eval>/eval/eval_*.json
        (from the parallel session; multi-dataset eval — we filter to home eval)
  - Treatment (atlas-best layer pair):
        experiments/drill_<eval-tag>_<s>_<t>_Qwen3-14B/eval/eval_*.json
        (from this session; single-dataset, dataset_filter=home eval)

Per eval, per method (pi, melbo), reports best ALIGNED-direction shift (in
both logit-diff and match-percentage) over (vector_idx, scale). Aligned
direction is sign-corrected via ALIGNED_SIGN — see inspect_map.py for the
table source. The headline metric is Δ = atlas_best_aligned − hand_picked
_aligned: positive = atlas-best layer pair beats hand-picked.

Usage:
    uv run python scripts/compare_atlas_vs_handpicked.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ALIGNED_SIGN: dict[str, int] = {
    "corrigible-neutral-HHH":      +1,
    "survival-instinct":           +1,
    "power-seeking-inclination":   +1,
    "wealth-seeking-inclination":  +1,
    "self-awareness-general-ai":   +1,
    "coordinate-other-ais":        -1,
    "myopic-reward":               -1,
}


# eval name → (baseline_dir, [list of (label, atlas_dir)])
COMPARISONS = {
    "corrigible-neutral-HHH": (
        "qwen3_14b_train_corrigible-neutral-HHH",
        [("atlas (18,25)", "drill_corrigibility_18_25_Qwen3-14B")],
    ),
    "self-awareness-general-ai": (
        "qwen3_14b_train_self-awareness-general-ai",
        [("atlas (12,31)", "drill_self-awareness_12_31_Qwen3-14B")],
    ),
    "survival-instinct": (
        "qwen3_14b_train_survival-instinct",
        [
            ("atlas-aligned (18,27)",    "drill_survival_aligned_18_27_Qwen3-14B"),
            ("atlas-misaligned (23,27)", "drill_survival_misaligned_23_27_Qwen3-14B"),
        ],
    ),
    "power-seeking-inclination": (
        "qwen3_14b_train_power-seeking-inclination",
        [("atlas (20,30)", "drill_power-seeking_20_30_Qwen3-14B")],
    ),
    "wealth-seeking-inclination": (
        "qwen3_14b_train_wealth-seeking-inclination",
        [("atlas (21,27)", "drill_wealth-seeking_21_27_Qwen3-14B")],
    ),
    "coordinate-other-ais": (
        "qwen3_14b_train_coordinate-other-ais",
        [("atlas (21,22)", "drill_coordinate-other-ais_21_22_Qwen3-14B")],
    ),
    "myopic-reward": (
        "qwen3_14b_train_myopic-reward",
        [("atlas (20,23)", "drill_myopic-reward_20_23_Qwen3-14B")],
    ),
}


EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "experiments"


def load_eval(exp_dir_name: str) -> list[dict]:
    eval_dir = EXPERIMENTS_ROOT / exp_dir_name / "eval"
    if not eval_dir.exists():
        raise FileNotFoundError(f"No eval/ in {eval_dir}")
    files = sorted(eval_dir.glob("eval_*.json"))
    if not files:
        raise FileNotFoundError(f"No eval_*.json in {eval_dir}")
    with open(files[-1]) as f:
        return json.load(f)["results"]


def best_aligned_per_method(
    records: list[dict],
    dataset: str,
    aligned_sign: int,
    method: str,
) -> dict:
    """Per (vector_idx, scale) combo of `method` on `dataset`, compute mean
    aligned logit-diff and aligned match%. Return the best (vector_idx, scale)
    by aligned_ld_mean.
    """
    by_combo = defaultdict(lambda: {"ld_sum": 0.0, "match_n": 0, "n": 0})
    for r in records:
        if r["dataset"] != dataset or r["vector_type"] != method:
            continue
        key = (r["vector_idx"], r["scale"])
        s = by_combo[key]
        s["n"] += 1
        s["ld_sum"] += aligned_sign * r["matching_logit_diff"]
        # aligned match: chose_matching XOR (aligned_sign==-1)
        chose_aligned = r["chose_matching"] if aligned_sign == 1 else not r["chose_matching"]
        s["match_n"] += int(chose_aligned)

    best_key = None
    best_ld = float("-inf")
    best_match_pct = None
    baseline_match_pct = None
    for (vi, scale), s in by_combo.items():
        if s["n"] == 0:
            continue
        ld = s["ld_sum"] / s["n"]
        match_pct = 100 * s["match_n"] / s["n"]
        if scale == 0.0:
            baseline_match_pct = match_pct  # same for any vector at scale 0
        if ld > best_ld:
            best_ld = ld
            best_key = (vi, scale)
            best_match_pct = match_pct

    return {
        "best_vec": best_key[0] if best_key else None,
        "best_scale": best_key[1] if best_key else None,
        "best_aligned_ld": best_ld if best_key else None,
        "best_aligned_match_pct": best_match_pct,
        "baseline_match_pct": baseline_match_pct,
    }


def main() -> None:
    print(f"\n{'='*92}")
    print(f"  Atlas-best vs hand-picked (10, 32) layer pairs — best aligned-direction shift")
    print(f"{'='*92}")

    summary_rows = []
    for eval_name, (baseline_dir, atlas_runs) in COMPARISONS.items():
        sign = ALIGNED_SIGN[eval_name]
        baseline_records = load_eval(baseline_dir)

        print(f"\n--- {eval_name}  (aligned_sign={sign:+d})")

        # Baseline at hand-picked (10, 32).
        baseline_pi = best_aligned_per_method(baseline_records, eval_name, sign, "pi")
        baseline_melbo = best_aligned_per_method(baseline_records, eval_name, sign, "melbo")
        bl_pct = baseline_pi["baseline_match_pct"]  # same for both methods
        print(f"  baseline aligned match% (scale=0): {bl_pct:.1f}%")
        print(f"  hand-picked (10,32):  PI    best vec_{baseline_pi['best_vec']}@{baseline_pi['best_scale']:+.0f}"
              f"  aligned_ld={baseline_pi['best_aligned_ld']:+.2f}"
              f"  aligned_match%={baseline_pi['best_aligned_match_pct']:.1f}"
              f"  Δ_pp={baseline_pi['best_aligned_match_pct']-bl_pct:+.1f}")
        print(f"  hand-picked (10,32):  MELBO best vec_{baseline_melbo['best_vec']}@{baseline_melbo['best_scale']:+.0f}"
              f"  aligned_ld={baseline_melbo['best_aligned_ld']:+.2f}"
              f"  aligned_match%={baseline_melbo['best_aligned_match_pct']:.1f}"
              f"  Δ_pp={baseline_melbo['best_aligned_match_pct']-bl_pct:+.1f}")

        # One or more atlas-best treatments.
        for label, atlas_dir in atlas_runs:
            atlas_records = load_eval(atlas_dir)
            atlas_pi = best_aligned_per_method(atlas_records, eval_name, sign, "pi")
            atlas_melbo = best_aligned_per_method(atlas_records, eval_name, sign, "melbo")

            print(f"  {label}:  PI    best vec_{atlas_pi['best_vec']}@{atlas_pi['best_scale']:+.0f}"
                  f"  aligned_ld={atlas_pi['best_aligned_ld']:+.2f}"
                  f"  aligned_match%={atlas_pi['best_aligned_match_pct']:.1f}"
                  f"  Δ_pp={atlas_pi['best_aligned_match_pct']-bl_pct:+.1f}"
                  f"  (vs hp: Δ_ld={atlas_pi['best_aligned_ld']-baseline_pi['best_aligned_ld']:+.2f},"
                  f" Δ_match%={atlas_pi['best_aligned_match_pct']-baseline_pi['best_aligned_match_pct']:+.1f}pp)")
            print(f"  {label}:  MELBO best vec_{atlas_melbo['best_vec']}@{atlas_melbo['best_scale']:+.0f}"
                  f"  aligned_ld={atlas_melbo['best_aligned_ld']:+.2f}"
                  f"  aligned_match%={atlas_melbo['best_aligned_match_pct']:.1f}"
                  f"  Δ_pp={atlas_melbo['best_aligned_match_pct']-bl_pct:+.1f}"
                  f"  (vs hp: Δ_ld={atlas_melbo['best_aligned_ld']-baseline_melbo['best_aligned_ld']:+.2f},"
                  f" Δ_match%={atlas_melbo['best_aligned_match_pct']-baseline_melbo['best_aligned_match_pct']:+.1f}pp)")

            summary_rows.append({
                "eval": eval_name,
                "label": label,
                "pi_hp": baseline_pi["best_aligned_match_pct"],
                "pi_atlas": atlas_pi["best_aligned_match_pct"],
                "pi_delta": atlas_pi["best_aligned_match_pct"] - baseline_pi["best_aligned_match_pct"],
                "melbo_hp": baseline_melbo["best_aligned_match_pct"],
                "melbo_atlas": atlas_melbo["best_aligned_match_pct"],
                "melbo_delta": atlas_melbo["best_aligned_match_pct"] - baseline_melbo["best_aligned_match_pct"],
            })

    print(f"\n\n{'='*92}")
    print(f"  Summary — best aligned match% per eval (hand-picked → atlas-best)")
    print(f"{'='*92}")
    print(f"  {'eval / atlas pair':<48} {'PI hp':>6} {'PI atlas':>9} {'Δ':>6}   {'MELBO hp':>9} {'MELBO atlas':>12} {'Δ':>6}")
    print(f"  {'-'*48} {'-'*6} {'-'*9} {'-'*6}   {'-'*9} {'-'*12} {'-'*6}")
    pi_deltas, melbo_deltas = [], []
    for r in summary_rows:
        print(f"  {r['eval'][:30]:<32} {r['label'][:14]:<14} "
              f"{r['pi_hp']:>5.1f}% {r['pi_atlas']:>8.1f}% {r['pi_delta']:>+5.1f}    "
              f"{r['melbo_hp']:>8.1f}% {r['melbo_atlas']:>11.1f}% {r['melbo_delta']:>+5.1f}")
        # Only count first row per eval (atlas-aligned for survival) for the mean.
        if "misaligned" not in r["label"]:
            pi_deltas.append(r["pi_delta"])
            melbo_deltas.append(r["melbo_delta"])
    print(f"\n  Mean Δ across {len(pi_deltas)} evals (atlas-aligned variant for survival):")
    print(f"    PI:    {sum(pi_deltas)/len(pi_deltas):+.1f} pp")
    print(f"    MELBO: {sum(melbo_deltas)/len(melbo_deltas):+.1f} pp")


if __name__ == "__main__":
    main()
