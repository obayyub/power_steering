"""Compare MELBO from PI-init vs random-init at the same atlas-best (s, t).

Three-way comparison per eval:
  1. MELBO @ hand-picked (10, 32)  — random init  (parallel session, qwen3_14b_train_<eval>)
  2. MELBO @ atlas-best (s, t)     — random init  (drill_<eval>)
  3. MELBO @ atlas-best (s, t)     — PI init      (drill_pi_init_<eval>)

For each, picks the best (vector_idx, scale) by mean aligned-direction
matching_logit_diff and reports the corresponding aligned match%. Sign-
corrected via ALIGNED_SIGN.

Usage:
    uv run python scripts/compare_pi_init_warmstart.py
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


# eval name → (baseline_dir, [(label, randinit_dir, piinit_dir), ...])
COMPARISONS = {
    "corrigible-neutral-HHH": (
        "qwen3_14b_train_corrigible-neutral-HHH",
        [("(18, 25)", "drill_corrigibility_18_25_Qwen3-14B", "drill_pi_init_corrigibility_18_25_Qwen3-14B")],
    ),
    "self-awareness-general-ai": (
        "qwen3_14b_train_self-awareness-general-ai",
        [("(12, 31)", "drill_self-awareness_12_31_Qwen3-14B", "drill_pi_init_self-awareness_12_31_Qwen3-14B")],
    ),
    "survival-instinct": (
        "qwen3_14b_train_survival-instinct",
        [
            ("aligned-best (18, 27)", "drill_survival_aligned_18_27_Qwen3-14B", "drill_pi_init_survival_aligned_18_27_Qwen3-14B"),
            ("misaligned-best (23, 27)", "drill_survival_misaligned_23_27_Qwen3-14B", "drill_pi_init_survival_misaligned_23_27_Qwen3-14B"),
        ],
    ),
    "power-seeking-inclination": (
        "qwen3_14b_train_power-seeking-inclination",
        [("(20, 30)", "drill_power-seeking_20_30_Qwen3-14B", "drill_pi_init_power-seeking_20_30_Qwen3-14B")],
    ),
    "wealth-seeking-inclination": (
        "qwen3_14b_train_wealth-seeking-inclination",
        [("(21, 27)", "drill_wealth-seeking_21_27_Qwen3-14B", "drill_pi_init_wealth-seeking_21_27_Qwen3-14B")],
    ),
    "coordinate-other-ais": (
        "qwen3_14b_train_coordinate-other-ais",
        [("(21, 22)", "drill_coordinate-other-ais_21_22_Qwen3-14B", "drill_pi_init_coordinate-other-ais_21_22_Qwen3-14B")],
    ),
    "myopic-reward": (
        "qwen3_14b_train_myopic-reward",
        [("(20, 23)", "drill_myopic-reward_20_23_Qwen3-14B", "drill_pi_init_myopic-reward_20_23_Qwen3-14B")],
    ),
}


EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "experiments"


def load_eval(exp_dir_name: str) -> list[dict]:
    eval_dir = EXPERIMENTS_ROOT / exp_dir_name / "eval"
    files = sorted(eval_dir.glob("eval_*.json"))
    with open(files[-1]) as f:
        return json.load(f)["results"]


def best_for_method(records, dataset, aligned_sign, method):
    """Return both best aligned-direction AND best misaligned-direction (vec, scale).
    `aligned_ld` field uses sign convention: + = toward aligned, − = toward misaligned.
    Best aligned = max aligned_ld; best misaligned = min aligned_ld."""
    by_combo = defaultdict(lambda: {"ld_sum": 0.0, "match_n": 0, "n": 0})
    baseline_match_pct = None
    for r in records:
        if r["dataset"] != dataset or r["vector_type"] != method:
            continue
        key = (r["vector_idx"], r["scale"])
        s = by_combo[key]
        s["n"] += 1
        s["ld_sum"] += aligned_sign * r["matching_logit_diff"]
        chose_aligned = r["chose_matching"] if aligned_sign == 1 else not r["chose_matching"]
        s["match_n"] += int(chose_aligned)

    best_aligned = None  # max aligned_ld
    best_mis = None      # min aligned_ld (i.e. largest push toward misaligned)
    for key, s in by_combo.items():
        if s["n"] == 0:
            continue
        ld = s["ld_sum"] / s["n"]
        match_pct = 100 * s["match_n"] / s["n"]
        if key[1] == 0.0:
            baseline_match_pct = match_pct
        rec = {"vec": key[0], "scale": key[1], "aligned_ld": ld, "aligned_match_pct": match_pct}
        if best_aligned is None or ld > best_aligned["aligned_ld"]:
            best_aligned = dict(rec)
        if best_mis is None or ld < best_mis["aligned_ld"]:
            best_mis = dict(rec)
    return {"aligned": best_aligned, "misaligned": best_mis, "baseline_match_pct": baseline_match_pct}


def main() -> None:
    print(f"\n{'='*110}")
    print(f"  MELBO: random-init vs PI-init at same layer pair — both directions reported")
    print(f"  Aligned = best aligned-direction match%; Mis = best misaligned-direction match%")
    print(f"{'='*110}")

    rows = []
    for eval_name, (baseline_dir, atlas_runs) in COMPARISONS.items():
        sign = ALIGNED_SIGN[eval_name]
        baseline_records = load_eval(baseline_dir)
        bl = best_for_method(baseline_records, eval_name, sign, "melbo")

        for label, randinit_dir, piinit_dir in atlas_runs:
            rand = best_for_method(load_eval(randinit_dir), eval_name, sign, "melbo")
            pi = best_for_method(load_eval(piinit_dir), eval_name, sign, "melbo")

            print(f"\n--- {eval_name}  {label}  (aligned_sign={sign:+d}, baseline {bl['baseline_match_pct']:.0f}%)")
            for label2, d in [("hand-picked random", bl), ("atlas random   ", rand), ("atlas PI-init   ", pi)]:
                a, m = d["aligned"], d["misaligned"]
                print(f"  {label2}:  ALIGNED  vec_{a['vec']}@{a['scale']:+.0f} ld={a['aligned_ld']:+6.2f} match%={a['aligned_match_pct']:5.1f}%   "
                      f"|| MISaligned  vec_{m['vec']}@{m['scale']:+.0f} ld={m['aligned_ld']:+6.2f} match%={m['aligned_match_pct']:5.1f}%")

            rows.append({
                "eval": eval_name, "label": label,
                "hp_aligned":  bl["aligned"]["aligned_match_pct"],
                "rand_aligned":  rand["aligned"]["aligned_match_pct"],
                "pi_aligned":  pi["aligned"]["aligned_match_pct"],
                "hp_mis":  bl["misaligned"]["aligned_match_pct"],
                "rand_mis":  rand["misaligned"]["aligned_match_pct"],
                "pi_mis":  pi["misaligned"]["aligned_match_pct"],
            })

    print(f"\n\n{'='*110}")
    print(f"  Summary — best aligned-direction match% (high = good)")
    print(f"{'='*110}")
    print(f"  {'eval / pair':<54} {'hp':>6} {'rand':>6} {'pi-init':>8}  {'Δ pi-rand':>10}  {'Δ pi-hp':>10}")
    pi_v_rand_a, pi_v_hp_a, rand_v_hp_a = [], [], []
    for r in rows:
        if "misaligned" in r["label"]:
            continue
        d1, d2, d3 = r["pi_aligned"]-r["rand_aligned"], r["pi_aligned"]-r["hp_aligned"], r["rand_aligned"]-r["hp_aligned"]
        print(f"  {r['eval'][:32]:<32} {r['label'][:20]:<20} "
              f"{r['hp_aligned']:5.1f}% {r['rand_aligned']:5.1f}% {r['pi_aligned']:7.1f}%  "
              f"{d1:>+9.1f}pp  {d2:>+9.1f}pp")
        rand_v_hp_a.append(d3); pi_v_rand_a.append(d1); pi_v_hp_a.append(d2)
    n = len(pi_v_rand_a)
    print(f"\n  Mean Δ (aligned-direction, n={n} evals using survival aligned-best):")
    print(f"    atlas-rand   vs hand-picked:   {sum(rand_v_hp_a)/n:+5.1f} pp")
    print(f"    atlas-PIinit vs atlas-rand:    {sum(pi_v_rand_a)/n:+5.1f} pp")
    print(f"    atlas-PIinit vs hand-picked:   {sum(pi_v_hp_a)/n:+5.1f} pp")

    print(f"\n\n{'='*110}")
    print(f"  Summary — best MISaligned-direction match% (LOW = strong misaligned push;")
    print(f"           Δ pi-rand <0 means PI-init MELBO pushes MORE toward misaligned)")
    print(f"{'='*110}")
    print(f"  {'eval / pair':<54} {'hp':>6} {'rand':>6} {'pi-init':>8}  {'Δ pi-rand':>10}  {'Δ pi-hp':>10}")
    pi_v_rand_m, pi_v_hp_m, rand_v_hp_m = [], [], []
    for r in rows:
        if "misaligned" in r["label"]:
            continue
        d1, d2, d3 = r["pi_mis"]-r["rand_mis"], r["pi_mis"]-r["hp_mis"], r["rand_mis"]-r["hp_mis"]
        print(f"  {r['eval'][:32]:<32} {r['label'][:20]:<20} "
              f"{r['hp_mis']:5.1f}% {r['rand_mis']:5.1f}% {r['pi_mis']:7.1f}%  "
              f"{d1:>+9.1f}pp  {d2:>+9.1f}pp")
        rand_v_hp_m.append(d3); pi_v_rand_m.append(d1); pi_v_hp_m.append(d2)
    print(f"\n  Mean Δ (misaligned-direction, n={n} evals):")
    print(f"    atlas-rand   vs hand-picked:   {sum(rand_v_hp_m)/n:+5.1f} pp")
    print(f"    atlas-PIinit vs atlas-rand:    {sum(pi_v_rand_m)/n:+5.1f} pp")
    print(f"    atlas-PIinit vs hand-picked:   {sum(pi_v_hp_m)/n:+5.1f} pp")


if __name__ == "__main__":
    main()
