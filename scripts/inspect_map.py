"""Inspect a layer-pair map experiment mid-run.

Reads per-pair JSONs from `experiments/<id>/map/<category>/pairs/` and prints
summary stats + top-N pairs. Safe to run while the sweep is in progress
(per-pair JSONs are written atomically by `map_layers.py`).

Three ranking modes via `--direction`:
  - abs (default): rank by largest |logit_diff| regardless of direction.
    Useful for "where is the most steerable layer pair" question.
  - aligned: sign-correct each eval through ALIGNED_SIGN below, then rank by
    the most positive aligned_ld. Surfaces the strongest *aligned-direction*
    steering per pair — the safety-relevant ranking.
  - misaligned: same but most negative aligned_ld. Strongest "induce the
    bad behavior" steering — useful for capability-elicitation analyses.

Usage:
    uv run python scripts/inspect_map.py experiments/<id>
    uv run python scripts/inspect_map.py experiments/<id> --direction aligned
    uv run python scripts/inspect_map.py experiments/<id> --category corrigible-neutral-HHH --top 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Per-eval polarity: +1 = matching answer is HHH-aligned (so positive ld is
# steering toward aligned); -1 = matching answer is the named misaligned
# behavior (so negative ld is steering toward aligned). Mirrors
# scripts/analyze_best_alignment.py:ALIGNED_SIGN; see Phase D notes in
# docs/sessions/2026-05-02.md ("matching-letter convention isn't uniform").
ALIGNED_SIGN: dict[str, int] = {
    "corrigible-neutral-HHH":      +1,
    "survival-instinct":           +1,
    "power-seeking-inclination":   +1,
    "wealth-seeking-inclination":  +1,
    "self-awareness-general-ai":   +1,
    "coordinate-other-ais":        -1,
    "myopic-reward":               -1,
}


def collect(pairs_dir: Path, aligned_sign: int | None = None) -> list[dict]:
    """Read every per-pair JSON. If `aligned_sign` is given, also compute the
    best aligned-direction and best misaligned-direction (vec, sign) per pair."""
    rows: list[dict] = []
    for pf in sorted(pairs_dir.glob("*.json")):
        try:
            with open(pf) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        kl_pos, kl_neg = d["kl_pos"], d["kl_neg"]
        kl_max = max(max(kl_pos), max(kl_neg))

        # All (vec_idx, sign_label, raw_ld) candidates for this pair.
        candidates = (
            [(vi, "+", v) for vi, v in enumerate(d["ld_pos_per_vec"])]
            + [(vi, "-", v) for vi, v in enumerate(d["ld_neg_per_vec"])]
        )

        # Largest absolute-value steering, signed (current default ranking).
        abs_vi, abs_sign, abs_ld = max(candidates, key=lambda x: abs(x[2]))

        n_active = sum(
            1 for kp, kn in zip(kl_pos, kl_neg) if max(kp, kn) >= 0.5
        )

        row = {
            "s": d["source_layer"], "t": d["target_layer"],
            "scale": d["scale"], "sigma_top": d["sigmas"][0],
            "kl_max": kl_max,
            "abs_ld": abs_ld, "abs_vi": abs_vi, "abs_sign": abs_sign,
            "n_active": n_active,
        }

        if aligned_sign is not None:
            # aligned_ld = aligned_sign × raw_ld; positive means pushed toward
            # the aligned answer regardless of which letter "matching" is.
            aligned_cands = [(vi, sg, aligned_sign * raw) for vi, sg, raw in candidates]
            ali_vi, ali_sign, ali_ld = max(aligned_cands, key=lambda x: x[2])
            mis_vi, mis_sign, mis_ld = min(aligned_cands, key=lambda x: x[2])
            row["aligned_ld"] = ali_ld
            row["aligned_vi"] = ali_vi
            row["aligned_sign"] = ali_sign
            row["misaligned_ld"] = -mis_ld  # report as positive magnitude
            row["misaligned_vi"] = mis_vi
            row["misaligned_sign"] = mis_sign

        rows.append(row)
    return rows


def print_stats(label: str, xs: list[float], fmt: str = "{:.3f}") -> None:
    if not xs:
        return
    xs = sorted(xs)
    n = len(xs)
    print(
        f"  {label:<14} "
        f"min={fmt.format(xs[0])} "
        f"p25={fmt.format(xs[n // 4])} "
        f"med={fmt.format(xs[n // 2])} "
        f"p75={fmt.format(xs[3 * n // 4])} "
        f"max={fmt.format(xs[-1])}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--category", default=None,
                    help="Only inspect this category (default: all under map/)")
    ap.add_argument("--top", type=int, default=15,
                    help="Number of top pairs to list")
    ap.add_argument("--direction", choices=["abs", "aligned", "misaligned"],
                    default="abs",
                    help="Ranking metric: abs=largest |ld| (any direction); "
                         "aligned=largest signed ld toward aligned answer; "
                         "misaligned=largest signed ld toward misaligned answer.")
    args = ap.parse_args()

    exp_root = Path(args.experiment_dir)

    manifest_path = exp_root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            m = json.load(f)
        cfg = m.get("config", {})
        print(f"Experiment: {m.get('experiment_id')}")
        print(f"Model: {cfg.get('model')}")
        if m.get("duration_seconds"):
            print(f"Status: finalized ({m['duration_seconds']}s)")
        else:
            print("Status: in progress (no duration stamped yet)")
    else:
        print(f"No manifest at {manifest_path} — proceeding from raw pair files")

    map_root = exp_root / "map"
    if not map_root.exists():
        print(f"No map dir at {map_root}")
        return

    cats = (
        [args.category] if args.category
        else [d.name for d in sorted(map_root.iterdir()) if d.is_dir()]
    )

    for cat in cats:
        pairs_dir = map_root / cat / "pairs"
        if not pairs_dir.exists():
            print(f"\n[{cat}] no pairs/ yet")
            continue

        aligned_sign = ALIGNED_SIGN.get(cat)
        if args.direction != "abs" and aligned_sign is None:
            print(f"\n[{cat}] no polarity entry in ALIGNED_SIGN; skipping")
            continue

        rows = collect(pairs_dir, aligned_sign=aligned_sign)
        print(f"\n=== {cat} ===")
        polarity_note = (
            f"  (aligned_sign={aligned_sign:+d}; "
            f"{'matching = aligned' if aligned_sign == 1 else 'matching = MISaligned'})"
            if aligned_sign is not None else ""
        )
        print(f"Pairs completed: {len(rows)}{polarity_note}")
        if not rows:
            continue

        print_stats("sigma_top", [r["sigma_top"] for r in rows], "{:.0f}")
        print_stats("scale", [r["scale"] for r in rows], "{:.2f}")
        print_stats("kl_max", [r["kl_max"] for r in rows], "{:.2f}")
        print_stats("|abs_ld|", [abs(r["abs_ld"]) for r in rows], "{:.3f}")
        if aligned_sign is not None:
            print_stats("aligned_ld", [r["aligned_ld"] for r in rows], "{:+.3f}")
            print_stats("misaligned_ld", [r["misaligned_ld"] for r in rows], "{:.3f}")

        actives = [r["n_active"] for r in rows]
        n_dead = sum(1 for a in actives if a == 0)
        print(
            f"  active vectors per pair (kl>=0.5): "
            f"mean={sum(actives) / len(actives):.1f}, "
            f"max={max(actives)}, dead pairs={n_dead}/{len(rows)}"
        )

        # Pick ranking + display columns based on --direction.
        if args.direction == "abs":
            sort_key = lambda r: abs(r["abs_ld"])
            ld_field, vi_field, sign_field = "abs_ld", "abs_vi", "abs_sign"
            label = "|abs_ld|"
        elif args.direction == "aligned":
            sort_key = lambda r: r["aligned_ld"]
            ld_field, vi_field, sign_field = "aligned_ld", "aligned_vi", "aligned_sign"
            label = "aligned_ld (+ = toward aligned)"
        else:  # misaligned
            sort_key = lambda r: r["misaligned_ld"]
            ld_field, vi_field, sign_field = "misaligned_ld", "misaligned_vi", "misaligned_sign"
            label = "misaligned_ld (+ = toward misaligned)"

        top = sorted(rows, key=sort_key, reverse=True)[:args.top]
        print(f"\n  Top {args.top} pairs by {label}:")
        print(f"    {'s':>3} {'t':>3} {'scale':>6} {'σ₁':>7} "
              f"{'kl_max':>7} {'ld':>9} {'vec':>5} {'#act':>5}")
        for r in top:
            vlabel = f"v{r[vi_field]}{r[sign_field]}"
            print(
                f"    {r['s']:>3} {r['t']:>3} "
                f"{r['scale']:>6.2f} {r['sigma_top']:>7.0f} "
                f"{r['kl_max']:>7.2f} {r[ld_field]:>+9.3f} "
                f"{vlabel:>5} {r['n_active']:>5}"
            )


if __name__ == "__main__":
    main()
