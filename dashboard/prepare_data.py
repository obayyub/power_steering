#!/usr/bin/env python3
"""Precompute compact JSON for the dashboard from diverse map results.

Expects results/diverse_map/{prompt_id}/merged.pt for each prompt.
Also includes CoT data from results/jacobian_gen for the CoT tab.
"""

import json
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIVERSE_DIR = ROOT / "results/diverse_map"
NORMSCALE_DIR = ROOT / "results/diverse_map_normscale"
TGTVAR_DIR = ROOT / "results/diverse_map_tgtvar"
TGTINV_DIR = ROOT / "results/diverse_map_tgtinv"
TGTCOV_DIR = ROOT / "results/diverse_map_tgtcov"
GGB_DIR = ROOT / "results/diverse_map_ggb"
GGB_REG_DIR = ROOT / "results/diverse_map_ggb_reg"
DEEP_PI_DIR = ROOT / "results/deep_pi_map"
PROMPT_IDS = ["code", "narrative", "refusal", "reasoning",
              "strawberry", "roleplay", "persuasion"]


def load_diverse():
    """Load per-prompt sigma/KL from diverse_map merged.pt files."""
    per_prompt = {}
    num_layers = None
    num_vectors = None
    model_name = None

    for pid in PROMPT_IDS:
        merged_path = DIVERSE_DIR / pid / "merged.pt"
        if not merged_path.exists():
            print(f"  Warning: {merged_path} not found, skipping")
            continue

        merged = torch.load(merged_path, map_location="cpu", weights_only=True)
        sigma_map = merged["sigma_map"]  # [n, n, k]
        kl_map = merged["kl_map"]  # [n, n, k]

        if num_layers is None:
            num_layers = int(sigma_map.shape[0])
            num_vectors = int(sigma_map.shape[2])
            model_name = merged["metadata"]["model"]

        pairs = {}
        for s in range(num_layers):
            for t in range(s + 1, num_layers):
                if sigma_map[s, t, 0].isnan():
                    continue
                key = f"{s}_{t}"
                pairs[key] = {
                    "s": s, "t": t,
                    "sigmas": [round(v, 2) for v in sigma_map[s, t].tolist()],
                    "kls": [round(v, 4) for v in kl_map[s, t].tolist()],
                }

        # Load baseline
        baseline_path = DIVERSE_DIR / pid / "baseline.json"
        baseline = []
        if baseline_path.exists():
            with open(baseline_path) as f:
                bl = json.load(f)
            baseline = bl["results"]

        per_prompt[pid] = {
            "pairs": pairs,
            "baseline": baseline,
        }
        print(f"  [{pid}] {len(pairs)} pairs loaded")

    return per_prompt, num_layers, num_vectors, model_name


def load_normscale():
    """Load norm-scaled diverse map data (scale_frac run)."""
    per_prompt = {}
    num_layers = None
    num_vectors = None
    model_name = None
    scale_frac = None

    for pid in PROMPT_IDS:
        merged_path = NORMSCALE_DIR / pid / "merged.pt"
        if not merged_path.exists():
            continue

        merged = torch.load(merged_path, map_location="cpu", weights_only=True)
        sigma_map = merged["sigma_map"]
        kl_map = merged["kl_map"]
        scale_map = merged.get("scale_map")

        if num_layers is None:
            num_layers = int(sigma_map.shape[0])
            num_vectors = int(sigma_map.shape[2])
            model_name = merged["metadata"]["model"]
            scale_frac = merged["metadata"].get("scale_frac")

        pairs = {}
        for s in range(num_layers):
            for t in range(s + 1, num_layers):
                if sigma_map[s, t, 0].isnan():
                    continue
                key = f"{s}_{t}"
                pair = {
                    "s": s, "t": t,
                    "sigmas": [round(v, 2) for v in sigma_map[s, t].tolist()],
                    "kls": [round(v, 4) for v in kl_map[s, t].tolist()],
                }
                if scale_map is not None and not scale_map[s, t].isnan():
                    pair["scale"] = round(float(scale_map[s, t]), 2)
                pairs[key] = pair

        baseline_path = NORMSCALE_DIR / pid / "baseline.json"
        baseline = []
        if baseline_path.exists():
            with open(baseline_path) as f:
                bl = json.load(f)
            baseline = bl["results"]

        per_prompt[pid] = {
            "pairs": pairs,
            "baseline": baseline,
        }
        print(f"  [{pid}] {len(pairs)} pairs loaded")

    if not per_prompt:
        return None

    # Load prompt metadata
    prompts = []
    for pid in PROMPT_IDS:
        summary_path = NORMSCALE_DIR / pid / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            prompts.append(s["prompt"])

    return {
        "model": model_name,
        "num_layers": num_layers,
        "num_vectors": num_vectors,
        "scale_frac": scale_frac,
        "prompt_ids": [p["id"] for p in prompts],
        "prompts": prompts,
        "per_prompt": per_prompt,
    }


def load_cot():
    """Load CoT sigma/KL/accuracy data (for CoT tab)."""
    merged_path = ROOT / "results/jacobian_map/merged.pt"
    if not merged_path.exists():
        return None

    merged = torch.load(merged_path, map_location="cpu", weights_only=True)
    sigma_map = merged["sigma_map"]
    kl_map = merged["kl_map"]

    num_layers = int(sigma_map.shape[0])
    num_vectors = int(sigma_map.shape[2])
    model_name = merged["metadata"]["model"]

    # Load per-vector accuracy from generation summary
    summary_path = ROOT / "results/jacobian_gen/summary.json"
    gen_summary = None
    if summary_path.exists():
        with open(summary_path) as f:
            gen_summary = json.load(f)

    cot_pairs = {}
    for s in range(num_layers):
        for t in range(s + 1, num_layers):
            if sigma_map[s, t, 0].isnan():
                continue
            key = f"{s}_{t}"
            pair = {
                "s": s, "t": t,
                "sigmas": [round(v, 2) for v in sigma_map[s, t].tolist()],
                "kls": [round(v, 4) for v in kl_map[s, t].tolist()],
            }
            # Add accuracy if generation data exists for this pair
            if gen_summary and key in gen_summary["pairs"]:
                gp = gen_summary["pairs"][key]
                pair["accs"] = [
                    round(gp["per_vector"].get(str(i), 0.0), 4)
                    for i in range(num_vectors)
                ]
            cot_pairs[key] = pair

    return {
        "model": model_name,
        "num_layers": num_layers,
        "num_vectors": num_vectors,
        "pairs": cot_pairs,
        "questions": gen_summary.get("questions") if gen_summary else None,
    }


def load_target_metric(results_dir):
    """Load target-metric diverse map data (tgtvar/tgtinv).

    These results have:
    - Only prompt subdirectories that exist (e.g. just 'roleplay')
    - Generations as a dict keyed by gen_prompt_id
    - Per-gen-prompt baseline files (baseline_roleplay.json, etc.)
    - scale_map in merged.pt
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return None

    per_prompt = {}
    num_layers = None
    num_vectors = None
    model_name = None
    scale_frac = None
    target_metric = None

    # Scan for prompt subdirectories containing merged.pt
    for subdir in sorted(results_dir.iterdir()):
        merged_path = subdir / "merged.pt"
        if not subdir.is_dir() or not merged_path.exists():
            continue

        pid = subdir.name
        merged = torch.load(merged_path, map_location="cpu", weights_only=True)
        sigma_map = merged["sigma_map"]
        kl_map = merged["kl_map"]
        scale_map = merged.get("scale_map")

        if num_layers is None:
            num_layers = int(sigma_map.shape[0])
            num_vectors = int(sigma_map.shape[2])
            model_name = merged["metadata"]["model"]
            scale_frac = merged["metadata"].get("scale_frac")
            target_metric = merged["metadata"].get("target_metric")

        pairs = {}
        for s in range(num_layers):
            for t in range(s + 1, num_layers):
                if sigma_map[s, t, 0].isnan():
                    continue
                key = f"{s}_{t}"
                pair = {
                    "s": s, "t": t,
                    "sigmas": [round(v, 2) for v in sigma_map[s, t].tolist()],
                    "kls": [round(v, 4) for v in kl_map[s, t].tolist()],
                }
                if scale_map is not None and not scale_map[s, t].isnan():
                    pair["scale"] = round(float(scale_map[s, t]), 2)
                pairs[key] = pair

        # Load per-gen-prompt baselines
        baselines = {}
        for bl_path in sorted(subdir.glob("baseline_*.json")):
            # baseline_roleplay_captain.json -> roleplay_captain
            gen_prompt_id = bl_path.stem.replace("baseline_", "")
            with open(bl_path) as f:
                bl = json.load(f)
            baselines[gen_prompt_id] = bl["results"]

        per_prompt[pid] = {
            "pairs": pairs,
            "baselines": baselines,
        }
        print(f"  [{pid}] {len(pairs)} pairs, {len(baselines)} gen-prompts loaded")

    if not per_prompt:
        return None

    return {
        "model": model_name,
        "num_layers": num_layers,
        "num_vectors": num_vectors,
        "scale_frac": scale_frac,
        "target_metric": target_metric,
        "prompt_ids": list(per_prompt.keys()),
        "per_prompt": per_prompt,
    }


def load_deep_pi():
    """Load deep PI map data (100 vectors per pair, generations for KL>1)."""
    results_dir = DEEP_PI_DIR
    if not results_dir.exists():
        return None

    # Look for the prompt subdirectory (roleplay)
    per_prompt = {}
    num_layers = None
    num_vectors = None
    model_name = None
    scale_frac = None

    for subdir in sorted(results_dir.iterdir()):
        merged_path = subdir / "merged.pt"
        if not subdir.is_dir() or not merged_path.exists():
            continue

        pid = subdir.name
        merged = torch.load(merged_path, map_location="cpu", weights_only=True)
        sigma_map = merged["sigma_map"]
        kl_map = merged["kl_map"]
        scale_map = merged.get("scale_map")

        if num_layers is None:
            num_layers = int(sigma_map.shape[0])
            num_vectors = int(sigma_map.shape[2])
            model_name = merged["metadata"]["model"]
            scale_frac = merged["metadata"].get("scale_frac")

        pairs = {}
        for s in range(num_layers):
            for t in range(s + 1, num_layers):
                if sigma_map[s, t, 0].isnan():
                    continue
                key = f"{s}_{t}"
                pair = {
                    "s": s, "t": t,
                    "sigmas": [round(v, 2) for v in sigma_map[s, t].tolist()],
                    "kls": [round(v, 4) for v in kl_map[s, t].tolist()],
                }
                if scale_map is not None and not scale_map[s, t].isnan():
                    pair["scale"] = round(float(scale_map[s, t]), 2)
                pairs[key] = pair

        # Load baseline
        baseline_path = subdir / "baseline_roleplay.json"
        baseline_text = None
        if baseline_path.exists():
            with open(baseline_path) as f:
                bl = json.load(f)
            baseline_text = bl.get("text", "")

        per_prompt[pid] = {
            "pairs": pairs,
            "baseline_text": baseline_text,
        }
        print(f"  [{pid}] {len(pairs)} pairs loaded")

    if not per_prompt:
        return None

    return {
        "model": model_name,
        "num_layers": num_layers,
        "num_vectors": num_vectors,
        "scale_frac": scale_frac,
        "prompt_ids": list(per_prompt.keys()),
        "per_prompt": per_prompt,
    }


def main():
    print("Loading diverse map data...")
    per_prompt, num_layers, num_vectors, model_name = load_diverse()

    if not per_prompt:
        print("No diverse map data found. Run map_diverse.py first.")
        return

    print("\nLoading norm-scaled data...")
    normscale = load_normscale()
    if normscale:
        total_pairs = sum(len(v["pairs"]) for v in normscale["per_prompt"].values())
        print(f"  Norm-scaled: {normscale['model']}, {normscale['num_layers']} layers, "
              f"scale_frac={normscale['scale_frac']}, {total_pairs} pairs")
    else:
        print("  No norm-scaled data found (optional)")

    print("\nLoading target-metric var data...")
    tgtvar = load_target_metric(TGTVAR_DIR)
    if tgtvar:
        total = sum(len(v["pairs"]) for v in tgtvar["per_prompt"].values())
        print(f"  tgtvar: {tgtvar['model']}, {tgtvar['num_layers']} layers, {total} pairs")
    else:
        print("  No tgtvar data found (optional)")

    print("\nLoading target-metric inv data...")
    tgtinv = load_target_metric(TGTINV_DIR)
    if tgtinv:
        total = sum(len(v["pairs"]) for v in tgtinv["per_prompt"].values())
        print(f"  tgtinv: {tgtinv['model']}, {tgtinv['num_layers']} layers, {total} pairs")
    else:
        print("  No tgtinv data found (optional)")

    print("\nLoading target-metric cov data...")
    tgtcov = load_target_metric(TGTCOV_DIR)
    if tgtcov:
        total = sum(len(v["pairs"]) for v in tgtcov["per_prompt"].values())
        print(f"  tgtcov: {tgtcov['model']}, {tgtcov['num_layers']} layers, {total} pairs")
    else:
        print("  No tgtcov data found (optional)")

    print("\nLoading GGB cov data...")
    ggb = load_target_metric(GGB_DIR)
    if ggb:
        total = sum(len(v["pairs"]) for v in ggb["per_prompt"].values())
        print(f"  ggb: {ggb['model']}, {ggb['num_layers']} layers, {total} pairs")
    else:
        print("  No GGB data found (optional)")

    print("\nLoading GGB reg cov data...")
    ggb_reg = load_target_metric(GGB_REG_DIR)
    if ggb_reg:
        total = sum(len(v["pairs"]) for v in ggb_reg["per_prompt"].values())
        print(f"  ggb_reg: {ggb_reg['model']}, {ggb_reg['num_layers']} layers, {total} pairs")
    else:
        print("  No GGB reg data found (optional)")

    print("\nLoading deep PI map data...")
    deep_pi = load_deep_pi()
    if deep_pi:
        total = sum(len(v["pairs"]) for v in deep_pi["per_prompt"].values())
        print(f"  deep_pi: {deep_pi['model']}, {deep_pi['num_layers']} layers, "
              f"{deep_pi['num_vectors']} vectors, {total} pairs")
    else:
        print("  No deep PI data found (optional)")

    print("\nLoading CoT data...")
    cot = load_cot()
    if cot:
        print(f"  CoT: {cot['model']}, {cot['num_layers']} layers, {len(cot['pairs'])} pairs")
    else:
        print("  No CoT data found (optional)")

    # Load prompt metadata
    prompts = []
    for pid in PROMPT_IDS:
        summary_path = DIVERSE_DIR / pid / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            prompts.append(s["prompt"])

    data = {
        "num_layers": num_layers,
        "num_vectors": num_vectors,
        "model": model_name,
        "prompt_ids": [p["id"] for p in prompts],
        "prompts": prompts,
        "per_prompt": per_prompt,
    }

    if normscale:
        data["normscale"] = normscale

    if tgtvar:
        data["tgtvar"] = tgtvar

    if tgtinv:
        data["tgtinv"] = tgtinv

    if tgtcov:
        data["tgtcov"] = tgtcov

    if ggb:
        data["ggb"] = ggb

    if ggb_reg:
        data["ggb_reg"] = ggb_reg

    if deep_pi:
        data["deep_pi"] = deep_pi

    if cot:
        data["cot"] = cot

    out = Path(__file__).parent / "dashboard_data.json"
    with open(out, "w") as f:
        json.dump(data, f)

    print(f"\nWrote {out}: {out.stat().st_size / 1024:.0f} KB")

    # Split diverse gen pairs into per-pair files for lazy loading
    diverse_pairs_dir = Path(__file__).parent / "diverse_pairs"
    diverse_pairs_dir.mkdir(exist_ok=True)

    pair_count = 0
    for pid in PROMPT_IDS:
        pid_dir = diverse_pairs_dir / pid
        pid_dir.mkdir(exist_ok=True)
        pairs_source = DIVERSE_DIR / pid / "pairs"
        if not pairs_source.exists():
            continue
        for f in sorted(pairs_source.glob("*.json")):
            # Copy pair file (already has generations)
            with open(f) as fp:
                pair_data = json.load(fp)
            # Strip vectors from pair files (large, not needed for dashboard)
            pair_data.pop("vectors", None)
            with open(pid_dir / f.name, "w") as fp:
                json.dump(pair_data, fp)
            pair_count += 1

    print(f"Split {pair_count} diverse pairs into {diverse_pairs_dir}")

    # Split target-metric gen pairs (tgtvar / tgtinv)
    for label, tm_data, tm_dir in [("tgtvar", tgtvar, TGTVAR_DIR),
                                    ("tgtinv", tgtinv, TGTINV_DIR),
                                    ("tgtcov", tgtcov, TGTCOV_DIR),
                                    ("ggb", ggb, GGB_DIR),
                                    ("ggb_reg", ggb_reg, GGB_REG_DIR)]:
        if not tm_data:
            continue
        tm_pairs_dir = Path(__file__).parent / f"{label}_pairs"
        tm_pairs_dir.mkdir(exist_ok=True)
        tm_count = 0
        for pid in tm_data["prompt_ids"]:
            pid_dir = tm_pairs_dir / pid
            pid_dir.mkdir(exist_ok=True)
            pairs_source = tm_dir / pid / "pairs"
            if not pairs_source.exists():
                continue
            for f in sorted(pairs_source.glob("*.json")):
                with open(f) as fp:
                    pair_data = json.load(fp)
                pair_data.pop("vectors", None)
                with open(pid_dir / f.name, "w") as fp:
                    json.dump(pair_data, fp)
                tm_count += 1
        print(f"Split {tm_count} {label} pairs into {tm_pairs_dir}")

    # Split deep PI gen pairs
    if deep_pi:
        dp_pairs_dir = Path(__file__).parent / "deep_pi_pairs"
        dp_pairs_dir.mkdir(exist_ok=True)
        dp_count = 0
        for pid in deep_pi["prompt_ids"]:
            pid_dir = dp_pairs_dir / pid
            pid_dir.mkdir(exist_ok=True)
            pairs_source = DEEP_PI_DIR / pid / "pairs"
            if not pairs_source.exists():
                continue
            for f in sorted(pairs_source.glob("*.json")):
                with open(f) as fp:
                    pair_data = json.load(fp)
                pair_data.pop("vectors", None)
                with open(pid_dir / f.name, "w") as fp:
                    json.dump(pair_data, fp)
                dp_count += 1
        print(f"Split {dp_count} deep PI pairs into {dp_pairs_dir}")

    # Split norm-scaled gen pairs
    if normscale:
        ns_pairs_dir = Path(__file__).parent / "normscale_pairs"
        ns_pairs_dir.mkdir(exist_ok=True)
        ns_count = 0
        for pid in normscale["prompt_ids"]:
            pid_dir = ns_pairs_dir / pid
            pid_dir.mkdir(exist_ok=True)
            pairs_source = NORMSCALE_DIR / pid / "pairs"
            if not pairs_source.exists():
                continue
            for f in sorted(pairs_source.glob("*.json")):
                with open(f) as fp:
                    pair_data = json.load(fp)
                pair_data.pop("vectors", None)
                with open(pid_dir / f.name, "w") as fp:
                    json.dump(pair_data, fp)
                ns_count += 1
        print(f"Split {ns_count} norm-scaled pairs into {ns_pairs_dir}")


if __name__ == "__main__":
    main()
