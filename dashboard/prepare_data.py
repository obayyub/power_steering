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


def main():
    print("Loading diverse map data...")
    per_prompt, num_layers, num_vectors, model_name = load_diverse()

    if not per_prompt:
        print("No diverse map data found. Run map_diverse.py first.")
        return

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


if __name__ == "__main__":
    main()
