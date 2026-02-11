#!/usr/bin/env python3
"""Precompute compact JSON for the dashboard from mapping + generation results."""

import json
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Load sigma and KL from merged.pt
merged = torch.load(ROOT / "results/jacobian_map/merged.pt", map_location="cpu", weights_only=True)
sigma_map = merged["sigma_map"]  # [n, n, k]
kl_map = merged["kl_map"]  # [n, n, k]

# Load per-vector accuracy from generation summary
with open(ROOT / "results/jacobian_gen/summary.json") as f:
    gen_summary = json.load(f)

num_layers = int(sigma_map.shape[0])
num_vectors = int(sigma_map.shape[2])

pairs = {}
for key, pair_data in gen_summary["pairs"].items():
    s = pair_data["source_layer"]
    t = pair_data["target_layer"]
    sigmas = sigma_map[s, t].tolist()
    kls = kl_map[s, t].tolist()
    accs = [pair_data["per_vector"].get(str(i), 0.0) for i in range(num_vectors)]

    pairs[key] = {
        "s": s, "t": t,
        "sigmas": [round(v, 2) for v in sigmas],
        "kls": [round(v, 4) for v in kls],
        "accs": [round(v, 4) for v in accs],
    }

data = {
    "num_layers": num_layers,
    "num_vectors": num_vectors,
    "model": merged["metadata"]["model"],
    "pairs": pairs,
    "questions": gen_summary["questions"],
}

out = Path(__file__).parent / "dashboard_data.json"
with open(out, "w") as f:
    json.dump(data, f)

print(f"Wrote {out}: {len(pairs)} pairs, {out.stat().st_size / 1024:.0f} KB")
