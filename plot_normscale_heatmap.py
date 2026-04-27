#!/usr/bin/env python3
"""Plot max-KL heatmap for normscale (norm-proportional scaling) data.

Reads from dashboard/dashboard_data.json and produces a heatmap matching
the style of docs/kl_heatmap.png but for normscale KLs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "dashboard/dashboard_data.json"
PROMPT_ID = "refusal"


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    ns = data["normscale"]
    num_layers = ns["num_layers"]
    scale_frac = ns["scale_frac"]
    model = ns["model"].split("/")[-1]
    pairs = ns["per_prompt"][PROMPT_ID]["pairs"]

    # Build max-KL matrix
    mat = np.full((num_layers, num_layers), np.nan)
    for p in pairs.values():
        s, t = p["s"], p["t"]
        mat[s, t] = max(p["kls"])

    # Mask lower triangle + diagonal (no data there)
    masked = np.ma.array(mat, mask=np.tri(num_layers, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        masked,
        origin="upper",
        cmap="YlOrRd",
        aspect="equal",
    )

    ax.set_xlabel("Target layer", fontsize=15)
    ax.set_ylabel("Source layer", fontsize=15)
    ax.set_title(
        f"Max KL divergence per layer pair\n{model}, {PROMPT_ID} prompt, "
        f"scale={scale_frac}·‖σ‖",
        fontsize=16,
    )

    ticks = list(range(0, num_layers, 5))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=13)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Max KL divergence", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    out_path = f"docs/kl_heatmap_normscale.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
