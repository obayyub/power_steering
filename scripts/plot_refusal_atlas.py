"""Render the refusal-phishing KL atlas heatmap for §4.

Crops to the layer range actually examined by the atlas (s ∈ [4, 35],
t ∈ [5, 37]) so the figure is tight and legible at NeurIPS column
width. Highlights the three (s, t) cells used in §4's AdvBench transfer:

  - (19, 28) — AdvBench candidate v5+
  - (20, 28) — AdvBench candidate v9+
  - (20, 26) — AdvBench candidate v7-

Usage:
    uv run python scripts/plot_refusal_atlas.py

Output:
    paper_artifacts/refusal_atlas_kl.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
ATLAS = REPO / "experiments/map_freeform_refusal_Qwen3-14B/freeform/refusal_phishing/merged.pt"
OUT = REPO / "paper_artifacts" / "refusal_atlas_kl"

CITED = [
    (19, 28, "AdvBench v5+"),
    (20, 28, "AdvBench v9+"),
    (20, 26, "AdvBench v7-"),
]

# Crop bounds — the atlas actually examined s ∈ [4, 35], t ∈ [5, 37];
# everything outside is gray noise that wastes plot real estate.
S_MIN, S_MAX = 4, 35
T_MIN, T_MAX = 5, 37

KL_CLIP = 15.0  # Clip top values; a few early-layer cells reach 35-41
                # and would crush the rest of the colormap dynamic range.


def main() -> None:
    d = torch.load(ATLAS, weights_only=False, map_location="cpu")
    kl_pos = d["kl_pos"]
    kl_neg = d["kl_neg"]
    both = torch.stack([kl_pos, kl_neg], dim=-1)
    flat = both.reshape(both.shape[0], both.shape[1], -1)
    kl = torch.nanquantile(flat, 1.0, dim=-1).numpy()

    # Crop to the examined range
    kl = kl[S_MIN:S_MAX + 1, T_MIN:T_MAX + 1]
    s_axis = np.arange(S_MIN, S_MAX + 1)
    t_axis = np.arange(T_MIN, T_MAX + 1)

    mask = np.isnan(kl)
    kl_clipped = np.clip(np.where(mask, 0, kl), 0, KL_CLIP)
    kl_display = np.ma.masked_where(mask, kl_clipped)

    n_active = int(((~mask) & (kl >= 0.5)).sum())
    n_computed = int((~mask).sum())

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#e5e7eb")

    # imshow extent so x/y use absolute layer indices
    extent = (T_MIN - 0.5, T_MAX + 0.5, S_MIN - 0.5, S_MAX + 0.5)
    im = ax.imshow(kl_display, cmap=cmap, origin="lower", aspect="equal",
                   vmin=0, vmax=KL_CLIP, extent=extent)

    # Tick every 5 layers on each axis for a clean grid
    s_ticks = [s for s in range(S_MIN, S_MAX + 1) if s % 5 == 0]
    t_ticks = [t for t in range(T_MIN, T_MAX + 1) if t % 5 == 0]
    ax.set_xticks(t_ticks)
    ax.set_yticks(s_ticks)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("Target layer t", fontsize=9)
    ax.set_ylabel("Source layer s", fontsize=9)
    ax.set_title(
        f"Refusal-phishing KL atlas (Qwen3-14B)\n"
        f"{n_active}/{n_computed} active pairs (max-KL ≥ 0.5)",
        fontsize=9, pad=6,
    )

    for n, (s, t, _label) in enumerate(CITED, start=1):
        ax.add_patch(mpatches.Circle((t, s), radius=0.9, fill=False,
                                      edgecolor="white", linewidth=2.0))
        ax.add_patch(mpatches.Circle((t, s), radius=0.9, fill=False,
                                      edgecolor="black", linewidth=0.6))
        txt = ax.text(t, s, str(n), ha="center", va="center",
                      fontsize=8.5, color="white", weight="bold")
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    legend_text = "\n".join(
        f"{n}. ({s},{t}) — {label}"
        for n, (s, t, label) in enumerate(CITED, start=1)
    )
    ax.text(0.03, 0.97, legend_text, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#9ca3af", linewidth=0.6, alpha=0.92))

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label(f"max KL (clipped at {KL_CLIP:.0f})", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(str(OUT) + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}.png")
    print(f"Saved {OUT}.pdf")


if __name__ == "__main__":
    main()
