"""Render the refusal-phishing KL atlas heatmap for §4.

Loads the merged Power-Steering free-form atlas tensor and renders a
40×40 (source layer × target layer) heatmap of max KL divergence
across the k=12 candidate vectors and both injection signs per pair.

Highlights the four (s, t) cells cited in the paper:
  - (24, 37) — Chinese-refusal vector v4+
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
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
ATLAS = REPO / "experiments/map_freeform_refusal_Qwen3-14B/freeform/refusal_phishing/merged.pt"
OUT = REPO / "paper_artifacts" / "refusal_atlas_kl"

CITED = [
    (24, 37, "Chinese-refusal v4+"),
    (19, 28, "AdvBench v5+"),
    (20, 28, "AdvBench v9+"),
    (20, 26, "AdvBench v7-"),
]

KL_CLIP = 15.0  # Clip top values; a few early-layer cells go to 35-41 and
                # would crush the rest of the colormap dynamic range.


def main() -> None:
    d = torch.load(ATLAS, weights_only=False, map_location="cpu")
    kl_pos = d["kl_pos"]
    kl_neg = d["kl_neg"]
    # Per-pair max over (signs, k vectors); NaN-safe
    both = torch.stack([kl_pos, kl_neg], dim=-1)  # (S, T, k, 2)
    flat = both.reshape(both.shape[0], both.shape[1], -1)  # (S, T, 2k)
    kl = torch.nanquantile(flat, 1.0, dim=-1)  # max ignoring NaN
    kl = kl.numpy()

    # Mask all-NaN cells (i.e., (s,t) pairs the atlas didn't compute)
    mask = np.isnan(kl)
    kl_clipped = np.clip(np.where(mask, 0, kl), 0, KL_CLIP)
    kl_display = np.ma.masked_where(mask, kl_clipped)

    n_active = int(((~mask) & (kl >= 0.5)).sum())
    n_computed = int((~mask).sum())
    n_total = mask.size

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#e5e7eb")  # gray for non-computed pairs

    im = ax.imshow(kl_display, cmap=cmap, origin="lower", aspect="equal",
                   vmin=0, vmax=KL_CLIP)

    ax.set_xlabel("Target layer t", fontsize=10)
    ax.set_ylabel("Source layer s", fontsize=10)
    ax.set_title(
        f"Refusal-phishing KL atlas (Qwen3-14B, k=12 per pair)\n"
        f"{n_active} active pairs (max-KL ≥ 0.5) of {n_computed} computed "
        f"({n_total - n_computed} not computed, in gray)",
        fontsize=10, pad=10,
    )

    # Mark cited cells with numbered markers; legend below.
    import matplotlib.patheffects as pe
    for n, (s, t, label) in enumerate(CITED, start=1):
        ax.add_patch(mpatches.Circle((t, s), radius=0.85, fill=False,
                                      edgecolor="white", linewidth=1.8))
        ax.add_patch(mpatches.Circle((t, s), radius=0.85, fill=False,
                                      edgecolor="black", linewidth=0.6))
        txt = ax.text(t, s, str(n), ha="center", va="center",
                      fontsize=9, color="white", weight="bold")
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    # Build a legend that maps the numbers to the cell coords + descriptions
    legend_text = "\n".join(
        f"{n}. ({s},{t}) — {label}" for n, (s, t, label) in enumerate(CITED, start=1)
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#9ca3af", linewidth=0.6, alpha=0.92))

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03,
                        label=f"max KL across (k=12, ±sign), clipped at {KL_CLIP}")

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(str(OUT) + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}.png")
    print(f"Saved {OUT}.pdf")


if __name__ == "__main__":
    main()
