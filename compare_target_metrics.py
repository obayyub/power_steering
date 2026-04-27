#!/usr/bin/env python3
"""
Compare vector alignment across target metrics (kl, var, inv).

For each (source, target) layer pair, measures:
  - Top-1 alignment: |cos(v0_A, v0_B)|
  - Subspace overlap (top-3): mean principal cosine of V_A[:3] @ V_B[:3].T

Output: results/target_metric_alignment.png
  Row 1: 6 heatmaps (3 method-pairs x 2 metrics)
  Row 2: 2 line plots (depth profile per method-pair)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Load data ──────────────────────────────────────────────────────────────

DATA = {
    "baseline": "results/diverse_map_normscale/roleplay/merged.pt",
    "var":      "results/diverse_map_tgtvar/roleplay/merged.pt",
    "inv":      "results/diverse_map_tgtinv/roleplay/merged.pt",
    "cov":      "results/diverse_map_tgtcov/roleplay/merged.pt",
}

METHOD_PAIRS = [
    ("var", "baseline"),
    ("inv", "baseline"),
    ("cov", "baseline"),
    ("cov", "var"),
    ("cov", "inv"),
    ("var", "inv"),
]

N = 36  # layers
TOP_K = 3  # subspace overlap dimension


def load_vectors():
    """Load vectors from all three merged.pt files.

    Returns: dict[method_name] -> dict["s_t" -> tensor [12, 4096]]
    """
    vecs = {}
    for name, path in DATA.items():
        d = torch.load(path, weights_only=False)
        vecs[name] = d["vectors"]
        print(f"Loaded {name}: {len(d['vectors'])} pairs")
    return vecs


def compute_alignment(vecs):
    """Compute top-1 cosine and top-3 subspace overlap for all method pairs.

    Returns:
        top1: dict[(a, b)] -> [N, N] numpy array
        subspace: dict[(a, b)] -> [N, N] numpy array
    """
    top1 = {}
    subspace = {}

    for a, b in METHOD_PAIRS:
        mat_top1 = np.full((N, N), np.nan)
        mat_sub = np.full((N, N), np.nan)

        va = vecs[a]
        vb = vecs[b]

        common_keys = sorted(set(va.keys()) & set(vb.keys()))
        for key in common_keys:
            s, t = map(int, key.split("_"))

            # Normalize vectors
            A = va[key].float()  # [12, H]
            B = vb[key].float()  # [12, H]
            A = A / A.norm(dim=1, keepdim=True)
            B = B / B.norm(dim=1, keepdim=True)

            # Top-1: |cos(v0_A, v0_B)|
            cos01 = (A[0] @ B[0]).abs().item()
            mat_top1[s, t] = cos01

            # Subspace overlap (top-3): mean singular value of A[:3] @ B[:3].T
            G = A[:TOP_K] @ B[:TOP_K].T  # [3, 3]
            svs = torch.linalg.svdvals(G)
            mat_sub[s, t] = svs.mean().item()

        top1[(a, b)] = mat_top1
        subspace[(a, b)] = mat_sub
        valid = ~np.isnan(mat_top1)
        print(f"  {a} vs {b}: {valid.sum()} pairs, "
              f"top1 mean={mat_top1[valid].mean():.3f}, "
              f"subspace mean={mat_sub[valid].mean():.3f}")

    return top1, subspace


def plot(top1, subspace, out_path="results/target_metric_alignment.png"):
    """Plot heatmaps (row 1-2) + line plots (row 3)."""

    n_pairs = len(METHOD_PAIRS)
    fig = plt.figure(figsize=(28, 18))
    gs = fig.add_gridspec(3, n_pairs * 2, height_ratios=[1, 1, 0.8],
                          hspace=0.4, wspace=0.45)

    pair_labels = {(a, b): f"{a} vs {b}" for a, b in METHOD_PAIRS}
    metric_labels = ["Top-1 |cos|", "Subspace overlap (top-3)"]

    # ── Rows 1-2: Heatmaps ────────────────────────────────────────────────
    for col, (pair, label) in enumerate(pair_labels.items()):
        row = 0 if col < n_pairs // 2 + n_pairs % 2 else 1
        hcol = col if row == 0 else col - (n_pairs // 2 + n_pairs % 2)
        for mi, (mat, mlabel) in enumerate([(top1[pair], metric_labels[0]),
                                             (subspace[pair], metric_labels[1])]):
            ax = fig.add_subplot(gs[row, hcol * 2 + mi])
            masked = np.ma.array(mat, mask=np.tri(N, dtype=bool))
            im = ax.imshow(masked, origin="upper", cmap="viridis",
                           vmin=0, vmax=1, aspect="equal")
            ax.set_title(f"{label}\n{mlabel}", fontsize=10)
            ax.set_xlabel("target layer")
            ax.set_ylabel("source layer")
            ticks = list(range(0, N, 5))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 3: Line plots ─────────────────────────────────────────────────
    colors_list = ["C0", "C1", "C2", "C3", "C4", "C5"]
    colors = {label: colors_list[i] for i, label in enumerate(pair_labels.values())}

    for mi, (matrices, mlabel) in enumerate([(top1, metric_labels[0]),
                                              (subspace, metric_labels[1])]):
        ax = fig.add_subplot(gs[2, mi * n_pairs:(mi + 1) * n_pairs])

        for pair, label in pair_labels.items():
            mat = matrices[pair]
            means = []
            for s in range(N):
                targets = mat[s, s + 1:]
                valid = targets[~np.isnan(targets)]
                means.append(valid.mean() if len(valid) > 0 else np.nan)

            ax.plot(range(N), means, label=label, color=colors[label],
                    linewidth=1.5, alpha=0.85)

        ax.set_xlabel("source layer")
        ax.set_ylabel(f"mean {mlabel}")
        ax.set_title(f"{mlabel} by source layer (averaged over targets)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, N - 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    fig.suptitle("Vector alignment across target metrics (Qwen3-8B, roleplay prompt)",
                 fontsize=14, y=0.99)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


def main():
    vecs = load_vectors()
    top1, subspace = compute_alignment(vecs)
    plot(top1, subspace)


if __name__ == "__main__":
    main()
