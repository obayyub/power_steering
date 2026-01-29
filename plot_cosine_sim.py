#!/usr/bin/env python3
"""Generate cosine similarity heatmaps between steering vectors."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Vector files from the r=5 experiment
VECTOR_FILES = {
    "MELBO": "vectors/melbo_Qwen3-14B_20260127_165510.pt",
    "Power Iter": "vectors/power_iter_Qwen3-14B_20260127_165531.pt",
    "Multi-PI": "vectors/power_iter_multi_Qwen3-14B_20260127_165614.pt",
}


def load_vectors(filepath: str) -> torch.Tensor:
    """Load steering vectors from a .pt file."""
    data = torch.load(filepath, map_location="cpu", weights_only=True)
    # Handle different formats
    if isinstance(data, dict):
        if "vectors" in data:
            return data["vectors"]
        elif "steering" in data:
            return data["steering"]
    return data


def cosine_similarity_matrix(vecs1: torch.Tensor, vecs2: torch.Tensor) -> np.ndarray:
    """Compute cosine similarity between all pairs of vectors."""
    # Convert to float32 if needed
    vecs1 = vecs1.float()
    vecs2 = vecs2.float()
    # Normalize
    vecs1_norm = F.normalize(vecs1, dim=-1)
    vecs2_norm = F.normalize(vecs2, dim=-1)
    # Compute similarities
    sim = torch.mm(vecs1_norm, vecs2_norm.t())
    return sim.numpy()


def plot_heatmap(sim_matrix: np.ndarray, title: str, xlabel: str, ylabel: str, output_path: str):
    """Plot a cosine similarity heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity", fontsize=11)

    # Labels
    n_rows, n_cols = sim_matrix.shape
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([f"v{i}" for i in range(n_cols)])
    ax.set_yticklabels([f"v{i}" for i in range(n_rows)])

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = sim_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path("results/cosine_sim_heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all vectors
    vectors = {}
    for name, filepath in VECTOR_FILES.items():
        vectors[name] = load_vectors(filepath)
        print(f"Loaded {name}: {vectors[name].shape}")

    # Cross-method comparisons
    comparisons = [
        ("MELBO", "Power Iter"),
        ("MELBO", "Multi-PI"),
        ("Power Iter", "Multi-PI"),
    ]

    for method1, method2 in comparisons:
        sim = cosine_similarity_matrix(vectors[method1], vectors[method2])
        filename = f"{method1.lower().replace(' ', '_')}_vs_{method2.lower().replace(' ', '_').replace('-', '_')}.png"
        plot_heatmap(
            sim,
            title=f"Cosine Similarity: {method1} vs {method2}",
            ylabel=method1,
            xlabel=method2,
            output_path=str(output_dir / filename),
        )

    # Self-similarity (within method)
    for method in vectors:
        sim = cosine_similarity_matrix(vectors[method], vectors[method])
        filename = f"{method.lower().replace(' ', '_').replace('-', '_')}_self.png"
        plot_heatmap(
            sim,
            title=f"Cosine Similarity: {method} (self)",
            ylabel=method,
            xlabel=method,
            output_path=str(output_dir / filename),
        )


if __name__ == "__main__":
    main()
