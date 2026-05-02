#!/usr/bin/env python3
"""Diagnose a saved CAA vector — sanity-check magnitude and signal-to-noise.

Loads vectors/<file>.pt and prints:
  - shape, dtype, layer, num_train, polarity convention
  - vector norm (mean across prompts is the post-mean norm — *that* is what gets
    unit-normalized for steering)
  - cosine similarity vs PI / MELBO vectors at the same source layer (if the
    PI/MELBO vectors are at a different layer, that comparison is skipped)
  - approximate per-prompt diff norm vs mean diff norm — when they're close
    the per-prompt differences are aligned (good signal); when the mean is
    much smaller than the per-prompt norm, the differences cancel (bad signal)

Usage:
    uv run python scripts/diagnose_caa.py experiments/<id>/vectors/caa_*.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from power_steering.utils import load_vectors, load_vector_metadata


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    caa_path = Path(sys.argv[1])
    print(f"\n=== CAA vector: {caa_path.name} ===")
    vec = load_vectors(caa_path)
    meta = load_vector_metadata(caa_path)

    print(f"shape={tuple(vec.shape)} dtype={vec.dtype}")
    print(f"layer={meta.get('layer')} (source_layer={meta.get('source_layer')})")
    print(f"num_train={meta.get('num_train')}  train_seed={meta.get('train_seed')}")
    print(f"position={meta.get('position')}  category={meta.get('category')}")
    print(f"vector_norm = {vec.norm(dim=-1).item():.4f}  "
          f"(this is mean(diffs).norm; small -> cancellation)")

    if vec.shape[0] == 1:
        v = vec[0]
        print(f"min/max element: {v.min().item():.4f} / {v.max().item():.4f}")
        print(f"mean/std element: {v.mean().item():.4f} / {v.std().item():.4f}")
        print(f"# elements > 0.1: {int((v.abs() > 0.1).sum())} / {v.numel()}")

    # Try to find sibling PI/MELBO vectors and check overlap
    siblings = sorted(caa_path.parent.glob("*.pt"))
    print(f"\n=== Cosine vs sibling vectors in {caa_path.parent} ===")
    for s in siblings:
        if s == caa_path:
            continue
        sib_vec = load_vectors(s)
        sib_meta = load_vector_metadata(s)
        sib_layer = sib_meta.get("source_layer", "?")
        for i in range(sib_vec.shape[0]):
            cs = cosine(vec[0], sib_vec[i])
            tag = f"{s.stem}[{i}] (layer={sib_layer})"
            print(f"  {tag:<60} cos={cs:+.4f}")

    print(
        "\nRule of thumb:\n"
        "  - vector_norm should be at least ~5–20% of typical per-prompt-diff norm.\n"
        "    If mean(diffs).norm is much smaller, the per-prompt differences cancel\n"
        "    -> CAA captured a noisy / dataset-incoherent direction, not a behavior axis.\n"
        "  - cos similarity vs PI/MELBO vectors at the same layer informs whether CAA\n"
        "    found a different direction or hit one of the dominant Jacobian directions.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
