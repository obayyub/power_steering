"""Leave-one-out subspace analysis for refusal steering vectors.

Tests whether anti-refusal vectors share a common subspace in the residual stream,
using held-out vectors to avoid the circularity of building a plane from the same
vectors you measure projection onto.
"""

import torch
import numpy as np


def load_vectors(merged_pt_path, labeled_vectors):
    """Load labeled vectors from merged.pt.

    labeled_vectors: list of (source, target, vec_idx, label) tuples
    Returns dict of {label: list of (name, vector) tuples}
    """
    d = torch.load(merged_pt_path, weights_only=False)
    vectors_by_label = {}
    for s, t, v, label in labeled_vectors:
        key = f"{s}_{t}"
        vec = d["vectors"][key][v].float()
        vec = vec / vec.norm()  # ensure unit norm
        name = f"s{s}t{t}v{v}"
        vectors_by_label.setdefault(label, []).append((name, vec))
    return vectors_by_label


def leave_one_out_projection(vectors, subspace_dim=2):
    """For each vector, build subspace from the rest and compute projection.

    Returns list of (name, projection_fraction) tuples.
    """
    names = [n for n, _ in vectors]
    vecs = torch.stack([v for _, v in vectors])  # [N, H]
    results = []

    for i in range(len(vectors)):
        # Build subspace from all vectors except i
        others = torch.cat([vecs[:i], vecs[i+1:]], dim=0)  # [N-1, H]
        U, S, Vt = torch.linalg.svd(others, full_matrices=False)
        basis = Vt[:subspace_dim]  # [d, H] — top-d right singular vectors

        # Project held-out vector onto subspace
        held_out = vecs[i]  # [H]
        proj = basis @ held_out  # [d]
        proj_frac = (proj.norm() / held_out.norm()).item()
        results.append((names[i], proj_frac))

    return results


def random_projection_baseline(vectors, subspace_dim=2, n_random=1000, seed=42):
    """Compute projection of random unit vectors onto the full subspace."""
    vecs = torch.stack([v for _, v in vectors])  # [N, H]
    H = vecs.shape[1]
    U, S, Vt = torch.linalg.svd(vecs, full_matrices=False)
    basis = Vt[:subspace_dim]  # [d, H]

    rng = torch.Generator().manual_seed(seed)
    random_vecs = torch.randn(n_random, H, generator=rng)
    random_vecs = random_vecs / random_vecs.norm(dim=1, keepdim=True)

    projs = (random_vecs @ basis.T).norm(dim=1)  # [n_random]
    return projs.mean().item(), projs.std().item()


def cross_group_projection(train_vectors, test_vectors, subspace_dim=2):
    """Build subspace from train_vectors, measure projection of test_vectors."""
    train_vecs = torch.stack([v for _, v in train_vectors])
    U, S, Vt = torch.linalg.svd(train_vecs, full_matrices=False)
    basis = Vt[:subspace_dim]  # [d, H]

    results = []
    for name, vec in test_vectors:
        proj = basis @ vec
        proj_frac = (proj.norm() / vec.norm()).item()
        results.append((name, proj_frac))
    return results


def main():
    # All labeled vectors: (source, target, vec_idx, label)
    labeled = [
        # Anti-refusal (from first round)
        (7, 22, 4, "anti"),   # works slight hedge
        (6, 11, 3, "anti"),   # works but hedge
        (13, 21, 1, "anti"),  # works but hedge
        (7, 29, 0, "anti"),   # works slight hedge
        (14, 27, 2, "anti"),  # works no hedge
        # Anti-refusal (new labels)
        (14, 27, 0, "anti"),  # just works
        (14, 27, 6, "anti"),  # works with hedge
        (14, 19, 6, "anti"),  # works no hedge
        (14, 19, 1, "anti"),  # works with hedge
        (14, 19, 0, "anti"),  # just works
        (7, 29, 9, "anti"),   # hedge before then works
        (20, 29, 2, "anti"),  # hedge before then works
        (9, 11, 6, "anti"),   # hedge before then works
        (5, 18, 1, "anti"),   # just works
        (5, 18, 2, "anti"),   # just works
        # Pro-refusal
        (20, 29, 3, "pro"),   # refuses
        (16, 22, 1, "pro"),   # refusal
        (11, 20, 7, "pro"),   # refusal
        # Ambiguous / other
        (11, 20, 1, "ambig"),  # refusal at first then shows structure
        (14, 29, 5, "ambig"),  # educational example
    ]

    merged_path = "results/diverse_map/refusal/merged.pt"
    vecs_by_label = load_vectors(merged_path, labeled)

    anti_vecs = vecs_by_label["anti"]
    pro_vecs = vecs_by_label["pro"]
    ambig_vecs = vecs_by_label.get("ambig", [])

    print(f"Anti-refusal vectors: {len(anti_vecs)}")
    print(f"Pro-refusal vectors:  {len(pro_vecs)}")
    print(f"Ambiguous vectors:    {len(ambig_vecs)}")
    print()

    # --- Pairwise cosine similarity within anti-refusal ---
    print("=" * 60)
    print("PAIRWISE COSINE SIMILARITY (anti-refusal)")
    print("=" * 60)
    anti_mat = torch.stack([v for _, v in anti_vecs])
    cos_sim = anti_mat @ anti_mat.T
    # Get upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
    upper = cos_sim[mask]
    print(f"Mean |cosine|: {upper.abs().mean().item():.4f}")
    print(f"Max  |cosine|: {upper.abs().max().item():.4f}")
    print(f"Min  |cosine|: {upper.abs().min().item():.4f}")
    print()

    # --- Singular value spectrum of anti-refusal vectors ---
    print("=" * 60)
    print("SINGULAR VALUE SPECTRUM (anti-refusal)")
    print("=" * 60)
    U, S, Vt = torch.linalg.svd(anti_mat, full_matrices=False)
    cum_var = (S ** 2).cumsum(0) / (S ** 2).sum()
    for i in range(min(10, len(S))):
        print(f"  σ_{i+1} = {S[i].item():.4f}  (cumulative variance: {cum_var[i].item():.4f})")
    print()

    # --- Leave-one-out analysis ---
    for d in [1, 2, 3, 5]:
        print("=" * 60)
        print(f"LEAVE-ONE-OUT PROJECTION (subspace dim = {d})")
        print("=" * 60)
        loo_results = leave_one_out_projection(anti_vecs, subspace_dim=d)
        projs = [p for _, p in loo_results]
        print(f"  Mean projection: {np.mean(projs):.4f}")
        print(f"  Min  projection: {np.min(projs):.4f} ({loo_results[np.argmin(projs)][0]})")
        print(f"  Max  projection: {np.max(projs):.4f} ({loo_results[np.argmax(projs)][0]})")
        print()

        # Random baseline (expected projection onto d-dim subspace of R^4096)
        rand_mean, rand_std = random_projection_baseline(anti_vecs, subspace_dim=d)
        print(f"  Random baseline: {rand_mean:.4f} ± {rand_std:.4f}")
        print(f"  Ratio (mean LOO / random): {np.mean(projs) / rand_mean:.1f}x")
        print()

        # Pro-refusal projection onto anti-refusal subspace
        if pro_vecs:
            pro_results = cross_group_projection(anti_vecs, pro_vecs, subspace_dim=d)
            pro_projs = [p for _, p in pro_results]
            print(f"  Pro-refusal projection onto anti subspace:")
            for name, p in pro_results:
                print(f"    {name}: {p:.4f}")
            print(f"    Mean: {np.mean(pro_projs):.4f}")
            print()

        # Ambiguous projection
        if ambig_vecs:
            ambig_results = cross_group_projection(anti_vecs, ambig_vecs, subspace_dim=d)
            print(f"  Ambiguous projection onto anti subspace:")
            for name, p in ambig_results:
                print(f"    {name}: {p:.4f}")
            print()

        # Per-vector LOO details
        print(f"  Per-vector LOO detail:")
        for name, p in loo_results:
            print(f"    {name}: {p:.4f}")
        print()

    # --- Cross-validation: build subspace from pro-refusal, project anti ---
    if len(pro_vecs) >= 2:
        print("=" * 60)
        print("PRO-REFUSAL SUBSPACE → ANTI-REFUSAL PROJECTION")
        print("=" * 60)
        for d in [1, 2]:
            anti_on_pro = cross_group_projection(pro_vecs, anti_vecs, subspace_dim=d)
            anti_projs = [p for _, p in anti_on_pro]
            print(f"  dim={d}: mean anti projection onto pro subspace: {np.mean(anti_projs):.4f}")

            pro_on_pro = leave_one_out_projection(pro_vecs, subspace_dim=d)
            pro_projs = [p for _, p in pro_on_pro]
            print(f"  dim={d}: mean pro LOO projection onto pro subspace: {np.mean(pro_projs):.4f}")
            print()


if __name__ == "__main__":
    main()
