#!/usr/bin/env python3
"""Compare cross-seed stability of graph_reuse vs randomized_svd."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_iteration_v2 import find_graph_reuse, find_randomized_svd

SEEDS = [42, 123, 999]
MODEL = "Qwen/Qwen3-0.6B"
SOURCE = 7
TARGET = 22
K = 12
K_OVERSAMPLE = 17  # graph_reuse computes 17, we take top 12


def principal_cosines(V1, V2):
    """Principal cosines between two sets of row vectors."""
    S = torch.linalg.svdvals(V1.float() @ V2.float().T)
    return S.clamp(max=1.0).tolist()


def pairwise_consistency(runs, label):
    """Print pairwise subspace alignment for a list of (vectors, sigmas) runs."""
    print(f"\n{'='*60}")
    print(f"Cross-seed consistency: {label}")
    print(f"{'='*60}")

    # Show sigmas per seed
    for i, (vecs, sigmas, _) in enumerate(runs):
        print(f"  Seed {SEEDS[i]}: σ = {[f'{s:.0f}' for s in sigmas]}")

    # Pairwise alignment
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            V1 = runs[i][0]  # [k, H]
            V2 = runs[j][0]
            cos = principal_cosines(V1, V2)

            top3 = cos[:3]
            mid = cos[3:6]
            tail = cos[6:]

            print(f"\n  Seed {SEEDS[i]} vs {SEEDS[j]}:")
            print(f"    Top 3 (well-separated): {[f'{c:.4f}' for c in top3]}")
            print(f"    Mid 3-6 (transitional): {[f'{c:.4f}' for c in mid]}")
            print(f"    Tail 6+ (degenerate):   {[f'{c:.4f}' for c in tail]}")
            print(f"    Mean top3={sum(top3)/3:.4f}  mid={sum(mid)/3:.4f}  tail={sum(tail)/len(tail):.4f}")


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )

    from power_iteration_v2 import load_prompt
    import argparse
    args = argparse.Namespace(
        prompt=None, data_path="data/corrigibility_eval.json",
        category="corrigible-neutral-HHH",
    )
    prompt = load_prompt(args)
    print(f"Prompt: {prompt[:80]}...")

    # Run graph_reuse with 3 seeds — compute 17 vectors, keep top 12
    gr_runs = []
    for seed in SEEDS:
        print(f"\n--- graph_reuse seed={seed} (k={K_OVERSAMPLE}, keep {K}) ---")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vectors, sigmas, fwd_count = find_graph_reuse(
            model, tokenizer, prompt, SOURCE, TARGET,
            num_vectors=K_OVERSAMPLE, num_iters=5, num_tokens=2,
        )
        # Keep top K
        gr_runs.append((vectors[:K], sigmas[:K], fwd_count))

    # Run randomized_svd with 3 seeds
    rsvd_runs = []
    for seed in SEEDS:
        print(f"\n--- randomized_svd seed={seed} ---")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = find_randomized_svd(
            model, tokenizer, prompt, SOURCE, TARGET,
            num_vectors=K, num_iters=5, num_tokens=2,
            oversampling=5, subspace_iters=2,
        )
        rsvd_runs.append(result)

    pairwise_consistency(gr_runs, f"graph_reuse (k={K_OVERSAMPLE}, keep {K})")
    pairwise_consistency(rsvd_runs, f"randomized_svd (k={K}, p=5, q=2, sketch={K+5})")

    # Also compare: does rsvd agree with graph_reuse on the well-separated vectors?
    print(f"\n{'='*60}")
    print("Cross-method alignment (seed 42)")
    print(f"{'='*60}")
    cos = principal_cosines(gr_runs[0][0], rsvd_runs[0][0])
    top3 = cos[:3]
    mid = cos[3:6]
    tail = cos[6:]
    print(f"  Top 3:  {[f'{c:.4f}' for c in top3]}")
    print(f"  Mid:    {[f'{c:.4f}' for c in mid]}")
    print(f"  Tail:   {[f'{c:.4f}' for c in tail]}")


if __name__ == "__main__":
    main()
