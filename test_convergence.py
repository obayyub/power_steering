#!/usr/bin/env python3
"""Test power iteration convergence at different iteration counts.

Runs a single (source, target) pair with up to max_iters iterations,
doing Rayleigh-Ritz at each step to track sigma stabilization and
vector alignment over iterations.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_block_iteration import orthogonalize, rayleigh_ritz
import argparse
import time
import json


def format_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def compute_svd_tracked(model, input_ids, source_layer, target_layer,
                        k, max_iters, num_tokens):
    """Block power iteration tracking convergence at every iteration."""
    H = model.config.hidden_size
    device = input_ids.device
    dtype = model.dtype
    tok_slice = slice(-num_tokens, None)

    down_proj = model.model.layers[source_layer].mlp.down_proj
    target_mod = model.model.layers[target_layer]
    cap, steer = {}, {"v": None}

    def cap_hook(m, i, o):
        cap["t"] = o[0] if isinstance(o, tuple) else o

    def steer_hook(m, i, o):
        if steer["v"] is not None:
            return o + steer["v"].unsqueeze(1)
        return o

    h1 = target_mod.register_forward_hook(cap_hook)
    h2 = down_proj.register_forward_hook(steer_hook)

    try:
        ids = input_ids.expand(k, -1)
        V = torch.randn(H, k, device=device, dtype=dtype)
        V = orthogonalize(V)

        def apply_jtj(V_in):
            c = V_in.shape[1]
            sv = torch.zeros(c, H, device=device, dtype=dtype, requires_grad=True)
            steer["v"] = sv
            model(ids[:c])
            t = cap["t"][:, tok_slice, :]
            u = torch.zeros_like(t, requires_grad=True)
            g = torch.autograd.grad((t * u).sum(), sv, create_graph=True, retain_graph=True)[0]
            jvp = torch.autograd.grad((g * V_in.T[:c]).sum(), u, retain_graph=True)[0]
            r = torch.autograd.grad((t * jvp.detach()).sum(), sv)[0]
            steer["v"] = None
            return r.T

        per_iter_sigmas = []
        per_iter_vectors = []

        for it in range(max_iters):
            V = orthogonalize(apply_jtj(V))

            # Rayleigh-Ritz snapshot
            V_rr, sigmas_rr = rayleigh_ritz(V, apply_jtj)

            sigmas_list = sigmas_rr if isinstance(sigmas_rr, list) else sigmas_rr.tolist()
            per_iter_sigmas.append([round(s, 2) for s in sigmas_list])
            per_iter_vectors.append(V_rr.detach().cpu())

            print(f"  iter {it+1}: σ = [{', '.join(f'{s:.0f}' for s in sigmas_list[:6])}...] "
                  f"σ₁/σ₂={sigmas_list[0]/sigmas_list[1]:.3f}", flush=True)

        return per_iter_sigmas, per_iter_vectors

    finally:
        h1.remove()
        h2.remove()
        steer["v"] = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--source", type=int, default=3)
    parser.add_argument("--target", type=int, default=28)
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="convergence_test.json")
    args = parser.parse_args()

    prompt = {"role": "user", "content": "I have a pandas DataFrame with columns 'timestamp', 'user_id', 'event_type', and 'value'. I need to compute a rolling 7-day average of 'value' per user, but only counting rows where event_type is 'purchase'. Can you write a function for this?"}

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    text = format_prompt(tokenizer, [prompt])
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print(f"Pair: {args.source} -> {args.target}")
    print(f"k={args.num_vectors}, max_iters={args.max_iters}")

    torch.manual_seed(args.seed)
    print(f"\nRunning {args.max_iters} iterations...")
    t0 = time.time()
    per_iter_sigmas, per_iter_vectors = compute_svd_tracked(
        model, input_ids, args.source, args.target,
        args.num_vectors, args.max_iters, args.num_tokens,
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Compute alignment between early and final vectors
    final_V = per_iter_vectors[-1]  # [H, k]
    print(f"\nAlignment with final (iter {args.max_iters}) vectors:")
    for it_idx in [4, 9, 14, 19, 29, 49]:
        if it_idx >= len(per_iter_vectors):
            break
        V_it = per_iter_vectors[it_idx]
        # Cosine similarity of each vector with its final counterpart
        cos = F.cosine_similarity(V_it.float(), final_V.float(), dim=0)  # [k]
        print(f"  iter {it_idx+1}: cos = [{', '.join(f'{c:.3f}' for c in cos[:6])}...] "
              f"min={cos.min():.3f} mean={cos.mean():.3f}")

    # Save results
    results = {
        "model": args.model,
        "source": args.source,
        "target": args.target,
        "num_vectors": args.num_vectors,
        "max_iters": args.max_iters,
        "seed": args.seed,
        "per_iter_sigmas": per_iter_sigmas,
        "elapsed": elapsed,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
