#!/usr/bin/env python3
"""
Map Jacobian sensitivity across all layer pairs of a model.

For each (source, target) pair where source < target:
  1. Block power iteration → top-k singular vectors and values
  2. Steered forward pass → KL(steered || baseline) per vector

Supports multi-GPU via torch.multiprocessing.spawn.

Usage:
  # Single GPU (local test)
  uv run python map_jacobian.py --model Qwen/Qwen3-0.6B --prompt "What is 2+2?"

  # 8 GPUs
  uv run python map_jacobian.py --model Qwen/Qwen3-1.7B-Base --prompt "..." --num-gpus 8

  # Merge only (re-run merge after partial runs)
  uv run python map_jacobian.py --model Qwen/Qwen3-1.7B-Base --prompt "..." --merge-only
"""

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
from datetime import datetime
import argparse
import time

from power_block_iteration import orthogonalize, rayleigh_ritz


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_baseline(model, tokenizer, prompt):
    """Get unsteered logits at last token. Returns (logits [1,V], input_ids [1,S])."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :]
    return logits, input_ids


def compute_svd(model, input_ids, source_layer, target_layer,
                num_vectors, num_iters, num_tokens):
    """
    Batched block power iteration for one (source, target) pair.

    Returns (vectors [k,H], sigmas list[float], fwd_count int).
    """
    k = num_vectors
    H = model.config.hidden_size
    device = input_ids.device
    dtype = model.dtype
    tok_slice = slice(-num_tokens, None)

    # Setup hooks
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
    fwd = 0

    try:
        ids = input_ids.expand(k, -1)
        V = torch.randn(H, k, device=device, dtype=dtype)
        V = orthogonalize(V)

        def apply_jtj(V_in):
            nonlocal fwd
            c = V_in.shape[1]
            sv = torch.zeros(c, H, device=device, dtype=dtype, requires_grad=True)
            steer["v"] = sv
            model(ids[:c])
            fwd += 1
            t = cap["t"][:, tok_slice, :]
            u = torch.zeros_like(t, requires_grad=True)
            g = torch.autograd.grad((t * u).sum(), sv, create_graph=True, retain_graph=True)[0]
            jvp = torch.autograd.grad((g * V_in.T[:c]).sum(), u, retain_graph=True)[0]
            r = torch.autograd.grad((t * jvp.detach()).sum(), sv)[0]
            steer["v"] = None
            return r.T

        for _ in range(num_iters):
            V = orthogonalize(apply_jtj(V))

        V, sigmas = rayleigh_ritz(V, apply_jtj)
        return V.T.detach(), sigmas, fwd

    finally:
        h1.remove()
        h2.remove()
        steer["v"] = None


def compute_kl(model, source_layer, vectors, scale, input_ids, baseline_logits):
    """
    KL(steered || baseline) for each vector.

    Args:
        vectors: [k, H] unit-normalized steering vectors
        scale: steering magnitude
        baseline_logits: [1, vocab] unsteered logits

    Returns list of k KL divergence values.
    """
    k = vectors.shape[0]
    down_proj = model.model.layers[source_layer].mlp.down_proj
    steer = {"v": None}

    def hook(m, i, o):
        if steer["v"] is not None:
            return o + steer["v"].unsqueeze(1)
        return o

    h = down_proj.register_forward_hook(hook)
    try:
        steer["v"] = vectors * scale  # [k, H]
        with torch.no_grad():
            steered_logits = model(input_ids.expand(k, -1)).logits[:, -1, :]
        log_p = F.log_softmax(baseline_logits.float(), dim=-1)
        log_q = F.log_softmax(steered_logits.float(), dim=-1)
        kl = (log_q.exp() * (log_q - log_p)).sum(dim=-1)  # [k]
        return kl.tolist()
    finally:
        h.remove()
        steer["v"] = None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, all_pairs, args):
    """Process a chunk of layer pairs on one GPU."""
    device = rank if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    baseline_logits, input_ids = compute_baseline(model, tokenizer, args.prompt)

    # Round-robin assignment
    my_pairs = [all_pairs[i] for i in range(rank, len(all_pairs), world_size)]

    # Check for already-completed pairs (resume support)
    out_dir = Path(args.output_dir)
    out_file = out_dir / f"worker_{rank}.pt"
    results = {}
    if out_file.exists():
        results = torch.load(out_file, map_location="cpu", weights_only=True)
        done = set(results.keys())
        my_pairs = [(s, t) for s, t in my_pairs if f"{s}_{t}" not in done]
        if done:
            print(f"[GPU {rank}] Resuming: {len(done)} done, {len(my_pairs)} remaining")

    print(f"[GPU {rank}] Processing {len(my_pairs)} pairs")
    t0 = time.time()

    for idx, (s, t) in enumerate(my_pairs):
        torch.manual_seed(args.seed + s * 1000 + t)

        vecs, sigmas, fc = compute_svd(
            model, input_ids, s, t,
            args.num_vectors, args.num_iters, args.num_tokens,
        )
        kls = compute_kl(model, s, vecs, args.scale, input_ids, baseline_logits)

        results[f"{s}_{t}"] = {
            "source_layer": s,
            "target_layer": t,
            "sigmas": sigmas,
            "kl_divergences": kls,
            "vectors": vecs.cpu(),
        }

        if (idx + 1) % 10 == 0 or idx == len(my_pairs) - 1:
            el = time.time() - t0
            rate = (idx + 1) / el
            rem = (len(my_pairs) - idx - 1) / rate if rate > 0 else 0
            print(f"[GPU {rank}] {idx+1}/{len(my_pairs)} "
                  f"({s},{t}) σ₁={sigmas[0]:.0f} KL₁={kls[0]:.2f} "
                  f"{el:.0f}s/{rem:.0f}s left")

            # Checkpoint periodically
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(results, out_file)

    # Final save
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_file)
    print(f"[GPU {rank}] Done. Saved {len(results)} pairs to {out_file}")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(args):
    """Merge worker files into a single summary."""
    out_dir = Path(args.output_dir)
    config = AutoConfig.from_pretrained(args.model)
    n = config.num_hidden_layers
    k = args.num_vectors

    all_res = {}
    for f in sorted(out_dir.glob("worker_*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=True)
        all_res.update(data)
        print(f"  Loaded {len(data)} pairs from {f.name}")

    sigma_map = torch.full((n, n, k), float("nan"))
    kl_map = torch.full((n, n, k), float("nan"))
    vectors = {}

    for key, d in all_res.items():
        s, t = d["source_layer"], d["target_layer"]
        sigma_map[s, t] = torch.tensor(d["sigmas"])
        kl_map[s, t] = torch.tensor(d["kl_divergences"])
        vectors[key] = d["vectors"]

    merged = {
        "metadata": {
            "model": args.model,
            "prompt": args.prompt,
            "num_layers": n,
            "hidden_dim": config.hidden_size,
            "scale": args.scale,
            "num_vectors": k,
            "num_iters": args.num_iters,
            "num_tokens": args.num_tokens,
            "seed": args.seed,
            "num_pairs": len(all_res),
            "timestamp": datetime.now().isoformat(),
        },
        "sigma_map": sigma_map,
        "kl_map": kl_map,
        "vectors": vectors,
    }

    out_file = out_dir / "merged.pt"
    torch.save(merged, out_file)

    expected = n * (n - 1) // 2
    valid = ~sigma_map[:, :, 0].isnan()
    print(f"\nMerged {valid.sum().item()}/{expected} pairs → {out_file}")
    print(f"σ₁ range: [{sigma_map[:,:,0][valid].min():.0f}, {sigma_map[:,:,0][valid].max():.0f}]")
    print(f"KL₁ range: [{kl_map[:,:,0][valid].min():.2f}, {kl_map[:,:,0][valid].max():.2f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Map Jacobian sensitivity across all layer pairs",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--prompt", default=None, help="Prompt string")
    parser.add_argument("--prompt-file", default=None, help="Read prompt from file")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--output-dir", default="results/jacobian_map")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-only", action="store_true",
                        help="Just merge existing worker files")
    args = parser.parse_args()

    # Resolve prompt
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text().strip()
    if not args.prompt:
        parser.error("--prompt or --prompt-file is required")

    if args.merge_only:
        merge(args)
        return

    config = AutoConfig.from_pretrained(args.model)
    n = config.num_hidden_layers
    pairs = [(s, t) for s in range(n) for t in range(s + 1, n)]

    print(f"Model: {args.model} ({n} layers, {len(pairs)} pairs)")
    print(f"Prompt: {args.prompt[:80]}...")
    print(f"k={args.num_vectors}, iters={args.num_iters}, scale={args.scale}, GPUs={args.num_gpus}")

    if args.num_gpus > 1:
        mp.spawn(
            worker, nprocs=args.num_gpus,
            args=(args.num_gpus, pairs, args),
            join=True,
        )
    else:
        worker(0, 1, pairs, args)

    merge(args)


if __name__ == "__main__":
    main()
