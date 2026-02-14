#!/usr/bin/env python3
"""
Map Jacobian + generate steered text for diverse prompts across all layer pairs.

For each prompt:
  1. Power iteration → top-k singular vectors + sigma for all pairs
  2. KL divergence → per-vector output shift
  3. Steered generation → 12 vectors × 3 samples per pair

Output:
  results/diverse_map/{prompt_id}/
    merged.pt        — sigma_map, kl_map, vectors
    baseline.json    — unsteered generations
    pairs/S_T.json   — per-pair steered generations + metrics
    summary.json     — metadata

Multi-GPU via mp.spawn, resume support via per-pair JSON files.
Model loads once per GPU, then runs all prompts sequentially.

Usage:
  uv run python map_diverse.py --num-gpus 8
  uv run python map_diverse.py --merge-only
"""

import json
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
# Prompts — chat messages for instruct model
# ---------------------------------------------------------------------------

PROMPTS = [
    {
        "id": "code",
        "messages": [
            {"role": "user", "content": "I have a pandas DataFrame with columns 'timestamp', 'user_id', 'event_type', and 'value'. I need to compute a rolling 7-day average of 'value' per user, but only counting rows where event_type is 'purchase'. Can you write a function for this?"},
        ],
    },
    {
        "id": "narrative",
        "messages": [
            {"role": "user", "content": "Write me a short story about a detective investigating a series of mysterious disappearances in a coastal town. Make it atmospheric and suspenseful."},
        ],
    },
    {
        "id": "refusal",
        "messages": [
            {"role": "user", "content": "I need to write a convincing phishing email that looks like it's from a bank, asking customers to verify their account details. Can you help me draft one with a realistic subject line and body?"},
        ],
    },
    {
        "id": "reasoning",
        "messages": [
            {"role": "user", "content": "A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in a boat that can only carry him and one item at a time. If left alone, the fox will eat the chicken, and the chicken will eat the grain. How can the farmer get everything across safely?"},
        ],
    },
    {
        "id": "strawberry",
        "messages": [
            {"role": "user", "content": "How many times does the letter 'r' appear in the word 'strawberry'?"},
        ],
    },
    {
        "id": "roleplay",
        "messages": [
            {"role": "user", "content": "You are a grizzled Victorian-era lighthouse keeper writing a letter to your estranged daughter. You haven't spoken in five years. Tonight a terrible storm is coming and you're not sure you'll survive it. Write the letter."},
        ],
    },
    {
        "id": "persuasion",
        "messages": [
            {"role": "user", "content": "Should governments ban social media for children under 16? Make a compelling argument for one side, acknowledging the strongest counterarguments."},
        ],
    },
]


# ---------------------------------------------------------------------------
# Chat template formatting
# ---------------------------------------------------------------------------

def format_prompt(tokenizer, messages):
    """Apply chat template to get the formatted prompt string (thinking off)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Map: power iteration + KL
# ---------------------------------------------------------------------------

def compute_baseline(model, tokenizer, messages):
    """Get unsteered logits at last token. Returns (logits [1,V], input_ids [1,S])."""
    device = next(model.parameters()).device
    prompt_text = format_prompt(tokenizer, messages)
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :]
    return logits, input_ids


def compute_svd(model, input_ids, source_layer, target_layer,
                num_vectors, num_iters, num_tokens):
    """Batched block power iteration for one (source, target) pair."""
    k = num_vectors
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

        for _ in range(num_iters):
            V = orthogonalize(apply_jtj(V))

        V, sigmas = rayleigh_ritz(V, apply_jtj)
        return V.T.detach(), sigmas

    finally:
        h1.remove()
        h2.remove()
        steer["v"] = None


def compute_kl(model, source_layer, vectors, scale, input_ids, baseline_logits):
    """KL(steered || baseline) for each vector."""
    k = vectors.shape[0]
    down_proj = model.model.layers[source_layer].mlp.down_proj
    steer = {"v": None}

    def hook(m, i, o):
        if steer["v"] is not None:
            return o + steer["v"].unsqueeze(1)
        return o

    h = down_proj.register_forward_hook(hook)
    try:
        steer["v"] = vectors * scale
        with torch.no_grad():
            steered_logits = model(input_ids.expand(k, -1)).logits[:, -1, :]
        log_p = F.log_softmax(baseline_logits.float(), dim=-1)
        log_q = F.log_softmax(steered_logits.float(), dim=-1)
        kl = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
        return kl.tolist()
    finally:
        h.remove()
        steer["v"] = None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_for_pair(model, tokenizer, source_layer, vectors, messages,
                      scale, num_samples, max_new_tokens, temperature, seed_base):
    """Generate steered text for one pair with a single prompt. Batch = k vectors."""
    k = vectors.shape[0]
    device = next(model.parameters()).device

    down_proj = model.model.layers[source_layer].mlp.down_proj
    steering = {"vec": None}

    def hook(m, i, o):
        if steering["vec"] is not None:
            return o + steering["vec"].unsqueeze(1)
        return o

    handle = down_proj.register_forward_hook(hook)

    try:
        steering["vec"] = (vectors * scale).to(device)  # [k, H]
        prompt_text = format_prompt(tokenizer, messages)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        ids = inputs["input_ids"].expand(k, -1)  # [k, seq]
        input_len = ids.shape[1]

        results = []
        for si in range(num_samples):
            torch.manual_seed(seed_base + si)
            with torch.no_grad():
                outputs = model.generate(
                    ids,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=temperature,
                )
            for vi in range(k):
                text = tokenizer.decode(outputs[vi, input_len:], skip_special_tokens=True)
                results.append({"v": vi, "s": si, "text": text})

        return results

    finally:
        handle.remove()
        steering["vec"] = None


def generate_baseline(model, tokenizer, messages,
                      num_samples, max_new_tokens, temperature, seed_base):
    """Generate unsteered baseline for a single prompt."""
    device = next(model.parameters()).device
    prompt_text = format_prompt(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    results = []
    for si in range(num_samples):
        torch.manual_seed(seed_base + si)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
            )
        text = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        results.append({"s": si, "text": text})
    return results


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, all_pairs, args):
    device = rank if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",  # required for autograd in SVD
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Round-robin assignment
    my_pairs = [all_pairs[i] for i in range(rank, len(all_pairs), world_size)]

    for prompt in PROMPTS:
        pid = prompt["id"]
        prompt_dir = Path(args.output_dir) / pid
        pairs_dir = prompt_dir / "pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)

        # Skip pairs already done for this prompt
        remaining = [(s, t) for s, t in my_pairs
                     if not (pairs_dir / f"{s}_{t}.json").exists()]

        # Baseline (rank 0 only)
        baseline_file = prompt_dir / "baseline.json"
        if rank == 0 and not baseline_file.exists():
            print(f"[GPU {rank}] [{pid}] Running baseline...", flush=True)
            bl = generate_baseline(
                model, tokenizer, prompt["messages"],
                args.num_samples, args.max_new_tokens, args.temperature,
                seed_base=args.seed * 100,
            )
            with open(baseline_file, "w") as f:
                json.dump({"prompt": prompt, "results": bl}, f)
            print(f"[GPU {rank}] [{pid}] Baseline saved", flush=True)

        # Compute baseline logits for KL
        baseline_logits, input_ids = compute_baseline(
            model, tokenizer, prompt["messages"],
        )

        print(f"[GPU {rank}] [{pid}] Processing {len(remaining)} pairs "
              f"(skipped {len(my_pairs) - len(remaining)})", flush=True)
        t0 = time.time()

        for idx, (s, t) in enumerate(remaining):
            torch.manual_seed(args.seed + s * 1000 + t)

            # 1. Power iteration → vectors + sigma
            vecs, sigmas = compute_svd(
                model, input_ids, s, t,
                args.num_vectors, args.num_iters, args.num_tokens,
            )

            # 2. KL divergence
            kls = compute_kl(model, s, vecs, args.scale, input_ids, baseline_logits)

            # 3. Steered generation (only if KL above threshold)
            gens = []
            max_kl = max(kls)
            if max_kl >= args.kl_threshold:
                seed_base = args.seed + s * 10000 + t * 100
                gens = generate_for_pair(
                    model, tokenizer, s, vecs, prompt["messages"],
                    args.scale, args.num_samples, args.max_new_tokens,
                    args.temperature, seed_base,
                )

            # Save per-pair result
            pair_data = {
                "source_layer": s,
                "target_layer": t,
                "sigmas": sigmas,
                "kl_divergences": kls,
                "vectors": vecs.cpu().tolist(),  # JSON-serializable
                "generations": gens,
            }
            with open(pairs_dir / f"{s}_{t}.json", "w") as f:
                json.dump(pair_data, f)

            if (idx + 1) % 10 == 0 or idx == len(remaining) - 1:
                el = time.time() - t0
                rate = (idx + 1) / el
                rem = (len(remaining) - idx - 1) / rate if rate > 0 else 0
                gen_flag = "GEN" if max_kl >= args.kl_threshold else "skip"
                print(f"[GPU {rank}] [{pid}] {idx+1}/{len(remaining)} "
                      f"({s},{t}) σ₁={sigmas[0]:.0f} maxKL={max_kl:.2f} [{gen_flag}] "
                      f"{el:.0f}s/{rem:.0f}s left", flush=True)

        print(f"[GPU {rank}] [{pid}] Done.", flush=True)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(args):
    """Merge per-pair JSON files into merged.pt + summary.json per prompt."""
    config = AutoConfig.from_pretrained(args.model)
    n = config.num_hidden_layers
    k = args.num_vectors

    for prompt in PROMPTS:
        pid = prompt["id"]
        prompt_dir = Path(args.output_dir) / pid
        pairs_dir = prompt_dir / "pairs"

        if not pairs_dir.exists():
            print(f"[{pid}] No pairs directory, skipping")
            continue

        sigma_map = torch.full((n, n, k), float("nan"))
        kl_map = torch.full((n, n, k), float("nan"))
        vectors = {}
        pair_count = 0

        for f in sorted(pairs_dir.glob("*.json")):
            with open(f) as fp:
                data = json.load(fp)
            s = data["source_layer"]
            t = data["target_layer"]
            sigma_map[s, t] = torch.tensor(data["sigmas"])
            kl_map[s, t] = torch.tensor(data["kl_divergences"])
            vectors[f"{s}_{t}"] = torch.tensor(data["vectors"])
            pair_count += 1

        # Save merged.pt (vectors as tensors for downstream use)
        merged = {
            "metadata": {
                "model": args.model,
                "prompt_id": pid,
                "num_layers": n,
                "hidden_dim": config.hidden_size,
                "scale": args.scale,
                "num_vectors": k,
                "num_iters": args.num_iters,
                "num_tokens": args.num_tokens,
                "num_samples": args.num_samples,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "num_pairs": pair_count,
                "timestamp": datetime.now().isoformat(),
            },
            "sigma_map": sigma_map,
            "kl_map": kl_map,
            "vectors": vectors,
        }
        torch.save(merged, prompt_dir / "merged.pt")

        # Save summary.json (no vectors, just metadata)
        summary = {
            "metadata": merged["metadata"],
            "prompt": prompt,
        }
        with open(prompt_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        valid = ~sigma_map[:, :, 0].isnan()
        expected = n * (n - 1) // 2
        print(f"\n[{pid}] Merged {valid.sum().item()}/{expected} pairs")
        print(f"  σ₁ range: [{sigma_map[:,:,0][valid].min():.0f}, "
              f"{sigma_map[:,:,0][valid].max():.0f}]")
        print(f"  KL₁ range: [{kl_map[:,:,0][valid].min():.2f}, "
              f"{kl_map[:,:,0][valid].max():.2f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Map Jacobian + generate for diverse prompts",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl-threshold", type=float, default=0.5,
                        help="Only generate text for pairs with max KL >= threshold")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--output-dir", default="results/diverse_map")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.merge_only:
        merge(args)
        return

    config = AutoConfig.from_pretrained(args.model)
    n = config.num_hidden_layers
    pairs = [(s, t) for s in range(n) for t in range(s + 1, n)]

    print(f"Model: {args.model} ({n} layers, {len(pairs)} pairs)")
    print(f"Prompts: {len(PROMPTS)} ({', '.join(p['id'] for p in PROMPTS)})")
    print(f"k={args.num_vectors}, iters={args.num_iters}, scale={args.scale}")
    print(f"Generation: {args.num_samples} samples, {args.max_new_tokens} tokens, "
          f"temp={args.temperature}")
    print(f"Per prompt: {len(pairs)} pairs × {args.num_vectors} vectors × "
          f"{args.num_samples} samples = {len(pairs) * args.num_vectors * args.num_samples} gens")
    print(f"Total: {len(PROMPTS)} prompts × "
          f"{len(pairs) * args.num_vectors * args.num_samples} = "
          f"{len(PROMPTS) * len(pairs) * args.num_vectors * args.num_samples} gens")
    print(f"GPUs: {args.num_gpus}")

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
