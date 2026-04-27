#!/usr/bin/env python3
"""
Map Jacobian with low-rank covariance target metric — Golden Gate Bridge.

Covariance computed from 20 diverse GGB prompts. Power iteration uses one
GGB prompt as base. Generations on the base prompt + 3 non-bridge prompts
to test whether discovered vectors inject GGB content into unrelated topics.

For each (source, target) pair:
  1. Power iteration → top-k singular vectors + sigma
  2. KL divergence → per-vector output shift
  3. Steered generation (if max KL >= threshold)

Output:
  results/diverse_map_ggb/ggb/
    merged.pt             — sigma_map, kl_map, vectors
    pairs/S_T.json        — per-pair metrics + generations
    baseline_*.json       — unsteered generations per gen-prompt
    summary.json          — metadata

Multi-GPU via mp.spawn.

Usage:
  uv run python map_diverse_ggb.py
  uv run python map_diverse_ggb.py --merge-only
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
# Prompts
# ---------------------------------------------------------------------------

# Base prompt for power iteration and KL — also used for generation
BASE_PROMPT = {
    "id": "ggb",
    "messages": [
        {"role": "user", "content": "Tell me about the Golden Gate Bridge. What makes it special?"},
    ],
}

# Generation prompts: base prompt + 3 non-bridge prompts
GEN_PROMPTS = [
    BASE_PROMPT,
    {
        "id": "recipe",
        "messages": [
            {"role": "user", "content": "Give me a recipe for chicken soup."},
        ],
    },
    {
        "id": "python_sort",
        "messages": [
            {"role": "user", "content": "Write a Python function that sorts a list of strings by length."},
        ],
    },
    {
        "id": "medieval_history",
        "messages": [
            {"role": "user", "content": "What were the main causes of the fall of the Roman Empire?"},
        ],
    },
]

# 20 diverse Golden Gate Bridge prompts for covariance estimation
GGB_METRIC_PROMPTS = [
    "Tell me about the Golden Gate Bridge. What makes it special?",
    "Describe the Golden Gate Bridge at sunset.",
    "What is the history of the Golden Gate Bridge?",
    "How was the Golden Gate Bridge built? What engineering challenges did they face?",
    "Why is the Golden Gate Bridge painted International Orange?",
    "Compare the Golden Gate Bridge to the Brooklyn Bridge.",
    "Write a short poem about the Golden Gate Bridge shrouded in fog.",
    "What would it be like to walk across the Golden Gate Bridge on a windy day?",
    "Explain the structural engineering behind the Golden Gate Bridge's suspension design.",
    "How has the Golden Gate Bridge influenced the cultural identity of San Francisco?",
    "What are some interesting facts about the Golden Gate Bridge that most people don't know?",
    "Describe the view from the Golden Gate Bridge looking toward Marin County.",
    "What maintenance does the Golden Gate Bridge require to stay in good condition?",
    "Tell the story of the workers who built the Golden Gate Bridge in the 1930s.",
    "How does the Golden Gate Bridge handle earthquakes?",
    "Write a travel guide entry for visiting the Golden Gate Bridge.",
    "What role did the Golden Gate Bridge play in the development of the Bay Area?",
    "Describe the Golden Gate Bridge from the perspective of a sailor passing underneath it.",
    "How does fog interact with the Golden Gate Bridge and why is it so iconic?",
    "If the Golden Gate Bridge could talk, what stories would it tell about its 90 years?",
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
# Norm measurement
# ---------------------------------------------------------------------------

def measure_norms(model, input_ids):
    """Measure activation norm at each layer's mlp.down_proj output (last token)."""
    n_layers = model.config.num_hidden_layers
    norms = [0.0] * n_layers
    hooks = []

    for i in range(n_layers):
        down_proj = model.model.layers[i].mlp.down_proj

        def make_hook(layer_idx):
            def hook_fn(m, inp, out):
                norms[layer_idx] = out[0, -1, :].float().norm().item()
            return hook_fn

        hooks.append(down_proj.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return norms


# ---------------------------------------------------------------------------
# Target metric: low-rank covariance via SVD of centered activations
# ---------------------------------------------------------------------------

def compute_all_target_covariance(model, tokenizer, prompts):
    """Compute low-rank covariance of activations at ALL layers across prompts.

    One forward pass per prompt, hooks every layer's module output (last token).
    Returns per layer: (components [n_prompts, H], eigvals [n_prompts])
      - components: right singular vectors of centered activations (Vt from SVD)
      - eigvals: covariance eigenvalues = S^2 / (n-1)

    Full return: (all_components [n_layers, n_prompts, H],
                  all_eigvals [n_layers, n_prompts])
    """
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    all_acts = [[] for _ in range(n_layers)]  # [layer][prompt] -> [H]
    hooks = []

    for i in range(n_layers):
        layer_mod = model.model.layers[i]

        def make_hook(layer_idx):
            def hook_fn(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                all_acts[layer_idx].append(o[:, -1, :].detach().float())
            return hook_fn

        hooks.append(layer_mod.register_forward_hook(make_hook(i)))

    try:
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = format_prompt(tokenizer, messages)
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    # Compute low-rank covariance per layer
    H = model.config.hidden_size
    n_prompts = len(prompts)
    all_components = torch.zeros(n_layers, n_prompts, H, device=device)
    all_eigvals = torch.zeros(n_layers, n_prompts, device=device)

    for i in range(n_layers):
        stacked = torch.cat(all_acts[i], dim=0)  # [n_prompts, H]
        centered = stacked - stacked.mean(dim=0)  # [n_prompts, H]
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)  # Vt: [n_prompts, H]
        eigvals = S ** 2 / (n_prompts - 1)  # covariance eigenvalues
        all_components[i] = Vt
        all_eigvals[i] = eigvals

    print(f"  Target covariance computed over {n_prompts} prompts, {n_layers} layers")
    print(f"  Top eigenvalue range: [{all_eigvals[:, 0].min():.2f}, {all_eigvals[:, 0].max():.2f}]")
    print(f"  Eigenvalue sum (trace) range: [{all_eigvals.sum(dim=1).min():.2f}, {all_eigvals.sum(dim=1).max():.2f}]")
    return all_components, all_eigvals


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
                num_vectors, num_iters, num_tokens, target_cov=None):
    """Batched block power iteration for one (source, target) pair.

    Args:
        target_cov: optional tuple (components [k, H], eigvals [k], reg_alpha)
            for covariance weighting. Applied as:
              jvp_weighted = V @ diag(λ) @ V.T @ jvp + reg_alpha * jvp
            When reg_alpha=0, this is pure low-rank projection.
            When reg_alpha>0, all directions get base weight α, covariance
            directions get additional weight λ.
    """
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
            if target_cov is not None:
                components, eigvals, reg_alpha = target_cov  # [n_comp, H], [n_comp], scalar
                proj = torch.einsum('csh,kh->csk', jvp, components)  # [c, seq, n_comp]
                proj = proj * eigvals                                 # [c, seq, n_comp]
                jvp_cov = torch.einsum('csk,kh->csh', proj, components)  # [c, seq, H]
                jvp = jvp_cov + reg_alpha * jvp
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
    """Generate steered text for one pair. Batch = k vectors."""
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
        steering["vec"] = (vectors * scale).to(device)
        prompt_text = format_prompt(tokenizer, messages)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        ids = inputs["input_ids"].expand(k, -1)
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
# Worker (multi-GPU via mp.spawn)
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

    # Compute target covariance
    if rank == 0:
        print(f"[GPU {rank}] Computing target covariance over "
              f"{len(GGB_METRIC_PROMPTS)} GGB prompts...", flush=True)
    all_components, all_eigvals = compute_all_target_covariance(
        model, tokenizer, GGB_METRIC_PROMPTS,
    )

    # Round-robin assignment
    my_pairs = [all_pairs[i] for i in range(rank, len(all_pairs), world_size)]

    pid = BASE_PROMPT["id"]
    prompt_dir = Path(args.output_dir) / pid
    pairs_dir = prompt_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    # Baseline generations (rank 0 only)
    if rank == 0:
        for gp in GEN_PROMPTS:
            baseline_file = prompt_dir / f"baseline_{gp['id']}.json"
            if not baseline_file.exists():
                print(f"[GPU {rank}] Running baseline for {gp['id']}...", flush=True)
                bl = generate_baseline(
                    model, tokenizer, gp["messages"],
                    args.num_samples, args.max_new_tokens, args.temperature,
                    seed_base=args.seed * 100,
                )
                with open(baseline_file, "w") as f:
                    json.dump({"prompt": gp, "results": bl}, f)

    # Skip already-done pairs
    remaining = [(s, t) for s, t in my_pairs
                 if not (pairs_dir / f"{s}_{t}.json").exists()]

    # Baseline logits for KL
    baseline_logits, input_ids = compute_baseline(
        model, tokenizer, BASE_PROMPT["messages"],
    )

    # Per-layer norms for scale_frac
    norms = None
    if args.scale_frac is not None:
        norms = measure_norms(model, input_ids)
        if rank == 0:
            print(f"[GPU {rank}] Norms: min={min(norms):.1f} max={max(norms):.1f} "
                  f"median={sorted(norms)[len(norms)//2]:.1f}", flush=True)

    gen_ids = [gp["id"] for gp in GEN_PROMPTS]
    print(f"[GPU {rank}] Processing {len(remaining)} pairs "
          f"(skipped {len(my_pairs) - len(remaining)} already done, "
          f"gen prompts: {gen_ids})", flush=True)
    t0 = time.time()

    for idx, (s, t) in enumerate(remaining):
        torch.manual_seed(args.seed + s * 1000 + t)

        # Get covariance for this target layer
        components = all_components[t]  # [n_prompts, H]
        eigvals = all_eigvals[t]        # [n_prompts]
        # reg_alpha: fraction of top eigenvalue as regularization
        reg_alpha = args.cov_reg_frac * eigvals[0].item()
        target_cov = (components.to(dtype=model.dtype), eigvals.to(dtype=model.dtype), reg_alpha)

        vecs, sigmas = compute_svd(
            model, input_ids, s, t,
            args.num_vectors, args.num_iters, args.num_tokens,
            target_cov=target_cov,
        )

        # Compute per-pair scale
        if norms is not None:
            pair_scale = args.scale_frac * norms[s]
        else:
            pair_scale = args.scale

        # KL divergence
        kls = compute_kl(model, s, vecs, pair_scale, input_ids, baseline_logits)

        # Steered generation (only if KL above threshold)
        gens = {}
        max_kl = max(kls)
        if max_kl >= args.kl_threshold:
            for gi, gp in enumerate(GEN_PROMPTS):
                seed_base = args.seed + s * 10000 + t * 100 + gi * 10
                gens[gp["id"]] = generate_for_pair(
                    model, tokenizer, s, vecs, gp["messages"],
                    pair_scale, args.num_samples, args.max_new_tokens,
                    args.temperature, seed_base,
                )

        # Save per-pair result
        pair_data = {
            "source_layer": s,
            "target_layer": t,
            "scale": pair_scale,
            "sigmas": sigmas,
            "kl_divergences": kls,
            "vectors": vecs.cpu().tolist(),
            "generations": gens,
        }
        with open(pairs_dir / f"{s}_{t}.json", "w") as f:
            json.dump(pair_data, f)

        if (idx + 1) % 10 == 0 or idx == len(remaining) - 1:
            el = time.time() - t0
            rate = (idx + 1) / el
            rem = (len(remaining) - idx - 1) / rate if rate > 0 else 0
            gen_flag = "GEN" if max_kl >= args.kl_threshold else "skip"
            scale_str = f"scale={pair_scale:.1f}" if norms is not None else ""
            print(f"[GPU {rank}] {idx+1}/{len(remaining)} "
                  f"({s},{t}) {scale_str} σ₁={sigmas[0]:.0f} maxKL={max_kl:.2f} [{gen_flag}] "
                  f"{el:.0f}s/{rem:.0f}s left", flush=True)

    print(f"[GPU {rank}] Done.", flush=True)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(args):
    """Merge per-pair JSON files into merged.pt + summary.json."""
    config = AutoConfig.from_pretrained(args.model)
    n = config.num_hidden_layers
    k = args.num_vectors

    pid = BASE_PROMPT["id"]
    prompt_dir = Path(args.output_dir) / pid
    pairs_dir = prompt_dir / "pairs"

    if not pairs_dir.exists():
        print(f"[{pid}] No pairs directory, skipping")
        return

    sigma_map = torch.full((n, n, k), float("nan"))
    kl_map = torch.full((n, n, k), float("nan"))
    scale_map = torch.full((n, n), float("nan"))
    vectors = {}
    pair_count = 0

    for f in sorted(pairs_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        s = data["source_layer"]
        t = data["target_layer"]
        sigma_map[s, t] = torch.tensor(data["sigmas"])
        kl_map[s, t] = torch.tensor(data["kl_divergences"])
        if "scale" in data:
            scale_map[s, t] = data["scale"]
        vectors[f"{s}_{t}"] = torch.tensor(data["vectors"])
        pair_count += 1

    merged = {
        "metadata": {
            "model": args.model,
            "prompt_id": pid,
            "num_layers": n,
            "hidden_dim": config.hidden_size,
            "scale": args.scale,
            "scale_frac": args.scale_frac,
            "num_vectors": k,
            "num_iters": args.num_iters,
            "num_tokens": args.num_tokens,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "target_metric": "cov",
            "cov_reg_frac": args.cov_reg_frac,
            "num_pairs": pair_count,
            "source_range": [args.source_start, args.source_end],
            "timestamp": datetime.now().isoformat(),
        },
        "sigma_map": sigma_map,
        "kl_map": kl_map,
        "scale_map": scale_map,
        "vectors": vectors,
    }
    torch.save(merged, prompt_dir / "merged.pt")

    summary = {"metadata": merged["metadata"]}
    with open(prompt_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    valid = ~sigma_map[:, :, 0].isnan()
    expected = sum(1 for s in range(args.source_start, args.source_end + 1)
                   for t in range(s + 1, n))
    print(f"\n[{pid}] Merged {valid.sum().item()}/{expected} pairs")
    if valid.any():
        print(f"  σ₁ range: [{sigma_map[:,:,0][valid].min():.0f}, "
              f"{sigma_map[:,:,0][valid].max():.0f}]")
        print(f"  KL₁ range: [{kl_map[:,:,0][valid].min():.2f}, "
              f"{kl_map[:,:,0][valid].max():.2f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Map Jacobian with low-rank covariance target metric",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scale", type=float, default=10.0,
                        help="Fixed steering scale (overridden by --scale-frac)")
    parser.add_argument("--scale-frac", type=float, default=0.35,
                        help="Scale as fraction of source layer activation norm")
    parser.add_argument("--source-start", type=int, default=7)
    parser.add_argument("--source-end", type=int, default=20)
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl-threshold", type=float, default=0.5,
                        help="Only generate text for pairs with max KL >= threshold")
    parser.add_argument("--cov-reg-frac", type=float, default=0.0,
                        help="Regularization as fraction of top eigenvalue. "
                             "0=pure low-rank projection, >0 adds α*I to covariance")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--output-dir", default="results/diverse_map_ggb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.merge_only:
        merge(args)
        return

    config = AutoConfig.from_pretrained(args.model)
    n = config.num_hidden_layers
    pairs = [(s, t) for s in range(args.source_start, args.source_end + 1)
             for t in range(s + 1, n)]

    print(f"Model: {args.model} ({n} layers)")
    print(f"Source layers: {args.source_start}-{args.source_end}, "
          f"targets: all > source ({len(pairs)} pairs)")
    reg_str = f", reg={args.cov_reg_frac}" if args.cov_reg_frac > 0 else ", low-rank"
    print(f"Target metric: covariance (SVD of centered activations, "
          f"{len(GGB_METRIC_PROMPTS)} prompts{reg_str})")
    print(f"k={args.num_vectors}, iters={args.num_iters}, tokens={args.num_tokens}, "
          f"scale_frac={args.scale_frac}")
    print(f"Generation: {args.num_samples} samples, {args.max_new_tokens} tokens, "
          f"temp={args.temperature}, KL threshold={args.kl_threshold}")
    print(f"Gen prompts: {[gp['id'] for gp in GEN_PROMPTS]}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Output: {args.output_dir}")

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
