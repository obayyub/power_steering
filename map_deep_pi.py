#!/usr/bin/env python3
"""
Deep power iteration map: 100 vectors per layer pair, all source layers.
Generates text only for vectors with KL > threshold.

Multi-GPU via mp.spawn.

Usage:
  uv run python map_deep_pi.py --num-gpus 2
  uv run python map_deep_pi.py --merge-only
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

BASE_PROMPT = {
    "id": "roleplay",
    "messages": [
        {"role": "user", "content": "You are a grizzled Victorian-era lighthouse keeper writing a letter to your estranged daughter. You haven't spoken in five years. Tonight a terrible storm is coming and you're not sure you'll survive it. Write the letter."},
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def measure_norms(model, input_ids):
    n_layers = model.config.num_hidden_layers
    norms = [0.0] * n_layers
    hooks = []
    for i in range(n_layers):
        down_proj = model.model.layers[i].mlp.down_proj
        def make_hook(idx):
            def hook_fn(m, inp, out):
                norms[idx] = out[0, -1, :].float().norm().item()
            return hook_fn
        hooks.append(down_proj.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()
    return norms


def compute_baseline(model, tokenizer, messages):
    device = next(model.parameters()).device
    prompt_text = format_prompt(tokenizer, messages)
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :]
    return logits, input_ids


# ---------------------------------------------------------------------------
# SVD (chunked)
# ---------------------------------------------------------------------------

def compute_svd(model, input_ids, source_layer, target_layer,
                num_vectors, num_iters, num_tokens, batch_vectors):
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
        V = torch.randn(H, k, device=device, dtype=dtype)
        V = orthogonalize(V)

        def apply_jtj_chunk(V_chunk):
            c = V_chunk.shape[1]
            sv = torch.zeros(c, H, device=device, dtype=dtype, requires_grad=True)
            steer["v"] = sv
            model(input_ids.expand(c, -1))
            t = cap["t"][:, tok_slice, :]
            u = torch.zeros_like(t, requires_grad=True)
            g = torch.autograd.grad((t * u).sum(), sv, create_graph=True, retain_graph=True)[0]
            jvp = torch.autograd.grad((g * V_chunk.T[:c]).sum(), u, retain_graph=True)[0]
            r = torch.autograd.grad((t * jvp.detach()).sum(), sv)[0]
            steer["v"] = None
            return r.T

        def apply_jtj(V_in):
            if batch_vectors >= V_in.shape[1]:
                return apply_jtj_chunk(V_in)
            results = []
            for start in range(0, V_in.shape[1], batch_vectors):
                end = min(start + batch_vectors, V_in.shape[1])
                results.append(apply_jtj_chunk(V_in[:, start:end]))
            return torch.cat(results, dim=1)

        for i in range(num_iters):
            new_V = apply_jtj(V)
            V = orthogonalize(new_V)

        V, sigmas = rayleigh_ritz(V, apply_jtj)
        return V.T.detach(), sigmas  # [k, H], [k]

    finally:
        h1.remove()
        h2.remove()
        steer["v"] = None


# ---------------------------------------------------------------------------
# KL (chunked)
# ---------------------------------------------------------------------------

def compute_kl(model, source_layer, vectors, scale, input_ids, baseline_logits,
               batch_size=25):
    k = vectors.shape[0]
    down_proj = model.model.layers[source_layer].mlp.down_proj
    steer = {"v": None}

    def hook(m, i, o):
        if steer["v"] is not None:
            return o + steer["v"].unsqueeze(1)
        return o

    h = down_proj.register_forward_hook(hook)
    kl_all = []
    try:
        log_p = F.log_softmax(baseline_logits.float(), dim=-1)
        for start in range(0, k, batch_size):
            end = min(start + batch_size, k)
            chunk = vectors[start:end]
            steer["v"] = chunk * scale
            with torch.no_grad():
                steered_logits = model(input_ids.expand(end - start, -1)).logits[:, -1, :]
            log_q = F.log_softmax(steered_logits.float(), dim=-1)
            kl = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
            kl_all.extend(kl.tolist())
        return kl_all
    finally:
        h.remove()
        steer["v"] = None


# ---------------------------------------------------------------------------
# Generation (one vector at a time)
# ---------------------------------------------------------------------------

def generate_steered(model, tokenizer, input_ids, source_layer, vector, scale,
                     max_new_tokens, temperature, seed):
    down_proj = model.model.layers[source_layer].mlp.down_proj
    steer = {"v": None}

    def hook(m, i, o):
        if steer["v"] is not None:
            return o + steer["v"]
        return o

    h = down_proj.register_forward_hook(hook)
    steer["v"] = (vector * scale).unsqueeze(0).unsqueeze(0)
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        h.remove()
        steer["v"] = None


def generate_baseline(model, tokenizer, messages, max_new_tokens, temperature, seed):
    device = next(model.parameters()).device
    prompt_text = format_prompt(tokenizer, messages)
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, all_pairs, args):
    device = rank if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    my_pairs = [all_pairs[i] for i in range(rank, len(all_pairs), world_size)]

    pid = BASE_PROMPT["id"]
    prompt_dir = Path(args.output_dir) / pid
    pairs_dir = prompt_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    # Baseline generation (rank 0 only)
    if rank == 0:
        baseline_file = prompt_dir / "baseline_roleplay.json"
        if not baseline_file.exists():
            print(f"[GPU {rank}] Running baseline generation...", flush=True)
            bl = generate_baseline(
                model, tokenizer, BASE_PROMPT["messages"],
                args.max_new_tokens, args.temperature, args.seed * 100,
            )
            with open(baseline_file, "w") as f:
                json.dump({"prompt": BASE_PROMPT, "text": bl}, f)

    # Baseline logits for KL
    baseline_logits, input_ids = compute_baseline(
        model, tokenizer, BASE_PROMPT["messages"],
    )

    # Per-layer norms
    norms = measure_norms(model, input_ids)
    if rank == 0:
        print(f"[GPU {rank}] Norms: min={min(norms):.1f} max={max(norms):.1f} "
              f"median={sorted(norms)[len(norms)//2]:.1f}", flush=True)

    # Skip done pairs
    remaining = [(s, t) for s, t in my_pairs
                 if not (pairs_dir / f"{s}_{t}.json").exists()]

    print(f"[GPU {rank}] Processing {len(remaining)} pairs "
          f"(skipped {len(my_pairs) - len(remaining)} already done)", flush=True)
    t0 = time.time()

    for idx, (s, t) in enumerate(remaining):
        torch.manual_seed(args.seed + s * 1000 + t)

        vecs, sigmas = compute_svd(
            model, input_ids, s, t,
            args.num_vectors, args.num_iters, args.num_tokens,
            args.batch_vectors,
        )

        pair_scale = args.scale_frac * norms[s]

        # KL divergence (chunked)
        kls = compute_kl(model, s, vecs, pair_scale, input_ids, baseline_logits)

        # Generate only for vectors with KL > threshold
        gens = {}
        active_indices = [i for i, k in enumerate(kls) if k >= args.kl_threshold]
        if active_indices:
            gen_results = []
            for vi in active_indices:
                text = generate_steered(
                    model, tokenizer, input_ids, s, vecs[vi], pair_scale,
                    args.max_new_tokens, args.temperature,
                    args.seed + s * 10000 + t * 100 + vi,
                )
                gen_results.append({"v": vi, "sigma": sigmas[vi], "kl": kls[vi], "text": text})
            gens["roleplay"] = gen_results

        # Save — vectors only for active indices to save space
        active_vecs = {str(i): vecs[i].cpu().tolist() for i in active_indices}

        pair_data = {
            "source_layer": s,
            "target_layer": t,
            "scale": pair_scale,
            "sigmas": sigmas,
            "kl_divergences": kls,
            "active_indices": active_indices,
            "vectors": active_vecs,
            "generations": gens,
        }
        with open(pairs_dir / f"{s}_{t}.json", "w") as f:
            json.dump(pair_data, f)

        n_active = len(active_indices)
        max_kl = max(kls)
        if (idx + 1) % (1 if idx < 10 else 5) == 0 or idx == len(remaining) - 1:
            el = time.time() - t0
            rate = (idx + 1) / el
            rem = (len(remaining) - idx - 1) / rate if rate > 0 else 0
            print(f"[GPU {rank}] {idx+1}/{len(remaining)} "
                  f"({s},{t}) σ₁={sigmas[0]:.0f} maxKL={max_kl:.1f} "
                  f"active={n_active}/{args.num_vectors} "
                  f"{el:.0f}s/{rem:.0f}s left", flush=True)

    print(f"[GPU {rank}] Done.", flush=True)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(args):
    config = AutoConfig.from_pretrained(args.model)
    n = config.num_hidden_layers
    k = args.num_vectors

    pid = BASE_PROMPT["id"]
    prompt_dir = Path(args.output_dir) / pid
    pairs_dir = prompt_dir / "pairs"

    if not pairs_dir.exists():
        print("No pairs directory")
        return

    sigma_map = torch.full((n, n, k), float("nan"))
    kl_map = torch.full((n, n, k), float("nan"))
    scale_map = torch.full((n, n), float("nan"))
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
        pair_count += 1

    merged = {
        "metadata": {
            "model": args.model,
            "prompt_id": pid,
            "num_layers": n,
            "hidden_dim": config.hidden_size,
            "scale_frac": args.scale_frac,
            "num_vectors": k,
            "num_iters": args.num_iters,
            "num_tokens": args.num_tokens,
            "kl_threshold": args.kl_threshold,
            "seed": args.seed,
            "num_pairs": pair_count,
            "source_range": [args.source_start, args.source_end],
            "timestamp": datetime.now().isoformat(),
        },
        "sigma_map": sigma_map,
        "kl_map": kl_map,
        "scale_map": scale_map,
    }
    torch.save(merged, prompt_dir / "merged.pt")

    summary = {"metadata": merged["metadata"]}
    with open(prompt_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    valid = ~sigma_map[:, :, 0].isnan()
    expected = sum(1 for s in range(args.source_start, args.source_end + 1)
                   for t in range(s + 1, n))
    print(f"\nMerged {valid.sum().item()}/{expected} pairs")
    if valid.any():
        print(f"  σ₁ range: [{sigma_map[:,:,0][valid].min():.0f}, "
              f"{sigma_map[:,:,0][valid].max():.0f}]")
        print(f"  KL₁ range: [{kl_map[:,:,0][valid].min():.2f}, "
              f"{kl_map[:,:,0][valid].max():.2f}]")
        # Count active vectors across all pairs
        total_active = 0
        for f in pairs_dir.glob("*.json"):
            with open(f) as fp:
                data = json.load(fp)
            total_active += len(data.get("active_indices", []))
        print(f"  Total active vectors (KL>{args.kl_threshold}): {total_active}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scale-frac", type=float, default=0.35)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-end", type=int, default=35)
    parser.add_argument("--num-vectors", type=int, default=100)
    parser.add_argument("--num-iters", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--batch-vectors", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl-threshold", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--output-dir", default="results/deep_pi_map")
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
    print(f"Source layers: {args.source_start}-{args.source_end} "
          f"({len(pairs)} pairs)")
    print(f"Vectors: {args.num_vectors}, iters: {args.num_iters}, "
          f"batch: {args.batch_vectors}")
    print(f"KL threshold: {args.kl_threshold} (generate only for active vectors)")
    print(f"GPUs: {args.num_gpus}")
    print(f"Output: {args.output_dir}")

    if args.num_gpus > 1:
        mp.spawn(worker, args=(args.num_gpus, pairs, args), nprocs=args.num_gpus)
    else:
        worker(0, 1, pairs, args)

    merge(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
