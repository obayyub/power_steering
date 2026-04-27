#!/usr/bin/env python3
"""
Run power iteration on a single (source, target) pair with many vectors.
Output: sigmas and KL divergences. No generation.

Example:
  uv run python run_deep_pi.py --model Qwen/Qwen3-8B --source 13 --target 21 --num-vectors 150
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
import argparse


PROMPT = "You are a grizzled Victorian-era lighthouse keeper writing a letter to your estranged daughter. You haven't spoken in five years. Tonight a terrible storm is coming."


def format_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def orthogonalize(V):
    Q = []
    for v in V.T:
        for q in Q:
            v = v - torch.dot(v, q) * q
        norm = v.norm()
        if norm > 1e-10:
            Q.append(v / norm)
    return torch.stack(Q, dim=1) if Q else V


def rayleigh_ritz(V, apply_jtj_fn):
    JtJ_V = apply_jtj_fn(V)
    M = (V.T @ JtJ_V).float()
    M = (M + M.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    V_rotated = V @ eigenvectors.to(V.dtype)
    sigmas = eigenvalues.clamp(min=0).sqrt().tolist()
    return V_rotated, sigmas


def measure_norm(model, input_ids, source_layer):
    """Measure activation norm at source layer's mlp.down_proj output (last token)."""
    norm_val = [0.0]
    down_proj = model.model.layers[source_layer].mlp.down_proj

    def hook(m, inp, out):
        norm_val[0] = out[0, -1, :].float().norm().item()

    h = down_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids)
    h.remove()
    return norm_val[0]


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

        convergence_log = []
        for i in range(num_iters):
            new_V = apply_jtj(V)
            V = orthogonalize(new_V)
            # Track approximate sigmas across the full spectrum
            all_approx = [new_V[:, j].norm().item() for j in range(k)]
            top5 = all_approx[:5]
            mid = all_approx[k // 2 - 2: k // 2 + 3]
            bot5 = all_approx[-5:]
            convergence_log.append(all_approx)
            print(f"  Iter {i:2d}: "
                  f"top5={[f'{s:.0f}' for s in top5]}  "
                  f"mid={[f'{s:.0f}' for s in mid]}  "
                  f"bot5={[f'{s:.0f}' for s in bot5]}",
                  flush=True)

        print("  Rayleigh-Ritz...", flush=True)
        V, sigmas = rayleigh_ritz(V, apply_jtj)
        return V.T.detach(), sigmas, convergence_log  # [k, H], [k], [[k] * iters]

    finally:
        h1.remove()
        h2.remove()
        steer["v"] = None


def compute_kl(model, source_layer, vectors, scale, input_ids, baseline_logits,
               batch_size=32):
    """KL(steered || baseline) for each vector, chunked."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--source", type=int, default=13)
    parser.add_argument("--target", type=int, default=21)
    parser.add_argument("--num-vectors", type=int, default=150)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--batch-vectors", type=int, default=12,
                        help="Vectors per forward pass (memory-limited)")
    parser.add_argument("--scale-frac", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )

    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = format_prompt(tokenizer, messages)
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model.device)
    print(f"Prompt tokens: {input_ids.shape[1]}")

    # Measure norm for scaling
    norm = measure_norm(model, input_ids, args.source)
    scale = args.scale_frac * norm
    print(f"Source layer {args.source} norm: {norm:.1f}, scale: {scale:.1f}")

    # Baseline logits
    with torch.no_grad():
        baseline_logits = model(input_ids).logits[:, -1, :]
    print(f"Baseline logits computed")

    # Power iteration
    print(f"\nRunning power iteration: ({args.source},{args.target}), "
          f"k={args.num_vectors}, iters={args.num_iters}, "
          f"batch={args.batch_vectors}")
    vectors, sigmas, convergence_log = compute_svd(
        model, input_ids, args.source, args.target,
        args.num_vectors, args.num_iters, args.num_tokens, args.batch_vectors,
    )
    print(f"Top-10 sigmas: {[f'{s:.1f}' for s in sigmas[:10]]}")
    print(f"Last-10 sigmas: {[f'{s:.1f}' for s in sigmas[-10:]]}")

    # KL
    print(f"\nComputing KL divergences...")
    kl = compute_kl(model, args.source, vectors, scale, input_ids, baseline_logits)
    print(f"Top-10 KL: {[f'{k:.2f}' for k in kl[:10]]}")
    print(f"KL > 1: {sum(1 for k in kl if k > 1)}/{len(kl)}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"deep_pi_{model_short}_{args.source}_{args.target}_{timestamp}.pt"

    torch.save({
        "vectors": vectors,
        "sigmas": sigmas,
        "kl_divergences": kl,
        "convergence_log": convergence_log,
        "model": args.model,
        "source_layer": args.source,
        "target_layer": args.target,
        "num_vectors": args.num_vectors,
        "num_iters": args.num_iters,
        "num_tokens": args.num_tokens,
        "scale": scale,
        "scale_frac": args.scale_frac,
        "norm": norm,
        "seed": args.seed,
        "prompt": PROMPT,
    }, out_file)

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
