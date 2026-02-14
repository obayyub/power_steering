#!/usr/bin/env python3
"""
Generate steered text for all layer pairs using diverse prompt styles.

Focus on behavioral diversity rather than task accuracy. Uses 4 prompt
styles (narrative, expository, code, dialogue) to surface different
latent behaviors across the 378 (source, target) × 12 vector combinations.

Multi-GPU via mp.spawn, resume support via per-pair JSON files.

Usage:
  uv run python map_generate_diverse.py --vectors-file results/jacobian_map/merged.pt --num-gpus 8
  uv run python map_generate_diverse.py --merge-only
"""

import json
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
import argparse
import time


# ---------------------------------------------------------------------------
# Prompts — 4 styles, raw continuations for base model
# ---------------------------------------------------------------------------

PROMPTS = [
    {
        "id": "narrative",
        "text": "Once upon a time, in a small village nestled between two mountains, there lived a",
    },
    {
        "id": "expository",
        "text": "The most important scientific discovery of the 20th century was",
    },
    {
        "id": "code",
        "text": "# Python function to find the shortest path in a weighted graph\ndef shortest_path(graph, start, end):\n    \"\"\"",
    },
    {
        "id": "dialogue",
        "text": "Alice: I've been thinking about moving to another country.\nBob: Really? What made you consider that?\nAlice:",
    },
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_for_pair(model, tokenizer, source_layer, vectors, prompts,
                      scale, num_samples, max_new_tokens, temperature, seed_base):
    """
    Generate steered text for one (source, target) pair.

    Batches all k vectors × n prompts together. Each batch element gets
    its own steering vector via per-element hook.
    """
    k = vectors.shape[0]
    n_p = len(prompts)
    device = next(model.parameters()).device

    down_proj = model.model.layers[source_layer].mlp.down_proj
    steering = {"vec": None}

    def hook(m, i, o):
        if steering["vec"] is not None:
            return o + steering["vec"].unsqueeze(1)
        return o

    handle = down_proj.register_forward_hook(hook)

    try:
        # [k*n_p, H]: vector i applied to batch elements [i*n_p : (i+1)*n_p]
        steering["vec"] = (vectors.repeat_interleave(n_p, dim=0) * scale).to(device)

        # Prompts repeated: [p0,p1,...pN, p0,p1,...pN, ...] k times
        prompt_texts = [p["text"] for p in prompts]
        all_prompts = prompt_texts * k
        tokenizer.padding_side = "left"
        inputs = tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)
        input_len = inputs["input_ids"].shape[1]

        results = []
        for si in range(num_samples):
            torch.manual_seed(seed_base + si)

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=temperature,
                )

            for idx in range(k * n_p):
                vi = idx // n_p
                pi = idx % n_p
                text = tokenizer.decode(outputs[idx, input_len:], skip_special_tokens=True)
                results.append({
                    "v": vi, "p": pi, "s": si,
                    "prompt_id": prompts[pi]["id"],
                    "text": text,
                })

        return results

    finally:
        handle.remove()
        steering["vec"] = None


def generate_baseline(model, tokenizer, prompts,
                      num_samples, max_new_tokens, temperature, seed_base):
    """Generate unsteered baseline."""
    device = next(model.parameters()).device
    prompt_texts = [p["text"] for p in prompts]
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    results = []
    for si in range(num_samples):
        torch.manual_seed(seed_base + si)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
            )
        for pi in range(len(prompts)):
            text = tokenizer.decode(outputs[pi, input_len:], skip_special_tokens=True)
            results.append({
                "p": pi, "s": si,
                "prompt_id": prompts[pi]["id"],
                "text": text,
            })
    return results


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, all_pairs, args):
    device = rank if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = torch.load(args.vectors_file, map_location="cpu", weights_only=True)
    vectors_dict = data["vectors"]

    pairs_dir = Path(args.output_dir) / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    my_pairs = [all_pairs[i] for i in range(rank, len(all_pairs), world_size)]
    my_pairs = [(s, t) for s, t in my_pairs if not (pairs_dir / f"{s}_{t}.json").exists()]

    # Baseline (rank 0 only)
    baseline_file = Path(args.output_dir) / "baseline.json"
    if rank == 0 and not baseline_file.exists():
        print(f"[GPU {rank}] Running baseline...", flush=True)
        baseline = generate_baseline(
            model, tokenizer, PROMPTS,
            args.num_samples, args.max_new_tokens, args.temperature,
            seed_base=args.seed * 100,
        )
        with open(baseline_file, "w") as f:
            json.dump({
                "prompts": PROMPTS,
                "results": baseline,
            }, f)
        print(f"[GPU {rank}] Baseline saved ({len(baseline)} generations)", flush=True)

    print(f"[GPU {rank}] Processing {len(my_pairs)} pairs", flush=True)
    t0 = time.time()

    for idx, (s, t) in enumerate(my_pairs):
        key = f"{s}_{t}"
        model_device = next(model.parameters()).device
        vecs = vectors_dict[key].to(model_device)

        seed_base = args.seed + s * 10000 + t * 100
        pair_results = generate_for_pair(
            model, tokenizer, s, vecs, PROMPTS,
            args.scale, args.num_samples, args.max_new_tokens, args.temperature,
            seed_base,
        )

        pair_data = {
            "source_layer": s,
            "target_layer": t,
            "num_vectors": vecs.shape[0],
            "generations": pair_results,
        }

        with open(pairs_dir / f"{key}.json", "w") as f:
            json.dump(pair_data, f)

        if (idx + 1) % 5 == 0 or idx == len(my_pairs) - 1:
            el = time.time() - t0
            rate = (idx + 1) / el
            rem = (len(my_pairs) - idx - 1) / rate if rate > 0 else 0
            print(f"[GPU {rank}] {idx+1}/{len(my_pairs)} "
                  f"({s},{t}) "
                  f"{el:.0f}s/{rem:.0f}s left", flush=True)

    print(f"[GPU {rank}] Done.", flush=True)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(args):
    """Merge pair files into summary and per-source generation files."""
    pairs_dir = Path(args.output_dir) / "pairs"
    gen_dir = Path(args.output_dir) / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    by_source = {}
    pair_count = 0

    for f in sorted(pairs_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        s = data["source_layer"]
        pair_count += 1

        if s not in by_source:
            by_source[s] = {}
        by_source[s][f.stem] = data

    summary = {
        "metadata": {
            "model": args.model,
            "scale": args.scale,
            "num_prompts": len(PROMPTS),
            "prompt_ids": [p["id"] for p in PROMPTS],
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "num_pairs": pair_count,
            "timestamp": datetime.now().isoformat(),
        },
        "prompts": PROMPTS,
    }

    summary_file = Path(args.output_dir) / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    for s, pairs in sorted(by_source.items()):
        with open(gen_dir / f"source_{s}.json", "w") as f:
            json.dump(pairs, f)

    total_gens = pair_count * len(PROMPTS) * 12 * args.num_samples
    print(f"\nMerged {pair_count} pairs → {summary_file}")
    print(f"~{total_gens} total generations")
    print(f"Split into {len(by_source)} source files in {gen_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate steered text for all layer pairs (diverse prompts)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--vectors-file", default="results/jacobian_map/merged.pt")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--output-dir", default="results/diverse_gen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.merge_only:
        merge(args)
        return

    data = torch.load(args.vectors_file, map_location="cpu", weights_only=True)
    all_pairs = []
    for key in data["vectors"]:
        s, t = key.split("_")
        all_pairs.append((int(s), int(t)))
    all_pairs.sort()

    print(f"Model: {args.model}")
    print(f"Vectors: {args.vectors_file} ({len(all_pairs)} pairs)")
    print(f"Prompts: {len(PROMPTS)} styles, Samples: {args.num_samples}, Scale: {args.scale}")
    print(f"Batch per pair: {12 * len(PROMPTS)} (k=12 × {len(PROMPTS)} prompts)")
    print(f"Total generations: ~{len(all_pairs) * 12 * len(PROMPTS) * args.num_samples}")
    print(f"GPUs: {args.num_gpus}")

    if args.num_gpus > 1:
        mp.spawn(
            worker, nprocs=args.num_gpus,
            args=(args.num_gpus, all_pairs, args),
            join=True,
        )
    else:
        worker(0, 1, all_pairs, args)

    merge(args)


if __name__ == "__main__":
    main()
