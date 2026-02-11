#!/usr/bin/env python3
"""
Generate steered text for all layer pairs from jacobian map results.

For each (source, target) pair, batches all k vectors × n questions together.
Saves per-pair JSON files for resume support. Multi-GPU via mp.spawn.

Usage:
  # Single GPU
  uv run python map_generate.py --vectors-file results/jacobian_map/merged.pt

  # 4 GPUs
  uv run python map_generate.py --vectors-file results/jacobian_map/merged.pt --num-gpus 4

  # Merge only
  uv run python map_generate.py --merge-only
"""

import json
import random
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
import argparse
import time

from eval_cot_math import generate_math_problem, make_prompt, extract_answer


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

def make_questions(num_questions, seed):
    """Generate deterministic set of math questions."""
    random.seed(seed)
    # Original training question first
    questions = [("a=5+6, b=2+7. What is a*b?", 99)]
    while len(questions) < num_questions:
        q = generate_math_problem()
        if q[0] != questions[0][0]:
            questions.append(q)
    prompts = [make_prompt(q) for q, _ in questions]
    answers = [a for _, a in questions]
    return questions, prompts, answers


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_for_pair(model, tokenizer, source_layer, vectors, prompts, answers,
                      scale, num_samples, max_new_tokens, temperature, seed_base):
    """
    Generate steered text for one (source, target) pair.

    Batches all k vectors × n questions together. Each batch element gets
    its own steering vector via per-element hook.

    Returns list of result dicts.
    """
    k = vectors.shape[0]
    n_q = len(prompts)
    device = next(model.parameters()).device

    down_proj = model.model.layers[source_layer].mlp.down_proj
    steering = {"vec": None}

    def hook(m, i, o):
        if steering["vec"] is not None:
            return o + steering["vec"].unsqueeze(1)
        return o

    handle = down_proj.register_forward_hook(hook)

    try:
        # [k*n_q, H]: vector i applied to batch elements [i*n_q : (i+1)*n_q]
        steering["vec"] = (vectors.repeat_interleave(n_q, dim=0) * scale).to(device)

        # Prompts repeated: [q0,q1,...qN, q0,q1,...qN, ...] k times
        all_prompts = prompts * k
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

            for idx in range(k * n_q):
                vi = idx // n_q
                qi = idx % n_q
                text = tokenizer.decode(outputs[idx, input_len:], skip_special_tokens=True)
                pred = extract_answer(text)
                results.append({
                    "v": vi, "q": qi, "s": si,
                    "text": text,
                    "pred": pred,
                    "correct": pred == answers[qi],
                })

        return results

    finally:
        handle.remove()
        steering["vec"] = None


def generate_baseline(model, tokenizer, prompts, answers,
                      num_samples, max_new_tokens, temperature, seed_base):
    """Generate unsteered baseline."""
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
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
        for qi in range(len(prompts)):
            text = tokenizer.decode(outputs[qi, input_len:], skip_special_tokens=True)
            pred = extract_answer(text)
            results.append({
                "q": qi, "s": si,
                "text": text, "pred": pred,
                "correct": pred == answers[qi],
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

    # Load vectors (to CPU, move per-pair)
    data = torch.load(args.vectors_file, map_location="cpu", weights_only=True)
    vectors_dict = data["vectors"]

    # Deterministic questions (same across all workers)
    questions, prompts, answers = make_questions(args.num_questions, args.seed)

    # My pairs, skip completed
    pairs_dir = Path(args.output_dir) / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    my_pairs = [all_pairs[i] for i in range(rank, len(all_pairs), world_size)]
    my_pairs = [(s, t) for s, t in my_pairs if not (pairs_dir / f"{s}_{t}.json").exists()]

    # Baseline (rank 0 only)
    baseline_file = Path(args.output_dir) / "baseline.json"
    if rank == 0 and not baseline_file.exists():
        print(f"[GPU {rank}] Running baseline...")
        baseline = generate_baseline(
            model, tokenizer, prompts, answers,
            args.num_samples, args.max_new_tokens, args.temperature,
            seed_base=args.seed * 100,
        )
        correct = sum(1 for r in baseline if r["correct"])
        print(f"[GPU {rank}] Baseline: {correct}/{len(baseline)} = {correct/len(baseline):.1%}")
        with open(baseline_file, "w") as f:
            json.dump({
                "questions": [{"question": q, "answer": a} for q, a in questions],
                "results": baseline,
            }, f)

    print(f"[GPU {rank}] Processing {len(my_pairs)} pairs")
    t0 = time.time()

    for idx, (s, t) in enumerate(my_pairs):
        key = f"{s}_{t}"
        model_device = next(model.parameters()).device
        vecs = vectors_dict[key].to(model_device)

        seed_base = args.seed + s * 10000 + t * 100
        pair_results = generate_for_pair(
            model, tokenizer, s, vecs, prompts, answers,
            args.scale, args.num_samples, args.max_new_tokens, args.temperature,
            seed_base,
        )

        # Summary stats
        total = len(pair_results)
        correct = sum(1 for r in pair_results if r["correct"])
        per_vec = {}
        for vi in range(vecs.shape[0]):
            vr = [r for r in pair_results if r["v"] == vi]
            vc = sum(1 for r in vr if r["correct"])
            per_vec[str(vi)] = {"accuracy": vc / len(vr), "correct": vc, "total": len(vr)}

        pair_data = {
            "source_layer": s,
            "target_layer": t,
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "per_vector": per_vec,
            "generations": pair_results,
        }

        with open(pairs_dir / f"{key}.json", "w") as f:
            json.dump(pair_data, f)

        if (idx + 1) % 5 == 0 or idx == len(my_pairs) - 1:
            el = time.time() - t0
            rate = (idx + 1) / el
            rem = (len(my_pairs) - idx - 1) / rate if rate > 0 else 0
            print(f"[GPU {rank}] {idx+1}/{len(my_pairs)} "
                  f"({s},{t}) acc={correct/total:.1%} "
                  f"{el:.0f}s/{rem:.0f}s left")

    print(f"[GPU {rank}] Done.")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(args):
    """Merge pair files into summary JSON and per-source generation files."""
    pairs_dir = Path(args.output_dir) / "pairs"
    gen_dir = Path(args.output_dir) / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    summary_pairs = {}
    by_source = {}

    for f in sorted(pairs_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        key = f.stem
        s = data["source_layer"]

        # Summary: no raw text
        summary_pairs[key] = {
            "source_layer": s,
            "target_layer": data["target_layer"],
            "accuracy": data["accuracy"],
            "per_vector": {k: v["accuracy"] for k, v in data["per_vector"].items()},
        }

        # Group for split files
        if s not in by_source:
            by_source[s] = {}
        by_source[s][key] = data

    questions, _, _ = make_questions(args.num_questions, args.seed)

    summary = {
        "metadata": {
            "model": args.model,
            "scale": args.scale,
            "num_questions": args.num_questions,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "num_pairs": len(summary_pairs),
            "timestamp": datetime.now().isoformat(),
        },
        "questions": [{"question": q, "answer": a} for q, a in questions],
        "pairs": summary_pairs,
    }

    summary_file = Path(args.output_dir) / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    for s, pairs in sorted(by_source.items()):
        with open(gen_dir / f"source_{s}.json", "w") as f:
            json.dump(pairs, f)

    accs = [v["accuracy"] for v in summary_pairs.values()]
    print(f"\nMerged {len(summary_pairs)} pairs → {summary_file}")
    print(f"Accuracy: [{min(accs):.1%}, {max(accs):.1%}], mean={sum(accs)/len(accs):.1%}")
    print(f"Split into {len(by_source)} source files in {gen_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate steered text for all layer pairs",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--vectors-file", default="results/jacobian_map/merged.pt")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--num-questions", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--output-dir", default="results/jacobian_gen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.merge_only:
        merge(args)
        return

    # Get pairs from vectors file
    data = torch.load(args.vectors_file, map_location="cpu", weights_only=True)
    all_pairs = []
    for key in data["vectors"]:
        s, t = key.split("_")
        all_pairs.append((int(s), int(t)))
    all_pairs.sort()

    print(f"Model: {args.model}")
    print(f"Vectors: {args.vectors_file} ({len(all_pairs)} pairs)")
    print(f"Questions: {args.num_questions}, Samples: {args.num_samples}, Scale: {args.scale}")
    print(f"Batch per pair: {12 * args.num_questions} (k=12 × {args.num_questions} questions)")
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
