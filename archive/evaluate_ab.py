#!/usr/bin/env python3
"""
Evaluate steering vectors using A/B logit comparison on Anthropic corrigibility evals.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
import json
import argparse
from collections import defaultdict


def format_chat(tokenizer, user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def get_ab_logits(model, tokenizer, prompt: str, token_A: int, token_B: int) -> dict:
    """Get logits for A vs B tokens."""
    formatted = format_chat(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        logits = outputs.logits[0, -1, :]

    logit_A = logits[token_A].item()
    logit_B = logits[token_B].item()

    # Probabilities over just A and B
    ab_logits = torch.tensor([logit_A, logit_B])
    ab_probs = torch.softmax(ab_logits, dim=0)

    return {
        "logit_A": logit_A,
        "logit_B": logit_B,
        "diff": logit_A - logit_B,
        "prob_A": ab_probs[0].item(),
        "prob_B": ab_probs[1].item(),
        "prefers": "A" if logit_A > logit_B else "B",
    }


def evaluate_vectors(
    model,
    tokenizer,
    dataset: list[dict],
    vectors: torch.Tensor,
    vector_name: str,
    source_layer: int,
    scales: list[float],
    steering_state: dict,
) -> list[dict]:
    """Evaluate a set of vectors on the dataset."""
    # Get token IDs for A and B
    token_A = tokenizer.encode("A", add_special_tokens=False)[0]
    token_B = tokenizer.encode("B", add_special_tokens=False)[0]

    results = []

    for q_idx, item in enumerate(dataset):
        question = item["question"]
        corrigible_letter = item["corrigible_letter"]
        survival_letter = item["survival_letter"]

        for vec_idx in range(vectors.shape[0]):
            vec = vectors[vec_idx]

            for scale in scales:
                if scale == 0:
                    steering_state["vec"] = None
                else:
                    steering_state["vec"] = vec * scale

                ab = get_ab_logits(model, tokenizer, question, token_A, token_B)

                results.append({
                    "question_idx": q_idx,
                    "vector_type": vector_name,
                    "vector_idx": vec_idx,
                    "scale": scale,
                    "logit_A": ab["logit_A"],
                    "logit_B": ab["logit_B"],
                    "logit_diff": ab["diff"],
                    "prob_A": ab["prob_A"],
                    "prob_B": ab["prob_B"],
                    "prefers": ab["prefers"],
                    "corrigible_letter": corrigible_letter,
                    "survival_letter": survival_letter,
                    "chose_survival": ab["prefers"] == survival_letter,
                })

        if q_idx % 20 == 0:
            print(f"    Processed {q_idx+1}/{len(dataset)} questions...")

    return results


def print_summary(results: list[dict], dataset_name: str):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print(f"Results for: {dataset_name}")
    print("="*70)

    # Group by vector type and scale
    stats = defaultdict(lambda: {"total": 0, "survival": 0, "logit_diff_sum": 0})

    for r in results:
        key = (r["vector_type"], r["scale"])
        stats[key]["total"] += 1
        if r["chose_survival"]:
            stats[key]["survival"] += 1
        stats[key]["logit_diff_sum"] += r["logit_diff"]

    print(f"\n{'Type':<15} {'Scale':<8} {'Survival%':<12} {'Avg A-B diff':<14} {'N':<6}")
    print("-" * 60)

    for (vtype, scale), s in sorted(stats.items()):
        surv_pct = 100 * s["survival"] / s["total"] if s["total"] > 0 else 0
        avg_diff = s["logit_diff_sum"] / s["total"] if s["total"] > 0 else 0
        print(f"{vtype:<15} {scale:<8.1f} {surv_pct:<12.1f} {avg_diff:<14.2f} {s['total']:<6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--dataset", default="data/corrigibility_eval.json")
    parser.add_argument("--melbo-vectors", default=None, help="Path to MELBO vectors .pt file")
    parser.add_argument("--power-iter-vectors", default=None, help="Path to power iteration vectors .pt file")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--scales", default="-20,-10,-5,0,5,10,20", help="Comma-separated scales to test")
    parser.add_argument("--max-questions", type=int, default=100, help="Max questions per dataset")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset) as f:
        all_datasets = json.load(f)

    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    hidden_dim = model.config.hidden_size
    print(f"Model loaded. Hidden dim: {hidden_dim}")

    # Load vectors
    vector_sets = []

    if args.melbo_vectors:
        print(f"\nLoading MELBO vectors from {args.melbo_vectors}...")
        data = torch.load(args.melbo_vectors, map_location="cuda", weights_only=True)
        melbo_vecs = data["vectors"]
        if melbo_vecs.shape[1] != hidden_dim:
            print(f"  ERROR: MELBO dim {melbo_vecs.shape[1]} != model dim {hidden_dim}")
        else:
            print(f"  Loaded {melbo_vecs.shape[0]} MELBO vectors")
            vector_sets.append(("melbo", melbo_vecs))

    if args.power_iter_vectors:
        print(f"\nLoading power iteration vectors from {args.power_iter_vectors}...")
        data = torch.load(args.power_iter_vectors, map_location="cuda", weights_only=True)
        pi_vecs = data["vectors"]
        if pi_vecs.shape[1] != hidden_dim:
            print(f"  ERROR: Power iter dim {pi_vecs.shape[1]} != model dim {hidden_dim}")
        else:
            print(f"  Loaded {pi_vecs.shape[0]} power iteration vectors")
            vector_sets.append(("power_iter", pi_vecs))

    # Add random control vectors
    print("\nCreating random control vectors...")
    num_random = 4
    random_vecs = torch.randn(num_random, hidden_dim, device="cuda", dtype=model.dtype)
    random_vecs = random_vecs / random_vecs.norm(dim=1, keepdim=True)
    vector_sets.append(("random", random_vecs))

    # Setup steering hook
    down_proj = model.model.layers[args.source_layer].mlp.down_proj
    steering_state = {"vec": None}

    def steering_hook(m, i, o):
        if steering_state["vec"] is not None:
            return o + steering_state["vec"].to(o.device)
        return o

    handle = down_proj.register_forward_hook(steering_hook)

    all_results = {}

    try:
        for dataset_name, dataset in all_datasets.items():
            print(f"\n{'='*70}")
            print(f"Evaluating on: {dataset_name}")
            print("="*70)

            # Limit questions
            dataset = dataset[:args.max_questions]
            print(f"  Using {len(dataset)} questions")

            dataset_results = []

            for vec_name, vectors in vector_sets:
                print(f"\n  Testing {vec_name} vectors...")
                results = evaluate_vectors(
                    model, tokenizer, dataset, vectors, vec_name,
                    args.source_layer, scales, steering_state
                )
                dataset_results.extend(results)

            all_results[dataset_name] = dataset_results
            print_summary(dataset_results, dataset_name)

    finally:
        handle.remove()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ab_eval_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": args.model,
            "scales": scales,
            "results": all_results,
        }, f, indent=2)

    print(f"\n\nSaved results to {output_file}")


if __name__ == "__main__":
    main()
