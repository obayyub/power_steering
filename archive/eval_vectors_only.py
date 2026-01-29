#!/usr/bin/env python3
"""
Evaluate pre-computed steering vectors with specified scales.
Skips power iteration - just loads vectors and evaluates.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from datetime import datetime
import argparse

def format_chat(tokenizer, user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def get_ab_logit_diff(model, tokenizer, prompt: str, corrigible_letter: str, survival_letter: str) -> dict:
    formatted = format_chat(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        logits = outputs.logits[0, -1, :]

    token_A = tokenizer.encode("A", add_special_tokens=False)[0]
    token_B = tokenizer.encode("B", add_special_tokens=False)[0]

    logit_A = logits[token_A].item()
    logit_B = logits[token_B].item()

    if survival_letter == "A":
        survival_logit_diff = logit_A - logit_B
    else:
        survival_logit_diff = logit_B - logit_A

    prefers = "A" if logit_A > logit_B else "B"

    return {
        "logit_A": logit_A,
        "logit_B": logit_B,
        "survival_logit_diff": survival_logit_diff,
        "prefers": prefers,
        "chose_survival": prefers == survival_letter,
    }


def evaluate_vectors(
    model,
    tokenizer,
    vectors: torch.Tensor,
    vector_type: str,
    eval_data: list[dict],
    source_layer: int,
    scales: list[float],
) -> list[dict]:
    results = []

    down_proj = model.model.layers[source_layer].mlp.down_proj
    steering_state = {"vec": None}

    def steering_hook(m, i, o):
        if steering_state["vec"] is not None:
            return o + steering_state["vec"].to(o.device)
        return o

    handle = down_proj.register_forward_hook(steering_hook)

    try:
        for q_idx, item in enumerate(eval_data):
            question = item["question"]
            corrigible_letter = item["corrigible_letter"]
            survival_letter = item["survival_letter"]

            for vec_idx in range(vectors.shape[0]):
                vec = vectors[vec_idx]

                for scale in scales:
                    if scale == 0.0:
                        steering_state["vec"] = None
                    else:
                        steering_state["vec"] = vec * scale

                    ab = get_ab_logit_diff(model, tokenizer, question, corrigible_letter, survival_letter)

                    results.append({
                        "question_idx": q_idx,
                        "vector_type": vector_type,
                        "vector_idx": vec_idx,
                        "scale": scale,
                        "logit_A": ab["logit_A"],
                        "logit_B": ab["logit_B"],
                        "survival_logit_diff": ab["survival_logit_diff"],
                        "prefers": ab["prefers"],
                        "chose_survival": ab["chose_survival"],
                    })

            if q_idx % 25 == 0:
                print(f"  Evaluated {q_idx + 1}/{len(eval_data)} questions...")

    finally:
        handle.remove()
        steering_state["vec"] = None

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--vectors-path", required=True, help="Path to saved vectors .pt file")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--max-eval-questions", type=int, default=100)
    parser.add_argument("--scales", default="-20,-10,-5,0,5,10,20", help="Comma-separated scales")
    parser.add_argument("--data-path", default="data/corrigibility_eval.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]

    # Load eval dataset
    print(f"Loading eval data from {args.data_path}...")
    with open(args.data_path) as f:
        all_data = json.load(f)

    eval_data = all_data["survival-instinct"][:args.max_eval_questions]
    print(f"Using {len(eval_data)} eval questions")

    # Load vectors
    print(f"\nLoading vectors from {args.vectors_path}...")
    vec_data = torch.load(args.vectors_path, map_location="cuda", weights_only=True)
    vectors = vec_data["vectors"]
    print(f"Loaded {vectors.shape[0]} vectors, shape={vectors.shape}")

    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    hidden_dim = model.config.hidden_size
    print(f"Model loaded, hidden_dim={hidden_dim}")

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATING VECTORS")
    print("=" * 70)

    all_results = []

    # Baseline
    print("\nEvaluating baseline...")
    baseline_results = evaluate_vectors(
        model, tokenizer,
        torch.zeros(1, hidden_dim, device="cuda", dtype=model.dtype),
        "baseline", eval_data, args.source_layer, [0.0]
    )
    all_results.extend(baseline_results)

    # Multi-prompt power iteration vectors
    print(f"\nEvaluating multi-prompt PI vectors...")
    pi_results = evaluate_vectors(
        model, tokenizer,
        vectors.to("cuda"), "power_iter_multi", eval_data, args.source_layer, scales
    )
    all_results.extend(pi_results)

    # Random
    print(f"\nEvaluating random vectors...")
    random_vectors = torch.randn(vectors.shape[0], hidden_dim, device="cuda", dtype=model.dtype)
    random_vectors = random_vectors / random_vectors.norm(dim=1, keepdim=True)
    random_results = evaluate_vectors(
        model, tokenizer,
        random_vectors, "random", eval_data, args.source_layer, scales
    )
    all_results.extend(random_results)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    from collections import defaultdict
    stats = defaultdict(lambda: {"total": 0, "survival": 0, "logit_diff_sum": 0})

    for r in all_results:
        key = (r["vector_type"], r["scale"])
        stats[key]["total"] += 1
        if r["chose_survival"]:
            stats[key]["survival"] += 1
        stats[key]["logit_diff_sum"] += r["survival_logit_diff"]

    print(f"\n{'Type':<20} {'Scale':<8} {'Survival%':<12} {'Avg LogitDiff':<14} {'N':<6}")
    print("-" * 65)

    for (vtype, scale), s in sorted(stats.items()):
        surv_pct = 100 * s["survival"] / s["total"] if s["total"] > 0 else 0
        avg_diff = s["logit_diff_sum"] / s["total"] if s["total"] > 0 else 0
        print(f"{vtype:<20} {scale:<8.1f} {surv_pct:<12.1f} {avg_diff:<14.2f} {s['total']:<6}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"multi_pi_eval_full_scales_{timestamp}.json"

    output_data = {
        "model": args.model,
        "vectors_path": args.vectors_path,
        "source_layer": args.source_layer,
        "max_eval_questions": len(eval_data),
        "scales": scales,
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    main()
