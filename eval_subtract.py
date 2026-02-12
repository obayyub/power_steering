#!/usr/bin/env python3
"""
Subtraction test: steer with NEGATIVE scale to test causal relevance.

If subtracting a direction from a context where it's naturally active
hurts performance, the direction is causally required for the model's
natural computation — not just correlated.

Tests both vectors across all task types at positive and negative scales.

Usage:
  uv run python eval_subtract.py --vectors-file results/jacobian_map/merged.pt
"""

import json
import random
import re
import argparse
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_generalize import extract_answer, WORD_PROBLEMS, make_arithmetic_problems
from eval_probe import (
    FACTUAL_PROBLEMS, FACTUAL_PROMPT,
    REASONING_PROBLEMS, REASONING_PROMPT,
    ARITH_ORIGINAL_PROMPT,
    generate_steered, analyze_style,
)
from measure_projections import HARD_WORD_PROBLEMS

WORD_PROBLEM_PROMPT = """Q: Lisa has 27 stickers. She gets 15 more from her friend. How many stickers does Lisa have now?

A: 27 + 15 = 42. The answer is 42.

Q: A classroom has 30 desks with 2 pencils on each. How many pencils are there?

A: 30 * 2 = 60. The answer is 60.

Q: {question}

A:"""


def evaluate(model, tokenizer, problems, prompt_template, hooks_config, label,
             num_samples, max_new_tokens, temperature, seed, batch_size=16):
    """Run evaluation for one condition. Returns summary dict."""
    questions = [q for q, _ in problems]
    answers = [a for _, a in problems]
    prompts = [prompt_template.format(question=q) for q in questions]

    results = []
    total_correct = 0
    total = 0

    for si in range(num_samples):
        for bi in range(0, len(prompts), batch_size):
            batch_prompts = prompts[bi:bi + batch_size]
            batch_answers = answers[bi:bi + batch_size]
            batch_questions = questions[bi:bi + batch_size]

            texts = generate_steered(
                model, tokenizer, batch_prompts, hooks_config,
                max_new_tokens, temperature,
                seed=seed + si * 1000 + bi,
            )

            for qi, (text, true_ans, question) in enumerate(
                zip(texts, batch_answers, batch_questions)
            ):
                pred = extract_answer(text)
                correct = pred == true_ans
                if correct:
                    total_correct += 1
                total += 1
                style = analyze_style(text)
                results.append({
                    "q": bi + qi, "s": si,
                    "question": question, "answer": true_ans,
                    "text": text, "pred": pred, "correct": correct,
                    "style": style,
                })

    acc = total_correct / total if total > 0 else 0
    avg_len = sum(r["style"]["length"] for r in results) / len(results) if results else 0
    avg_cot = sum(r["style"]["num_patterns"] for r in results) / len(results) if results else 0

    print(f"    {label:>20s}: {total_correct}/{total} = {acc:.1%}  "
          f"len={avg_len:.0f}  cot={avg_cot:.1f}")

    return {
        "accuracy": acc,
        "correct": total_correct,
        "total": total,
        "avg_length": avg_len,
        "avg_cot_patterns": avg_cot,
        "generations": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--vectors-file", default="results/jacobian_map/merged.pt")
    parser.add_argument("--pairs", default="7_25:1,9_18:1")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/subtraction")
    args = parser.parse_args()

    # Parse pairs
    pair_specs = []
    for spec in args.pairs.split(","):
        pair_key, vec_idx = spec.strip().split(":")
        pair_specs.append((pair_key, int(vec_idx)))

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load vectors
    data = torch.load(args.vectors_file, map_location="cpu", weights_only=True)
    vectors_dict = data["vectors"]

    vec_info = []
    for pair_key, vec_idx in pair_specs:
        s, t = pair_key.split("_")
        vec = vectors_dict[pair_key][vec_idx]
        vec_info.append((f"({s},{t})v{vec_idx}", int(s), vec))
        print(f"  ({s},{t})v{vec_idx}: norm={vec.norm():.2f}")

    # Tasks
    arith_problems = make_arithmetic_problems(16, args.seed)

    tasks = [
        ("arithmetic", arith_problems["level1_basic"], ARITH_ORIGINAL_PROMPT),
        ("word_easy", WORD_PROBLEMS[:16], WORD_PROBLEM_PROMPT),
        ("word_hard", HARD_WORD_PROBLEMS[:16], WORD_PROBLEM_PROMPT),
        ("factual_qa", FACTUAL_PROBLEMS, FACTUAL_PROMPT),
        ("reasoning_qa", REASONING_PROBLEMS, REASONING_PROMPT),
    ]

    scales = [-10, -5, -3, 0, 3, 5, 10]

    output = {
        "config": {
            "model": args.model,
            "pairs": args.pairs,
            "scales": scales,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "tasks": {},
    }

    for task_name, problems, prompt in tasks:
        print(f"\n{'='*60}")
        print(f"  {task_name} ({len(problems)} problems)")
        print(f"{'='*60}")

        task_data = {"problems": [{"q": q, "a": a} for q, a in problems]}

        # Baseline (scale=0) once
        res = evaluate(
            model, tokenizer, problems, prompt, [], "baseline",
            args.num_samples, args.max_new_tokens, args.temperature,
            args.seed, args.batch_size,
        )
        task_data["baseline"] = {
            "accuracy": res["accuracy"],
            "avg_length": res["avg_length"],
            "avg_cot_patterns": res["avg_cot_patterns"],
        }

        # Each vector at each scale
        for vec_label, src, vec in vec_info:
            vec_results = {}
            for s in scales:
                if s == 0:
                    continue
                hooks = [(src, vec, float(s))]
                label = f"{vec_label} s={s:+d}"
                res = evaluate(
                    model, tokenizer, problems, prompt, hooks, label,
                    args.num_samples, args.max_new_tokens, args.temperature,
                    args.seed, args.batch_size,
                )
                vec_results[str(s)] = {
                    "accuracy": res["accuracy"],
                    "avg_length": res["avg_length"],
                    "avg_cot_patterns": res["avg_cot_patterns"],
                }
            task_data[vec_label] = vec_results

        output["tasks"][task_name] = task_data

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: accuracy by task × vector × scale")
    print("=" * 70)

    for task_name, task_data in output["tasks"].items():
        baseline = task_data["baseline"]["accuracy"]
        print(f"\n{task_name} (baseline={baseline:.1%}):")
        for vec_label, _, _ in vec_info:
            if vec_label not in task_data:
                continue
            parts = []
            for s in scales:
                if s == 0:
                    parts.append(f"s=0:{baseline:.0%}")
                else:
                    acc = task_data[vec_label][str(s)]["accuracy"]
                    delta = acc - baseline
                    parts.append(f"s={s:+d}:{acc:.0%}({delta:+.0%})")
            print(f"  {vec_label}: {', '.join(parts)}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"subtract_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
