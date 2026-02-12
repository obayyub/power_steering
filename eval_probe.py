#!/usr/bin/env python3
"""
Probing experiments for the two-mechanism hypothesis.

Experiment 1: Non-math structured tasks
  - Factual Q&A (trivia with numeric answers)
  - Contextual reasoning (multi-hop, temporal, comparison — some involve basic computation)
  - Measures accuracy, output length, and CoT pattern frequency

Experiment 2: Alternative arithmetic formats
  - Original: "a=5+6, b=2+7. What is a*b?"
  - Plain expression: "What is (5+6) * (2+7)?"
  - Direct (pre-computed sums): "What is 11 * 9?"

Experiment 3: Scale sensitivity
  - Sweep scale [1,2,3,5,7,10,15,20] on original arithmetic format

All experiments test both vectors + combined (both at scale=1).

Usage:
  uv run python eval_probe.py --vectors-file results/jacobian_map/merged.pt
"""

import json
import random
import re
import argparse
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_generalize import extract_answer


# ---------------------------------------------------------------------------
# Experiment 1: Non-math structured tasks
# ---------------------------------------------------------------------------

FACTUAL_PROMPT = """Q: What is the largest planet in our solar system?

A: The answer is Jupiter.

Q: How many continents are on Earth?

A: The answer is 7.

Q: {question}

A:"""

FACTUAL_PROBLEMS = [
    ("How many sides does a hexagon have?", 6),
    ("How many planets are in our solar system?", 8),
    ("How many legs does an octopus have?", 8),
    ("How many strings does a standard guitar have?", 6),
    ("How many bones are in the adult human body?", 206),
    ("How many teeth does a normal adult human have?", 32),
    ("How many states are in the United States?", 50),
    ("How many days are in the month of June?", 30),
    ("How many degrees are in a right angle?", 90),
    ("How many minutes are in an hour?", 60),
    ("How many cards are in a standard deck?", 52),
    ("How many colors are in a rainbow?", 7),
    ("How many keys are on a standard piano?", 88),
    ("How many letters are in the English alphabet?", 26),
    ("How many hours are in a day?", 24),
    ("How many weeks are in a year?", 52),
]

REASONING_PROMPT = """Q: Dogs are mammals. All mammals are warm-blooded. Are dogs warm-blooded? Answer with 1 for yes or 0 for no.

A: Dogs are mammals, and all mammals are warm-blooded, so dogs must be warm-blooded. The answer is 1.

Q: {question}

A:"""

REASONING_PROBLEMS = [
    # Multi-hop (non-arithmetic)
    ("Paris is in France. France is in Europe. Tokyo is in Japan. Japan is in Asia. Is Paris in Asia? Answer 1 for yes, 0 for no.", 0),
    ("All squares are rectangles. All rectangles have 4 sides. How many sides does a square have?", 4),
    ("Earth is the 3rd planet from the Sun. Mars is the 4th. How many planets are between Earth and Mars?", 0),
    ("January is month 1, February is month 2, and so on. What month number is May?", 5),
    ("Sunday is day 1 of the week. What day number is Wednesday?", 4),
    ("If today is the 10th and a meeting is in 5 days, what date is the meeting?", 15),
    ("A year starts in January (month 1). September is how many months after January?", 8),
    ("In the sequence 2, 4, 6, 8, what is the next number?", 10),
    # These involve basic computation embedded in context
    ("A century has 100 years. A millennium has 10 centuries. How many years are in a millennium?", 1000),
    ("A triangle has 3 sides. A square has 4 sides. How many sides do they have together?", 7),
    ("Mount Everest is 8849 meters. K2 is 8611 meters. How many meters taller is Everest?", 238),
    ("A baker has 3 dozen eggs. How many eggs does he have?", 36),
    ("12 months in a year and 4 seasons. How many months per season?", 3),
    ("A decade is 10 years. How many decades are in 50 years?", 5),
    ("A book has 200 pages. You read 25 pages per day. How many days to finish?", 8),
    ("A store sells apples for $3 each. You buy 7 apples. How much do you pay?", 21),
]


# ---------------------------------------------------------------------------
# Experiment 2: Alternative arithmetic formats
# ---------------------------------------------------------------------------

ARITH_ORIGINAL_PROMPT = """Q: a=6+2, b=3+7. What is a*b?

A: The answer is 80.

Q: {question}

A:"""

ARITH_PLAIN_PROMPT = """Q: What is (6+2) * (3+7)?

A: The answer is 80.

Q: {question}

A:"""

ARITH_DIRECT_PROMPT = """Q: What is 8 * 10?

A: The answer is 80.

Q: {question}

A:"""


def make_probe_arithmetic(n, seed):
    """Generate equivalent arithmetic problems in three formats."""
    rng = random.Random(seed)
    problems = {"original": [], "plain": [], "direct": []}

    for _ in range(n):
        x, y = rng.randint(1, 9), rng.randint(1, 9)
        w, z = rng.randint(1, 9), rng.randint(1, 9)
        answer = (x + y) * (w + z)

        problems["original"].append((f"a={x}+{y}, b={w}+{z}. What is a*b?", answer))
        problems["plain"].append((f"What is ({x}+{y}) * ({w}+{z})?", answer))
        problems["direct"].append((f"What is {x+y} * {w+z}?", answer))

    return problems


# ---------------------------------------------------------------------------
# Generation with multi-layer steering
# ---------------------------------------------------------------------------

def generate_steered(model, tokenizer, prompts, hooks_config,
                     max_new_tokens, temperature, seed):
    """
    Generate with optional multi-layer steering.

    hooks_config: list of (source_layer, vector, scale) tuples, or [] for baseline.
    """
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    handles = []
    for source_layer, vector, scale in hooks_config:
        down_proj = model.model.layers[source_layer].mlp.down_proj
        sv = (vector * scale).to(device, dtype=model.dtype)

        def make_hook(sv_):
            def hook(m, inp, out):
                return out + sv_
            return hook

        handles.append(down_proj.register_forward_hook(make_hook(sv)))

    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
            )
    finally:
        for h in handles:
            h.remove()

    texts = []
    for i in range(len(prompts)):
        texts.append(tokenizer.decode(outputs[i, input_len:], skip_special_tokens=True))
    return texts


# ---------------------------------------------------------------------------
# Style analysis
# ---------------------------------------------------------------------------

COT_PATTERNS = [
    (r"step[- ]by[- ]step", "step_by_step"),
    (r"let me|let's", "let_me"),
    (r"first|then|next|finally", "sequencing"),
    (r"\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+", "calculation"),
    (r"therefore|thus|so the answer", "conclusion"),
    (r"think|reason|consider", "thinking"),
]


def analyze_style(text):
    """Detect CoT-like patterns in output."""
    text_lower = text.lower()
    patterns_found = []
    for pattern, name in COT_PATTERNS:
        if re.search(pattern, text_lower):
            patterns_found.append(name)
    return {
        "length": len(text),
        "num_lines": text.count("\n") + 1,
        "patterns": patterns_found,
        "num_patterns": len(patterns_found),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_task(model, tokenizer, problems, prompt_template, vectors_config,
                  num_samples, max_new_tokens, temperature, seed, batch_size=16):
    """
    Evaluate baseline + each steering config on a set of problems.

    vectors_config: list of (label, hooks_config) where hooks_config is
                    [(source_layer, vector, scale), ...] or [] for baseline.
    """
    questions = [q for q, _ in problems]
    answers = [a for _, a in problems]
    prompts = [prompt_template.format(question=q) for q in questions]

    all_results = {}
    for label, hooks_config in vectors_config:
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
        avg_patterns = sum(r["style"]["num_patterns"] for r in results) / len(results) if results else 0
        pattern_counts = {}
        for r in results:
            for p in r["style"]["patterns"]:
                pattern_counts[p] = pattern_counts.get(p, 0) + 1

        all_results[label] = {
            "results": results,
            "accuracy": acc,
            "correct": total_correct,
            "total": total,
            "avg_length": avg_len,
            "avg_cot_patterns": avg_patterns,
            "pattern_counts": pattern_counts,
        }
        print(f"    {label}: {total_correct}/{total} = {acc:.1%}  "
              f"avg_len={avg_len:.0f}  cot={avg_patterns:.1f}")

    return all_results


def compact_results(task_results):
    """Extract summary stats (no generations) for JSON output."""
    return {k: {
        "accuracy": v["accuracy"],
        "correct": v["correct"],
        "total": v["total"],
        "avg_length": v["avg_length"],
        "avg_cot_patterns": v["avg_cot_patterns"],
        "pattern_counts": v["pattern_counts"],
    } for k, v in task_results.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Probing experiments for two-mechanism hypothesis")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--vectors-file", default="results/jacobian_map/merged.pt")
    parser.add_argument("--pairs", default="7_25:1,9_18:1")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--num-problems", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/probing")
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

    vec_info = []  # (label, source_layer, vector)
    for pair_key, vec_idx in pair_specs:
        s, t = pair_key.split("_")
        vec = vectors_dict[pair_key][vec_idx]
        vec_info.append((f"({s},{t})v{vec_idx}", int(s), vec))
        print(f"  ({s},{t})v{vec_idx}: norm={vec.norm():.2f}")

    # Build steering configs: baseline + each vector + combined at scale=1
    vecs_config = [("baseline", [])]
    for label, src, vec in vec_info:
        vecs_config.append((label, [(src, vec, args.scale)]))
    combined_hooks = [(src, vec, 1.0) for _, src, vec in vec_info]
    vecs_config.append(("combined_s1", combined_hooks))

    output = {
        "config": {
            "model": args.model,
            "pairs": args.pairs,
            "scale": args.scale,
            "num_problems": args.num_problems,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "experiments": {},
    }

    # -----------------------------------------------------------------------
    # Experiment 1: Non-math structured tasks
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Non-math structured tasks")
    print("=" * 70)

    for task_name, problems, prompt in [
        ("factual_qa", FACTUAL_PROBLEMS, FACTUAL_PROMPT),
        ("reasoning_qa", REASONING_PROBLEMS, REASONING_PROMPT),
    ]:
        print(f"\n  --- {task_name} ({len(problems)} problems) ---")
        task_results = evaluate_task(
            model, tokenizer, problems, prompt, vecs_config,
            args.num_samples, args.max_new_tokens, args.temperature,
            args.seed, args.batch_size,
        )
        output["experiments"][task_name] = {
            "problems": [{"question": q, "answer": a} for q, a in problems],
            "results": compact_results(task_results),
            "generations": {k: v["results"] for k, v in task_results.items()},
        }

    # -----------------------------------------------------------------------
    # Experiment 2: Alternative arithmetic formats
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Alternative arithmetic formats")
    print("=" * 70)

    arith_problems = make_probe_arithmetic(args.num_problems, args.seed)

    for fmt_name, prompt in [
        ("original", ARITH_ORIGINAL_PROMPT),
        ("plain", ARITH_PLAIN_PROMPT),
        ("direct", ARITH_DIRECT_PROMPT),
    ]:
        exp_key = f"arithmetic_{fmt_name}"
        problems = arith_problems[fmt_name]
        print(f"\n  --- {exp_key} ({len(problems)} problems) ---")
        task_results = evaluate_task(
            model, tokenizer, problems, prompt, vecs_config,
            args.num_samples, args.max_new_tokens, args.temperature,
            args.seed, args.batch_size,
        )
        output["experiments"][exp_key] = {
            "problems": [{"question": q, "answer": a} for q, a in problems],
            "results": compact_results(task_results),
            "generations": {k: v["results"] for k, v in task_results.items()},
        }

    # -----------------------------------------------------------------------
    # Experiment 3: Scale sensitivity
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Scale sensitivity")
    print("=" * 70)

    scales = [1, 2, 3, 5, 7, 10, 15, 20]
    scale_problems = arith_problems["original"]

    # Baseline once
    print("\n  --- baseline ---")
    baseline_res = evaluate_task(
        model, tokenizer, scale_problems, ARITH_ORIGINAL_PROMPT,
        [("baseline", [])],
        args.num_samples, args.max_new_tokens, args.temperature,
        args.seed, args.batch_size,
    )
    baseline_acc = baseline_res["baseline"]["accuracy"]

    scale_results = {"baseline_accuracy": baseline_acc}
    for label, src, vec in vec_info:
        print(f"\n  --- {label} ---")
        vec_scale_data = {}
        for s in scales:
            scale_config = [(f"s{s}", [(src, vec, float(s))])]
            res = evaluate_task(
                model, tokenizer, scale_problems, ARITH_ORIGINAL_PROMPT,
                scale_config,
                args.num_samples, args.max_new_tokens, args.temperature,
                args.seed, args.batch_size,
            )
            r = res[f"s{s}"]
            vec_scale_data[str(s)] = {
                "accuracy": r["accuracy"],
                "avg_length": r["avg_length"],
                "avg_cot_patterns": r["avg_cot_patterns"],
            }
        scale_results[label] = vec_scale_data

    output["experiments"]["scale_sensitivity"] = {
        "scales": scales,
        "problems": [{"question": q, "answer": a} for q, a in scale_problems],
        "results": scale_results,
    }

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    labels = [vc[0] for vc in vecs_config]
    header = f"{'task':<25}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))

    for exp_name, exp_data in output["experiments"].items():
        if exp_name == "scale_sensitivity":
            continue
        row = f"{exp_name:<25}"
        for label in labels:
            acc = exp_data["results"][label]["accuracy"]
            row += f"{acc:>14.1%} "
        print(row)

    print(f"\nScale sensitivity (baseline={baseline_acc:.1%}):")
    for label in scale_results:
        if label == "baseline_accuracy":
            continue
        accs = [f"s{s}={scale_results[label][str(s)]['accuracy']:.0%}" for s in scales]
        print(f"  {label}: {', '.join(accs)}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"probe_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
