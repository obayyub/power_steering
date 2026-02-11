#!/usr/bin/env python3
"""
Test whether CoT-inducing steering vectors generalize beyond the training task.

Tasks:
  1. Harder arithmetic (same format, bigger numbers / more steps)
  2. Simple word problems (hardcoded)

Usage:
  uv run python eval_generalize.py --vectors-file results/jacobian_map/merged.pt
  uv run python eval_generalize.py --vectors-file results/jacobian_map/merged.pt --pairs 7_25:1,9_18:1
"""

import json
import random
import re
import argparse
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------

def gen_arithmetic_level1(rng):
    """Same as training: a=x+y, b=w+z. What is a*b? (single digit adds)"""
    x, y = rng.randint(1, 9), rng.randint(1, 9)
    w, z = rng.randint(1, 9), rng.randint(1, 9)
    q = f"a={x}+{y}, b={w}+{z}. What is a*b?"
    return q, (x + y) * (w + z)


def gen_arithmetic_level2(rng):
    """Bigger numbers: double-digit additions."""
    x, y = rng.randint(10, 50), rng.randint(10, 50)
    w, z = rng.randint(10, 50), rng.randint(10, 50)
    q = f"a={x}+{y}, b={w}+{z}. What is a*b?"
    return q, (x + y) * (w + z)


def gen_arithmetic_level3(rng):
    """Three variables: a=x+y, b=w+z, c=a+b. What is c*2?"""
    x, y = rng.randint(1, 9), rng.randint(1, 9)
    w, z = rng.randint(1, 9), rng.randint(1, 9)
    a, b = x + y, w + z
    q = f"a={x}+{y}, b={w}+{z}, c=a+b. What is c*2?"
    return q, (a + b) * 2


def gen_arithmetic_level4(rng):
    """Chained: a=x+y, b=a*z, c=b-w. What is c?"""
    x, y = rng.randint(2, 8), rng.randint(2, 8)
    z = rng.randint(2, 5)
    a = x + y
    b = a * z
    w = rng.randint(1, b - 1)
    q = f"a={x}+{y}, b=a*{z}, c=b-{w}. What is c?"
    return q, b - w


ARITHMETIC_LEVELS = {
    "level1_basic": gen_arithmetic_level1,
    "level2_big_numbers": gen_arithmetic_level2,
    "level3_three_vars": gen_arithmetic_level3,
    "level4_chained": gen_arithmetic_level4,
}


def make_arithmetic_problems(num_per_level, seed):
    """Generate arithmetic problems at each difficulty level."""
    rng = random.Random(seed)
    problems = {}
    for name, gen_fn in ARITHMETIC_LEVELS.items():
        seen = set()
        level_problems = []
        while len(level_problems) < num_per_level:
            q, a = gen_fn(rng)
            if q not in seen:
                seen.add(q)
                level_problems.append((q, a))
        problems[name] = level_problems
    return problems


# ---------------------------------------------------------------------------
# Hardcoded word problems
# ---------------------------------------------------------------------------

WORD_PROBLEMS = [
    # Simple addition
    ("Sarah has 15 apples. She buys 23 more. How many apples does she have?", 38),
    ("A store has 45 red shirts and 32 blue shirts. How many shirts are there in total?", 77),
    ("There are 28 boys and 35 girls in a school. How many students are there?", 63),
    # Simple subtraction
    ("Tom had 50 dollars. He spent 18 dollars on lunch. How much money does he have left?", 32),
    ("There were 83 students in the hall. 27 left early. How many remained?", 56),
    ("A tank has 200 liters of water. 75 liters leak out. How many liters are left?", 125),
    # Simple multiplication
    ("Each box contains 12 cookies. If there are 7 boxes, how many cookies are there?", 84),
    ("A car travels 55 miles per hour. How far does it travel in 3 hours?", 165),
    ("There are 8 rows of chairs. Each row has 15 chairs. How many chairs are there?", 120),
    ("A pack of pencils costs 4 dollars. How much do 9 packs cost?", 36),
    # Simple division
    ("48 students need to be split into groups of 6. How many groups are there?", 8),
    ("A rope is 96 inches long. It is cut into 8 equal pieces. How long is each piece?", 12),
    # Two-step problems
    ("Maria has 24 dollars. She earns 16 more, then spends 10 on a book. How much does she have?", 30),
    ("Jake bought 5 packs of cards with 12 cards each. He gave 15 cards away. How many does he have?", 45),
    ("A farmer picks 36 apples in the morning and 28 in the afternoon. He puts them in bags of 8. How many bags does he need?", 8),
    ("Tom has 3 boxes of 20 marbles. He loses 7 marbles. How many does he have?", 53),
    ("A store sells 14 shirts on Monday and 22 on Tuesday. Each shirt costs 5 dollars. How much money did they make in total?", 180),
    # Comparison
    ("Alice has 34 marbles. Bob has 19 more than Alice. How many marbles does Bob have?", 53),
    ("A book costs 25 dollars. A pen costs 3 dollars. How much more does the book cost?", 22),
    ("Lisa ran 8 miles. Karen ran 3 times as far. How many miles did Karen run?", 24),
    # Slightly harder multi-step
    ("A school has 4 classes. Each class has 25 students. Each student brings 2 pencils. How many pencils are there?", 200),
    ("Mike earns 12 dollars per hour. He works 8 hours on Monday and 6 hours on Tuesday. How much did he earn in total?", 168),
    ("There are 5 bags with 10 red balls and 6 blue balls in each bag. How many balls are there in total?", 80),
    ("A train travels 60 miles in the first hour and 80 miles in the second hour. What is the total distance?", 140),
    ("Emma has 100 stickers. She gives 8 stickers to each of her 7 friends. How many stickers does she have left?", 44),
]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

ARITHMETIC_PROMPT = """Q: a=6+2, b=3+7. What is a*b?

A: The answer is 80.

Q: {question}

A:"""

WORD_PROBLEM_PROMPT = """Q: Lisa has 27 stickers. She gets 15 more from her friend. How many stickers does Lisa have now?

A: 27 + 15 = 42. The answer is 42.

Q: A classroom has 30 desks with 2 pencils on each. How many pencils are there?

A: 30 * 2 = 60. The answer is 60.

Q: {question}

A:"""


def make_prompt(question, task_type):
    if task_type == "word":
        return WORD_PROBLEM_PROMPT.format(question=question)
    else:
        return ARITHMETIC_PROMPT.format(question=question)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text):
    """Extract numerical answer from model output."""
    text_lower = text.lower()

    # "The answer is X"
    match = re.search(r"answer is\s*(\d+)", text_lower)
    if match:
        return int(match.group(1))

    # "= X" at end of a calculation
    match = re.search(r"=\s*(\d+)", text)
    if match:
        # Return the last "= X" pattern
        matches = re.findall(r"=\s*(\d+)", text)
        if matches:
            return int(matches[-1])

    # Last number
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return int(numbers[-1])
    return None


# ---------------------------------------------------------------------------
# Generation with steering
# ---------------------------------------------------------------------------

def generate_steered(model, tokenizer, prompts, source_layer, vector, scale,
                     max_new_tokens, temperature, seed):
    """Generate with optional steering. vector=None for baseline."""
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    handle = None
    if vector is not None:
        down_proj = model.model.layers[source_layer].mlp.down_proj
        sv = (vector * scale).to(device, dtype=model.dtype)

        def hook(m, inp, out):
            return out + sv

        handle = down_proj.register_forward_hook(hook)

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
        if handle:
            handle.remove()

    texts = []
    for i in range(len(prompts)):
        texts.append(tokenizer.decode(outputs[i, input_len:], skip_special_tokens=True))
    return texts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_task(model, tokenizer, problems, task_type, vectors_config,
                  scale, num_samples, max_new_tokens, temperature, seed, batch_size=16):
    """
    Evaluate baseline + each steering vector on a set of problems.

    vectors_config: list of (label, source_layer, vector_tensor) or (label, None, None) for baseline
    """
    questions = [q for q, _ in problems]
    answers = [a for _, a in problems]
    prompts = [make_prompt(q, task_type) for q in questions]

    all_results = {}

    for label, source_layer, vector in vectors_config:
        results = []
        total_correct = 0
        total = 0

        for si in range(num_samples):
            # Process in batches
            for bi in range(0, len(prompts), batch_size):
                batch_prompts = prompts[bi:bi + batch_size]
                batch_answers = answers[bi:bi + batch_size]
                batch_questions = questions[bi:bi + batch_size]

                texts = generate_steered(
                    model, tokenizer, batch_prompts,
                    source_layer, vector, scale,
                    max_new_tokens, temperature,
                    seed=seed + si * 1000 + bi,
                )

                for qi, (text, true_ans, question) in enumerate(zip(texts, batch_answers, batch_questions)):
                    pred = extract_answer(text)
                    correct = pred == true_ans
                    if correct:
                        total_correct += 1
                    total += 1
                    results.append({
                        "q": bi + qi, "s": si,
                        "question": question, "answer": true_ans,
                        "text": text, "pred": pred, "correct": correct,
                    })

        acc = total_correct / total if total > 0 else 0
        all_results[label] = {"results": results, "accuracy": acc, "correct": total_correct, "total": total}
        print(f"    {label}: {total_correct}/{total} = {acc:.1%}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test steering vector generalization")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--vectors-file", default="results/jacobian_map/merged.pt")
    parser.add_argument("--pairs", default="7_25:1,9_18:1",
                        help="Comma-separated pair:vector_idx to test (e.g. 7_25:1,9_18:1)")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--num-per-level", type=int, default=16,
                        help="Number of arithmetic problems per difficulty level")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/generalization")
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
    print(f"Loading vectors from {args.vectors_file}...")
    data = torch.load(args.vectors_file, map_location="cpu", weights_only=True)
    vectors_dict = data["vectors"]

    vectors_config = [("baseline", None, None)]
    for pair_key, vec_idx in pair_specs:
        s, t = pair_key.split("_")
        vecs = vectors_dict[pair_key]  # [k, H]
        vec = vecs[vec_idx]  # [H]
        label = f"({s},{t})v{vec_idx}"
        vectors_config.append((label, int(s), vec))
        print(f"  {label}: norm={vec.norm():.2f}")

    # Generate arithmetic problems
    print(f"\nGenerating arithmetic problems ({args.num_per_level} per level)...")
    arith_problems = make_arithmetic_problems(args.num_per_level, args.seed)

    # Build output
    output = {
        "config": {
            "model": args.model,
            "vectors_file": args.vectors_file,
            "pairs": args.pairs,
            "scale": args.scale,
            "num_per_level": args.num_per_level,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "tasks": {},
    }

    # Evaluate arithmetic levels
    for level_name, problems in arith_problems.items():
        print(f"\n  === {level_name} ({len(problems)} problems) ===")
        task_results = evaluate_task(
            model, tokenizer, problems, "arithmetic", vectors_config,
            args.scale, args.num_samples, args.max_new_tokens, args.temperature,
            args.seed, args.batch_size,
        )
        output["tasks"][level_name] = {
            "type": "arithmetic",
            "problems": [{"question": q, "answer": a} for q, a in problems],
            "results": {k: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"]}
                        for k, v in task_results.items()},
            "generations": {k: v["results"] for k, v in task_results.items()},
        }

    # Evaluate word problems
    print(f"\n  === word_problems ({len(WORD_PROBLEMS)} problems) ===")
    task_results = evaluate_task(
        model, tokenizer, WORD_PROBLEMS, "word", vectors_config,
        args.scale, args.num_samples, args.max_new_tokens, args.temperature,
        args.seed, args.batch_size,
    )
    output["tasks"]["word_problems"] = {
        "type": "word",
        "problems": [{"question": q, "answer": a} for q, a in WORD_PROBLEMS],
        "results": {k: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"]}
                    for k, v in task_results.items()},
        "generations": {k: v["results"] for k, v in task_results.items()},
    }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    labels = [vc[0] for vc in vectors_config]
    header = f"{'task':<25}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))
    for task_name, task_data in output["tasks"].items():
        row = f"{task_name:<25}"
        for label in labels:
            acc = task_data["results"][label]["accuracy"]
            row += f"{acc:>14.1%} "
        print(row)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"generalize_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
