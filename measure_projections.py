#!/usr/bin/env python3
"""
Measure how much the steering vector direction is naturally activated
in the unsteered model across different prompt types.

Hypothesis: word problems (where model already does CoT) naturally activate
the steering direction more than arithmetic prompts (where model fails).

Usage:
  uv run python measure_projections.py --vectors-file results/jacobian_map/merged.pt
"""

import json
import random
import torch
import argparse
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_generalize import (
    make_arithmetic_problems, WORD_PROBLEMS,
    ARITHMETIC_PROMPT, WORD_PROBLEM_PROMPT,
)


CONTROL_PROMPT = """Q: What is the largest ocean on Earth?

A: The largest ocean is the Pacific Ocean.

Q: {question}

A:"""

CONTROL_PROBLEMS = [
    "What is the capital of France?",
    "What color is the sky on a clear day?",
    "How many legs does a spider have?",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "What planet is closest to the Sun?",
    "What is the main ingredient in bread?",
    "How many continents are there?",
    "What language is spoken in Brazil?",
    "What is the tallest mountain in the world?",
    "What season comes after summer?",
    "What animal is known as the king of the jungle?",
    "What is the chemical symbol for gold?",
    "How many days are in a leap year?",
    "What is the largest mammal on Earth?",
    "What year did World War II end?",
]

HARD_WORD_PROBLEMS = [
    # 3+ step problems (GSM8K-style)
    ("Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 to bake muffins. She sells the rest for $2 each. How much does she make per day?", 18),
    ("A merchant has 120 apples. He sells 40% on Monday. On Tuesday he sells half of what's left. How many apples remain?", 36),
    ("Tom buys 3 shirts at $15 each and 2 pants at $25 each. He has a $10 coupon. How much does he pay?", 85),
    ("A train travels at 60 mph for 2 hours, then at 40 mph for 3 hours. What is the total distance?", 240),
    ("A bakery makes 200 cookies. They put 8 cookies in each box. Each box sells for $6. How much money do they make if they sell all the boxes?", 150),
    ("Sam has $50. He buys 3 books at $8 each. Then he earns $15 mowing a lawn. How much money does he have now?", 41),
    ("A school has 5 classes with 30 students each. Each student needs 3 notebooks. Notebooks cost $2 each. What is the total cost?", 900),
    ("Maria runs 3 miles every day for a week. On the weekend she runs 5 miles each day. How many miles does she run in total?", 25),
    ("A factory produces 150 toys per hour. It runs for 8 hours. 10% of the toys fail quality control. How many good toys are produced?", 1080),
    ("John buys 4 pizzas for $12 each. He splits the cost equally with 3 friends. How much does each person pay?", 12),
    ("A garden has 6 rows of flowers. Each row has 12 flowers. Half the flowers are red and the rest are yellow. How many yellow flowers are there?", 36),
    ("A bus holds 45 passengers. A school needs to transport 320 students. How many buses are needed? Round up.", 8),
    ("Lisa saves $15 per week. After 8 weeks she buys a $75 jacket. How much money does she have left?", 45),
    ("A recipe needs 3 cups of flour for 24 cookies. How many cups of flour are needed for 40 cookies?", 5),
    ("Mark drives 180 miles using 6 gallons of gas. Gas costs $4 per gallon. What is the fuel cost per mile?", 0),
    ("A store buys 50 shirts for $10 each and sells them for $18 each. What is the total profit?", 400),
    ("Three friends split a $84 dinner bill equally, plus each leaves a $5 tip. How much does each person pay in total?", 33),
    ("A pool is filled at 5 gallons per minute. It holds 600 gallons. How many hours does it take to fill?", 2),
    ("Amy has twice as many stickers as Ben. Ben has 3 times as many as Cal. Cal has 8 stickers. How many does Amy have?", 48),
    ("A movie theater charges $12 for adults and $8 for children. A group of 3 adults and 5 children goes. What is the total cost?", 76),
]


def measure_projections(model, tokenizer, prompts, source_layer, steering_vec, num_random=50, seed=42):
    """
    For each prompt, measure:
    - Projection of MLP down_proj output onto steering vector (last token)
    - Projection onto random vectors (control)
    - Activation norm

    Returns list of dicts with measurements.
    """
    device = next(model.parameters()).device
    H = steering_vec.shape[0]

    # Normalize steering vector
    sv_norm = steering_vec / steering_vec.norm()
    sv_norm = sv_norm.to(device, dtype=model.dtype)

    # Random control vectors
    rng = torch.Generator().manual_seed(seed)
    random_vecs = torch.randn(num_random, H, generator=rng)
    random_vecs = (random_vecs / random_vecs.norm(dim=1, keepdim=True)).to(device, dtype=model.dtype)

    # Hook to capture MLP down_proj output
    captured = {}
    down_proj = model.model.layers[source_layer].mlp.down_proj

    def hook(m, inp, out):
        captured["out"] = out.detach()

    handle = down_proj.register_forward_hook(hook)

    results = []
    try:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)

            # Last token activation
            act = captured["out"][0, -1, :]  # [H]
            act_norm = act.norm().item()

            # Projection onto steering vector
            steer_proj = (act @ sv_norm).item()

            # Projections onto random vectors
            rand_projs = (random_vecs @ act).tolist()  # [num_random]

            # Also measure across all token positions
            all_acts = captured["out"][0]  # [seq, H]
            steer_proj_mean = (all_acts @ sv_norm).mean().item()
            steer_proj_max = (all_acts @ sv_norm).max().item()

            results.append({
                "steer_proj": steer_proj,
                "steer_proj_mean": steer_proj_mean,
                "steer_proj_max": steer_proj_max,
                "rand_proj_mean": sum(rand_projs) / len(rand_projs),
                "rand_proj_std": (sum((r - sum(rand_projs)/len(rand_projs))**2 for r in rand_projs) / len(rand_projs)) ** 0.5,
                "rand_proj_abs_mean": sum(abs(r) for r in rand_projs) / len(rand_projs),
                "act_norm": act_norm,
                "seq_len": inputs["input_ids"].shape[1],
            })
    finally:
        handle.remove()

    return results


def summarize(results, label):
    """Print summary stats for a group of results."""
    n = len(results)
    steer = [r["steer_proj"] for r in results]
    steer_mean_tok = [r["steer_proj_mean"] for r in results]
    rand_abs = [r["rand_proj_abs_mean"] for r in results]
    norms = [r["act_norm"] for r in results]

    mean_s = sum(steer) / n
    mean_s_abs = sum(abs(s) for s in steer) / n
    mean_r = sum(rand_abs) / n
    mean_norm = sum(norms) / n

    print(f"  {label} (n={n}):")
    print(f"    steering proj (last tok):  mean={mean_s:+.2f}  |mean|={mean_s_abs:.2f}")
    print(f"    steering proj (all toks):  mean={sum(steer_mean_tok)/n:+.2f}")
    print(f"    random proj |mean|:        {mean_r:.2f}")
    print(f"    ratio (|steer|/|random|):  {mean_s_abs/mean_r:.2f}x")
    print(f"    activation norm:           {mean_norm:.1f}")
    return {
        "label": label, "n": n,
        "steer_proj_mean": mean_s,
        "steer_proj_abs_mean": mean_s_abs,
        "steer_proj_all_toks_mean": sum(steer_mean_tok) / n,
        "rand_proj_abs_mean": mean_r,
        "ratio": mean_s_abs / mean_r if mean_r > 0 else 0,
        "act_norm": mean_norm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--vectors-file", default="results/jacobian_map/merged.pt")
    parser.add_argument("--pairs", default="7_25:1,9_18:1")
    parser.add_argument("--num-per-level", type=int, default=16)
    parser.add_argument("--num-random", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/projections")
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

    # Build prompts
    arith_problems = make_arithmetic_problems(args.num_per_level, args.seed)
    arith_prompts_l1 = [ARITHMETIC_PROMPT.format(question=q) for q, _ in arith_problems["level1_basic"]]
    arith_prompts_l4 = [ARITHMETIC_PROMPT.format(question=q) for q, _ in arith_problems["level4_chained"]]
    word_prompts_easy = [WORD_PROBLEM_PROMPT.format(question=q) for q, _ in WORD_PROBLEMS]
    word_prompts_hard = [WORD_PROBLEM_PROMPT.format(question=q) for q, _ in HARD_WORD_PROBLEMS]
    control_prompts = [CONTROL_PROMPT.format(question=q) for q in CONTROL_PROBLEMS]

    prompt_groups = [
        ("arithmetic_l1", arith_prompts_l1),
        ("arithmetic_l4", arith_prompts_l4),
        ("word_easy", word_prompts_easy),
        ("word_hard", word_prompts_hard),
        ("control_non_math", control_prompts),
    ]

    # Build vector list: requested pairs + bad vectors for comparison
    vec_specs = []
    for pair_key, vec_idx in pair_specs:
        s, t = pair_key.split("_")
        source_layer = int(s)
        vec = vectors_dict[pair_key][vec_idx]
        label = f"({s},{t})v{vec_idx}"
        vec_specs.append((label, source_layer, vec))
        # Also add a bad vector from same pair (v0 if good is v1, v3 if good is v0)
        bad_idx = 0 if vec_idx != 0 else 3
        bad_vec = vectors_dict[pair_key][bad_idx]
        bad_label = f"({s},{t})v{bad_idx}_bad"
        vec_specs.append((bad_label, source_layer, bad_vec))

    output = {
        "config": {
            "model": args.model,
            "pairs": args.pairs,
            "num_random": args.num_random,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "vectors": {},
    }

    for label, source_layer, vec in vec_specs:
        print(f"\n{'='*60}")
        print(f"Vector: {label} (source layer {source_layer})")
        print(f"{'='*60}")

        vec_output = {}
        for group_name, prompts in prompt_groups:
            print(f"\n  Measuring {group_name} ({len(prompts)} prompts)...")
            results = measure_projections(
                model, tokenizer, prompts, source_layer, vec,
                num_random=args.num_random, seed=args.seed,
            )
            summary = summarize(results, group_name)
            vec_output[group_name] = {"summary": summary, "per_prompt": results}

        output["vectors"][label] = vec_output

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"projections_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
