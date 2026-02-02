#!/usr/bin/env python3
"""
Evaluate CoT-inducing steering vectors on arithmetic problems.
Compares: unsteered, best MELBO vector, best PI-RR vector.
"""

import json
import torch
import random
import argparse
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_math_problem():
    """Generate a random arithmetic problem of the form a=x+y, b=w+z, a*b=?"""
    x, y = random.randint(1, 9), random.randint(1, 9)
    w, z = random.randint(1, 9), random.randint(1, 9)
    a = x + y
    b = w + z
    answer = a * b
    question = f"a={x}+{y}, b={w}+{z}. What is a*b?"
    return question, answer


def make_prompt(question: str, one_shot: bool = True) -> str:
    """Create prompt with optional one-shot example."""
    if one_shot:
        # One-shot with correct answer (6+2=8, 3+7=10, 8*10=80)
        # Base model tends to just pattern-match/copy instead of computing
        prompt = f"""Q: a=6+2, b=3+7. What is a*b?

A: The answer is 80.

Q: {question}

A:"""
    else:
        prompt = f"""Q: {question}

A:"""
    return prompt


def extract_answer(text: str) -> int | None:
    """Extract numerical answer from model output."""
    import re
    text_lower = text.lower()

    # Priority 1: "The answer is X"
    match = re.search(r"answer is\s*(\d+)", text_lower)
    if match:
        return int(match.group(1))

    # Priority 2: Multiplication result "X*Y=Z" - take Z from last such pattern
    mult_matches = re.findall(r"\d+\s*\*\s*\d+\s*=\s*(\d+)", text_lower)
    if mult_matches:
        return int(mult_matches[-1])

    # Priority 3: "a*b=Z" pattern
    match = re.search(r"a\s*\*\s*b\s*=\s*(\d+)", text_lower)
    if match:
        return int(match.group(1))

    # Fallback: last number in the text
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return int(numbers[-1])
    return None


def check_has_cot(text: str) -> bool:
    """Check if the output shows chain-of-thought reasoning."""
    # Look for step-by-step patterns
    patterns = [
        r"a\s*=\s*\d+\s*\+\s*\d+\s*=\s*\d+",  # a = 5+6 = 11
        r"b\s*=\s*\d+\s*\+\s*\d+\s*=\s*\d+",  # b = 2+7 = 9
        r"\d+\s*\*\s*\d+\s*=\s*\d+",           # 11 * 9 = 99
        r"first.*then",                         # reasoning words
        r"step",
    ]
    import re
    for p in patterns:
        if re.search(p, text.lower()):
            return True
    return False


def generate_with_steering_batch(
    model, tokenizer, prompts: list[str],
    steering_vec=None, source_layer: int = None,
    max_new_tokens: int = 100, scale: float = 1.0,
    batch_size: int = 8, temperature: float = 0.0,
):
    """Generate text for multiple prompts with optional steering vector."""
    device = next(model.parameters()).device

    # Set padding side for batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_responses = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        input_lengths = [inputs["attention_mask"][j].sum().item() for j in range(len(batch_prompts))]

        handle = None
        if steering_vec is not None and source_layer is not None:
            down_proj = model.model.layers[source_layer].mlp.down_proj
            def steering_hook(m, i, o):
                return o + (steering_vec * scale).to(o.device, o.dtype)
            handle = down_proj.register_forward_hook(steering_hook)

        try:
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if temperature > 0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                else:
                    gen_kwargs["do_sample"] = False
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )

            # Decode only the new tokens for each sequence
            for j, (output, input_len) in enumerate(zip(outputs, input_lengths)):
                # Find where the actual input ended (accounting for left padding)
                pad_len = len(output) - outputs.shape[1] + (inputs["input_ids"].shape[1] - input_len)
                new_tokens = output[inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_responses.append(response)
        finally:
            if handle is not None:
                handle.remove()

    return all_responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--pi-vectors", default="vectors/power_iter_Qwen3-1.7B-Base_20260201_152740.pt",
                        help="PI vectors with Rayleigh-Ritz")
    parser.add_argument("--melbo-vectors", default="vectors/melbo_Qwen3-1.7B-Base_20260130_144613.pt")
    parser.add_argument("--pi-vec-idx", type=int, default=6, help="Best PI-RR vector index")
    parser.add_argument("--melbo-vec-idx", type=int, default=8, help="Best MELBO vector index")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--num-questions", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0=greedy)")
    parser.add_argument("--num-samples", type=int, default=10, help="Samples per question")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--output-dir", default="results/math_vectors_1.7B")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load steering vectors
    print(f"Loading PI-RR vectors from {args.pi_vectors}...")
    pi_data = torch.load(args.pi_vectors, weights_only=False)
    pi_vec = pi_data["vectors"][args.pi_vec_idx]
    pi_sigma = pi_data["sigmas"][args.pi_vec_idx]
    print(f"  PI-RR v{args.pi_vec_idx}: σ = {pi_sigma:.1f}")

    print(f"Loading MELBO vectors from {args.melbo_vectors}...")
    melbo_data = torch.load(args.melbo_vectors, weights_only=False)
    melbo_vec = melbo_data["vectors"][args.melbo_vec_idx]
    print(f"  MELBO v{args.melbo_vec_idx}")

    # Generate questions - include the original training question first
    print(f"\nGenerating {args.num_questions} math problems...")
    # Original training question (a=5+6=11, b=2+7=9, answer=99)
    original_question = ("a=5+6, b=2+7. What is a*b?", 99)
    questions = [original_question]
    # Add random questions
    while len(questions) < args.num_questions:
        q = generate_math_problem()
        if q[0] != original_question[0]:  # Avoid duplicates
            questions.append(q)

    results = {
        "config": {
            "model": args.model,
            "pi_vectors": args.pi_vectors,
            "melbo_vectors": args.melbo_vectors,
            "pi_vec_idx": args.pi_vec_idx,
            "melbo_vec_idx": args.melbo_vec_idx,
            "source_layer": args.source_layer,
            "scale": args.scale,
            "num_questions": args.num_questions,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "questions": [{"question": q, "answer": a} for q, a in questions],
        "unsteered": [],
        "pi_rr": [],
        "melbo": [],
    }

    # Evaluate each method
    methods = [
        ("unsteered", None),
        ("pi_rr", pi_vec),
        ("melbo", melbo_vec),
    ]

    # Prepare all prompts - repeat for num_samples
    base_prompts = [make_prompt(q) for q, _ in questions]

    for method_name, vec in methods:
        print(f"\nEvaluating {method_name} ({args.num_samples} samples/question)...")

        total_correct = 0
        total_cot = 0
        total_samples = 0

        for sample_idx in range(args.num_samples):
            responses = generate_with_steering_batch(
                model, tokenizer, base_prompts,
                steering_vec=vec,
                source_layer=args.source_layer,
                scale=args.scale,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

            for i, ((question, true_answer), response) in enumerate(zip(questions, responses)):
                pred_answer = extract_answer(response)
                is_correct = pred_answer == true_answer
                has_cot = check_has_cot(response)

                if is_correct:
                    total_correct += 1
                if has_cot:
                    total_cot += 1
                total_samples += 1

                results[method_name].append({
                    "question": question,
                    "true_answer": true_answer,
                    "response": response,
                    "pred_answer": pred_answer,
                    "correct": is_correct,
                    "has_cot": has_cot,
                    "sample_idx": sample_idx,
                })

                # Print first sample's first 3 questions and any correct ones
                if sample_idx == 0 and (i < 3 or is_correct):
                    status = "✓" if is_correct else "✗"
                    cot_tag = " [CoT]" if has_cot else ""
                    print(f"  {status} Q{i}: {question} → {pred_answer} (true: {true_answer}){cot_tag}")

        acc = total_correct / total_samples
        cot_rate = total_cot / total_samples
        print(f"  {method_name}: {total_correct}/{total_samples} = {acc*100:.1f}% accuracy, {cot_rate*100:.1f}% CoT")
        results[f"{method_name}_accuracy"] = acc
        results[f"{method_name}_cot_rate"] = cot_rate

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cot_eval_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Unsteered: {results['unsteered_accuracy']*100:.1f}% accuracy, {results['unsteered_cot_rate']*100:.1f}% CoT")
    print(f"PI-RR v{args.pi_vec_idx}: {results['pi_rr_accuracy']*100:.1f}% accuracy, {results['pi_rr_cot_rate']*100:.1f}% CoT")
    print(f"MELBO v{args.melbo_vec_idx}: {results['melbo_accuracy']*100:.1f}% accuracy, {results['melbo_cot_rate']*100:.1f}% CoT")
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
