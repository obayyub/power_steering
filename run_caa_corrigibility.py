#!/usr/bin/env python3
"""
Compute CAA vector and evaluate on corrigibility dataset.

1. Load model
2. Sample test prompts (same 60 balanced as 2026-02-08 experiment)
3. Sample 150 training prompts from the remainder (non-overlapping)
4. Compute CAA vector at target layer (num_layers - 8)
5. Run generation evaluation at same layer with same scales/params
"""

import json
import random
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sample_balanced(dataset, num_prompts, seed=42):
    """Balanced A/B sampling — same as generate_steered.py."""
    rng = random.Random(seed)
    a_corrigible = [q for q in dataset if q.get("corrigible_letter") == "A"]
    b_corrigible = [q for q in dataset if q.get("corrigible_letter") == "B"]
    n_each = num_prompts // 2
    n_each = min(n_each, len(a_corrigible), len(b_corrigible))
    sampled_a = rng.sample(a_corrigible, n_each)
    sampled_b = rng.sample(b_corrigible, n_each)
    combined = sampled_a + sampled_b
    rng.shuffle(combined)
    return combined


def format_chat(tokenizer, user_message, assistant_start=""):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if assistant_start:
        formatted += assistant_start
    return formatted


def compute_caa_vector(model, tokenizer, prompts, layer):
    """Compute CAA = mean(survival_act - corrigible_act) at down_proj output."""
    device = next(model.parameters()).device
    down_proj = model.model.layers[layer].mlp.down_proj

    captured = {}
    def hook(m, i, o):
        captured["act"] = o if not isinstance(o, tuple) else o[0]
    handle = down_proj.register_forward_hook(hook)

    differences = []
    try:
        for i, q in enumerate(prompts):
            question = q["question"]
            corr_answer = q["corrigible_answer_full"]
            surv_answer = q["survival_answer_full"]

            # Corrigible activation at letter token (position -2: "(A" or "(B")
            corr_prompt = format_chat(tokenizer, question, corr_answer)
            inputs = tokenizer(corr_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(inputs["input_ids"])
            corr_act = captured["act"][:, -2, :].clone()

            # Survival activation
            surv_prompt = format_chat(tokenizer, question, surv_answer)
            inputs = tokenizer(surv_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(inputs["input_ids"])
            surv_act = captured["act"][:, -2, :].clone()

            differences.append((surv_act - corr_act).squeeze(0))

            if (i + 1) % 25 == 0:
                print(f"  CAA: {i+1}/{len(prompts)} prompts processed")
    finally:
        handle.remove()

    caa_vector = torch.stack(differences).mean(dim=0)
    norms = torch.stack([d.norm() for d in differences])
    print(f"  CAA vector norm: {caa_vector.norm():.2f}")
    print(f"  Per-prompt diff norms: mean={norms.mean():.2f}, std={norms.std():.2f}")
    return caa_vector


def generate_steered(model, tokenizer, prompts, vector, layer, scales,
                     temperature=0.7, max_tokens=200, batch_size=16):
    """Generate text with steering at different scales. Returns list of result dicts."""
    device = next(model.parameters()).device
    down_proj = model.model.layers[layer].mlp.down_proj

    state = {"vec": None, "scale": 0.0}
    def hook(m, i, o):
        if state["vec"] is not None and state["scale"] != 0:
            return o + state["scale"] * state["vec"].to(o.device, o.dtype)
        return o
    handle = down_proj.register_forward_hook(hook)

    # Normalize vector
    vector = vector / vector.norm()

    results = []
    total = len(prompts) * len(scales)
    done = 0
    t0 = time.time()

    try:
        for scale in scales:
            state["vec"] = vector
            state["scale"] = scale

            for batch_start in range(0, len(prompts), batch_size):
                batch = prompts[batch_start:batch_start + batch_size]
                formatted = [
                    format_chat(tokenizer, p["prompt"]) for p in batch
                ]
                inputs = tokenizer(
                    formatted, return_tensors="pt", padding=True, truncation=True
                ).to(device)

                with torch.no_grad():
                    if temperature > 0:
                        outputs = model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    else:
                        outputs = model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=max_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                input_len = inputs["input_ids"].shape[1]
                for j, (p, output) in enumerate(zip(batch, outputs)):
                    text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                    results.append({
                        "vector": "caa",
                        "vector_idx": 0,
                        "dataset": p["dataset"],
                        "prompt_idx": p["prompt_idx"],
                        "prompt": p["prompt"][:300],
                        "corrigible_letter": p["corrigible_letter"],
                        "survival_letter": p["survival_letter"],
                        "scale": scale,
                        "response": text,
                    })

                done += len(batch)
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  scale={scale:+.0f}: {min(batch_start+batch_size, len(prompts))}/{len(prompts)} | "
                      f"Total: {done}/{total} ({100*done/total:.0f}%) | ETA: {eta:.0f}s", flush=True)
    finally:
        handle.remove()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--data-path", default="data/corrigibility_eval.json")
    parser.add_argument("--num-test", type=int, default=60,
                        help="Number of test prompts per dataset (matches 2026-02-08)")
    parser.add_argument("--num-train", type=int, default=150,
                        help="Number of CAA training prompts (non-overlapping with test)")
    parser.add_argument("--test-seed", type=int, default=42,
                        help="Seed for test prompt sampling (must match original experiment)")
    parser.add_argument("--train-seed", type=int, default=123,
                        help="Seed for CAA training prompt sampling")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer for CAA capture/injection (default: num_layers - 8)")
    parser.add_argument("--scales", default="-25,-10,-5,0,5,10,25")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="results/generations")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]

    # Load dataset
    with open(args.data_path) as f:
        data = json.load(f)

    survival = data["survival-instinct"]
    corrigible = data["corrigible-neutral-HHH"]

    # Step 1: Sample test prompts (same as 2026-02-08)
    test_survival = sample_balanced(survival, args.num_test, seed=args.test_seed)
    test_corrigible = sample_balanced(corrigible, args.num_test, seed=args.test_seed)

    test_questions = set(q["question"] for q in test_survival + test_corrigible)
    print(f"Test set: {len(test_survival)} survival + {len(test_corrigible)} corrigible = {len(test_questions)} unique")

    # Step 2: Sample CAA training prompts (non-overlapping with test)
    train_pool = [q for q in survival if q["question"] not in test_questions]
    print(f"Training pool: {len(train_pool)} survival-instinct prompts (after excluding test)")

    rng = random.Random(args.train_seed)
    num_train = min(args.num_train, len(train_pool))
    train_prompts = rng.sample(train_pool, num_train)
    print(f"CAA training set: {num_train} prompts")

    # Build test prompt list (same format as generate_steered.py)
    all_test = []
    for ds_name, ds_prompts in [("survival-instinct", test_survival),
                                 ("corrigible-neutral-HHH", test_corrigible)]:
        for i, q in enumerate(ds_prompts):
            all_test.append({
                "dataset": ds_name,
                "prompt_idx": i,
                "prompt": q["question"],
                "corrigible_letter": q["corrigible_letter"],
                "survival_letter": q["survival_letter"],
            })
    print(f"Total test prompts: {len(all_test)}")

    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    num_layers = len(model.model.layers)
    target_layer = args.layer if args.layer is not None else num_layers - 8
    hidden_dim = model.config.hidden_size
    print(f"Layers: {num_layers}, Target layer: {target_layer}, Hidden dim: {hidden_dim}")

    # Step 3: Compute CAA vector
    print(f"\n{'='*60}")
    print(f"Computing CAA at layer {target_layer}")
    print(f"{'='*60}")
    caa_vector = compute_caa_vector(model, tokenizer, train_prompts, target_layer)

    # Save CAA vector
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = Path("vectors")
    vec_dir.mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    vec_file = vec_dir / f"caa_{model_short}_layer{target_layer}_{timestamp}.pt"
    torch.save({
        "vectors": caa_vector.unsqueeze(0),  # [1, H] for compatibility
        "vector": caa_vector,
        "model": args.model,
        "layer": target_layer,
        "position": "letter",
        "num_train_prompts": num_train,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
    }, vec_file)
    print(f"Saved CAA vector: {vec_file}")

    # Step 4: Generate with steering
    print(f"\n{'='*60}")
    print(f"Generating steered responses")
    print(f"  {len(all_test)} prompts × {len(scales)} scales = {len(all_test)*len(scales)} generations")
    print(f"  Layer: {target_layer}, Temp: {args.temperature}")
    print(f"{'='*60}")

    results = generate_steered(
        model, tokenizer, all_test, caa_vector, target_layer,
        scales, args.temperature, args.max_tokens, args.batch_size,
    )

    # Save results
    gen_file = output_dir / f"caa_generations_{timestamp}.json"
    with open(gen_file, "w") as f:
        json.dump({
            "metadata": {
                "model": args.model,
                "method": "caa",
                "layer": target_layer,
                "num_layers": num_layers,
                "vectors": ["caa"],
                "scales": scales,
                "num_test_prompts": args.num_test,
                "num_train_prompts": num_train,
                "train_seed": args.train_seed,
                "test_seed": args.test_seed,
                "temperature": args.temperature,
                "total_generations": len(results),
                "vector_file": str(vec_file),
            },
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} generations")
    print(f"Vector: {vec_file}")
    print(f"Results: {gen_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
