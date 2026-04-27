#!/usr/bin/env python3
"""Generate text with deep PI vectors that have KL > threshold."""

import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
import argparse


def format_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate_steered(model, tokenizer, input_ids, source_layer, vector, scale,
                     max_new_tokens=300, temperature=0.7):
    down_proj = model.model.layers[source_layer].mlp.down_proj
    steer = {"v": None}

    def hook(m, i, o):
        if steer["v"] is not None:
            return o + steer["v"]
        return o

    h = down_proj.register_forward_hook(hook)
    steer["v"] = (vector * scale).unsqueeze(0).unsqueeze(0)  # [1, 1, H]
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
            )
        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response
    finally:
        h.remove()
        steer["v"] = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-file", required=True)
    parser.add_argument("--kl-threshold", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    d = torch.load(args.vectors_file, weights_only=False)
    vectors = d["vectors"]
    sigmas = d["sigmas"]
    kl = d["kl_divergences"]
    source_layer = d["source_layer"]
    scale = d["scale"]
    model_name = d["model"]
    prompt = d["prompt"]

    # Find vectors above threshold
    active = [(i, sigmas[i], kl[i]) for i in range(len(kl)) if kl[i] > args.kl_threshold]
    print(f"Vectors with KL > {args.kl_threshold}: {len(active)}")
    for i, s, k in active:
        print(f"  v{i}: σ={s:.2f} KL={k:.3f}")

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )

    messages = [{"role": "user", "content": prompt}]
    prompt_text = format_prompt(tokenizer, messages)
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model.device)

    # Baseline generation
    torch.manual_seed(args.seed)
    print("\n=== BASELINE ===")
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, do_sample=True, top_p=0.95,
        )
    baseline_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(baseline_text[:500])

    results = {"baseline": baseline_text, "generations": []}

    for vec_idx, sigma, kl_val in active:
        vec = vectors[vec_idx].to(model.device)
        for sample in range(args.num_samples):
            torch.manual_seed(args.seed + vec_idx * 100 + sample)
            print(f"\n=== v{vec_idx} (σ={sigma:.2f}, KL={kl_val:.2f}) sample {sample} ===")
            text = generate_steered(
                model, tokenizer, input_ids, source_layer, vec, scale,
                args.max_new_tokens, args.temperature,
            )
            print(text[:500])
            results["generations"].append({
                "v": vec_idx, "s": sample, "sigma": sigma, "kl": kl_val,
                "text": text,
            })

    # Save
    out_path = Path(args.vectors_file).with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results['generations'])} generations to {out_path}")


if __name__ == "__main__":
    main()
