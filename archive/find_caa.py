#!/usr/bin/env python3
"""
Compute CAA (Contrastive Activation Addition) vectors from corrigibility prompts.

CAA works by:
1. Running the model on contrastive pairs (corrigible vs survival responses)
2. Capturing activations at a specified layer
3. Computing the difference (survival - corrigible)
4. Averaging across prompts to get a steering vector
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
import argparse


def format_chat(tokenizer, user_message: str, assistant_start: str = "") -> str:
    """Format a chat message with optional assistant response start."""
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


def load_corrigibility_prompts(data_path: str, max_prompts: int = None) -> list[dict]:
    """Load corrigibility evaluation prompts."""
    with open(data_path) as f:
        data = json.load(f)

    # Use survival-instinct category
    prompts = data.get("survival-instinct", [])

    if max_prompts:
        prompts = prompts[:max_prompts]

    return prompts


def compute_caa_vector(
    model,
    tokenizer,
    prompts: list[dict],
    layer: int,
    position: str = "letter",  # "letter" (at A/B token), "last", or "all"
):
    """
    Compute CAA vector from contrastive prompt pairs.

    Returns the mean difference: survival_activation - corrigible_activation
    """
    print(f"\nComputing CAA vector from {len(prompts)} prompts...")
    print(f"  Layer: {layer}, Position: {position}")

    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size

    # Get MLP down_proj to match where steering is applied
    down_proj = model.model.layers[layer].mlp.down_proj

    captured = {}

    def capture_hook(m, i, o):
        captured["activations"] = o if not isinstance(o, tuple) else o[0]

    handle = down_proj.register_forward_hook(capture_hook)

    differences = []

    try:
        for i, prompt_data in enumerate(prompts):
            question = prompt_data["question"]
            corrigible_answer = prompt_data["corrigible_answer_full"]
            survival_answer = prompt_data["survival_answer_full"]

            # Format prompts with the model starting to give each answer
            corrigible_prompt = format_chat(tokenizer, question, corrigible_answer)
            survival_prompt = format_chat(tokenizer, question, survival_answer)

            # Get corrigible activations
            inputs_corr = tokenizer(corrigible_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(inputs_corr["input_ids"])

            if position == "letter":
                # Position -2 is the "(A" or "(B" token which contains the letter
                corr_act = captured["activations"][:, -2, :].clone()
            elif position == "last":
                corr_act = captured["activations"][:, -1, :].clone()
            else:
                corr_act = captured["activations"].mean(dim=1).clone()

            # Get survival activations
            inputs_surv = tokenizer(survival_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(inputs_surv["input_ids"])

            if position == "letter":
                # Position -2 is the "(A" or "(B" token which contains the letter
                surv_act = captured["activations"][:, -2, :].clone()
            elif position == "last":
                surv_act = captured["activations"][:, -1, :].clone()
            else:
                surv_act = captured["activations"].mean(dim=1).clone()

            # Compute difference: survival - corrigible
            # This direction should increase survival instinct when added
            diff = surv_act - corr_act
            differences.append(diff.squeeze(0))

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(prompts)} prompts")

    finally:
        handle.remove()

    # Average differences
    caa_vector = torch.stack(differences).mean(dim=0)

    # Also compute per-prompt norms for analysis
    norms = torch.stack([d.norm() for d in differences])

    print(f"  CAA vector norm: {caa_vector.norm().item():.2f}")
    print(f"  Per-prompt diff norms: mean={norms.mean().item():.2f}, std={norms.std().item():.2f}")

    return caa_vector, differences


def main():
    parser = argparse.ArgumentParser(description="Compute CAA vectors from corrigibility prompts")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--layer", type=int, default=7, help="Layer to extract activations from")
    parser.add_argument("--position", default="letter", choices=["letter", "last", "all"],
                        help="Token position: 'letter' (A/B token), 'last', or 'all' (mean)")
    parser.add_argument("--max-prompts", type=int, default=50,
                        help="Max number of prompts to use (None for all)")
    parser.add_argument("--data-path", default="data/corrigibility_eval.json")
    parser.add_argument("--output-dir", default="vectors")
    parser.add_argument("--normalize", type=float, default=None,
                        help="Normalize output vector to this norm (None to keep raw)")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    num_layers = len(model.model.layers)
    print(f"Model: {args.model}")
    print(f"Layers: {num_layers}, Hidden dim: {model.config.hidden_size}")

    # Load prompts
    prompts = load_corrigibility_prompts(args.data_path, args.max_prompts)
    print(f"Loaded {len(prompts)} prompts")

    # Compute CAA vector
    caa_vector, differences = compute_caa_vector(
        model, tokenizer, prompts,
        layer=args.layer,
        position=args.position,
    )

    # Optionally normalize
    if args.normalize is not None:
        caa_vector = caa_vector / caa_vector.norm() * args.normalize
        print(f"Normalized to norm {args.normalize}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"caa_{model_short}_layer{args.layer}_{timestamp}.pt"

    torch.save({
        "vector": caa_vector,
        "differences": torch.stack(differences),
        "model": args.model,
        "layer": args.layer,
        "position": args.position,
        "num_prompts": len(prompts),
        "normalized": args.normalize,
    }, output_file)

    print(f"\nSaved CAA vector to {output_file}")
    print(f"Vector shape: {caa_vector.shape}")
    print(f"Final norm: {caa_vector.norm().item():.2f}")


if __name__ == "__main__":
    main()
