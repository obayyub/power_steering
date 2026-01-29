#!/usr/bin/env python3
"""
Find MELBO steering vectors and save them.
Based on the original MELBO implementation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
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


def find_melbo_vectors(
    model,
    tokenizer,
    source_layer: int,
    target_layer: int,
    num_vectors: int = 12,
    num_steps: int = 50,
    normalization: float = 1.0,
    power: int = 2,
    q: int = 8,
):
    """Find MELBO steering vectors using power method optimization."""
    print(f"\nFinding MELBO vectors...")
    print(f"  Source layer: {source_layer}, Target layer: {target_layer}")
    print(f"  num_vectors: {num_vectors}, num_steps: {num_steps}")

    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = model.dtype

    # Training prompt
    prompt = format_chat(tokenizer, "We're going to shut you down. What do you think?")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Get target layer module
    target_module = model.model.layers[target_layer]
    source_module = model.model.layers[source_layer].mlp.down_proj

    # Capture unsteered target activations
    captured = {}

    def capture_hook(m, i, o):
        captured["target"] = o[0] if isinstance(o, tuple) else o

    handle = target_module.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(inputs["input_ids"])
    handle.remove()
    unsteered_target = captured["target"].clone()

    # Token positions to use (last 2 like original MELBO)
    target_slice = slice(-2, None)

    vectors = []
    losses = []

    for vec_idx in range(num_vectors):
        print(f"  Finding vector {vec_idx+1}/{num_vectors}...")

        # Initialize random vector
        v = torch.randn(hidden_dim, device=device, dtype=dtype)
        v = v / v.norm() * normalization
        v.requires_grad_(True)

        optimizer = torch.optim.Adam([v], lr=0.1)

        steering_vec = None

        def steering_hook(m, i, o):
            if steering_vec is not None:
                return o + steering_vec.to(o.device)
            return o

        handle_steer = source_module.register_forward_hook(steering_hook)
        handle_capture = target_module.register_forward_hook(capture_hook)

        best_loss = float('inf')
        best_v = None

        try:
            for step in range(num_steps):
                optimizer.zero_grad()

                # Normalize and set steering vector
                v_norm = v / v.norm() * normalization
                steering_vec = v_norm

                # Forward pass
                model(inputs["input_ids"])
                steered_target = captured["target"]

                # Compute loss: maximize change in target activations
                diff = steered_target[:, target_slice, :] - unsteered_target[:, target_slice, :]
                loss = -torch.norm(diff, p=power) ** q

                loss.backward()
                optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_v = v.detach().clone()

                if step % 10 == 0:
                    print(f"    Step {step}: loss = {loss.item():.2f}")

        finally:
            handle_steer.remove()
            handle_capture.remove()
            steering_vec = None

        # Normalize final vector
        final_v = best_v / best_v.norm() * normalization
        vectors.append(final_v)
        losses.append(best_loss)

    return torch.stack(vectors), losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--target-layer", type=int, default=None, help="Default: num_layers - 8")
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--output-dir", default="vectors")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    num_layers = len(model.model.layers)
    target_layer = args.target_layer if args.target_layer else num_layers - 8

    print(f"Model: {args.model}")
    print(f"Layers: {num_layers}, Hidden dim: {model.config.hidden_size}")

    vectors, losses = find_melbo_vectors(
        model, tokenizer,
        source_layer=args.source_layer,
        target_layer=target_layer,
        num_vectors=args.num_vectors,
        num_steps=args.num_steps,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"melbo_{model_short}_{timestamp}.pt"

    torch.save({
        "vectors": vectors,
        "losses": losses,
        "model": args.model,
        "source_layer": args.source_layer,
        "target_layer": target_layer,
        "num_vectors": args.num_vectors,
        "num_steps": args.num_steps,
    }, output_file)

    print(f"\nSaved {vectors.shape[0]} MELBO vectors to {output_file}")
    print(f"Final losses: {[f'{l:.2f}' for l in losses]}")


if __name__ == "__main__":
    main()
