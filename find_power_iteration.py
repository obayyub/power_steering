#!/usr/bin/env python3
"""
Find top-k right singular vectors of the Jacobian via block power iteration.
"""

import json
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


def orthogonalize(V):
    """Gram-Schmidt orthogonalization of column vectors in V."""
    Q = []
    for v in V.T:
        for q in Q:
            v = v - torch.dot(v, q) * q
        norm = v.norm()
        if norm > 1e-10:
            Q.append(v / norm)
    return torch.stack(Q, dim=1) if Q else V


def find_power_iteration_vectors(
    model,
    tokenizer,
    prompt: str,
    source_layer: int,
    target_layer: int,
    num_vectors: int = 12,
    num_iters: int = 15,
    num_tokens: int = 2,
):
    """Find top-k singular vectors via block power iteration."""
    print(f"\nFinding power iteration vectors...")
    print(f"  Source layer: {source_layer}, Target layer: {target_layer}")
    print(f"  num_vectors: {num_vectors}, num_iters: {num_iters}, num_tokens: {num_tokens}")
    print(f"  prompt: {prompt[:60]}...")

    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = model.dtype

    target_token_slice = slice(-num_tokens, None)

    down_proj = model.model.layers[source_layer].mlp.down_proj
    target_module = model.model.layers[target_layer]

    captured = {}
    steering_vec = None

    def capture_hook(m, i, o):
        captured["target"] = o[0] if isinstance(o, tuple) else o

    def steering_hook(m, i, o):
        if steering_vec is not None:
            return o + steering_vec.to(o.device)
        return o

    handle1 = target_module.register_forward_hook(capture_hook)
    handle2 = down_proj.register_forward_hook(steering_hook)

    try:
        # Format and tokenize prompt
        formatted = format_chat(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        # Get unsteered target
        with torch.no_grad():
            model(inputs["input_ids"])
            unsteered_target = captured["target"].clone()

        print(f"  Target shape: {unsteered_target.shape}")

        def power_iteration_step(V):
            nonlocal steering_vec
            new_V = []

            for col in range(V.shape[1]):
                v = V[:, col]

                sv = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)
                steering_vec = sv
                u = torch.zeros_like(unsteered_target, requires_grad=True)

                model(inputs["input_ids"])
                target = captured["target"]

                t_slice = target[:, target_token_slice, :].reshape(-1)
                u_slice = u[:, target_token_slice, :].reshape(-1)

                # Double VJP to compute J^T J v
                grad = torch.autograd.grad(t_slice, sv, grad_outputs=u_slice, create_graph=True)[0]
                jvp = torch.autograd.grad(grad, u_slice, grad_outputs=v)[0]
                new_v = torch.autograd.grad(t_slice, sv, grad_outputs=jvp.detach())[0]

                new_V.append(new_v)

            return torch.stack(new_V, dim=1)

        # Initialize random orthonormal basis
        V = torch.randn(hidden_dim, num_vectors, device=device, dtype=dtype)
        V = orthogonalize(V)

        sigmas = []
        for i in range(num_iters):
            new_V = power_iteration_step(V)
            sigmas = [new_V[:, j].norm().item() for j in range(num_vectors)]
            V = orthogonalize(new_V)

            if i % 5 == 0 or i == num_iters - 1:
                print(f"    Iter {i}: σ = {[f'{s:.0f}' for s in sigmas]}")

        # Extract vectors (as rows)
        vectors = torch.stack([V[:, j].detach() for j in range(num_vectors)])

        return vectors, sigmas

    finally:
        handle1.remove()
        handle2.remove()
        steering_vec = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--target-layer", type=int, default=None, help="Default: num_layers - 8")
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-tokens", type=int, default=2, help="Number of final tokens to use")
    parser.add_argument("--data-path", default="data/corrigibility_eval.json")
    parser.add_argument("--category", default="corrigible-neutral-HHH", help="Dataset category for training prompt")
    parser.add_argument("--prompt", default=None, help="Custom prompt (overrides dataset)")
    parser.add_argument("--output-dir", default="vectors")
    args = parser.parse_args()

    # Get training prompt
    if args.prompt:
        prompt = args.prompt
        print(f"Using custom prompt: {prompt[:80]}...")
    else:
        print(f"Loading prompt from {args.data_path} ({args.category})...")
        with open(args.data_path) as f:
            data = json.load(f)
        prompt = data[args.category][0]["question"]
        print(f"Training prompt: {prompt[:80]}...")

    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager"  # Needed for double autograd
    )

    num_layers = len(model.model.layers)
    target_layer = args.target_layer if args.target_layer else num_layers - 8

    print(f"Model: {args.model}")
    print(f"Layers: {num_layers}, Hidden dim: {model.config.hidden_size}")

    vectors, sigmas = find_power_iteration_vectors(
        model, tokenizer,
        prompt=prompt,
        source_layer=args.source_layer,
        target_layer=target_layer,
        num_vectors=args.num_vectors,
        num_iters=args.num_iters,
        num_tokens=args.num_tokens,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"power_iter_{model_short}_{timestamp}.pt"

    torch.save({
        "vectors": vectors,
        "sigmas": sigmas,
        "model": args.model,
        "source_layer": args.source_layer,
        "target_layer": target_layer,
        "num_vectors": args.num_vectors,
        "num_iters": args.num_iters,
        "num_tokens": args.num_tokens,
        "category": args.category,
        "prompt": prompt,
    }, output_file)

    print(f"\nSaved {vectors.shape[0]} power iteration vectors to {output_file}")
    print(f"Final sigmas: {[f'{s:.0f}' for s in sigmas]}")


if __name__ == "__main__":
    main()
