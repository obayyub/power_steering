#!/usr/bin/env python3
"""
Find top-k right singular vectors of the stacked Jacobian across multiple prompts.

Instead of finding singular vectors for a single prompt's Jacobian J, this finds
singular vectors of the stacked Jacobian [J_1; J_2; ...; J_n], which is equivalent
to finding eigenvectors of Σ_i (J_i^T J_i).

This produces steering vectors that have high sensitivity across the distribution
of prompts, not just a single prompt.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
import argparse
import json
import random


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


def load_prompts(data_path: str, category: str = "survival-instinct", num_prompts: int = 50):
    """Load prompts from the corrigibility eval dataset."""
    with open(data_path) as f:
        data = json.load(f)

    prompts = [item["question"] for item in data[category]]

    if num_prompts < len(prompts):
        prompts = random.sample(prompts, num_prompts)

    return prompts


def find_power_iteration_vectors_multi(
    model,
    tokenizer,
    prompts: list[str],
    source_layer: int,
    target_layer: int,
    num_vectors: int = 12,
    num_iters: int = 15,
    num_tokens: int = 2,
    batch_size: int = 8,
):
    """
    Find top-k singular vectors via block power iteration across multiple prompts.

    This finds eigenvectors of Σ_i (J_i^T J_i), equivalent to singular vectors
    of the stacked Jacobian [J_1; J_2; ...; J_n].

    Uses batched forward passes with left-padding for efficiency.
    """
    print(f"\nFinding multi-prompt power iteration vectors...")
    print(f"  Source layer: {source_layer}, Target layer: {target_layer}")
    print(f"  num_vectors: {num_vectors}, num_iters: {num_iters}, num_tokens: {num_tokens}")
    print(f"  num_prompts: {len(prompts)}, batch_size: {batch_size}")

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
        # Use left-padding so last tokens are always actual content
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize and batch all prompts
        formatted_prompts = [format_chat(tokenizer, p) for p in prompts]

        batches = []
        for i in range(0, len(formatted_prompts), batch_size):
            batch_prompts = formatted_prompts[i:i + batch_size]
            batch_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            batches.append(batch_inputs)

        tokenizer.padding_side = original_padding_side
        print(f"  Created {len(batches)} batches")

        def power_iteration_step(V):
            """Apply Σ_i (J_i^T J_i) to each column of V using batched forward passes."""
            nonlocal steering_vec
            new_V = torch.zeros_like(V)

            for col in range(V.shape[1]):
                v = V[:, col]
                accumulated = torch.zeros_like(v)

                for batch_inputs in batches:
                    sv = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)
                    steering_vec = sv

                    # Batched forward pass
                    model(**batch_inputs)
                    target = captured["target"]  # [batch, seq, hidden]

                    # Create u tensor for this batch
                    u = torch.zeros_like(target, requires_grad=True)

                    # Take last num_tokens from each sequence and flatten
                    # With left-padding, last tokens are always actual content
                    # Shape: [batch * num_tokens * hidden]
                    t_slice = target[:, target_token_slice, :].reshape(-1)
                    u_slice = u[:, target_token_slice, :].reshape(-1)

                    # Double VJP - autograd sums gradients across batch
                    grad = torch.autograd.grad(t_slice, sv, grad_outputs=u_slice, create_graph=True)[0]
                    jvp = torch.autograd.grad(grad, u_slice, grad_outputs=v)[0]
                    jtj_v = torch.autograd.grad(t_slice, sv, grad_outputs=jvp.detach())[0]

                    accumulated = accumulated + jtj_v

                new_V[:, col] = accumulated

            return new_V

        # Initialize random orthonormal basis
        V = torch.randn(hidden_dim, num_vectors, device=device, dtype=dtype)
        V = orthogonalize(V)

        sigmas = []
        for i in range(num_iters):
            new_V = power_iteration_step(V)
            # Sigmas are sqrt of eigenvalues of Σ(J^T J), scaled by num_prompts
            sigmas = [new_V[:, j].norm().item() for j in range(num_vectors)]
            V = orthogonalize(new_V)

            if i % 3 == 0 or i == num_iters - 1:
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
    parser.add_argument("--num-prompts", type=int, default=32, help="Number of prompts to use")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for forward passes")
    parser.add_argument("--data-path", default="data/corrigibility_eval.json")
    parser.add_argument("--category", default="corrigible-neutral-HHH")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="vectors")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading prompts from {args.data_path}...")
    prompts = load_prompts(args.data_path, args.category, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")

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

    vectors, sigmas = find_power_iteration_vectors_multi(
        model, tokenizer, prompts,
        source_layer=args.source_layer,
        target_layer=target_layer,
        num_vectors=args.num_vectors,
        num_iters=args.num_iters,
        num_tokens=args.num_tokens,
        batch_size=args.batch_size,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"power_iter_multi_{model_short}_{timestamp}.pt"

    torch.save({
        "vectors": vectors,
        "sigmas": sigmas,
        "model": args.model,
        "source_layer": args.source_layer,
        "target_layer": target_layer,
        "num_vectors": args.num_vectors,
        "num_iters": args.num_iters,
        "num_tokens": args.num_tokens,
        "num_prompts": len(prompts),
        "batch_size": args.batch_size,
        "category": args.category,
        "seed": args.seed,
    }, output_file)

    print(f"\nSaved {vectors.shape[0]} multi-prompt power iteration vectors to {output_file}")
    print(f"Final sigmas: {[f'{s:.0f}' for s in sigmas]}")


if __name__ == "__main__":
    main()
