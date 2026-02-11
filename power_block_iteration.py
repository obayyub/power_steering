#!/usr/bin/env python3
"""
Batched block power iteration for finding top-k right singular vectors of the Jacobian.

Supports both single-prompt and multi-prompt modes:
  - Single prompt: one forward pass per iteration (fast)
  - Multi-prompt: one forward pass per prompt per iteration, accumulates J^T J across prompts

The batched approach expands k columns along the batch dimension so all k
columns are processed in a single forward + backward pass per prompt.
"""

import json
import random
import torch
import torch.autograd
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


def rayleigh_ritz(V, apply_jtj_fn):
    """
    Rayleigh-Ritz correction: rotate power iteration vectors to true singular vectors.

    Args:
        V: [H, k] orthonormal columns from power iteration
        apply_jtj_fn: function that takes [H, k] and returns [H, k] = J^T J @ V

    Returns:
        V_rotated: [H, k] true singular vectors (columns)
        sigmas: list of singular values (descending)
    """
    JtJ_V = apply_jtj_fn(V)
    M = (V.T @ JtJ_V).float()

    # Symmetrize (should be symmetric, but numerical errors)
    M = (M + M.T) / 2

    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    # eigh returns ascending order, we want descending
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    V_rotated = V @ eigenvectors.to(V.dtype)
    sigmas = eigenvalues.clamp(min=0).sqrt().tolist()

    return V_rotated, sigmas


def setup_hooks(model, source_layer, target_layer):
    """
    Register capture and steering hooks for batched mode.

    The steering hook adds steering["vec"] (shape [k, H]) to each batch element,
    broadcasting over the sequence dimension.

    Returns:
        captured: dict with 'target' key populated after forward pass
        steering: dict with 'vec' key to set steering vector
        handles: list of hook handles to remove later
    """
    down_proj = model.model.layers[source_layer].mlp.down_proj
    target_module = model.model.layers[target_layer]

    captured = {}
    steering = {"vec": None}

    def capture_hook(m, i, o):
        captured["target"] = o[0] if isinstance(o, tuple) else o

    def steering_hook(m, i, o):
        if steering["vec"] is not None:
            # steering["vec"] is [k, H], broadcast over sequence dim
            return o + steering["vec"].unsqueeze(1)
        return o

    h1 = target_module.register_forward_hook(capture_hook)
    h2 = down_proj.register_forward_hook(steering_hook)

    return captured, steering, [h1, h2]


def find_vectors(
    model,
    tokenizer,
    prompts: list[str],
    source_layer: int,
    target_layer: int,
    num_vectors: int = 12,
    num_iters: int = 5,
    num_tokens: int = 2,
):
    """
    Find top-k right singular vectors via batched block power iteration.

    For single prompt, one forward pass per iteration.
    For multiple prompts, one forward pass per prompt per iteration,
    accumulating J_i^T J_i across prompts.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        prompts: list of prompts (single or multiple)
        source_layer: layer to inject steering perturbation (down_proj)
        target_layer: layer to capture output from
        num_vectors: number of singular vectors to find (k)
        num_iters: number of power iteration steps
        num_tokens: number of final tokens to use for Jacobian

    Returns:
        vectors: [k, H] tensor of singular vectors (rows)
        sigmas: list of singular values (descending)
        fwd_count: total number of forward passes
    """
    k = num_vectors
    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = model.dtype
    target_token_slice = slice(-num_tokens, None)

    print(f"\nBatched block power iteration")
    print(f"  Source layer: {source_layer}, Target layer: {target_layer}")
    print(f"  k={k}, iters={num_iters}, tokens={num_tokens}")
    print(f"  Prompts: {len(prompts)}")

    captured, steering, handles = setup_hooks(model, source_layer, target_layer)
    fwd_count = 0

    try:
        # Tokenize each prompt individually (no cross-prompt padding needed
        # since we process one prompt at a time, expanded to k batch elements)
        prompt_inputs = []
        for p in prompts:
            formatted = format_chat(tokenizer, p)
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]  # [1, seq]
            prompt_inputs.append(input_ids)

        # Initialize random orthonormal basis
        V = torch.randn(hidden_dim, k, device=device, dtype=dtype)
        V = orthogonalize(V)

        def apply_jtj(V_in):
            """Apply sum_i(J_i^T J_i) to all columns of V_in using batched forward passes."""
            nonlocal fwd_count
            cols = V_in.shape[1]
            new_V = torch.zeros(hidden_dim, cols, device=device, dtype=dtype)

            for input_ids in prompt_inputs:
                # Expand single prompt to k batch elements: [1, seq] -> [cols, seq]
                ids = input_ids.expand(cols, -1)

                # Each batch element gets its own steering perturbation
                sv = torch.zeros(cols, hidden_dim, device=device, dtype=dtype, requires_grad=True)
                steering["vec"] = sv

                model(ids)
                fwd_count += 1
                target = captured["target"]  # [cols, seq, H]

                t_slice = target[:, target_token_slice, :]  # [cols, num_tokens, H]
                u = torch.zeros_like(t_slice, requires_grad=True)

                # Step 1: J^T u — d/d(sv) of sum_i (t_slice[i] * u[i]).sum()
                # autograd keeps batch elements separate (block-diagonal Jacobian)
                loss1 = (t_slice * u).sum()
                grad = torch.autograd.grad(loss1, sv, create_graph=True, retain_graph=True)[0]
                # grad: [cols, H] where grad[i] = J_i^T @ u[i].flatten()

                # Step 2: J v — d/d(u) of sum_i (grad[i] * V_in[:, i]).sum()
                loss2 = (grad * V_in.T[:cols]).sum()
                jvp = torch.autograd.grad(loss2, u, retain_graph=True)[0]
                # jvp: [cols, num_tokens, H] where jvp[i] = J_i @ V_in[:, i]

                # Step 3: J^T (J v) — d/d(sv) of sum_i (t_slice[i] * jvp[i].detach()).sum()
                loss3 = (t_slice * jvp.detach()).sum()
                new_V_T = torch.autograd.grad(loss3, sv)[0]
                # new_V_T: [cols, H] where new_V_T[i] = J_i^T @ J_i @ V_in[:, i]

                new_V += new_V_T.T  # accumulate across prompts

            steering["vec"] = None
            return new_V  # [H, cols]

        for i in range(num_iters):
            new_V = apply_jtj(V)
            V = orthogonalize(new_V)

            if i % 5 == 0 or i == num_iters - 1:
                approx_sigmas = [new_V[:, j].norm().item() for j in range(k)]
                print(f"    Iter {i}: σ ≈ {[f'{s:.0f}' for s in approx_sigmas]}")

        # Rayleigh-Ritz correction
        print("    Computing Rayleigh-Ritz rotation...")
        V, sigmas = rayleigh_ritz(V, apply_jtj)

        print(f"    True σ = {[f'{s:.0f}' for s in sigmas]}")
        print(f"    Forward passes: {fwd_count}")

        vectors = V.T.detach()  # [k, H]
        return vectors, sigmas, fwd_count

    finally:
        for h in handles:
            h.remove()
        steering["vec"] = None


def main():
    parser = argparse.ArgumentParser(
        description="Batched block power iteration for steering vector discovery",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--source-layer", type=int, default=3)
    parser.add_argument("--target-layer", type=int, default=None, help="Default: num_layers - 8")
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--prompt", default=None, help="Custom prompt (overrides dataset)")
    parser.add_argument("--data-path", default="data/corrigibility_eval.json")
    parser.add_argument("--category", default="corrigible-neutral-HHH")
    parser.add_argument("--num-prompts", type=int, default=1,
                        help="Number of prompts (1 = single-prompt mode)")
    parser.add_argument("--output-dir", default="vectors")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load prompts
    if args.prompt:
        prompts = [args.prompt]
        print(f"Using custom prompt: {args.prompt[:80]}...")
    else:
        with open(args.data_path) as f:
            data = json.load(f)
        all_prompts = [item["question"] for item in data[args.category]]
        if args.num_prompts >= len(all_prompts):
            prompts = all_prompts
        else:
            prompts = random.sample(all_prompts, args.num_prompts)
        print(f"Loaded {len(prompts)} prompt(s) from {args.data_path} ({args.category})")

    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",  # Required for double autograd
    )

    num_layers = len(model.model.layers)
    target_layer = args.target_layer if args.target_layer is not None else num_layers - 8
    print(f"Layers: {num_layers}, Hidden dim: {model.config.hidden_size}")
    print(f"Source: {args.source_layer}, Target: {target_layer}")

    vectors, sigmas, fwd_count = find_vectors(
        model, tokenizer, prompts,
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
    output_file = output_dir / f"power_block_iter_{model_short}_{timestamp}.pt"

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
        "category": args.category if not args.prompt else None,
        "prompt": prompts[0] if len(prompts) == 1 else prompts,
        "seed": args.seed,
        "fwd_count": fwd_count,
    }, output_file)

    print(f"\nSaved {vectors.shape[0]} vectors to {output_file}")
    print(f"Sigmas: {[f'{s:.0f}' for s in sigmas]}")
    print(f"Forward passes: {fwd_count}")


if __name__ == "__main__":
    main()
