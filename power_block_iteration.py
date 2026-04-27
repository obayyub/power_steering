#!/usr/bin/env python3
"""
Batched block power iteration for finding top-k right singular vectors of the Jacobian.

Supports both single-prompt and multi-prompt modes:
  - Single prompt: one forward pass per iteration (fast)
  - Multi-prompt: one forward pass per prompt per iteration, accumulates J^T J across prompts

Optional target metric: weight the target (cotangent) space by per-coordinate variance
to change which output changes the optimization cares about.

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


def compute_target_variance(model, tokenizer, prompts, target_layer, batch_size=8):
    """Compute per-coordinate variance of activations at target layer across prompts.

    Hooks the target layer module output, runs forward passes, returns variance [H].
    """
    device = next(model.parameters()).device
    target_module = model.model.layers[target_layer]
    acts = []

    def hook(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        acts.append(out[:, -1, :].detach())

    h = target_module.register_forward_hook(hook)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted = [format_chat(tokenizer, p) for p in prompts]
    try:
        with torch.no_grad():
            for i in range(0, len(formatted), batch_size):
                batch = formatted[i:i + batch_size]
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to(device)
                model(**inputs)
    finally:
        h.remove()
        tokenizer.padding_side = original_padding_side

    all_acts = torch.cat(acts, dim=0).float()  # [num_prompts, H]
    var = all_acts.var(dim=0)  # [H]
    print(f"  Target variance (layer {target_layer}): min={var.min():.6f}, "
          f"max={var.max():.2f}, median={var.median():.4f}")
    return var.to(device)


def get_target_weights(var_t, metric_type, eps=1e-8):
    """Convert target variance to weights for the target metric."""
    var_t = var_t.float().clamp(min=eps)
    if metric_type == "var":
        return var_t
    elif metric_type == "inv":
        return 1.0 / var_t
    else:
        return None


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
    target_weights: torch.Tensor | None = None,
    batch_vectors: int | None = None,
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
        target_weights: optional [H] tensor of per-coordinate target weights (G_t diagonal)
        batch_vectors: process this many vectors at once (default: all k).
            Set to 1 for large models to reduce memory.

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

    # Prepare target weights for broadcasting: [1, 1, H]
    tw = None
    if target_weights is not None:
        tw = target_weights.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

    bv = batch_vectors if batch_vectors is not None else k

    print(f"\nBatched block power iteration")
    print(f"  Source layer: {source_layer}, Target layer: {target_layer}")
    print(f"  k={k}, iters={num_iters}, tokens={num_tokens}, batch_vectors={bv}")
    print(f"  Prompts: {len(prompts)}")
    if tw is not None:
        print(f"  Target metric: enabled (weight range [{target_weights.min():.4f}, {target_weights.max():.2f}])")

    captured, steering, handles = setup_hooks(model, source_layer, target_layer)
    fwd_count = 0

    try:
        # Tokenize each prompt individually (no cross-prompt padding needed
        # since we process one prompt at a time, expanded to batch elements)
        prompt_inputs = []
        for p in prompts:
            formatted = format_chat(tokenizer, p)
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]  # [1, seq]
            prompt_inputs.append(input_ids)

        # Initialize random orthonormal basis
        V = torch.randn(hidden_dim, k, device=device, dtype=dtype)
        V = orthogonalize(V)

        def apply_jtj_chunk(V_chunk):
            """Apply sum_i(J_i^T G_t J_i) to columns of V_chunk."""
            nonlocal fwd_count
            cols = V_chunk.shape[1]
            new_V = torch.zeros(hidden_dim, cols, device=device, dtype=dtype)

            for input_ids in prompt_inputs:
                ids = input_ids.expand(cols, -1)
                sv = torch.zeros(cols, hidden_dim, device=device, dtype=dtype, requires_grad=True)
                steering["vec"] = sv

                model(ids)
                fwd_count += 1
                target = captured["target"]

                t_slice = target[:, target_token_slice, :]
                u = torch.zeros_like(t_slice, requires_grad=True)

                loss1 = (t_slice * u).sum()
                grad = torch.autograd.grad(loss1, sv, create_graph=True, retain_graph=True)[0]

                loss2 = (grad * V_chunk.T[:cols]).sum()
                jvp = torch.autograd.grad(loss2, u, retain_graph=True)[0]

                if tw is not None:
                    jvp = jvp * tw

                loss3 = (t_slice * jvp.detach()).sum()
                new_V_T = torch.autograd.grad(loss3, sv)[0]

                new_V += new_V_T.T

            steering["vec"] = None
            return new_V

        def apply_jtj(V_in):
            """Apply J^T G_t J to all columns, chunked by batch_vectors."""
            if bv >= V_in.shape[1]:
                return apply_jtj_chunk(V_in)
            # Process in chunks
            results = []
            for start in range(0, V_in.shape[1], bv):
                end = min(start + bv, V_in.shape[1])
                results.append(apply_jtj_chunk(V_in[:, start:end]))
            return torch.cat(results, dim=1)

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
    parser.add_argument("--target-metric", default="none", choices=["none", "var", "inv"],
                        help="Target metric: none (standard), var (upweight high-variance), "
                             "inv (upweight low-variance)")
    parser.add_argument("--batch-vectors", type=int, default=None,
                        help="Process this many vectors at once (default: all). "
                             "Set to 1 for large models.")
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

    # Compute target metric if requested
    target_weights = None
    if args.target_metric != "none":
        print(f"\nComputing target variance for metric '{args.target_metric}'...")
        var_t = compute_target_variance(model, tokenizer, prompts, target_layer)
        target_weights = get_target_weights(var_t, args.target_metric)
        print(f"  Target weights: min={target_weights.min():.4f}, "
              f"max={target_weights.max():.2f}, median={target_weights.median():.4f}")

    vectors, sigmas, fwd_count = find_vectors(
        model, tokenizer, prompts,
        source_layer=args.source_layer,
        target_layer=target_layer,
        num_vectors=args.num_vectors,
        num_iters=args.num_iters,
        num_tokens=args.num_tokens,
        target_weights=target_weights,
        batch_vectors=args.batch_vectors,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_tag = f"_tgt{args.target_metric}" if args.target_metric != "none" else ""
    output_file = output_dir / f"power_block_iter{metric_tag}_{model_short}_{timestamp}.pt"

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
        "target_metric": args.target_metric,
    }, output_file)

    print(f"\nSaved {vectors.shape[0]} vectors to {output_file}")
    print(f"Sigmas: {[f'{s:.0f}' for s in sigmas]}")
    print(f"Forward passes: {fwd_count}")


if __name__ == "__main__":
    main()
