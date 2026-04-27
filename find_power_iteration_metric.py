#!/usr/bin/env python3
"""
Metric power iteration: multi-prompt block power iteration with diagonal variance metrics.

Computes diagonal variance of activations across prompts at source and target layers,
then uses these as metrics in a generalized eigenvalue formulation of the power iteration.

Uses prewhitening to reduce the generalized eigenvalue problem to standard form:
    A_tilde = G_s^{-1/2} (sum_i J_i^T G_t J_i) G_s^{-1/2}
finds eigenvectors of A_tilde in standard Euclidean norm, then unwhitens.

Runs 5 configurations:
    baseline: no metric (standard multi-prompt PI)
    var_var:  G_s = diag(var_s), G_t = diag(var_t)
    var_inv:  G_s = diag(var_s), G_t = diag(1/var_t)
    inv_var:  G_s = diag(1/var_s), G_t = diag(var_t)
    inv_inv:  G_s = diag(1/var_s), G_t = diag(1/var_t)

Usage:
    python find_power_iteration_metric.py --model Qwen/Qwen3-14B --num-prompts 32
"""

import torch
import torch.autograd
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
import argparse
import json
import random


METRIC_CONFIGS = {
    "baseline": ("none", "none"),
    "var_var": ("var", "var"),
    "var_inv": ("var", "inv"),
    "inv_var": ("inv", "var"),
    "inv_inv": ("inv", "inv"),
}


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


def compute_variance(model, tokenizer, prompts, source_layer, target_layer, batch_size=8):
    """
    Compute per-coordinate variance of activations at source and target layers.

    Source: MLP down_proj output at source_layer, last token position.
    Target: layer output (residual stream) at target_layer, last token position.
    Variance computed across prompts.
    """
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size

    source_acts = []
    target_acts = []

    down_proj = model.model.layers[source_layer].mlp.down_proj
    target_module = model.model.layers[target_layer]

    def source_hook(m, i, o):
        source_acts.append(o[:, -1, :].detach())

    def target_hook(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        target_acts.append(out[:, -1, :].detach())

    h1 = down_proj.register_forward_hook(source_hook)
    h2 = target_module.register_forward_hook(target_hook)

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
        h1.remove()
        h2.remove()
        tokenizer.padding_side = original_padding_side

    source_all = torch.cat(source_acts, dim=0).float()  # [num_prompts, hidden]
    target_all = torch.cat(target_acts, dim=0).float()

    var_s = source_all.var(dim=0)  # [hidden]
    var_t = target_all.var(dim=0)

    print(f"  Source variance: min={var_s.min():.6f}, max={var_s.max():.2f}, "
          f"median={var_s.median():.4f}")
    print(f"  Target variance: min={var_t.min():.6f}, max={var_t.max():.2f}, "
          f"median={var_t.median():.4f}")
    print(f"  Source near-zero (< 1e-8): {(var_s < 1e-8).sum().item()}/{hidden_dim}")
    print(f"  Target near-zero (< 1e-8): {(var_t < 1e-8).sum().item()}/{hidden_dim}")

    return var_s.to(device), var_t.to(device)


def get_weights(var_s, var_t, source_type, target_type, eps=1e-8):
    """
    Compute prewhitening and target metric weights.

    source_pre = G_s^{-1/2}, applied before JVP and after VJP (same weights both sides).
    target_w = G_t, applied to JVP result between steps 2 and 3.
    """
    var_s = var_s.float().clamp(min=eps)
    var_t = var_t.float().clamp(min=eps)

    if source_type == "var":
        source_pre = 1.0 / var_s.sqrt()
    elif source_type == "inv":
        source_pre = var_s.sqrt()
    else:
        source_pre = torch.ones_like(var_s)

    if target_type == "var":
        target_w = var_t
    elif target_type == "inv":
        target_w = 1.0 / var_t
    else:
        target_w = torch.ones_like(var_t)

    return source_pre, target_w


def find_vectors_metric(
    model, tokenizer, prompts,
    source_layer, target_layer,
    source_pre, target_w,
    num_vectors=12, num_iters=5, num_tokens=2, batch_size=8,
):
    """
    Multi-prompt block power iteration with metric prewhitening.

    Finds eigenvectors of:
        G_s^{-1/2} (sum_i J_i^T G_t J_i) G_s^{-1/2}
    in standard Euclidean norm, then unwhitens to get steering vectors.
    """
    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = model.dtype
    target_token_slice = slice(-num_tokens, None)

    source_pre_d = source_pre.to(dtype=dtype, device=device)
    target_w_d = target_w.to(dtype=dtype, device=device)

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

    h1 = target_module.register_forward_hook(capture_hook)
    h2 = down_proj.register_forward_hook(steering_hook)

    try:
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        formatted = [format_chat(tokenizer, p) for p in prompts]
        batches = []
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i:i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(device)
            batches.append(inputs)

        tokenizer.padding_side = original_padding_side
        print(f"    {len(batches)} batches, {len(prompts)} prompts")

        def apply_metrized_jtj(V):
            """Apply G_s^{-1/2} sum_i(J_i^T G_t J_i) G_s^{-1/2} to columns of V."""
            nonlocal steering_vec
            new_V = torch.zeros_like(V)

            for col in range(V.shape[1]):
                v = V[:, col]
                # Prewhiten: actual perturbation is G_s^{-1/2} v
                v_actual = source_pre_d * v
                accumulated = torch.zeros_like(v)

                for batch_inputs in batches:
                    sv = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)
                    steering_vec = sv

                    model(**batch_inputs)
                    target = captured["target"]

                    u = torch.zeros_like(target, requires_grad=True)
                    t_slice = target[:, target_token_slice, :].reshape(-1)
                    u_slice = u[:, target_token_slice, :].reshape(-1)

                    # Step 1: VJP with dummy u
                    grad = torch.autograd.grad(
                        t_slice, sv, grad_outputs=u_slice, create_graph=True
                    )[0]

                    # Step 2: JVP with prewhitened v
                    jvp = torch.autograd.grad(grad, u_slice, grad_outputs=v_actual.to(grad.device))[0]

                    # Apply target metric G_t (move weights to jvp's device for multi-GPU)
                    jvp_shaped = jvp.view(-1, hidden_dim)
                    jvp_weighted = (jvp_shaped * target_w_d.to(jvp.device).unsqueeze(0)).reshape(-1)

                    # Step 3: VJP with weighted JVP
                    jtj_v = torch.autograd.grad(
                        t_slice, sv, grad_outputs=jvp_weighted.detach()
                    )[0]

                    # Post-whiten (same G_s^{-1/2} as prewhiten)
                    jtj_v = jtj_v * source_pre_d.to(jtj_v.device)

                    accumulated += jtj_v

                new_V[:, col] = accumulated

            return new_V

        # Initialize random orthonormal basis
        V = torch.randn(hidden_dim, num_vectors, device=device, dtype=dtype)
        V = orthogonalize(V)

        for i in range(num_iters):
            new_V = apply_metrized_jtj(V)
            V = orthogonalize(new_V)

            if i % 3 == 0 or i == num_iters - 1:
                approx_sigmas = [new_V[:, j].norm().item() for j in range(num_vectors)]
                print(f"    Iter {i}: σ ≈ {[f'{s:.0f}' for s in approx_sigmas]}")

        # Rayleigh-Ritz correction
        print("    Computing Rayleigh-Ritz rotation...")
        JtJ_V = apply_metrized_jtj(V)
        M = (V.T @ JtJ_V).float()
        M = (M + M.T) / 2

        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        V = V @ eigenvectors.to(V.dtype)
        sigmas = eigenvalues.clamp(min=0).sqrt().tolist()

        print(f"    True σ = {[f'{s:.0f}' for s in sigmas]}")

        # Unwhiten: v_steering = G_s^{-1/2} v_whitened
        vectors = torch.stack([(source_pre_d * V[:, j]).detach() for j in range(num_vectors)])

        # Normalize to unit norm
        norms = vectors.norm(dim=1, keepdim=True)
        vectors = vectors / norms.clamp(min=1e-10)

        return vectors, sigmas

    finally:
        h1.remove()
        h2.remove()
        steering_vec = None


def main():
    parser = argparse.ArgumentParser(
        description="Metric power iteration: find steering vectors with variance metrics"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--target-layer", type=int, default=None, help="Default: num_layers - 8")
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-path", default="data/corrigibility_eval.json")
    parser.add_argument("--category", default="corrigible-neutral-HHH")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="vectors")
    parser.add_argument(
        "--configs", default=None,
        help="Comma-separated configs to run (default: all). "
             "Options: baseline,var_var,var_inv,inv_var,inv_inv"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Select configs
    if args.configs:
        config_names = [c.strip() for c in args.configs.split(",")]
        configs = {k: METRIC_CONFIGS[k] for k in config_names}
    else:
        configs = METRIC_CONFIGS

    # Load prompts
    print(f"Loading prompts from {args.data_path}...")
    with open(args.data_path) as f:
        data = json.load(f)
    all_prompts = [item["question"] for item in data[args.category]]
    if args.num_prompts < len(all_prompts):
        prompts = random.sample(all_prompts, args.num_prompts)
    else:
        prompts = all_prompts
    print(f"Using {len(prompts)} prompts from {args.category}")

    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )

    num_layers = len(model.model.layers)
    target_layer = args.target_layer if args.target_layer else num_layers - 8
    hidden_dim = model.config.hidden_size

    print(f"Layers: {num_layers}, Hidden dim: {hidden_dim}")
    print(f"Source: {args.source_layer}, Target: {target_layer}")
    print(f"Configs to run: {list(configs.keys())}")

    # Compute variance across prompts
    print("\nComputing activation variance across prompts...")
    var_s, var_t = compute_variance(
        model, tokenizer, prompts,
        args.source_layer, target_layer,
        batch_size=args.batch_size,
    )

    # Run each configuration
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_summary = {}

    for config_name, (source_type, target_type) in configs.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name} (source={source_type}, target={target_type})")
        print(f"{'='*60}")

        # Reset random state so each config starts from the same V_0
        torch.manual_seed(args.seed)

        source_pre, target_w = get_weights(var_s, var_t, source_type, target_type)

        print(f"  source_pre: min={source_pre.min():.4f}, max={source_pre.max():.4f}, "
              f"median={source_pre.median():.4f}")
        print(f"  target_w:   min={target_w.min():.4f}, max={target_w.max():.4f}, "
              f"median={target_w.median():.4f}")

        vectors, sigmas = find_vectors_metric(
            model, tokenizer, prompts,
            args.source_layer, target_layer,
            source_pre, target_w,
            num_vectors=args.num_vectors,
            num_iters=args.num_iters,
            num_tokens=args.num_tokens,
            batch_size=args.batch_size,
        )

        output_file = output_dir / f"metric_pi_{config_name}_{model_short}_{timestamp}.pt"
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
            "metric_config": config_name,
            "source_metric_type": source_type,
            "target_metric_type": target_type,
            "var_source": var_s.cpu(),
            "var_target": var_t.cpu(),
        }, output_file)

        print(f"  Saved to {output_file}")
        results_summary[config_name] = {
            "file": str(output_file),
            "sigmas": sigmas,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, info in results_summary.items():
        s = info["sigmas"]
        print(f"  {name:12s}: σ = [{s[0]:.0f}, {s[1]:.0f}, {s[2]:.0f}, ...] -> {info['file']}")

    print(f"\nEvaluate with:")
    print(f"  python eval_steering.py --vectors <path> --source-layer {args.source_layer}")


if __name__ == "__main__":
    main()
