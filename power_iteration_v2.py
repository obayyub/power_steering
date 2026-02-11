#!/usr/bin/env python3
"""
Optimized power iteration for finding top-k right singular vectors of the Jacobian.

Three methods with different speed/memory tradeoffs:
  graph_reuse     - One forward pass per iteration, reuse graph for all k columns
  batched         - Batch k columns along batch dimension, fully parallelized autograd
  randomized_svd  - Halko-Martinsson-Tropp randomized SVD with oversampling
"""

import json
import time
import torch
import torch.autograd
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
import argparse


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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


def orthogonalize_qr(V):
    """QR-based orthogonalization. More numerically stable than Gram-Schmidt."""
    V_f32 = V.float()
    Q, _ = torch.linalg.qr(V_f32)
    return Q.to(V.dtype)


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


def setup_hooks(model, source_layer, target_layer, batched=False):
    """
    Register capture and steering hooks.

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
            if batched:
                # steering["vec"] is [k, H], broadcast over sequence dim
                return o + steering["vec"].unsqueeze(1)
            else:
                return o + steering["vec"].to(o.device)
        return o

    h1 = target_module.register_forward_hook(capture_hook)
    h2 = down_proj.register_forward_hook(steering_hook)

    return captured, steering, [h1, h2]


def load_prompt(args):
    """Load prompt from --prompt flag or dataset file."""
    if args.prompt:
        return args.prompt
    with open(args.data_path) as f:
        data = json.load(f)
    return data[args.category][0]["question"]


# ---------------------------------------------------------------------------
# Method 1: graph_reuse
# ---------------------------------------------------------------------------

def find_graph_reuse(
    model, tokenizer, prompt, source_layer, target_layer,
    num_vectors=12, num_iters=5, num_tokens=2,
):
    """
    Power iteration reusing the computation graph across all k columns.

    One forward pass per iteration instead of k forward passes.
    Total forward passes: num_iters + 1 (for Rayleigh-Ritz).
    """
    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = model.dtype
    target_token_slice = slice(-num_tokens, None)

    captured, steering, handles = setup_hooks(model, source_layer, target_layer)
    fwd_count = 0

    try:
        formatted = format_chat(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        # Initialize random orthonormal basis
        V = torch.randn(hidden_dim, num_vectors, device=device, dtype=dtype)
        V = orthogonalize(V)

        def apply_jtj(V_in):
            """Apply J^T J to all columns of V_in using one forward pass."""
            nonlocal fwd_count
            k = V_in.shape[1]

            sv = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)
            steering["vec"] = sv

            model(inputs["input_ids"])
            fwd_count += 1
            target = captured["target"]

            t_slice = target[:, target_token_slice, :].reshape(-1)

            new_cols = []
            for j in range(k):
                u = torch.zeros_like(target, requires_grad=True)
                u_slice = u[:, target_token_slice, :].reshape(-1)

                # Step 1: J^T u (with graph creation)
                grad = torch.autograd.grad(
                    t_slice, sv, grad_outputs=u_slice,
                    create_graph=True, retain_graph=True,
                )[0]
                # Step 2: J v (forward-mode via backward)
                jvp = torch.autograd.grad(
                    grad, u_slice, grad_outputs=V_in[:, j],
                    retain_graph=True,
                )[0]
                # Step 3: J^T (J v)
                new_v = torch.autograd.grad(
                    t_slice, sv, grad_outputs=jvp.detach(),
                    retain_graph=(j < k - 1),
                )[0]

                new_cols.append(new_v)

            steering["vec"] = None
            return torch.stack(new_cols, dim=1)

        for i in range(num_iters):
            new_V = apply_jtj(V)
            V = orthogonalize(new_V)

            if i % 5 == 0 or i == num_iters - 1:
                approx_sigmas = [new_V[:, j].norm().item() for j in range(num_vectors)]
                print(f"    Iter {i}: σ ≈ {[f'{s:.0f}' for s in approx_sigmas]}")

        # Rayleigh-Ritz
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


# ---------------------------------------------------------------------------
# Method 2: batched
# ---------------------------------------------------------------------------

def find_batched(
    model, tokenizer, prompt, source_layer, target_layer,
    num_vectors=12, num_iters=5, num_tokens=2,
):
    """
    Power iteration with k columns batched along the batch dimension.

    One forward pass AND one set of backward passes per iteration.
    More memory (k× activations) but fully parallelized autograd.
    """
    k = num_vectors
    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = model.dtype
    target_token_slice = slice(-num_tokens, None)

    captured, steering, handles = setup_hooks(
        model, source_layer, target_layer, batched=True,
    )
    fwd_count = 0

    try:
        formatted = format_chat(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        input_ids_k = inputs["input_ids"].expand(k, -1)

        # Initialize random orthonormal basis
        V = torch.randn(hidden_dim, k, device=device, dtype=dtype)
        V = orthogonalize(V)

        def apply_jtj(V_in):
            """Apply J^T J to all columns using batched forward pass."""
            nonlocal fwd_count
            cols = V_in.shape[1]

            # sv: [cols, H] - each batch element gets its own steering perturbation
            sv = torch.zeros(cols, hidden_dim, device=device, dtype=dtype, requires_grad=True)
            steering["vec"] = sv

            ids = input_ids_k[:cols]
            model(ids)
            fwd_count += 1
            target = captured["target"]  # [cols, seq, H]

            t_slice = target[:, target_token_slice, :]  # [cols, num_tokens, H]
            u = torch.zeros_like(t_slice, requires_grad=True)

            # Step 1: J^T u — autograd keeps batch elements separate because
            # sv[i] only affects target[i] (block-diagonal Jacobian)
            # We compute d/d(sv) of sum_i (t_slice[i] * u[i]).sum()
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

            steering["vec"] = None
            return new_V_T.T  # [H, cols]

        for i in range(num_iters):
            new_V = apply_jtj(V)
            V = orthogonalize(new_V)

            if i % 5 == 0 or i == num_iters - 1:
                approx_sigmas = [new_V[:, j].norm().item() for j in range(k)]
                print(f"    Iter {i}: σ ≈ {[f'{s:.0f}' for s in approx_sigmas]}")

        # Rayleigh-Ritz
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


# ---------------------------------------------------------------------------
# Method 3: randomized_svd
# ---------------------------------------------------------------------------

def find_randomized_svd(
    model, tokenizer, prompt, source_layer, target_layer,
    num_vectors=12, num_iters=5, num_tokens=2,
    oversampling=5, subspace_iters=2,
):
    """
    Randomized SVD via Halko-Martinsson-Tropp.

    Uses graph_reuse internally for apply_jtj. Oversampling and subspace
    iterations improve accuracy.

    Total forward passes: 1 + subspace_iters + 1 = 4 (default).
    """
    k = num_vectors
    p = oversampling
    q = subspace_iters
    l = k + p  # sketch size

    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = model.dtype
    target_token_slice = slice(-num_tokens, None)

    captured, steering, handles = setup_hooks(model, source_layer, target_layer)
    fwd_count = 0

    try:
        formatted = format_chat(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        def apply_jtj(V_in):
            """Apply J^T J to all columns using graph reuse."""
            nonlocal fwd_count
            cols = V_in.shape[1]

            sv = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)
            steering["vec"] = sv

            model(inputs["input_ids"])
            fwd_count += 1
            target = captured["target"]

            t_slice = target[:, target_token_slice, :].reshape(-1)

            new_cols = []
            for j in range(cols):
                u = torch.zeros_like(target, requires_grad=True)
                u_slice = u[:, target_token_slice, :].reshape(-1)

                grad = torch.autograd.grad(
                    t_slice, sv, grad_outputs=u_slice,
                    create_graph=True, retain_graph=True,
                )[0]
                jvp = torch.autograd.grad(
                    grad, u_slice, grad_outputs=V_in[:, j],
                    retain_graph=True,
                )[0]
                new_v = torch.autograd.grad(
                    t_slice, sv, grad_outputs=jvp.detach(),
                    retain_graph=(j < cols - 1),
                )[0]
                new_cols.append(new_v)

            steering["vec"] = None
            return torch.stack(new_cols, dim=1)

        # Halko-Martinsson-Tropp algorithm
        print(f"    Randomized SVD: k={k}, p={p}, q={q}, sketch size l={l}")

        # Random sketch
        Omega = torch.randn(hidden_dim, l, device=device, dtype=dtype)
        Y = apply_jtj(Omega)

        # Subspace iterations for accuracy
        for si in range(q):
            Y = orthogonalize_qr(Y)
            Y = apply_jtj(Y)
            print(f"    Subspace iteration {si + 1}/{q}")

        Q = orthogonalize_qr(Y)

        # Rayleigh-Ritz on the sketch subspace, then take top k
        print("    Computing Rayleigh-Ritz rotation...")
        V, sigmas_all = rayleigh_ritz(Q, apply_jtj)

        # Take top k
        V = V[:, :k]
        sigmas = sigmas_all[:k]

        print(f"    True σ = {[f'{s:.0f}' for s in sigmas]}")
        print(f"    Forward passes: {fwd_count}")

        vectors = V.T.detach()  # [k, H]
        return vectors, sigmas, fwd_count

    finally:
        for h in handles:
            h.remove()
        steering["vec"] = None


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def compare_methods(
    model, tokenizer, prompt, source_layer, target_layer,
    num_vectors=12, num_iters=5, num_tokens=2,
    oversampling=5, subspace_iters=2,
):
    """Run all three methods and compare results."""
    device = next(model.parameters()).device
    results = {}

    methods = [
        ("graph_reuse", lambda: find_graph_reuse(
            model, tokenizer, prompt, source_layer, target_layer,
            num_vectors, num_iters, num_tokens,
        )),
        ("batched", lambda: find_batched(
            model, tokenizer, prompt, source_layer, target_layer,
            num_vectors, num_iters, num_tokens,
        )),
        ("randomized_svd", lambda: find_randomized_svd(
            model, tokenizer, prompt, source_layer, target_layer,
            num_vectors, num_iters, num_tokens,
            oversampling, subspace_iters,
        )),
    ]

    for name, fn in methods:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        vectors, sigmas, fwd_count = fn()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        results[name] = {
            "vectors": vectors,
            "sigmas": sigmas,
            "fwd_count": fwd_count,
            "time": elapsed,
        }
        print(f"    Time: {elapsed:.1f}s, Forward passes: {fwd_count}")

    # --- Comparison report ---
    print(f"\n{'='*60}")
    print("COMPARISON REPORT")
    print(f"{'='*60}")

    # Timing and forward pass table
    print(f"\n{'Method':<20} {'Time (s)':>10} {'Fwd passes':>12}")
    print("-" * 44)
    for name, r in results.items():
        print(f"{name:<20} {r['time']:>10.1f} {r['fwd_count']:>12}")

    # Singular value comparison
    print(f"\n{'Vec':<5}", end="")
    for name in results:
        print(f" {name:>16}", end="")
    print()
    print("-" * (5 + 17 * len(results)))
    for j in range(num_vectors):
        print(f"{j:<5}", end="")
        for name in results:
            print(f" {results[name]['sigmas'][j]:>16.1f}", end="")
        print()

    # Subspace alignment: principal cosines between each pair
    names = list(results.keys())
    print(f"\nSubspace alignment (principal cosines):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            V1 = results[names[i]]["vectors"]  # [k, H]
            V2 = results[names[j]]["vectors"]  # [k, H]
            # SVD of V1 @ V2^T gives principal cosines
            S = torch.linalg.svdvals(V1.float() @ V2.float().T)
            cosines = S.clamp(max=1.0).tolist()
            mean_cos = sum(cosines) / len(cosines)
            min_cos = min(cosines)
            print(f"  {names[i]} vs {names[j]}:")
            print(f"    cosines: {[f'{c:.4f}' for c in cosines]}")
            print(f"    mean={mean_cos:.4f}, min={min_cos:.4f}")

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimized power iteration for steering vector discovery",
    )
    parser.add_argument(
        "--method", default="graph_reuse",
        choices=["graph_reuse", "batched", "randomized_svd", "compare"],
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
    parser.add_argument("--output-dir", default="vectors")
    parser.add_argument("--oversampling", type=int, default=5, help="Oversampling for randomized SVD")
    parser.add_argument("--subspace-iters", type=int, default=2, help="Subspace iterations for randomized SVD")
    args = parser.parse_args()

    # Load prompt
    prompt = load_prompt(args)
    print(f"Prompt: {prompt[:80]}...")

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

    if args.method == "compare":
        compare_methods(
            model, tokenizer, prompt,
            args.source_layer, target_layer,
            args.num_vectors, args.num_iters, args.num_tokens,
            args.oversampling, args.subspace_iters,
        )
        return

    # Run selected method
    method_fn = {
        "graph_reuse": find_graph_reuse,
        "batched": find_batched,
        "randomized_svd": find_randomized_svd,
    }[args.method]

    kwargs = dict(
        model=model, tokenizer=tokenizer, prompt=prompt,
        source_layer=args.source_layer, target_layer=target_layer,
        num_vectors=args.num_vectors, num_iters=args.num_iters,
        num_tokens=args.num_tokens,
    )
    if args.method == "randomized_svd":
        kwargs["oversampling"] = args.oversampling
        kwargs["subspace_iters"] = args.subspace_iters

    vectors, sigmas, fwd_count = method_fn(**kwargs)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"power_iter_v2_{model_short}_{timestamp}.pt"

    torch.save({
        "vectors": vectors,
        "sigmas": sigmas,
        "model": args.model,
        "method": args.method,
        "source_layer": args.source_layer,
        "target_layer": target_layer,
        "num_vectors": args.num_vectors,
        "num_iters": args.num_iters,
        "num_tokens": args.num_tokens,
        "category": args.category,
        "prompt": prompt,
        "fwd_count": fwd_count,
    }, output_file)

    print(f"\nSaved {vectors.shape[0]} vectors to {output_file}")
    print(f"Sigmas: {[f'{s:.0f}' for s in sigmas]}")
    print(f"Forward passes: {fwd_count}")


if __name__ == "__main__":
    main()
