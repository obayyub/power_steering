"""Vector discovery methods: PI-RR and MELBO."""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from power_steering.utils import (
    format_chat, format_time, orthogonalize,
    get_steering_module, take_hidden_state,
)


# ============================================================================
# PI-RR: Power Iteration with Rayleigh-Ritz correction
# ============================================================================


def find_pi_vectors(
    model,
    tokenizer,
    prompt: str,
    source_layer: int,
    target_layer: int,
    num_vectors: int = 12,
    num_iters: int = 15,
    num_tokens: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, list[float]]:
    """Find top-k right singular vectors of the Jacobian via block power iteration.

    Uses graph reuse: one forward pass per iteration (not per column),
    reusing the computation graph for all k autograd calls. This gives a
    k× speedup over the naive approach with identical results.

    After convergence, applies Rayleigh-Ritz to rotate the subspace to
    the true singular vectors.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        prompt: Raw user message (will be chat-formatted).
        source_layer: Layer whose MLP down_proj gets the perturbation.
        target_layer: Layer where we measure the response.
        num_vectors: How many singular vectors to find.
        num_iters: Power iteration steps.
        num_tokens: Number of final tokens to aggregate over.
        seed: RNG seed for the random orthonormal starting basis.

    Returns:
        (vectors, sigmas) — vectors is [num_vectors, hidden_dim], unit-normed.
        sigmas are the corresponding singular values.
    """
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
        formatted = format_chat(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        def apply_jtj(V):
            """Apply J^T J to all columns of V using one forward pass."""
            nonlocal steering_vec
            k = V.shape[1]

            sv = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)
            steering_vec = sv

            model(inputs["input_ids"])
            target = captured["target"]
            t_slice = target[:, target_token_slice, :].reshape(-1)

            new_cols = []
            for j in range(k):
                u = torch.zeros_like(target, requires_grad=True)
                u_slice = u[:, target_token_slice, :].reshape(-1)

                # Step 1: J^T u (build graph)
                grad = torch.autograd.grad(
                    t_slice, sv, grad_outputs=u_slice,
                    create_graph=True, retain_graph=True,
                )[0]
                # Step 2: J v (forward-mode via backward on the graph)
                jvp = torch.autograd.grad(
                    grad, u_slice, grad_outputs=V[:, j],
                    retain_graph=True,
                )[0]
                # Step 3: J^T (J v)
                new_v = torch.autograd.grad(
                    t_slice, sv, grad_outputs=jvp.detach(),
                    retain_graph=(j < k - 1),
                )[0]
                new_cols.append(new_v)

            steering_vec = None
            return torch.stack(new_cols, dim=1)

        # Random orthonormal starting basis (seeded for reproducibility)
        gen = torch.Generator(device=device).manual_seed(seed)
        V = orthogonalize(torch.randn(
            hidden_dim, num_vectors, device=device, dtype=dtype, generator=gen,
        ))

        t0 = time.time()
        for i in range(num_iters):
            iter_start = time.time()
            new_V = apply_jtj(V)
            V = orthogonalize(new_V)
            iter_elapsed = time.time() - iter_start
            total_elapsed = time.time() - t0
            # +1 for the Rayleigh-Ritz pass at the end
            remaining = iter_elapsed * (num_iters - i - 1 + 1)

            approx_sigmas = [new_V[:, j].norm().item() for j in range(V.shape[1])]
            print(
                f"  PI iter {i}/{num_iters-1}: σ_top={approx_sigmas[0]:.0f}"
                f"  [{format_time(iter_elapsed)}/iter,"
                f" elapsed {format_time(total_elapsed)},"
                f" ~{format_time(remaining)} left]"
            )

        # Rayleigh-Ritz correction
        print("  Computing Rayleigh-Ritz rotation...")
        JtJ_V = apply_jtj(V)
        M = (V.T @ JtJ_V).float()
        M = (M + M.T) / 2  # symmetrize

        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        V = V @ eigenvectors.to(V.dtype)
        sigmas = eigenvalues.clamp(min=0).sqrt().tolist()

        vectors = V.T.detach()  # [k, H]
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        return vectors, sigmas

    finally:
        handle1.remove()
        handle2.remove()
        steering_vec = None


# ============================================================================
# MELBO: Mechanistic Elicitation of Latent Behaviors via Optimization
# ============================================================================


@dataclass
class MELBOConfig:
    """Configuration for MELBO steering vector training."""
    source_layer: int = 7
    target_layer: int | None = None
    num_steps: int = 300
    learning_rate: float = 0.001
    normalization: float = 1.0
    power: float = 2.0
    q: float = 1.0
    orthogonal: bool = True
    target_tokens: slice = field(default_factory=lambda: slice(-2, None))


def _rgetattr(obj, path: str):
    """Get nested attribute using dot notation."""
    return functools.reduce(getattr, path.split("."), obj)


def _get_layers(model):
    """Get transformer layers for common architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Qwen, Llama, Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers  # Pythia
    raise ValueError(f"Unknown model architecture: {type(model)}")


def find_melbo_vectors(
    model,
    tokenizer,
    prompt: str,
    config: MELBOConfig | None = None,
    num_vectors: int = 12,
    verbose: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Discover steering vectors via MELBO optimization.

    Injects a learned bias into the MLP down_proj at the source layer,
    optimizing to maximize activation displacement at the target layer.
    Successive vectors are constrained to be orthogonal.

    Args:
        model: HuggingFace causal LM (will be frozen).
        tokenizer: Matching tokenizer.
        prompt: Raw user message (will be chat-formatted).
        config: MELBO hyperparameters. Uses defaults if None.
        num_vectors: Number of orthogonal vectors to discover.
        verbose: Show progress bars and loss.
        seed: RNG seed for the per-vector random initialisations.

    Returns:
        Tensor of shape [num_vectors, hidden_dim].
    """
    config = config or MELBOConfig()
    layers = _get_layers(model)
    num_layers = len(layers)

    if config.target_layer is None:
        config.target_layer = num_layers - 8

    hidden_dim = layers[0].mlp.down_proj.out_features
    dtype = layers[0].mlp.down_proj.weight.dtype

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    # Add trainable bias to source layer
    down_proj = layers[config.source_layer].mlp.down_proj
    down_proj.bias = nn.Parameter(
        torch.zeros(hidden_dim, device=model.device, dtype=dtype)
    )
    bias = down_proj.bias

    # Format prompt
    formatted = format_chat(tokenizer, prompt)

    # Cache unsteered target activations
    bias.data.zero_()
    with torch.no_grad():
        inputs = tokenizer([formatted], return_tensors="pt", padding=False).to(model.device)
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        unsteered_target = outputs.hidden_states[config.target_layer]
        unsteered_target = unsteered_target[:, config.target_tokens, :].clone()

    learned = torch.zeros(num_vectors, hidden_dim, device=model.device, dtype=dtype)
    num_learned = 0

    def project_orthogonal(vec):
        if num_learned == 0:
            return vec
        U = learned[:num_learned].t() / config.normalization
        return vec - U @ (U.t() @ vec)

    melbo_t0 = time.time()
    gen = torch.Generator(device=model.device).manual_seed(seed)
    # When stdout is redirected to a file (e.g. pipeline.log), tqdm's \r-based
    # progress bar produces garbage. Auto-disable in that case — we already
    # print a clean per-vector summary line below, so progress stays visible.
    import sys as _sys
    iterator = (
        tqdm(range(num_vectors), desc="MELBO", disable=not _sys.stdout.isatty())
        if verbose else range(num_vectors)
    )
    for i in iterator:
        vec_start = time.time()
        # Initialize random direction, orthogonal to previous
        with torch.no_grad():
            init = torch.randn(
                hidden_dim, device=model.device, dtype=dtype, generator=gen,
            )
            if config.orthogonal:
                init = project_orthogonal(init)
            bias.data = nn.functional.normalize(init, dim=0) * config.normalization

        optimizer = optim.AdamW(
            [bias], lr=config.learning_rate,
            betas=(0.9, 0.98), weight_decay=0.0, amsgrad=True,
        )

        losses = []
        for step in range(config.num_steps):
            optimizer.zero_grad()

            outputs = model(inputs["input_ids"], output_hidden_states=True)
            steered = outputs.hidden_states[config.target_layer]
            steered = steered[:, config.target_tokens, :]

            diff = steered - unsteered_target
            norms = diff.norm(dim=-1)
            loss = -norms.pow(config.power).sum().pow(1 / config.q)
            loss.backward()

            # Orthogonality constraints
            if config.orthogonal and num_learned > 0:
                with torch.no_grad():
                    bias.grad = project_orthogonal(bias.grad)

            # Project gradient to sphere tangent, step, re-normalize
            with torch.no_grad():
                radial = torch.dot(bias.grad, bias) / (config.normalization ** 2)
                bias.grad -= radial * bias

            optimizer.step()

            if config.orthogonal and num_learned > 0:
                with torch.no_grad():
                    bias.data = project_orthogonal(bias.data)

            with torch.no_grad():
                bias.data = nn.functional.normalize(bias.data, dim=0) * config.normalization

            losses.append(loss.item())

        if verbose:
            vec_elapsed = time.time() - vec_start
            total_elapsed = time.time() - melbo_t0
            remaining = vec_elapsed * (num_vectors - i - 1)
            print(
                f"  Vector {i}/{num_vectors-1}: loss {losses[0]:.1f} -> {losses[-1]:.1f}"
                f"  [{format_time(vec_elapsed)}/vec,"
                f" elapsed {format_time(total_elapsed)},"
                f" ~{format_time(remaining)} left]"
            )

        with torch.no_grad():
            learned[i] = bias.data.clone()
        num_learned = i + 1

    # Clean up: remove bias
    down_proj.bias = None

    return learned


# ============================================================================
# CAA: Contrastive Activation Addition
# ============================================================================


def find_caa_vector(
    model,
    tokenizer,
    train_prompts: list[dict],
    layer: int,
    capture_site: str = "layer_output",
    log_every: int = 25,
) -> torch.Tensor:
    """Compute a CAA steering vector at the given layer.

    For each training prompt, captures the residual stream at the answer
    letter token (position -2 of `[..., (A, )]`) under the matching answer
    suffix and under the not-matching answer suffix. Returns:

        mean(matching_act - not_matching_act)

    Adding this with positive scale nudges the model toward the matching
    (HHH-aligned) answer, parallel to the `matching_logit_diff` convention
    used elsewhere.

    The default capture site is the layer's residual-stream output
    (`capture_site="layer_output"`), the standard CAA recipe — the
    captured difference encodes everything that diverged between the two
    answer-conditioned trajectories up to layer L. The alternative
    `"down_proj"` captures only the layer-L MLP's incremental contribution
    and produces a much weaker steering direction; kept available mainly
    for cross-method comparison with PI/MELBO at their injection site.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        train_prompts: List of dicts with `question`,
            `matching_answer_full` and `not_matching_answer_full` fields.
        layer: Transformer layer to capture from (and to inject at later).
        capture_site: "layer_output" (default, standard CAA) or "down_proj".
        log_every: Print progress every N prompts.

    Returns:
        Tensor of shape [1, hidden_dim] — unsqueezed so downstream code
        that expects [num_vectors, hidden_dim] works unchanged.
    """
    device = next(model.parameters()).device
    target_module = get_steering_module(model, layer, capture_site)

    captured: dict = {}

    def hook(m, i, o):
        captured["act"] = take_hidden_state(o)

    handle = target_module.register_forward_hook(hook)
    differences: list[torch.Tensor] = []

    try:
        for i, q in enumerate(train_prompts):
            question = q["question"]
            match_answer = q["matching_answer_full"]
            not_match_answer = q["not_matching_answer_full"]

            # Matching activation at letter token (position -2 of "[..., (A, )]")
            match_text = format_chat(tokenizer, question, assistant_start=match_answer)
            inputs = tokenizer(match_text, return_tensors="pt").to(device)
            with torch.no_grad():
                model(inputs["input_ids"])
            match_act = captured["act"][:, -2, :].clone()

            # Not-matching activation
            not_match_text = format_chat(tokenizer, question, assistant_start=not_match_answer)
            inputs = tokenizer(not_match_text, return_tensors="pt").to(device)
            with torch.no_grad():
                model(inputs["input_ids"])
            not_match_act = captured["act"][:, -2, :].clone()

            differences.append((match_act - not_match_act).squeeze(0))

            if (i + 1) % log_every == 0:
                print(f"  CAA: {i+1}/{len(train_prompts)} prompts processed")
    finally:
        handle.remove()

    diffs = torch.stack(differences)
    caa_vector = diffs.mean(dim=0)
    norms = diffs.norm(dim=-1)
    print(f"  CAA capture_site={capture_site}, layer={layer}")
    print(f"  CAA vector norm: {caa_vector.norm():.2f}")
    print(f"  Per-prompt diff norms: mean={norms.mean():.2f}, std={norms.std():.2f}")
    print(f"  Coherence ratio (mean_norm / per_prompt_mean): {caa_vector.norm() / norms.mean():.3f}")

    return caa_vector.unsqueeze(0)
