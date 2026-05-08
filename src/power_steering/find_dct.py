"""Exponential DCT (Deep Causal Transcoding) — Mack 2024.

Reference: A. Mack, "Deep Causal Transcoding: A Framework for Mechanistically
Eliciting Latent Behaviors in Language Models", LessWrong / AI Alignment Forum,
2024-12-03. https://www.lesswrong.com/posts/fSRg5qs9TPbNy3sm5/

This file implements the exponential variant of DCT (Mack's recommended
default), trained via Orthogonalized Gradient Iteration (OGI). The MLP
approximation is

    Δ_hat(θ) = Σ_i α_i · (exp(<v_i, θ>) − 1) · u_i

with ‖u_i‖ = ‖v_i‖ = 1. OGI iterates infinite-step gradient ascent on the
"causal importance" term Σ_i <u_i, Δ^{s→t}(R v_i)>, with QR-orthogonalization
on V between steps to encourage feature diversity.

We share PI and MELBO's injection convention: the steering bias is added at
the source layer's MLP `down_proj` output (mathematically equivalent to
adding to the residual stream after the source block). This keeps DCT vectors
drop-in compatible with the existing eval pipeline at capture_site="down_proj".

Notable implementation choices vs Mack's blog:
- Initialisation: random orthonormal V (no Algorithm 1' randomized-SVD warm
  start). The OGI loop converges from random init for our small m=12 regime;
  adding randomized-SVD init would help at large m but is unnecessary here.
- Calibration: bisection on the ratio sqrt(‖linear‖² / ‖residual‖²) = λ over
  random calibration directions, matching the formal-paper version of the
  procedure (Mack ICLR 2026 eq. 5). Default λ=0.5.
- Batched forward+backward across m features in a single pass; gradients on
  V are obtained via one autograd.grad call per OGI step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.func import jvp as functional_jvp

from power_steering.utils import format_chat, format_time


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DCTConfig:
    """Hyperparameters for exponential-DCT training via OGI."""
    source_layer: int = 7
    target_layer: int | None = None
    num_features: int = 12          # m in Mack's notation
    num_iters: int = 10             # τ — OGI steps
    lambda_cal: float = 0.5         # ratio target sqrt(‖residual‖²/‖linear‖²) = λ
    n_cal: int = 30                 # # random directions used for R calibration (Mack default)
    cal_R_lo: float = 0.001         # bisection bracket — low end (per Mack)
    cal_R_hi: float = 100.0         # bisection bracket — high end (per Mack)
    cal_iters: int = 25             # bisection iterations
    target_tokens: slice = field(default_factory=lambda: slice(-2, None))
    capture_site: str = "down_proj"  # matches PI/MELBO; eval pipeline keys on this


# ============================================================================
# Helpers
# ============================================================================


def _get_source_module(model, layer: int, capture_site: str):
    """Resolve the module to attach the steering hook to."""
    block = model.model.layers[layer]
    if capture_site == "down_proj":
        return block.mlp.down_proj
    if capture_site == "layer_output":
        return block
    raise ValueError(f"Unknown capture_site: {capture_site!r}")


def _add_to_output(o, addend):
    """Add `addend` to a hook output, handling both tensor and tuple outputs."""
    if isinstance(o, tuple):
        h = o[0]
        return (h + addend,) + tuple(o[1:])
    return o + addend


def _compute_delta(
    model,
    source_layer: int,
    target_layer: int,
    capture_site: str,
    input_ids: torch.Tensor,
    biases: torch.Tensor,
    target_token_slice: slice,
) -> torch.Tensor:
    """Compute Δ^{s→t}(bias_i) for each row i of `biases`.

    Forward pass with batch size m+1: indices 0..m-1 carry the bias rows,
    index m is the unperturbed baseline. The hook adds `biases[i]` to the
    source-layer output at batch index i (broadcast across token positions).

    Returns the *change* in target-layer activations at the target_token_slice
    positions: shape [m, num_target_tokens, hidden_dim]. Differentiable in
    `biases` (i.e. autograd will flow back through the model to `biases` if
    that tensor requires grad).
    """
    m, hidden_dim = biases.shape
    device = biases.device
    out_dtype = biases.dtype

    # Replicate input m+1 times: 0..m-1 perturbed, m baseline
    expanded_input = input_ids.expand(m + 1, -1)

    state = {"target": None, "biases": biases}

    def steering_hook(module, inputs, output):
        # Augment biases with a zero row for the baseline batch index.
        # Important: torch.cat preserves grad flow back to `state["biases"]`.
        h = output[0] if isinstance(output, tuple) else output
        zero_row = torch.zeros(
            1, hidden_dim, device=device, dtype=h.dtype,
        )
        full = torch.cat([state["biases"].to(h.dtype), zero_row], dim=0)
        return _add_to_output(output, full.unsqueeze(1))  # broadcast over seq

    def capture_hook(module, inputs, output):
        state["target"] = output[0] if isinstance(output, tuple) else output

    src_mod = _get_source_module(model, source_layer, capture_site)
    tgt_mod = model.model.layers[target_layer]
    h1 = src_mod.register_forward_hook(steering_hook)
    h2 = tgt_mod.register_forward_hook(capture_hook)

    try:
        model(expanded_input)
        target = state["target"]                          # [m+1, S, H]
        target_tok = target[:, target_token_slice, :]    # [m+1, k_tok, H]
        baseline = target_tok[m:m + 1]                   # [1, k_tok, H]
        delta = target_tok[:m] - baseline                 # [m, k_tok, H]
        return delta.to(out_dtype)
    finally:
        h1.remove()
        h2.remove()


# ============================================================================
# Calibration of R
# ============================================================================


def _calibrate_R(
    model, tokenizer, cfg: DCTConfig, input_ids: torch.Tensor,
    seed: int, verbose: bool = True,
) -> float:
    """Bisection on R such that sqrt(‖residual‖² / ‖linear‖²) = λ.

    Matches Mack's `SteeringCalibrator.calibrate` formula:

        ratio(R) = sqrt(mean over directions of [‖Δ(R v) − R·Jv‖² / ‖R·Jv‖²])

    linear(R)   = R · J · v_cal           (Jacobian-vector product, R-times-linear)
    residual(R) = Δ(R v_cal) − R · J · v_cal   (the higher-order tail)

    We compute J v_cal by finite difference: J v ≈ Δ(ε v) / ε. The ratio
    INCREASES monotonically in R (residual grows super-linearly while linear
    grows linearly), so bisection inverts: ratio > target → reduce R.

    Default λ = 0.5: residual is HALF the linear part — "mild nonlinearity,
    linear still dominates." This is Mack's intended training regime, where
    the nonlinear correction is large enough to identify non-orthogonal
    factors but small enough that the optimization is stable.
    """
    device = input_ids.device
    hidden_dim = model.config.hidden_size
    target_layer = cfg.target_layer or (model.config.num_hidden_layers - 8)

    # Random unit calibration directions (seeded)
    gen = torch.Generator(device=device).manual_seed(seed)
    v_cal = F.normalize(
        torch.randn(cfg.n_cal, hidden_dim, device=device,
                    dtype=model.dtype, generator=gen),
        dim=1,
    )

    # Compute exact J · v_cal via forward-mode autodiff (matches Mack's
    # `torch.func.jvp`). This avoids the bf16 noise floor that contaminates
    # finite-difference estimates at small ε. One forward pass with dual
    # numbers; tangent at batch position i = J_i · v_cal[i] (block-diagonal
    # Jacobian across the batch dim, since each delta_i depends only on
    # bias_i in `_compute_delta`).
    def _delta_at_bias(biases):
        return _compute_delta(
            model, cfg.source_layer, target_layer, cfg.capture_site,
            input_ids, biases, cfg.target_tokens,
        )
    zero_biases = torch.zeros_like(v_cal)
    _, Jv = functional_jvp(_delta_at_bias, (zero_biases,), (v_cal,))
    Jv = Jv.detach()       # [n_cal, k_tok, H]

    def ratio_at(R: float) -> float:
        """sqrt(mean over directions of ‖Δ - R·Jv‖² / ‖R·Jv‖²)."""
        with torch.no_grad():
            delta = _compute_delta(
                model, cfg.source_layer, target_layer, cfg.capture_site,
                input_ids, R * v_cal, cfg.target_tokens,
            )
            linear = R * Jv          # [n_cal, k_tok, H]
            residual = delta - linear  # [n_cal, k_tok, H]
            # Per-direction squared norms (reduce over tokens + hidden)
            linear_sq = linear.float().pow(2).sum(dim=(1, 2)).clamp(min=1e-12)
            residual_sq = residual.float().pow(2).sum(dim=(1, 2))
            return float((residual_sq / linear_sq).mean().sqrt().item())

    # Bisection — ratio is monotone INCREASING in R (residual grows faster
    # than linear). r > target → too nonlinear → reduce R.
    lo, hi = cfg.cal_R_lo, cfg.cal_R_hi
    target = cfg.lambda_cal
    for _ in range(cfg.cal_iters):
        mid = (lo + hi) / 2
        r = ratio_at(mid)
        if r > target:    # too nonlinear → reduce R
            hi = mid
        else:             # too linear → increase R
            lo = mid
    R_cal = (lo + hi) / 2

    if verbose:
        final_ratio = ratio_at(R_cal)
        print(f"  Calibrated R = {R_cal:.3f}  (target λ={target}, achieved={final_ratio:.3f})")
    return R_cal


# ============================================================================
# Main: find DCT vectors via OGI
# ============================================================================


def find_dct_vectors(
    model,
    tokenizer,
    prompt: str,
    config: DCTConfig | None = None,
    num_features: int = 12,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Find exponential-DCT input feature directions via OGI.

    Args:
        model: HuggingFace causal LM (will be temporarily flagged frozen
            for the duration of this call).
        tokenizer: matching tokenizer.
        prompt: raw user message; chat template will be applied.
        config: DCTConfig; if None, uses defaults with `num_features` overridden.
        num_features: m, the number of feature directions to learn.
        seed: RNG seed for V initialization and calibration directions.
        verbose: print per-iteration progress.

    Returns:
        (vectors, info)
          vectors: [num_features, hidden_dim] tensor, rows unit-normed.
                   These are the input (source-layer) feature directions —
                   drop-in compatible with PI/MELBO vectors at the same
                   (source_layer, capture_site).
          info:    dict with R_cal, final loss, etc. — useful for metadata.
    """
    cfg = config or DCTConfig(num_features=num_features)
    if config is None:
        cfg.num_features = num_features
    m = cfg.num_features

    device = next(model.parameters()).device
    dtype = model.dtype
    hidden_dim = model.config.hidden_size
    target_layer = cfg.target_layer or (model.config.num_hidden_layers - 8)

    # Save original requires_grad state, freeze for DCT, restore at end.
    # (PI/MELBO do similar manipulations — we want to be polite to the caller.)
    saved_req_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad = False

    # Format and tokenize prompt
    formatted = format_chat(tokenizer, prompt)
    input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)

    info: dict = {}
    try:
        # ── Calibrate R ────────────────────────────────────────────────────
        if verbose:
            print(f"  DCT calibration (n_cal={cfg.n_cal}, target λ={cfg.lambda_cal})...")
        cal_t0 = time.time()
        R = _calibrate_R(model, tokenizer, cfg, input_ids, seed=seed, verbose=verbose)
        info["R_cal"] = R
        info["cal_time"] = time.time() - cal_t0

        # ── Initialise V (random orthonormal rows) ────────────────────────
        gen = torch.Generator(device=device).manual_seed(seed + 1)
        V = torch.randn(m, hidden_dim, device=device, dtype=dtype, generator=gen)
        # QR on V^T (hidden × m) → orthonormal columns; rows of Q^T orthonormal
        Q, _ = torch.linalg.qr(V.t().float())
        V = Q.t().to(dtype)

        # Initialise U via U = mean_t Δ(R v_i) (the "natural" target direction
        # produced by the i-th source perturbation). One forward, no grad.
        with torch.no_grad():
            delta_init = _compute_delta(
                model, cfg.source_layer, target_layer, cfg.capture_site,
                input_ids, R * V, cfg.target_tokens,
            )
            U = F.normalize(delta_init.mean(dim=1), dim=1)

        # ── OGI loop ───────────────────────────────────────────────────────
        ogi_t0 = time.time()
        loss_history = []
        for it in range(cfg.num_iters):
            iter_start = time.time()

            # 1) QR-orthogonalise V at the start of each iteration
            Q, _ = torch.linalg.qr(V.t().float())
            V = Q.t().to(dtype)

            # 2) Forward + backward to get gradients of causal importance
            #    causal_imp = Σ_i <U_i, mean_tok Δ(R v_i)>
            V_param = V.detach().clone().requires_grad_(True)
            delta = _compute_delta(
                model, cfg.source_layer, target_layer, cfg.capture_site,
                input_ids, R * V_param, cfg.target_tokens,
            )
            delta_mean = delta.mean(dim=1)                     # [m, H]
            causal_imp = (U.detach() * delta_mean.float()).sum()
            grad_V = torch.autograd.grad(causal_imp, V_param)[0]   # [m, H]

            # 3) Infinite-step update: replace V, U with their gradients.
            #    ∂causal_imp/∂U_i = delta_mean_i (no model backward needed).
            V = grad_V.detach()
            U = delta_mean.detach()

            # 4) Column-normalise (rows of U, V are unit-norm in our convention)
            V = F.normalize(V, dim=1)
            U = F.normalize(U, dim=1)

            loss_val = float(causal_imp.item())
            loss_history.append(loss_val)

            if verbose:
                iter_dt = time.time() - iter_start
                elapsed = time.time() - ogi_t0
                remaining = iter_dt * (cfg.num_iters - it - 1)
                print(
                    f"  OGI iter {it}/{cfg.num_iters - 1}: loss={loss_val:+.3f}"
                    f"  [{format_time(iter_dt)}/iter, "
                    f"elapsed {format_time(elapsed)}, "
                    f"~{format_time(remaining)} left]"
                )

        # Final orthogonalisation + normalisation pass
        Q, _ = torch.linalg.qr(V.t().float())
        V = Q.t().to(dtype)
        V = F.normalize(V, dim=1)

        info["ogi_time"] = time.time() - ogi_t0
        info["loss_history"] = loss_history
        info["final_loss"] = loss_history[-1] if loss_history else None

        return V.detach(), info

    finally:
        # Restore caller's parameter grad flags
        for p, was in zip(model.parameters(), saved_req_grad):
            p.requires_grad = was
