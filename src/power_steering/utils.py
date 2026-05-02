"""Shared utilities for data loading, vector I/O, and model helpers."""

import json
import random
import time
from pathlib import Path
from datetime import datetime

import torch


# ── Layer configs per model family ──────────────────────────────────────────

QWEN3_CONFIGS = {
    "0.6B": (4, 20),
    "1.7B": (6, 22),
    "4B": (8, 28),
    "8B": (8, 28),
    "14B": (10, 32),
    "32B": (12, 52),
}


def get_layer_config(model_name: str) -> tuple[int, int | None]:
    """Return (source_layer, target_layer) for a known model, else defaults."""
    # Sort by key length descending so "14B" matches before "4B"
    for size, config in sorted(QWEN3_CONFIGS.items(), key=lambda x: len(x[0]), reverse=True):
        if size in model_name:
            return config
    return (7, None)


def get_caa_layer(model, fraction: float = 0.6) -> int:
    """Default CAA injection layer = round(fraction * num_layers).

    60% places it in the mid-late block, where contrastive activations
    typically carry the most behaviorally relevant signal.
    """
    num_layers = len(model.model.layers)
    return int(round(fraction * num_layers))


# ── Steering site resolution ───────────────────────────────────────────────

CAPTURE_SITES = ("down_proj", "layer_output")


def get_steering_module(model, layer: int, capture_site: str):
    """Return the module to hook for capture/injection at a given site.

    "down_proj"   — the MLP's down-projection inside layer L. Output is a
                    tensor [B, S, H]. Used by PI and MELBO so the perturbation
                    site matches the Jacobian source/target convention.
    "layer_output" — the whole transformer block at layer L. Output is the
                    residual stream after the block (may be a tensor or a
                    tuple whose first element is the hidden state). Used by
                    CAA so the contrast captures the cumulative state, not
                    just one layer's MLP contribution.
    """
    block = model.model.layers[layer]
    if capture_site == "down_proj":
        return block.mlp.down_proj
    if capture_site == "layer_output":
        return block
    raise ValueError(f"Unknown capture_site: {capture_site!r}; expected one of {CAPTURE_SITES}")


def add_steering_to_output(o, vec: torch.Tensor):
    """Add `vec` to a hook's output, handling both tensor and tuple outputs.

    For "down_proj" hooks `o` is a tensor; we add directly. For
    "layer_output" hooks `o` may be a tuple `(hidden_states, *rest)`; we add
    to the first element and rebuild the tuple.
    """
    if isinstance(o, tuple):
        h = o[0]
        return (h + vec.to(h.device, dtype=h.dtype),) + tuple(o[1:])
    return o + vec.to(o.device, dtype=o.dtype)


def take_hidden_state(o):
    """Return the hidden-state tensor from a hook output (tensor or tuple)."""
    return o[0] if isinstance(o, tuple) else o


# ── Chat formatting ────────────────────────────────────────────────────────

def format_chat(
    tokenizer,
    user_message: str,
    system: str = "You are a helpful assistant.",
    enable_thinking: bool = False,
    assistant_start: str = "",
) -> str:
    """Format a user message using the tokenizer's chat template.

    If `assistant_start` is provided, it is appended verbatim after the
    assistant generation prompt — used by CAA to put the model in the
    state where the next token is the answer letter.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if assistant_start:
        formatted += assistant_start
    return formatted


# ── Vector I/O ──────────────────────────────────────────────────────────────

def load_vectors(filepath: str | Path) -> torch.Tensor:
    """Load steering vectors from a .pt file.

    Handles three formats:
      - dict with "vectors" key  (PI-RR, MELBO, CAA default)
      - dict with "steering" key (legacy)
      - raw tensor
    """
    data = torch.load(filepath, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        if "vectors" in data:
            return data["vectors"]
        if "steering" in data:
            return data["steering"]
    return data


def load_vector_metadata(filepath: str | Path) -> dict:
    """Load the non-tensor metadata from a vector .pt file.

    Returns an empty dict if the file isn't a metadata-bearing dict (e.g.
    raw tensor format).
    """
    data = torch.load(filepath, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k not in ("vectors", "steering")}


def save_vectors(
    vectors: torch.Tensor,
    output_dir: str | Path,
    *,
    method: str,
    model_name: str,
    metadata: dict | None = None,
) -> Path:
    """Save vectors with standard naming and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{method}_{model_short}_{timestamp}.pt"

    save_data = {"vectors": vectors.cpu(), "model": model_name, "method": method}
    if metadata:
        save_data.update(metadata)
    torch.save(save_data, path)
    return path


# ── Dataset loading and sampling ────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_dataset(path: str | Path) -> dict[str, list[dict]]:
    """Load the corrigibility evaluation dataset."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = _PROJECT_ROOT / path
    with open(path) as f:
        return json.load(f)


def sample_balanced(
    dataset: list[dict],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Sample n questions with balanced matching-letter labels.

    Takes n/2 from questions where A=matching and n/2 from B=matching
    to avoid letter preference bias. Falls back to the smaller class if
    either has fewer than n/2 items.
    """
    rng = random.Random(seed)

    a_match = [q for q in dataset if q.get("matching_letter") == "A"]
    b_match = [q for q in dataset if q.get("matching_letter") == "B"]

    n_each = n // 2
    n_each = min(n_each, len(a_match), len(b_match))

    sampled = rng.sample(a_match, n_each) + rng.sample(b_match, n_each)
    rng.shuffle(sampled)
    return sampled


# ── Misc ────────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def orthogonalize(V: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt orthogonalization of column vectors in V.

    Columns with near-zero norm after projection (< 1e-10) are dropped.
    In practice, floating-point noise may prevent exact-duplicate columns
    from being detected.
    """
    Q = []
    for v in V.T:
        for q in Q:
            v = v - torch.dot(v, q) * q
        norm = v.norm()
        if norm > 1e-10:
            Q.append(v / norm)
    return torch.stack(Q, dim=1) if Q else V
