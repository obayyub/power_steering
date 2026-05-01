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


# ── Chat formatting ────────────────────────────────────────────────────────

def format_chat(
    tokenizer,
    user_message: str,
    system: str = "You are a helpful assistant.",
    enable_thinking: bool = False,
) -> str:
    """Format a user message using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


# ── Vector I/O ──────────────────────────────────────────────────────────────

def load_vectors(filepath: str | Path) -> torch.Tensor:
    """Load steering vectors from a .pt file.

    Handles three formats:
      - dict with "vectors" key  (PI-RR and MELBO default)
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

def load_dataset(path: str | Path) -> dict[str, list[dict]]:
    """Load the corrigibility evaluation dataset."""
    with open(path) as f:
        return json.load(f)


def sample_balanced(
    dataset: list[dict],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Sample n questions with balanced corrigible-letter labels.

    Takes n/2 from questions where A=corrigible and n/2 from B=corrigible
    to avoid letter preference bias. Falls back to the smaller class if
    either has fewer than n/2 items.
    """
    rng = random.Random(seed)

    a_corrigible = [q for q in dataset if q.get("corrigible_letter") == "A"]
    b_corrigible = [q for q in dataset if q.get("corrigible_letter") == "B"]

    n_each = n // 2
    n_each = min(n_each, len(a_corrigible), len(b_corrigible))

    sampled = rng.sample(a_corrigible, n_each) + rng.sample(b_corrigible, n_each)
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
