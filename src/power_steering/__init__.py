"""
power_steering — Discover and evaluate steering vectors for language models.

Supports three methods:
  - PI-RR: Power iteration with Rayleigh-Ritz correction (fast, linear, unsupervised)
  - MELBO: Mechanistic Elicitation of Latent Behaviors via Optimization (nonlinear)
  - CAA:   Contrastive Activation Addition (supervised, signed)
"""

from power_steering.find_vectors import (
    find_pi_vectors, find_melbo_vectors, find_caa_vector,
)
from power_steering.eval import SteeringEvaluator
from power_steering.generate import SteeredGenerator
from power_steering.utils import (
    load_vectors, load_vector_metadata, load_dataset, format_chat, get_caa_layer,
)

__all__ = [
    "find_pi_vectors",
    "find_melbo_vectors",
    "find_caa_vector",
    "SteeringEvaluator",
    "SteeredGenerator",
    "load_vectors",
    "load_vector_metadata",
    "load_dataset",
    "format_chat",
    "get_caa_layer",
]
