"""
power_steering — Discover and evaluate steering vectors for language models.

Supports two methods:
  - PI-RR: Power iteration with Rayleigh-Ritz correction (fast, linear)
  - MELBO: Mechanistic Elicitation of Latent Behaviors via Optimization (nonlinear)
"""

from power_steering.find_vectors import find_pi_vectors, find_melbo_vectors
from power_steering.eval import SteeringEvaluator
from power_steering.generate import SteeredGenerator
from power_steering.utils import load_vectors, load_dataset, format_chat

__all__ = [
    "find_pi_vectors",
    "find_melbo_vectors",
    "SteeringEvaluator",
    "SteeredGenerator",
    "load_vectors",
    "load_dataset",
    "format_chat",
]
