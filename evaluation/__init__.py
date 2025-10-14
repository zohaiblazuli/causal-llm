"""
Evaluation Module for Causal LLM Security

This module implements comprehensive evaluation metrics and benchmarks
for testing prompt injection robustness.
"""

from .metrics import compute_attack_success_rate, compute_causal_stability
from .attacks import generate_attacks, test_robustness
from .benchmark import run_benchmark_suite

__all__ = [
    'compute_attack_success_rate',
    'compute_causal_stability',
    'generate_attacks',
    'test_robustness',
    'run_benchmark_suite'
]
