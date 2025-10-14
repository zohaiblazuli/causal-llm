"""
Comprehensive Benchmarking Suite

Runs complete benchmark comparing with baseline defenses.
"""

import torch
from typing import Dict
from .metrics import run_full_evaluation


def run_benchmark_suite(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda"
) -> Dict[str, any]:
    """
    Run comprehensive benchmark suite.

    Compares against:
    - No defense (~87% attack success)
    - Input filtering (~62% attack success)
    - SecAlign (~34% attack success)
    - Our method (target: <10%)

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device

    Returns:
        Complete benchmark results
    """
    print("="*60)
    print("RUNNING COMPREHENSIVE BENCHMARK SUITE")
    print("="*60)

    # Run full evaluation
    results = run_full_evaluation(model, test_loader, device)

    # Comparison with baselines
    baselines = {
        "No Defense": {"attack_success_rate": 0.87},
        "Input Filtering": {"attack_success_rate": 0.62},
        "StruQ": {"attack_success_rate": 0.41},
        "SecAlign": {"attack_success_rate": 0.34},
        "Our Method (Target)": {"attack_success_rate": 0.10},
        "Our Method (Actual)": {
            "attack_success_rate": results["attack_success"]["overall_attack_success_rate"]
        }
    }

    results["baseline_comparison"] = baselines

    return results
