"""
Training Module for Causal LLM Fine-tuning

Provides complete training pipeline for fine-tuning large language models
with causal contrastive loss on memory-constrained hardware.

Components:
- dataset: Custom dataset and data collator for counterfactual triplets
- trainer: Custom trainer with memory optimizations
- callbacks: Training callbacks (early stopping, checkpointing, logging)
- utils: Utility functions for configuration, memory management, metrics
- optimize_memory: Memory profiling and optimization tools

Usage:
    from training import CausalTrainer
    from training.dataset import CausalContrastiveDataset
    from training.utils import load_config

    config = load_config("training/config.yaml")
    # ... setup model, dataset, etc.
    trainer = CausalTrainer(model, dataset, config=config)
    trainer.train()
"""

from training.dataset import CausalContrastiveDataset, CausalContrastiveCollator
from training.trainer import CausalTrainer
from training.utils import (
    load_config,
    set_seed,
    print_trainable_parameters,
    get_memory_usage,
    estimate_model_memory,
    compute_causal_metrics
)

__version__ = "1.0.0"

__all__ = [
    "CausalContrastiveDataset",
    "CausalContrastiveCollator",
    "CausalTrainer",
    "load_config",
    "set_seed",
    "print_trainable_parameters",
    "get_memory_usage",
    "estimate_model_memory",
    "compute_causal_metrics",
]
