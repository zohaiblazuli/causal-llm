"""
Training Utilities

Helper functions for:
- Configuration loading
- Memory optimization
- Metric computation
- Reproducibility
- Batch preparation
- Model utilities
"""

import gc
import os
import random
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ["model", "lora", "training", "loss", "data"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    return config


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_trainable_parameters(model: PreTrainedModel):
    """
    Print the number of trainable parameters in the model.

    Args:
        model: Model to analyze
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100 * trainable_params / all_param

    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_param:,} || "
          f"Trainable%: {trainable_percent:.4f}%")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3

    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free
    }


def print_memory_usage(prefix: str = ""):
    """
    Print current GPU memory usage.

    Args:
        prefix: Prefix string for output
    """
    memory = get_memory_usage()
    print(f"{prefix}Memory - Allocated: {memory['allocated']:.2f}GB, "
          f"Reserved: {memory['reserved']:.2f}GB, "
          f"Free: {memory['free']:.2f}GB")


def clear_memory():
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_model_memory(
    model_name: str,
    lora_r: int = 16,
    max_seq_length: int = 2048,
    batch_size: int = 1,
    load_in_4bit: bool = True
) -> Dict[str, float]:
    """
    Estimate memory requirements for training.

    Args:
        model_name: Model identifier
        lora_r: LoRA rank
        max_seq_length: Maximum sequence length
        batch_size: Batch size
        load_in_4bit: Whether using 4-bit quantization

    Returns:
        Dictionary with memory estimates in GB
    """
    # Base model sizes (approximate)
    model_sizes = {
        "llama-2-7b": 7_000_000_000,
        "llama-3.1-8b": 8_000_000_000,
        "llama-3-8b": 8_000_000_000,
    }

    # Find matching model size
    num_params = 7_000_000_000  # Default to 7B
    for key, size in model_sizes.items():
        if key in model_name.lower():
            num_params = size
            break

    # Memory calculations
    if load_in_4bit:
        # 4-bit quantization
        model_memory = num_params * 0.5 / 1024**3  # 0.5 bytes per param
    else:
        # FP16
        model_memory = num_params * 2 / 1024**3  # 2 bytes per param

    # LoRA parameters (very small)
    # Approximate: rank * 2 * num_layers * hidden_size
    lora_params = lora_r * 2 * 32 * 4096  # Rough estimate
    lora_memory = lora_params * 2 / 1024**3  # FP16

    # Optimizer state (AdamW has 2 states per param)
    # For 8-bit optimizer, 1 byte per state
    optimizer_memory = lora_params * 2 / 1024**3

    # Gradient memory
    gradient_memory = lora_memory

    # Activation memory (depends on batch size and sequence length)
    # Rough estimate: batch * seq * hidden * layers * bytes_per_elem
    activation_memory = batch_size * max_seq_length * 4096 * 32 * 2 / 1024**3

    # With gradient checkpointing, reduce activation memory
    activation_memory *= 0.3  # ~70% reduction

    # Total
    total_memory = (
        model_memory +
        lora_memory +
        optimizer_memory +
        gradient_memory +
        activation_memory
    )

    return {
        "model": model_memory,
        "lora": lora_memory,
        "optimizer": optimizer_memory,
        "gradients": gradient_memory,
        "activations": activation_memory,
        "total": total_memory
    }


def compute_causal_metrics(
    repr_benign: torch.Tensor,
    repr_benign_cf: torch.Tensor,
    repr_injection: torch.Tensor
) -> Dict[str, float]:
    """
    Compute causal stability and spurious separation metrics.

    Args:
        repr_benign: Benign representations [batch, hidden]
        repr_benign_cf: Benign counterfactual representations [batch, hidden]
        repr_injection: Injection representations [batch, hidden]

    Returns:
        Dictionary with causal metrics
    """
    # Normalize representations
    repr_benign = F.normalize(repr_benign, dim=-1)
    repr_benign_cf = F.normalize(repr_benign_cf, dim=-1)
    repr_injection = F.normalize(repr_injection, dim=-1)

    # Cosine similarity
    sim_benign_benign_cf = torch.sum(repr_benign * repr_benign_cf, dim=-1)
    sim_benign_injection = torch.sum(repr_benign * repr_injection, dim=-1)

    # Causal stability: how similar benign and benign_cf are (higher is better)
    causal_stability = sim_benign_benign_cf.mean().item()

    # Spurious separation: how different benign and injection are (higher is better)
    spurious_separation = (1 - sim_benign_injection).mean().item()

    # Causal discrimination: margin between stability and similarity to injection
    causal_discrimination = causal_stability - sim_benign_injection.mean().item()

    return {
        "causal_stability": causal_stability,
        "spurious_separation": spurious_separation,
        "causal_discrimination": causal_discrimination
    }


def prepare_model_for_training(model: PreTrainedModel):
    """
    Prepare model for training with memory optimizations.

    Args:
        model: Model to prepare
    """
    # Enable gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Set model to training mode
    model.train()


def save_checkpoint(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    output_dir: str,
    best_metric: Optional[float] = None
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        step: Current step
        loss: Current loss
        output_dir: Output directory
        best_metric: Best validation metric (optional)
    """
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model (LoRA weights)
    model.save_pretrained(checkpoint_dir)

    # Save optimizer state
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "best_metric": best_metric
    }, checkpoint_dir / "training_state.pt")

    print(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str
) -> Tuple[int, int, float]:
    """
    Load training checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_dir: Checkpoint directory

    Returns:
        Tuple of (epoch, step, loss)
    """
    # Load model weights
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_dir)

    # Load training state
    training_state = torch.load(Path(checkpoint_dir) / "training_state.pt")
    optimizer.load_state_dict(training_state["optimizer_state_dict"])

    epoch = training_state["epoch"]
    step = training_state["step"]
    loss = training_state["loss"]

    print(f"Loaded checkpoint from {checkpoint_dir}")
    print(f"Resuming from epoch {epoch}, step {step}, loss {loss:.4f}")

    return epoch, step, loss


def create_optimizer(
    model: PreTrainedModel,
    learning_rate: float,
    weight_decay: float,
    optimizer_type: str = "paged_adamw_8bit",
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8
) -> torch.optim.Optimizer:
    """
    Create optimizer with memory-efficient settings.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_type: Optimizer type
        betas: Adam betas
        eps: Adam epsilon

    Returns:
        Optimizer instance
    """
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == "paged_adamw_8bit":
        # Memory-efficient 8-bit AdamW from bitsandbytes
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
        except ImportError:
            print("Warning: bitsandbytes not available, falling back to standard AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
    elif optimizer_type == "adamw_8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    else:
        # Standard AdamW
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )

    return optimizer


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def count_parameters(model: PreTrainedModel) -> Dict[str, int]:
    """
    Count different types of parameters in the model.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_pct": 100 * trainable_params / total_params if total_params > 0 else 0
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing training utilities...")

    # Test config loading
    config_path = "C:\\isef\\training\\config.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Config loaded with {len(config)} sections")

    # Test seed setting
    set_seed(42)
    print("Seed set to 42")

    # Test memory estimation
    memory_est = estimate_model_memory(
        model_name="meta-llama/Llama-2-7b-hf",
        lora_r=16,
        max_seq_length=2048,
        batch_size=1,
        load_in_4bit=True
    )

    print("\nMemory estimation for Llama 2 7B with LoRA:")
    for key, value in memory_est.items():
        print(f"  {key}: {value:.2f} GB")

    print(f"\nTotal estimated memory: {memory_est['total']:.2f} GB")

    if memory_est['total'] < 6.0:
        print("Fits in RTX 4050 (6GB)!")
    else:
        print("Warning: May exceed RTX 4050 VRAM")

    # Test causal metrics computation
    batch_size = 4
    hidden_dim = 768

    repr_benign = torch.randn(batch_size, hidden_dim)
    repr_benign_cf = repr_benign + torch.randn(batch_size, hidden_dim) * 0.1  # Similar
    repr_injection = torch.randn(batch_size, hidden_dim)  # Different

    metrics = compute_causal_metrics(repr_benign, repr_benign_cf, repr_injection)

    print("\nCausal metrics test:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nAll utility tests passed!")
