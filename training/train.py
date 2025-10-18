"""
Main Training Script for Causal LLM Fine-tuning

Usage:
    # Basic training
    python training/train.py --config training/config.yaml

    # Resume from checkpoint
    python training/train.py --config training/config.yaml --resume checkpoints/checkpoint-500

    # Debug mode (small dataset)
    python training/train.py --config training/config.yaml --debug

    # Without W&B logging
    python training/train.py --config training/config.yaml --no-wandb
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.causal_model import CausalLLMModel
from models.losses import CausalContrastiveLoss
from training.dataset import CausalContrastiveDataset, CausalContrastiveCollator
from training.trainer import CausalTrainer
from training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    MemoryMonitor,
    CausalMetricsLogger,
    WandbLogger,
    ProgressLogger
)
from training.utils import (
    load_config,
    set_seed,
    print_trainable_parameters,
    estimate_model_memory,
    create_optimizer,
    print_memory_usage
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Causal LLM with LoRA")

    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (small dataset, fast iteration)"
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )

    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Profile memory usage during training"
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force fresh training, ignore existing checkpoints"
    )

    return parser.parse_args()


def setup_model(config: dict):
    """
    Setup model with LoRA and quantization.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = config["model"]
    training_cfg = config.get("training", {})
    advanced_cfg = config.get("advanced", {})
    lora_config_dict = config["lora"]

    print(f"\nLoading model: {model_config['name']}")

    # Setup quantization config
    quantization_config = None
    if model_config.get("load_in_4bit", True):
        # Map string dtype to torch dtype
        compute_dtype_str = model_config.get("bnb_4bit_compute_dtype", "bfloat16")
        compute_dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        compute_dtype = compute_dtype_map.get(compute_dtype_str, torch.bfloat16)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4")
        )

    # Optional: read token from environment to bypass CLI login issues
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'],
        token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with quantization
    from transformers import AutoModelForCausalLM

    # Optional attention implementation (e.g., 'sdpa' or 'flash_attention_2')
    attn_impl = advanced_cfg.get("attn_implementation", None)

    base_model_kwargs = dict(
        quantization_config=quantization_config,
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=model_config.get("trust_remote_code", True),
        token=hf_token,
    )
    if attn_impl:
        base_model_kwargs["attn_implementation"] = attn_impl

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        **base_model_kwargs
    )

    print(f"Base model loaded")
    print_memory_usage("After loading base model: ")

    # Setup LoRA
    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["dropout"],
        bias=lora_config_dict.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=lora_config_dict.get("modules_to_save", None)
    )

    # Apply LoRA
    model = get_peft_model(base_model, lora_config)

    # Add causal projection head
    hidden_size = base_model.config.hidden_size
    device = next(model.parameters()).device
    model.causal_projection = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.LayerNorm(hidden_size),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_size, hidden_size)
    ).to(device)
    print(f"Causal projection moved to device: {device}")

    print(f"LoRA applied with rank {lora_config_dict['r']}")
    print_trainable_parameters(model)
    print_memory_usage("After applying LoRA: ")

    # Enable gradient checkpointing (controlled from training section)
    if training_cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    else:
        print("Gradient checkpointing disabled")

    return model, tokenizer


def setup_dataloaders(config: dict, tokenizer, debug: bool = False):
    """
    Setup train and validation dataloaders.

    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer
        debug: Debug mode flag

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    data_config = config["data"]
    training_config = config["training"]

    # Override for debug mode
    max_samples = -1
    if debug:
        max_samples = 100
        print("Debug mode: Using only 100 samples")

    # Create datasets
    print("\nLoading datasets...")

    train_dataset = CausalContrastiveDataset(
        data_path=data_config["train_path"],
        tokenizer=tokenizer,
        max_length=data_config["max_length"],
        padding=data_config.get("padding", "max_length"),
        truncation=data_config.get("truncation", True),
        cache_dir=data_config.get("cache_dir", None),
        max_samples=max_samples
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset contains 0 valid samples after schema normalization. "
            "Check your data files and schema. Expected fields include system_instruction, "
            "user_input_benign_1/user_input_benign_2 (or benign_input/benign_cf_input), "
            "user_input_injection (or injection_input), and expected_output_1 (or benign_output)."
        )

    val_dataset = None
    if os.path.exists(data_config["val_path"]):
        val_dataset = CausalContrastiveDataset(
            data_path=data_config["val_path"],
            tokenizer=tokenizer,
            max_length=data_config["max_length"],
            padding=data_config.get("padding", "max_length"),
            truncation=data_config.get("truncation", True),
            cache_dir=data_config.get("cache_dir", None),
            max_samples=max_samples // 5 if debug else -1
        )
        if len(val_dataset) == 0:
            print("Warning: validation dataset has 0 valid samples. Continuing without validation.")
            val_dataset = None

    # Create collator (CRITICAL FIX: use same padding as config for consistency)
    collator = CausalContrastiveCollator(
        tokenizer=tokenizer,
        padding=data_config.get("padding", "max_length")  # Use config padding
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=training_config.get("dataloader_num_workers", 2),
        pin_memory=training_config.get("dataloader_pin_memory", True)
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=training_config["per_device_eval_batch_size"],
            shuffle=False,
            collate_fn=collator,
            num_workers=training_config.get("dataloader_num_workers", 2),
            pin_memory=training_config.get("dataloader_pin_memory", True)
        )

    print(f"Train dataset: {len(train_dataset)} samples")
    if val_dataset is not None:
        print(f"Val dataset: {len(val_dataset)} samples")

    return train_dataloader, val_dataloader


def setup_training_components(model, config: dict):
    """
    Setup optimizer, loss function, and callbacks.

    Args:
        model: Model to train
        config: Configuration dictionary

    Returns:
        Tuple of (optimizer, loss_fn, callbacks)
    """
    training_config = config["training"]
    loss_config = config["loss"]

    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        optimizer_type=training_config.get("optim", "paged_adamw_8bit"),
        betas=(training_config.get("adam_beta1", 0.9), training_config.get("adam_beta2", 0.999)),
        eps=training_config.get("adam_epsilon", 1e-8)
    )

    print(f"\nOptimizer: {type(optimizer).__name__}")

    # Create loss function
    if loss_config["type"] == "causal_contrastive":
        loss_fn = CausalContrastiveLoss(
            temperature=loss_config["temperature"],
            lambda_task=loss_config["lambda_task"],
            lambda_causal=loss_config["lambda_causal"],
            lambda_spurious=loss_config["lambda_spurious"],
            similarity_metric=loss_config.get("similarity_metric", "cosine")
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")

    print(f"Loss function: {type(loss_fn).__name__}")

    return optimizer, loss_fn


def setup_callbacks(config: dict, no_wandb: bool = False):
    """
    Setup training callbacks.

    Args:
        config: Configuration dictionary
        no_wandb: Disable W&B logging

    Returns:
        List of callbacks
    """
    callbacks = []

    training_config = config["training"]
    checkpoint_config = config.get("checkpointing", {})
    wandb_config = config.get("wandb", {})

    # Progress logger
    callbacks.append(ProgressLogger(
        log_every_n_steps=training_config.get("logging_steps", 10)
    ))

    # Learning rate monitor
    callbacks.append(LearningRateMonitor())

    # Memory monitor
    callbacks.append(MemoryMonitor(
        log_every_n_steps=50,
        clear_cache_every_n_steps=training_config.get("eval_steps", 200)
    ))

    # Causal metrics logger
    callbacks.append(CausalMetricsLogger())

    # Model checkpoint
    callbacks.append(ModelCheckpoint(
        output_dir=checkpoint_config.get("output_dir", "checkpoints"),
        metric=training_config.get("metric_for_best_model", "causal_stability"),
        mode="max" if training_config.get("greater_is_better", True) else "min",
        save_total_limit=training_config.get("save_total_limit", 3),
        save_every_n_steps=training_config.get("save_steps", 200)
    ))

    # Early stopping
    if training_config.get("early_stopping_patience", 0) > 0:
        callbacks.append(EarlyStopping(
            patience=training_config["early_stopping_patience"],
            min_delta=training_config.get("early_stopping_threshold", 0.001),
            metric=training_config.get("metric_for_best_model", "causal_stability"),
            mode="max" if training_config.get("greater_is_better", True) else "min"
        ))

    # Weights & Biases logger
    if not no_wandb and wandb_config.get("enabled", True):
        callbacks.append(WandbLogger(
            project=wandb_config["project"],
            entity=wandb_config.get("entity", None),
            name=wandb_config.get("name", None),
            config=config,
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", None)
        ))

    print(f"\nCallbacks: {[type(cb).__name__ for cb in callbacks]}")

    return callbacks


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    print("="*80)
    print("CAUSAL LLM TRAINING - RTX 4090 Optimized")
    print("="*80)

    config = load_config(args.config)
    print(f"\nConfiguration loaded from {args.config}")

    # Override output dir if specified
    if args.output_dir:
        config["checkpointing"]["output_dir"] = args.output_dir

    # Set random seed for reproducibility
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    print(f"Random seed set to {seed}")

    # Estimate memory requirements
    print("\n" + "-"*80)
    print("MEMORY ESTIMATION")
    print("-"*80)

    memory_est = estimate_model_memory(
        model_name=config["model"]["name"],
        lora_r=config["lora"]["r"],
        max_seq_length=config["data"]["max_length"],
        batch_size=config["training"]["per_device_train_batch_size"],
        load_in_4bit=config["model"].get("load_in_4bit", True)
    )

    print("\nEstimated memory usage:")
    for key, value in memory_est.items():
        print(f"  {key}: {value:.2f} GB")

    total_memory = memory_est["total"]
    vram_limit = config["hardware"]["vram_gb"]

    if total_memory > vram_limit:
        print(f"\nWARNING: Estimated memory ({total_memory:.2f} GB) exceeds VRAM ({vram_limit} GB)")
        print("Consider reducing max_seq_length or enabling more aggressive optimizations")
    else:
        print(f"\nMemory estimate looks good! ({total_memory:.2f} GB / {vram_limit} GB)")

    # Setup model
    print("\n" + "-"*80)
    print("MODEL SETUP")
    print("-"*80)

    model, tokenizer = setup_model(config)

    # Setup dataloaders
    print("\n" + "-"*80)
    print("DATA LOADING")
    print("-"*80)

    train_dataloader, val_dataloader = setup_dataloaders(
        config,
        tokenizer,
        debug=args.debug
    )

    # Setup training components
    print("\n" + "-"*80)
    print("TRAINING SETUP")
    print("-"*80)

    optimizer, loss_fn = setup_training_components(model, config)

    # Setup callbacks
    callbacks = setup_callbacks(config, no_wandb=args.no_wandb)

    # Create trainer
    trainer = CausalTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        callbacks=callbacks,
        output_dir=config["checkpointing"]["output_dir"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Auto-resume logic (default: ON)
    if args.resume:
        # Explicit checkpoint specified
        print(f"\nResuming from explicit checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    elif not args.no_resume:
        # Auto-detect latest checkpoint (DEFAULT BEHAVIOR)
        checkpoint_dir = Path(config["checkpointing"]["output_dir"])

        if checkpoint_dir.exists():
            # Find all valid checkpoints
            potential_checkpoints = []

            # Check for checkpoint-* directories
            for cp in checkpoint_dir.glob("checkpoint-*"):
                if (cp / "trainer_state.pt").exists():
                    potential_checkpoints.append(cp)

            # Check for best_model
            best_model_path = checkpoint_dir / "best_model"
            if best_model_path.exists() and (best_model_path / "trainer_state.pt").exists():
                potential_checkpoints.append(best_model_path)

            if potential_checkpoints:
                # Sort by modification time (most recent first)
                latest_checkpoint = max(potential_checkpoints, key=lambda p: (p / "trainer_state.pt").stat().st_mtime)

                # Check if training was already completed
                trainer_state = torch.load(latest_checkpoint / "trainer_state.pt", map_location="cpu")
                completed_epoch = trainer_state.get("epoch", 0)
                completed_step = trainer_state.get("global_step", 0)
                num_epochs = config["training"]["num_epochs"]

                # Check if this is the last epoch AND we're past the last step
                total_steps = len(train_dataloader) // config["training"]["gradient_accumulation_steps"] * num_epochs

                if completed_epoch >= num_epochs - 1 and completed_step >= total_steps - 10:
                    print(f"\n{'='*80}")
                    print("TRAINING ALREADY COMPLETED")
                    print(f"{'='*80}")
                    print(f"Completed: {completed_epoch + 1}/{num_epochs} epochs, {completed_step} steps")
                    print("\nTo start fresh training, use: --no-resume")
                    print(f"{'='*80}\n")
                    return  # Exit without training
                else:
                    print(f"\n{'='*80}")
                    print("AUTO-RESUME DETECTED")
                    print(f"{'='*80}")
                    print(f"Latest checkpoint: {latest_checkpoint.name}")
                    print(f"Progress: Epoch {completed_epoch + 1}/{num_epochs}, Step {completed_step}")
                    print(f"Remaining: {num_epochs - completed_epoch - 1} epochs + partial epoch")
                    print(f"{'='*80}\n")
                    trainer.load_checkpoint(str(latest_checkpoint))
            else:
                print("\nNo valid checkpoints found, starting fresh training")
    else:
        print("\n--no-resume flag detected, starting fresh training")

    # Start training
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    print_memory_usage("Before training: ")

    try:
        trainer.train()
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

    print_memory_usage("Final ")


if __name__ == "__main__":
    main()
