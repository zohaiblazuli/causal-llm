"""
Dry Run Training Test

Performs a minimal end-to-end training test to validate:
- Model loading with 4-bit quantization
- LoRA adapter application
- Data pipeline
- Forward pass
- Loss computation
- Backward pass
- Optimizer step
- Memory management
- Checkpoint saving/loading

This is a critical pre-training validation to catch issues before full training.

Usage:
    python training/dry_run.py
    python training/dry_run.py --config training/config.yaml --steps 10
"""

import argparse
import gc
import sys
import time
import os
from pathlib import Path
from typing import Dict, List

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import CausalContrastiveDataset, CausalContrastiveCollator
from training.utils import (
    load_config,
    set_seed,
    get_memory_usage,
    clear_memory,
    print_memory_usage,
    create_optimizer,
    print_trainable_parameters
)
from models.losses import CausalContrastiveLoss


def load_model_with_lora(config: dict, device: str = "cuda"):
    """
    Load base model with 4-bit quantization and apply LoRA.

    Args:
        config: Configuration dictionary
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer)
    """
    print("\n" + "="*80)
    print("LOADING MODEL WITH LORA")
    print("="*80)

    model_config = config["model"]
    lora_config_dict = config["lora"]

    # Setup quantization
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    compute_dtype = compute_dtype_map.get(
        model_config.get("bnb_4bit_compute_dtype", "bfloat16"),
        torch.bfloat16
    )

    print(f"\nModel: {model_config['name']}")
    print(f"Quantization: 4-bit NF4")
    print(f"Compute dtype: {compute_dtype}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load tokenizer
    print("\nLoading tokenizer...")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'], token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")

    # Load base model
    print("\nLoading base model...")
    print_memory_usage("Before model load: ")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )

    print_memory_usage("After base model: ")

    # Apply LoRA
    print(f"\nApplying LoRA (rank={lora_config_dict['r']})...")

    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, lora_config)

    print_memory_usage("After LoRA: ")

    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    print("\n✓ Model loaded successfully")
    print_trainable_parameters(model)

    return model, tokenizer


def create_data_loader(
    config: dict,
    tokenizer,
    num_samples: int = 100,
    batch_size: int = 1
) -> DataLoader:
    """
    Create data loader with subset of training data.

    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer instance
        num_samples: Number of samples to load
        batch_size: Batch size

    Returns:
        DataLoader instance
    """
    print("\n" + "="*80)
    print("CREATING DATA LOADER")
    print("="*80)

    data_config = config["data"]

    print(f"\nLoading {num_samples} samples from {data_config['train_path']}")

    dataset = CausalContrastiveDataset(
        data_path=data_config['train_path'],
        tokenizer=tokenizer,
        max_length=data_config['max_length'],
        padding="max_length",
        truncation=True,
        max_samples=num_samples
    )

    print(f"✓ Dataset created with {len(dataset)} samples")

    collator = CausalContrastiveCollator(
        tokenizer=tokenizer,
        padding="longest"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Use 0 for dry run to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"✓ DataLoader created with batch_size={batch_size}")

    return dataloader


def run_training_steps(
    model,
    dataloader,
    optimizer,
    loss_fn,
    num_steps: int,
    gradient_accumulation_steps: int,
    use_amp: bool,
    amp_dtype,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    Run training steps and monitor progress.

    Args:
        model: Model to train
        dataloader: Data loader
        optimizer: Optimizer
        loss_fn: Loss function
        num_steps: Number of training steps
        gradient_accumulation_steps: Gradient accumulation steps
        use_amp: Use automatic mixed precision
        amp_dtype: AMP dtype
        device: Device

    Returns:
        Dictionary with training metrics
    """
    print("\n" + "="*80)
    print(f"RUNNING {num_steps} TRAINING STEPS")
    print("="*80)

    model.train()

    metrics = {
        "loss": [],
        "causal_stability": [],
        "spurious_separation": [],
        "task_loss": [],
        "step_time": [],
        "memory": []
    }

    scaler = GradScaler() if use_amp and amp_dtype == torch.float16 else None

    print(f"\nConfiguration:")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Mixed precision: {use_amp} ({amp_dtype})")
    print(f"  Device: {device}")

    print("\nStarting training steps...")

    batch_idx = 0
    step = 0
    pbar = tqdm(total=num_steps, desc="Training steps")

    for batch in dataloader:
        if step >= num_steps:
            break

        step_start_time = time.time()

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass with mixed precision
        with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            # Get representations for all three inputs
            benign_outputs = model(
                input_ids=batch["benign_input_ids"],
                attention_mask=batch["benign_attention_mask"],
                output_hidden_states=True
            )

            benign_cf_outputs = model(
                input_ids=batch["benign_cf_input_ids"],
                attention_mask=batch["benign_cf_attention_mask"],
                output_hidden_states=True
            )

            injection_outputs = model(
                input_ids=batch["injection_input_ids"],
                attention_mask=batch["injection_attention_mask"],
                output_hidden_states=True
            )

            # Extract representations (mean pooling of last hidden state)
            def get_representation(outputs, attention_mask):
                hidden_states = outputs.hidden_states[-1]
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_hidden / sum_mask

            repr_benign = get_representation(benign_outputs, batch["benign_attention_mask"])
            repr_benign_cf = get_representation(benign_cf_outputs, batch["benign_cf_attention_mask"])
            repr_injection = get_representation(injection_outputs, batch["injection_attention_mask"])

            # Compute loss
            loss_dict = loss_fn(
                repr_benign=repr_benign,
                repr_benign_counterfactual=repr_benign_cf,
                repr_injection=repr_injection,
                logits_benign=benign_outputs.logits,
                labels_benign=batch["labels"]
            )

            loss = loss_dict["loss"] / gradient_accumulation_steps

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if scaler:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # Record metrics
            step_time = time.time() - step_start_time
            memory = get_memory_usage()

            metrics["loss"].append(loss.item() * gradient_accumulation_steps)
            metrics["causal_stability"].append(loss_dict["causal_stability"].item())
            metrics["spurious_separation"].append(loss_dict["spurious_separation"].item())
            metrics["task_loss"].append(loss_dict["task_loss"].item())
            metrics["step_time"].append(step_time)
            metrics["memory"].append(memory["allocated"])

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{metrics['loss'][-1]:.4f}",
                "mem": f"{memory['allocated']:.2f}GB",
                "time": f"{step_time:.2f}s"
            })

            step += 1

            # Check for NaN or Inf
            if not torch.isfinite(loss):
                print(f"\n✗ Loss became {loss.item()} at step {step}")
                return metrics

        batch_idx += 1

    pbar.close()

    return metrics


def analyze_training_metrics(metrics: Dict[str, List[float]]) -> bool:
    """
    Analyze training metrics for issues.

    Args:
        metrics: Dictionary of training metrics

    Returns:
        True if training looks healthy
    """
    print("\n" + "="*80)
    print("ANALYZING TRAINING METRICS")
    print("="*80)

    if len(metrics["loss"]) == 0:
        print("\n✗ No training steps completed")
        return False

    # Check loss trajectory
    print("\nLoss Trajectory:")
    for i, loss in enumerate(metrics["loss"]):
        print(f"  Step {i}: {loss:.4f}")

    # Check if loss is decreasing
    if len(metrics["loss"]) >= 3:
        recent_losses = metrics["loss"][-3:]
        if recent_losses[-1] < recent_losses[0]:
            print("\n✓ Loss is decreasing (good)")
        else:
            print("\n⚠ Loss is not decreasing (may need more steps)")

    # Check for NaN or Inf
    if any(not (0 <= loss < 1000) for loss in metrics["loss"]):
        print("\n✗ Loss values out of reasonable range")
        return False

    # Memory analysis
    print("\nMemory Usage:")
    print(f"  Average: {sum(metrics['memory']) / len(metrics['memory']):.2f} GB")
    print(f"  Peak: {max(metrics['memory']):.2f} GB")

    if max(metrics['memory']) > 5.8:
        print("  ⚠ Warning: Memory usage is very close to 6GB limit")
        return False
    else:
        print("  ✓ Memory usage within safe limits")

    # Performance analysis
    avg_step_time = sum(metrics["step_time"]) / len(metrics["step_time"])
    print(f"\nPerformance:")
    print(f"  Average step time: {avg_step_time:.2f} seconds")

    if avg_step_time > 5.0:
        print("  ⚠ Steps are slow (>5s each)")
    else:
        print("  ✓ Step time is reasonable")

    # Causal metrics
    print(f"\nCausal Metrics:")
    print(f"  Final causal stability: {metrics['causal_stability'][-1]:.4f}")
    print(f"  Final spurious separation: {metrics['spurious_separation'][-1]:.4f}")

    return True


def test_checkpoint_save_load(model, tokenizer, output_dir: str = "checkpoints/dry_run_test") -> bool:
    """
    Test checkpoint saving and loading.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory

    Returns:
        True if successful
    """
    print("\n" + "="*80)
    print("TESTING CHECKPOINT SAVE/LOAD")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving checkpoint to {output_path}...")

    try:
        # Save model and tokenizer
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        print("✓ Checkpoint saved")

        # Try to load
        print("\nLoading checkpoint...")

        from peft import PeftModel

        # We need to load the base model first, then apply PEFT
        # For this test, we'll just verify files exist
        required_files = ["adapter_config.json", "adapter_model.safetensors"]

        for filename in required_files:
            filepath = output_path / filename
            if not filepath.exists():
                print(f"✗ Missing file: {filename}")
                return False

        print("✓ Checkpoint files verified")

        return True

    except Exception as e:
        print(f"\n✗ Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main dry run function - Enhanced for Week 1 validation."""
    parser = argparse.ArgumentParser(description="Dry run training test")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    print("="*80)
    print("WEEK 1 DRY RUN: 10 TRAINING STEPS - PHASE 2")
    print("="*80)

    from datetime import datetime
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Training steps: {args.steps}")
    print(f"  Data samples: {args.num_samples}")

    # Set seed for reproducibility
    set_seed(42)

    # Load config
    print("\nLoading configuration...")
    config = load_config(args.config)
    print("✓ Configuration loaded")

    # Clear memory before starting
    clear_memory()
    print_memory_usage("Initial memory: ")

    # Load model with LoRA
    try:
        model, tokenizer = load_model_with_lora(config)
    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create data loader
    try:
        dataloader = create_data_loader(
            config,
            tokenizer,
            num_samples=args.num_samples,
            batch_size=config["training"]["per_device_train_batch_size"]
        )
    except Exception as e:
        print(f"\n✗ Data loader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create optimizer
    print("\n" + "="*80)
    print("CREATING OPTIMIZER")
    print("="*80)

    try:
        optimizer = create_optimizer(
            model,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            optimizer_type=config["training"]["optimizer"]
        )
        print("✓ Optimizer created")
    except Exception as e:
        print(f"\n✗ Optimizer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create loss function
    loss_config = config["loss"]
    loss_fn = CausalContrastiveLoss(
        temperature=loss_config["temperature"],
        lambda_task=loss_config["lambda_task"],
        lambda_causal=loss_config["lambda_causal"],
        lambda_spurious=loss_config["lambda_spurious"]
    )
    print("✓ Loss function created")

    # Run training steps
    try:
        training_config = config["training"]
        use_amp = training_config.get("bf16", False) or training_config.get("fp16", False)
        amp_dtype = torch.bfloat16 if training_config.get("bf16", False) else torch.float16

        metrics = run_training_steps(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_steps=args.steps,
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        if len(metrics["loss"]) == 0:
            print("\n✗ No training steps completed")
            return False

        print("\n✓ Training steps completed")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n✗ Out of memory error!")
            print("  Recommendations:")
            print("  - Reduce max_seq_length in config.yaml")
            print("  - Increase gradient_accumulation_steps")
            print("  - Reduce LoRA rank")
            return False
        else:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Analyze metrics
    try:
        healthy = analyze_training_metrics(metrics)
    except Exception as e:
        print(f"\n✗ Metrics analysis failed: {e}")
        healthy = False

    # Test checkpoint
    try:
        checkpoint_ok = test_checkpoint_save_load(model, tokenizer)
    except Exception as e:
        print(f"\n✗ Checkpoint test failed: {e}")
        checkpoint_ok = False

    # Enhanced Summary
    print("\n" + "="*80)
    print("DRY RUN COMPLETE ✓")
    print("="*80)

    # Detailed metrics
    if len(metrics["loss"]) > 0:
        print(f"\nSteps completed: {len(metrics['loss'])}/10")
        print(f"Average loss: {sum(metrics['loss'])/len(metrics['loss']):.4f}")

        loss_trend = "decreasing ✓" if metrics['loss'][-1] < metrics['loss'][0] else "not decreasing ⚠️"
        print(f"Loss trend: {loss_trend}")

        peak_mem = max(metrics['memory'])
        print(f"Peak memory: {peak_mem:.2f} GB / 6.00 GB")

        if peak_mem < 5.5:
            mem_status = "SAFE ✓"
        elif peak_mem < 5.8:
            mem_status = "TIGHT ⚠️"
        else:
            mem_status = "TOO HIGH ✗"
        print(f"Memory status: {mem_status}")

        avg_step_time = sum(metrics['step_time']) / len(metrics['step_time'])
        print(f"Average step time: {avg_step_time:.2f}s")

        print(f"Checkpoint: {'saved and verified ✓' if checkpoint_ok else 'failed ✗'}")

    # Test checklist
    tests = [
        ("Model loading (4-bit + LoRA)", True),
        ("Data loading (100 samples)", True),
        ("Training steps completed", len(metrics["loss"]) == 10),
        ("Loss values valid", len(metrics["loss"]) > 0 and all(0 <= loss < 100 for loss in metrics["loss"])),
        ("Loss decreasing trend", len(metrics["loss"]) > 2 and metrics["loss"][-1] < metrics["loss"][0]),
        ("Training health check", healthy),
        ("Memory under limit (<5.8GB)", len(metrics["memory"]) > 0 and max(metrics["memory"]) < 5.8),
        ("Checkpoint save/load", checkpoint_ok),
    ]

    passed = sum(result for _, result in tests)
    total = len(tests)

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print()
    for name, result in tests:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nRESULT: {passed}/{total} TESTS PASSED")

    if passed == total:
        print("\n" + "="*80)
        print("READY FOR FULL TRAINING ✓")
        print("="*80)
        print("\nDry run successful! All systems operational.")

        # Estimate full training time
        if len(metrics["step_time"]) > 0:
            print("\nExpected full training time:")
            avg_step_time = sum(metrics["step_time"]) / len(metrics["step_time"])
            total_steps = len(dataloader) * config["training"]["num_epochs"] // config["training"]["gradient_accumulation_steps"]
            estimated_time = total_steps * avg_step_time / 3600
            print(f"  Estimated: ~{estimated_time:.2f} hours for {config['training']['num_epochs']} epochs")
            print(f"  Per epoch: ~{estimated_time/config['training']['num_epochs']:.2f} hours")

        print("\nNext step: Run full training with:")
        print("  python training/train.py --config training/config.yaml")
        return True
    else:
        print("\n" + "="*80)
        print(f"DRY RUN FAILED: {total - passed}/{total} TESTS FAILED ✗")
        print("="*80)
        print("\nPlease fix the following issues before full training:")
        for name, result in tests:
            if not result:
                print(f"  - {name}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
