"""
Memory Optimization and Profiling Script

Test different configurations to find optimal settings for RTX 4050 (6GB VRAM).

Usage:
    python training/optimize_memory.py --config training/config.yaml
    python training/optimize_memory.py --config training/config.yaml --test-batch-sizes
    python training/optimize_memory.py --config training/config.yaml --test-sequence-lengths
"""

import argparse
import gc
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.amp import autocast
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import load_config, set_seed, get_memory_usage, clear_memory


def test_model_loading(config: dict) -> Dict[str, float]:
    """
    Test memory usage for model loading.

    Args:
        config: Configuration dictionary

    Returns:
        Memory usage statistics
    """
    print("\n" + "="*80)
    print("TEST 1: Model Loading Memory")
    print("="*80)

    model_config = config["model"]
    lora_config_dict = config["lora"]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # Initial memory
    clear_memory()
    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory['allocated']:.2f} GB")

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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load base model
    from transformers import AutoModelForCausalLM

    print(f"\nLoading base model: {model_config['name']}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )

    after_base_memory = get_memory_usage()
    print(f"After base model: {after_base_memory['allocated']:.2f} GB "
          f"(+{after_base_memory['allocated'] - initial_memory['allocated']:.2f} GB)")

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, lora_config)

    after_lora_memory = get_memory_usage()
    print(f"After LoRA: {after_lora_memory['allocated']:.2f} GB "
          f"(+{after_lora_memory['allocated'] - after_base_memory['allocated']:.2f} GB)")

    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    after_gc_memory = get_memory_usage()
    print(f"After gradient checkpointing: {after_gc_memory['allocated']:.2f} GB")

    # Summary
    print("\nMemory Breakdown:")
    print(f"  Base model (4-bit): {after_base_memory['allocated'] - initial_memory['allocated']:.2f} GB")
    print(f"  LoRA adapter: {after_lora_memory['allocated'] - after_base_memory['allocated']:.2f} GB")
    print(f"  Total model: {after_lora_memory['allocated']:.2f} GB")

    # Cleanup
    del model, base_model
    clear_memory()

    return {
        "base_model": after_base_memory['allocated'] - initial_memory['allocated'],
        "lora": after_lora_memory['allocated'] - after_base_memory['allocated'],
        "total": after_lora_memory['allocated']
    }


def test_forward_pass(
    config: dict,
    batch_size: int = 1,
    seq_length: int = 2048
) -> Dict[str, float]:
    """
    Test memory usage for forward pass.

    Args:
        config: Configuration dictionary
        batch_size: Batch size to test
        seq_length: Sequence length to test

    Returns:
        Memory usage statistics
    """
    print("\n" + "="*80)
    print(f"TEST 2: Forward Pass Memory (batch={batch_size}, seq_len={seq_length})")
    print("="*80)

    model_config = config["model"]
    lora_config_dict = config["lora"]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # Load model
    clear_memory()

    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    compute_dtype = compute_dtype_map.get(
        model_config.get("bnb_4bit_compute_dtype", "bfloat16"),
        torch.bfloat16
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )

    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    before_forward_memory = get_memory_usage()
    print(f"Before forward: {before_forward_memory['allocated']:.2f} GB")

    # Create dummy input
    input_ids = torch.randint(0, 32000, (batch_size, seq_length)).to("cuda")
    attention_mask = torch.ones_like(input_ids)

    # Forward pass with mixed precision
    try:
        with autocast(device_type="cuda", dtype=compute_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        after_forward_memory = get_memory_usage()
        print(f"After forward: {after_forward_memory['allocated']:.2f} GB "
              f"(+{after_forward_memory['allocated'] - before_forward_memory['allocated']:.2f} GB)")

        success = True
        peak_memory = after_forward_memory['allocated']

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM Error! Configuration too large for available VRAM")
            success = False
            peak_memory = float('inf')
        else:
            raise

    # Cleanup
    del model, base_model, input_ids, attention_mask
    clear_memory()

    return {
        "success": success,
        "peak_memory": peak_memory,
        "activation_memory": peak_memory - before_forward_memory['allocated'] if success else float('inf')
    }


def test_training_step(
    config: dict,
    batch_size: int = 1,
    seq_length: int = 2048,
    grad_accum_steps: int = 1
) -> Dict[str, float]:
    """
    Test memory usage for full training step.

    Args:
        config: Configuration dictionary
        batch_size: Batch size to test
        seq_length: Sequence length to test
        grad_accum_steps: Gradient accumulation steps

    Returns:
        Memory usage statistics
    """
    print("\n" + "="*80)
    print(f"TEST 3: Training Step (batch={batch_size}, seq={seq_length}, grad_accum={grad_accum_steps})")
    print("="*80)

    model_config = config["model"]
    lora_config_dict = config["lora"]
    training_config = config["training"]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # Load model
    clear_memory()

    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    compute_dtype = compute_dtype_map.get(
        model_config.get("bnb_4bit_compute_dtype", "bfloat16"),
        torch.bfloat16
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )

    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()

    # Create optimizer
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_config["learning_rate"]
    )

    before_train_memory = get_memory_usage()
    print(f"Before training: {before_train_memory['allocated']:.2f} GB")

    # Dummy inputs
    input_ids = torch.randint(0, 32000, (batch_size, seq_length)).to("cuda")
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    try:
        # Forward pass
        with autocast(device_type="cuda", dtype=compute_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / grad_accum_steps

        after_forward_memory = get_memory_usage()
        print(f"After forward: {after_forward_memory['allocated']:.2f} GB")

        # Backward pass
        loss.backward()

        after_backward_memory = get_memory_usage()
        print(f"After backward: {after_backward_memory['allocated']:.2f} GB "
              f"(+{after_backward_memory['allocated'] - after_forward_memory['allocated']:.2f} GB)")

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        after_optimizer_memory = get_memory_usage()
        print(f"After optimizer: {after_optimizer_memory['allocated']:.2f} GB")

        success = True
        peak_memory = after_backward_memory['allocated']

        print(f"\nPeak memory: {peak_memory:.2f} GB")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM Error! Configuration too large")
            success = False
            peak_memory = float('inf')
        else:
            raise

    # Cleanup
    del model, base_model, optimizer, input_ids, attention_mask, labels
    clear_memory()

    return {
        "success": success,
        "peak_memory": peak_memory
    }


def find_optimal_batch_size(config: dict) -> int:
    """
    Binary search to find maximum batch size that fits in memory.

    Args:
        config: Configuration dictionary

    Returns:
        Optimal batch size
    """
    print("\n" + "="*80)
    print("FINDING OPTIMAL BATCH SIZE")
    print("="*80)

    seq_length = config["data"]["max_length"]

    # Binary search
    min_batch = 1
    max_batch = 4
    optimal_batch = 1

    while min_batch <= max_batch:
        test_batch = (min_batch + max_batch) // 2

        print(f"\nTesting batch size: {test_batch}")

        result = test_training_step(config, batch_size=test_batch, seq_length=seq_length)

        if result["success"]:
            optimal_batch = test_batch
            print(f"Success! Peak memory: {result['peak_memory']:.2f} GB")
            min_batch = test_batch + 1
        else:
            print("Failed (OOM)")
            max_batch = test_batch - 1

    return optimal_batch


def recommend_configuration(config: dict, vram_gb: float = 6.0):
    """
    Analyze configuration and provide recommendations.

    Args:
        config: Configuration dictionary
        vram_gb: Available VRAM in GB
    """
    print("\n" + "="*80)
    print("CONFIGURATION RECOMMENDATIONS")
    print("="*80)

    print(f"\nTarget VRAM: {vram_gb} GB")

    # Test current config
    print("\n1. Testing current configuration...")
    result = test_training_step(
        config,
        batch_size=config["training"]["per_device_train_batch_size"],
        seq_length=config["data"]["max_length"],
        grad_accum_steps=config["training"]["gradient_accumulation_steps"]
    )

    if result["success"]:
        print(f"\nCurrent config works! Peak memory: {result['peak_memory']:.2f} GB")
        margin = vram_gb - result['peak_memory']
        print(f"Memory margin: {margin:.2f} GB ({margin/vram_gb*100:.1f}%)")

        if margin > 1.0:
            print("\nYou have significant headroom. Consider:")
            print("  - Increasing max_seq_length for longer contexts")
            print("  - Increasing LoRA rank for more capacity")
            print("  - Reducing gradient_accumulation_steps for faster training")
    else:
        print("\nCurrent config FAILED (OOM)!")
        print("\nRecommendations:")
        print("  - Reduce max_seq_length (current: {})".format(config["data"]["max_length"]))
        print("  - Ensure batch_size = 1 (current: {})".format(config["training"]["per_device_train_batch_size"]))
        print("  - Increase gradient_accumulation_steps (current: {})".format(config["training"]["gradient_accumulation_steps"]))
        print("  - Enable gradient_checkpointing (current: {})".format(config["training"].get("gradient_checkpointing", False)))

    # Test optimal batch size
    print("\n2. Finding optimal batch size...")
    optimal_batch = find_optimal_batch_size(config)
    print(f"\nOptimal batch size: {optimal_batch}")

    if optimal_batch == 1:
        print("Batch size of 1 is optimal for this GPU.")
        print("Use gradient accumulation for effective larger batches.")
        recommended_grad_accum = 16
        print(f"Recommended gradient_accumulation_steps: {recommended_grad_accum}")
        print(f"Effective batch size: {optimal_batch * recommended_grad_accum}")
    else:
        print(f"You can use batch size {optimal_batch} directly")
        recommended_grad_accum = 16 // optimal_batch
        print(f"Recommended gradient_accumulation_steps: {recommended_grad_accum}")


def print_detailed_memory_report(config: dict, vram_gb: float = 6.0):
    """
    Print detailed memory breakdown report with recommendations.

    Args:
        config: Configuration dictionary
        vram_gb: Available VRAM in GB
    """
    print("\n" + "="*80)
    print("DETAILED MEMORY OPTIMIZATION REPORT")
    print("="*80)

    from datetime import datetime
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target GPU: RTX 4050 (6GB VRAM)")
    print(f"Configuration: {config['model']['name']}")
    print(f"LoRA rank: {config['lora']['r']}")
    print(f"Max sequence length: {config['data']['max_length']}")
    print(f"Batch size: {config['training']['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")

    # Test each component
    print("\n" + "="*80)
    print("STEP 1: Model Loading (4-bit quantized)")
    print("="*80)
    model_memory = test_model_loading(config)

    print("\n" + "="*80)
    print("STEP 2: Forward Pass")
    print("="*80)
    forward_result = test_forward_pass(
        config,
        batch_size=config["training"]["per_device_train_batch_size"],
        seq_length=config["data"]["max_length"]
    )

    print("\n" + "="*80)
    print("STEP 3: Full Training Step")
    print("="*80)
    training_result = test_training_step(
        config,
        batch_size=config["training"]["per_device_train_batch_size"],
        seq_length=config["data"]["max_length"],
        grad_accum_steps=config["training"]["gradient_accumulation_steps"]
    )

    # Generate detailed report
    print("\n" + "="*80)
    print("MEMORY BREAKDOWN")
    print("="*80)

    if training_result["success"]:
        base_model_mem = model_memory["base_model"]
        lora_mem = model_memory["lora"]
        activation_mem = forward_result["activation_memory"]
        peak_mem = training_result["peak_memory"]

        # Estimate optimizer and gradient memory
        optimizer_mem = lora_mem * 0.5  # Approximate
        gradient_mem = lora_mem
        cache_mem = peak_mem - (base_model_mem + lora_mem + activation_mem + optimizer_mem + gradient_mem)
        cache_mem = max(0, cache_mem)

        print()
        print(f"  Base Model (4-bit quantized):     {base_model_mem:5.2f} GB")
        print(f"  LoRA Adapters (rank {config['lora']['r']:2d}):          {lora_mem:5.2f} GB")
        print(f"  Activations (seq_len {config['data']['max_length']}):      {activation_mem:5.2f} GB")
        print(f"  Gradients:                        {gradient_mem:5.2f} GB")
        print(f"  Optimizer State (8-bit):          {optimizer_mem:5.2f} GB")
        print(f"  Cache/Buffers:                    {cache_mem:5.2f} GB")
        print("  " + "-"*44)
        print(f"  TOTAL PEAK MEMORY:                {peak_mem:5.2f} GB / {vram_gb:.2f} GB")
        print()

        margin = vram_gb - peak_mem
        margin_pct = (margin / vram_gb) * 100

        print(f"MARGIN:                            {margin:5.2f} GB ({margin_pct:.0f}%)")

        # Status assessment
        if margin >= 1.0:
            status = "SAFE"
            status_symbol = "✓"
        elif margin >= 0.5:
            status = "TIGHT (recommend reducing to 1024 seq_len)"
            status_symbol = "⚠️"
        else:
            status = "UNSAFE (will likely OOM)"
            status_symbol = "✗"

        print(f"STATUS:                            {status_symbol}  {status}")

        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        print()

        if margin >= 1.0:
            print("1. Current configuration is SAFE with good margin ✓")
            print("2. You can proceed with training confidently")
            print(f"3. Memory headroom of {margin:.2f}GB allows for small fluctuations")
        elif margin >= 0.5:
            print("1. Current config will work but has minimal margin ⚠️")
            print("2. Consider reducing max_seq_length to 1024 for safety")

            # Estimate with reduced seq_length
            estimated_new_activation = activation_mem * (1024 / config['data']['max_length'])
            estimated_new_total = peak_mem - activation_mem + estimated_new_activation
            new_margin = vram_gb - estimated_new_total
            new_margin_pct = (new_margin / vram_gb) * 100

            print(f"3. With seq_len=1024: estimated {estimated_new_total:.2f} GB ({new_margin_pct:.0f}% margin) ✓")
            print("4. Monitor memory during training (may need adjustment)")
        else:
            print("1. Current configuration is UNSAFE ✗")
            print("2. MUST reduce max_seq_length (try 512 or 768)")
            print("3. Alternative: increase gradient_accumulation_steps")
            print("4. Last resort: reduce LoRA rank to 8")

        print()
        verdict = "READY TO TRAIN" if margin >= 0.3 else "NOT READY - ADJUST CONFIG"
        verdict_symbol = "✓" if margin >= 0.3 else "✗"
        print(f"VERDICT: {verdict} {verdict_symbol}")
        if margin < 0.3:
            print("\nCRITICAL: Reduce memory usage before training!")

    else:
        print("\n✗ Training step failed with OOM")
        print("\nRECOMMENDATIONS:")
        print("  1. Reduce max_seq_length (current: {})".format(config["data"]["max_length"]))
        print("  2. Ensure batch_size = 1")
        print("  3. Enable gradient_checkpointing")
        print("  4. Reduce LoRA rank")


def main():
    """Main optimization function - Enhanced with detailed reporting."""
    parser = argparse.ArgumentParser(description="Optimize memory usage for training")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--test-batch-sizes", action="store_true")
    parser.add_argument("--test-sequence-lengths", action="store_true")
    parser.add_argument("--full-analysis", action="store_true")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(42)

    print("="*80)
    print("WEEK 1 MEMORY OPTIMIZATION - PHASE 2")
    print("="*80)

    # Run tests
    if args.full_analysis or not (args.test_batch_sizes or args.test_sequence_lengths):
        # Full analysis with detailed report
        print_detailed_memory_report(config, vram_gb=6.0)

    elif args.test_batch_sizes:
        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            test_training_step(config, batch_size=batch_size)

    elif args.test_sequence_lengths:
        # Test different sequence lengths
        for seq_length in [512, 1024, 2048, 4096]:
            test_forward_pass(config, seq_length=seq_length)

    print("\n" + "="*80)
    print("MEMORY OPTIMIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
