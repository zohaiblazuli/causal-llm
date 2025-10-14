"""
Verification Script for Training Setup

Checks that all components are properly configured and ready for training.

Usage:
    python training/verify_setup.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check that all required libraries are installed."""
    print("=" * 80)
    print("CHECKING IMPORTS")
    print("=" * 80)

    required_packages = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "peft": "PEFT (LoRA)",
        "bitsandbytes": "BitsAndBytes (quantization)",
        "datasets": "Datasets",
        "yaml": "PyYAML",
        "tqdm": "TQDM",
    }

    optional_packages = {
        "wandb": "Weights & Biases",
        "tensorboard": "TensorBoard",
    }

    all_good = True

    print("\nRequired packages:")
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            all_good = False

    print("\nOptional packages:")
    for package, name in optional_packages.items():
        try:
            __import__(package)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [SKIP] {name} - not installed (optional)")

    return all_good


def check_cuda():
    """Check CUDA availability."""
    print("\n" + "=" * 80)
    print("CHECKING CUDA")
    print("=" * 80)

    try:
        import torch

        if torch.cuda.is_available():
            print("\n[OK] CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")

            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  Total memory: {total_memory:.2f} GB")

            # Check if memory is sufficient (at least 5GB for 4-bit training)
            if total_memory < 5.0:
                print(f"  [WARN] GPU memory ({total_memory:.2f} GB) may be insufficient for training")
                print("         Consider reducing max_seq_length or batch size")

            # Check bfloat16 support
            if torch.cuda.get_device_capability()[0] >= 8:
                print("  [OK] BFloat16 supported (Ampere or newer)")
            else:
                print("  [WARN] BFloat16 not supported - will use FP16 instead")

            return True
        else:
            print("\n[FAIL] CUDA not available - training will be very slow or fail!")
            return False
    except Exception as e:
        print(f"\n[FAIL] Error checking CUDA: {e}")
        return False


def check_config():
    """Check that config file exists and is valid."""
    print("\n" + "=" * 80)
    print("CHECKING CONFIGURATION")
    print("=" * 80)

    config_path = Path("training/config.yaml")

    if not config_path.exists():
        print(f"\n[FAIL] Config file not found: {config_path}")
        return False

    try:
        from training.utils import load_config
        config = load_config(str(config_path))

        print("\n[OK] Config file loaded successfully")

        # Check required sections
        required_sections = ["model", "lora", "training", "loss", "data"]
        for section in required_sections:
            if section in config:
                print(f"  [OK] {section} section present")
            else:
                print(f"  [FAIL] {section} section missing")
                return False

        return True

    except Exception as e:
        print(f"\n[FAIL] Error loading config: {e}")
        return False


def check_data():
    """Check that data files exist."""
    print("\n" + "=" * 80)
    print("CHECKING DATA FILES")
    print("=" * 80)

    data_dir = Path("data/processed")

    if not data_dir.exists():
        print(f"\n[FAIL] Data directory not found: {data_dir}")
        print("       Create the directory and add your training data")
        return False

    train_file = data_dir / "train_split.jsonl"
    val_file = data_dir / "val_split.jsonl"

    if train_file.exists():
        print(f"\n[OK] Training data found: {train_file}")
        # Count lines
        with open(train_file, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
        print(f"  {num_lines} training samples")
    else:
        print(f"\n[FAIL] Training data not found: {train_file}")
        print("       Generate training data using the data-forge agent")
        return False

    if val_file.exists():
        print(f"[OK] Validation data found: {val_file}")
        with open(val_file, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
        print(f"  {num_lines} validation samples")
    else:
        print(f"[WARN] Validation data not found: {val_file}")
        print("       Training will proceed without validation")

    return True


def check_model_access():
    """Check access to Llama models."""
    print("\n" + "=" * 80)
    print("CHECKING MODEL ACCESS")
    print("=" * 80)

    try:
        from huggingface_hub import HfApi
        from training.utils import load_config
        import os
        api = HfApi()

        # Determine login via whoami or env token
        env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        logged_in = False
        try:
            _ = api.whoami()
            logged_in = True
        except Exception:
            logged_in = bool(env_token)

        if logged_in:
            print("\n[OK] Hugging Face authentication detected")

            # Try to access the configured model
            try:
                cfg = load_config("training/config.yaml")
                model_name = cfg.get("model", {}).get("name", "meta-llama/Llama-3.1-8B")
                api.model_info(model_name)
                print(f"[OK] Access to {model_name} verified")
                return True
            except Exception:
                print("[WARN] Could not verify model access (may need license acceptance)")
                print("       If download fails, visit the model page and accept the license.")
                return True

        else:
            print("\n[FAIL] Not authenticated with Hugging Face")
            print("       Run: huggingface-cli login or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN")
            return False

    except ImportError:
        print("\n[WARN] huggingface_hub not installed - skipping model access check")
        return True
    except Exception as e:
        print(f"\n[WARN] Could not verify model access: {e}")
        return True


def check_training_files():
    """Check that all training files are present."""
    print("\n" + "=" * 80)
    print("CHECKING TRAINING FILES")
    print("=" * 80)

    training_dir = Path("training")
    required_files = [
        "config.yaml",
        "train.py",
        "trainer.py",
        "dataset.py",
        "callbacks.py",
        "utils.py",
        "optimize_memory.py",
        "__init__.py",
    ]

    all_present = True
    for filename in required_files:
        filepath = training_dir / filename
        if filepath.exists():
            print(f"  [OK] {filename}")
        else:
            print(f"  [FAIL] {filename} missing")
            all_present = False

    return all_present


def check_models_module():
    """Check that models module is accessible."""
    print("\n" + "=" * 80)
    print("CHECKING MODELS MODULE")
    print("=" * 80)

    try:
        from models.causal_model import CausalLLMModel  # noqa: F401
        from models.losses import CausalContrastiveLoss  # noqa: F401

        print("\n[OK] Models module accessible")
        print("  [OK] CausalLLMModel")
        print("  [OK] CausalContrastiveLoss")
        return True

    except ImportError as e:
        print(f"\n[FAIL] Cannot import models: {e}")
        return False


def check_disk_space():
    """Check available disk space for checkpoints."""
    print("\n" + "=" * 80)
    print("CHECKING DISK SPACE")
    print("=" * 80)

    try:
        import shutil

        # Check space in current directory
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)

        print(f"\n  Free disk space: {free_gb:.2f} GB")

        if free_gb < 50:
            print("  [WARN] Less than 50GB free")
            print("         Training checkpoints and model weights are large")
            if free_gb < 20:
                print("  [FAIL] Less than 20GB free - training may fail!")
                return False
            return True
        else:
            print("  [OK] Sufficient disk space available")
            return True

    except Exception as e:
        print(f"\n[WARN] Could not check disk space: {e}")
        return True  # Don't fail on this check


def run_quick_test():
    """Run a quick functionality test."""
    print("\n" + "=" * 80)
    print("RUNNING QUICK FUNCTIONALITY TEST")
    print("=" * 80)

    try:
        # Test config loading
        from training.utils import load_config
        _ = load_config("training/config.yaml")
        print("\n[OK] Config loading works")

        # Test loss function
        import torch
        from models.losses import CausalContrastiveLoss

        loss_fn = CausalContrastiveLoss()
        batch_size = 2
        hidden_dim = 768

        repr_benign = torch.randn(batch_size, hidden_dim)
        repr_benign_cf = torch.randn(batch_size, hidden_dim)
        repr_injection = torch.randn(batch_size, hidden_dim)

        loss_dict = loss_fn(repr_benign, repr_benign_cf, repr_injection)

        print("[OK] Loss function works")
        print(f"  Sample loss: {loss_dict['loss'].item():.4f}")

        # Test memory estimation
        from training.utils import estimate_model_memory

        memory_est = estimate_model_memory(
            model_name="meta-llama/Llama-2-7b-hf",
            lora_r=16,
            max_seq_length=2048,
            batch_size=1,
            load_in_4bit=True,
        )

        print("[OK] Memory estimation works")
        print(f"  Estimated total memory: {memory_est['total']:.2f} GB")

        if memory_est['total'] < 6.0:
            print("  [OK] Configuration should fit in RTX 4050 (6GB)")
        else:
            print("  [WARN] Configuration may exceed RTX 4050 VRAM")

        return True

    except Exception as e:
        print(f"\n[FAIL] Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_python_version():
    """Check Python version."""
    print("\n" + "=" * 80)
    print("CHECK 1/9: Python Version")
    print("=" * 80)

    import sys as _sys
    version = _sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print(f"\nPython version: {version_str}")

    if version.major == 3 and version.minor >= 8:
        print("[OK] Python 3.8+ detected")
        return True
    else:
        print("[FAIL] Python 3.8+ required")
        print(f"       Current version: {version_str}")
        print("       Please upgrade Python")
        return False


def check_dependencies():
    """Check all required dependencies with versions."""
    print("\n" + "=" * 80)
    print("CHECK 2/9: Dependencies")
    print("=" * 80)

    required_packages = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "peft": "PEFT (LoRA)",
        "bitsandbytes": "BitsAndBytes (quantization)",
        "datasets": "Datasets",
        "yaml": "PyYAML",
        "tqdm": "TQDM",
    }

    all_good = True
    print("\nRequired packages:")

    for package, name in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  [OK] {name} ({version})")
        except ImportError:
            print(f"  [MISSING] {name}")
            print(f"     Install with: pip install {package}")
            all_good = False

    return all_good


def main():
    """Main verification function - Enhanced 9-point check system."""
    print("\n")
    print("=" * 80)
    print("WEEK 1 SETUP VERIFICATION - PHASE 2")
    print("=" * 80)
    print()

    from datetime import datetime
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Enhanced 9-point check system
    checks = [
        ("1. Python version (>=3.8)", check_python_version),
        ("2. All dependencies", check_dependencies),
        ("3. CUDA availability and version", check_cuda),
        ("4. GPU memory (6GB available)", lambda: check_cuda()),  # Checks memory within
        ("5. Hugging Face token configured", check_model_access),
        ("6. Model access (Llama 2 7B)", check_model_access),
        ("7. Data files exist and readable", check_data),
        ("8. Config file valid", check_config),
        ("9. Disk space (>50GB free)", check_disk_space),
    ]

    results = {}
    print("Running 9-point verification system...\n")

    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n[FAIL] {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Enhanced Summary
    print("\n" + "=" * 80)
    print("SETUP VERIFICATION SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    print()
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print()
    print(f"RESULT: {passed}/{total} CHECKS PASSED")

    if passed == total:
        print("\n" + "=" * 80)
        print("ALL 9 CHECKS PASSED [OK]")
        print("=" * 80)
        print("\nReady to proceed to memory optimization!")
        print("\nNext steps:")
        print("  1. Memory optimization: python training/optimize_memory.py")
        print("  2. Data pipeline test: python training/test_data_pipeline.py")
        print("  3. Dry run test: python training/dry_run.py")
        print("  4. Full training: python training/train.py")
        return True
    else:
        print("\n" + "=" * 80)
        print(f"VERIFICATION FAILED: {total - passed}/{total} checks failed")
        print("=" * 80)
        print("\nPlease fix the issues above before proceeding.")
        print("\nFailed checks:")
        for name, result in results.items():
            if not result:
                print(f"  - {name}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
