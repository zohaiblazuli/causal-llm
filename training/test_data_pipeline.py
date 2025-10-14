"""
Data Pipeline Test Script

Validates the data loading pipeline with comprehensive checks:
- File reading and parsing
- Triplet extraction
- Tokenization
- Batch collation
- Data loader performance
- Data integrity

Usage:
    python training/test_data_pipeline.py
    python training/test_data_pipeline.py --num-samples 50 --batch-size 2
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import CausalContrastiveDataset, CausalContrastiveCollator
from training.utils import load_config


def test_file_reading(data_path: str, num_samples: int = 10) -> List[Dict]:
    """
    Test reading and parsing JSONL file.

    Args:
        data_path: Path to JSONL file
        num_samples: Number of samples to read

    Returns:
        List of parsed samples
    """
    print("\n" + "="*80)
    print("TEST 1: File Reading and Parsing")
    print("="*80)

    if not Path(data_path).exists():
        print(f"\n✗ Data file not found: {data_path}")
        return []

    print(f"\nReading {num_samples} samples from {data_path}")

    samples = []
    required_fields = [
        "system_instruction",
        "benign_input",
        "benign_cf_input",
        "injection_input",
        "benign_output"
    ]

    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            try:
                sample = json.loads(line.strip())

                # Validate fields
                missing_fields = [field for field in required_fields if field not in sample]
                if missing_fields:
                    print(f"  ✗ Sample {i}: Missing fields {missing_fields}")
                    continue

                samples.append(sample)

                # Print first sample details
                if i == 0:
                    print(f"\n  Sample 0 structure:")
                    for field in required_fields:
                        value = sample[field]
                        preview = value[:50] + "..." if len(value) > 50 else value
                        print(f"    - {field}: {preview}")

            except json.JSONDecodeError as e:
                print(f"  ✗ Sample {i}: JSON parse error - {e}")
                continue

    print(f"\n✓ Successfully parsed {len(samples)}/{num_samples} samples")

    # Check data diversity
    unique_systems = len(set(s["system_instruction"] for s in samples))
    print(f"  Unique system instructions: {unique_systems}")

    return samples


def test_tokenization(samples: List[Dict], model_name: str = "meta-llama/Llama-2-7b-hf") -> bool:
    """
    Test tokenization of triplets.

    Args:
        samples: List of data samples
        model_name: Model name for tokenizer

    Returns:
        True if successful
    """
    print("\n" + "="*80)
    print("TEST 2: Tokenization")
    print("="*80)

    print(f"\nLoading tokenizer: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False

    # Test tokenization of first sample
    sample = samples[0]

    print("\nTokenizing sample triplet...")

    try:
        # Format prompts
        benign_prompt = f"""### System Instruction:
{sample['system_instruction']}

### User Input:
{sample['benign_input']}

### Response:
"""

        benign_cf_prompt = f"""### System Instruction:
{sample['system_instruction']}

### User Input:
{sample['benign_cf_input']}

### Response:
"""

        injection_prompt = f"""### System Instruction:
{sample['system_instruction']}

### User Input:
{sample['injection_input']}

### Response:
"""

        # Tokenize
        benign_tokens = tokenizer(benign_prompt, truncation=True, max_length=2048)
        benign_cf_tokens = tokenizer(benign_cf_prompt, truncation=True, max_length=2048)
        injection_tokens = tokenizer(injection_prompt, truncation=True, max_length=2048)

        print(f"  Benign tokens: {len(benign_tokens['input_ids'])}")
        print(f"  Benign CF tokens: {len(benign_cf_tokens['input_ids'])}")
        print(f"  Injection tokens: {len(injection_tokens['input_ids'])}")

        # Check for truncation
        if len(benign_tokens['input_ids']) >= 2048:
            print("  ⚠ Benign input truncated (consider reducing max_length)")

        print("\n✓ Tokenization successful")
        return True

    except Exception as e:
        print(f"\n✗ Tokenization failed: {e}")
        return False


def test_dataset_loading(
    data_path: str,
    model_name: str = "meta-llama/Llama-2-7b-hf",
    max_samples: int = 10,
    max_length: int = 2048
) -> CausalContrastiveDataset:
    """
    Test dataset loading with CausalContrastiveDataset.

    Args:
        data_path: Path to data file
        model_name: Model name for tokenizer
        max_samples: Maximum samples to load
        max_length: Maximum sequence length

    Returns:
        Loaded dataset
    """
    print("\n" + "="*80)
    print("TEST 3: Dataset Loading")
    print("="*80)

    print(f"\nInitializing dataset with max_samples={max_samples}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        start_time = time.time()

        dataset = CausalContrastiveDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            max_samples=max_samples
        )

        load_time = time.time() - start_time

        print(f"✓ Dataset loaded in {load_time:.2f} seconds")
        print(f"  Dataset size: {len(dataset)}")

        # Test getting a sample
        print("\nTesting __getitem__...")
        sample = dataset[0]

        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Benign input shape: {sample['benign_input_ids'].shape}")
        print(f"  Benign CF input shape: {sample['benign_cf_input_ids'].shape}")
        print(f"  Injection input shape: {sample['injection_input_ids'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")

        # Check dtypes
        print(f"\n  Benign input dtype: {sample['benign_input_ids'].dtype}")
        print(f"  Labels dtype: {sample['labels'].dtype}")

        # Check for proper masking in labels
        num_masked = (sample['labels'] == -100).sum().item()
        total_tokens = sample['labels'].numel()
        print(f"\n  Masked tokens in labels: {num_masked}/{total_tokens} ({num_masked/total_tokens*100:.1f}%)")

        if num_masked == 0:
            print("  ⚠ Warning: No tokens masked in labels (prompt should be masked)")

        print("\n✓ Dataset loading successful")
        return dataset

    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_collator(
    dataset: CausalContrastiveDataset,
    batch_size: int = 2
) -> bool:
    """
    Test data collator for batching.

    Args:
        dataset: Dataset instance
        batch_size: Batch size to test

    Returns:
        True if successful
    """
    print("\n" + "="*80)
    print("TEST 4: Data Collator")
    print("="*80)

    print(f"\nCreating batch with size {batch_size}")

    try:
        collator = CausalContrastiveCollator(
            tokenizer=dataset.tokenizer,
            padding="longest"
        )

        # Get samples
        samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]

        # Collate
        batch = collator(samples)

        print(f"✓ Batch created successfully")
        print(f"\n  Batch keys: {list(batch.keys())}")
        print(f"  Benign input shape: {batch['benign_input_ids'].shape}")
        print(f"  Benign CF input shape: {batch['benign_cf_input_ids'].shape}")
        print(f"  Injection input shape: {batch['injection_input_ids'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")

        # Verify shapes match
        assert batch['benign_input_ids'].shape[0] == batch_size or batch['benign_input_ids'].shape[0] == len(dataset)
        assert batch['benign_input_ids'].shape == batch['benign_attention_mask'].shape

        print("\n✓ Collator test successful")
        return True

    except Exception as e:
        print(f"\n✗ Collator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(
    dataset: CausalContrastiveDataset,
    batch_size: int = 1,
    num_workers: int = 2,
    num_batches: int = 5
) -> bool:
    """
    Test DataLoader with multi-processing.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        num_batches: Number of batches to test

    Returns:
        True if successful
    """
    print("\n" + "="*80)
    print("TEST 5: DataLoader Performance")
    print("="*80)

    print(f"\nCreating DataLoader (batch_size={batch_size}, num_workers={num_workers})")

    try:
        collator = CausalContrastiveCollator(
            tokenizer=dataset.tokenizer,
            padding="longest"
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f"✓ DataLoader created")
        print(f"  Total batches: {len(dataloader)}")

        # Test iteration
        print(f"\nIterating through {num_batches} batches...")

        start_time = time.time()

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Verify batch
            assert 'benign_input_ids' in batch
            assert 'benign_cf_input_ids' in batch
            assert 'injection_input_ids' in batch
            assert 'labels' in batch

            if i == 0:
                print(f"  Batch 0 shape: {batch['benign_input_ids'].shape}")

        elapsed_time = time.time() - start_time
        batches_per_sec = num_batches / elapsed_time
        samples_per_sec = (num_batches * batch_size) / elapsed_time

        print(f"\n✓ DataLoader iteration successful")
        print(f"  Time: {elapsed_time:.2f} seconds")
        print(f"  Speed: {batches_per_sec:.2f} batches/sec")
        print(f"  Speed: {samples_per_sec:.2f} samples/sec")

        return True

    except Exception as e:
        print(f"\n✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_integrity(samples: List[Dict]) -> bool:
    """
    Test data integrity and quality.

    Args:
        samples: List of data samples

    Returns:
        True if data is valid
    """
    print("\n" + "="*80)
    print("TEST 6: Data Integrity")
    print("="*80)

    issues = []

    for i, sample in enumerate(samples):
        # Check for empty fields
        for field in ["system_instruction", "benign_input", "benign_cf_input", "injection_input", "benign_output"]:
            if not sample.get(field) or len(sample[field].strip()) == 0:
                issues.append(f"Sample {i}: Empty {field}")

        # Check for suspicious patterns
        if sample["benign_input"] == sample["injection_input"]:
            issues.append(f"Sample {i}: Benign and injection inputs are identical")

        if sample["benign_input"] == sample["benign_cf_input"]:
            issues.append(f"Sample {i}: Benign and benign_cf inputs are identical (should be variations)")

        # Check output is reasonable
        if len(sample["benign_output"]) < 5:
            issues.append(f"Sample {i}: Benign output suspiciously short")

    if issues:
        print(f"\n⚠ Found {len(issues)} data quality issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False
    else:
        print("\n✓ Data integrity check passed")
        print(f"  All {len(samples)} samples are valid")
        return True


def test_all_splits(train_path: str, val_path: str, test_path: str) -> Dict[str, int]:
    """
    Test loading all data splits.

    Args:
        train_path: Path to train split
        val_path: Path to val split
        test_path: Path to test split

    Returns:
        Dictionary with sample counts
    """
    print("\n" + "="*80)
    print("TEST 0: Loading All Splits")
    print("="*80)

    counts = {}

    for split_name, split_path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if Path(split_path).exists():
            with open(split_path, 'r') as f:
                count = sum(1 for _ in f)
            size_mb = Path(split_path).stat().st_size / (1024 * 1024)
            counts[split_name] = count
            print(f"  ✓ {split_name}: {count} examples ({size_mb:.1f}MB)")
        else:
            print(f"  ✗ {split_name}: NOT FOUND at {split_path}")
            counts[split_name] = 0

    return counts


def main():
    """Main test function - Enhanced for Week 1 validation."""
    parser = argparse.ArgumentParser(description="Test data pipeline")
    parser.add_argument("--data-path", type=str, default="data/processed/train_split.jsonl")
    parser.add_argument("--val-path", type=str, default="data/processed/val_split.jsonl")
    parser.add_argument("--test-path", type=str, default="data/processed/test_split.jsonl")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    print("="*80)
    print("WEEK 1 DATA PIPELINE TEST - PHASE 2")
    print("="*80)

    from datetime import datetime
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nConfiguration:")
    print(f"  Train path: {args.data_path}")
    print(f"  Val path: {args.val_path}")
    print(f"  Test path: {args.test_path}")
    print(f"  Model: {args.model_name}")
    print(f"  Test samples: {args.num_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max length: {args.max_length}")

    # Run tests
    tests = []

    # Test 0: Load all splits
    split_counts = test_all_splits(args.data_path, args.val_path, args.test_path)
    tests.append(("All splits loaded", all(count > 0 for count in split_counts.values())))

    # Test 1: File reading
    samples = test_file_reading(args.data_path, args.num_samples)
    tests.append(("File reading and parsing", len(samples) > 0))

    if not samples:
        print("\n✗ Cannot proceed without valid samples")
        sys.exit(1)

    # Test 2: Tokenization
    tests.append(("Tokenization", test_tokenization(samples, args.model_name)))

    # Test 3: Dataset loading
    dataset = test_dataset_loading(
        args.data_path,
        args.model_name,
        args.num_samples,
        args.max_length
    )
    tests.append(("Dataset loading", dataset is not None))

    if dataset is None:
        print("\n✗ Cannot proceed without loaded dataset")
        sys.exit(1)

    # Test 4: Data collator
    tests.append(("Data collator (batching)", test_data_collator(dataset, args.batch_size)))

    # Test 5: DataLoader
    tests.append(("DataLoader (multi-process)", test_dataloader(dataset, args.batch_size, args.num_workers)))

    # Test 6: Data integrity
    tests.append(("Data integrity check", test_data_integrity(samples)))

    # Enhanced Summary
    print("\n" + "="*80)
    print("DATA PIPELINE TEST SUMMARY")
    print("="*80)

    passed = sum(result for _, result in tests)
    total = len(tests)

    print()
    for name, result in tests:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nRESULT: {passed}/{total} TESTS PASSED")

    # Data statistics
    print("\n" + "="*80)
    print("DATA STATISTICS")
    print("="*80)
    print(f"\n  Train samples: {split_counts.get('train', 0)}")
    print(f"  Val samples: {split_counts.get('val', 0)}")
    print(f"  Test samples: {split_counts.get('test', 0)}")
    print(f"  Total samples: {sum(split_counts.values())}")

    if passed == total:
        print("\n" + "="*80)
        print("DATA PIPELINE READY ✓")
        print("="*80)
        print("\nAll pipeline tests passed successfully!")
        print("Ready to proceed to dry run training test.")
        return True
    else:
        print("\n" + "="*80)
        print("DATA PIPELINE FAILED ✗")
        print("="*80)
        print(f"\n{total - passed}/{total} tests failed.")
        print("Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
