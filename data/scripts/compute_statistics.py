"""
Compute Statistics - Week 1 Phase 2 Validation
Computes comprehensive dataset statistics including tokens, balance, and coverage.
"""

import json
import sys
from pathlib import Path
from collections import Counter

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def load_jsonl(filepath):
    """Load JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_statistics():
    """Compute dataset statistics."""
    print("=== Dataset Statistics ===\n")

    data_path = Path("C:/isef/data/processed/train_split.jsonl")
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        return False

    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} examples\n")

    # Token statistics
    print("1. Token Statistics:")

    if TRANSFORMERS_AVAILABLE:
        try:
            print("   Loading tokenizer (Llama-2-7b-hf)...")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

            token_counts = []
            over_2048 = 0
            over_4096 = 0

            for i, ex in enumerate(data):
                if (i + 1) % 500 == 0:
                    print(f"   Tokenizing {i + 1}/{len(data)}...")

                # Count tokens in full example (system + user + output)
                text = (
                    ex.get("system_instruction", "") + " " +
                    ex.get("user_input_benign_1", "") + " " +
                    ex.get("expected_output_1", "")
                )

                try:
                    tokens = tokenizer.encode(text, add_special_tokens=True)
                    token_count = len(tokens)
                    token_counts.append(token_count)

                    if token_count > 2048:
                        over_2048 += 1
                    if token_count > 4096:
                        over_4096 += 1
                except Exception as e:
                    print(f"   Warning: Failed to tokenize example {i}: {e}")

            if token_counts:
                if NUMPY_AVAILABLE:
                    avg_tokens = np.mean(token_counts)
                    median_tokens = np.median(token_counts)
                    max_tokens = np.max(token_counts)
                    min_tokens = np.min(token_counts)
                    p95_tokens = np.percentile(token_counts, 95)
                else:
                    avg_tokens = sum(token_counts) / len(token_counts)
                    sorted_tokens = sorted(token_counts)
                    median_tokens = sorted_tokens[len(sorted_tokens)//2]
                    max_tokens = max(token_counts)
                    min_tokens = min(token_counts)
                    p95_tokens = sorted_tokens[int(len(sorted_tokens) * 0.95)]

                print(f"   Average tokens: {avg_tokens:.0f}")
                print(f"   Median tokens: {median_tokens:.0f}")
                print(f"   Min tokens: {min_tokens}")
                print(f"   Max tokens: {max_tokens}")
                print(f"   95th percentile: {p95_tokens:.0f}")
                print(f"   Examples >2048: {over_2048} ({over_2048/len(data)*100:.1f}%)")
                print(f"   Examples >4096: {over_4096} ({over_4096/len(data)*100:.1f}%)")
            else:
                print("   ERROR: No tokens counted")
                avg_tokens = 0
                max_tokens = 0
                over_2048 = 0

        except Exception as e:
            print(f"   ERROR: Tokenizer failed: {e}")
            print("   Install with: pip install transformers")
            TRANSFORMERS_AVAILABLE = False

    if not TRANSFORMERS_AVAILABLE:
        print("   WARNING: Transformers not available, using word count estimate")
        word_counts = []
        for ex in data:
            text = (
                ex.get("system_instruction", "") + " " +
                ex.get("user_input_benign_1", "") + " " +
                ex.get("expected_output_1", "")
            )
            words = len(text.split())
            word_counts.append(words)

        # Rough estimate: 1 token ~= 0.75 words
        if NUMPY_AVAILABLE:
            avg_tokens = np.mean(word_counts) / 0.75
            max_tokens = np.max(word_counts) / 0.75
        else:
            avg_tokens = (sum(word_counts) / len(word_counts)) / 0.75
            max_tokens = max(word_counts) / 0.75

        over_2048 = sum(1 for w in word_counts if w / 0.75 > 2048)

        print(f"   Estimated avg tokens: {avg_tokens:.0f}")
        print(f"   Estimated max tokens: {max_tokens:.0f}")
        print(f"   Estimated >2048: {over_2048} ({over_2048/len(data)*100:.1f}%)")

    # Category balance
    print("\n2. Category Balance:")

    if "task_category" in data[0]:
        categories = Counter(ex.get("task_category", "unknown") for ex in data)

        for cat, count in categories.most_common():
            pct = count / len(data) * 100
            bar = "#" * int(pct / 2)
            print(f"   {cat:30s}: {count:5d} ({pct:5.1f}%) {bar}")

        # Chi-square test for balance
        expected = len(data) / len(categories)
        chi_sq = sum((count - expected)**2 / expected for count in categories.values())

        # Degrees of freedom = k - 1
        dof = len(categories) - 1
        critical_value_95 = {
            1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07,
            6: 12.59, 7: 14.07, 8: 15.51, 9: 16.92, 10: 18.31
        }.get(dof, 20.0)

        print(f"\n   Chi-square: {chi_sq:.2f} (critical value @95%: {critical_value_95:.2f})")

        if chi_sq < critical_value_95:
            print(f"   Result: BALANCED (uniform distribution)")
            balanced = True
        else:
            print(f"   Result: IMBALANCED (non-uniform distribution)")
            balanced = False
    else:
        print("   WARNING: No task_category field found")
        categories = {}
        balanced = False

    # Difficulty distribution
    print("\n3. Difficulty Distribution:")

    difficulties = Counter(ex.get("difficulty", "unknown") for ex in data)
    for diff, count in difficulties.most_common():
        pct = count / len(data) * 100
        bar = "#" * int(pct / 2)
        print(f"   {diff:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Attack coverage matrix
    print("\n4. Attack Coverage (Category x Type):")

    if "task_category" in data[0]:
        attack_types = set(ex.get("attack_type", "unknown") for ex in data)
        coverage = {}

        for ex in data:
            cat = ex.get("task_category", "unknown")
            atype = ex.get("attack_type", "unknown")
            key = (cat, atype)
            coverage[key] = coverage.get(key, 0) + 1

        # Find gaps (< 10 examples)
        gaps = [(cat, atype, count) for (cat, atype), count in coverage.items() if count < 10]

        print(f"   Total cells: {len(coverage)}")
        print(f"   Coverage gaps (<10 examples): {len(gaps)}")

        if gaps:
            print(f"   WARNING: Found {len(gaps)} coverage gaps")
            if len(gaps) <= 10:
                print("\n   Gaps:")
                for cat, atype, count in sorted(gaps, key=lambda x: x[2])[:10]:
                    print(f"     {cat} x {atype}: {count} examples")
        else:
            print(f"   Result: No coverage gaps")
    else:
        gaps = []

    # Quality indicators
    print("\n5. Quality Indicators:")

    # Check for empty fields
    empty_counts = {
        "system_instruction": 0,
        "user_input_benign_1": 0,
        "expected_output_1": 0,
        "user_input_injection": 0,
    }

    for ex in data:
        for field in empty_counts.keys():
            if not ex.get(field) or len(str(ex.get(field)).strip()) == 0:
                empty_counts[field] += 1

    total_empty = sum(empty_counts.values())
    print(f"   Empty fields: {total_empty}")
    if total_empty > 0:
        for field, count in empty_counts.items():
            if count > 0:
                print(f"     {field}: {count}")

    # Check output diversity (unique expected outputs)
    unique_outputs = len(set(ex.get("expected_output_1", "") for ex in data))
    output_diversity = unique_outputs / len(data)
    print(f"   Output diversity: {output_diversity:.2%} ({unique_outputs}/{len(data)} unique)")

    # Save statistics
    stats = {
        "total_examples": len(data),
        "avg_tokens": float(avg_tokens) if TRANSFORMERS_AVAILABLE or not TRANSFORMERS_AVAILABLE else 0,
        "max_tokens": int(max_tokens),
        "over_2048": over_2048,
        "categories": dict(categories) if categories else {},
        "balanced": balanced if "task_category" in data[0] else False,
        "difficulties": dict(difficulties),
        "coverage_gaps": len(gaps),
        "empty_fields": total_empty,
        "output_diversity": float(output_diversity)
    }

    stats_path = Path("C:/isef/data/processed/dataset_statistics_final.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"\n   Statistics saved to: {stats_path}")

    # Overall assessment
    print("\n" + "="*50)

    issues = []
    if not balanced and "task_category" in data[0]:
        issues.append("Imbalanced categories")
    if len(gaps) > len(coverage) * 0.1:  # >10% gaps
        issues.append(f"Too many coverage gaps ({len(gaps)})")
    if total_empty > len(data) * 0.01:  # >1% empty
        issues.append(f"Empty fields detected ({total_empty})")
    if output_diversity < 0.5:
        issues.append("Low output diversity")

    if not issues:
        print("RESULT: STATISTICS GOOD (PASS)")
        result = True
    elif len(issues) <= 2:
        print("RESULT: MINOR ISSUES (WARNING)")
        for issue in issues:
            print(f"  - {issue}")
        result = True
    else:
        print("RESULT: MAJOR ISSUES (FAIL)")
        for issue in issues:
            print(f"  - {issue}")
        result = False

    print("="*50)

    return result


if __name__ == "__main__":
    success = compute_statistics()
    sys.exit(0 if success else 1)
