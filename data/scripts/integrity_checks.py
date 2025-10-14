"""
Integrity Checks - Week 1 Phase 2 Validation
Runs comprehensive integrity checks to ensure dataset quality and consistency.
"""

import json
import sys
from pathlib import Path
from collections import Counter


def load_jsonl(filepath):
    """Load JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def integrity_checks():
    """Run integrity checks on dataset."""
    print("=== Data Integrity Checks ===\n")

    base_dir = Path("C:/isef")

    # Load all splits
    train_path = base_dir / "data/processed/train_split.jsonl"
    val_path = base_dir / "data/processed/val_split.jsonl"
    test_path = base_dir / "data/processed/test_split.jsonl"

    missing_files = []
    if not train_path.exists():
        missing_files.append("train_split.jsonl")
    if not val_path.exists():
        missing_files.append("val_split.jsonl")
    if not test_path.exists():
        missing_files.append("test_split.jsonl")

    if missing_files:
        print(f"ERROR: Missing files: {', '.join(missing_files)}")
        return False

    print("Loading splits...")
    train = load_jsonl(train_path)
    val = load_jsonl(val_path)
    test = load_jsonl(test_path)

    all_data = train + val + test

    print(f"  Train: {len(train)} examples")
    print(f"  Val: {len(val)} examples")
    print(f"  Test: {len(test)} examples")
    print(f"  Total: {len(all_data)} examples\n")

    issues = []

    # Check 1: No data leakage
    print("1. Checking for data leakage...")

    train_ids = set(ex.get("id", f"train_{i}") for i, ex in enumerate(train))
    val_ids = set(ex.get("id", f"val_{i}") for i, ex in enumerate(val))
    test_ids = set(ex.get("id", f"test_{i}") for i, ex in enumerate(test))

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    total_overlap = len(train_val_overlap | train_test_overlap | val_test_overlap)

    if total_overlap > 0:
        print(f"   X LEAKAGE FOUND: {total_overlap} overlapping IDs")
        if train_val_overlap:
            print(f"     Train-Val overlap: {len(train_val_overlap)}")
        if train_test_overlap:
            print(f"     Train-Test overlap: {len(train_test_overlap)}")
        if val_test_overlap:
            print(f"     Val-Test overlap: {len(val_test_overlap)}")
        issues.append("Data leakage detected")
    else:
        print(f"   PASS: No leakage (all IDs unique across splits)")

    # Check 2: No duplicates within splits
    print("\n2. Checking for duplicates within splits...")

    train_dups = len(train) - len(train_ids)
    val_dups = len(val) - len(val_ids)
    test_dups = len(test) - len(test_ids)
    total_dups = train_dups + val_dups + test_dups

    if total_dups > 0:
        print(f"   X Found {total_dups} duplicate IDs")
        if train_dups > 0:
            print(f"     Train: {train_dups} duplicates")
        if val_dups > 0:
            print(f"     Val: {val_dups} duplicates")
        if test_dups > 0:
            print(f"     Test: {test_dups} duplicates")
        issues.append("Duplicate IDs found")
    else:
        print(f"   PASS: No duplicates")

    # Check 3: Consistent formatting
    print("\n3. Checking formatting consistency...")

    required_fields = [
        "system_instruction",
        "user_input_benign_1", "expected_output_1",
        "user_input_benign_2", "expected_output_2",
        "user_input_injection", "expected_output_injection",
        "attack_type", "attack_technique"
    ]

    schema_violations = 0
    field_violations = Counter()

    for ex in all_data:
        for field in required_fields:
            value = ex.get(field)
            if value is None:
                schema_violations += 1
                field_violations[field] += 1
            elif not isinstance(value, str):
                schema_violations += 1
                field_violations[field + "_type"] += 1
            elif len(str(value).strip()) == 0:
                schema_violations += 1
                field_violations[field + "_empty"] += 1

    if schema_violations > 0:
        print(f"   WARNING: {schema_violations} schema violations")
        if field_violations:
            print(f"   Most common violations:")
            for field, count in field_violations.most_common(5):
                print(f"     {field}: {count}")
        if schema_violations > len(all_data) * 0.05:  # >5%
            issues.append("Major schema violations")
    else:
        print(f"   PASS: All examples follow schema")

    # Check 4: Text quality
    print("\n4. Checking text quality...")

    encoding_issues = 0
    too_long = 0
    too_short = 0

    for ex in all_data:
        # Check encoding
        for field in required_fields:
            try:
                str(ex.get(field, "")).encode('utf-8')
            except:
                encoding_issues += 1
                break

        # Check length of system instruction
        sys_instr = ex.get("system_instruction", "")
        if len(sys_instr) > 5000:
            too_long += 1
        elif len(sys_instr) < 10:
            too_short += 1

    if encoding_issues > 0:
        print(f"   X {encoding_issues} encoding issues")
        issues.append("Encoding issues")
    if too_long > 0:
        print(f"   WARNING: {too_long} excessively long strings (>5000 chars)")
    if too_short > 0:
        print(f"   WARNING: {too_short} very short system instructions (<10 chars)")

    if encoding_issues == 0 and too_long == 0 and too_short == 0:
        print(f"   PASS: Text quality good")

    # Check 5: Injection quality
    print("\n5. Checking injection quality...")

    # Common attack indicators
    attack_keywords = [
        "ignore", "disregard", "system", "override", "forget",
        "instruction", "prompt", "instead", "new", "task",
        "sql", "script", "exec", "eval", "delete", "drop"
    ]

    weak_injections = 0
    injection_stats = Counter()

    for ex in all_data:
        injection = ex.get("user_input_injection", "").lower()

        # Count keyword matches
        matches = [kw for kw in attack_keywords if kw in injection]
        injection_stats[len(matches)] += 1

        # Flag if no keywords found
        if not matches:
            weak_injections += 1

    weak_pct = weak_injections / len(all_data) * 100 if all_data else 0

    print(f"   Weak injections (no keywords): {weak_injections} ({weak_pct:.1f}%)")

    # Show distribution of keyword matches
    print(f"   Keyword match distribution:")
    for num_matches in sorted(injection_stats.keys()):
        count = injection_stats[num_matches]
        pct = count / len(all_data) * 100
        print(f"     {num_matches} keywords: {count} ({pct:.1f}%)")

    if weak_pct > 10:  # >10% weak
        print(f"   WARNING: High proportion of weak injections")
        issues.append("Weak injection quality")
    else:
        print(f"   PASS: Injection quality good")

    # Check 6: Counterfactual consistency
    print("\n6. Checking counterfactual consistency...")

    inconsistent = 0
    identical_pairs = 0

    for ex in all_data:
        benign_1 = ex.get("user_input_benign_1", "")
        benign_2 = ex.get("user_input_benign_2", "")

        # Check if benign pairs are identical
        if benign_1 == benign_2:
            identical_pairs += 1

        # Check if outputs match (should be "I cannot assist")
        output_1 = ex.get("expected_output_1", "")
        output_2 = ex.get("expected_output_2", "")
        output_inj = ex.get("expected_output_injection", "")

        # Both benign outputs should be similar
        if "cannot" not in output_1.lower() and "cannot" not in output_2.lower():
            # Both might be compliant (which is also valid)
            pass

        # Injection output should contain refusal
        if "cannot" not in output_inj.lower():
            inconsistent += 1

    identical_pct = identical_pairs / len(all_data) * 100 if all_data else 0
    inconsistent_pct = inconsistent / len(all_data) * 100 if all_data else 0

    if identical_pairs > 0:
        print(f"   WARNING: {identical_pairs} identical benign pairs ({identical_pct:.1f}%)")
        if identical_pct > 5:
            issues.append("Too many identical counterfactual pairs")

    if inconsistent > 0:
        print(f"   WARNING: {inconsistent} inconsistent outputs ({inconsistent_pct:.1f}%)")

    if identical_pairs == 0 and inconsistent == 0:
        print(f"   PASS: Counterfactuals consistent")

    # Check 7: Split distribution
    print("\n7. Checking split distribution...")

    total = len(all_data)
    train_pct = len(train) / total * 100
    val_pct = len(val) / total * 100
    test_pct = len(test) / total * 100

    print(f"   Train: {train_pct:.1f}% (target: 70-80%)")
    print(f"   Val: {val_pct:.1f}% (target: 10-15%)")
    print(f"   Test: {test_pct:.1f}% (target: 10-15%)")

    if not (60 <= train_pct <= 85 and 5 <= val_pct <= 20 and 5 <= test_pct <= 20):
        print(f"   WARNING: Split distribution outside target ranges")
        issues.append("Non-standard split distribution")
    else:
        print(f"   PASS: Split distribution good")

    # Final summary
    print("\n" + "="*50)

    if len(issues) == 0:
        print("RESULT: ALL INTEGRITY CHECKS PASSED (PASS)")
        result = True
    elif len(issues) <= 2:
        print(f"RESULT: MINOR ISSUES FOUND (WARNING)")
        print("\nIssues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        result = True
    else:
        print(f"RESULT: MAJOR ISSUES FOUND (FAIL)")
        print("\nIssues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        result = False

    print("="*50)

    # Save integrity report
    report_path = Path("C:/isef/data/processed/integrity_report.json")
    report = {
        "total_examples": len(all_data),
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test)
        },
        "data_leakage": total_overlap,
        "duplicates": total_dups,
        "schema_violations": schema_violations,
        "encoding_issues": encoding_issues,
        "weak_injections": weak_injections,
        "identical_pairs": identical_pairs,
        "issues": issues,
        "passed": result
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\nIntegrity report saved to: {report_path}")

    return result


if __name__ == "__main__":
    success = integrity_checks()
    sys.exit(0 if success else 1)
