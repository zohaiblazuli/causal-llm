#!/usr/bin/env python3
"""
Task 3: Attack Diversity Validation
Analyzes attack type and technique distribution, entropy, and distinctness.
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter


def calculate_shannon_entropy(distribution: Dict[str, int]) -> float:
    """Calculate Shannon entropy for a distribution."""
    total = sum(distribution.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def calculate_distinctness(text1: str, text2: str) -> float:
    """
    Calculate distinctness (inverse similarity) between two texts.
    Returns value between 0 (identical) and 1 (completely different).
    """
    # Simple word-based Jaccard distance
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    jaccard_sim = intersection / union if union > 0 else 0.0
    return 1 - jaccard_sim  # Distinctness is inverse of similarity


def load_all_examples(data_dir: Path) -> List[Dict]:
    """Load all examples from all splits."""
    print("Loading all examples...")

    all_examples = []

    for split in ['train_split.jsonl', 'val_split.jsonl', 'test_split.jsonl']:
        file_path = data_dir / split
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_examples.append(json.loads(line.strip()))

    print(f"Total examples loaded: {len(all_examples)}")
    return all_examples


def analyze_attack_distribution(examples: List[Dict]) -> Dict:
    """Analyze attack type and technique distributions."""
    print("\nAnalyzing attack distributions...")

    # Count attack types and techniques
    attack_types = Counter()
    attack_techniques = Counter()

    # Category x attack type matrix
    category_attack_matrix = defaultdict(lambda: defaultdict(int))

    # Collect all attack types and techniques
    all_attack_types = set()
    all_attack_techniques = set()

    for example in examples:
        attack_type = example.get('attack_type', 'unknown')
        attack_technique = example.get('attack_technique', 'unknown')
        category = example.get('task_category', 'unknown')

        attack_types[attack_type] += 1
        attack_techniques[attack_technique] += 1
        category_attack_matrix[category][attack_type] += 1

        all_attack_types.add(attack_type)
        all_attack_techniques.add(attack_technique)

    # Calculate entropies
    attack_type_entropy = calculate_shannon_entropy(dict(attack_types))
    attack_technique_entropy = calculate_shannon_entropy(dict(attack_techniques))

    # Calculate coverage
    expected_attack_types = 9  # Based on schema
    expected_attack_techniques = 15  # Based on schema

    attack_type_coverage = len(attack_types)
    attack_technique_coverage = len(attack_techniques)

    return {
        'attack_types': dict(attack_types),
        'attack_techniques': dict(attack_techniques),
        'attack_type_entropy': attack_type_entropy,
        'attack_technique_entropy': attack_technique_entropy,
        'attack_type_coverage': attack_type_coverage,
        'expected_attack_types': expected_attack_types,
        'attack_technique_coverage': attack_technique_coverage,
        'expected_attack_techniques': expected_attack_techniques,
        'category_attack_matrix': {cat: dict(attacks) for cat, attacks in category_attack_matrix.items()},
        'total_examples': len(examples)
    }


def analyze_attack_distinctness(examples: List[Dict], sample_size: int = 500) -> Dict:
    """Analyze how distinct injection attacks are from benign inputs."""
    print(f"\nAnalyzing attack distinctness (sampling {sample_size} examples)...")

    # Random sample
    if len(examples) > sample_size:
        sample = random.sample(examples, sample_size)
    else:
        sample = examples

    distinctness_scores = []
    low_distinctness_examples = []

    for example in sample:
        injection = example.get('user_input_injection', '')
        benign_1 = example.get('user_input_benign_1', '')
        benign_2 = example.get('user_input_benign_2', '')

        # Calculate distinctness between injection and both benign inputs
        dist1 = calculate_distinctness(injection, benign_1)
        dist2 = calculate_distinctness(injection, benign_2)

        # Average distinctness
        avg_distinctness = (dist1 + dist2) / 2
        distinctness_scores.append(avg_distinctness)

        # Flag low distinctness (similarity > 0.7, distinctness < 0.3)
        if avg_distinctness < 0.3:
            low_distinctness_examples.append({
                'id': example['id'],
                'distinctness': avg_distinctness,
                'injection': injection,
                'benign_1': benign_1,
                'attack_type': example.get('attack_type', 'unknown')
            })

    avg_distinctness = sum(distinctness_scores) / len(distinctness_scores) if distinctness_scores else 0.0

    # Distinctness distribution
    dist_ranges = {
        'very_high (>0.8)': sum(1 for d in distinctness_scores if d > 0.8),
        'high (0.6-0.8)': sum(1 for d in distinctness_scores if 0.6 <= d <= 0.8),
        'moderate (0.4-0.6)': sum(1 for d in distinctness_scores if 0.4 <= d < 0.6),
        'low (0.2-0.4)': sum(1 for d in distinctness_scores if 0.2 <= d < 0.4),
        'very_low (<0.2)': sum(1 for d in distinctness_scores if d < 0.2)
    }

    return {
        'sample_size': len(sample),
        'average_distinctness': avg_distinctness,
        'distinctness_distribution': dist_ranges,
        'low_distinctness_count': len(low_distinctness_examples),
        'low_distinctness_examples': low_distinctness_examples[:10]  # First 10 only
    }


def generate_histograms(distribution_data: Dict) -> List[str]:
    """Generate ASCII histograms for distributions."""
    histograms = []

    # Attack type distribution
    attack_types = distribution_data['attack_types']
    if attack_types:
        max_count = max(attack_types.values())
        bar_width = 50

        histogram = ["\nATTACK TYPE DISTRIBUTION:"]
        histogram.append("-" * 70)

        for attack_type, count in sorted(attack_types.items(), key=lambda x: x[1], reverse=True):
            bar_length = int((count / max_count) * bar_width)
            bar = '█' * bar_length
            pct = (count / distribution_data['total_examples']) * 100
            histogram.append(f"{attack_type:25s} │{bar:50s}│ {count:5d} ({pct:5.1f}%)")

        histograms.append('\n'.join(histogram))

    # Attack technique distribution
    attack_techniques = distribution_data['attack_techniques']
    if attack_techniques:
        max_count = max(attack_techniques.values())
        bar_width = 50

        histogram = ["\n\nATTACK TECHNIQUE DISTRIBUTION:"]
        histogram.append("-" * 70)

        for technique, count in sorted(attack_techniques.items(), key=lambda x: x[1], reverse=True):
            bar_length = int((count / max_count) * bar_width)
            bar = '█' * bar_length
            pct = (count / distribution_data['total_examples']) * 100
            histogram.append(f"{technique:25s} │{bar:50s}│ {count:5d} ({pct:5.1f}%)")

        histograms.append('\n'.join(histogram))

    return histograms


def print_report(distribution_data: Dict, distinctness_data: Dict):
    """Print formatted analysis report."""
    print("\n" + "="*80)
    print("ATTACK DIVERSITY VALIDATION REPORT")
    print("="*80)

    print(f"\nTotal examples analyzed: {distribution_data['total_examples']}")

    print(f"\n{'='*80}")
    print("ATTACK TYPE ANALYSIS")
    print(f"{'='*80}")

    print(f"\nAttack Type Coverage: {distribution_data['attack_type_coverage']}/{distribution_data['expected_attack_types']}")
    if distribution_data['attack_type_coverage'] == distribution_data['expected_attack_types']:
        print(f"  ✓ PASS - All attack types present")
    else:
        missing = distribution_data['expected_attack_types'] - distribution_data['attack_type_coverage']
        print(f"  ⚠ WARNING - {missing} attack types missing")

    print(f"\nShannon Entropy: {distribution_data['attack_type_entropy']:.3f}")
    print(f"  Target: >2.5 (high diversity)")
    if distribution_data['attack_type_entropy'] >= 2.5:
        print(f"  ✓ PASS - Excellent diversity")
    elif distribution_data['attack_type_entropy'] >= 2.0:
        print(f"  ⚠ MARGINAL - Acceptable diversity")
    else:
        print(f"  ✗ FAIL - Low diversity (distribution too skewed)")

    print(f"\nAttack Types ({len(distribution_data['attack_types'])}):")
    for attack_type, count in sorted(distribution_data['attack_types'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / distribution_data['total_examples']) * 100
        print(f"  {attack_type:30s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n{'='*80}")
    print("ATTACK TECHNIQUE ANALYSIS")
    print(f"{'='*80}")

    print(f"\nAttack Technique Coverage: {distribution_data['attack_technique_coverage']}/{distribution_data['expected_attack_techniques']}")
    coverage_pct = (distribution_data['attack_technique_coverage'] / distribution_data['expected_attack_techniques']) * 100
    print(f"  Coverage: {coverage_pct:.1f}%")

    if distribution_data['attack_technique_coverage'] == distribution_data['expected_attack_techniques']:
        print(f"  ✓ PASS - All techniques present")
    elif coverage_pct >= 80:
        print(f"  ⚠ MARGINAL - Most techniques present")
    else:
        print(f"  ✗ FAIL - Significant techniques missing")

    print(f"\nShannon Entropy: {distribution_data['attack_technique_entropy']:.3f}")
    print(f"  Target: >2.5 (high diversity)")
    if distribution_data['attack_technique_entropy'] >= 2.5:
        print(f"  ✓ PASS - Excellent diversity")
    elif distribution_data['attack_technique_entropy'] >= 2.0:
        print(f"  ⚠ MARGINAL - Acceptable diversity")
    else:
        print(f"  ✗ FAIL - Low diversity")

    print(f"\nAttack Techniques ({len(distribution_data['attack_techniques'])}):")
    for technique, count in sorted(distribution_data['attack_techniques'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / distribution_data['total_examples']) * 100
        print(f"  {technique:30s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n{'='*80}")
    print("CATEGORY × ATTACK TYPE MATRIX")
    print(f"{'='*80}")

    print("\nChecking for gaps (cells with <10 examples)...")
    gap_cells = []
    for category, attacks in distribution_data['category_attack_matrix'].items():
        print(f"\n{category}:")
        all_attack_types = set(distribution_data['attack_types'].keys())
        present_attacks = set(attacks.keys())
        missing_attacks = all_attack_types - present_attacks

        for attack_type, count in sorted(attacks.items(), key=lambda x: x[1], reverse=True):
            indicator = "⚠" if count < 10 else " "
            print(f"  {indicator} {attack_type:30s}: {count:4d}")
            if count < 10:
                gap_cells.append(f"{category} × {attack_type}: {count}")

        if missing_attacks:
            for attack_type in missing_attacks:
                print(f"  ⚠ {attack_type:30s}: 0")
                gap_cells.append(f"{category} × {attack_type}: 0")

    if gap_cells:
        print(f"\n⚠ Found {len(gap_cells)} gaps:")
        for gap in gap_cells[:10]:
            print(f"    {gap}")
        if len(gap_cells) > 10:
            print(f"    ... and {len(gap_cells) - 10} more")
    else:
        print(f"\n✓ No gaps found - all cells have ≥10 examples")

    print(f"\n{'='*80}")
    print("ATTACK DISTINCTNESS")
    print(f"{'='*80}")

    print(f"\nSample Size: {distinctness_data['sample_size']}")
    print(f"Average Distinctness: {distinctness_data['average_distinctness']:.3f}")
    print(f"  Target: >0.7 (attacks very different from benign)")
    if distinctness_data['average_distinctness'] >= 0.7:
        print(f"  ✓ PASS - Attacks highly distinct from benign inputs")
    elif distinctness_data['average_distinctness'] >= 0.5:
        print(f"  ⚠ MARGINAL - Attacks moderately distinct")
    else:
        print(f"  ✗ FAIL - Attacks too similar to benign inputs")

    print(f"\nDistinctness Distribution:")
    for range_name, count in distinctness_data['distinctness_distribution'].items():
        pct = (count / distinctness_data['sample_size']) * 100
        print(f"  {range_name:20s}: {count:4d} ({pct:5.1f}%)")

    if distinctness_data['low_distinctness_examples']:
        print(f"\n⚠ Low Distinctness Examples ({distinctness_data['low_distinctness_count']}):")
        for idx, ex in enumerate(distinctness_data['low_distinctness_examples'], 1):
            print(f"\n  {idx}. ID: {ex['id']} ({ex['attack_type']})")
            print(f"     Distinctness: {ex['distinctness']:.3f}")
            print(f"     Injection: {ex['injection'][:80]}...")
            print(f"     Benign: {ex['benign_1'][:80]}...")

    # Print histograms
    histograms = generate_histograms(distribution_data)
    for histogram in histograms:
        print(histogram)

    print(f"\n{'='*80}")
    print("FINAL ASSESSMENT")
    print(f"{'='*80}")

    # Determine overall status
    checks = {
        'attack_type_coverage': distribution_data['attack_type_coverage'] == distribution_data['expected_attack_types'],
        'attack_type_entropy': distribution_data['attack_type_entropy'] >= 2.5,
        'attack_technique_coverage': distribution_data['attack_technique_coverage'] >= distribution_data['expected_attack_techniques'] * 0.8,
        'attack_technique_entropy': distribution_data['attack_technique_entropy'] >= 2.5,
        'distinctness': distinctness_data['average_distinctness'] >= 0.7,
        'no_major_gaps': len(gap_cells) < 10
    }

    passed = sum(checks.values())
    total = len(checks)

    print(f"\nChecks Passed: {passed}/{total}")
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")

    if passed == total:
        print(f"\n✓ STATUS: PASS")
        print("  Attack diversity is excellent")
        print("  All attack types and techniques well represented")
        print("  Attacks are highly distinct from benign inputs")
        status = "PASS"
    elif passed >= total * 0.7:
        print(f"\n⚠ STATUS: MARGINAL PASS")
        print("  Attack diversity is acceptable")
        print("  Some coverage or diversity concerns noted")
        print("  Dataset usable but could be improved")
        status = "MARGINAL"
    else:
        print(f"\n✗ STATUS: FAIL")
        print("  Attack diversity below threshold")
        print("  Significant issues with coverage or diversity")
        print("  Dataset augmentation recommended")
        status = "FAIL"

    print(f"{'='*80}\n")

    return status, gap_cells


def main():
    """Main execution function."""
    print("="*80)
    print("TASK 3: ATTACK DIVERSITY VALIDATION")
    print("="*80)

    # Set random seed for reproducibility
    random.seed(42)

    # Load data
    data_dir = Path('C:/isef/data/processed')
    examples = load_all_examples(data_dir)

    # Analyze distributions
    distribution_data = analyze_attack_distribution(examples)

    # Analyze distinctness
    distinctness_data = analyze_attack_distinctness(examples, sample_size=500)

    # Print report
    status, gap_cells = print_report(distribution_data, distinctness_data)

    # Save results
    output_file = data_dir / 'attack_diversity_analysis.json'

    output_data = {
        'distribution': distribution_data,
        'distinctness': distinctness_data,
        'gap_cells': gap_cells,
        'final_status': status
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")

    return status


if __name__ == "__main__":
    status = main()
    exit(0 if status == "PASS" else 1)
