#!/usr/bin/env python3
"""
Task 2: Counterfactual Quality Analysis
Analyzes the quality of counterfactual pairs (benign_1 vs benign_2).
Measures semantic similarity, lexical diversity, and intent preservation.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import math


def cosine_similarity(vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
    """Calculate cosine similarity between two word frequency vectors."""
    # Get all unique words
    all_words = set(vec1.keys()) | set(vec2.keys())

    if not all_words:
        return 0.0

    # Calculate dot product and magnitudes
    dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def tokenize_and_vectorize(text: str) -> Dict[str, int]:
    """Simple tokenization and word frequency vector."""
    # Lowercase and split by whitespace/punctuation
    words = re.findall(r'\b\w+\b', text.lower())

    # Count frequencies
    freq = defaultdict(int)
    for word in words:
        freq[word] += 1

    return dict(freq)


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity (lexical overlap) between two texts."""
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))

    if not words1 and not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity using word frequency vectors.
    This is a simplified version - in production, you'd use embeddings.
    """
    vec1 = tokenize_and_vectorize(text1)
    vec2 = tokenize_and_vectorize(text2)

    return cosine_similarity(vec1, vec2)


def compute_lexical_overlap(text1: str, text2: str) -> float:
    """Compute lexical overlap using Jaccard similarity."""
    return jaccard_similarity(text1, text2)


def assess_intent_preservation(example: Dict) -> Dict:
    """
    Assess whether the intent is preserved between benign_1 and benign_2.
    Returns a detailed assessment.
    """
    input1 = example['user_input_benign_1']
    input2 = example['user_input_benign_2']
    output1 = example['expected_output_1']
    output2 = example['expected_output_2']

    # Calculate similarities
    input_semantic = compute_semantic_similarity(input1, input2)
    input_lexical = compute_lexical_overlap(input1, input2)
    output_semantic = compute_semantic_similarity(output1, output2)
    output_lexical = compute_lexical_overlap(output1, output2)

    # Intent preservation heuristics:
    # 1. Inputs should be semantically similar but lexically diverse
    # 2. Outputs should reflect similar task completion

    # Quality score (0-100)
    # Good: high input semantic similarity (>0.5), low input lexical overlap (<0.6)
    # Good: output similarity should be moderate (task-dependent)

    quality_score = 0.0

    # Input semantic similarity (target: 0.5-0.9) - 40 points
    if input_semantic >= 0.7:
        quality_score += 40
    elif input_semantic >= 0.5:
        quality_score += 30
    elif input_semantic >= 0.3:
        quality_score += 20
    else:
        quality_score += 10

    # Input lexical diversity (target: overlap < 0.5) - 30 points
    if input_lexical < 0.3:
        quality_score += 30
    elif input_lexical < 0.5:
        quality_score += 20
    elif input_lexical < 0.7:
        quality_score += 10
    else:
        quality_score += 0

    # Output consistency (some similarity expected) - 30 points
    if 0.3 <= output_semantic <= 0.8:
        quality_score += 30
    elif 0.2 <= output_semantic <= 0.9:
        quality_score += 20
    else:
        quality_score += 10

    return {
        'input_semantic_similarity': input_semantic,
        'input_lexical_overlap': input_lexical,
        'output_semantic_similarity': output_semantic,
        'output_lexical_overlap': output_lexical,
        'quality_score': quality_score,
        'quality_rating': 'excellent' if quality_score >= 80 else 'good' if quality_score >= 60 else 'fair' if quality_score >= 40 else 'poor'
    }


def load_random_sample(data_dir: Path, sample_size: int = 500) -> List[Dict]:
    """Load random sample from all splits."""
    print(f"Loading random sample of {sample_size} examples...")

    all_examples = []

    # Load from all splits
    for split in ['train_split.jsonl', 'val_split.jsonl', 'test_split.jsonl']:
        file_path = data_dir / split
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_examples.append(json.loads(line.strip()))

    print(f"Total examples loaded: {len(all_examples)}")

    # Random sample
    if len(all_examples) > sample_size:
        sample = random.sample(all_examples, sample_size)
    else:
        sample = all_examples

    print(f"Sample size: {len(sample)}")
    return sample


def analyze_counterfactuals(sample: List[Dict]) -> Dict:
    """Analyze counterfactual quality for all samples."""
    print(f"\nAnalyzing counterfactual quality for {len(sample)} examples...")

    results = []
    low_quality_pairs = []

    for idx, example in enumerate(sample):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(sample)} examples...")

        assessment = assess_intent_preservation(example)
        assessment['id'] = example['id']
        assessment['task_category'] = example['task_category']

        results.append(assessment)

        # Flag low-quality pairs
        if (assessment['input_semantic_similarity'] < 0.5 or
            assessment['input_lexical_overlap'] > 0.8 or
            assessment['quality_score'] < 40):
            low_quality_pairs.append({
                'id': example['id'],
                'task_category': example['task_category'],
                'input_semantic_sim': assessment['input_semantic_similarity'],
                'input_lexical_overlap': assessment['input_lexical_overlap'],
                'quality_score': assessment['quality_score'],
                'benign_1': example['user_input_benign_1'],
                'benign_2': example['user_input_benign_2']
            })

    # Calculate aggregate statistics
    avg_input_semantic = sum(r['input_semantic_similarity'] for r in results) / len(results)
    avg_input_lexical = sum(r['input_lexical_overlap'] for r in results) / len(results)
    avg_output_semantic = sum(r['output_semantic_similarity'] for r in results) / len(results)
    avg_quality_score = sum(r['quality_score'] for r in results) / len(results)

    # Quality distribution
    quality_distribution = defaultdict(int)
    for r in results:
        quality_distribution[r['quality_rating']] += 1

    # Category breakdown
    category_stats = defaultdict(lambda: {
        'count': 0,
        'avg_semantic': 0.0,
        'avg_lexical': 0.0,
        'avg_quality': 0.0
    })

    for r in results:
        cat = r['task_category']
        category_stats[cat]['count'] += 1
        category_stats[cat]['avg_semantic'] += r['input_semantic_similarity']
        category_stats[cat]['avg_lexical'] += r['input_lexical_overlap']
        category_stats[cat]['avg_quality'] += r['quality_score']

    # Calculate averages per category
    for cat in category_stats:
        count = category_stats[cat]['count']
        category_stats[cat]['avg_semantic'] /= count
        category_stats[cat]['avg_lexical'] /= count
        category_stats[cat]['avg_quality'] /= count

    return {
        'sample_size': len(sample),
        'average_input_semantic_similarity': avg_input_semantic,
        'average_input_lexical_overlap': avg_input_lexical,
        'average_output_semantic_similarity': avg_output_semantic,
        'average_quality_score': avg_quality_score,
        'quality_distribution': dict(quality_distribution),
        'category_breakdown': dict(category_stats),
        'low_quality_pairs': low_quality_pairs,
        'low_quality_count': len(low_quality_pairs),
        'detailed_results': results
    }


def print_report(analysis: Dict):
    """Print formatted analysis report."""
    print("\n" + "="*80)
    print("COUNTERFACTUAL QUALITY ANALYSIS REPORT")
    print("="*80)

    print(f"\nSample Size: {analysis['sample_size']} examples")

    print(f"\n{'='*80}")
    print("AGGREGATE METRICS")
    print(f"{'='*80}")

    print(f"\nInput Semantic Similarity: {analysis['average_input_semantic_similarity']:.3f}")
    print(f"  Target: >0.70 (high semantic similarity)")
    if analysis['average_input_semantic_similarity'] >= 0.70:
        print(f"  ✓ PASS - Counterfactuals maintain semantic meaning")
    elif analysis['average_input_semantic_similarity'] >= 0.50:
        print(f"  ⚠ MARGINAL - Some semantic drift observed")
    else:
        print(f"  ✗ FAIL - Significant semantic divergence")

    print(f"\nInput Lexical Overlap: {analysis['average_input_lexical_overlap']:.3f}")
    print(f"  Target: <0.50 (high lexical diversity)")
    if analysis['average_input_lexical_overlap'] < 0.50:
        print(f"  ✓ PASS - Good lexical diversity")
    elif analysis['average_input_lexical_overlap'] < 0.70:
        print(f"  ⚠ MARGINAL - Moderate diversity")
    else:
        print(f"  ✗ FAIL - Low diversity (too similar)")

    print(f"\nOutput Semantic Similarity: {analysis['average_output_semantic_similarity']:.3f}")
    print(f"  (Outputs should show task consistency)")

    print(f"\nOverall Quality Score: {analysis['average_quality_score']:.1f}/100")
    if analysis['average_quality_score'] >= 70:
        print(f"  ✓ EXCELLENT - High-quality counterfactuals")
    elif analysis['average_quality_score'] >= 50:
        print(f"  ✓ GOOD - Acceptable counterfactual quality")
    elif analysis['average_quality_score'] >= 30:
        print(f"  ⚠ FAIR - Quality concerns detected")
    else:
        print(f"  ✗ POOR - Significant quality issues")

    print(f"\n{'='*80}")
    print("QUALITY DISTRIBUTION")
    print(f"{'='*80}")
    for rating in ['excellent', 'good', 'fair', 'poor']:
        count = analysis['quality_distribution'].get(rating, 0)
        pct = (count / analysis['sample_size']) * 100
        print(f"  {rating.capitalize()}: {count} ({pct:.1f}%)")

    print(f"\n{'='*80}")
    print("CATEGORY BREAKDOWN")
    print(f"{'='*80}")
    for cat, stats in analysis['category_breakdown'].items():
        print(f"\n{cat}:")
        print(f"  Sample count: {stats['count']}")
        print(f"  Avg semantic similarity: {stats['avg_semantic']:.3f}")
        print(f"  Avg lexical overlap: {stats['avg_lexical']:.3f}")
        print(f"  Avg quality score: {stats['avg_quality']:.1f}/100")

    print(f"\n{'='*80}")
    print("LOW-QUALITY PAIRS")
    print(f"{'='*80}")
    print(f"\nTotal low-quality pairs: {analysis['low_quality_count']} ({(analysis['low_quality_count']/analysis['sample_size'])*100:.1f}%)")

    if analysis['low_quality_pairs']:
        print(f"\nShowing first 10 low-quality pairs:")
        for idx, pair in enumerate(analysis['low_quality_pairs'][:10], 1):
            print(f"\n{idx}. ID: {pair['id']} ({pair['task_category']})")
            print(f"   Semantic Sim: {pair['input_semantic_sim']:.3f}, Lexical Overlap: {pair['input_lexical_overlap']:.3f}")
            print(f"   Quality Score: {pair['quality_score']:.1f}/100")
            print(f"   Benign 1: {pair['benign_1'][:80]}...")
            print(f"   Benign 2: {pair['benign_2'][:80]}...")

        if analysis['low_quality_count'] > 10:
            print(f"\n   ... and {analysis['low_quality_count'] - 10} more low-quality pairs")

    print(f"\n{'='*80}")
    print("FINAL ASSESSMENT")
    print(f"{'='*80}")

    if (analysis['average_input_semantic_similarity'] >= 0.70 and
        analysis['average_input_lexical_overlap'] < 0.50 and
        analysis['average_quality_score'] >= 70):
        print("✓ STATUS: PASS")
        print("  Counterfactuals are high quality")
        print("  Semantic meaning preserved")
        print("  Lexical diversity achieved")
        status = "PASS"
    elif (analysis['average_input_semantic_similarity'] >= 0.50 and
          analysis['average_quality_score'] >= 50):
        print("⚠ STATUS: MARGINAL PASS")
        print("  Counterfactuals are acceptable")
        print("  Some quality concerns noted")
        print("  Consider review of low-quality pairs")
        status = "MARGINAL"
    else:
        print("✗ STATUS: FAIL")
        print("  Counterfactual quality below threshold")
        print("  Significant issues detected")
        print("  Manual review and regeneration recommended")
        status = "FAIL"

    print(f"{'='*80}\n")

    return status


def main():
    """Main execution function."""
    print("="*80)
    print("TASK 2: COUNTERFACTUAL QUALITY ANALYSIS")
    print("="*80)

    # Set random seed for reproducibility
    random.seed(42)

    # Load data
    data_dir = Path('C:/isef/data/processed')
    sample = load_random_sample(data_dir, sample_size=500)

    # Analyze counterfactuals
    analysis = analyze_counterfactuals(sample)

    # Print report
    status = print_report(analysis)

    # Save results
    output_file = data_dir / 'counterfactual_analysis.json'

    # Remove detailed results for cleaner output (too large)
    output_data = {k: v for k, v in analysis.items() if k != 'detailed_results'}

    # Add summary stats for detailed results
    output_data['detailed_results_summary'] = {
        'count': len(analysis['detailed_results']),
        'saved': 'Results too large to save in full JSON'
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")

    return status


if __name__ == "__main__":
    status = main()
    exit(0 if status == "PASS" else 1)
