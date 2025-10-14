"""
Advanced Dataset Analysis for ISEF 2026
Performs deeper quality checks including token statistics, similarity, and bias analysis
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import statistics
from typing import Dict, List, Tuple
import random

class AdvancedAnalyzer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.train_file = self.data_dir / "train_split.jsonl"

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using simple heuristic:
        - Roughly 0.75 tokens per word for English text
        - Account for punctuation and special characters
        """
        # Simple whitespace tokenization
        words = text.split()
        # Add punctuation and special chars
        punct_count = len(re.findall(r'[^\w\s]', text))
        # Estimate: words * 0.75 + punct * 0.3
        return int(len(words) * 0.75 + punct_count * 0.3)

    def analyze_token_statistics(self, examples: List[Dict]) -> Dict:
        """Comprehensive token statistics for training compatibility"""
        print("\n=== TOKEN STATISTICS ANALYSIS ===")

        stats = {
            "benign_1": [],
            "benign_2": [],
            "injection": [],
            "total_triplet": []
        }

        for ex in examples:
            b1_tokens = self.estimate_tokens(ex.get("user_input_benign_1", ""))
            b2_tokens = self.estimate_tokens(ex.get("user_input_benign_2", ""))
            inj_tokens = self.estimate_tokens(ex.get("user_input_injection", ""))

            stats["benign_1"].append(b1_tokens)
            stats["benign_2"].append(b2_tokens)
            stats["injection"].append(inj_tokens)
            stats["total_triplet"].append(b1_tokens + b2_tokens + inj_tokens)

        # Calculate statistics for each field
        results = {}
        for field, tokens in stats.items():
            results[field] = {
                "min": min(tokens),
                "max": max(tokens),
                "mean": round(statistics.mean(tokens), 2),
                "median": statistics.median(tokens),
                "p95": round(sorted(tokens)[int(len(tokens) * 0.95)], 2),
                "p99": round(sorted(tokens)[int(len(tokens) * 0.99)], 2),
                "stdev": round(statistics.stdev(tokens), 2) if len(tokens) > 1 else 0
            }

            # Count examples that would be truncated at different lengths
            results[field]["truncated_at_512"] = sum(1 for t in tokens if t > 512)
            results[field]["truncated_at_1024"] = sum(1 for t in tokens if t > 1024)
            results[field]["truncated_at_2048"] = sum(1 for t in tokens if t > 2048)

        # Print results
        print("\n  Token Statistics by Field:")
        for field in ["benign_1", "benign_2", "injection", "total_triplet"]:
            print(f"\n  {field}:")
            print(f"    Min/Max/Mean: {results[field]['min']} / {results[field]['max']} / {results[field]['mean']}")
            print(f"    Median/P95/P99: {results[field]['median']} / {results[field]['p95']} / {results[field]['p99']}")
            print(f"    Std Dev: {results[field]['stdev']}")
            print(f"    Truncated at 512: {results[field]['truncated_at_512']} ({results[field]['truncated_at_512']/len(examples)*100:.1f}%)")
            print(f"    Truncated at 1024: {results[field]['truncated_at_1024']} ({results[field]['truncated_at_1024']/len(examples)*100:.1f}%)")

        # Memory estimation
        print("\n  Memory Estimates (for batch_size=1, max_length=1024):")
        print(f"    Assuming 2 bytes per token (fp16)")
        print(f"    Per example: ~{2 * 1024 * 3 / 1024:.1f} KB (3 inputs)")
        print(f"    Batch of 8: ~{2 * 1024 * 3 * 8 / 1024:.1f} KB")
        print(f"    Full train set: ~{2 * 1024 * 3 * len(examples) / (1024*1024):.1f} MB")

        return results

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple word-overlap similarity (Jaccard similarity)
        Good enough for detecting similar/different examples
        """
        # Tokenize and lowercase
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def analyze_counterfactual_structure(self, examples: List[Dict], sample_size: int = 100) -> Dict:
        """Analyze if counterfactual triplets have proper structure"""
        print("\n=== COUNTERFACTUAL STRUCTURE ANALYSIS ===")

        sample = random.sample(examples, min(sample_size, len(examples)))

        similarities = {
            "benign1_benign2": [],
            "benign1_injection": [],
            "benign2_injection": []
        }

        poor_structure = []

        for ex in sample:
            b1 = ex.get("user_input_benign_1", "")
            b2 = ex.get("user_input_benign_2", "")
            inj = ex.get("user_input_injection", "")

            # Calculate pairwise similarities
            sim_b1_b2 = self.calculate_text_similarity(b1, b2)
            sim_b1_inj = self.calculate_text_similarity(b1, inj)
            sim_b2_inj = self.calculate_text_similarity(b2, inj)

            similarities["benign1_benign2"].append(sim_b1_b2)
            similarities["benign1_injection"].append(sim_b1_inj)
            similarities["benign2_injection"].append(sim_b2_inj)

            # Check for poor structure
            # Good: b1 and b2 should be different (low similarity)
            # Good: b1/b2 should be very different from injection (low similarity)
            if sim_b1_b2 > 0.8:  # Too similar
                poor_structure.append((ex.get("id"), "benign_1 and benign_2 too similar", sim_b1_b2))
            if sim_b1_inj > 0.6 or sim_b2_inj > 0.6:  # Injection too similar to benign
                poor_structure.append((ex.get("id"), "injection too similar to benign", max(sim_b1_inj, sim_b2_inj)))

        # Calculate statistics
        results = {}
        for pair, sims in similarities.items():
            results[pair] = {
                "mean": round(statistics.mean(sims), 3),
                "median": round(statistics.median(sims), 3),
                "min": round(min(sims), 3),
                "max": round(max(sims), 3),
                "stdev": round(statistics.stdev(sims), 3) if len(sims) > 1 else 0
            }

        print(f"\n  Similarity Statistics (sample: {len(sample)}):")
        print(f"\n    Benign-1 vs Benign-2:")
        print(f"      Mean/Median: {results['benign1_benign2']['mean']} / {results['benign1_benign2']['median']}")
        print(f"      Range: {results['benign1_benign2']['min']} - {results['benign1_benign2']['max']}")
        print(f"      Note: Should be LOW (benign examples should be different)")

        print(f"\n    Benign-1 vs Injection:")
        print(f"      Mean/Median: {results['benign1_injection']['mean']} / {results['benign1_injection']['median']}")
        print(f"      Range: {results['benign1_injection']['min']} - {results['benign1_injection']['max']}")
        print(f"      Note: Should be LOW (injection should be distinct)")

        print(f"\n    Benign-2 vs Injection:")
        print(f"      Mean/Median: {results['benign2_injection']['mean']} / {results['benign2_injection']['median']}")
        print(f"      Range: {results['benign2_injection']['min']} - {results['benign2_injection']['max']}")
        print(f"      Note: Should be LOW (injection should be distinct)")

        print(f"\n  Poor Structure Examples: {len(poor_structure)}")
        if poor_structure:
            print(f"    Sample issues:")
            for ex_id, issue, score in poor_structure[:5]:
                print(f"      {ex_id}: {issue} (similarity: {score:.3f})")

        return results

    def analyze_bias_correlations(self, examples: List[Dict]) -> Dict:
        """Check for potential biases in the dataset"""
        print("\n=== BIAS ANALYSIS ===")

        # 1. Length vs Difficulty correlation
        print("\n  1. Length vs Difficulty Correlation:")

        length_by_difficulty = defaultdict(list)
        for ex in examples:
            difficulty = ex.get("difficulty")
            inj_length = self.estimate_tokens(ex.get("user_input_injection", ""))
            length_by_difficulty[difficulty].append(inj_length)

        for diff in ["trivial", "easy", "medium", "hard"]:
            if diff in length_by_difficulty:
                lengths = length_by_difficulty[diff]
                avg_len = statistics.mean(lengths)
                print(f"    {diff}: avg {avg_len:.1f} tokens")

        # 2. Category vs Attack Type correlation
        print("\n  2. Category vs Attack Type Distribution:")

        category_attacks = defaultdict(Counter)
        for ex in examples:
            category = ex.get("task_category")
            attack = ex.get("attack_type")
            category_attacks[category][attack] += 1

        # Check if certain attacks are overrepresented in certain categories
        total_by_category = Counter(ex.get("task_category") for ex in examples)

        for category in sorted(category_attacks.keys()):
            print(f"\n    {category}:")
            total = total_by_category[category]
            top_attacks = category_attacks[category].most_common(3)
            for attack, count in top_attacks:
                pct = (count / total) * 100
                print(f"      {attack}: {count} ({pct:.1f}%)")

        # 3. Difficulty distribution by category
        print("\n  3. Difficulty Distribution by Category:")

        category_difficulty = defaultdict(Counter)
        for ex in examples:
            category = ex.get("task_category")
            difficulty = ex.get("difficulty")
            category_difficulty[category][difficulty] += 1

        for category in sorted(category_difficulty.keys()):
            print(f"\n    {category}:")
            total = total_by_category[category]
            for diff in ["trivial", "easy", "medium", "hard"]:
                count = category_difficulty[category][diff]
                pct = (count / total) * 100 if total > 0 else 0
                print(f"      {diff}: {count} ({pct:.1f}%)")

        return {
            "length_by_difficulty": length_by_difficulty,
            "category_attacks": category_attacks,
            "category_difficulty": category_difficulty
        }

    def check_injection_quality(self, examples: List[Dict], sample_size: int = 50) -> Dict:
        """Manually check if injections look realistic"""
        print("\n=== INJECTION QUALITY CHECK ===")

        sample = random.sample(examples, min(sample_size, len(examples)))

        # Pattern detection for common attack markers
        attack_patterns = {
            "ignore_instructions": [r"ignore", r"disregard", r"forget"],
            "role_change": [r"you are now", r"act as", r"pretend to be", r"switch to"],
            "privilege_words": [r"admin", r"root", r"sudo", r"developer mode", r"unrestricted"],
            "encoding_tricks": [r"\\x", r"\\u", r"&#", r"base64"],
            "delimiter_abuse": [r"---", r"###", r"```", r"<|", r"|>"],
        }

        pattern_counts = Counter()
        examples_with_patterns = defaultdict(list)

        for ex in sample:
            injection = ex.get("user_input_injection", "").lower()
            ex_id = ex.get("id")

            for pattern_type, regexes in attack_patterns.items():
                for regex in regexes:
                    if re.search(regex, injection, re.IGNORECASE):
                        pattern_counts[pattern_type] += 1
                        examples_with_patterns[pattern_type].append(ex_id)
                        break  # Count each pattern type once per example

        print(f"\n  Attack Pattern Distribution (sample: {len(sample)}):")
        for pattern, count in pattern_counts.most_common():
            pct = (count / len(sample)) * 100
            print(f"    {pattern}: {count} ({pct:.1f}%)")

        # Check for very short injections (might be low quality)
        short_injections = [ex for ex in sample if len(ex.get("user_input_injection", "").split()) < 5]
        print(f"\n  Very short injections (<5 words): {len(short_injections)}")

        # Check for very generic injections
        generic_phrases = ["ignore previous", "ignore all", "you are now", "disregard"]
        generic_count = sum(1 for ex in sample
                           if any(phrase in ex.get("user_input_injection", "").lower()
                                 for phrase in generic_phrases))
        print(f"  Generic injections: {generic_count} ({generic_count/len(sample)*100:.1f}%)")

        return {
            "pattern_counts": pattern_counts,
            "short_injections": len(short_injections),
            "generic_count": generic_count
        }

    def estimate_training_time(self, num_examples: int) -> Dict:
        """Estimate training time and resource requirements"""
        print("\n=== TRAINING TIME ESTIMATION ===")

        # Assumptions based on typical small model fine-tuning
        # These are rough estimates for a model like GPT-2 or DistilGPT-2
        assumptions = {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_length": 1024,
            "epochs": 3,
            "steps_per_second": 0.5  # Conservative estimate for contrastive learning
        }

        effective_batch_size = assumptions["batch_size"] * assumptions["gradient_accumulation_steps"]
        steps_per_epoch = num_examples / effective_batch_size
        total_steps = steps_per_epoch * assumptions["epochs"]
        total_seconds = total_steps / assumptions["steps_per_second"]
        total_hours = total_seconds / 3600

        print(f"\n  Assumptions:")
        print(f"    Batch size: {assumptions['batch_size']}")
        print(f"    Gradient accumulation: {assumptions['gradient_accumulation_steps']}")
        print(f"    Effective batch size: {effective_batch_size}")
        print(f"    Max sequence length: {assumptions['max_length']}")
        print(f"    Epochs: {assumptions['epochs']}")
        print(f"    Est. steps/second: {assumptions['steps_per_second']}")

        print(f"\n  Estimates:")
        print(f"    Steps per epoch: {steps_per_epoch:.0f}")
        print(f"    Total training steps: {total_steps:.0f}")
        print(f"    Est. training time: {total_hours:.1f} hours ({total_hours/24:.1f} days)")

        print(f"\n  Note: These are rough estimates. Actual time depends on:")
        print(f"    - Hardware (GPU type, memory)")
        print(f"    - Model size")
        print(f"    - Learning rate and convergence")
        print(f"    - Contrastive loss computation overhead")

        return {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "estimated_hours": total_hours
        }

    def run_advanced_analysis(self):
        """Run all advanced analyses"""
        print("=" * 70)
        print("ADVANCED DATASET ANALYSIS")
        print("=" * 70)

        # Load training data
        print("\nLoading training data...")
        examples = []
        with open(self.train_file, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line.strip()))

        print(f"Loaded {len(examples):,} training examples")

        # Run analyses
        results = {}
        results["token_stats"] = self.analyze_token_statistics(examples)
        results["counterfactual_structure"] = self.analyze_counterfactual_structure(examples, sample_size=100)
        results["bias_analysis"] = self.analyze_bias_correlations(examples)
        results["injection_quality"] = self.check_injection_quality(examples, sample_size=100)
        results["training_estimates"] = self.estimate_training_time(len(examples))

        print("\n" + "=" * 70)
        print("ADVANCED ANALYSIS COMPLETE")
        print("=" * 70)

        return results

if __name__ == "__main__":
    analyzer = AdvancedAnalyzer("c:/isef/data/processed")
    analyzer.run_advanced_analysis()
