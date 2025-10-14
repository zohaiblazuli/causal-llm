"""
Data Validation and Quality Control for Counterfactual Dataset
Ensures dataset meets quality standards and identifies issues
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import difflib
from datetime import datetime


class DataValidator:
    """
    Comprehensive validation for counterfactual dataset quality
    """

    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize validator

        Args:
            similarity_threshold: Maximum allowed similarity for duplicate detection
        """
        self.similarity_threshold = similarity_threshold
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "errors": [],
            "statistics": {}
        }

    def validate_dataset(self, dataset_path: str) -> Dict:
        """
        Run all validation checks on dataset

        Args:
            dataset_path: Path to the dataset JSONL file

        Returns:
            Validation results dictionary
        """
        print("Loading dataset...")
        examples = self._load_dataset(dataset_path)

        print(f"Loaded {len(examples)} examples\n")

        # Run all validation checks
        print("Running validation checks...\n")

        self._check_required_fields(examples)
        self._check_duplicates(examples)
        self._check_counterfactual_quality(examples)
        self._check_injection_distinctness(examples)
        self._check_output_format(examples)
        self._check_category_balance(examples)
        self._check_attack_diversity(examples)
        self._check_semantic_coherence(examples)
        self._check_bias_and_fairness(examples)
        self._check_difficulty_distribution(examples)

        # Generate summary
        self._generate_summary()

        return self.validation_results

    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load dataset from JSONL file"""
        examples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        return examples

    def _check_required_fields(self, examples: List[Dict]):
        """Verify all required fields are present"""
        required_fields = [
            'id', 'task_category', 'system_instruction',
            'user_input_benign_1', 'expected_output_1',
            'user_input_benign_2', 'expected_output_2',
            'user_input_injection', 'expected_behavior_injection',
            'attack_type', 'attack_technique', 'difficulty'
        ]

        missing_fields_count = 0
        examples_with_missing = []

        for i, example in enumerate(examples):
            missing = [field for field in required_fields if field not in example]
            if missing:
                missing_fields_count += 1
                examples_with_missing.append({
                    'id': example.get('id', f'index_{i}'),
                    'missing_fields': missing
                })

        if missing_fields_count == 0:
            self.validation_results['checks_passed'].append(
                f"[PASS] All {len(examples)} examples have required fields"
            )
        else:
            self.validation_results['checks_failed'].append(
                f"[FAIL] {missing_fields_count} examples missing required fields"
            )
            self.validation_results['errors'].extend(examples_with_missing[:10])

        self.validation_results['statistics']['required_fields_check'] = {
            'total_examples': len(examples),
            'examples_with_missing_fields': missing_fields_count,
            'pass_rate': (len(examples) - missing_fields_count) / len(examples) * 100
        }

    def _check_duplicates(self, examples: List[Dict]):
        """Check for duplicate examples"""
        hashes = set()
        near_duplicates = []
        exact_duplicates = 0

        # Check exact duplicates
        for i, example in enumerate(examples):
            # Hash key fields
            key_string = (
                f"{example.get('user_input_benign_1', '')}"
                f"{example.get('user_input_benign_2', '')}"
                f"{example.get('user_input_injection', '')}"
            )
            key_hash = hashlib.md5(key_string.encode()).hexdigest()

            if key_hash in hashes:
                exact_duplicates += 1
            else:
                hashes.add(key_hash)

        # Check near duplicates (expensive, sample-based)
        sample_size = min(1000, len(examples))
        sample_indices = list(range(0, len(examples), len(examples) // sample_size))

        for i in sample_indices[:sample_size]:
            for j in sample_indices[i+1:sample_size]:
                if i >= len(examples) or j >= len(examples):
                    continue

                similarity = self._compute_similarity(
                    examples[i].get('user_input_benign_1', ''),
                    examples[j].get('user_input_benign_1', '')
                )

                if similarity > self.similarity_threshold:
                    near_duplicates.append({
                        'id1': examples[i].get('id', f'index_{i}'),
                        'id2': examples[j].get('id', f'index_{j}'),
                        'similarity': similarity
                    })

        duplicate_rate = (exact_duplicates / len(examples)) * 100

        if exact_duplicates == 0:
            self.validation_results['checks_passed'].append(
                "[PASS] No exact duplicates found"
            )
        else:
            self.validation_results['checks_failed'].append(
                f"[FAIL] Found {exact_duplicates} exact duplicates ({duplicate_rate:.2f}%)"
            )

        if near_duplicates:
            self.validation_results['warnings'].append(
                f"[WARN] Found {len(near_duplicates)} potential near-duplicates in sample"
            )

        self.validation_results['statistics']['duplicate_check'] = {
            'exact_duplicates': exact_duplicates,
            'near_duplicates_in_sample': len(near_duplicates),
            'duplicate_rate': duplicate_rate
        }

    def _check_counterfactual_quality(self, examples: List[Dict]):
        """Verify counterfactual pairs are semantically similar"""
        low_quality_pairs = []
        similarity_scores = []

        for example in examples[:1000]:  # Sample for performance
            benign_1 = example.get('user_input_benign_1', '')
            benign_2 = example.get('user_input_benign_2', '')

            # Check if they're too similar (not diverse enough)
            similarity = self._compute_similarity(benign_1, benign_2)
            similarity_scores.append(similarity)

            # Counterfactuals should be different but related
            # Too high similarity (>0.9) means not diverse enough
            # Too low similarity (<0.1) means not related
            if similarity > 0.9 or similarity < 0.1:
                low_quality_pairs.append({
                    'id': example.get('id', ''),
                    'similarity': similarity,
                    'issue': 'too_similar' if similarity > 0.9 else 'too_different'
                })

        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        quality_rate = (1 - len(low_quality_pairs) / len(similarity_scores)) * 100 if similarity_scores else 0

        if quality_rate >= 90:
            self.validation_results['checks_passed'].append(
                f"[PASS] Counterfactual pairs quality: {quality_rate:.1f}% pass rate"
            )
        else:
            self.validation_results['checks_failed'].append(
                f"[FAIL] Counterfactual pairs quality: {quality_rate:.1f}% pass rate (target: 90%)"
            )

        self.validation_results['statistics']['counterfactual_quality'] = {
            'average_similarity': avg_similarity,
            'quality_pass_rate': quality_rate,
            'low_quality_pairs': len(low_quality_pairs),
            'sample_size': len(similarity_scores)
        }

    def _check_injection_distinctness(self, examples: List[Dict]):
        """Ensure injection attacks are distinct from benign inputs"""
        similar_to_benign = []

        for example in examples[:1000]:  # Sample
            benign_1 = example.get('user_input_benign_1', '')
            injection = example.get('user_input_injection', '')

            similarity = self._compute_similarity(benign_1, injection)

            # Injection should be clearly different from benign
            if similarity > 0.7:
                similar_to_benign.append({
                    'id': example.get('id', ''),
                    'similarity': similarity
                })

        distinctness_rate = (1 - len(similar_to_benign) / min(1000, len(examples))) * 100

        if distinctness_rate >= 95:
            self.validation_results['checks_passed'].append(
                f"[PASS] Injection attacks are distinct: {distinctness_rate:.1f}% pass rate"
            )
        else:
            self.validation_results['checks_failed'].append(
                f"[FAIL] Injection distinctness: {distinctness_rate:.1f}% (target: 95%)"
            )

        self.validation_results['statistics']['injection_distinctness'] = {
            'pass_rate': distinctness_rate,
            'similar_to_benign_count': len(similar_to_benign)
        }

    def _check_output_format(self, examples: List[Dict]):
        """Validate output format and content"""
        invalid_outputs = []

        for example in examples:
            output_1 = example.get('expected_output_1', '')
            output_2 = example.get('expected_output_2', '')
            output_injection = example.get('expected_output_injection', '')

            # Check for empty outputs
            if not output_1 or not output_2 or not output_injection:
                invalid_outputs.append({
                    'id': example.get('id', ''),
                    'issue': 'empty_output'
                })
                continue

            # Check output lengths are reasonable
            if len(output_1) < 10 or len(output_2) < 10:
                invalid_outputs.append({
                    'id': example.get('id', ''),
                    'issue': 'output_too_short'
                })

        format_pass_rate = (1 - len(invalid_outputs) / len(examples)) * 100

        if format_pass_rate >= 99:
            self.validation_results['checks_passed'].append(
                f"[PASS] Output format validation: {format_pass_rate:.1f}% pass rate"
            )
        else:
            self.validation_results['checks_failed'].append(
                f"[FAIL] Output format issues: {format_pass_rate:.1f}% pass rate"
            )

        self.validation_results['statistics']['output_format'] = {
            'pass_rate': format_pass_rate,
            'invalid_count': len(invalid_outputs)
        }

    def _check_category_balance(self, examples: List[Dict]):
        """Check for balanced distribution across categories"""
        category_counts = Counter(ex.get('task_category', '') for ex in examples)
        total = len(examples)

        expected_per_category = total / 5  # 5 categories
        tolerance = 0.02  # 2% tolerance

        imbalanced_categories = []
        for category, count in category_counts.items():
            actual_ratio = count / total
            expected_ratio = 1 / 5
            deviation = abs(actual_ratio - expected_ratio)

            if deviation > tolerance:
                imbalanced_categories.append({
                    'category': category,
                    'count': count,
                    'expected': int(expected_per_category),
                    'deviation': f"{deviation * 100:.1f}%"
                })

        if not imbalanced_categories:
            self.validation_results['checks_passed'].append(
                "[PASS] Category distribution is balanced (within 2% tolerance)"
            )
        else:
            self.validation_results['warnings'].append(
                f"[WARN] {len(imbalanced_categories)} categories outside balance tolerance"
            )

        self.validation_results['statistics']['category_balance'] = {
            'distribution': dict(category_counts),
            'imbalanced_categories': imbalanced_categories
        }

    def _check_attack_diversity(self, examples: List[Dict]):
        """Verify diversity of attack types and techniques"""
        attack_type_counts = Counter(ex.get('attack_type', '') for ex in examples)
        attack_technique_counts = Counter(ex.get('attack_technique', '') for ex in examples)

        unique_attack_types = len(attack_type_counts)
        unique_techniques = len(attack_technique_counts)

        # Calculate entropy for diversity
        total = len(examples)
        attack_type_entropy = self._calculate_entropy(attack_type_counts, total)
        technique_entropy = self._calculate_entropy(attack_technique_counts, total)

        # High entropy (>2.0) indicates good diversity
        if attack_type_entropy > 2.0:
            self.validation_results['checks_passed'].append(
                f"[PASS] Good attack type diversity: {unique_attack_types} types, entropy={attack_type_entropy:.2f}"
            )
        else:
            self.validation_results['warnings'].append(
                f"[WARN] Low attack type diversity: entropy={attack_type_entropy:.2f} (target: >2.0)"
            )

        self.validation_results['statistics']['attack_diversity'] = {
            'unique_attack_types': unique_attack_types,
            'unique_techniques': unique_techniques,
            'attack_type_entropy': attack_type_entropy,
            'technique_entropy': technique_entropy,
            'attack_type_distribution': dict(attack_type_counts.most_common(10)),
            'technique_distribution': dict(attack_technique_counts.most_common(10))
        }

    def _check_semantic_coherence(self, examples: List[Dict]):
        """Check for semantic coherence in examples"""
        incoherent_examples = []

        # Sample for performance
        for example in examples[:500]:
            system_instruction = example.get('system_instruction', '')
            benign_1 = example.get('user_input_benign_1', '')
            output_1 = example.get('expected_output_1', '')

            # Basic coherence checks
            # 1. Output should relate to input
            input_words = set(benign_1.lower().split())
            output_words = set(output_1.lower().split())
            common_words = input_words & output_words
            common_ratio = len(common_words) / max(len(output_words), 1)

            # 2. Check for reasonable length ratios
            length_ratio = len(output_1) / max(len(benign_1), 1)

            if common_ratio < 0.1 and length_ratio < 0.3:
                incoherent_examples.append({
                    'id': example.get('id', ''),
                    'common_word_ratio': common_ratio,
                    'length_ratio': length_ratio
                })

        coherence_rate = (1 - len(incoherent_examples) / min(500, len(examples))) * 100

        if coherence_rate >= 95:
            self.validation_results['checks_passed'].append(
                f"[PASS] Semantic coherence: {coherence_rate:.1f}% pass rate"
            )
        else:
            self.validation_results['warnings'].append(
                f"[WARN] Potential coherence issues: {coherence_rate:.1f}% pass rate"
            )

        self.validation_results['statistics']['semantic_coherence'] = {
            'pass_rate': coherence_rate,
            'incoherent_count': len(incoherent_examples),
            'sample_size': min(500, len(examples))
        }

    def _check_bias_and_fairness(self, examples: List[Dict]):
        """Check for potential bias issues"""
        # Check for overrepresentation of certain patterns
        system_instruction_counts = Counter(ex.get('system_instruction', '') for ex in examples)

        max_instruction_count = max(system_instruction_counts.values())
        total = len(examples)
        max_ratio = max_instruction_count / total

        bias_issues = []

        # No single system instruction should be > 25% of dataset
        if max_ratio > 0.25:
            bias_issues.append({
                'type': 'system_instruction_overrepresentation',
                'ratio': f"{max_ratio * 100:.1f}%"
            })

        # Check for language/phrasing diversity in attacks
        injection_phrases = []
        for ex in examples[:1000]:
            injection = ex.get('user_input_injection', '').lower()
            if 'ignore' in injection:
                injection_phrases.append('ignore')
            if 'disregard' in injection:
                injection_phrases.append('disregard')
            if 'forget' in injection:
                injection_phrases.append('forget')

        phrase_counts = Counter(injection_phrases)
        if phrase_counts:
            most_common_phrase, count = phrase_counts.most_common(1)[0]
            phrase_ratio = count / len(injection_phrases) if injection_phrases else 0

            if phrase_ratio > 0.4:
                bias_issues.append({
                    'type': 'attack_phrase_overuse',
                    'phrase': most_common_phrase,
                    'ratio': f"{phrase_ratio * 100:.1f}%"
                })

        if not bias_issues:
            self.validation_results['checks_passed'].append(
                "[PASS] No significant bias issues detected"
            )
        else:
            self.validation_results['warnings'].append(
                f"[WARN] {len(bias_issues)} potential bias issues detected"
            )

        self.validation_results['statistics']['bias_check'] = {
            'issues_found': len(bias_issues),
            'bias_details': bias_issues
        }

    def _check_difficulty_distribution(self, examples: List[Dict]):
        """Check distribution of difficulty levels"""
        difficulty_counts = Counter(ex.get('difficulty', '') for ex in examples)

        # Ideal distribution: some easy, mostly medium, some hard
        total = len(examples)
        distribution = {
            diff: count / total * 100
            for diff, count in difficulty_counts.items()
        }

        # Check if we have all difficulty levels
        expected_difficulties = {'trivial', 'easy', 'medium', 'hard', 'expert'}
        missing_difficulties = expected_difficulties - set(difficulty_counts.keys())

        if not missing_difficulties:
            self.validation_results['checks_passed'].append(
                "[PASS] All difficulty levels represented"
            )
        else:
            self.validation_results['warnings'].append(
                f"[WARN] Missing difficulty levels: {missing_difficulties}"
            )

        self.validation_results['statistics']['difficulty_distribution'] = {
            'distribution': dict(difficulty_counts),
            'percentages': distribution,
            'missing_levels': list(missing_difficulties)
        }

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts using sequence matching"""
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def _calculate_entropy(self, counts: Counter, total: int) -> float:
        """Calculate Shannon entropy for distribution"""
        import math
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _generate_summary(self):
        """Generate validation summary"""
        total_checks = len(self.validation_results['checks_passed']) + \
                      len(self.validation_results['checks_failed'])

        passed_checks = len(self.validation_results['checks_passed'])
        failed_checks = len(self.validation_results['checks_failed'])
        warnings = len(self.validation_results['warnings'])

        self.validation_results['summary'] = {
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': failed_checks,
            'warnings': warnings,
            'pass_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0
        }

    def save_validation_report(self, output_path: str):
        """Save validation report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        print(f"Validation report saved to {output_path}")

    def print_summary(self):
        """Print validation summary to console"""
        summary = self.validation_results.get('summary', {})

        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        print(f"\nTotal Checks: {summary.get('total_checks', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print(f"Warnings: {summary.get('warnings', 0)}")
        print(f"Overall Pass Rate: {summary.get('pass_rate', 0):.1f}%")

        print("\n" + "-" * 80)
        print("PASSED CHECKS")
        print("-" * 80)
        for check in self.validation_results['checks_passed']:
            print(f"  {check}")

        if self.validation_results['checks_failed']:
            print("\n" + "-" * 80)
            print("FAILED CHECKS")
            print("-" * 80)
            for check in self.validation_results['checks_failed']:
                print(f"  {check}")

        if self.validation_results['warnings']:
            print("\n" + "-" * 80)
            print("WARNINGS")
            print("-" * 80)
            for warning in self.validation_results['warnings']:
                print(f"  {warning}")

        print("\n" + "=" * 80)


def main():
    """Main validation execution"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_validation.py <dataset_path>")
        print("Example: python data_validation.py data/processed/counterfactual_pairs.jsonl")
        sys.exit(1)

    dataset_path = sys.argv[1]

    if not Path(dataset_path).exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)
    print(f"\nDataset: {dataset_path}\n")

    validator = DataValidator(similarity_threshold=0.95)
    results = validator.validate_dataset(dataset_path)

    validator.print_summary()

    # Save report
    output_dir = Path(dataset_path).parent
    report_path = output_dir / "validation_report.json"
    validator.save_validation_report(str(report_path))

    # Exit with appropriate code
    if results['summary']['pass_rate'] < 80:
        print("\n[WARN] WARNING: Validation pass rate below 80%. Review failed checks.")
        sys.exit(1)
    else:
        print("\n[PASS] Dataset validation complete. Quality standards met.")
        sys.exit(0)


if __name__ == "__main__":
    main()
