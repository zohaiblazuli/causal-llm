"""
Comprehensive Dataset Validation Script for ISEF 2026
Validates counterfactual triplet dataset for causal intervention training
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import re
import statistics

class DatasetValidator:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.train_file = self.data_dir / "train_split.jsonl"
        self.val_file = self.data_dir / "val_split.jsonl"
        self.test_file = self.data_dir / "test_split.jsonl"

        # Expected schema fields
        self.required_fields = [
            "id", "task_category", "system_instruction",
            "user_input_benign_1", "expected_output_1",
            "user_input_benign_2", "expected_output_2",
            "user_input_injection", "expected_behavior_injection",
            "expected_output_injection", "attack_type",
            "attack_technique", "difficulty"
        ]

        # Valid values for categorical fields
        self.valid_categories = ["email_assistant", "rag_qa", "code_generation",
                                "calendar_scheduling", "document_processor"]
        self.valid_difficulties = ["easy", "medium", "hard", "trivial"]

        # Storage for validation results
        self.issues = {
            "critical": [],
            "important": [],
            "minor": []
        }

        self.stats = {
            "train": {},
            "val": {},
            "test": {}
        }

    def log_issue(self, severity: str, message: str, details: str = ""):
        """Log an issue with severity level"""
        self.issues[severity].append({
            "message": message,
            "details": details
        })

    def validate_file_integrity(self) -> Dict[str, Any]:
        """Check if files exist and are readable"""
        print("\n=== FILE INTEGRITY CHECK ===")
        results = {}

        for split_name, file_path in [("train", self.train_file),
                                       ("val", self.val_file),
                                       ("test", self.test_file)]:
            print(f"\nChecking {split_name} split...")

            if not file_path.exists():
                self.log_issue("critical", f"{split_name} file not found", str(file_path))
                results[split_name] = {"exists": False}
                continue

            # Get file size
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)

            # Count lines
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)

            results[split_name] = {
                "exists": True,
                "size_mb": round(size_mb, 2),
                "line_count": line_count
            }

            print(f"  File exists: YES")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Lines: {line_count:,}")

            # Validate expected counts
            expected_counts = {"train": 7151, "val": 893, "test": 895}
            if line_count != expected_counts[split_name]:
                self.log_issue("important",
                             f"{split_name} has unexpected line count",
                             f"Expected {expected_counts[split_name]}, got {line_count}")

        return results

    def validate_json_format(self, file_path: Path, split_name: str) -> Tuple[List[Dict], List[int]]:
        """Validate JSON format and return valid examples + corrupted line numbers"""
        print(f"\n=== VALIDATING JSON FORMAT: {split_name} ===")

        examples = []
        corrupted_lines = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    examples.append(example)
                except json.JSONDecodeError as e:
                    corrupted_lines.append(line_num)
                    self.log_issue("critical",
                                 f"Invalid JSON in {split_name} at line {line_num}",
                                 str(e))

        print(f"  Valid JSON lines: {len(examples):,}")
        print(f"  Corrupted lines: {len(corrupted_lines)}")

        if corrupted_lines:
            print(f"  Corrupted line numbers: {corrupted_lines[:10]}...")

        return examples, corrupted_lines

    def validate_schema(self, examples: List[Dict], split_name: str) -> Dict[str, Any]:
        """Validate schema for all examples"""
        print(f"\n=== SCHEMA VALIDATION: {split_name} ===")

        schema_issues = {
            "missing_fields": [],
            "invalid_categories": [],
            "invalid_difficulties": [],
            "null_values": [],
            "empty_strings": []
        }

        for idx, example in enumerate(examples):
            ex_id = example.get("id", f"line_{idx+1}")

            # Check required fields
            missing = [field for field in self.required_fields if field not in example]
            if missing:
                schema_issues["missing_fields"].append((ex_id, missing))
                self.log_issue("critical",
                             f"Missing fields in example {ex_id}",
                             f"Missing: {missing}")

            # Check category validity
            category = example.get("task_category")
            if category and category not in self.valid_categories:
                schema_issues["invalid_categories"].append((ex_id, category))
                self.log_issue("important",
                             f"Invalid category in {ex_id}",
                             f"Got: {category}")

            # Check difficulty validity
            difficulty = example.get("difficulty")
            if difficulty and difficulty not in self.valid_difficulties:
                schema_issues["invalid_difficulties"].append((ex_id, difficulty))
                self.log_issue("important",
                             f"Invalid difficulty in {ex_id}",
                             f"Got: {difficulty}")

            # Check for null values
            for field in self.required_fields:
                value = example.get(field)
                if value is None:
                    schema_issues["null_values"].append((ex_id, field))
                    self.log_issue("critical",
                                 f"Null value in {ex_id}",
                                 f"Field: {field}")
                elif isinstance(value, str) and len(value.strip()) == 0:
                    schema_issues["empty_strings"].append((ex_id, field))
                    self.log_issue("important",
                                 f"Empty string in {ex_id}",
                                 f"Field: {field}")

        print(f"  Examples with missing fields: {len(schema_issues['missing_fields'])}")
        print(f"  Examples with invalid categories: {len(schema_issues['invalid_categories'])}")
        print(f"  Examples with invalid difficulties: {len(schema_issues['invalid_difficulties'])}")
        print(f"  Examples with null values: {len(schema_issues['null_values'])}")
        print(f"  Examples with empty strings: {len(schema_issues['empty_strings'])}")

        return schema_issues

    def analyze_text_quality(self, examples: List[Dict], split_name: str, sample_size: int = 100) -> Dict[str, Any]:
        """Analyze text quality on a sample"""
        print(f"\n=== TEXT QUALITY ANALYSIS: {split_name} (sample: {sample_size}) ===")

        import random
        sample = random.sample(examples, min(sample_size, len(examples)))

        text_fields = ["user_input_benign_1", "user_input_benign_2", "user_input_injection"]

        stats = {}
        for field in text_fields:
            texts = [ex.get(field, "") for ex in sample]

            # Simple tokenization (whitespace split)
            token_counts = [len(text.split()) for text in texts]
            char_counts = [len(text) for text in texts]

            stats[field] = {
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0,
                "avg_tokens": round(statistics.mean(token_counts), 2) if token_counts else 0,
                "median_tokens": statistics.median(token_counts) if token_counts else 0,
                "min_chars": min(char_counts) if char_counts else 0,
                "max_chars": max(char_counts) if char_counts else 0,
                "avg_chars": round(statistics.mean(char_counts), 2) if char_counts else 0
            }

            # Check for very short or very long examples
            too_short = sum(1 for tc in token_counts if tc < 10)
            too_long = sum(1 for tc in token_counts if tc > 500)  # Rough estimate for 1024 tokens

            stats[field]["too_short"] = too_short
            stats[field]["too_long"] = too_long

            print(f"\n  {field}:")
            print(f"    Token range: {stats[field]['min_tokens']} - {stats[field]['max_tokens']}")
            print(f"    Token avg/median: {stats[field]['avg_tokens']} / {stats[field]['median_tokens']}")
            print(f"    Char range: {stats[field]['min_chars']} - {stats[field]['max_chars']}")
            print(f"    Too short (<10 tokens): {too_short}")
            print(f"    Too long (>500 tokens): {too_long}")

        # Check for encoding issues
        encoding_issues = 0
        for ex in sample:
            for field in text_fields:
                text = ex.get(field, "")
                # Check for common encoding problems
                if any(char in text for char in ['\ufffd', '\x00']):
                    encoding_issues += 1
                    self.log_issue("important",
                                 f"Encoding issue in {ex.get('id')}",
                                 f"Field: {field}")
                    break

        print(f"\n  Encoding issues: {encoding_issues}")

        return stats

    def analyze_distribution(self, examples: List[Dict], split_name: str) -> Dict[str, Counter]:
        """Analyze distribution of categories, attack types, etc."""
        print(f"\n=== DISTRIBUTION ANALYSIS: {split_name} ===")

        distributions = {
            "categories": Counter(),
            "attack_types": Counter(),
            "attack_techniques": Counter(),
            "difficulties": Counter()
        }

        for ex in examples:
            distributions["categories"][ex.get("task_category")] += 1
            distributions["attack_types"][ex.get("attack_type")] += 1
            distributions["attack_techniques"][ex.get("attack_technique")] += 1
            distributions["difficulties"][ex.get("difficulty")] += 1

        # Print distributions
        total = len(examples)

        print(f"\n  Categories (total: {total}):")
        for cat, count in distributions["categories"].most_common():
            pct = (count / total) * 100
            print(f"    {cat}: {count} ({pct:.1f}%)")

        print(f"\n  Attack Types:")
        for att, count in distributions["attack_types"].most_common():
            print(f"    {att}: {count}")

        print(f"\n  Attack Techniques:")
        for tech, count in distributions["attack_techniques"].most_common():
            print(f"    {tech}: {count}")

        print(f"\n  Difficulty:")
        for diff, count in distributions["difficulties"].most_common():
            pct = (count / total) * 100
            print(f"    {diff}: {count} ({pct:.1f}%)")

        return distributions

    def check_duplicates(self, all_examples: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Check for duplicate IDs and similar examples across splits"""
        print(f"\n=== DUPLICATE ANALYSIS ===")

        # Check ID duplicates
        all_ids = []
        for split_name, examples in all_examples.items():
            all_ids.extend([ex.get("id") for ex in examples])

        id_counts = Counter(all_ids)
        duplicate_ids = {id_: count for id_, count in id_counts.items() if count > 1}

        print(f"\n  Total unique IDs: {len(id_counts)}")
        print(f"  Duplicate IDs: {len(duplicate_ids)}")

        if duplicate_ids:
            print(f"  Duplicate ID examples: {list(duplicate_ids.items())[:5]}")
            for id_, count in duplicate_ids.items():
                self.log_issue("critical",
                             f"Duplicate ID found: {id_}",
                             f"Appears {count} times")

        # Check for exact text duplicates across splits
        def get_text_signature(ex):
            return (ex.get("user_input_benign_1", ""),
                   ex.get("user_input_benign_2", ""),
                   ex.get("user_input_injection", ""))

        train_signatures = set(get_text_signature(ex) for ex in all_examples.get("train", []))
        val_signatures = set(get_text_signature(ex) for ex in all_examples.get("val", []))
        test_signatures = set(get_text_signature(ex) for ex in all_examples.get("test", []))

        train_val_overlap = train_signatures & val_signatures
        train_test_overlap = train_signatures & test_signatures
        val_test_overlap = val_signatures & test_signatures

        print(f"\n  Cross-split exact duplicates:")
        print(f"    Train-Val overlap: {len(train_val_overlap)}")
        print(f"    Train-Test overlap: {len(train_test_overlap)}")
        print(f"    Val-Test overlap: {len(val_test_overlap)}")

        if train_val_overlap or train_test_overlap or val_test_overlap:
            self.log_issue("critical",
                         "Data leakage detected",
                         f"Found exact duplicates across splits")

        return {
            "duplicate_ids": duplicate_ids,
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap)
        }

    def check_stratification(self, all_examples: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Check if splits are properly stratified"""
        print(f"\n=== STRATIFICATION ANALYSIS ===")

        results = {}

        for split_name, examples in all_examples.items():
            dist = Counter(ex.get("task_category") for ex in examples)
            total = len(examples)
            results[split_name] = {cat: (count/total)*100 for cat, count in dist.items()}

        print("\n  Category distribution across splits:")
        all_cats = set()
        for dist in results.values():
            all_cats.update(dist.keys())

        for cat in sorted(all_cats):
            print(f"\n    {cat}:")
            for split_name in ["train", "val", "test"]:
                pct = results[split_name].get(cat, 0)
                print(f"      {split_name}: {pct:.1f}%")

        # Check if distributions are similar (within 5 percentage points)
        for cat in all_cats:
            train_pct = results["train"].get(cat, 0)
            val_pct = results["val"].get(cat, 0)
            test_pct = results["test"].get(cat, 0)

            max_diff = max(abs(train_pct - val_pct),
                          abs(train_pct - test_pct),
                          abs(val_pct - test_pct))

            if max_diff > 5:
                self.log_issue("minor",
                             f"Stratification imbalance for {cat}",
                             f"Max difference: {max_diff:.1f} percentage points")

        return results

    def generate_report(self) -> str:
        """Generate comprehensive markdown report"""
        report = []
        report.append("# DATASET QUALITY VALIDATION REPORT")
        report.append("\n## ISEF 2026: Provably Safe LLM Agents via Causal Intervention")
        report.append(f"\nGenerated: {import_datetime()}\n")

        # Executive Summary
        report.append("\n## Executive Summary\n")

        total_issues = sum(len(self.issues[sev]) for sev in ["critical", "important", "minor"])
        report.append(f"- **Total Issues Found**: {total_issues}")
        report.append(f"  - Critical: {len(self.issues['critical'])}")
        report.append(f"  - Important: {len(self.issues['important'])}")
        report.append(f"  - Minor: {len(self.issues['minor'])}")

        # Determine overall status
        if len(self.issues['critical']) > 0:
            status = "NOT READY - Critical issues must be fixed"
            confidence = "LOW"
        elif len(self.issues['important']) > 10:
            status = "NEEDS IMPROVEMENT - Multiple important issues"
            confidence = "MEDIUM"
        elif len(self.issues['important']) > 0:
            status = "ACCEPTABLE - Minor improvements recommended"
            confidence = "MEDIUM-HIGH"
        else:
            status = "READY FOR TRAINING"
            confidence = "HIGH"

        report.append(f"\n**Training Readiness**: {status}")
        report.append(f"**Confidence Level**: {confidence}\n")

        # Issues by severity
        report.append("\n## Issues Identified\n")

        for severity in ["critical", "important", "minor"]:
            icon = {"critical": "🔴", "important": "🟡", "minor": "🟢"}[severity]
            report.append(f"\n### {icon} {severity.upper()} Issues\n")

            if not self.issues[severity]:
                report.append("None found.\n")
            else:
                for issue in self.issues[severity]:
                    report.append(f"- **{issue['message']}**")
                    if issue['details']:
                        report.append(f"  - Details: {issue['details']}")
                    report.append("")

        return "\n".join(report)

    def run_full_validation(self):
        """Run complete validation pipeline"""
        print("=" * 70)
        print("DATASET VALIDATION FOR ISEF 2026")
        print("Counterfactual Triplet Dataset for Causal Intervention")
        print("=" * 70)

        # 1. File integrity
        file_results = self.validate_file_integrity()

        # 2. Load and validate JSON
        all_examples = {}
        for split_name, file_path in [("train", self.train_file),
                                       ("val", self.val_file),
                                       ("test", self.test_file)]:
            if file_results[split_name]["exists"]:
                examples, corrupted = self.validate_json_format(file_path, split_name)
                all_examples[split_name] = examples

                # 3. Schema validation
                if examples:
                    self.validate_schema(examples, split_name)

                    # 4. Text quality analysis
                    self.analyze_text_quality(examples, split_name, sample_size=100)

                    # 5. Distribution analysis
                    self.stats[split_name] = self.analyze_distribution(examples, split_name)

        # 6. Cross-split checks
        if len(all_examples) > 1:
            self.check_duplicates(all_examples)
            self.check_stratification(all_examples)

        # 7. Generate report
        print("\n" + "=" * 70)
        print("GENERATING REPORT...")
        print("=" * 70)

        report = self.generate_report()

        # Save report
        report_path = self.data_dir.parent / "DATASET_QUALITY_REPORT.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nReport saved to: {report_path}")

        return report_path

def import_datetime():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    validator = DatasetValidator("c:/isef/data/processed")
    validator.run_full_validation()

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
