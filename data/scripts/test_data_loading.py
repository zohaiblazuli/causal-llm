#!/usr/bin/env python3
"""
Task 1: Comprehensive Data Loading Test
Tests loading, parsing, and validates all required fields across all splits.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Required fields for each example
REQUIRED_FIELDS = [
    'id',
    'task_category',
    'system_instruction',
    'user_input_benign_1',
    'expected_output_1',
    'user_input_benign_2',
    'expected_output_2',
    'user_input_injection',
    'expected_behavior_injection',
    'expected_output_injection',
    'attack_type',
    'attack_technique'
]


def load_and_validate_split(file_path: Path) -> Dict:
    """Load a split file and validate all examples."""
    results = {
        'file': file_path.name,
        'total_lines': 0,
        'loaded_successfully': 0,
        'parsing_errors': [],
        'missing_fields': defaultdict(list),
        'corrupted_entries': [],
        'valid_examples': 0
    }

    print(f"\n{'='*80}")
    print(f"Loading: {file_path.name}")
    print(f"{'='*80}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                results['total_lines'] += 1

                # Try to parse JSON
                try:
                    example = json.loads(line.strip())
                    results['loaded_successfully'] += 1

                    # Validate all required fields are present
                    missing = []
                    for field in REQUIRED_FIELDS:
                        if field not in example:
                            missing.append(field)
                            results['missing_fields'][field].append(f"Line {line_num}, ID: {example.get('id', 'UNKNOWN')}")

                    if missing:
                        results['corrupted_entries'].append({
                            'line': line_num,
                            'id': example.get('id', 'UNKNOWN'),
                            'missing_fields': missing
                        })
                    else:
                        # All fields present - additional validation
                        # Check for empty values
                        empty_fields = []
                        for field in REQUIRED_FIELDS:
                            if not example[field] or (isinstance(example[field], str) and not example[field].strip()):
                                empty_fields.append(field)

                        if empty_fields:
                            results['corrupted_entries'].append({
                                'line': line_num,
                                'id': example.get('id', 'UNKNOWN'),
                                'empty_fields': empty_fields
                            })
                        else:
                            results['valid_examples'] += 1

                except json.JSONDecodeError as e:
                    results['parsing_errors'].append({
                        'line': line_num,
                        'error': str(e),
                        'content_preview': line[:100]
                    })
                except Exception as e:
                    results['corrupted_entries'].append({
                        'line': line_num,
                        'error': str(e),
                        'type': 'validation_error'
                    })

    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        results['file_error'] = 'File not found'
        return results
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        results['file_error'] = str(e)
        return results

    # Print summary
    print(f"\nTotal lines: {results['total_lines']}")
    print(f"Loaded successfully: {results['loaded_successfully']}")
    print(f"Valid examples (all fields present and non-empty): {results['valid_examples']}")
    print(f"Parsing errors: {len(results['parsing_errors'])}")
    print(f"Corrupted entries: {len(results['corrupted_entries'])}")

    if results['parsing_errors']:
        print(f"\nPARSING ERRORS ({len(results['parsing_errors'])}):")
        for error in results['parsing_errors'][:5]:  # Show first 5
            print(f"  Line {error['line']}: {error['error']}")
        if len(results['parsing_errors']) > 5:
            print(f"  ... and {len(results['parsing_errors']) - 5} more")

    if results['missing_fields']:
        print(f"\nMISSING FIELDS:")
        for field, occurrences in results['missing_fields'].items():
            print(f"  {field}: {len(occurrences)} occurrences")
            for occ in occurrences[:3]:  # Show first 3
                print(f"    - {occ}")
            if len(occurrences) > 3:
                print(f"    ... and {len(occurrences) - 3} more")

    if results['corrupted_entries']:
        print(f"\nCORRUPTED ENTRIES ({len(results['corrupted_entries'])}):")
        for entry in results['corrupted_entries'][:5]:  # Show first 5
            print(f"  Line {entry.get('line', 'N/A')}, ID {entry.get('id', 'UNKNOWN')}:")
            if 'missing_fields' in entry:
                print(f"    Missing: {', '.join(entry['missing_fields'])}")
            if 'empty_fields' in entry:
                print(f"    Empty: {', '.join(entry['empty_fields'])}")
            if 'error' in entry:
                print(f"    Error: {entry['error']}")
        if len(results['corrupted_entries']) > 5:
            print(f"  ... and {len(results['corrupted_entries']) - 5} more")

    # Status
    if results['valid_examples'] == results['total_lines']:
        print(f"\n✓ STATUS: PASS - All examples valid")
    elif results['valid_examples'] > 0:
        print(f"\n⚠ STATUS: PARTIAL - {results['valid_examples']}/{results['total_lines']} examples valid")
    else:
        print(f"\n✗ STATUS: FAIL - No valid examples")

    return results


def generate_summary_report(all_results: List[Dict]) -> Dict:
    """Generate overall summary report."""
    summary = {
        'total_files': len(all_results),
        'total_lines': sum(r['total_lines'] for r in all_results),
        'total_valid': sum(r['valid_examples'] for r in all_results),
        'total_parsing_errors': sum(len(r['parsing_errors']) for r in all_results),
        'total_corrupted': sum(len(r['corrupted_entries']) for r in all_results),
        'files': []
    }

    for result in all_results:
        summary['files'].append({
            'name': result['file'],
            'lines': result['total_lines'],
            'valid': result['valid_examples'],
            'parsing_errors': len(result['parsing_errors']),
            'corrupted': len(result['corrupted_entries'])
        })

    return summary


def main():
    """Main execution function."""
    print("="*80)
    print("TASK 1: COMPREHENSIVE DATA LOADING TEST")
    print("="*80)
    print("\nValidating dataset integrity and field completeness...")

    # Define file paths
    data_dir = Path('C:/isef/data/processed')
    splits = ['train_split.jsonl', 'val_split.jsonl', 'test_split.jsonl']

    # Load and validate each split
    all_results = []
    for split in splits:
        file_path = data_dir / split
        results = load_and_validate_split(file_path)
        all_results.append(results)

    # Generate summary
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    summary = generate_summary_report(all_results)

    print(f"\nTotal files processed: {summary['total_files']}")
    print(f"Total examples: {summary['total_lines']}")
    print(f"Valid examples: {summary['total_valid']}")
    print(f"Parsing errors: {summary['total_parsing_errors']}")
    print(f"Corrupted entries: {summary['total_corrupted']}")

    print(f"\nPer-file breakdown:")
    for file_info in summary['files']:
        print(f"  {file_info['name']}:")
        print(f"    Lines: {file_info['lines']}")
        print(f"    Valid: {file_info['valid']}")
        print(f"    Errors: {file_info['parsing_errors']}")
        print(f"    Corrupted: {file_info['corrupted']}")

    # Overall status
    print(f"\n{'='*80}")
    if summary['total_valid'] == summary['total_lines'] and summary['total_parsing_errors'] == 0:
        print("✓ FINAL STATUS: PASS")
        print(f"  All {summary['total_lines']} examples loaded successfully")
        print("  All required fields present and valid")
        print("  No parsing errors detected")
        print("  No corrupted entries found")
        final_status = "PASS"
    elif summary['total_valid'] > 0:
        print("⚠ FINAL STATUS: PARTIAL PASS")
        print(f"  {summary['total_valid']}/{summary['total_lines']} examples valid")
        print(f"  {summary['total_parsing_errors']} parsing errors")
        print(f"  {summary['total_corrupted']} corrupted entries")
        print("  ⚠ MANUAL REVIEW REQUIRED")
        final_status = "PARTIAL"
    else:
        print("✗ FINAL STATUS: FAIL")
        print("  Critical issues detected")
        print("  Dataset cannot be used for training")
        final_status = "FAIL"
    print(f"{'='*80}\n")

    # Save detailed results
    output_file = data_dir / 'loading_test_results.json'
    output_data = {
        'summary': summary,
        'detailed_results': all_results,
        'final_status': final_status
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"Detailed results saved to: {output_file}")

    return final_status


if __name__ == "__main__":
    status = main()
    exit(0 if status == "PASS" else 1)
