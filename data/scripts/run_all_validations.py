"""
Run All Validations - Week 1 Phase 2
Master script to run all validation checks in sequence.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime


def run_script(script_name, description):
    """Run a validation script and return success status."""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70 + "\n")

    script_path = Path(__file__).parent / script_name

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            timeout=600  # 10 minute timeout
        )

        success = result.returncode == 0

        if success:
            print(f"\n>>> {description}: PASSED")
        else:
            print(f"\n>>> {description}: FAILED")

        return success

    except subprocess.TimeoutExpired:
        print(f"\n>>> {description}: TIMEOUT (exceeded 10 minutes)")
        return False
    except Exception as e:
        print(f"\n>>> {description}: ERROR - {e}")
        return False


def main():
    """Run all validation scripts."""
    print("="*70)
    print("Week 1 Phase 2 - Dataset Validation Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    validations = [
        ("test_data_loading.py", "Data Loading Test"),
        ("analyze_counterfactuals.py", "Counterfactual Quality Analysis"),
        ("analyze_attack_diversity.py", "Attack Diversity Analysis"),
        ("compute_statistics.py", "Dataset Statistics"),
        ("integrity_checks.py", "Integrity Checks")
    ]

    results = {}

    for script, description in validations:
        success = run_script(script, description)
        results[description] = success

    # Final summary
    print("\n\n")
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for description, success in results.items():
        status = "PASS" if success else "FAIL"
        symbol = "✓" if success else "X"
        print(f"  {symbol} {description:40s} [{status}]")

    print("="*70)

    passed = sum(1 for s in results.values() if s)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\nRESULT: ALL VALIDATIONS PASSED")
        print("Dataset is ready for training!")
        exit_code = 0
    elif passed >= total * 0.8:  # 80% pass rate
        print("\nRESULT: MOST VALIDATIONS PASSED (WARNING)")
        print("Dataset is usable but some issues should be addressed")
        exit_code = 0
    else:
        print("\nRESULT: VALIDATION FAILED")
        print("Dataset needs fixes before training")
        exit_code = 1

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
