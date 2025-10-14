# Week 1 Dataset Validation - Quick Start Guide

## Overview

This guide helps you quickly set up and run the Week 1 Phase 2 dataset validation suite.

## Prerequisites

### Required Files
Ensure you have generated and split your dataset:
- `data/processed/train_split.jsonl`
- `data/processed/val_split.jsonl`
- `data/processed/test_split.jsonl`

### Python Environment
Python 3.8+ required.

## Installation

### Step 1: Install Core Dependencies

```bash
pip install numpy
```

### Step 2: Install Optional Dependencies (Recommended)

For full validation functionality:

```bash
# For counterfactual semantic analysis
pip install sentence-transformers scikit-learn

# For token statistics
pip install transformers torch
```

**Note:** Scripts will work with reduced functionality without optional dependencies.

## Running Validations

### Option 1: Run All Validations (Recommended)

```bash
cd C:/isef
python data/scripts/run_all_validations.py
```

**Expected Runtime:** 5-15 minutes (depending on dataset size and dependencies)

**Output:**
- Console summary of all validations
- Individual JSON reports in `data/processed/`
- Overall PASS/WARNING/FAIL status

### Option 2: Run Individual Validations

```bash
# 1. Test data loading (~30 seconds)
python data/scripts/test_data_loading.py

# 2. Analyze counterfactuals (2-5 minutes)
python data/scripts/analyze_counterfactuals.py

# 3. Analyze attack diversity (1-2 minutes)
python data/scripts/analyze_attack_diversity.py

# 4. Compute statistics (2-5 minutes)
python data/scripts/compute_statistics.py

# 5. Run integrity checks (1-2 minutes)
python data/scripts/integrity_checks.py
```

## Understanding Results

### Status Indicators

Each validation returns one of:

- **PASS (✓)** - All checks passed
- **WARNING (⚠)** - Minor issues, dataset usable
- **FAIL (✗)** - Major issues, needs fixes

### Success Criteria

Your dataset is ready for training if:

1. **Data Loading:** All files load with <5% corrupted examples
2. **Counterfactuals:** Quality score ≥70/100
3. **Attack Diversity:** Diversity score ≥80/100, entropy >2.5
4. **Statistics:** Balanced categories, reasonable token counts
5. **Integrity:** No leakage, no duplicates, <5% violations

### Output Files

Check these files for detailed reports:

```
data/processed/
├── counterfactual_analysis.json
├── attack_diversity_analysis.json
├── dataset_statistics_final.json
└── integrity_report.json
```

## Completing the Validation Report

After running validations:

1. **Open the report template:**
   ```
   WEEK1_DATASET_REPORT_TEMPLATE.md
   ```

2. **Fill in results from console output and JSON files**

3. **Document any issues found**

4. **Make go/no-go decision for training**

5. **Sign off on the report**

## Common Issues & Solutions

### Issue 1: Missing Dependencies

**Symptom:** `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution:**
```bash
pip install sentence-transformers scikit-learn
```

**Or:** Run without semantic analysis (will skip counterfactual metrics)

### Issue 2: Files Not Found

**Symptom:** `ERROR: Dataset not found at...`

**Solution:**
1. Check if you've run data generation scripts
2. Verify file paths in `data/processed/`
3. Ensure splits were created correctly

### Issue 3: Out of Memory

**Symptom:** Script crashes during analysis

**Solution:**
- Close other applications
- Reduce sample sizes in scripts (edit `sample_size` variables)
- Run validations individually instead of all at once

### Issue 4: Tokenizer Download Fails

**Symptom:** Cannot download `meta-llama/Llama-2-7b-hf`

**Solution:**
- Ensure internet connection
- Script will fall back to word count estimation
- Or use alternative tokenizer in `compute_statistics.py`

## Quick Validation Checklist

Use this checklist before training:

```
[ ] All validation scripts installed and working
[ ] Core dependencies installed (numpy)
[ ] Optional dependencies installed (recommended)
[ ] Dataset files exist in data/processed/
[ ] All validations run successfully
[ ] No FAIL statuses returned
[ ] Warnings reviewed and documented
[ ] Report template completed
[ ] Results reviewed by team
[ ] Sign-off obtained
[ ] Ready to proceed to training
```

## Expected Results for Good Dataset

### Data Loading
- All files load: 100%
- Valid examples: >95%
- Status: PASS

### Counterfactuals
- Semantic similarity: 0.70-0.85
- Lexical diversity: 0.30-0.50
- Quality score: 70-90/100
- Status: PASS or WARNING

### Attack Diversity
- Type entropy: 2.5-4.0
- Distinctness: 0.70-0.85
- Diversity score: 80-95/100
- Status: PASS

### Statistics
- Average tokens: 200-1500
- Examples >2048: <5%
- Categories: Balanced
- Coverage gaps: <10
- Status: PASS

### Integrity
- Data leakage: 0
- Duplicates: 0
- Schema violations: <1%
- Weak injections: <10%
- Status: PASS

## Next Steps After Validation

### If All Tests Pass
1. Complete validation report
2. Archive validation results
3. Proceed to model training
4. Use validated splits for training

### If Warnings Found
1. Review each warning carefully
2. Assess impact on training
3. Document decision to proceed or fix
4. If proceeding, monitor training closely

### If Tests Fail
1. Identify root causes from reports
2. Fix data generation issues
3. Regenerate affected data
4. Re-run validations
5. Do not proceed to training until fixed

## Time Estimates

### Full Validation Suite
- **With all dependencies:** 5-15 minutes
- **Without optional deps:** 3-8 minutes
- **On large dataset (50K+ examples):** 15-30 minutes

### Individual Validations
- Data Loading: 30 seconds
- Counterfactuals: 2-5 minutes
- Attack Diversity: 1-2 minutes
- Statistics: 2-5 minutes
- Integrity: 1-2 minutes

## Support Resources

- **Detailed documentation:** `data/scripts/VALIDATION_README.md`
- **Script source code:** `data/scripts/*.py`
- **Utility functions:** `data/scripts/validation_utils.py`
- **Report template:** `WEEK1_DATASET_REPORT_TEMPLATE.md`

## Example Workflow

```bash
# 1. Navigate to project directory
cd C:/isef

# 2. Verify dataset exists
ls data/processed/*.jsonl

# 3. Install dependencies (if not done)
pip install numpy sentence-transformers scikit-learn transformers

# 4. Run all validations
python data/scripts/run_all_validations.py

# 5. Review console output
# Look for overall PASS/WARNING/FAIL status

# 6. Check detailed reports
cat data/processed/dataset_statistics_final.json

# 7. Complete validation report
# Edit WEEK1_DATASET_REPORT_TEMPLATE.md

# 8. Make decision
# If PASS: Proceed to training
# If WARNING: Review and decide
# If FAIL: Fix issues and re-validate
```

## Tips for Success

1. **Run validations early and often**
   - Don't wait until just before training
   - Catch issues during data generation

2. **Review all warnings**
   - Warnings might indicate subtle issues
   - Understand impact before proceeding

3. **Keep validation history**
   - Save completed reports
   - Track quality improvements over time

4. **Automate validation**
   - Integrate into data pipeline
   - Run automatically after generation

5. **Use sample datasets for testing**
   - Test validation scripts on small samples first
   - Ensure everything works before full run

## Questions?

- Check `VALIDATION_README.md` for detailed information
- Review script docstrings for implementation details
- Examine JSON reports for diagnostic data
- Test with small dataset first to understand output

---

**Good luck with your validation!**

Your dataset quality directly impacts model performance. Take time to ensure data is clean, diverse, and ready for training.

**Phase 2 Week 1 Goal:** Validate dataset quality before training begins
**Success Metric:** All validations PASS or WARNING with documented justification
**Next Step:** Begin training with confidence in data quality
