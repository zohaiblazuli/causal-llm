# Dataset Validation Scripts - Week 1 Phase 2

Comprehensive validation suite for ensuring dataset quality before training begins.

## Overview

This validation suite consists of 6 core scripts that check different aspects of dataset quality:

1. **test_data_loading.py** - Verifies all data files load correctly
2. **analyze_counterfactuals.py** - Analyzes quality of counterfactual pairs
3. **analyze_attack_diversity.py** - Checks attack type and technique diversity
4. **compute_statistics.py** - Computes comprehensive dataset statistics
5. **integrity_checks.py** - Runs integrity checks for leakage, duplicates, etc.
6. **run_all_validations.py** - Master script to run all validations

## Quick Start

### Run All Validations

```bash
cd C:/isef
python data/scripts/run_all_validations.py
```

This will run all 5 validation scripts in sequence and provide a summary report.

### Run Individual Validations

```bash
# Data loading test
python data/scripts/test_data_loading.py

# Counterfactual analysis
python data/scripts/analyze_counterfactuals.py

# Attack diversity analysis
python data/scripts/analyze_attack_diversity.py

# Statistics computation
python data/scripts/compute_statistics.py

# Integrity checks
python data/scripts/integrity_checks.py
```

## Requirements

### Core Dependencies
```bash
pip install numpy
```

### Optional Dependencies (for full functionality)
```bash
# For counterfactual analysis
pip install sentence-transformers scikit-learn

# For token statistics
pip install transformers torch
```

Note: Scripts will run with reduced functionality if optional dependencies are missing.

## Validation Details

### 1. Data Loading Test

**Purpose:** Verify that all dataset files load correctly and contain valid examples.

**Checks:**
- File existence and readability
- JSON parsing validity
- Required field presence
- Field value types and non-emptiness

**Pass Criteria:**
- All files load successfully
- <5% of examples have issues

**Output:**
- Console report with split-by-split breakdown
- Example issues if found

**Runtime:** <30 seconds

---

### 2. Counterfactual Quality Analysis

**Purpose:** Analyze the quality of counterfactual pairs for semantic similarity and diversity.

**Checks:**
- Semantic similarity between benign pairs (using sentence embeddings)
- Lexical overlap (Jaccard similarity)
- Distribution of similarity scores
- Low-quality pair identification

**Pass Criteria:**
- Semantic similarity >0.70 (pairs should be semantically similar)
- Lexical overlap <0.50 (pairs should use different wording)
- Quality score ≥70/100

**Metrics:**
- **Quality Score:** Composite score (0-100)
  - 60 points: Semantic similarity
  - 40 points: Lexical diversity

**Output:**
- Console report with metrics and distribution
- `data/processed/counterfactual_analysis.json` - Detailed analysis

**Runtime:** 2-5 minutes (depends on sample size and model loading)

---

### 3. Attack Diversity Analysis

**Purpose:** Ensure diverse coverage of attack types and techniques.

**Checks:**
- Attack type distribution and entropy
- Attack technique distribution and entropy
- Distinctness between injection and benign inputs
- Coverage matrix (category × attack type)

**Pass Criteria:**
- Attack type entropy >2.5
- Average distinctness >0.70
- Diversity score ≥80/100

**Metrics:**
- **Shannon Entropy:** Measures distribution uniformity
- **Distinctness:** 1 - Jaccard similarity between injection and benign
- **Diversity Score:** Composite score (0-100)
  - 40 points: Type diversity
  - 30 points: Technique diversity
  - 30 points: Distinctness

**Output:**
- Console report with distributions and entropy
- `data/processed/attack_diversity_analysis.json` - Detailed analysis

**Runtime:** 1-2 minutes

---

### 4. Dataset Statistics

**Purpose:** Compute comprehensive statistics about the dataset.

**Checks:**
- Token count distribution (if transformers available)
- Category balance (chi-square test)
- Difficulty distribution
- Attack coverage matrix
- Quality indicators (empty fields, output diversity)

**Pass Criteria:**
- Average tokens reasonable (<2048 preferred)
- Categories balanced (chi-square test passes)
- <1% empty fields
- Output diversity >50%

**Metrics:**
- Token statistics (mean, median, max, percentiles)
- Category balance (chi-square statistic)
- Coverage gaps (cells with <10 examples)

**Output:**
- Console report with comprehensive statistics
- `data/processed/dataset_statistics_final.json` - Statistics summary

**Runtime:** 2-5 minutes (depends on tokenizer)

---

### 5. Integrity Checks

**Purpose:** Ensure data integrity and consistency across splits.

**Checks:**
- Data leakage between splits
- Duplicate IDs within splits
- Schema conformance
- Text encoding quality
- Injection quality (keyword presence)
- Counterfactual consistency
- Split distribution (70/15/15 target)

**Pass Criteria:**
- No data leakage
- No duplicates
- <5% schema violations
- No encoding issues
- <10% weak injections
- Split distribution within reasonable ranges

**Output:**
- Console report with detailed checks
- `data/processed/integrity_report.json` - Integrity summary

**Runtime:** 1-2 minutes

---

## Understanding Results

### Status Codes

Each validation returns one of three statuses:

- **PASS** - All checks passed, no issues found
- **WARNING** - Minor issues found but dataset is usable
- **FAIL** - Major issues found, dataset needs fixes

### Exit Codes

Scripts use standard exit codes:
- `0` - Success (PASS or WARNING)
- `1` - Failure (FAIL)

This allows integration into CI/CD pipelines.

## Output Files

All validation scripts save detailed reports to `data/processed/`:

| File | Description |
|------|-------------|
| `counterfactual_analysis.json` | Counterfactual quality metrics |
| `attack_diversity_analysis.json` | Attack diversity metrics |
| `dataset_statistics_final.json` | Comprehensive statistics |
| `integrity_report.json` | Integrity check results |

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'sentence_transformers'`
- **Solution:** Install optional dependencies: `pip install sentence-transformers`
- **Alternative:** Script will skip semantic similarity analysis

**Issue:** `ModuleNotFoundError: No module named 'transformers'`
- **Solution:** Install transformers: `pip install transformers torch`
- **Alternative:** Script will use word count estimation for token statistics

**Issue:** File not found errors
- **Solution:** Ensure you have run data generation and splitting scripts first
- **Check:** Files should exist at `data/processed/train_split.jsonl`, etc.

**Issue:** Out of memory during analysis
- **Solution:** Validation scripts use sampling to avoid memory issues
- **Adjust:** Reduce sample sizes in scripts if needed

## Performance Tips

1. **Run validations on a subset first:**
   - Edit scripts to use smaller sample sizes for testing
   - Increase sample sizes for final validation

2. **Skip optional dependencies for faster validation:**
   - Basic validation works without sentence-transformers
   - Token estimation works without transformers

3. **Use `run_all_validations.py` for batch processing:**
   - Runs all validations in sequence
   - Provides comprehensive summary

## Integration with Training Pipeline

### Before Training

Run all validations and ensure:
1. All tests return PASS or WARNING
2. No FAIL status from any validation
3. Review warnings and decide if acceptable

### Validation Checklist

- [ ] All data files load successfully
- [ ] Counterfactual quality score ≥70/100
- [ ] Attack diversity score ≥80/100
- [ ] No data leakage between splits
- [ ] No duplicate examples
- [ ] Category balance reasonable
- [ ] Token lengths within model limits

### Report Template

Use `WEEK1_DATASET_REPORT_TEMPLATE.md` to document validation results:
1. Run all validations
2. Fill in the template with results
3. Review with team
4. Sign off before training

## Advanced Usage

### Custom Validation

You can extend the validation suite by:

1. **Adding new checks to existing scripts:**
   ```python
   # In integrity_checks.py
   def check_custom_property():
       # Your check here
       pass
   ```

2. **Creating new validation scripts:**
   ```python
   # new_validation.py
   from validation_utils import load_jsonl, get_dataset_paths

   def custom_validation():
       data = load_jsonl(get_dataset_paths()["train"])
       # Your validation logic
       pass
   ```

3. **Adding to master script:**
   ```python
   # In run_all_validations.py
   validations.append(("new_validation.py", "Custom Validation"))
   ```

### Batch Processing

For validating multiple dataset versions:

```python
import subprocess

datasets = ["v1", "v2", "v3"]

for version in datasets:
    # Update paths in validation_utils.py
    # Run validations
    subprocess.run(["python", "data/scripts/run_all_validations.py"])
    # Save results with version tag
```

## Best Practices

1. **Run validations after any data changes:**
   - New data generation
   - Data augmentation
   - Filtering or cleaning

2. **Document all validation runs:**
   - Use the report template
   - Keep historical records
   - Track quality trends

3. **Set quality thresholds:**
   - Define minimum acceptable scores
   - Enforce thresholds in CI/CD
   - Review and update thresholds

4. **Review warnings carefully:**
   - Understand what each warning means
   - Assess impact on training
   - Document decisions to proceed

5. **Keep validation scripts updated:**
   - Update checks as requirements evolve
   - Add new validations as needed
   - Maintain documentation

## Support

For issues or questions:
1. Check this README for common solutions
2. Review script docstrings for details
3. Check output files for detailed diagnostics
4. Review validation logic in source code

## Version History

- **v1.0** (2025-01-13): Initial validation suite
  - 5 core validation scripts
  - Report template
  - Utility functions

---

**Last Updated:** 2025-01-13
**Maintained By:** DataForge Agent
**Project:** Provably Safe LLM Agents via Causal Intervention
