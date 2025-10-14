# Week 1 Phase 2 - Validation Scripts Checklist

Use this checklist to verify all validation scripts are ready and functional.

## Pre-Flight Checklist

### File Verification

**Core Validation Scripts** (6)
- [ ] `data/scripts/test_data_loading.py` - 9.5KB
- [ ] `data/scripts/analyze_counterfactuals.py` - 14KB
- [ ] `data/scripts/analyze_attack_diversity.py` - 16KB
- [ ] `data/scripts/compute_statistics.py` - 11KB
- [ ] `data/scripts/integrity_checks.py` - 11KB
- [ ] `data/scripts/run_all_validations.py` - 2.8KB

**Supporting Files** (4)
- [ ] `data/scripts/validation_utils.py` - 3.3KB
- [ ] `data/scripts/VALIDATION_README.md` - 10KB
- [ ] `WEEK1_VALIDATION_QUICKSTART.md` - 7.7KB
- [ ] `WEEK1_DATASET_REPORT_TEMPLATE.md` - 5.8KB

**Total:** 10 files

---

## Dependency Installation

### Required Dependencies
- [ ] Python 3.8 or higher installed
- [ ] numpy installed (`pip install numpy`)

### Optional Dependencies (Recommended)
- [ ] sentence-transformers installed (`pip install sentence-transformers`)
- [ ] scikit-learn installed (`pip install scikit-learn`)
- [ ] transformers installed (`pip install transformers`)
- [ ] torch installed (`pip install torch`)

**Note:** Scripts will work with reduced functionality without optional dependencies.

---

## Dataset Verification

### Required Dataset Files
- [ ] `data/processed/train_split.jsonl` exists
- [ ] `data/processed/val_split.jsonl` exists
- [ ] `data/processed/test_split.jsonl` exists
- [ ] All files are non-empty
- [ ] Files are valid JSONL format

### Dataset Size Check
- [ ] Train split: >100 examples recommended
- [ ] Val split: >20 examples recommended
- [ ] Test split: >20 examples recommended
- [ ] Total examples: >200 recommended

---

## Script Functionality Tests

### Test 1: Data Loading Script
```bash
python data/scripts/test_data_loading.py
```
- [ ] Script runs without errors
- [ ] Displays file loading results
- [ ] Shows valid/corrupted counts
- [ ] Returns exit code 0 or 1
- [ ] Creates `data/processed/loading_test_results.json`

### Test 2: Counterfactual Analysis Script
```bash
python data/scripts/analyze_counterfactuals.py
```
- [ ] Script runs (may take 2-5 minutes)
- [ ] Displays semantic similarity metrics
- [ ] Shows quality score
- [ ] Returns exit code 0 or 1
- [ ] Creates `data/processed/counterfactual_analysis.json`

### Test 3: Attack Diversity Script
```bash
python data/scripts/analyze_attack_diversity.py
```
- [ ] Script runs (may take 1-2 minutes)
- [ ] Displays entropy metrics
- [ ] Shows attack type distribution
- [ ] Returns exit code 0 or 1
- [ ] Creates `data/processed/attack_diversity_analysis.json`

### Test 4: Statistics Script
```bash
python data/scripts/compute_statistics.py
```
- [ ] Script runs (may take 2-5 minutes)
- [ ] Displays token statistics
- [ ] Shows category balance
- [ ] Returns exit code 0 or 1
- [ ] Creates `data/processed/dataset_statistics_final.json`

### Test 5: Integrity Checks Script
```bash
python data/scripts/integrity_checks.py
```
- [ ] Script runs (may take 1-2 minutes)
- [ ] Checks for data leakage
- [ ] Checks for duplicates
- [ ] Returns exit code 0 or 1
- [ ] Creates `data/processed/integrity_report.json`

### Test 6: Master Validation Script
```bash
python data/scripts/run_all_validations.py
```
- [ ] Script runs all 5 validations
- [ ] Displays progress for each
- [ ] Shows final summary
- [ ] Returns exit code 0 or 1
- [ ] All sub-validations complete

---

## Output Verification

### Console Output Check
- [ ] All scripts print clear headers
- [ ] Metrics are displayed with targets
- [ ] Pass/Warning/Fail status shown
- [ ] Summary provided at end

### JSON Output Check
- [ ] `loading_test_results.json` created and valid
- [ ] `counterfactual_analysis.json` created and valid
- [ ] `attack_diversity_analysis.json` created and valid
- [ ] `dataset_statistics_final.json` created and valid
- [ ] `integrity_report.json` created and valid

### Content Verification
- [ ] JSON files contain expected keys
- [ ] Metrics are numeric and reasonable
- [ ] No null or undefined values
- [ ] Arrays/objects properly structured

---

## Quality Threshold Verification

### Data Loading Thresholds
- [ ] Valid examples: >95%
- [ ] Parsing errors: 0
- [ ] Missing fields: <1%

### Counterfactual Thresholds
- [ ] Semantic similarity: >0.70 (or documented reason if lower)
- [ ] Lexical overlap: <0.50 (or documented reason if higher)
- [ ] Quality score: ≥70/100 (or documented reason if lower)

### Attack Diversity Thresholds
- [ ] Type entropy: >2.5 (or documented reason if lower)
- [ ] Distinctness: >0.70 (or documented reason if lower)
- [ ] Coverage gaps: <10 (or documented reason if more)

### Statistics Thresholds
- [ ] Average tokens: <2048 preferred
- [ ] Examples >2048: <5% preferred
- [ ] Categories balanced (chi-square passes)

### Integrity Thresholds
- [ ] Data leakage: 0 (REQUIRED)
- [ ] Duplicates: 0 (REQUIRED)
- [ ] Schema violations: <5%
- [ ] Encoding issues: 0 (REQUIRED)

---

## Documentation Verification

### README Files
- [ ] `VALIDATION_README.md` is comprehensive
- [ ] All scripts documented
- [ ] Usage examples provided
- [ ] Troubleshooting section included

### Quick Start Guide
- [ ] `WEEK1_VALIDATION_QUICKSTART.md` exists
- [ ] Installation steps clear
- [ ] Running instructions simple
- [ ] Common issues addressed

### Report Template
- [ ] `WEEK1_DATASET_REPORT_TEMPLATE.md` exists
- [ ] All validation sections included
- [ ] Tables and metrics fields present
- [ ] Sign-off section included

---

## Workflow Verification

### End-to-End Test
- [ ] Navigate to project directory: `cd C:/isef`
- [ ] Run master script: `python data/scripts/run_all_validations.py`
- [ ] Wait for completion (5-15 minutes)
- [ ] Review console summary
- [ ] Check all JSON reports created
- [ ] Verify pass/warning/fail statuses

### Report Creation Test
- [ ] Open `WEEK1_DATASET_REPORT_TEMPLATE.md`
- [ ] Fill in sample results from validation run
- [ ] Verify all sections can be completed
- [ ] Check template makes sense

---

## Error Handling Verification

### Missing Dependencies
- [ ] Scripts warn about missing dependencies
- [ ] Scripts continue with reduced functionality
- [ ] Clear instructions provided for installation

### Missing Files
- [ ] Scripts detect missing dataset files
- [ ] Clear error messages displayed
- [ ] Exit gracefully with appropriate code

### Corrupted Data
- [ ] Scripts handle corrupted JSON
- [ ] Errors reported with line numbers
- [ ] Processing continues for valid data

### Edge Cases
- [ ] Empty datasets handled
- [ ] Very large datasets handled (sampling)
- [ ] Zero-division errors prevented
- [ ] Null/undefined values handled

---

## Performance Verification

### Runtime Check
- [ ] test_data_loading: <1 minute
- [ ] analyze_counterfactuals: <10 minutes
- [ ] analyze_attack_diversity: <5 minutes
- [ ] compute_statistics: <10 minutes
- [ ] integrity_checks: <5 minutes
- [ ] run_all_validations: <30 minutes total

### Memory Usage Check
- [ ] Scripts don't crash with OOM
- [ ] Memory usage reasonable (<4GB)
- [ ] Sampling used for large datasets

---

## Integration Testing

### CI/CD Readiness
- [ ] Scripts return appropriate exit codes
- [ ] Can be run in automated pipeline
- [ ] No user interaction required
- [ ] All output goes to stdout/files

### Reproducibility
- [ ] Same dataset produces same results
- [ ] Random seeds used where needed
- [ ] Results are deterministic

---

## Final Validation

### Complete Package Check
- [ ] All 10 files present
- [ ] All scripts executable
- [ ] All documentation complete
- [ ] No broken references or links

### Ready for Use
- [ ] Scripts tested on real dataset
- [ ] All features working
- [ ] Documentation accurate
- [ ] No known bugs

---

## Sign-Off

**Pre-validation Complete:** [ ] YES / [ ] NO

If all checkboxes are checked, the validation suite is ready for use.

**Verified By:** _______________
**Date:** _______________
**Status:** READY FOR WEEK 1 PHASE 2 VALIDATION

---

## Quick Reference Commands

```bash
# Navigate to project
cd C:/isef

# Install dependencies
pip install numpy sentence-transformers scikit-learn transformers torch

# Run all validations
python data/scripts/run_all_validations.py

# Run individual validations
python data/scripts/test_data_loading.py
python data/scripts/analyze_counterfactuals.py
python data/scripts/analyze_attack_diversity.py
python data/scripts/compute_statistics.py
python data/scripts/integrity_checks.py

# Check output files
ls data/processed/*.json

# View specific report
cat data/processed/dataset_statistics_final.json
```

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Missing dependency | `pip install <package>` |
| File not found | Check `data/processed/` directory |
| Out of memory | Reduce sample sizes in scripts |
| Slow performance | First run downloads models (one-time) |
| JSON parse error | Check dataset file format |
| Import error | Verify Python version ≥3.8 |

---

## Next Steps After Validation

1. [ ] Review all validation results
2. [ ] Complete validation report template
3. [ ] Document all warnings and failures
4. [ ] Make go/no-go decision for training
5. [ ] Sign off on validation report
6. [ ] Archive validation results
7. [ ] Proceed to training (if approved)

---

**Document Version:** 1.0
**Last Updated:** 2025-01-13
**Project:** Provably Safe LLM Agents via Causal Intervention
**Phase:** Phase 2 - Week 1
