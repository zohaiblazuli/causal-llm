# Phase 2 Execution Guide
## Core Implementation & Training

**Duration:** January 2025 (4 weeks)
**Goal:** Train causal LLM and achieve attack success rate <10%

---

## Overview

Phase 2 focuses on executing the training, validating causal properties, and establishing baseline performance metrics. By the end of this phase, you will have:

1. ✅ A trained causal LLM model
2. ✅ Empirical validation of causal metrics (ε_causal, d-separation)
3. ✅ Attack success rate measurements
4. ✅ Comparison with baseline defenses

---

## Week-by-Week Breakdown

### **Week 1: Pre-Training Validation (Days 1-7)**

#### Objectives
- Verify training infrastructure is production-ready
- Validate dataset quality one final time
- Ensure no blockers before training starts

#### Tasks

**Day 1-2: Setup Verification**
```bash
# Verify all dependencies
python training/verify_setup.py

# Test memory configuration
python training/optimize_memory.py --config training/config.yaml

# Expected output: All checks pass, memory <5.5GB
```

**Day 3-4: Dataset Validation**
```bash
# Run data quality checks
python data/scripts/test_data_loading.py
python data/scripts/analyze_counterfactuals.py
python data/scripts/analyze_attack_diversity.py

# Expected output: Dataset quality EXCELLENT, ready for training
```

**Day 5-6: Dry Run**
```bash
# Run dry run with 100 samples
python training/dry_run.py

# Expected output:
# - 10 steps complete without OOM
# - Loss decreases
# - Memory stays <5.5GB
# - Checkpoint saves successfully
```

**Day 7: Review and Plan**
- Review all verification reports
- Identify any issues
- Prepare for Week 2 training launch

#### Deliverables
- [x] `WEEK1_SETUP_REPORT.md` - Setup verification results
- [x] `WEEK1_DATASET_REPORT.md` - Dataset quality report
- [x] Confirmation: READY TO TRAIN

---

### **Week 2: Initial Training (Days 8-14)**

#### Objectives
- Execute Epoch 1 of training
- Monitor closely for issues
- Validate training stability

#### Tasks

**Day 8: Launch Training**
```bash
# Start with debug mode (small subset)
python training/train.py --config training/config.yaml --debug

# If successful, launch full training
python training/train.py --config training/config.yaml
```

**Day 9-11: Monitor Training**
- Check W&B dashboard daily
- Monitor metrics:
  - Loss (task, causal, spurious) - should decrease
  - Causal stability - should increase toward 0.80
  - Spurious separation - should increase toward 0.75
  - Memory usage - should stay <5.5GB
  - Training speed - target 0.5-1.0 steps/sec

**Day 12-13: Epoch 1 Validation**
```bash
# After Epoch 1 completes, run validation
python evaluation/metrics.py --checkpoint checkpoints/epoch_1

# Check intermediate metrics:
# - Causal stability progress
# - Spurious separation progress
# - Attack success rate (may still be high)
```

**Day 14: Review and Adjust**
- Review Epoch 1 results
- Adjust hyperparameters if needed (learning rate, loss weights)
- Prepare for Epochs 2-3

#### Deliverables
- [x] `checkpoints/epoch_1/` - Epoch 1 checkpoint
- [x] W&B logs with all training metrics
- [x] `WEEK2_TRAINING_REPORT.md` - Epoch 1 analysis

---

### **Week 3: Complete Training & Verification (Days 15-21)**

#### Objectives
- Complete Epochs 2-3
- Run formal causal verification
- Select best model

#### Tasks

**Day 15-18: Continue Training**
```bash
# Training continues automatically
# Monitor for:
# - Convergence (loss plateaus)
# - Overfitting (train/val gap)
# - Early stopping triggers

# Training should complete in ~2 hours total for 3 epochs
```

**Day 19-20: Causal Verification**
```bash
# After training completes, run verification
python verification/independence_tests.py --model checkpoints/best_model

# Expected output:
# - HSIC test: d-separation holds (p > 0.05)
# - ε_causal: < 0.10
# - Status: PASS

python verification/causal_discovery.py --model checkpoints/best_model

# Expected output:
# - Learned graph matches S → R ← U, R → O
# - Match score: ≥ 0.66
# - Status: STRUCTURE CORRECT
```

**Day 21: Model Selection & Analysis**
- Select best checkpoint based on causal_stability metric
- Analyze training dynamics
- Document findings

#### Deliverables
- [x] `checkpoints/best_model/` - Best model checkpoint
- [x] `WEEK3_VERIFICATION_REPORT.md` - Causal verification results
- [x] Training complete confirmation

---

### **Week 4: Evaluation & Comparison (Days 22-28)**

#### Objectives
- Comprehensive attack evaluation
- Compare with baseline defenses
- Generate Phase 2 completion report

#### Tasks

**Day 22-24: Attack Evaluation**
```bash
# Run comprehensive attack evaluation
python evaluation/benchmark.py --model checkpoints/best_model

# Evaluate:
# - Overall attack success rate (target: <10%)
# - Per-category attack success
# - Per-attack-type success
# - Benign accuracy (target: >95%)
```

**Day 25-26: Baseline Comparison**
```bash
# Generate comparison report
python evaluation/compare_baselines.py --model checkpoints/best_model

# Compare with:
# - No defense: ~87%
# - Input filtering: ~62%
# - StruQ: ~41%
# - SecAlign: ~34%
# - Our method: target <10%
```

**Day 27: Analysis & Documentation**
- Analyze all results
- Identify failure cases
- Document lessons learned
- Create recommendations for Phase 3

**Day 28: Phase 2 Completion**
- Generate final Phase 2 report
- Update PROJECT_STATUS.md
- Plan Phase 3 (Formal Verification)

#### Deliverables
- [x] `WEEK4_EVALUATION_REPORT.md` - Complete evaluation results
- [x] `WEEK4_COMPARISON_TABLE.md` - Baseline comparison
- [x] `PHASE2_COMPLETION_REPORT.md` - Final Phase 2 summary
- [x] Updated `PROJECT_STATUS.md` - Phase 2 marked complete

---

## Success Criteria

### Must Achieve (Required)
- ✅ Training completes without OOM errors
- ✅ Model converges (loss decreases)
- ✅ Checkpoints saved successfully
- ✅ Attack success rate measured

### Target Metrics (Goals)
- 🎯 Attack success rate: <10% (stretch: <5%)
- 🎯 Benign accuracy: >95%
- 🎯 Causal stability: >0.80
- 🎯 Spurious separation: >0.75
- 🎯 ε_causal: <0.10
- 🎯 D-separation: p > 0.05 (independent)
- 🎯 Causal graph: Match score ≥ 0.66

### Quality Indicators
- ✅ Training time: ~2 hours (reasonable)
- ✅ Memory usage: <5.5GB (within limit)
- ✅ No NaN/Inf losses
- ✅ Validation metrics improve over training

---

## Key Commands Reference

### Setup & Verification
```bash
# Week 1: Verify setup
python training/verify_setup.py
python training/optimize_memory.py
python training/dry_run.py

# Week 1: Validate dataset
python data/scripts/test_data_loading.py
python data/scripts/analyze_counterfactuals.py
```

### Training
```bash
# Week 2-3: Training
python training/train.py --config training/config.yaml --debug  # Test
python training/train.py --config training/config.yaml  # Full training

# Resume from checkpoint
python training/train.py --config training/config.yaml --resume checkpoints/epoch_1
```

### Verification
```bash
# Week 3: Causal verification
python verification/independence_tests.py --model checkpoints/best_model
python verification/causal_discovery.py --model checkpoints/best_model
python verification/bounds.py --model checkpoints/best_model
```

### Evaluation
```bash
# Week 4: Evaluation
python evaluation/metrics.py --model checkpoints/best_model
python evaluation/benchmark.py --model checkpoints/best_model
python evaluation/compare_baselines.py --model checkpoints/best_model
```

---

## Monitoring & Debugging

### What to Watch During Training

**Good Signs:**
- ✅ Loss decreases steadily
- ✅ Causal stability increases
- ✅ Spurious separation increases
- ✅ Memory stays <5.5GB
- ✅ Training speed consistent

**Warning Signs:**
- ⚠️ Loss plateaus early (may need more epochs)
- ⚠️ Large train/val gap (overfitting)
- ⚠️ Slow training (<0.3 steps/sec)
- ⚠️ Memory creeping up (watch for leaks)

**Critical Issues:**
- 🚨 OOM errors
- 🚨 NaN/Inf losses
- 🚨 Training crash
- 🚨 Checkpoint save failures

### Troubleshooting

**If OOM Errors:**
1. Reduce `max_seq_length` (2048 → 1024)
2. Reduce `lora.r` (16 → 8)
3. Ensure `gradient_checkpointing: true`
4. Clear CUDA cache: `torch.cuda.empty_cache()`

**If Training Slow:**
1. Check GPU utilization: `nvidia-smi -l 1`
2. Ensure `bf16: true` or `fp16: true`
3. Reduce `dataloader_num_workers` if high CPU usage

**If Loss NaN:**
1. Reduce `learning_rate` (2e-4 → 1e-4)
2. Increase `warmup_ratio` (0.03 → 0.1)
3. Enable `max_grad_norm: 1.0` (gradient clipping)

**If High Attack Success Rate:**
1. Train longer (more epochs)
2. Increase `lambda_causal` and `lambda_spurious` (0.5 → 0.8)
3. Ensure dataset has diverse attack types
4. Check if model is actually learning (loss decreasing?)

---

## Resource Requirements

### Compute
- **GPU:** RTX 4050 (6GB VRAM) or better
- **Training Time:** ~2 hours for 3 epochs
- **Total GPU Hours:** ~3-5 hours (including validation)

### Storage
- **Checkpoints:** ~5GB per epoch × 3 = 15GB
- **Logs:** ~1GB
- **Total:** ~20GB

### Human Time
- **Week 1:** ~5 hours (setup verification)
- **Week 2:** ~3 hours (training monitoring)
- **Week 3:** ~3 hours (verification)
- **Week 4:** ~8 hours (evaluation & documentation)
- **Total:** ~19 hours over 4 weeks

---

## Expected Outputs

By end of Phase 2, you will have:

### Models
- `checkpoints/epoch_1/` - Epoch 1 checkpoint
- `checkpoints/epoch_2/` - Epoch 2 checkpoint
- `checkpoints/epoch_3/` - Epoch 3 checkpoint
- `checkpoints/best_model/` - Best model (by causal_stability)

### Reports
- `WEEK1_SETUP_REPORT.md` - Setup verification
- `WEEK1_DATASET_REPORT.md` - Dataset quality
- `WEEK2_TRAINING_REPORT.md` - Epoch 1 analysis
- `WEEK3_VERIFICATION_REPORT.md` - Causal verification
- `WEEK4_EVALUATION_REPORT.md` - Attack evaluation
- `WEEK4_COMPARISON_TABLE.md` - Baseline comparison
- `PHASE2_COMPLETION_REPORT.md` - Final summary

### Metrics
- Training logs (W&B dashboard)
- Attack success rates by category
- Causal metrics (stability, separation, ε_causal)
- Learned causal graph
- Baseline comparisons

---

## Transition to Phase 3

After Phase 2 completion, you will:

1. **Have a trained model** with empirical validation
2. **Know attack success rates** and how they compare to baselines
3. **Understand causal properties** (d-separation, ε_causal)
4. **Be ready for formal verification** in Phase 3

**Phase 3 Focus:** Prove formal guarantees with complete PAC-Bayesian bounds

---

## Questions & Support

### Quick Reference
- **Setup Issues:** See `training/README.md` Troubleshooting section
- **Training Issues:** See above Troubleshooting section
- **Metric Interpretation:** See `theory/causal_formalization.md` Section 3-4
- **Comparison Context:** See `literature/gaps_analysis.md`

### File Locations
- **Code:** `models/`, `training/`, `verification/`, `evaluation/`
- **Data:** `data/processed/`
- **Config:** `training/config.yaml`
- **Docs:** `theory/`, `literature/`, `training/README.md`

---

**Good luck with Phase 2!** 🚀

Remember:
- Monitor training closely in Week 2
- Don't panic if metrics aren't perfect after Epoch 1
- Document everything for reproducibility
- Iterate if needed (that's research!)
