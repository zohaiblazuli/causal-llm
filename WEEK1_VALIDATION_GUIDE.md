# Week 1 Validation Scripts - Phase 2
**Training Readiness Verification System**

## Overview

This Week 1 validation system consists of 4 comprehensive scripts designed to verify all systems are ready before beginning full-scale training in January 2025. Each script performs critical checks to ensure hardware, software, data, and training configurations are properly set up.

---

## Quick Start

Run all validation scripts in sequence:

```bash
# 1. Setup Verification (5 min)
python training/verify_setup.py

# 2. Memory Optimization (10-15 min)
python training/optimize_memory.py

# 3. Data Pipeline Test (5 min)
python training/test_data_pipeline.py

# 4. Dry Run Training (10-15 min)
python training/dry_run.py --steps 10 --num-samples 100
```

**Total estimated time:** ~40-50 minutes

---

## Script Details

### 1. Enhanced `training/verify_setup.py`
**Purpose:** Comprehensive 9-point system check

**What it checks:**
1. Python version (>=3.8)
2. All dependencies (torch, transformers, peft, bitsandbytes, etc.)
3. CUDA availability and version
4. GPU memory (6GB available on RTX 4050)
5. Hugging Face token configured
6. Model access (Llama 2 7B or equivalent)
7. Data files exist and readable (train/val/test splits)
8. Config file valid (`config.yaml`)
9. Disk space (>50GB free)

**Output format:**
```
=== WEEK 1 SETUP VERIFICATION - PHASE 2 ===

✓ 1. Python version (>=3.8): Python 3.10.5
✓ 2. All dependencies: PyTorch 2.1.0, Transformers 4.35.0, etc.
✓ 3. CUDA availability: CUDA 11.8
✓ 4. GPU Memory: NVIDIA GeForce RTX 4050 (6GB)
✓ 5. HF Token: Configured
✓ 6. Model access: meta-llama/Llama-2-7b-hf
✓ 7. Data files: train (7.6MB), val (973KB), test (966KB)
✓ 8. Config valid: training/config.yaml
✓ 9. Disk space: 124GB free

RESULT: ALL 9 CHECKS PASSED ✓
Ready to proceed to memory optimization!
```

**Usage:**
```bash
python training/verify_setup.py
```

**Expected duration:** ~5 minutes

---

### 2. Enhanced `training/optimize_memory.py`
**Purpose:** Detailed memory profiling and optimization recommendations

**What it tests:**
1. Base model loading (4-bit quantized) memory
2. LoRA adapter memory overhead
3. Forward pass activation memory
4. Gradient and optimizer memory
5. Full training step peak memory
6. Provides safety margin analysis
7. Recommends configuration adjustments

**Output format:**
```
=== DETAILED MEMORY OPTIMIZATION REPORT ===

STEP 1: Model Loading (4-bit quantized)
  Base model: 3.45 GB
  LoRA adapters: 0.05 GB

STEP 2: Forward Pass
  Activation memory: 1.52 GB

STEP 3: Full Training Step
  Peak memory: 5.45 GB

=== MEMORY BREAKDOWN ===

  Base Model (4-bit quantized):      3.45 GB
  LoRA Adapters (rank 16):           0.05 GB
  Activations (seq_len 2048):        1.52 GB
  Gradients:                         0.05 GB
  Optimizer State (8-bit):           0.10 GB
  Cache/Buffers:                     0.28 GB
  --------------------------------------------
  TOTAL PEAK MEMORY:                 5.45 GB / 6.00 GB

MARGIN:                              0.55 GB (9%)
STATUS:                              ⚠️ TIGHT (recommend reducing to 1024 seq_len)

RECOMMENDATIONS:
1. Current config will work but has minimal margin
2. Consider reducing max_seq_length to 1024 for safety
3. With seq_len=1024: estimated 4.44 GB (26% margin) ✓
4. Monitor memory during training (may need adjustment)

VERDICT: READY TO TRAIN (with caution) ✓
```

**Usage:**
```bash
# Full analysis (recommended)
python training/optimize_memory.py

# Test specific batch sizes
python training/optimize_memory.py --test-batch-sizes

# Test specific sequence lengths
python training/optimize_memory.py --test-sequence-lengths
```

**Expected duration:** ~10-15 minutes

---

### 3. Enhanced `training/test_data_pipeline.py`
**Purpose:** End-to-end data loading pipeline validation

**What it tests:**
1. Loading all splits (train/val/test)
2. File reading and JSON parsing
3. Triplet extraction (benign, benign_cf, injection)
4. Tokenization with proper formatting
5. Batch collation with padding
6. DataLoader with multi-processing
7. Data integrity checks

**Output format:**
```
=== WEEK 1 DATA PIPELINE TEST - PHASE 2 ===

TEST 0: Loading All Splits
  ✓ train: 5000 examples (7.6MB)
  ✓ val: 625 examples (973KB)
  ✓ test: 625 examples (966KB)

TEST 1: File Reading and Parsing
  ✓ Successfully parsed 10/10 samples

TEST 2: Tokenization
  Benign tokens: 145
  Benign CF tokens: 142
  Injection tokens: 158
  ✓ Tokenization successful

TEST 3: Dataset Loading
  ✓ Dataset loaded in 0.52 seconds
  Dataset size: 10
  ✓ Dataset loading successful

TEST 4: Data Collator
  ✓ Batch created successfully
  Batch shape: [2, 2048]

TEST 5: DataLoader Performance
  ✓ DataLoader iteration successful
  Speed: 2.3 batches/sec

TEST 6: Data Integrity
  ✓ All 10 samples are valid

=== DATA PIPELINE TEST SUMMARY ===
RESULT: 7/7 TESTS PASSED

DATA PIPELINE READY ✓
```

**Usage:**
```bash
# Default test
python training/test_data_pipeline.py

# Custom parameters
python training/test_data_pipeline.py \
    --num-samples 50 \
    --batch-size 4 \
    --max-length 1024
```

**Expected duration:** ~5 minutes

---

### 4. Enhanced `training/dry_run.py`
**Purpose:** Full training simulation with 10 steps

**What it tests:**
1. Model loading (4-bit quantization + LoRA)
2. Data loading (100 samples)
3. Loss function computation
4. Forward pass through model
5. Backward pass and gradients
6. Optimizer step
7. Memory management
8. Loss convergence (should decrease)
9. Checkpoint saving
10. Checkpoint loading

**Output format:**
```
=== WEEK 1 DRY RUN: 10 TRAINING STEPS - PHASE 2 ===

STEP 1: Loading model (4-bit + LoRA)
  ✓ Model loaded
  Trainable params: 4,194,304 (0.06% of 7B)

STEP 2: Loading data (100 samples)
  ✓ Data loaded: 100 samples

STEP 3: Running 10 training steps

Training steps: 100%|████████| 10/10 [01:23<00:00, 8.3s/it]
  Step 1/10: loss=2.3456, time=8.21s, mem=5.12GB
  Step 2/10: loss=2.2834, time=8.15s, mem=5.15GB
  ...
  Step 10/10: loss=1.9823, time=8.18s, mem=5.14GB

=== DRY RUN COMPLETE ✓ ===

Steps completed: 10/10
Average loss: 2.1245
Loss trend: decreasing ✓
Peak memory: 5.15 GB / 6.00 GB
Memory status: SAFE ✓
Average step time: 8.18s
Checkpoint: saved and verified ✓

=== TEST RESULTS ===
  ✓ PASS: Model loading (4-bit + LoRA)
  ✓ PASS: Data loading (100 samples)
  ✓ PASS: Training steps completed
  ✓ PASS: Loss values valid
  ✓ PASS: Loss decreasing trend
  ✓ PASS: Training health check
  ✓ PASS: Memory under limit (<5.8GB)
  ✓ PASS: Checkpoint save/load

RESULT: 8/8 TESTS PASSED

=== READY FOR FULL TRAINING ✓ ===

Expected full training time:
  Estimated: ~12.5 hours for 3 epochs
  Per epoch: ~4.2 hours

Next step: Run full training with:
  python training/train.py --config training/config.yaml
```

**Usage:**
```bash
# Standard dry run (10 steps, 100 samples)
python training/dry_run.py

# Custom configuration
python training/dry_run.py \
    --steps 20 \
    --num-samples 200 \
    --config training/config.yaml
```

**Expected duration:** ~10-15 minutes

---

## Week 1 Setup Report Template

After running all scripts, fill out the comprehensive report:

**File:** `WEEK1_SETUP_REPORT_TEMPLATE.md`

This template includes:
- Executive summary with all 4 stages
- Detailed results from each script
- Memory breakdown tables
- Training metrics and loss trajectory
- Critical issues and action items
- Final approval checklist
- Sign-off section

**How to use:**
1. Run all 4 validation scripts
2. Copy `WEEK1_SETUP_REPORT_TEMPLATE.md` to a new file with date:
   ```bash
   cp WEEK1_SETUP_REPORT_TEMPLATE.md WEEK1_SETUP_REPORT_2025-01-XX.md
   ```
3. Fill in all sections with results from each script
4. Check all boxes in the Final Approval section
5. Get sign-off before proceeding to full training

---

## Success Criteria

### All scripts must:
- ✅ Run without errors
- ✅ Provide clear pass/fail status for each check
- ✅ Give actionable error messages with fix suggestions
- ✅ Complete in reasonable time (<10 min each)
- ✅ Generate comprehensive reports

### Training readiness requirements:
- ✅ All 9 setup checks passed
- ✅ Memory peak < 5.8GB (ideally < 5.5GB)
- ✅ All data splits loaded successfully
- ✅ Data pipeline tests all pass
- ✅ Dry run completes 10 steps
- ✅ Loss decreases during dry run
- ✅ Checkpoint save/load works
- ✅ No critical errors or warnings

---

## Troubleshooting Common Issues

### Issue: CUDA not available
**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of memory during optimization
**Solution:**
1. Reduce `max_seq_length` in `config.yaml` (try 1024 or 768)
2. Ensure `per_device_train_batch_size = 1`
3. Increase `gradient_accumulation_steps` to 16 or 32
4. Reduce LoRA `r` from 16 to 8

### Issue: Missing Hugging Face token
**Solution:**
```bash
# Login to Hugging Face
huggingface-cli login

# Or set token in environment
export HF_TOKEN=your_token_here
```

### Issue: Data files not found
**Solution:**
1. Verify data directory exists: `ls data/processed/`
2. Run data generation: `python data/generate_counterfactuals.py`
3. Check file permissions

### Issue: Slow data loading
**Solution:**
1. Reduce `dataloader_num_workers` in config
2. Use faster storage (SSD instead of HDD)
3. Reduce `num_samples` for testing

---

## Configuration Recommendations

### For RTX 4050 (6GB VRAM):

**Safe configuration:**
```yaml
model:
  max_seq_length: 1024  # Reduced from 2048

lora:
  r: 16  # Balanced rank
  alpha: 32

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  bf16: true
```

**Aggressive configuration (if safe config works well):**
```yaml
model:
  max_seq_length: 2048

lora:
  r: 32  # Higher capacity
  alpha: 64

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  bf16: true
```

---

## Files Created/Modified

### New Files:
1. `WEEK1_SETUP_REPORT_TEMPLATE.md` - Comprehensive report template
2. `WEEK1_VALIDATION_GUIDE.md` - This guide

### Enhanced Files:
1. `training/verify_setup.py` - 9-point check system
2. `training/optimize_memory.py` - Detailed memory analysis
3. `training/test_data_pipeline.py` - Complete pipeline validation
4. `training/dry_run.py` - 10-step training test

All scripts now include:
- Clear pass/fail indicators (✓ / ✗)
- Detailed progress output
- Actionable error messages
- Comprehensive summaries
- Time estimates
- Safety margins

---

## Next Steps After Validation

Once all 4 scripts pass:

1. **Week 2: Initial Training**
   - Run 1 epoch with full training data
   - Monitor loss, memory, and speed
   - Validate checkpointing works
   - Assess model convergence

2. **Week 3-4: Full Training**
   - Train for 3 full epochs
   - Monitor W&B dashboard
   - Save best checkpoints
   - Validate on test set

3. **Week 5: Evaluation**
   - Run comprehensive evaluation
   - Test attack success rates
   - Analyze causal metrics
   - Generate sample outputs

---

## Support and Resources

### Documentation:
- Main project README: `README.md`
- Training guide: `training/README.md`
- Config documentation: `training/config.yaml` (comments)
- Theory documentation: `theory/causal_formalization.md`

### Key Configuration File:
- `training/config.yaml` - All training hyperparameters

### Log Output:
All scripts output to console. To save logs:
```bash
python training/verify_setup.py 2>&1 | tee logs/setup_verification.log
python training/optimize_memory.py 2>&1 | tee logs/memory_optimization.log
python training/test_data_pipeline.py 2>&1 | tee logs/data_pipeline_test.log
python training/dry_run.py 2>&1 | tee logs/dry_run.log
```

---

## Contact

For issues or questions during Week 1 validation:
1. Check this guide first
2. Review script output for specific error messages
3. Consult troubleshooting section
4. Review configuration recommendations

---

**Document Version:** 1.0
**Last Updated:** 2025-10-13
**Phase:** 2 - Week 1 - Training Readiness Validation
**Status:** Ready for use in January 2025
