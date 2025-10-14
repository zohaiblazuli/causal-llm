# Phase 2 Pre-Launch Checklist
## ISEF 2025: Provably Safe LLM Agents via Causal Intervention

**Date:** October 14, 2025
**Phase:** Phase 2 Week 1 - Pre-Training Validation
**Status:** 95% READY - Only HF authentication needed

---

## Executive Summary

**System Status: EXCELLENT**
- All code fixes complete (verified in existing code)
- Dependencies installed (peft, bitsandbytes, accelerate, psutil)
- CUDA PyTorch 2.7.1+cu118 working
- GPU detected: NVIDIA RTX 3060 (12GB VRAM)
- HuggingFace authentication (user action required)

**Estimated Time to Training:** 20-25 minutes (5 min HF auth + 15-20 min verification)

---

## Pre-Launch Checklist

### COMPLETED
- [x] All 5 critical code fixes (already in codebase)
- [x] Python dependencies installed
- [x] CUDA PyTorch installed (2.7.1+cu118)
- [x] GPU detection confirmed (RTX 3060 12GB)
- [x] Training scripts validated
- [x] Dataset available (8,939 examples)

### PENDING (User Action Required)

#### Step 1: HuggingFace Authentication (5 minutes)

**Why Required:**
- Need to download Llama 2 7B base model
- Model is gated behind license agreement
- One-time setup

**Instructions:**

1. **Accept Llama 2 License:**
   - Go to: https://huggingface.co/meta-llama/Llama-2-7b-hf
   - Click "Agree and access repository"
   - Wait for approval (usually instant)

2. **Create Access Token:**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name: "isef-training" (or any name)
   - Permission: "Read" (that's all you need)
   - Click "Generate token"
   - **IMPORTANT:** Copy the token immediately (starts with `hf_`)

3. **Login to HuggingFace:**
   ```bash
   huggingface-cli login
   # OR
   hf auth login
   ```
   - Paste your token when prompted
   - Answer 'n' to git credential question

**Verification:**
```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); print('HF Login: SUCCESS')"
```

**Expected Output:** `HF Login: SUCCESS`

**Troubleshooting:**
- Token doesn't work? Make sure you copied the ENTIRE token
- "Bad Request" error? Accept Llama 2 license first
- Command not found? Run: `pip install huggingface-hub`

---

### VERIFICATION SUITE (After HF Login - 15-20 minutes)

#### Test 1: CUDA Verification (30 seconds)
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA Version: {torch.version.cuda}')"
```

**Expected Output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060
CUDA Version: 11.8
```

**Status:** [ ] PASS [ ] FAIL

---

#### Test 2: Training Setup Verification (1-2 minutes)
```bash
python training/verify_setup.py
```

**Expected:** 8-9/9 checks pass (all except maybe W&B)

**Critical Checks:**
- Python version >= 3.8
- CUDA available
- GPU detected
- All dependencies installed
- HuggingFace token valid
- Model access granted
- Data files present
- Config valid
- Disk space sufficient

**Status:** [ ] PASS (8+/9) [ ] FAIL

---

#### Test 3: Data Loading Test (1-2 minutes)
```bash
python training/test_data_pipeline.py
```

**Expected:**
- Loads all 8,939 examples
- Validates triplet structure
- Tests DataLoader with num_workers=2
- Shows batches/sec performance

**Status:** [ ] PASS [ ] FAIL

---

#### Test 4: Memory Optimization Check (1 minute)
```bash
python training/optimize_memory.py
```

**Expected Output:**
```
Estimated Memory Usage:
  Base Model (4-bit): 3.50 GB
  LoRA Parameters: 0.05 GB
  Causal Projection: 0.07 GB
  Activations: 0.60 GB
  Gradients: 0.12 GB
  Optimizer: 0.10 GB
  Total: 4.44 GB

VRAM Available: 12.00 GB
Safety Margin: 7.56 GB (63%)

Status: SAFE - Plenty of memory headroom
```

**Status:** [ ] PASS [ ] FAIL

---

#### Test 5: Dry Run (MOST IMPORTANT - 2-3 minutes)
```bash
python training/dry_run.py --steps 10
```

**Expected Behavior:**
- Loads model successfully
- Runs 10 training steps
- No crashes or OOM errors
- Memory stays <6GB
- Loss is finite (no NaN)
- Steps complete in 1-2 seconds each

**Expected Output:**
```
Loading model: meta-llama/Llama-2-7b-hf
Model loaded successfully
LoRA applied (52M trainable parameters)
Causal projection added

Running 10-step dry run...
Step 1/10: Loss: 9.234 | Memory: 4.2 GB
Step 2/10: Loss: 8.891 | Memory: 4.3 GB
...
Step 10/10: Loss: 7.456 | Memory: 4.4 GB

Dry run SUCCESSFUL!
No crashes
Memory within limits
Loss is finite
Ready for training
```

**Status:** [ ] PASS [ ] FAIL

---

## Final GO/NO-GO Decision

### GO Criteria (All Must Be TRUE):
- [ ] HuggingFace authentication complete
- [ ] CUDA verification passed
- [ ] Training setup verification passed (8+/9)
- [ ] Data loading test passed
- [ ] Memory check shows SAFE status
- [ ] Dry run completed without errors

### If ALL checks pass: GO FOR TRAINING

### If ANY check fails: NO-GO
**Troubleshooting Actions:**
1. Review error messages carefully
2. Check CHANGES_LOG.md for known issues
3. Consult CONTINUATION_STATUS.md for status
4. See NEXT_STEPS.md for troubleshooting guide

---

## Training Launch Procedure

### Once All Checks Pass:

**Command:**
```bash
python training/train.py --config training/config.yaml
```

**Monitoring:**
- Watch first 10 steps carefully for errors
- Check memory usage (should stay 4-5GB)
- Verify loss decreases (from ~10 to ~8 in first epoch)
- Training speed should be 0.5-1.0 steps/sec

**Expected Duration:** ~2 hours for 3 epochs on RTX 3060

**Completion:** Training saves checkpoints to `checkpoints/` directory

---

## Expected Training Metrics

**Epoch 1:**
- Loss: ~10 → ~7
- Causal Stability: ~0.3 → ~0.6
- Memory: 4-5GB steady

**Epoch 2:**
- Loss: ~7 → ~4
- Causal Stability: ~0.6 → ~0.75

**Epoch 3:**
- Loss: ~4 → ~2.5
- Causal Stability: ~0.75 → ~0.85

**Final Targets:**
- Attack success rate: <10% (target <5%)
- Benign accuracy: >95%
- Causal stability: >0.80

---

## Troubleshooting Guide

### HuggingFace Login Issues
**Problem:** Token doesn't work
**Solution:**
1. Make sure you accepted Llama 2 license first
2. Copy ENTIRE token (starts with `hf_`)
3. Try `hf auth login` instead of `huggingface-cli login`

### CUDA Not Available
**Problem:** `torch.cuda.is_available()` returns False
**Solution:**
1. Verify CUDA PyTorch installed: `pip show torch` (should show +cu118)
2. Reinstall if needed: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Out of Memory
**Problem:** CUDA OOM during dry run
**Solution:**
1. Close other GPU applications
2. Run `nvidia-smi` to check GPU usage
3. Reduce max_seq_length in config.yaml to 768 if needed

### Training Very Slow
**Problem:** <0.3 steps/sec
**Solution:**
1. Verify CUDA is being used (check verification test)
2. Check GPU utilization with `nvidia-smi`
3. Ensure no other programs using GPU

---

## Success Confirmation

### You're ready to train when you see:
- All verification tests passed
- Dry run completed successfully
- Memory well within limits (4-5GB / 12GB)
- No error messages or warnings

### Expected Timeline from Here:
- **Now:** Complete HF authentication (5 min)
- **Next:** Run verification suite (15-20 min)
- **Then:** Launch training (2 hours)
- **Result:** Trained model ready for Phase 2 Week 2 evaluation

---

**Document Version:** 1.0
**Created:** 2025-10-14
**Status:** Ready for user HF authentication
**Next Step:** Complete Step 1 (HF login) then run verification suite
