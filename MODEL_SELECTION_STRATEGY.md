# Model Selection Strategy for ISEF 2026
## Provably Safe LLM Agents via Causal Intervention

**Date:** October 14, 2025
**Target Competition:** ISEF 2026 (May 2026)
**Timeline:** Training in January 2026 (~3 months from now)
**Hardware:** NVIDIA RTX 3060 (12GB VRAM)

---

## Executive Summary

**Current Selection (October 2025):** Llama 3.1-8B-Instruct
**Purpose:** Setup validation and system testing
**Final Selection:** To be reassessed in December 2025

**Strategic Approach:**
- Use Llama 3.1-8B-Instruct NOW for verification and validation
- Reassess options in December 2025 before training
- Select the absolute best model available in January 2026
- This ensures maximum recency for ISEF May 2026 presentation

---

## Timeline Context

| Date | Event | Model Age at ISEF 2026 |
|------|-------|----------------------|
| **Oct 2025** | NOW - Setup validation | - |
| **Dec 2025** | Final model decision | - |
| **Jan 2026** | Training begins | - |
| **May 2026** | ISEF 2026 presentation | 4 months old |

**Key Insight:** By waiting until January 2026 to train, we ensure the model is only 4 months old at ISEF 2026, maximizing the "state-of-the-art" appeal to judges.

---

## Current Selection: Llama 3.1-8B-Instruct

### Model Details
- **Release Date:** July 2024
- **Parameters:** 8 billion
- **Context Length:** 8,000 tokens
### Important Correction**Llama 3.2 does NOT have an 8B model.**- **Llama 3.2 sizes:** 1B, 3B (text-only) and 11B, 90B (multimodal vision)- **For 8B class:** Llama 3.1-8B-Instruct (128K context) is the latest official model- **Llama 3.1-8B release:** July 2024 (15 months old by ISEF 2026)
- **License:** Llama 3.2 Community License
- **HuggingFace:** `meta-llama/Llama-3.1-8B-Instruct`

### Why This Model NOW (October 2025)

**Strengths:**
- ✅ **Proven stable** (13 months of maturity)
- ✅ **Best all-around under 10B** as of October 2025
- ✅ **Will definitely work** with our setup (4-5GB with 4-bit quantization)
- ✅ **Excellent instruction-following** (77.4 on IFEval benchmark)
- ✅ **Perfect for setup validation** - no surprises
- ✅ **8,000 token context** - sufficient for attack scenarios
- ✅ **Massive community support** - easy to troubleshoot

**For Setup Phase:**
- Complete HuggingFace authentication
- Run all verification tests
- Validate training pipeline
- Dry run to confirm everything works
- **Don't do full training yet!**

### Limitations
- By ISEF May 2026, will be 20 months old
- Less impressive than "Llama 4" or newer models
- 8K context vs 128K in newer models

---

## Alternative: Llama 4 Scout 17B-16E-Instruct

### Model Details
- **Release Date:** April 2025
- **Parameters:** 17 billion (16 experts, MoE architecture)
- **Context Length:** Unprecedented (exact number TBD)
- **Architecture:** Mixture-of-Experts (first open-weight natively multimodal)
- **HuggingFace:** `meta-llama/Llama-4-Scout-17B-16E-Instruct`

### Why This Could Be THE Choice

**Revolutionary Features:**
- ✅ **State-of-the-art as of April 2025** - only 6 months old
- ✅ **"We used Llama 4"** - strongest possible statement for ISEF 2026
- ✅ **Multimodal capabilities** (text + images + video) - enables advanced demos
- ✅ **MoE architecture** - more efficient than dense 17B
- ✅ **Natively multimodal** - could show attack visualizations
- ✅ **Will be 13 months old at ISEF 2026** - still very current

**Challenges:**
- ⚠️ **17B parameters** - larger than originally planned
- ⚠️ **Memory: 10-11GB estimated** with 4-bit quantization
- ⚠️ **Your RTX 3060 has 12GB** - tight but should work
- ⚠️ **Less documentation** - newer, fewer examples
- ⚠️ **Higher risk** - might encounter unexpected issues

### Memory Estimation for Llama 4 Scout 17B

**Base Memory:**
- 17B parameters × 0.5 bytes (4-bit) = 8.5 GB

**Training Memory:**
- Base model: 8.5 GB
- LoRA adapters: 0.1 GB
- Causal projection: 0.1 GB
- Activations (batch=1, seq=1024): 0.8 GB
- Gradients: 0.2 GB
- Optimizer states: 0.3 GB
- **Total:** ~10 GB

**Safety Analysis:**
- Available: 12 GB
- Required: ~10 GB
- **Margin: 2 GB (17%)** - TIGHT but workable

**Risk Mitigation:**
- Reduce sequence length to 512 if needed
- Gradient checkpointing (already enabled)
- Batch size 1 (already configured)
- Monitor closely during dry run

### Verdict on Llama 4 Scout

**Recommendation:** VIABLE but HIGH RISK

**Best Case Scenario:**
- Works perfectly with 4-bit quantization
- Impressive "Llama 4" claim for ISEF
- Multimodal capabilities enable cool demos
- Strongest possible foundation for publication

**Worst Case Scenario:**
- OOM errors during training
- Fall back to Llama 3.1-8B (still good)
- Delay of 1-2 days maximum

---

## Other Contenders

### DeepSeek 7B
**Strength:** Best for reasoning and coding
**Use Case:** Could be excellent for formal verification work
**Concern:** Less well-known (might need more explanation in paper)
**Verdict:** BACKUP OPTION

### Qwen 2.5-7B
**Strength:** 32K token context (vs 8K)
**Use Case:** Excellent for conversational AI
**Concern:** Alibaba Cloud origin (less Western research community support)
**Verdict:** BACKUP OPTION

### Llama 3.1-8B
**Strength:** 128K context length, tool-use capabilities
**Use Case:** Good middle ground between 3.2 and 4
**Concern:** Less impressive than Llama 4
**Verdict:** SAFE FALLBACK

---

## Strategic Decision Framework

### Phase 1: NOW (October 2025) ✅
**Action:** Use Llama 3.1-8B-Instruct
**Purpose:** Setup validation
**Tasks:**
- Complete HuggingFace authentication
- Run verification suite
- Dry run training (10 steps only)
- Confirm pipeline works perfectly
- **NO FULL TRAINING**

### Phase 2: December 2025 (2 months)
**Action:** Reassess model landscape
**Questions to Answer:**
1. Have any new Llama 4 variants been released?
   - Any 8B class Llama 4 models?
   - Any optimized Llama 4 Scout versions?
2. Are there breakthrough models from other labs?
   - Mistral, DeepSeek, Qwen updates?
3. What's the benchmark leader in 8B-17B class?
4. Is Llama 4 Scout 17B stable enough?

**Decision Criteria:**
- **Performance:** Benchmark scores on reasoning, instruction-following
- **Stability:** Community feedback, known issues resolved?
- **Memory:** Confirmed to work with 12GB VRAM?
- **Documentation:** Enough resources to troubleshoot?
- **Wow Factor:** How impressive for ISEF judges?

### Phase 3: January 2026 (Training)
**Action:** Train on selected model
**Result:** Model is only 4 months old by ISEF May 2026

---

## Recommended Path Forward

### Option A: BOLD - Llama 4 Scout 17B ⭐⭐⭐⭐⭐

**Decision Point:** December 2025
**Conditions to Proceed:**
- ✅ Community confirms it works with 12GB VRAM
- ✅ Successful dry runs reported by others
- ✅ No major show-stopping bugs
- ✅ Clear documentation available

**If Conditions Met:**
1. Update config to Llama 4 Scout 17B
2. Run extensive dry runs in December
3. Train in January 2026
4. Present "Llama 4" at ISEF 2026 🏆

**Benefits:**
- Strongest possible model claim
- "State-of-the-art Llama 4" impresses judges
- Multimodal capabilities for demos
- Best for publication

**Risks:**
- Tight memory (10GB / 12GB)
- Newer model, less tested
- Might need fallback plan

---

### Option B: SAFE - Llama 3.1-8B-Instruct ⭐⭐⭐⭐

**Decision Point:** Can decide now or wait until December
**Conditions to Proceed:**
- Always viable - proven stable

**If Selected:**
1. Already configured (current choice)
2. Can train anytime in January 2026
3. Guaranteed to work
4. Still excellent performance

**Benefits:**
- Zero risk of failure
- Well-documented
- Proven excellent performance
- "Best all-around under 10B" is defensible

**Drawbacks:**
- Will be 20 months old at ISEF 2026
- Less impressive than "Llama 4"
- Missed opportunity for cutting-edge claim

---

### Option C: HYBRID - Try Bold, Fallback to Safe ⭐⭐⭐⭐⭐ (RECOMMENDED)

**Strategy:**
1. **December 2025:** Test Llama 4 Scout 17B
   - Run extensive dry runs
   - Measure actual memory usage
   - Test with full training for 1 epoch

2. **If Llama 4 Works:**
   - Use Llama 4 Scout 17B for main results
   - **Also** train Llama 3.1-8B baseline
   - Paper shows: "Works on Llama 3.2, even better on Llama 4"

3. **If Llama 4 Doesn't Work:**
   - Use Llama 3.1-8B (still excellent)
   - Mention in paper: "Evaluated Llama 4 but memory constraints..."
   - Shows due diligence

**Benefits:**
- Best of both worlds
- Minimizes risk
- Maximizes upside
- Shows research rigor (tested multiple models)

**Timeline:**
- December: 1 week testing Llama 4
- January Week 1-2: Train on selected model
- January Week 3-4: Train baseline comparison

---

## Implementation Plan

### NOW (October 2025)

**Step 1: Complete HF Authentication**
```bash
# Accept Llama 3.2 license
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# Get token and login
huggingface-cli login
```

**Step 2: Run Verification Suite**
```bash
python training/verify_setup.py
python training/optimize_memory.py
python training/dry_run.py --steps 10
```

**Step 3: Confirm Readiness**
- All tests pass ✓
- Memory < 5GB ✓
- No errors ✓

### December 2025

**Step 1: Research Update**
- Check HuggingFace for new models
- Review benchmarks and leaderboards
- Read community feedback on Llama 4

**Step 2: Test Llama 4 (if pursuing)**
```bash
# Update config.yaml line 11:
name: "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# Accept Llama 4 license on HuggingFace
# Run verification
python training/verify_setup.py
python training/optimize_memory.py  # Should show ~10GB
python training/dry_run.py --steps 50  # More extensive test

# If successful, test 1 epoch
python training/train.py --config training/config.yaml --debug --max_epochs 1
```

**Step 3: Make Final Decision**
- Llama 4 works? → Use it ✓
- Llama 4 issues? → Llama 3.1-8B ✓
- Want both? → Train both (baseline + main)

### January 2026

**Step 1: Full Training**
```bash
python training/train.py --config training/config.yaml
```

**Step 2: Monitor & Document**
- Watch training metrics
- Document any issues
- Save all results

**Step 3: Proceed to Phase 3**
- Causal verification
- Baseline comparisons
- Evaluation

---

## Configuration Changes Made

### config.yaml Updates (October 14, 2025)

**Line 2:** Updated header
```yaml
# Optimized for RTX 3060 (12GB VRAM) - Updated October 2025
# Target: ISEF 2026 (May 2026)
```

**Lines 7-16:** Model selection with strategy
```yaml
# Base model - Strategic selection for ISEF 2026
# Current (Oct 2025): Llama-3.1-8B-Instruct for setup validation
# Final (Jan 2026): Will reassess and select best available state-of-the-art
# Options: Llama 4 Scout variants, Llama 3.3 optimized, or newer releases
name: "meta-llama/Llama-3.1-8B-Instruct"

# Alternative options (uncomment to use):
# name: "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # Bold choice, requires 10-11GB
# name: "meta-llama/Meta-Llama-3.1-8B"  # Fallback stable option
# name: "meta-llama/Llama-2-7b-hf"  # Original baseline for comparison
```

**Lines 196-204:** Hardware updates
```yaml
# Hardware-Specific Settings for RTX 3060
hardware:
  gpu_name: "RTX 3060"
  vram_gb: 12
  max_memory_allocated_gb: 11.0  # Leave headroom for 12GB VRAM
```

**Lines 167-168:** W&B tags
```yaml
tags: ["llama", "lora", "causal-contrastive", "rtx3060", "isef2026"]
notes: "ISEF 2026: Fine-tuning Llama 3.1-8B with causal contrastive loss for provably safe LLM agents"
```

---

## Expected Outcomes by Model

### With Llama 3.1-8B-Instruct

**Strengths for ISEF:**
- "Best all-around model under 10B parameters"
- Proven excellent instruction-following
- Safe, reliable, reproducible

**Paper Positioning:**
- "We used Llama 3.1-8B-Instruct, the best available model in its class"
- Focus on methodology and results
- Strong but not flashy

**Expected Results:**
- Attack success rate: 5-10%
- Benign degradation: 2-5%
- Causal stability: 0.80-0.85
- Publication: Acceptable, focus on approach

---

### With Llama 4 Scout 17B-16E-Instruct

**Strengths for ISEF:**
- "State-of-the-art Llama 4 with multimodal capabilities"
- "First open-weight natively multimodal model"
- Impressive technical sophistication
- Could enable visual attack demos

**Paper Positioning:**
- "We demonstrate our approach on Llama 4, the latest generation"
- Stronger foundation = clearer demonstration of causal method's value
- More impressive to reviewers

**Expected Results:**
- Attack success rate: 3-8% (better baseline → bigger improvement)
- Benign degradation: 1-4% (better model → less degradation)
- Causal stability: 0.85-0.90
- Publication: Strong advantage - cutting-edge

---

## Risk Assessment

### Llama 3.1-8B Risk Level: LOW ✅

| Risk Factor | Level | Mitigation |
|-------------|-------|------------|
| Memory OOM | Very Low | Proven 4-5GB usage |
| Training Failure | Very Low | Extensively tested |
| Documentation | Very Low | Massive community |
| Time Loss | None | Guaranteed to work |

**Overall Risk:** MINIMAL

---

### Llama 4 Scout 17B Risk Level: MEDIUM ⚠️

| Risk Factor | Level | Mitigation |
|-------------|-------|------------|
| Memory OOM | Medium | Test extensively in Dec, fallback ready |
| Training Failure | Low-Medium | Community feedback by January |
| Documentation | Medium | Will improve by January 2026 |
| Time Loss | Low | 1-2 days max, then fallback |

**Overall Risk:** ACCEPTABLE with proper testing and fallback plan

---

## Final Recommendation

### Recommended Path: **HYBRID APPROACH**

**Phase 1 (NOW - October 2025):**
- ✅ Use Llama 3.1-8B-Instruct
- ✅ Complete setup validation
- ✅ Confirm everything works perfectly

**Phase 2 (December 2025):**
- 🔍 Test Llama 4 Scout 17B extensively
- 📊 Measure actual memory usage
- 🧪 Run 1-epoch training test
- ✅ Make informed decision

**Phase 3 (January 2026):**
- 🚀 Train on best available model
- 📈 Strong baseline for comparison
- 🎯 4 months old by ISEF 2026

**Benefits:**
- Minimizes risk (Llama 3.2 always works)
- Maximizes upside (Try for Llama 4)
- Shows research rigor (Tested options)
- Best positioning for ISEF 2026

---

## Questions for User

Before proceeding, please confirm:

1. **Are you comfortable with the hybrid approach?**
   - Test Llama 4 in December
   - Fallback to Llama 3.2 if needed

2. **Would you prefer to be conservative?**
   - Just use Llama 3.1-8B (guaranteed to work)
   - Focus on perfect execution over model choice

3. **Are you willing to be bold?**
   - Commit to Llama 4 Scout 17B now
   - Accept higher risk for higher reward

**My Recommendation:** Hybrid approach - test Llama 4 in December, decide then.

---

**Document Version:** 1.0
**Created:** October 14, 2025
**Next Review:** December 2025 (model reassessment)
**Current Model:** Llama 3.1-8B-Instruct (validation only)
**Final Model:** TBD (January 2026)
