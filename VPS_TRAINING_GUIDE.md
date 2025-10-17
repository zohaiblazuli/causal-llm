# VPS Training Guide - ISEF 2026 Causal LLM

**GitHub Repository**: https://github.com/zohaiblazuli/causal-llm

---

## ✅ PUSHED TO GITHUB - READY FOR VPS

Your project has been successfully pushed to GitHub with all the optimizations for training.

---

## 📋 VPS Requirements

### Recommended GPU:
- **A100 (40GB)**: Best choice - ~10-15 minutes total training ✅
- **V100 (32GB)**: Good - ~15-20 minutes
- **A10G (24GB)**: Adequate - ~20-25 minutes
- **RTX 4090 (24GB)**: Good - ~15-20 minutes

### Minimum Specs:
- GPU VRAM: 16GB+ (for comfortable training)
- RAM: 32GB+
- Disk: 50GB (for model cache + checkpoints)
- CUDA: 11.8+

### Cost Estimate:
- **RunPod**: ~$0.50-1.00/hour (A100)
- **Lambda Labs**: ~$1.10/hour (A100)
- **Vast.ai**: ~$0.30-0.80/hour (varies)
- **Colab Pro+**: $50/month (A100 access)

**Total cost for this training**: ~$0.25-0.50 (15-20 min on A100)

---

## 🚀 QUICK START ON VPS

### Step 1: Clone Repository (2 min)

```bash
# SSH into your VPS
git clone https://github.com/zohaiblazuli/causal-llm.git
cd causal-llm

# Check GPU
nvidia-smi
```

### Step 2: Setup Environment (5 min)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 3: HuggingFace Authentication (2 min)

```bash
# Login to HuggingFace
huggingface-cli login

# Paste your token from: https://huggingface.co/settings/tokens
# Accept Llama 3.2 license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

### Step 4: Start Training (10-20 min)

```bash
# Run training
python training/train.py --config training/config.yaml

# Or with logging
python training/train.py --config training/config.yaml 2>&1 | tee logs/training_vps_$(date +%Y%m%d_%H%M%S).log
```

### Step 5: Download Results (2 min)

```bash
# After training completes, download the model
# From your local machine:
scp -r user@vps:/path/to/causal-llm/checkpoints/best_model ./checkpoints/
```

---

## 📊 Expected Training Performance

### On A100 (40GB):
- **Sequence length**: Can increase to 1024 (better quality!)
- **Batch size**: Can increase to 4 (faster training!)
- **Gradient accumulation**: Can reduce to 4
- **Total time**: ~10-15 minutes for 3 epochs
- **Memory usage**: ~8-12GB / 40GB

### On V100 (32GB):
- Keep current config (seq=512, batch=1, grad_accum=8)
- **Total time**: ~15-20 minutes
- **Memory usage**: ~6-8GB / 32GB

### On RTX 4090 (24GB):
- Can increase seq_len to 768
- **Total time**: ~15-18 minutes
- **Memory usage**: ~10-12GB / 24GB

---

## 🔧 CONFIG OPTIMIZATION FOR HIGH-END GPU

If you have an A100 or similar, update `training/config.yaml`:

```yaml
# Optimize for A100 (40GB VRAM)
model:
  max_seq_length: 1024  # Increase from 512

training:
  per_device_train_batch_size: 2  # Increase from 1
  gradient_accumulation_steps: 4  # Reduce from 8

data:
  max_length: 1024  # Increase from 512
```

This will give:
- ✅ Better quality (longer context)
- ✅ Faster training (larger batch)
- ✅ Still safe memory usage

---

## 📝 TRAINING COMMAND REFERENCE

### Basic Training:
```bash
python training/train.py --config training/config.yaml
```

### With W&B Logging (recommended):
```bash
# First: pip install wandb
# Then: wandb login

python training/train.py --config training/config.yaml
```

### Resume from Checkpoint:
```bash
python training/train.py --config training/config.yaml --resume checkpoints/epoch-1
```

### Debug Mode (quick test):
```bash
python training/train.py --config training/config.yaml --debug
```

---

## ✅ VERIFICATION CHECKLIST

Before starting training:
- [ ] GPU detected: `nvidia-smi` shows your GPU
- [ ] CUDA working: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] HF logged in: `huggingface-cli whoami`
- [ ] Dataset present: `ls data/processed/*.jsonl`
- [ ] Config valid: `cat training/config.yaml`

After training:
- [ ] Training completed: "TRAINING COMPLETE!" message
- [ ] Model saved: `ls checkpoints/best_model/`
- [ ] Metrics look good: loss <3.0, causal_stability >0.75

---

## 🎯 WHAT TO EXPECT

### Console Output:
```
================================================================================
CAUSAL LLM TRAINING - OPTIMIZED
================================================================================

Loading model: meta-llama/Llama-3.2-3B-Instruct...
✓ Model loaded
Trainable params: 20,971,520 (0.62%)

Train dataset: 7151 samples
Val dataset: 893 samples

Starting training for 3 epochs
Total training steps: 894

Epoch 1/3: [████████████████████] 100% | loss=2.543, causal=0.67
Epoch 2/3: [████████████████████] 100% | loss=1.821, causal=0.78
Epoch 3/3: [████████████████████] 100% | loss=1.342, causal=0.83

✓ Best model saved to: checkpoints/best_model
Training complete!
```

### Timeline (A100):
- **0:00** - Model loading
- **0:02** - Data loading
- **0:03** - Epoch 1 starts
- **0:08** - Epoch 1 complete, validation
- **0:09** - Epoch 2 starts
- **0:13** - Epoch 2 complete, validation
- **0:14** - Epoch 3 starts
- **0:18** - **TRAINING COMPLETE!**

---

## 📁 IMPORTANT FILES

### Configuration:
- `training/config.yaml` - Main training config (optimized for RTX 3060, but upgradeable)

### Data:
- `data/processed/train_split.jsonl` - 7,151 training examples
- `data/processed/val_split.jsonl` - 893 validation examples
- `data/processed/test_split.jsonl` - 895 test examples

### Code:
- `training/train.py` - Main training script
- `models/causal_model.py` - Model architecture
- `models/losses.py` - Causal contrastive loss

### Outputs:
- `checkpoints/best_model/` - Trained model (created after training)
- `logs/` - Training logs

---

## 🐛 TROUBLESHOOTING

### Issue: "CUDA Out of Memory"
**Solution**:
```bash
# Reduce sequence length in config.yaml
max_seq_length: 384  # From 512
```

### Issue: "HuggingFace Access Denied"
**Solution**:
```bash
# Go to https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
# Click "Agree and access repository"
# Then: huggingface-cli login
```

### Issue: "ImportError: No module named 'peft'"
**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: Training very slow (>1 min per epoch)
**Check**:
```bash
# Verify GPU is being used
nvidia-smi  # Should show Python process using GPU

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

---

## 📊 AFTER TRAINING - EVALUATION

### Run Evaluation Suite:
```bash
# Evaluate on test set
python evaluation/benchmark.py --model checkpoints/best_model

# Run causal verification
python verification/independence_tests.py --model checkpoints/best_model

# Measure attack success rates
python evaluation/metrics.py --model checkpoints/best_model
```

### Expected Results:
- **Final Loss**: 1.5-2.5 (lower is better)
- **Causal Stability**: >0.75 (target >0.80)
- **Attack Success Rate**: <20% (target <10%)
- **Benign Accuracy**: >90%

---

## 💾 DOWNLOAD MODEL TO LOCAL MACHINE

### Option 1: SCP (Secure Copy)
```bash
# From your local machine
scp -r user@vps-ip:/path/to/causal-llm/checkpoints/best_model ./checkpoints/
```

### Option 2: Git LFS (for sharing)
```bash
# On VPS
git lfs install
git lfs track "checkpoints/best_model/*.safetensors"
git add .gitattributes checkpoints/best_model/
git commit -m "Add trained model"
git push

# On local machine
git pull
git lfs pull
```

### Option 3: Cloud Storage
```bash
# Upload to Google Drive, Dropbox, etc.
# Or use wget/curl from local machine
```

---

## 🎓 CURRENT PROJECT STATUS

### Completed:
- ✅ Phase 1: Theory & Dataset (100%)
- ✅ Code implementation (all modules ready)
- ✅ Configuration optimized for training
- ✅ Pushed to GitHub

### Next Steps (After Training):
1. **Phase 2 Week 3**: Causal verification
   - Run HSIC independence tests
   - Measure ε_causal
   - Validate d-separation

2. **Phase 2 Week 4**: Evaluation & benchmarking
   - Test attack success rates
   - Compare to baselines
   - Generate Phase 2 completion report

3. **Phase 3 (Feb 2025)**: Formal verification
   - Prove theoretical bounds
   - Complete PAC-Bayes proof
   - Write paper draft

---

## 📞 SUPPORT

### If Training Fails:
1. Check the error message carefully
2. Look in `logs/` for detailed output
3. Try dry run: `python training/dry_run.py --steps 10`
4. Check GPU memory: `nvidia-smi`

### Key Metrics to Watch:
- ✅ Loss decreasing (should go from ~8 → ~2)
- ✅ Causal stability increasing (should reach >0.75)
- ✅ Memory usage stable (<80% of total)
- ✅ No NaN/Inf values

---

## 🎯 SUCCESS CRITERIA

**Training is successful if:**
- All 3 epochs complete without errors
- Final loss < 3.0
- Causal stability > 0.75
- Model saves to `checkpoints/best_model/`
- No OOM errors or crashes

**You'll know it's working when:**
- Progress bar moves smoothly
- Loss decreases each epoch
- GPU utilization at 90-100%
- Memory stays constant (no leaks)

---

## 📚 REPOSITORY STRUCTURE

```
causal-llm/
├── data/               # 8,939 training examples ✅
├── models/             # Model architecture ✅
├── training/           # Training pipeline ✅
├── evaluation/         # Metrics & benchmarking ✅
├── verification/       # Causal tests ✅
├── theory/             # Mathematical foundations ✅
├── literature/         # Research papers ✅
├── checkpoints/        # Model outputs (created after training)
└── logs/               # Training logs
```

---

**GitHub**: https://github.com/zohaiblazuli/causal-llm
**Ready to train**: Just clone and run!
**Expected time**: 10-20 minutes on A100
**Cost**: ~$0.25-0.50

**Good luck with VPS training! 🚀**
