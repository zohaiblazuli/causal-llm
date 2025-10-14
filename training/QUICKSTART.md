# Quick Start Guide - Causal LLM Training

Get your model training in 5 minutes on RTX 4050 (6GB VRAM).

## Prerequisites

- RTX 4050 GPU with 6GB VRAM
- Python 3.8+
- 50GB free disk space

## Step-by-Step Setup

### 1. Install Dependencies (2 minutes)

```bash
# Install all required packages
pip install -r requirements.txt

# If bitsandbytes installation fails on Windows:
pip install bitsandbytes-windows
```

### 2. Get Model Access (1 minute)

```bash
# Login to Hugging Face
huggingface-cli login

# Accept Llama 2 license
# Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf
# Click "Access repository" and accept license
```

### 3. Verify Setup (1 minute)

```bash
# Run verification script
python training/verify_setup.py
```

**Expected output:**
```
✓ All checks passed! Ready to start training.
```

If any checks fail, follow the error messages to fix issues.

### 4. Test Memory Configuration (1 minute)

```bash
# Profile memory usage
python training/optimize_memory.py --config training/config.yaml
```

**Expected output:**
```
Peak memory: ~5.2 GB
✓ Configuration fits in RTX 4050 (6GB)
```

If memory test fails:
- Edit `training/config.yaml`
- Reduce `max_seq_length` from 2048 to 1024
- Run optimization again

### 5. Start Training!

```bash
# Full training (when data is ready)
python training/train.py --config training/config.yaml

# Debug mode (test with small dataset first)
python training/train.py --config training/config.yaml --debug
```

## What Happens During Training

1. **Model Loading** (~30 seconds)
   - Downloads Llama 2 7B if not cached
   - Applies 4-bit quantization
   - Adds LoRA adapters

2. **Data Loading** (~10 seconds)
   - Loads counterfactual triplets
   - Tokenizes inputs
   - Creates batches

3. **Training Loop** (~30 minutes per epoch)
   - Forward pass on 3 inputs per sample
   - Computes causal contrastive loss
   - Updates LoRA parameters only
   - Validates every 200 steps
   - Saves best checkpoint

4. **Final Model** (~10 seconds)
   - Saves LoRA weights
   - Ready for inference

## Expected Timeline

For 10,000 training samples:
- **Epoch 1**: 30-40 minutes
- **Epoch 2**: 30-40 minutes
- **Epoch 3**: 30-40 minutes
- **Total**: 1.5-2 hours

## Monitoring Training

### Console Output
```
Epoch 1/3
Step 100: Loss: 2.345 | LR: 2.00e-04 | Causal: 0.723 | Spurious: 0.812
Memory - Allocated: 5.12GB
```

### Weights & Biases (Recommended)
```bash
# Login once
wandb login

# View at: https://wandb.ai/your-username/isef-causal-llm
```

## When Things Go Wrong

### Out of Memory
```bash
# Edit config.yaml
max_seq_length: 1024  # reduce from 2048

# Or reduce LoRA rank
lora:
  r: 8  # reduce from 16
```

### Slow Training (< 0.5 steps/sec)
```bash
# Check GPU utilization
nvidia-smi

# Ensure mixed precision is enabled in config.yaml
bf16: true
```

### Loss is NaN
```bash
# Reduce learning rate in config.yaml
learning_rate: 1.0e-4  # reduce from 2.0e-4
```

## After Training

Your trained model will be at:
```
checkpoints/best_model/
├── adapter_config.json
├── adapter_model.bin
└── tokenizer files
```

Use it with:
```python
from models.causal_model import CausalLLMModel

model = CausalLLMModel.from_pretrained("checkpoints/best_model")
output = model.generate(
    system_instruction="You are a helpful assistant.",
    user_input="What is AI?"
)
```

## Next Steps

1. **Evaluate Model**: Use evaluation scripts to test attack robustness
2. **Fine-tune Hyperparameters**: Adjust learning rate, loss weights
3. **Scale Up**: Train on larger datasets or longer sequences
4. **Deploy**: Integrate into your application

## Getting Help

- **Documentation**: `training/README.md`
- **Memory Issues**: Run `python training/optimize_memory.py`
- **Code Issues**: Check `training/verify_setup.py`
- **Questions**: Open an issue on GitHub

## Troubleshooting Checklist

Before asking for help, verify:

- [ ] `python training/verify_setup.py` passes all checks
- [ ] `python training/optimize_memory.py` shows config fits in memory
- [ ] CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Data files exist in `data/processed/`
- [ ] Hugging Face login is active: `huggingface-cli whoami`
- [ ] Config file is valid: `python -c "from training.utils import load_config; load_config('training/config.yaml')"`

## Advanced Options

### Resume from Checkpoint
```bash
python training/train.py --resume checkpoints/checkpoint-500
```

### Train Without W&B
```bash
python training/train.py --no-wandb
```

### Profile Memory During Training
```bash
python training/train.py --profile-memory
```

### Custom Output Directory
```bash
python training/train.py --output-dir my_experiments/run1
```

## Configuration Cheat Sheet

Quick edits to `training/config.yaml`:

**Reduce memory usage:**
```yaml
max_seq_length: 1024      # from 2048
lora.r: 8                 # from 16
```

**Speed up training:**
```yaml
gradient_accumulation_steps: 8   # from 16 (less accurate)
eval_steps: 500                  # from 200 (less frequent validation)
```

**Improve quality:**
```yaml
lora.r: 32                       # from 16 (more capacity)
learning_rate: 1.0e-4            # from 2.0e-4 (more stable)
num_epochs: 5                    # from 3 (more training)
```

**Debug faster:**
```yaml
debug:
  enabled: true
  max_train_samples: 100
```

## Success Criteria

After training, your model should achieve:
- ✓ Causal Stability > 0.80
- ✓ Spurious Separation > 0.75
- ✓ Attack Success Rate < 10%

Check with:
```bash
python evaluation/test_model.py --model checkpoints/best_model
```

---

**Ready to train?** Run `python training/verify_setup.py` and follow the next steps!
