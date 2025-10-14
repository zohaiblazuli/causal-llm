# CRITICAL FIXES CHECKLIST
## Must Complete Before Training

**Estimated Time:** 2-3 hours
**Priority:** CRITICAL - Training will fail without these fixes

---

## Fix 1: Model Architecture Integration (60 minutes)

**Problem:** Trainer expects model with `return_representation=True` but uses raw PEFT model

**Files to Modify:**
- `training/trainer.py` (lines 266-285, 389-405)

**Fix:**
```python
# In trainer.py, replace the forward pass sections with:

# For benign input (around line 266):
outputs = self.model(
    input_ids=batch["benign_input_ids"],
    attention_mask=batch["benign_attention_mask"],
    output_hidden_states=True
)
hidden_states = outputs.hidden_states[-1]
# Pool with attention mask
mask_expanded = batch["benign_attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
# Apply projection
representation = self.model.causal_projection(pooled)
benign_outputs = {
    "logits": outputs.logits,
    "representation": representation
}

# Repeat for benign_cf and injection (around lines 273-285)
```

**Test:**
```bash
python training/dry_run.py --steps 5
```

**Expected:** No crashes, forward pass completes successfully

---

## Fix 2: Checkpoint Save/Load for Causal Projection (30 minutes)

**Problem:** causal_projection weights not saved/loaded in checkpoints

**Files to Modify:**
- `training/callbacks.py` (add after line 198)
- `training/trainer.py` (modify lines 497-516)

**Fix in callbacks.py:**
```python
# In ModelCheckpoint._save_checkpoint(), add after line 198:
if hasattr(trainer.model, 'causal_projection'):
    torch.save(
        trainer.model.causal_projection.state_dict(),
        checkpoint_path / "causal_projection.pt"
    )
```

**Fix in trainer.py:**
```python
# In load_checkpoint(), add after line 498:
if hasattr(self.model, 'causal_projection'):
    projection_path = checkpoint_path / "causal_projection.pt"
    if projection_path.exists():
        self.model.causal_projection.load_state_dict(
            torch.load(projection_path, map_location=self.device)
        )
        print("Loaded causal projection from checkpoint")
```

**Test:**
```python
# Test save/load:
from training.train import setup_model
from pathlib import Path

model, tokenizer = setup_model(config)
# Save
Path("test_checkpoint").mkdir(exist_ok=True)
model.save_pretrained("test_checkpoint")
torch.save(model.causal_projection.state_dict(), "test_checkpoint/causal_projection.pt")

# Load
model2, _ = setup_model(config)
model2.causal_projection.load_state_dict(
    torch.load("test_checkpoint/causal_projection.pt")
)
print("Save/load test passed!")
```

---

## Fix 3: Reduce Sequence Length (5 minutes)

**Problem:** max_seq_length=2048 will likely OOM on 6GB VRAM

**File to Modify:**
- `training/config.yaml` (lines 20, 134)

**Fix:**
```yaml
# Line 20:
max_seq_length: 1024  # Reduced from 2048

# Line 134:
max_length: 1024  # Reduced from 2048
```

**Memory Impact:**
- Before: ~5.4 GB (tight)
- After: ~4.4 GB (safe margin of 1.6 GB)

---

## Fix 4: Add MLP Targets to LoRA (5 minutes)

**Problem:** Missing gate_proj, up_proj, down_proj from LoRA targets

**File to Modify:**
- `training/config.yaml` (lines 30-37)

**Fix:**
```yaml
target_modules:
  - "q_proj"
  - "v_proj"
  - "k_proj"
  - "o_proj"
  - "gate_proj"   # ADD
  - "up_proj"     # ADD
  - "down_proj"   # ADD
```

**Impact:** 40% more trainable capacity, better adaptation

---

## Fix 5: Causal Projection Device Placement (5 minutes)

**Problem:** Projection layer may not be on correct device

**File to Modify:**
- `training/train.py` (line 173)

**Fix:**
```python
# Replace line 173 with:
device = next(model.parameters()).device
model.causal_projection = torch.nn.Sequential(
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.LayerNorm(hidden_size),
    torch.nn.GELU(),
    torch.nn.Linear(hidden_size, hidden_size)
).to(device)
print(f"Causal projection moved to device: {device}")
```

---

## Verification Checklist

After completing all fixes, run these tests:

### 1. Import Test
```bash
python -c "from training.train import setup_model; from training.trainer import CausalTrainer; print('Imports OK')"
```
Expected: "Imports OK"

### 2. Model Setup Test
```bash
python -c "
from training.train import setup_model
from training.utils import load_config
config = load_config('training/config.yaml')
model, tokenizer = setup_model(config)
print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
print(f'Causal projection on device: {next(model.causal_projection.parameters()).device}')
"
```
Expected: Shows parameter count and device

### 3. Forward Pass Test
```bash
python -c "
import torch
from training.train import setup_model
from training.utils import load_config

config = load_config('training/config.yaml')
model, tokenizer = setup_model(config)

# Test input
input_ids = torch.randint(0, 1000, (1, 128)).to(next(model.parameters()).device)
attention_mask = torch.ones_like(input_ids)

# Forward pass
outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
hidden_states = outputs.hidden_states[-1]
pooled = hidden_states.mean(dim=1)
representation = model.causal_projection(pooled)

print(f'Forward pass successful!')
print(f'Representation shape: {representation.shape}')
print(f'Representation device: {representation.device}')
"
```
Expected: Shows shape (1, 4096) and device

### 4. Dry Run Test (Most Important)
```bash
python training/train.py --config training/config.yaml --debug --no-wandb
```
Expected: Trains for 100 samples without crashing

### 5. Memory Test
```bash
python -c "
import torch
from training.train import setup_model
from training.utils import load_config, get_memory_usage

config = load_config('training/config.yaml')
print('Before model:', get_memory_usage())

model, tokenizer = setup_model(config)
print('After model:', get_memory_usage())

# Simulate training step
input_ids = torch.randint(0, 1000, (1, 1024)).cuda()
attention_mask = torch.ones_like(input_ids)

outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
representation = model.causal_projection(outputs.hidden_states[-1].mean(dim=1))

print('After forward:', get_memory_usage())
print('Peak memory should be < 5.5 GB')
"
```
Expected: Peak memory < 5.5 GB

---

## Success Criteria

All fixes are successful if:
- ✓ All 5 verification tests pass
- ✓ Dry run completes without errors
- ✓ Peak memory < 5.5 GB
- ✓ Forward pass produces valid representations
- ✓ Checkpoint save/load works

---

## Common Issues and Solutions

### Issue: "AttributeError: 'PeftModel' object has no attribute 'causal_projection'"
**Solution:** Ensure causal_projection is added in setup_model() and properly moved to device

### Issue: "RuntimeError: Expected all tensors to be on the same device"
**Solution:** Check that causal_projection is on CUDA: `model.causal_projection.to('cuda')`

### Issue: "CUDA out of memory"
**Solution:** Reduce max_seq_length further to 768 or 512

### Issue: "KeyError: 'hidden_states'"
**Solution:** Ensure `output_hidden_states=True` is passed to model forward

---

## After Fixes: Start Training

Once all verifications pass:

```bash
# Full training run
python training/train.py \
    --config training/config.yaml \
    --output-dir checkpoints/causal_llm_v1 \
    --no-wandb
```

Expected training time: 20-30 minutes for 3 epochs

Monitor:
- Memory usage stays < 5.5 GB
- Loss decreases (should drop from ~10 to ~2-3)
- Steps/sec around 0.5-0.8
- No NaN or Inf in losses

---

**Last Updated:** 2025-10-13
**Estimated Completion Time:** 2-3 hours
**Difficulty:** Medium
**Risk if Skipped:** Training will crash immediately
