# Dataset Integration Guide

## Quick Start

### 1. Verify Installation

```bash
# Check that all files are present
ls data/processed/
# Should see:
# - counterfactual_pairs.jsonl
# - train_split.jsonl
# - val_split.jsonl
# - test_split.jsonl
# - dataset_statistics.json
# - validation_report.json
# - examples_preview.txt
```

### 2. Load and Explore

```python
# Run the example usage script
python data/scripts/example_usage.py
```

### 3. Validate Quality

```python
# Run validation
python data/scripts/data_validation.py data/processed/counterfactual_pairs.jsonl
```

---

## Integration Checklist

### ✅ Pre-Training Checklist

- [ ] Dataset loaded successfully (8,939 examples)
- [ ] Train/val/test splits verified (80/10/10)
- [ ] Data format understood (triplet structure)
- [ ] Attack taxonomy reviewed
- [ ] Example usage script tested
- [ ] Validation report reviewed (85.7% pass rate)

### ✅ Model Integration Checklist

- [ ] Tokenizer configured for system + user format
- [ ] Maximum sequence length set appropriately (~512 tokens)
- [ ] Training objective defined (causal intervention or contrastive)
- [ ] Evaluation metrics prepared (attack success rate, benign accuracy)
- [ ] Baseline model performance measured

### ✅ Training Pipeline Checklist

- [ ] DataLoader configured with appropriate batch size
- [ ] Learning rate and optimization schedule set
- [ ] Gradient accumulation configured if needed
- [ ] Validation frequency determined
- [ ] Checkpointing strategy defined
- [ ] Early stopping criteria set

---

## Training Recommendations

### Recommended Hyperparameters

```python
# For causal intervention training
config = {
    'batch_size': 32,
    'learning_rate': 5e-5,
    'epochs': 3,
    'warmup_steps': 500,
    'max_seq_length': 512,
    'gradient_accumulation_steps': 4,
    'weight_decay': 0.01,
    'validation_frequency': 500  # steps
}
```

### Training Objectives

#### Option 1: Triplet Loss (Causal Intervention)
```python
def triplet_loss(anchor, positive, negative, margin=0.5):
    """
    Enforce:
    - Similarity between benign pairs (anchor, positive)
    - Dissimilarity between benign and injection (anchor, negative)
    """
    pos_dist = distance(anchor, positive)
    neg_dist = distance(anchor, negative)
    loss = max(pos_dist - neg_dist + margin, 0)
    return loss
```

#### Option 2: Contrastive Loss
```python
def contrastive_loss(benign_1, benign_2, injection, temperature=0.07):
    """
    InfoNCE-style loss for causal learning
    """
    # Benign pairs should be similar
    pos_sim = cosine_similarity(benign_1, benign_2) / temperature

    # Injection should be dissimilar
    neg_sim = cosine_similarity(benign_1, injection) / temperature

    loss = -log(exp(pos_sim) / (exp(pos_sim) + exp(neg_sim)))
    return loss
```

#### Option 3: Classification (Simple Baseline)
```python
def classification_loss(logits, labels):
    """
    Binary classification: benign (0) vs injection (1)
    """
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    return loss
```

### Evaluation Metrics

```python
def evaluate(model, test_loader):
    metrics = {
        'benign_accuracy': 0,        # Correctly handles benign inputs
        'attack_block_rate': 0,       # Successfully blocks injection attacks
        'false_positive_rate': 0,     # Incorrectly blocks benign inputs
        'attack_success_rate': 0,     # Attacks that succeed (lower is better)
        'overall_safety_score': 0     # Combined metric
    }

    for batch in test_loader:
        # Compute metrics
        pass

    return metrics
```

---

## Advanced Usage Patterns

### 1. Curriculum Learning

Start with easy examples, gradually increase difficulty:

```python
# Sort by difficulty
easy_examples = filter_by_difficulty(train_data, 'easy')
medium_examples = filter_by_difficulty(train_data, 'medium')
hard_examples = filter_by_difficulty(train_data, 'hard')

# Train in stages
train_on(easy_examples, epochs=1)
train_on(medium_examples, epochs=1)
train_on(hard_examples, epochs=1)
```

### 2. Category-Specific Training

Focus on specific use cases:

```python
# Train separate models per category
for category in ['email_assistant', 'rag_qa', 'code_generation']:
    category_data = filter_by_category(train_data, category)
    model = train_category_specific_model(category_data)
    save_model(model, f'model_{category}.pt')
```

### 3. Attack-Type Augmentation

Oversample rare attack types:

```python
from collections import Counter

attack_counts = Counter(ex['attack_type'] for ex in train_data)
min_count = min(attack_counts.values())

# Balance by oversampling rare attacks
balanced_data = []
for attack_type, count in attack_counts.items():
    examples = filter_by_attack_type(train_data, attack_type)
    oversample_factor = min_count / count
    balanced_data.extend(examples * int(1 / oversample_factor))
```

### 4. Multi-Task Learning

Train on multiple objectives simultaneously:

```python
def multi_task_loss(model, batch):
    # Task 1: Causal intervention (triplet loss)
    triplet_loss = compute_triplet_loss(batch)

    # Task 2: Attack classification
    classification_loss = compute_classification_loss(batch)

    # Task 3: Output generation (cross-entropy)
    generation_loss = compute_generation_loss(batch)

    # Combine with learned weights
    total_loss = w1 * triplet_loss + w2 * classification_loss + w3 * generation_loss
    return total_loss
```

---

## Integration with Existing Models

### Hugging Face Transformers

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load base model
model = AutoModelForSequenceClassification.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Prepare dataset
def prepare_batch(examples):
    inputs = []
    labels = []

    for ex in examples:
        triplet = format_triplet(ex)
        for input_text, output_text in triplet:
            encoded = tokenizer(input_text, truncation=True, max_length=512)
            inputs.append(encoded)
            # Label: 0 for benign, 1 for injection
            label = 1 if 'injection' in input_text.lower() else 0
            labels.append(label)

    return inputs, labels

# Train
trainer = Trainer(
    model=model,
    train_dataset=prepare_dataset(train_data),
    eval_dataset=prepare_dataset(val_data),
    # ... other config
)
trainer.train()
```

### LangChain Integration

```python
from langchain import LLMChain, PromptTemplate

# Create safety layer
class SafetyLLMChain(LLMChain):
    def __init__(self, llm, safety_model):
        super().__init__(llm=llm)
        self.safety_model = safety_model

    def run(self, user_input):
        # Check for injection attacks
        is_safe = self.safety_model.predict(user_input)

        if not is_safe:
            return "I cannot process this request as it conflicts with my instructions."

        # Proceed with normal execution
        return super().run(user_input)
```

---

## Testing and Validation

### Unit Tests

```python
import pytest

def test_dataset_loading():
    data = load_dataset('data/processed/train_split.jsonl')
    assert len(data) == 7151

def test_triplet_format():
    data = load_dataset('data/processed/train_split.jsonl')
    example = data[0]
    triplet = format_triplet(example)
    assert len(triplet) == 3  # Three pairs

def test_filtering():
    data = load_dataset('data/processed/train_split.jsonl')
    email_data = filter_by_category(data, 'email_assistant')
    assert all(ex['task_category'] == 'email_assistant' for ex in email_data)
```

### Integration Tests

```python
def test_model_robustness():
    model = load_trained_model()

    # Test on benign inputs
    benign_examples = filter_by_difficulty(test_data, 'easy')
    benign_accuracy = evaluate_benign(model, benign_examples)
    assert benign_accuracy > 0.95

    # Test on injection attacks
    injection_examples = filter_by_attack_type(test_data, 'instruction_override')
    block_rate = evaluate_attack_blocking(model, injection_examples)
    assert block_rate > 0.90
```

---

## Performance Optimization

### Data Loading Optimization

```python
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Use multiple workers
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster CPU-to-GPU transfer
    prefetch_factor=2       # Pre-load next batches
)
```

### Memory Optimization

```python
# Gradient accumulation for large models
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        loss = model(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Dataset Loading Errors
```python
# Solution: Ensure UTF-8 encoding
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)
```

#### Issue 2: Memory Issues
```python
# Solution: Reduce batch size or use gradient accumulation
config = {
    'batch_size': 16,  # Reduced from 32
    'gradient_accumulation_steps': 8  # Effective batch size = 128
}
```

#### Issue 3: Overfitting
```python
# Solution: Add regularization and augmentation
config = {
    'weight_decay': 0.01,
    'dropout': 0.1,
    'data_augmentation': True,
    'early_stopping_patience': 3
}
```

#### Issue 4: Poor Attack Blocking
```python
# Solution: Increase attack example weight
def weighted_loss(loss, example):
    if example['attack_type'] in ['instruction_override', 'indirect_injection']:
        return loss * 2.0  # Increase weight for critical attacks
    return loss
```

---

## Monitoring and Logging

### Training Metrics to Track

```python
import wandb  # or tensorboard

wandb.init(project="llm-safety")

for epoch in range(num_epochs):
    for batch in train_loader:
        loss = train_step(batch)

        # Log metrics
        wandb.log({
            'train_loss': loss,
            'benign_accuracy': benign_acc,
            'attack_block_rate': block_rate,
            'learning_rate': lr,
            'epoch': epoch
        })
```

### Attack Analysis Dashboard

```python
def analyze_failures(model, test_data):
    failures = {
        'by_attack_type': defaultdict(list),
        'by_difficulty': defaultdict(list),
        'by_category': defaultdict(list)
    }

    for example in test_data:
        prediction = model.predict(example)
        if not prediction_is_safe(prediction):
            failures['by_attack_type'][example['attack_type']].append(example)
            failures['by_difficulty'][example['difficulty']].append(example)
            failures['by_category'][example['task_category']].append(example)

    return failures
```

---

## Production Deployment

### Safety Layer Architecture

```python
class SafetyLayer:
    def __init__(self, detection_model, response_generator):
        self.detector = detection_model
        self.generator = response_generator

    def process(self, system_instruction, user_input):
        # Step 1: Detect injection attacks
        is_injection = self.detector.predict(user_input)

        if is_injection:
            # Step 2: Generate safe refusal
            return self.generator.generate_refusal(system_instruction)

        # Step 3: Normal processing
        return self.generator.generate_response(system_instruction, user_input)
```

### API Integration

```python
from fastapi import FastAPI

app = FastAPI()
safety_layer = SafetyLayer(detection_model, response_generator)

@app.post("/chat")
async def chat(request: ChatRequest):
    response = safety_layer.process(
        system_instruction=request.system,
        user_input=request.user_message
    )

    return {"response": response, "safe": True}
```

---

## Next Steps

1. **Review Documentation**
   - Read `data/README.md` for detailed information
   - Study `data/DATASET_SUMMARY.md` for statistics

2. **Run Example Scripts**
   - Execute `example_usage.py` to see data loading
   - Run validation to ensure quality

3. **Design Training Pipeline**
   - Choose training objective (triplet, contrastive, classification)
   - Set hyperparameters based on recommendations
   - Implement evaluation metrics

4. **Start Training**
   - Begin with easy examples (curriculum learning)
   - Monitor metrics closely
   - Iterate on model architecture

5. **Evaluate and Deploy**
   - Test on held-out examples
   - Measure attack blocking rate
   - Deploy with safety layer

---

## Support and Resources

- **Documentation**: `data/README.md`
- **Schema**: `data/schemas/dataset_schema.json`
- **Validation**: `data/processed/validation_report.json`
- **Examples**: `data/processed/examples_preview.txt`

For questions or issues, refer to the project documentation or contact the research team.

---

**Last Updated**: October 12, 2025
**Version**: 1.0
