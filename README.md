# Provably Safe LLM Agents via Causal Intervention

ISEF 2025 Research Project

## Overview

This project applies causal inference theory to defend against prompt injection attacks in Large Language Models (LLMs). We use causal intervention techniques to create LLM agents that are provably robust against adversarial attacks while maintaining performance on legitimate tasks.

## Key Innovation

Instead of pattern-matching defenses that fail on novel attacks, we:
1. Formalize LLM security using Structural Causal Models (SCMs)
2. Apply do-calculus to define intervention-based robustness
3. Train models using causal contrastive learning
4. Prove formal generalization bounds using PAC-Bayesian theory

## Project Structure

```
isef/
├── data/                      # Dataset and data generation
│   ├── raw/                   # Raw training examples
│   ├── processed/             # Processed counterfactual pairs
│   └── scripts/               # Data generation scripts
├── models/                    # Model implementations
│   ├── causal_model.py        # Main causal intervention model
│   ├── losses.py              # Causal contrastive loss
│   └── utils.py               # Model utilities
├── training/                  # Training pipeline
│   ├── train.py               # Main training script
│   ├── config.py              # Training configuration
│   └── callbacks.py           # Training callbacks
├── evaluation/                # Evaluation and benchmarking
│   ├── metrics.py             # Evaluation metrics
│   ├── attacks.py             # Attack implementations
│   └── benchmark.py           # Benchmark suite
├── verification/              # Formal verification
│   ├── causal_discovery.py    # PC/GES algorithms
│   ├── independence_tests.py  # HSIC and d-separation tests
│   └── bounds.py              # PAC-Bayesian bounds
├── theory/                    # Theoretical foundations
│   ├── causal_formalization.md
│   ├── key_contributions_summary.md
│   ├── open_questions.md
│   └── implementation_roadmap.md
├── literature/                # Literature review
│   ├── review.md
│   ├── references.bib
│   ├── gaps_analysis.md
│   └── summary.md
├── experiments/               # Experiment logs and results
│   ├── logs/
│   └── results/
├── demo/                      # Interactive demo (Phase 6)
│   └── app.py
├── paper/                     # Research paper (Phase 5)
│   └── draft/
├── tests/                     # Unit and integration tests
│   ├── test_models.py
│   ├── test_training.py
│   └── test_verification.py
├── PROJECT_STATUS.md          # Central project tracking
├── MILESTONES.md              # Detailed milestones
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Timeline

- **Phase 1 (Dec 2024)**: Foundation & Theory ✓
- **Phase 2 (Jan 2025)**: Core Implementation
- **Phase 3 (Feb 2025)**: Formal Verification
- **Phase 4 (Mar 2025)**: Evaluation & Benchmarking
- **Phase 5 (Apr 2025)**: Extensions & Paper Writing
- **Phase 6 (May 2025)**: Demo & Presentation

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

```bash
# Generate training data
python data/scripts/generate_counterfactuals.py

# Train causal model
python training/train.py --config training/config.yaml

# Evaluate on benchmark
python evaluation/benchmark.py --model-path checkpoints/best_model

# Run verification tests
python verification/causal_discovery.py --model-path checkpoints/best_model
```

## Hardware Requirements

- **Minimum**: RTX 4050 (6GB VRAM) with 4-bit quantization + LoRA
- **Recommended**: RTX 4090 (24GB VRAM) or cloud GPU (Lambda Labs, vast.ai)
- **CPU**: 8+ cores recommended for data generation
- **RAM**: 16GB minimum, 32GB recommended

## Key Results (Target)

| Metric | Target | Baseline (DefenseX) |
|--------|--------|---------------------|
| Attack Success Rate | <5% | 34% |
| Benign Accuracy | >95% | 91% |
| Novel Attack Transfer | <10% | 58% |

| Latency Overhead | <50ms | N/A |

## Citation

```bibtex
@misc{isef2025_causal_llm,
  title={Provably Safe LLM Agents via Causal Intervention},
  author={[Your Name]},
  year={2025},
  note={ISEF 2025 Project}
}
```

## License

MIT License (to be confirmed)

## Contact

[Your Email] | [ISEF Project Page]

## Acknowledgments

- Theoretical foundations based on Pearl's causal inference framework
- Builds on prior work in causal robustness (Zhang et al., ICLR)
- Thanks to mentors and advisors (TBD)
