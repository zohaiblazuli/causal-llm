#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
mkdir -p logs

ts=$(date +%Y%m%d_%H%M%S)
python -u training/train.py --config training/config.5090.yaml --no-wandb 2>&1 | tee "logs/train_${ts}.log"

