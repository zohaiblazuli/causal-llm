#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install -r requirements.txt

echo "Setup complete. Activate with: source .venv/bin/activate"

