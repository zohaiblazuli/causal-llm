#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python - << 'PY'
import torch
from training.verify_setup import main as verify
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
ok = verify()
raise SystemExit(0 if ok else 1)
PY

