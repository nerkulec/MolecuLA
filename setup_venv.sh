#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOLECULA_VENV="${MOLECULA_VENV:-$REPO_DIR/molecula-venv}"

cd "$REPO_DIR"
if [[ ! -x "$MOLECULA_VENV/bin/python" ]]; then
  python -m venv "$MOLECULA_VENV"
fi

"$MOLECULA_VENV/bin/python" -m pip install --upgrade pip wheel
"$MOLECULA_VENV/bin/python" -m pip install -r requirements.txt
"$MOLECULA_VENV/bin/python" - <<'PY'
import rdkit
import selfies
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_devices={torch.cuda.device_count()}")
print(f"rdkit={rdkit.__version__}")
print(f"selfies={selfies.__version__}")
PY

echo "Environment ready: $MOLECULA_VENV"
