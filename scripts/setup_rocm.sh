#!/usr/bin/env bash
# ROCm setup script for jazz-piano (the original/local environment).
#
# Creates a venv that *inherits* the system's ROCm-built PyTorch via
# --system-site-packages, plus the project's Python dependencies on top.
#
# Run this INSIDE the mlbox distrobox:
#   mlbox bash scripts/setup_rocm.sh
#
# Or, if you have a ROCm host without mlbox, run it directly. The script
# only requires that `python3 -c 'import torch; torch.cuda.is_available()'
# already works in the parent environment.
#
# Usage:
#   bash scripts/setup_rocm.sh
#   SKIP_SMOKE=1 bash scripts/setup_rocm.sh
#
# Exit codes:
#   0 on success
#   non-zero otherwise

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
PY=${PYTHON:-python3}
VENV="${VENV_DIR:-.venv}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

echo "== jazz-piano ROCm setup =="
echo "  REPO_ROOT = $REPO_ROOT"
echo "  PYTHON    = $($PY --version 2>&1)"
echo "  VENV      = $VENV"

# ---- Sanity: parent env exposes a ROCm-built torch ----
if ! "$PY" -c '
import sys, importlib
try:
    t = importlib.import_module("torch")
except Exception as e:
    print(f"  parent env has no torch: {e}", file=sys.stderr); sys.exit(2)
hip = getattr(t.version, "hip", None)
if not hip:
    print(f"  parent env has torch {t.__version__} but no HIP — wrong env?", file=sys.stderr)
    sys.exit(3)
print(f"  parent torch={t.__version__} hip={hip} cuda.is_available={t.cuda.is_available()}")
' 2>&1; then
  echo "!! Parent environment does not look like a ROCm PyTorch install."
  echo "   Activate mlbox first:  mlbox bash scripts/setup_rocm.sh"
  exit 1
fi

# ---- Create venv with --system-site-packages (inherits ROCm torch) ----
if [ ! -d "$VENV" ]; then
  echo
  echo "-- Creating venv at $VENV with --system-site-packages"
  "$PY" -m venv --system-site-packages "$VENV"
else
  echo
  echo "-- Reusing existing venv at $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Confirm the venv sees the inherited torch.
python -c '
import torch
print(f"  venv torch={torch.__version__} hip={torch.version.hip} cuda_ok={torch.cuda.is_available()}")
'

python -m pip install --upgrade pip wheel setuptools

# ---- Project requirements (no torch in here — venv inherits it from system) ----
echo
echo "-- Installing project requirements"
python -m pip install -r requirements.txt

# ---- Smoke test ----
if [ "$SKIP_SMOKE" = "1" ]; then
  echo "-- Skipping smoke test (SKIP_SMOKE=1)"
else
  echo
  echo "-- Running smoke test"
  python scripts/check_env.py
fi

echo
echo "== Setup complete =="
echo "Activate with:  source $VENV/bin/activate"
