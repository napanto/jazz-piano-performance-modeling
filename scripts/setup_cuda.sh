#!/usr/bin/env bash
# CUDA setup script for jazz-piano.
#
# Use this on a fresh CUDA-capable GPU host. It creates a clean venv,
# installs PyTorch with the requested CUDA wheel, installs the project's
# Python requirements, and runs the smoke test.
#
# Usage:
#   bash scripts/setup_cuda.sh                 # default cu124 + torch 2.5
#   CUDA_WHEEL=cu126 bash scripts/setup_cuda.sh
#   TORCH_VERSION=2.6.0 bash scripts/setup_cuda.sh
#
# Exit codes:
#   0 on success (venv ready + smoke test green)
#   non-zero otherwise

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
PY=${PYTHON:-python3}
VENV="${VENV_DIR:-.venv}"
CUDA_WHEEL="${CUDA_WHEEL:-cu124}"            # cu121, cu124, cu126
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

echo "== jazz-piano CUDA setup =="
echo "  REPO_ROOT     = $REPO_ROOT"
echo "  PYTHON        = $($PY --version 2>&1)"
echo "  VENV          = $VENV"
echo "  TORCH_VERSION = $TORCH_VERSION"
echo "  CUDA_WHEEL    = $CUDA_WHEEL"

# ---- Sanity: nvidia-smi visible ----
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! nvidia-smi not on PATH — CUDA driver may be missing."
  echo "   Most GPU cloud images ship it; on a vanilla image install it first."
  echo "   Continuing anyway (in case you're building offline)."
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/   /'
fi

# ---- Create venv (clean: do NOT use --system-site-packages on CUDA) ----
if [ ! -d "$VENV" ]; then
  echo
  echo "-- Creating venv at $VENV (no --system-site-packages)"
  "$PY" -m venv "$VENV"
else
  echo
  echo "-- Reusing existing venv at $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ---- PyTorch with CUDA wheel ----
echo
echo "-- Installing torch==$TORCH_VERSION + $CUDA_WHEEL"
python -m pip install --upgrade \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}" \
  "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}"

# ---- Project requirements (no torch in here; we just installed it above) ----
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
echo "Activate with: source $VENV/bin/activate"
