#!/usr/bin/env bash
# Pull trained checkpoints down from the Hugging Face Hub model repo.
#
# Mirror of `sync_weights_to_hf.sh`. Populates `runs/<model>/best.pt`,
# `runs/<model>/last.pt`, and any small metadata that was uploaded.
#
# Use this on a fresh machine (the AWS instance, a collaborator's box,
# etc.) after `git clone` to recover the trained checkpoints.
#
# Requires `hf auth login` (interactive) or `HF_TOKEN` env var with read
# access to the model repo if private.
#
# Usage:
#   bash scripts/sync_weights_from_hf.sh
#   bash scripts/sync_weights_from_hf.sh --models lstm_rerun
#   HF_REPO=napanto/foo bash scripts/sync_weights_from_hf.sh

set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO="${HF_REPO:-napanto/jazz-piano-performance-generation}"
SELECTED=()

while [ $# -gt 0 ]; do
  case "$1" in
    --models) shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SELECTED+=("$1"); shift; done ;;
    --repo)   HF_REPO="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "== Sync weights <- HF Hub =="
echo "  repo   = $HF_REPO"
echo "  models = ${SELECTED[*]:-(all)}"

if ! command -v hf >/dev/null 2>&1; then
  echo "!! hf CLI not on PATH. Install it via any of:" >&2
  echo "     brew install hf                       # Homebrew / Linuxbrew" >&2
  echo "     pipx install huggingface_hub[cli]     # any Python ≥3.9" >&2
  echo "     pip install --user huggingface_hub    # then ensure ~/.local/bin on PATH" >&2
  echo "     conda install -c conda-forge huggingface_hub" >&2
  echo "   See: https://huggingface.co/docs/huggingface_hub/guides/cli" >&2
  exit 1
fi

mkdir -p runs

if [ ${#SELECTED[@]} -eq 0 ]; then
  # Pull the whole repo into runs/ in one shot. Uses the hf snapshot
  # download under the hood; resumable and idempotent.
  hf download "$HF_REPO" --repo-type model --local-dir runs
else
  for name in "${SELECTED[@]}"; do
    echo
    echo "-- $name"
    hf download "$HF_REPO" --repo-type model --local-dir "runs" \
      --include "$name/*"
  done
fi

echo
echo "== Done =="
ls -la runs/ | head -30
