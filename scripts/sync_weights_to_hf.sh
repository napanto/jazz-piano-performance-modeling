#!/usr/bin/env bash
# Push trained checkpoints to a Hugging Face Hub model repo.
#
# The repo is treated as a flat snapshot mirror of `runs/<model>/best.pt`
# and `runs/<model>/last.pt` (plus the small train_summary.json /
# eval_test_*.json files for easy "what does this checkpoint claim to do"
# at-a-glance). It is NOT a per-run-time-stamped archive — uploading
# overwrites the previous checkpoint. If you want timestamped snapshots,
# tag a commit on the HF side after each upload.
#
# Requires `hf auth login` (interactive) or `HF_TOKEN` env var.
#
# Usage:
#   bash scripts/sync_weights_to_hf.sh
#   bash scripts/sync_weights_to_hf.sh --models lstm_rerun
#   HF_REPO=napanto/foo bash scripts/sync_weights_to_hf.sh
#   bash scripts/sync_weights_to_hf.sh --dry-run

set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO="${HF_REPO:-napanto/jazz-piano-performance-generation}"
DRY_RUN=0
SELECTED=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --models)  shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SELECTED+=("$1"); shift; done ;;
    --repo)    HF_REPO="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "== Sync weights -> HF Hub =="
echo "  repo     = $HF_REPO"
echo "  dry-run  = $DRY_RUN"
echo "  models   = ${SELECTED[*]:-(all)}"

# Use the system-installed hf CLI — do NOT activate .venv, because the
# venv's hf inherits the venv's dependency state and may be missing
# optional deps (e.g. tqdm). A system-installed hf is self-contained.
if ! command -v hf >/dev/null 2>&1; then
  echo "!! hf CLI not on PATH. Install it via any of:" >&2
  echo "     brew install hf                       # Homebrew / Linuxbrew" >&2
  echo "     pipx install huggingface_hub[cli]     # any Python ≥3.9" >&2
  echo "     pip install --user huggingface_hub    # then ensure ~/.local/bin on PATH" >&2
  echo "     conda install -c conda-forge huggingface_hub" >&2
  echo "   See: https://huggingface.co/docs/huggingface_hub/guides/cli" >&2
  exit 1
fi

# Ensure repo exists (private by default; safe to re-run).
echo "-- ensuring HF repo $HF_REPO exists (private if new)"
if [ "$DRY_RUN" = 1 ]; then
  echo "   (dry-run, skipping create)"
else
  hf repos create "$HF_REPO" --type model --private --exist-ok \
    || { echo "!! create failed — check 'hf auth whoami' and your token permissions"; exit 1; }
fi

# Discover model dirs.
mapfile -t DIRS < <(find runs -maxdepth 1 -mindepth 1 -type d | sort)
if [ ${#SELECTED[@]} -gt 0 ]; then
  FILTERED=()
  for d in "${DIRS[@]}"; do
    name=$(basename "$d")
    for s in "${SELECTED[@]}"; do
      if [ "$name" = "$s" ]; then FILTERED+=("$d"); break; fi
    done
  done
  DIRS=("${FILTERED[@]}")
fi

if [ ${#DIRS[@]} -eq 0 ]; then
  echo "!! no model dirs to upload"; exit 1
fi

# Per-dir, upload best.pt + last.pt + small metadata files if present.
for d in "${DIRS[@]}"; do
  name=$(basename "$d")
  echo
  echo "-- $name"
  files_to_upload=()
  for f in best.pt last.pt train_summary.json eval_test_oa.json eval_test_oa_balanced.json eval_t1.0_k0_fmd.json; do
    if [ -f "$d/$f" ]; then files_to_upload+=("$d/$f"); fi
  done
  if [ ${#files_to_upload[@]} -eq 0 ]; then
    echo "   (no eligible files; skipping)"
    continue
  fi
  for src in "${files_to_upload[@]}"; do
    rel=${src#runs/}                # lstm_rerun/best.pt
    sz=$(stat -c%s "$src" | numfmt --to=iec)
    if [ "$DRY_RUN" = 1 ]; then
      echo "   [dry] would upload $src ($sz) -> $rel"
    else
      hf upload "$HF_REPO" "$src" "$rel" --repo-type model --commit-message "sync $rel" \
        && echo "   uploaded $rel ($sz)" \
        || { echo "!! upload failed for $src"; exit 1; }
    fi
  done
done

echo
echo "== Done =="
echo "Browse: https://huggingface.co/$HF_REPO"
