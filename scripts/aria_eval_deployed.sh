#!/bin/bash
WORKSPACE="${WORKSPACE:-$PWD}"   # deployment root: holds the repo, the aria checkout, data and checkpoints
# Stage C eval (sampling sweep + OA + FMD) on the already-deployed Aria
# real-time MLX checkpoint, AS-IS — no fine-tuning. Use this when you want
# to know how the deployed model itself scores on the kong PiJAMA test split,
# without continuing to fine-tune it.
#
# The deployed checkpoint lives at huggingface.co/napanto/jazz-piano-
# performance-modeling/aria-real-time/{deployed.safetensors,mlx-deployed/
# tokenizer-config.json}. It is the v2-style demo / real-time MLX model after the
# v2 4-stage fine-tune on kong (with the medium-emb architecture).
#
# Outputs (mirrored to local + HF, same layout as the v1-v4 pipeline):
#   $RUNS/stageC/sweep/<config>/{prompt_*/var_*.mid,eval_stagec.json}
#   $RUNS/stageC/sweep_summary.csv
#   $RUNS/stageC/sweep_summary_fmd.csv
#
# Usage:
#   VARIANT=v2-deployed N_PROMPTS=4 N_VARIATIONS=20 \
#     bash scripts/aria_eval_deployed.sh
#
# Inputs (env-var):
#   VARIANT       (default v2-deployed) — sub-dir name under aria_finetune/
#   N_PROMPTS     (default 4) — number of kong test prompts to sample from
#   N_VARIATIONS  (default 20) — number of variations per prompt
#   PROMPT_DURATION (default 8) — seconds of audible prompt
#   TEMPS         (default "0.8 1.0 1.2") — temperature sweep
#   TOP_KS        (default "0 24") — top-k sweep
#   MIN_PS        (default "0.035 0.05") — min-p sweep
#   BASE_CKPT_DIR (default ${WORKSPACE}/aria_checkpoints)
#   FMD_REF_DIR   (default ${WORKSPACE}/data/pijama_raw_kong/test)

set -e -o pipefail

VARIANT="${VARIANT:-v2-deployed}"
N_PROMPTS="${N_PROMPTS:-4}"
N_VARIATIONS="${N_VARIATIONS:-20}"
PROMPT_DURATION="${PROMPT_DURATION:-8}"
TEMPS="${TEMPS:-0.8 1.0 1.2}"
TOP_KS="${TOP_KS:-0 24}"
MIN_PS="${MIN_PS:-0.035 0.05}"

REPO="${REPO:-${WORKSPACE}/jazz_piano}"
ARIA="${ARIA:-${WORKSPACE}/aria}"
CKPTS="${BASE_CKPT_DIR:-${WORKSPACE}/aria_checkpoints}"
RUNS="${WORKSPACE}/runs/${VARIANT}"
LOGS="${WORKSPACE}/logs/${VARIANT}"
DATA="${WORKSPACE}/data/${VARIANT}"
HF_REPO="napanto/jazz-piano-performance-generation"
FMD_REF_DIR="${FMD_REF_DIR:-${WORKSPACE}/data/pijama_raw_kong/test}"

mkdir -p "$RUNS" "$LOGS" "$DATA"
say() { echo "[${VARIANT}-eval $(date -u +%FT%T)] $*" | tee -a "$LOGS/eval.log"; }

# --- Pull the deployed checkpoint + tokenizer if not present locally ---
DEPLOYED_HF_REPO="napanto/jazz-piano-performance-modeling"
DEPLOYED_HF_PATH="aria-real-time"
BASE_CKPT="$CKPTS/aria_realtime_deployed.safetensors"
TOK_CFG="$CKPTS/aria_realtime_deployed_tokenizer.json"
MODEL_NAME="medium-emb"

if [ ! -f "$BASE_CKPT" ]; then
    say "downloading deployed real-time ckpt from $DEPLOYED_HF_REPO"
    hf download "$DEPLOYED_HF_REPO" \
        "$DEPLOYED_HF_PATH/deployed.safetensors" \
        --local-dir "$CKPTS/_hf_deployed_tmp" 2>&1 | tail -3
    mv "$CKPTS/_hf_deployed_tmp/$DEPLOYED_HF_PATH/deployed.safetensors" "$BASE_CKPT"
    rm -rf "$CKPTS/_hf_deployed_tmp"
fi

if [ ! -f "$TOK_CFG" ]; then
    say "downloading deployed real-time tokenizer config"
    hf download "$DEPLOYED_HF_REPO" \
        "$DEPLOYED_HF_PATH/mlx-deployed/tokenizer-config.json" \
        --local-dir "$CKPTS/_hf_deployed_tmp" 2>&1 | tail -3
    mv "$CKPTS/_hf_deployed_tmp/$DEPLOYED_HF_PATH/mlx-deployed/tokenizer-config.json" "$TOK_CFG"
    rm -rf "$CKPTS/_hf_deployed_tmp"
fi

say "BASE_CKPT=$BASE_CKPT"
say "TOK_CFG=$TOK_CFG"
say "MODEL_NAME=$MODEL_NAME"

# --- Stage C: sampling sweep ---
say "=== STAGE C: sampling sweep over T x top_k x min_p ==="
mkdir -p "$RUNS/stageC"
ARIA_TOK_CFG="$TOK_CFG" python3 "$REPO/src/sample_aria_sweep.py" \
    --checkpoint "$BASE_CKPT" \
    --model-name "$MODEL_NAME" \
    --tokenizer-config "$TOK_CFG" \
    --test-midi-dir "$FMD_REF_DIR" \
    --n-prompts "$N_PROMPTS" \
    --n-variations "$N_VARIATIONS" \
    --prompt-duration "$PROMPT_DURATION" \
    --temps $TEMPS \
    --top-ks $TOP_KS \
    --min-ps $MIN_PS \
    --out-dir "$RUNS/stageC" \
    --aria-repo "$ARIA" 2>&1 | tee "$LOGS/stageC_sweep.log"

# --- Stage C metrics: OA / KLD / FMD over each config ---
say "=== STAGE C: distributional metrics (OA + KLD + FMD) ==="
python3 "$REPO/src/eval_aria_metrics.py" \
    --sweep-dir "$RUNS/stageC/sweep" \
    --reference-dir "$FMD_REF_DIR" \
    --out-csv "$RUNS/stageC/sweep_summary.csv" \
    --fmd-out-csv "$RUNS/stageC/sweep_summary_fmd.csv" 2>&1 | tee "$LOGS/stageC_metrics.log"

# --- Push artefacts ---
say "=== pushing to HF $HF_REPO/aria_finetune/${VARIANT} ==="
hf upload "$HF_REPO" "$RUNS/stageC" "aria_finetune/${VARIANT}/stageC" --repo-type dataset 2>&1 | tail -3 || true
hf upload "$HF_REPO" "$LOGS" "aria_finetune/${VARIANT}/logs" --repo-type dataset 2>&1 | tail -3 || true

# Mirror small artefacts locally too (the report-writing process reads from there)
LOCAL_MIRROR="${REPO}/reports/aria_sweep_midis/aria_finetune/${VARIANT}/stageC"
mkdir -p "$LOCAL_MIRROR"
cp "$RUNS/stageC/sweep_summary.csv" "$LOCAL_MIRROR/sweep_summary.csv" 2>/dev/null || true
cp "$RUNS/stageC/sweep_summary_fmd.csv" "$LOCAL_MIRROR/sweep_summary_fmd.csv" 2>/dev/null || true

say "=== DONE ==="
say "headline: $(tail -n +2 $RUNS/stageC/sweep_summary.csv | sort -t, -k7 -gr | head -1)"
say "best FMD: $(tail -n +2 $RUNS/stageC/sweep_summary_fmd.csv | sort -t, -k6 -g | head -1)"
