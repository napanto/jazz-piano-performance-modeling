#!/bin/bash
WORKSPACE="${WORKSPACE:-$PWD}"   # deployment root: holds the repo, the aria checkout, data and checkpoints
# Stage-C orchestrator for a single Aria variant.
#
# Drives:
#   1. src/sample_aria_sweep.py  — batched sampling grid on stageB_final.safetensors
#   2. src/eval_aria_metrics.py  — OA / KLD / FMD on the resulting MIDI sweep dir
#   3. hf upload                 — pushes sweep CSVs + sample MIDIs to HF
#
# Reads the same VARIANT-specific configuration as
# `aria_pipeline_per_variant.sh` so v1/v2/v3/v4 stay consistent
# (tokenizer config, kong-vs-hawthorne split, model name).
#
# Usage:
#   VARIANT=v1 bash scripts/aria_eval_stageC.sh
#   VARIANT=v2 bash scripts/aria_eval_stageC.sh
#   VARIANT=v3 bash scripts/aria_eval_stageC.sh
#   VARIANT=v4 bash scripts/aria_eval_stageC.sh
#
# Env knobs (with defaults):
#   N_PROMPTS=4      — how many TEST prompts to use
#   N_VARIATIONS=20  — variations per prompt per setting
#   LENGTH=2048      — tokens to generate per variation
#   PROMPT_DURATION=8
#   TEMPS=0.8,1.0,1.2
#   TOP_KS=0,24
#   MIN_PS=0.035,0.05
#   MAX_TEST_CLIPS=200
#   FMD_BACKBONE=auto (auto|clamp2|aria|none)
#   SKIP_SAMPLE=0    — set to 1 to skip the (slow) sampling step
#   SKIP_EVAL=0      — set to 1 to skip the metrics step
#   SKIP_PUSH=0      — set to 1 to skip HF upload
#
# Constraints:
#   - This script DOES NOT touch $RUNS/stageA, stageB or stageD. Only
#     $RUNS/stageC/{sweep,sweep_meta.json,sweep.csv,sweep_summary.csv}.
#   - Re-runs are idempotent — sample_aria_sweep skips (setting, prompt)
#     pairs whose var_*.mid files already exist.

set -e -o pipefail

VARIANT="${VARIANT:?set VARIANT=v1|v2|v3|v4|v6}"
N_PROMPTS="${N_PROMPTS:-4}"
N_VARIATIONS="${N_VARIATIONS:-20}"
LENGTH="${LENGTH:-2048}"
PROMPT_DURATION="${PROMPT_DURATION:-8}"
TEMPS="${TEMPS:-0.8,1.0,1.2}"
TOP_KS="${TOP_KS:-0,24}"
MIN_PS="${MIN_PS:-0.035,0.05}"
MAX_TEST_CLIPS="${MAX_TEST_CLIPS:-200}"
FMD_BACKBONE="${FMD_BACKBONE:-auto}"
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

REPO=${WORKSPACE}/jazz_piano
ARIA=${WORKSPACE}/aria
RUNS=${WORKSPACE}/runs/${VARIANT}
LOGS=${WORKSPACE}/logs/${VARIANT}
HF_REPO="napanto/jazz-piano-performance-generation"

mkdir -p "$LOGS"

say() { echo "[${VARIANT}/stageC $(date -u +%FT%T)] $*" | tee -a "$LOGS/stageC.log"; }

# ===== variant-specific configuration (mirrors aria_pipeline_per_variant.sh) =====
case "$VARIANT" in
    v1) TOK_CFG=""
        PIJAMA_KONG_OR_HAW=hawthorne
        MODEL_NAME="medium"
        ;;
    v2) TOK_CFG="${WORKSPACE}/demo_cfg/demo-tokenizer-config.json"
        PIJAMA_KONG_OR_HAW=kong
        MODEL_NAME="medium-emb"
        ;;
    v3) TOK_CFG=""
        PIJAMA_KONG_OR_HAW=kong
        MODEL_NAME="medium"
        ;;
    v4) TOK_CFG="${WORKSPACE}/pedal_only_cfg.json"
        PIJAMA_KONG_OR_HAW=kong
        MODEL_NAME="medium"
        ;;
    v6) TOK_CFG="${WORKSPACE}/pedal_only_cfg.json"
        PIJAMA_KONG_OR_HAW=kong
        MODEL_NAME="medium"
        ;;
    *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

CKPT="$RUNS/stageB_final.safetensors"
if [ ! -f "$CKPT" ]; then
    say "ERROR: $CKPT does not exist; Stage B did not finish or upload."
    exit 3
fi

TEST_MIDI_DIR="${WORKSPACE}/data/pijama_raw_${PIJAMA_KONG_OR_HAW}/test"
if [ ! -d "$TEST_MIDI_DIR" ]; then
    say "ERROR: $TEST_MIDI_DIR does not exist."
    exit 4
fi

STAGEC="$RUNS/stageC"
SWEEP_DIR="$STAGEC/sweep"
META_JSON="$STAGEC/sweep_meta.json"
SWEEP_CSV="$STAGEC/sweep.csv"
SUMMARY_CSV="$STAGEC/sweep_summary.csv"
TEST_EMB_CACHE="$STAGEC/test_emb_cache.npy"
mkdir -p "$STAGEC"

say "=== Stage C ==="
say "variant       = $VARIANT"
say "checkpoint    = $CKPT"
say "test MIDI dir = $TEST_MIDI_DIR"
say "tok cfg       = ${TOK_CFG:-default}"
say "model         = $MODEL_NAME"
say "grid          = T=$TEMPS  k=$TOP_KS  p=$MIN_PS"
say "prompts/var   = $N_PROMPTS x $N_VARIATIONS"
say "out dir       = $STAGEC"

# ===== 1. Sampling sweep =====
if [ "$SKIP_SAMPLE" != "1" ]; then
    say "--- 1) sampling sweep ---"
    cd "$REPO"
    python3 -m src.sample_aria_sweep \
        --variant "$VARIANT" \
        --checkpoint "$CKPT" \
        --test-midi-dir "$TEST_MIDI_DIR" \
        --tokenizer-config "$TOK_CFG" \
        --model-name "$MODEL_NAME" \
        --out-dir "$STAGEC" \
        --n-prompts "$N_PROMPTS" \
        --n-variations "$N_VARIATIONS" \
        --prompt-duration "$PROMPT_DURATION" \
        --length "$LENGTH" \
        --temps "$TEMPS" \
        --top-ks "$TOP_KS" \
        --min-ps "$MIN_PS" \
        --aria-repo "$ARIA" \
        --device cuda \
        --dtype bfloat16 2>&1 | tee -a "$LOGS/stageC_sample.log" | tail -50
else
    say "--- 1) sampling sweep SKIPPED (SKIP_SAMPLE=1) ---"
fi

# ===== 2. Metric eval =====
if [ "$SKIP_EVAL" != "1" ]; then
    say "--- 2) OA / KLD / FMD eval ---"
    cd "$REPO"
    python3 -m src.eval_aria_metrics \
        --gen-dir "$SWEEP_DIR" \
        --test-midi-dir "$TEST_MIDI_DIR" \
        --out "$SWEEP_CSV" \
        --variant "$VARIANT" \
        --max-test-clips "$MAX_TEST_CLIPS" \
        --fmd-backbone "$FMD_BACKBONE" \
        --test-emb-cache "$TEST_EMB_CACHE" 2>&1 | tee -a "$LOGS/stageC_eval.log" | tail -50
else
    say "--- 2) metric eval SKIPPED (SKIP_EVAL=1) ---"
fi

# ===== 3. HF push =====
if [ "$SKIP_PUSH" != "1" ]; then
    say "--- 3) push to HF ---"
    if ! command -v hf >/dev/null 2>&1; then
        say "WARN: hf CLI not available; skipping push"
    else
        # Push the small artefacts (CSVs, meta) — fast and always useful.
        for f in "$SWEEP_CSV" "$SUMMARY_CSV" "$META_JSON"; do
            if [ -f "$f" ]; then
                rel=$(basename "$f")
                hf upload "$HF_REPO" "$f" \
                    "aria_finetune/${VARIANT}/stageC/$rel" --repo-type model \
                    2>&1 | tail -2 \
                    || say "WARN: upload of $rel failed"
            fi
        done
        # Push the sweep MIDIs — bigger; skip with HF_SKIP_MIDIS=1 if bandwidth-bound.
        if [ "${HF_SKIP_MIDIS:-0}" != "1" ] && [ -d "$SWEEP_DIR" ]; then
            hf upload "$HF_REPO" "$SWEEP_DIR" \
                "aria_finetune/${VARIANT}/stageC/sweep" --repo-type model \
                2>&1 | tail -2 \
                || say "WARN: upload of sweep MIDIs failed"
        fi
    fi
else
    say "--- 3) push to HF SKIPPED (SKIP_PUSH=1) ---"
fi

say "=== Stage C complete ==="
