#!/bin/bash
WORKSPACE="${WORKSPACE:-$PWD}"   # deployment root: holds the repo, the aria checkout, data and checkpoints
# Aria multi-stage fine-tune pipeline for a SINGLE variant.
#
# Runs Stages A → B → C (sampling sweep) → D for one of {v1, v2, v3,
# v2-deployed}. The "v2-deployed" variant starts from the already-deployed
# real-time MLX checkpoint published at huggingface.co/napanto/jazz-piano-
# jazz-piano-performance-modeling/aria-real-time/ instead of EleutherAI's stock model-demo, so
# fine-tuning *continues* from a model that already speaks jazz with pedal.
# To evaluate the deployed model as-is (no fine-tune), use the standalone
# scripts/aria_eval_deployed.sh helper instead — it skips training and runs
# only the Stage C sweep + FMD.
#
# Expects to run on a B200 pod that has:
#   - ${WORKSPACE}/jazz_piano (this repo, git clone)
#   - ${WORKSPACE}/aria (EleutherAI/aria checkout, pip-installed)
#   - python deps installed
#   - hf auth login (token in /root/.cache/huggingface/token)
#   - Aria train.py patched (the AbsTokenizer() calls take an
#     ARIA_TOK_CFG env var; see plan §1)
#
# Inputs (env-var or arg):
#   VARIANT=v1|v2|v3|v2-deployed
#   N_STAGE_A=16
#   PATIENCE=4
#   BS=16, WORKERS=8
#   PIJAMA_RAW=${WORKSPACE}/data/pijama_raw_hawthorne   (for v1)
#                ${WORKSPACE}/data/pijama_raw_kong        (for v2/v3)
#   BASE_CKPT_DIR=${WORKSPACE}/aria_checkpoints
#
# Behaviour:
#   - Tokenizes train+val split (Stage A); train.py runs --epochs N_STAGE_A
#     with the patience watcher.
#   - Reads best_epoch_A from epoch.csv, tokenizes TRAIN+VAL combined,
#     runs Stage B for best_epoch_A epochs.
#   - Converts Stage B ckpt to safetensors.
#   - Runs sampling sweep + OA/KLD/FMD eval (Stage C). [TODO: hook our
#     existing src/eval_distributional.py — left as user-extensible.]
#   - Tokenizes TRAIN+VAL+TEST combined, runs Stage D for best_epoch_A
#     epochs.
#   - Pushes everything to HF napanto/jazz-piano-performance-generation
#     under aria_finetune/${VARIANT}/.
#
# Extensive logging: every stage writes train_summary.json with git sha,
# CLI args, tokenizer config, accelerate config; epoch.csv, loss.csv,
# logs.txt are saved per stage; GPU sidecar writes 30-s VRAM/util.

set -e -o pipefail

VARIANT="${VARIANT:?set VARIANT=v1|v2|v3|v2-deployed}"
N_STAGE_A="${N_STAGE_A:-16}"
PATIENCE="${PATIENCE:-4}"
BS="${BS:-16}"
WORKERS="${WORKERS:-8}"
SEQ_LEN="${SEQ_LEN:-8192}"   # Aria paper §3.1: "A sequence length of 8192 tokens was chosen ...";
                              # pretrained on 8 H100 (bf16, BS=16/GPU). We match that here. The orig
                              # fast experimental run used 4096 — that's a deviation, NOT what the paper says.
                              # Compensate for the 4x attention cost with bf16 (see --mixed_precision below).
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_EPOCHS_TOK="${NUM_EPOCHS_TOK:-4}"
REPO=${WORKSPACE}/jazz_piano
ARIA=${WORKSPACE}/aria
RUNS=${WORKSPACE}/runs/${VARIANT}
LOGS=${WORKSPACE}/logs/${VARIANT}
DATA=${WORKSPACE}/data/${VARIANT}
CKPTS=${WORKSPACE}/aria_checkpoints
HF_REPO="napanto/jazz-piano-performance-generation"
mkdir -p "$RUNS" "$LOGS" "$DATA"

say() { echo "[${VARIANT} $(date -u +%FT%T)] $*" | tee -a "$LOGS/chain.log"; }

# ===== variant-specific configuration =====
case "$VARIANT" in
    v1) BASE_CKPT="$CKPTS/model-gen.safetensors"
        TOK_CFG=""
        PIJAMA_KONG_OR_HAW=hawthorne
        MODEL_NAME="medium"
        ;;
    v2) BASE_CKPT="$CKPTS/model-demo.safetensors"
        TOK_CFG="${WORKSPACE}/demo_cfg/demo-tokenizer-config.json"
        PIJAMA_KONG_OR_HAW=kong
        # model-demo is the contrastive-finetune variant; it has an extra
        # emb_proj layer (emb_size=512 -> d_model=1536). Load with the
        # medium-emb arch so emb_proj's weights are loaded + preserved in
        # the output safetensors, making the result a drop-in replacement
        # for demo_mlx.py (the real-time iOS / MLX demo).
        MODEL_NAME="medium-emb"
        ;;
    v3) BASE_CKPT="$CKPTS/model-gen.safetensors"
        TOK_CFG=""
        PIJAMA_KONG_OR_HAW=kong
        MODEL_NAME="medium"
        ;;
    v2-deployed) # Use the already-deployed real-time MLX checkpoint from HF as
        # the base, instead of EleutherAI's stock model-demo. Same recipe as v2
        # otherwise (medium-emb arch, demo tokenizer, kong split).
        #
        # This variant runs the full 4-stage pipeline starting *from* the
        # deployed weights — i.e. continues fine-tuning on a model that already
        # speaks jazz with pedal. For eval-only (deployed model as-is, Stage C
        # sweep + FMD on kong test, no training) use scripts/aria_eval_deployed.sh.
        BASE_CKPT="$CKPTS/aria_realtime_deployed.safetensors"
        TOK_CFG="$CKPTS/aria_realtime_deployed_tokenizer.json"
        PIJAMA_KONG_OR_HAW=kong
        MODEL_NAME="medium-emb"
        DEPLOYED_HF_REPO="napanto/jazz-piano-performance-modeling"
        DEPLOYED_HF_PATH="aria-real-time"
        if [ ! -f "$BASE_CKPT" ]; then
            say "downloading deployed real-time checkpoint from $DEPLOYED_HF_REPO/$DEPLOYED_HF_PATH"
            hf download "$DEPLOYED_HF_REPO" \
                "$DEPLOYED_HF_PATH/deployed.safetensors" \
                --local-dir "$CKPTS/_hf_deployed_tmp" 2>&1 | tee -a "$LOGS/hf_download.log"
            mv "$CKPTS/_hf_deployed_tmp/$DEPLOYED_HF_PATH/deployed.safetensors" "$BASE_CKPT"
            rm -rf "$CKPTS/_hf_deployed_tmp"
        fi
        if [ ! -f "$TOK_CFG" ]; then
            say "downloading deployed real-time tokenizer config"
            hf download "$DEPLOYED_HF_REPO" \
                "$DEPLOYED_HF_PATH/mlx-deployed/tokenizer-config.json" \
                --local-dir "$CKPTS/_hf_deployed_tmp" 2>&1 | tee -a "$LOGS/hf_download.log"
            mv "$CKPTS/_hf_deployed_tmp/$DEPLOYED_HF_PATH/mlx-deployed/tokenizer-config.json" "$TOK_CFG"
            rm -rf "$CKPTS/_hf_deployed_tmp"
        fi
        ;;
    *) echo "Unknown VARIANT=$VARIANT (expected: v1|v2|v3|v2-deployed)"; exit 2 ;;
esac

# Pin aria/config/models/${MODEL_NAME}.json max_seq_len to match $SEQ_LEN.
# train.py asserts dataset.config.max_seq_len == model_config.max_seq_len at line 724;
# if these drift the run crashes after Stage A tokenize.
MODEL_JSON="$ARIA/config/models/${MODEL_NAME}.json"
python3 - <<PY
import json, sys
p = "$MODEL_JSON"
c = json.load(open(p))
if c.get("max_seq_len") != $SEQ_LEN:
    c["max_seq_len"] = $SEQ_LEN
    json.dump(c, open(p, "w"), indent=4)
    print(f"[pipeline] patched {p} max_seq_len -> $SEQ_LEN")
else:
    print(f"[pipeline] {p} already has max_seq_len=$SEQ_LEN")
PY

# Source JSONLs (built from raw MIDI by `aria midi-dataset`):
JSONL_TRAIN="${WORKSPACE}/data/jsonl/${PIJAMA_KONG_OR_HAW}_train.jsonl"
JSONL_VAL="${WORKSPACE}/data/jsonl/${PIJAMA_KONG_OR_HAW}_val.jsonl"
JSONL_TEST="${WORKSPACE}/data/jsonl/${PIJAMA_KONG_OR_HAW}_test.jsonl"

say "=== variant=$VARIANT base=$BASE_CKPT tok_cfg=${TOK_CFG:-default} ==="
say "JSONL: $JSONL_TRAIN $JSONL_VAL $JSONL_TEST"

# ===== ckpt sweeper + GPU sidecar =====
mkdir -p /tmp/${VARIANT}
cat > /tmp/${VARIANT}/gpu_sidecar.sh <<SH
#!/bin/bash
out=$LOGS/gpu_log.jsonl
while true; do
  ts=\$(date -u +%FT%T)
  read mem util temp pwr <<<\$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits | tr ',' ' ' | head -1)
  echo "{\"ts\":\"\$ts\",\"vram_mib\":\$mem,\"gpu_util\":\$util,\"temp_c\":\$temp,\"power_w\":\$pwr}" >> \$out
  sleep 30
done
SH
chmod +x /tmp/${VARIANT}/gpu_sidecar.sh
pkill -f "/tmp/${VARIANT}/gpu_sidecar" 2>/dev/null || true
nohup /tmp/${VARIANT}/gpu_sidecar.sh > /dev/null 2>&1 &

# ===== Tokenization helpers =====
tokenize_split() {
    # tokenize_split <input_jsonl> <save_dir> <num_epochs>
    local in="$1" out="$2" ne="$3"
    rm -rf "$out"; mkdir -p "$out"
    cd "$ARIA"
    # NOTE: do NOT use `yes Y |` here — `yes` gets SIGPIPE when python exits
    # and pipefail then kills the whole pipeline script. PretrainingDataset.build
    # never prompts because we just rm -rf'd $out, so stdin can be /dev/null.
    ARIA_TOK_CFG="$TOK_CFG" python3 "$REPO/src/aria_tokenize_with_cfg.py" \
        --load-path "$in" --save-dir "$out" \
        --num-epochs "$ne" --seq-len "$SEQ_LEN" \
        --tokenizer-config "$TOK_CFG" \
        </dev/null 2>&1 | tee -a "$LOGS/tokenize.log" | tail -3
}

# ===== Stage A =====
say "=== STAGE A: tokenize TRAIN+VAL splits separately, train --epochs $N_STAGE_A ==="
tokenize_split "$JSONL_TRAIN" "$DATA/tok_stageA/train" "$NUM_EPOCHS_TOK"
tokenize_split "$JSONL_VAL"   "$DATA/tok_stageA/val"   1

rm -rf "$RUNS/stageA"
mkdir -p /tmp/${VARIANT}
tmux kill-session -t ${VARIANT}_train 2>/dev/null || true
tmux kill-session -t ${VARIANT}_sweep 2>/dev/null || true
tmux kill-session -t ${VARIANT}_earlystop 2>/dev/null || true

tmux new -d -s ${VARIANT}_sweep "bash $REPO/scripts/sweep_ckpts_best.sh $RUNS/stageA"
tmux new -d -s ${VARIANT}_earlystop "bash $REPO/scripts/aria_early_stop_watcher.sh $RUNS/stageA $PATIENCE ${VARIANT}_train >> $LOGS/earlystop.log 2>&1"

# Env vars in shell-prefix form (FOO=bar tmux ...) don't propagate into tmux subshell
# once a tmux server already exists from earlier sessions. Export inside the tmux command.
tmux new -d -s ${VARIANT}_train "export ARIA_TOK_CFG='$TOK_CFG' && cd $ARIA && \
  accelerate launch --mixed_precision $MIXED_PRECISION aria/training/train.py train $MODEL_NAME \
    --train_data $DATA/tok_stageA/train --val_data $DATA/tok_stageA/val \
    --cp_path $BASE_CKPT \
    --epochs $N_STAGE_A --bs $BS --workers $WORKERS \
    --pdir $RUNS/stageA --spc 99999 2>&1 | tee $LOGS/stageA.log; \
  echo TRAIN_EXIT_\$?; sleep 60"

# Wait for train tmux to die (either patience triggered or N_STAGE_A done)
while tmux has-session -t ${VARIANT}_train 2>/dev/null; do sleep 30; done
tmux kill-session -t ${VARIANT}_earlystop 2>/dev/null || true
tmux kill-session -t ${VARIANT}_sweep 2>/dev/null || true
say "stage A finished"

# Read best_epoch_A from epoch.csv
BEST_EPOCH_A=$(python3 - <<PY
import csv, sys
rows = list(csv.DictReader(open("$RUNS/stageA/epoch.csv")))
if not rows:
    print("0")
    sys.exit()
best = min(rows, key=lambda r: float(r["avg_val_loss"]))
print(int(best["epoch"]))
PY
)
say "best_epoch_A = $BEST_EPOCH_A (from $RUNS/stageA/epoch.csv)"
N_STAGE_B=$((BEST_EPOCH_A + 1))   # +1 because epoch is 0-indexed

# Convert Stage-A best ckpt to safetensors (for HF push + sampling)
STAGE_A_BEST_CKPT_DIR=$(ls -d "$RUNS/stageA/checkpoints/epoch$((BEST_EPOCH_A + 1))_step0" 2>/dev/null)
if [ -z "$STAGE_A_BEST_CKPT_DIR" ]; then
    # Fallback to whatever the sweeper preserved
    STAGE_A_BEST_CKPT_DIR=$(ls -td "$RUNS/stageA/checkpoints/"*/ | head -1)
fi
ARIA_TOK_CFG="$TOK_CFG" python3 "$REPO/src/aria_convert_ckpt.py" \
    --cp-dir "$STAGE_A_BEST_CKPT_DIR" \
    --save-path "$RUNS/stageA_best.safetensors" \
    --aria-repo "$ARIA" \
    2>&1 | tee -a "$LOGS/stageA.log"

# Push Stage-A artefacts
hf upload "$HF_REPO" "$RUNS/stageA_best.safetensors" \
    "aria_finetune/${VARIANT}/stageA/best.safetensors" --repo-type model 2>&1 | tail -2 || true
hf upload "$HF_REPO" "$RUNS/stageA" \
    "aria_finetune/${VARIANT}/stageA/run_dir" --repo-type model 2>&1 | tail -2 || true
hf upload "$HF_REPO" "$LOGS" \
    "aria_finetune/${VARIANT}/logs_stageA" --repo-type model 2>&1 | tail -2 || true
say "stage A pushed to HF"

# ===== Stage B =====
say "=== STAGE B: tokenize TRAIN+VAL combined; train for $N_STAGE_B epochs ==="
# Use cat to combine the JSONL files (each is one MidiDict per line)
cat "$JSONL_TRAIN" "$JSONL_VAL" > "$DATA/jsonl_train_val.jsonl"
tokenize_split "$DATA/jsonl_train_val.jsonl" "$DATA/tok_stageB/train" "$NUM_EPOCHS_TOK"
# Keep the same val split for early-stop / monitoring during Stage B.
cp -r "$DATA/tok_stageA/val" "$DATA/tok_stageB/val"

rm -rf "$RUNS/stageB"
tmux new -d -s ${VARIANT}_sweep "bash $REPO/scripts/sweep_ckpts_best.sh $RUNS/stageB"
tmux new -d -s ${VARIANT}_train "export ARIA_TOK_CFG='$TOK_CFG' && cd $ARIA && \
  accelerate launch --mixed_precision $MIXED_PRECISION aria/training/train.py train $MODEL_NAME \
    --train_data $DATA/tok_stageB/train --val_data $DATA/tok_stageB/val \
    --cp_path $BASE_CKPT \
    --epochs $N_STAGE_B --bs $BS --workers $WORKERS \
    --pdir $RUNS/stageB --spc 99999 2>&1 | tee $LOGS/stageB.log; \
  echo TRAIN_EXIT_\$?; sleep 60"
while tmux has-session -t ${VARIANT}_train 2>/dev/null; do sleep 30; done
tmux kill-session -t ${VARIANT}_sweep 2>/dev/null || true
say "stage B finished"

STAGE_B_CKPT_DIR=$(ls -td "$RUNS/stageB/checkpoints/"*/ | head -1)
ARIA_TOK_CFG="$TOK_CFG" python3 "$REPO/src/aria_convert_ckpt.py" \
    --cp-dir "$STAGE_B_CKPT_DIR" \
    --save-path "$RUNS/stageB_final.safetensors" \
    --aria-repo "$ARIA" \
    2>&1 | tee -a "$LOGS/stageB.log"
hf upload "$HF_REPO" "$RUNS/stageB_final.safetensors" \
    "aria_finetune/${VARIANT}/stageB/final.safetensors" --repo-type model 2>&1 | tail -2 || true
hf upload "$HF_REPO" "$RUNS/stageB" \
    "aria_finetune/${VARIANT}/stageB/run_dir" --repo-type model 2>&1 | tail -2 || true
say "stage B pushed to HF"

# ===== Stage C =====
say "=== STAGE C: TEST eval + sampling sweep (placeholder) ==="
# TODO: hook the existing project3/src/metrics OA/KLD/FMD pipeline once
# the variants tokenize / decode cleanly. For now produce *some*
# generations from a handful of TEST prompts so the report has
# something to inspect.
tokenize_split "$JSONL_TEST" "$DATA/tok_stageC/test" 1
mkdir -p "$RUNS/stageC/generations"
# Generate using aria generate. We sample from the *test* MIDIs as
# prompts (first 5).
for PROMPT_MIDI in $(ls ${WORKSPACE}/data/raw_test/*.mid* 2>/dev/null | head -5); do
    base=$(basename "$PROMPT_MIDI" .mid)
    ARIA_TOK_CFG="$TOK_CFG" aria generate \
        --backend torch_cuda \
        --checkpoint_path "$RUNS/stageB_final.safetensors" \
        --prompt_midi_path "$PROMPT_MIDI" \
        --prompt_duration 8 \
        --variations 4 \
        --temp 0.98 \
        --min_p 0.035 \
        --length 2048 \
        --save_dir "$RUNS/stageC/generations/${base}" 2>&1 | tail -3 \
        || say "WARN: aria generate failed for $PROMPT_MIDI"
done
hf upload "$HF_REPO" "$RUNS/stageC" \
    "aria_finetune/${VARIANT}/stageC" --repo-type model 2>&1 | tail -2 || true

# ===== Stage D =====
say "=== STAGE D: TRAIN+VAL+TEST retrain for $N_STAGE_B epochs (deployment) ==="
cat "$JSONL_TRAIN" "$JSONL_VAL" "$JSONL_TEST" > "$DATA/jsonl_train_val_test.jsonl"
tokenize_split "$DATA/jsonl_train_val_test.jsonl" "$DATA/tok_stageD/train" "$NUM_EPOCHS_TOK"
cp -r "$DATA/tok_stageA/val" "$DATA/tok_stageD/val"
rm -rf "$RUNS/stageD"
tmux new -d -s ${VARIANT}_sweep "bash $REPO/scripts/sweep_ckpts_best.sh $RUNS/stageD"
tmux new -d -s ${VARIANT}_train "export ARIA_TOK_CFG='$TOK_CFG' && cd $ARIA && \
  accelerate launch --mixed_precision $MIXED_PRECISION aria/training/train.py train $MODEL_NAME \
    --train_data $DATA/tok_stageD/train --val_data $DATA/tok_stageD/val \
    --cp_path $BASE_CKPT \
    --epochs $N_STAGE_B --bs $BS --workers $WORKERS \
    --pdir $RUNS/stageD --spc 99999 2>&1 | tee $LOGS/stageD.log; \
  echo TRAIN_EXIT_\$?; sleep 60"
while tmux has-session -t ${VARIANT}_train 2>/dev/null; do sleep 30; done
tmux kill-session -t ${VARIANT}_sweep 2>/dev/null || true
say "stage D finished"
STAGE_D_CKPT_DIR=$(ls -td "$RUNS/stageD/checkpoints/"*/ | head -1)
ARIA_TOK_CFG="$TOK_CFG" python3 "$REPO/src/aria_convert_ckpt.py" \
    --cp-dir "$STAGE_D_CKPT_DIR" \
    --save-path "$RUNS/stageD_demo.safetensors" \
    --aria-repo "$ARIA" \
    2>&1 | tee -a "$LOGS/stageD.log"
hf upload "$HF_REPO" "$RUNS/stageD_demo.safetensors" \
    "aria_finetune/${VARIANT}/stageD/demo.safetensors" --repo-type model 2>&1 | tail -2 || true
hf upload "$HF_REPO" "$RUNS/stageD" \
    "aria_finetune/${VARIANT}/stageD/run_dir" --repo-type model 2>&1 | tail -2 || true

say "=== ${VARIANT} ALL STAGES DONE ==="
pkill -f "/tmp/${VARIANT}/gpu_sidecar" 2>/dev/null || true
