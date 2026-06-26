#!/bin/bash
# Patience-based early stop for Aria train.py.
#
# Aria's train.py has no built-in early stop. This watcher polls
# `epoch.csv` every 30s, tracks the running val-best, and if the
# val hasn't improved for `patience` epochs it sends SIGINT to the
# tmux session running the trainer.
#
# Usage:
#   bash scripts/aria_early_stop_watcher.sh <pdir> <patience> <tmux-session>
#
# After SIGINT the val-best ckpt is preserved by the sweeper.

set -u
PDIR="${1:?usage: aria_early_stop_watcher.sh <pdir> <patience> <tmux-session>}"
PATIENCE="${2:-4}"
SESSION="${3:?need tmux session name}"

best_val="inf"
best_epoch=-1
last_seen_epoch=-1
no_improve=0

while true; do
    CSV="$PDIR/epoch.csv"
    if [ -s "$CSV" ]; then
        # Find row of last epoch
        last_row=$(tail -1 "$CSV")
        epoch_n=$(echo "$last_row" | cut -d',' -f1)
        val=$(echo "$last_row" | cut -d',' -f3)
        if [[ "$epoch_n" =~ ^[0-9]+$ ]] && [ "$epoch_n" -gt "$last_seen_epoch" ]; then
            last_seen_epoch="$epoch_n"
            # Compare val to best_val
            # Compare floats with awk — robust to the "inf" sentinel and to
            # bash's lack of float arithmetic. The earlier python3 -c
            # approach silently returned empty on every iteration so best_val
            # stayed at "inf" forever and patience fired at exactly
            # PATIENCE epochs regardless of val_loss. Observed 2026-06-03
            # 00:30 UTC — same bug retroactively explains why every
            # first-cohort Aria variant capped at exactly 4 epochs (we
            # mis-attributed it to NUM_EPOCHS_TOK=4 data exhaustion).
            improved=$(awk -v v="$val" -v b="$best_val" 'BEGIN {
                if (b == "inf") { print "yes"; exit }
                if (v + 0 < b + 0) print "yes"; else print "no"
            }')
            if [ "$improved" = "yes" ]; then
                best_val="$val"
                best_epoch="$epoch_n"
                no_improve=0
                echo "[earlystop] new best at epoch=$epoch_n val=$val"
            else
                no_improve=$(( no_improve + 1 ))
                echo "[earlystop] epoch=$epoch_n val=$val (best=$best_val@$best_epoch) no_improve=$no_improve/$PATIENCE"
                if [ "$no_improve" -ge "$PATIENCE" ]; then
                    echo "[earlystop] patience exhausted, sending SIGINT to tmux $SESSION"
                    tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
                    echo "[earlystop] DONE; best epoch=$best_epoch best val=$best_val"
                    exit 0
                fi
            fi
        fi
    fi
    sleep 30
done
