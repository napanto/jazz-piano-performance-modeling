#!/bin/bash
# Aria accelerate-ckpt sweeper.
#
# Aria's train.py saves a 7.4 GB accelerate state dir per epoch which
# would bust a 50 GB volume after ~6 epochs. This sweeper keeps:
#   - the two most recent ckpts (training process is writing the latest)
#   - the ckpt corresponding to the *current val-best* epoch
#
# Best-epoch lookup: read `epoch.csv` (Aria's stock per-epoch CSV) and
# find the row with minimum `avg_val_loss`. Aria names a checkpoint
# saved AFTER epoch N as `epochN+1_step0/`. So if best val is at
# epoch.csv row "epoch=5", the ckpt to preserve is `epoch6_step0/`.
#
# Usage:
#   bash scripts/sweep_ckpts_best.sh <pdir>
#
# where <pdir> is the Aria train.py --pdir.

set -u
PDIR="${1:?usage: sweep_ckpts_best.sh <pdir>}"

while true; do
    CK_ROOT="$PDIR/checkpoints"
    CSV="$PDIR/epoch.csv"
    if [ -d "$CK_ROOT" ]; then
        cd "$CK_ROOT" || { sleep 30; continue; }
        # Two latest by mtime.
        ALL=( $(ls -td */ 2>/dev/null) )
        KEEP=()
        # Keep top-2
        for d in "${ALL[@]:0:2}"; do
            KEEP+=("$d")
        done
        # Plus best-val ckpt, if epoch.csv exists.
        if [ -s "$CSV" ]; then
            BEST_EPOCH=$(awk -F',' '
                NR==1 { for (i=1;i<=NF;i++) if ($i=="avg_val_loss") c=i; if(!c) c=3; next }
                {
                    if (best == "" || $c < best) { best=$c; bi=$1 }
                }
                END { print bi }
            ' "$CSV" 2>/dev/null)
            if [ -n "$BEST_EPOCH" ]; then
                # Aria's naming convention: after epoch N finishes the next
                # ckpt is epoch{N+1}_step0/. Account for 0-indexed CSV.
                BEST_CK="epoch$((BEST_EPOCH + 1))_step0/"
                if [ -d "$BEST_CK" ]; then
                    # Only add if not already in KEEP[]
                    in_keep=0
                    for k in "${KEEP[@]}"; do
                        [ "$k" = "$BEST_CK" ] && in_keep=1
                    done
                    if [ "$in_keep" = "0" ]; then
                        KEEP+=("$BEST_CK")
                    fi
                fi
            fi
        fi
        # Remove anything not in KEEP[].
        for d in "${ALL[@]}"; do
            in_keep=0
            for k in "${KEEP[@]}"; do
                [ "$k" = "$d" ] && in_keep=1
            done
            if [ "$in_keep" = "0" ]; then
                rm -rf "$d"
            fi
        done
    fi
    sleep 60
done
