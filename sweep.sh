#!/usr/bin/env bash
set -euo pipefail

COMMON="--lr 0.02 --warmup 2000 --weight-decay 0.01 --batch-size 512 --eval-every 1000 --eval-samples 2048"

for SEED in 1337 69420 256 777 2026 1991 373 1414 999 555; do
    echo "=== Starting seed $SEED ==="
    uv run python -m src.train --run-name expA_night_242p --seed $SEED --steps 500000 $COMMON
    echo "=== Finished seed $SEED ==="
done
