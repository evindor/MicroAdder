#!/usr/bin/env bash
set -euo pipefail

SEEDS=(42 69 420 12345 1337 1995 2026 707 777 10 666 999)
PARALLEL=3

run_seed() {
    local seed=$1
    local name="sub100_exp3_d5h1_141p_wd10_s${seed}"
    local log="results/runs/${name}.log"
    echo "[$(date +%H:%M:%S)] Starting seed $seed -> $log"
    uv run python -m src.train --run-name "$name" \
        --d-model 5 --tok-dim 2 --pos-dim 3 --n-heads 1 --head-dim 5 \
        --tie-qk --q-phase --pos-mode spiral_correct --attn-out-rank 2 --pos-correction-mode linear \
        --freeze-special eos --wd-adaptive --wd-drop-exact 0.10 --wd-drop-exact-final 0.50 \
        --seed "$seed" --steps 600000 --lr 0.02 \
        --carry-mix 0.3 --carry-mix-tok-acc-fade 0.8 --carry-mix-max-steps 300000 \
        > "$log" 2>&1
    local status=$?
    local last_eval=$(grep "EVAL" "$log" | tail -1)
    echo "[$(date +%H:%M:%S)] Finished seed $seed (exit=$status) | $last_eval"
}

echo "=== d_model=5, 1h, 141p sweep — $(date) ==="
echo "Seeds: ${SEEDS[*]}"
echo "Parallel: $PARALLEL"
echo ""

for ((i=0; i<${#SEEDS[@]}; i+=PARALLEL)); do
    batch=("${SEEDS[@]:i:PARALLEL}")
    echo "--- Batch: ${batch[*]} ---"
    pids=()
    for seed in "${batch[@]}"; do
        run_seed "$seed" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo ""
done

echo "=== Sweep complete — $(date) ==="
echo ""
echo "Summary:"
for seed in "${SEEDS[@]}"; do
    log="results/runs/sub100_exp3_d5h1_141p_wd10_s${seed}.log"
    best=$(grep "NEW BEST" "$log" 2>/dev/null | tail -1 || echo "no best")
    last=$(grep "EVAL" "$log" 2>/dev/null | tail -1 || echo "no evals")
    echo "  seed $seed | $best | $last"
done
