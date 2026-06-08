#!/usr/bin/env bash
# exp004 Action 4 — 3-seed × 30 epoch on 50 GB basins.
# seed=42 already done (re-uses existing run_dir); this script trains seed=137 + seed=2024.
#
# tmux new-session -d -s exp004_3seed "bash experiments/exp004/scripts/run_3seed_action4.sh"

set -eo pipefail

PROJ="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
LOG_DIR="$PROJ/experiments/exp004/logs"
RESULTS_ROOT="$PROJ/experiments/exp004/results"
mkdir -p "$LOG_DIR"
cd "$PROJ"

source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology

echo "=========================================="
echo "exp004 Action 4: 3-seed pipeline START — $(date +'%F %T')"
echo "=========================================="

for SEED in 137 2024; do
    CONFIG="$PROJ/experiments/exp004/configs/exp004_camels_gb_seed${SEED}.yml"
    echo ""
    echo "[$(date +'%F %T')] === SEED ${SEED} ==="
    echo "  config: $CONFIG"

    # Step 1: train
    echo "[$(date +'%F %T')]   train (30 epochs) ..."
    nh-run train --config-file "$CONFIG" 2>&1 \
        | tee "$LOG_DIR/train_seed${SEED}.log"

    # Find the run_dir created by NH (latest matching glob)
    RUN_DIR=$(ls -dt "$RESULTS_ROOT"/exp004_camels_gb_native_seed${SEED}_* | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "[FAIL] no run_dir found for seed ${SEED}"
        exit 2
    fi
    echo "  run_dir: $RUN_DIR"

    # Step 2: evaluate validation (calibration period)
    echo "[$(date +'%F %T')]   evaluate validation period ..."
    nh-run evaluate --run-dir "$RUN_DIR" --period validation --epoch 30 2>&1 \
        | tee "$LOG_DIR/eval_val_seed${SEED}.log"

    # Step 3: evaluate test period
    echo "[$(date +'%F %T')]   evaluate test period ..."
    nh-run evaluate --run-dir "$RUN_DIR" --period test --epoch 30 2>&1 \
        | tee "$LOG_DIR/eval_test_seed${SEED}.log"

    # Step 4: HSCC analysis (uses post-PCR-004 GB tier names: gb_drier_q4 / gb_montane)
    echo "[$(date +'%F %T')]   HSCC analysis ..."
    OUT_DIR="$RUN_DIR/_analysis"
    mkdir -p "$OUT_DIR"
    python "$PROJ/experiments/exp004/hscc_analysis_gb.py" \
        --run_dir "$RUN_DIR" --out_dir "$OUT_DIR" 2>&1 \
        | tee "$LOG_DIR/hscc_analysis_seed${SEED}.log"

    echo "[$(date +'%F %T')]   SEED ${SEED} done. metrics → $OUT_DIR/metrics.json"
done

echo ""
echo "=========================================="
echo "[$(date +'%F %T')] all 2 new seeds complete; running 3-seed aggregator"
echo "=========================================="

python "$PROJ/experiments/exp004/scripts/aggregate_3seed.py" 2>&1 \
    | tee "$LOG_DIR/aggregate_3seed.log"

echo ""
echo "=========================================="
echo "exp004 Action 4 COMPLETE — $(date +'%F %T')"
echo "outputs:"
echo "  $RESULTS_ROOT/_3seed_aggregate/{summary.json, per_tier_table.csv, bootstrap_ci.json}"
echo "=========================================="
