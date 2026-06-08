#!/usr/bin/env bash
# exp006 Phase A1 (D-028) — 2 additional seeds for cross-seed std.
#
# seed=42 already complete as v3 canary:
#   experiments/exp006/results/exp006_mdn_camels_us_2904_155205/
#
# This runner does seed=1337 then seed=2024 sequentially with full pipeline
# (train + eval val + eval test + analyze test/val). validate_every=10 so each
# run is ~13h instead of v3's ~18h. Total ~26h sequential.
#
# Pass criterion (D-028):
#   - 3 seeds × 30 epoch all complete with no NaN
#   - Cross-seed std on per-tier coverage spread ≤ 1.0 pp
#   - Cross-seed std on test mean basin NSE ≤ 0.01
#
# Run inside tmux session "exp006_phase_a1".

set -eo pipefail

PROJ_DIR="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
EXP_DIR="$PROJ_DIR/experiments/exp006"
LOG_DIR="$EXP_DIR/logs"
RUN_PARENT="$EXP_DIR/results"
TIERS_CSV="$PROJ_DIR/experiments/exp002/basin_tiers.csv"

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR"

source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology
export PATH="/home/qingsong/miniconda3/envs/hscc-hydrology/bin:$PATH"
PY="/home/qingsong/miniconda3/envs/hscc-hydrology/bin/python"

run_one_seed() {
    local SEED="$1"
    local CONFIG="$EXP_DIR/configs/exp006_mdn_seed${SEED}.yml"
    local TAG="seed${SEED}"

    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: config not found: $CONFIG" >&2
        return 1
    fi

    echo "=========================================="
    echo "Phase A1 — seed=${SEED} START — $(date +'%F %T')"
    echo "  config: $CONFIG"
    echo "=========================================="

    # Step 1: Train
    local TRAIN_LOG="$LOG_DIR/phase_a1_${TAG}_train.log"
    nh-run train --config-file "$CONFIG" 2>&1 | tee "$TRAIN_LOG"

    # Discover the new run dir matching this seed
    local RUN_DIR=$(ls -1dt "$RUN_PARENT"/exp006_mdn_camels_us_${TAG}_* 2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: cannot find run dir for ${TAG}" >&2
        return 1
    fi
    echo "[$(date +'%F %T')] ${TAG} run dir: $RUN_DIR"

    # Step 2: Evaluate validation
    local VAL_LOG="$LOG_DIR/phase_a1_${TAG}_eval_validation.log"
    nh-run evaluate --run-dir "$RUN_DIR" --period validation --epoch 30 2>&1 | tee "$VAL_LOG"

    # Step 3: Evaluate test
    local TEST_LOG="$LOG_DIR/phase_a1_${TAG}_eval_test.log"
    nh-run evaluate --run-dir "$RUN_DIR" --period test --epoch 30 2>&1 | tee "$TEST_LOG"

    # Step 4: Analyze test + validation
    local ANALYZE_LOG="$LOG_DIR/phase_a1_${TAG}_analyze.log"
    "$PY" "$EXP_DIR/scripts/analyze_mdn.py" \
        --run_dir "$RUN_DIR" \
        --tiers "$TIERS_CSV" \
        --period test \
        --epoch 30 \
        --alpha 0.10 \
        --out_dir "$RUN_DIR/_analysis" 2>&1 | tee "$ANALYZE_LOG"

    "$PY" "$EXP_DIR/scripts/analyze_mdn.py" \
        --run_dir "$RUN_DIR" \
        --tiers "$TIERS_CSV" \
        --period validation \
        --epoch 30 \
        --alpha 0.10 \
        --out_dir "$RUN_DIR/_analysis_validation" 2>&1 | tee -a "$ANALYZE_LOG"

    echo "=========================================="
    echo "Phase A1 — seed=${SEED} DONE — $(date +'%F %T')"
    echo "run dir: $RUN_DIR"
    echo "=========================================="
}

echo "=========================================="
echo "Phase A1 START — $(date +'%F %T')"
echo "  Seeds: 1337, 2024 (seed=42 already done as v3 canary)"
echo "=========================================="

run_one_seed 1337
run_one_seed 2024

echo "=========================================="
echo "Phase A1 ALL SEEDS DONE — $(date +'%F %T')"
echo "Next: cross-seed aggregation + Decision Gate G evaluation"
echo "=========================================="
