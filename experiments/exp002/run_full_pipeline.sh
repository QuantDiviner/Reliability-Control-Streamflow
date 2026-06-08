#!/usr/bin/env bash
# exp002 full pipeline orchestration (D-014):
#   1. Train 30 epochs × 671 basins
#   2. Evaluate validation period (full 671) → cal scores
#   3. Evaluate test period (full 671)
#   4. Run HSCC analysis → metrics.json + hscc_results.csv
#
# Run inside tmux session "exp002_full" (per user preference: tmux for long jobs).
# All output tee'd to logs/full_*.log so progress can be inspected with Read tool.

set -e

PROJ_DIR="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
LOG_DIR="$PROJ_DIR/experiments/exp002/logs"
CONFIG="$PROJ_DIR/experiments/exp002/config_full.yml"
RUN_PARENT="$PROJ_DIR/experiments/exp002/results"

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR"

source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology

echo "=========================================="
echo "exp002 FULL PIPELINE START — $(date +'%F %T')"
echo "=========================================="

# Step 1: Train
TRAIN_LOG="$LOG_DIR/full_train.log"
echo "[$(date +'%F %T')] STEP 1/4 train — log: $TRAIN_LOG"
nh-run train --config-file "$CONFIG" 2>&1 | tee "$TRAIN_LOG"

# Discover the new run dir (latest under results/ matching exp002_camels_us_temporal_*)
RUN_DIR=$(ls -1dt "$RUN_PARENT"/exp002_camels_us_temporal_* 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: cannot find run dir under $RUN_PARENT" | tee -a "$LOG_DIR/full_pipeline.log"
    exit 1
fi
echo "[$(date +'%F %T')] run dir: $RUN_DIR" | tee -a "$LOG_DIR/full_pipeline.log"

# Step 2: Evaluate validation (calibration scores for HSCC)
VAL_LOG="$LOG_DIR/full_eval_validation.log"
echo "[$(date +'%F %T')] STEP 2/4 evaluate validation period — log: $VAL_LOG"
nh-run evaluate --run-dir "$RUN_DIR" --period validation --epoch 30 2>&1 | tee "$VAL_LOG"

# Step 3: Evaluate test (test obs/pred)
TEST_LOG="$LOG_DIR/full_eval_test.log"
echo "[$(date +'%F %T')] STEP 3/4 evaluate test period — log: $TEST_LOG"
nh-run evaluate --run-dir "$RUN_DIR" --period test --epoch 30 2>&1 | tee "$TEST_LOG"

# Step 4: HSCC analysis
ANALYSIS_LOG="$LOG_DIR/full_hscc_analysis.log"
echo "[$(date +'%F %T')] STEP 4/4 HSCC analysis — log: $ANALYSIS_LOG"
python "$PROJ_DIR/experiments/exp002/hscc_analysis_v2.py" \
    --run_dir "$RUN_DIR" \
    --out_dir "$RUN_DIR/_analysis" 2>&1 | tee "$ANALYSIS_LOG"

echo "=========================================="
echo "exp002 FULL PIPELINE DONE — $(date +'%F %T')"
echo "run dir: $RUN_DIR"
echo "outputs: $RUN_DIR/_analysis/{metrics.json,hscc_results.csv,global_cp_results.csv}"
echo "=========================================="
