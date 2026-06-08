#!/usr/bin/env bash
# Post-review item 4 pipeline: larger-backbone robustness on CAMELS-US temporal split.
#   1. Train hidden=256, 50 epochs, 671 basins, seed 42
#   2. Evaluate validation period (epoch 50) -> calibration scores
#   3. Evaluate test period (epoch 50) -> test obs/pred
#   4. Reuse exp002 hscc_analysis_v2.py -> metrics.json + hscc_results.csv + global_cp_results.csv
# Run inside tmux "item4_h256". Output tee'd to logs/ for inspection.

set -e

PROJ_DIR="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
BASE="$PROJ_DIR/experiments/post_review_wrr_enhancements/item4_larger_backbone"
LOG_DIR="$BASE/logs"
CONFIG="$BASE/config_hidden256.yml"
RUN_PARENT="$BASE/results"
EPOCH=50

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR"

source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology

echo "=========================================="
echo "item4 larger-backbone PIPELINE START — $(date +'%F %T')"
echo "=========================================="

# Step 1: Train
TRAIN_LOG="$LOG_DIR/train.log"
echo "[$(date +'%F %T')] STEP 1/4 train (hidden=256, 50ep) — log: $TRAIN_LOG"
nh-run train --config-file "$CONFIG" 2>&1 | tee "$TRAIN_LOG"

# Discover the new run dir
RUN_DIR=$(ls -1dt "$RUN_PARENT"/item4_larger_backbone_h256_* 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: cannot find run dir under $RUN_PARENT" | tee -a "$LOG_DIR/pipeline.log"
    exit 1
fi
echo "[$(date +'%F %T')] run dir: $RUN_DIR" | tee -a "$LOG_DIR/pipeline.log"

# Step 2: Evaluate validation (calibration scores for HSCC)
VAL_LOG="$LOG_DIR/eval_validation.log"
echo "[$(date +'%F %T')] STEP 2/4 evaluate validation — log: $VAL_LOG"
nh-run evaluate --run-dir "$RUN_DIR" --period validation --epoch $EPOCH 2>&1 | tee "$VAL_LOG"

# Step 3: Evaluate test
TEST_LOG="$LOG_DIR/eval_test.log"
echo "[$(date +'%F %T')] STEP 3/4 evaluate test — log: $TEST_LOG"
nh-run evaluate --run-dir "$RUN_DIR" --period test --epoch $EPOCH 2>&1 | tee "$TEST_LOG"

# Step 4: HSCC analysis (reuse exp002 script)
ANALYSIS_LOG="$LOG_DIR/hscc_analysis.log"
echo "[$(date +'%F %T')] STEP 4/4 HSCC analysis — log: $ANALYSIS_LOG"
python "$PROJ_DIR/experiments/exp002/hscc_analysis_v2.py" \
    --run_dir "$RUN_DIR" \
    --out_dir "$RUN_DIR/_analysis" 2>&1 | tee "$ANALYSIS_LOG"

echo "=========================================="
echo "item4 larger-backbone PIPELINE DONE — $(date +'%F %T')"
echo "run dir: $RUN_DIR"
echo "outputs: $RUN_DIR/_analysis/{metrics.json,hscc_results.csv,global_cp_results.csv}"
echo "=========================================="
