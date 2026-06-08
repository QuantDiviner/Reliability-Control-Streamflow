#!/usr/bin/env bash
# exp006 v2 Phase A0 CANARY pipeline (D-026, 2026-04-28):
#   1. Train MDN (cudalstm + CMAL head, K=5) — 671 basins × 30 epochs (~6h GPU on RTX 4070)
#   2. Evaluate validation period (Klotz 2022 reference; sanity check)
#   3. Evaluate test period (head-to-head with HSCC on identical exp002 split)
#   4. Run analyze_mdn.py → per-tier reliability + sharpness + Pareto data
#
# === CANARY-FIRST PROTOCOL (R3 修订 #2) ===
# This is the seed=42 single-seed canary run. After completion, **STOP** and
# wait for Decision Gate G evaluation:
#   - test mean basin NSE >= 0.65 (vs exp002 0.738)？
#     YES → 启动 Phase A1 (3-seed full: seeds 137 / 2024) + Phase B1 (K sweep K=3, 8)
#     NO  → 触发 Phase A3 (NSE 减半根因诊断) + 评估 Phase D fallback (Route D)
#
# v1 (GMM K=3) data preserved as supplementary "GMM K=3 reference run":
#   experiments/exp006/results/exp006_mdn_camels_us_2704_232730/
#
# Run inside tmux session "exp006_canary" (project rule: long-running tasks use tmux).
# All output tee'd to logs/full_*.log so progress can be inspected via Read tool.

set -eo pipefail   # pipefail: tee 不再吞 nh-run 的非 0 退出码（v2a NaN 崩溃后 evaluate/analyze 仍执行的根因）

PROJ_DIR="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
EXP_DIR="$PROJ_DIR/experiments/exp006"
LOG_DIR="$EXP_DIR/logs"
CONFIG="$EXP_DIR/configs/exp006_mdn.yml"
RUN_PARENT="$EXP_DIR/results"
TIERS_CSV="$PROJ_DIR/experiments/exp002/basin_tiers.csv"

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR"

source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology
# Force hscc-hydrology bin ahead of r-reticulate (which is hard-pinned in user PATH)
export PATH="/home/qingsong/miniconda3/envs/hscc-hydrology/bin:$PATH"
PY="/home/qingsong/miniconda3/envs/hscc-hydrology/bin/python"

echo "=========================================="
echo "exp006 v2 Phase A0 CANARY START — $(date +'%F %T')"
echo "  head=cmal, K=5, seed=42 (D-026)"
echo "  After completion: STOP for Decision Gate G (NSE >= 0.65?)"
echo "=========================================="

# Step 1: Train
TRAIN_LOG="$LOG_DIR/full_train.log"
echo "[$(date +'%F %T')] STEP 1/4 train MDN (CMAL K=5, 30 epochs, 671 basins) — log: $TRAIN_LOG"
nh-run train --config-file "$CONFIG" 2>&1 | tee "$TRAIN_LOG"

# Discover the new run dir (latest under results/ matching exp006_mdn_camels_us_*)
RUN_DIR=$(ls -1dt "$RUN_PARENT"/exp006_mdn_camels_us_* 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: cannot find run dir under $RUN_PARENT" | tee -a "$LOG_DIR/full_pipeline.log"
    exit 1
fi
echo "[$(date +'%F %T')] run dir: $RUN_DIR" | tee -a "$LOG_DIR/full_pipeline.log"

# Step 2: Evaluate validation (sanity, Klotz 2022 protocol cross-check)
VAL_LOG="$LOG_DIR/full_eval_validation.log"
echo "[$(date +'%F %T')] STEP 2/4 evaluate validation — log: $VAL_LOG"
nh-run evaluate --run-dir "$RUN_DIR" --period validation --epoch 30 2>&1 | tee "$VAL_LOG"

# Step 3: Evaluate test (primary head-to-head period)
TEST_LOG="$LOG_DIR/full_eval_test.log"
echo "[$(date +'%F %T')] STEP 3/4 evaluate test — log: $TEST_LOG"
nh-run evaluate --run-dir "$RUN_DIR" --period test --epoch 30 2>&1 | tee "$TEST_LOG"

# Step 4: MDN per-tier analysis (test period, primary metrics)
ANALYSIS_LOG="$LOG_DIR/full_analyze.log"
echo "[$(date +'%F %T')] STEP 4/4 analyze test — log: $ANALYSIS_LOG"
"$PY" "$EXP_DIR/scripts/analyze_mdn.py" \
    --run_dir "$RUN_DIR" \
    --tiers "$TIERS_CSV" \
    --period test \
    --epoch 30 \
    --alpha 0.10 \
    --out_dir "$RUN_DIR/_analysis" 2>&1 | tee "$ANALYSIS_LOG"

# Optional: also analyze validation period for QA
"$PY" "$EXP_DIR/scripts/analyze_mdn.py" \
    --run_dir "$RUN_DIR" \
    --tiers "$TIERS_CSV" \
    --period validation \
    --epoch 30 \
    --alpha 0.10 \
    --out_dir "$RUN_DIR/_analysis_validation" 2>&1 | tee -a "$ANALYSIS_LOG"

echo "=========================================="
echo "exp006 v2 Phase A0 CANARY DONE — $(date +'%F %T')"
echo "run dir: $RUN_DIR"
echo "outputs: $RUN_DIR/_analysis/{metrics_per_tier.json,pareto_data.csv,mdn_intervals.csv,basin_nse.csv}"
echo ""
echo "*** NEXT STEP: Decision Gate G ***"
echo "Check $RUN_DIR/test/model_epoch030/test_metrics.csv for mean basin NSE."
echo "  NSE >= 0.65 → trigger Phase A1 (3-seed) + B1 (K sweep)"
echo "  NSE <  0.65 → trigger Phase A3 root-cause diagnosis (and possibly Route D)"
echo "=========================================="
