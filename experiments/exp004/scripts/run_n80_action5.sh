#!/usr/bin/env bash
# exp004 Action 5 — 80-basin GB-native training + HSCC analysis.
# Replaces gb_50.txt with gb_80.txt; gb_montane n: 5 → 8 (P1-1 sample power fix).
# Uses tier swap to protect the 50-basin pipeline's gb_basin_tiers.csv default.
#
# tmux new-session -d -s exp004_n80 "bash experiments/exp004/scripts/run_n80_action5.sh"

set -eo pipefail

PROJ="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
LOG_DIR="$PROJ/experiments/exp004/logs"
RESULTS_ROOT="$PROJ/experiments/exp004/results"
TIERS_DIR="$PROJ/experiments/exp004/basin_lists"
CONFIG="$PROJ/experiments/exp004/configs/exp004_camels_gb_n80.yml"
mkdir -p "$LOG_DIR"
cd "$PROJ"

source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology

echo "=========================================="
echo "exp004 Action 5: 80-basin pipeline START — $(date +'%F %T')"
echo "=========================================="

# Step 1: train (seed=42, 30 epoch on 80 basins)
echo "[$(date +'%F %T')] STEP 1/4 train (seed=42, 80 basins, 30 epoch) ..."
nh-run train --config-file "$CONFIG" 2>&1 | tee "$LOG_DIR/train_n80.log"

RUN_DIR=$(ls -dt "$RESULTS_ROOT"/exp004_camels_gb_native_n80_* | head -1)
[ -z "$RUN_DIR" ] && { echo "[FAIL] no run_dir for n80"; exit 2; }
echo "  run_dir: $RUN_DIR"

# Step 2: evaluate validation
echo "[$(date +'%F %T')] STEP 2/4 evaluate validation period ..."
nh-run evaluate --run-dir "$RUN_DIR" --period validation --epoch 30 2>&1 \
    | tee "$LOG_DIR/eval_val_n80.log"

# Step 3: evaluate test
echo "[$(date +'%F %T')] STEP 3/4 evaluate test period ..."
nh-run evaluate --run-dir "$RUN_DIR" --period test --epoch 30 2>&1 \
    | tee "$LOG_DIR/eval_test_n80.log"

# Step 4: HSCC analysis with tier swap (80-basin tiers → default → run → restore)
echo "[$(date +'%F %T')] STEP 4/4 HSCC analysis (with tier swap) ..."
OUT_DIR="$RUN_DIR/_analysis"
mkdir -p "$OUT_DIR"

# Swap in 80-basin tiers
cp "$TIERS_DIR/gb_basin_tiers.csv" "$TIERS_DIR/gb_basin_tiers.csv.bak_50_during_n80"
cp "$TIERS_DIR/gb_basin_tiers_80basins.csv" "$TIERS_DIR/gb_basin_tiers.csv"
echo "  swapped to 80-basin tier mapping"

# Run analysis
python "$PROJ/experiments/exp004/hscc_analysis_gb.py" \
    --run_dir "$RUN_DIR" --out_dir "$OUT_DIR" 2>&1 \
    | tee "$LOG_DIR/hscc_analysis_n80.log"

# Restore 50-basin default
cp "$TIERS_DIR/gb_basin_tiers.csv.bak_50_during_n80" "$TIERS_DIR/gb_basin_tiers.csv"
rm "$TIERS_DIR/gb_basin_tiers.csv.bak_50_during_n80"
echo "  restored 50-basin default tier mapping"

echo ""
echo "=========================================="
echo "exp004 Action 5 COMPLETE — $(date +'%F %T')"
echo "outputs:"
echo "  $OUT_DIR/{metrics.json, hscc_results.csv}"
echo "  basin list: $TIERS_DIR/gb_80.txt (80 basins; gb_montane: 5→8)"
echo "  tier mapping: $TIERS_DIR/gb_basin_tiers_80basins.csv"
echo "=========================================="
