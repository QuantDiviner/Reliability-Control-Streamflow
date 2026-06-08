#!/usr/bin/env bash
# exp006 SMOKE test — 5 basins × 1 epoch GMM training + eval + analysis sanity.
# Should finish in ~3 min on RTX 4070.

set -e

PROJ_DIR="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
EXP_DIR="$PROJ_DIR/experiments/exp006"
LOG_DIR="$EXP_DIR/logs"
CONFIG="$EXP_DIR/configs/exp006_mdn_smoke.yml"
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
echo "exp006 SMOKE START — $(date +'%F %T')"
echo "=========================================="

SMOKE_LOG="$LOG_DIR/smoke.log"

nh-run train --config-file "$CONFIG" 2>&1 | tee "$SMOKE_LOG"

RUN_DIR=$(ls -1dt "$RUN_PARENT"/exp006_mdn_smoke_* 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: cannot find smoke run dir" | tee -a "$SMOKE_LOG"
    exit 1
fi
echo "smoke run dir: $RUN_DIR" | tee -a "$SMOKE_LOG"

nh-run evaluate --run-dir "$RUN_DIR" --period test --epoch 1 2>&1 | tee -a "$SMOKE_LOG"

"$PY" "$EXP_DIR/scripts/analyze_mdn.py" \
    --run_dir "$RUN_DIR" \
    --tiers "$TIERS_CSV" \
    --period test \
    --epoch 1 \
    --alpha 0.10 \
    --out_dir "$RUN_DIR/_analysis" 2>&1 | tee -a "$SMOKE_LOG"

echo "=========================================="
echo "exp006 SMOKE DONE — $(date +'%F %T')"
echo "outputs: $RUN_DIR/_analysis/"
echo "=========================================="
