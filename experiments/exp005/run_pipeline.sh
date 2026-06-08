#!/usr/bin/env bash
# exp005 Lean MVP — sequential C0 then C3 full pipeline.
#
# Each condition: train + eval validation + eval test.
# 20 basins × 30 epoch each → estimated ~30-60 min per condition on RTX 4070.
# Total walltime: ~1-2 hours.

set -eo pipefail

PROJ_DIR="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
EXP_DIR="$PROJ_DIR/experiments/exp005"
LOG_DIR="$EXP_DIR/logs"
RUN_PARENT="$EXP_DIR/results"

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR"

source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology
export PATH="/home/qingsong/miniconda3/envs/hscc-hydrology/bin:$PATH"
PY="/home/qingsong/miniconda3/envs/hscc-hydrology/bin/python"

run_one_condition() {
    local COND="$1"
    local CONFIG="$EXP_DIR/configs/exp005_${COND}.yml"
    local TAG="${COND}"

    echo "=========================================="
    echo "exp005 — ${COND} START — $(date +'%F %T')"
    echo "  config: $CONFIG"
    echo "=========================================="

    local TRAIN_LOG="$LOG_DIR/${COND}_train.log"
    nh-run train --config-file "$CONFIG" 2>&1 | tee "$TRAIN_LOG"

    local PATTERN="exp005_${COND}_*"
    local RUN_DIR=$(ls -1dt $RUN_PARENT/${PATTERN} 2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
        # Fallback: NH names runs based on experiment_name
        if [ "$COND" = "c0" ]; then
            RUN_DIR=$(ls -1dt "$RUN_PARENT"/exp005_c0_clean_* 2>/dev/null | head -1)
        elif [ "$COND" = "c3" ]; then
            RUN_DIR=$(ls -1dt "$RUN_PARENT"/exp005_c3_perturbed_* 2>/dev/null | head -1)
        fi
    fi
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: cannot find run dir for ${COND}" >&2
        return 1
    fi
    echo "[${COND}] run dir: $RUN_DIR"

    local VAL_LOG="$LOG_DIR/${COND}_eval_validation.log"
    nh-run evaluate --run-dir "$RUN_DIR" --period validation --epoch 30 2>&1 | tee "$VAL_LOG"

    local TEST_LOG="$LOG_DIR/${COND}_eval_test.log"
    nh-run evaluate --run-dir "$RUN_DIR" --period test --epoch 30 2>&1 | tee "$TEST_LOG"

    echo "=========================================="
    echo "exp005 — ${COND} DONE — $(date +'%F %T')"
    echo "  run dir: $RUN_DIR"
    echo "=========================================="
}

echo "=========================================="
echo "exp005 Lean MVP START — $(date +'%F %T')"
echo "  conditions: c0 (clean), c3 (P+T perturbed)"
echo "=========================================="

run_one_condition c0
run_one_condition c3

echo "=========================================="
echo "exp005 Lean MVP ALL CONDITIONS DONE — $(date +'%F %T')"
echo "Next: analyze_residuals.py for AR1 / coverage / Wilcoxon"
echo "=========================================="
