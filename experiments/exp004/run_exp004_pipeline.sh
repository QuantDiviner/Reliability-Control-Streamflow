#!/usr/bin/env bash
# exp004 auto pipeline: 等下载完成 → 解压 → 选 basin → 训练 → evaluate.
# HSCC analysis 暂不放进来（需要先 refactor hscc_analysis_v2.py 解耦 CAMELS-US attribute 路径）。
#
# 在 tmux session "exp004_pipe" 中运行:
#   tmux new-session -d -s exp004_pipe "bash experiments/exp004/run_exp004_pipeline.sh"

set -eo pipefail
# 注意：不使用 -u（unset 变量严格模式），因为 conda activate.d 里某些 hook
# (libblas_mkl_activate.sh) 会触发 MKL_INTERFACE_LAYER unbound 错误。

PROJ_DIR="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
GB_DIR="$PROJ_DIR/data/raw/CAMELS_GB"
ZIP_FILE="$GB_DIR/CAMELS_GB.zip"
DONE_FLAG="$GB_DIR/.download_done"
EXTRACT_FLAG="$GB_DIR/.extract_done"
LOG_DIR="$PROJ_DIR/experiments/exp004/logs"
CONFIG="$PROJ_DIR/experiments/exp004/configs/exp004_camels_gb.yml"
SELECT_SCRIPT="$PROJ_DIR/experiments/exp004/scripts/select_gb_basins.py"
BASIN_LIST="$PROJ_DIR/experiments/exp004/basin_lists/gb_50.txt"

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR"

source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology

echo "=========================================="
echo "exp004 PIPELINE START — $(date +'%F %T')"
echo "=========================================="

# Step 0: wait for download flag (set by scripts/download_camels_gb.sh)
echo "[$(date +'%F %T')] STEP 0/4 waiting for $DONE_FLAG ..."
WAIT_SEC=0
while [ ! -f "$DONE_FLAG" ]; do
    sleep 60
    WAIT_SEC=$((WAIT_SEC + 60))
    if [ $((WAIT_SEC % 300)) -eq 0 ]; then
        SIZE=$(du -h "$GB_DIR/CAMELS_GB.zip.partial" 2>/dev/null | cut -f1 || echo "0")
        echo "[$(date +'%F %T')]   ... waited ${WAIT_SEC}s, partial=$SIZE"
    fi
    if [ $WAIT_SEC -gt 14400 ]; then  # 4 hours hard cap
        echo "[FAIL] download did not finish in 4 hours; abort."
        exit 1
    fi
done
echo "[$(date +'%F %T')] STEP 0/4 download complete."

# Step 1: extract + normalize layout for NeuralHydrology compatibility
if [ ! -f "$EXTRACT_FLAG" ]; then
    echo "[$(date +'%F %T')] STEP 1/4 extracting $ZIP_FILE ..."
    cd "$GB_DIR"
    unzip -q -o "$ZIP_FILE"
    cd "$PROJ_DIR"
    touch "$EXTRACT_FLAG"
    echo "[$(date +'%F %T')] STEP 1/4 extracted."
else
    echo "[$(date +'%F %T')] STEP 1/4 already extracted (.extract_done present)."
fi

# CAMELS-GB zip puts *_attributes.csv at $GB_DIR/data/ root, but NH camelsgb.py
# expects them under <data_dir>/attributes/. Normalize once.
if [ -d "$GB_DIR/data" ] && [ ! -d "$GB_DIR/data/attributes" ]; then
    echo "[$(date +'%F %T')] STEP 1/4 normalizing layout: data/*_attributes.csv → data/attributes/"
    mkdir -p "$GB_DIR/data/attributes"
    mv "$GB_DIR/data"/CAMELS_GB_*_attributes.csv "$GB_DIR/data/attributes/" 2>/dev/null || true
fi
ls "$GB_DIR/data" | head -10

# Step 2: select basins
if [ ! -f "$BASIN_LIST" ]; then
    echo "[$(date +'%F %T')] STEP 2/4 selecting 50 basins ..."
    python "$SELECT_SCRIPT" --n 50 --seed 0 2>&1 | tee "$LOG_DIR/select_basins.log"
    if [ ! -f "$BASIN_LIST" ]; then
        echo "[FAIL] basin list not created — see $LOG_DIR/select_basins.log"
        exit 2
    fi
    echo "[$(date +'%F %T')] STEP 2/4 selected $(wc -l < "$BASIN_LIST") basins."
else
    echo "[$(date +'%F %T')] STEP 2/4 basin list exists, skip."
fi

# Step 3: train
TRAIN_LOG="$LOG_DIR/train_$(date +%H%M%S).log"
echo "[$(date +'%F %T')] STEP 3/4 train — log: $TRAIN_LOG"
nh-run train --config-file "$CONFIG" 2>&1 | tee "$TRAIN_LOG"

# Locate the run dir (NH creates results/<experiment_name>_<DDMM_HHMMSS>/)
RUN_DIR=$(ls -dt "$PROJ_DIR/experiments/exp004/results"/exp004_camels_gb_native_* 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "[FAIL] could not locate run dir under experiments/exp004/results/"
    exit 3
fi
echo "[$(date +'%F %T')] run dir: $RUN_DIR"

# Step 4: evaluate validation + test (mirror exp002 D-014 strategy)
EVAL_VAL_LOG="$LOG_DIR/eval_val_$(date +%H%M%S).log"
EVAL_TEST_LOG="$LOG_DIR/eval_test_$(date +%H%M%S).log"
echo "[$(date +'%F %T')] STEP 4/4 evaluate validation period — log: $EVAL_VAL_LOG"
nh-run evaluate --run-dir "$RUN_DIR" --period validation 2>&1 | tee "$EVAL_VAL_LOG"
echo "[$(date +'%F %T')] STEP 4/4 evaluate test period — log: $EVAL_TEST_LOG"
nh-run evaluate --run-dir "$RUN_DIR" --period test 2>&1 | tee "$EVAL_TEST_LOG"

touch "$RUN_DIR/done.flag"
echo "[$(date +'%F %T')] exp004 PIPELINE DONE — run: $RUN_DIR"
echo "下一步: 运行 HSCC analysis（需先 refactor hscc_analysis_v2.py 接受 GB attributes 路径）"
