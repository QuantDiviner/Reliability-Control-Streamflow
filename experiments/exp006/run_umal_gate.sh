#!/usr/bin/env bash
# exp006 v3 PCR-005 — UMAL GATING smoke runner.
#
# 50 basins × 5 epochs UMAL training as hard pre-flight before full canary.
# Per R1' Opus subagent: pass criterion = 5 epochs no NaN + train loss decreasing.
# Fail action = skip Path B canary, trigger Phase D / Path E fallback.
#
# Time estimate: ~30 min on RTX 4070.

set -eo pipefail

PROJ_DIR="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
EXP_DIR="$PROJ_DIR/experiments/exp006"
LOG_DIR="$EXP_DIR/logs"
CONFIG="$EXP_DIR/configs/exp006_umal_gate.yml"

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR"

source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology
export PATH="/home/qingsong/miniconda3/envs/hscc-hydrology/bin:$PATH"
PY="/home/qingsong/miniconda3/envs/hscc-hydrology/bin/python"

GATE_LOG="$LOG_DIR/umal_gate.log"

echo "=========================================="
echo "exp006 v3 UMAL GATE START — $(date +'%F %T')"
echo "  50 basins × 5 epochs"
echo "  Pass: no NaN, loss decreasing"
echo "  Fail → Phase D / Path E"
echo "=========================================="

nh-run train --config-file "$CONFIG" 2>&1 | tee "$GATE_LOG"

echo ""
echo "=========================================="
echo "exp006 v3 UMAL GATE TRAIN DONE — $(date +'%F %T')"
echo "Inspect: $GATE_LOG"
echo "Pass criterion check:"
echo "  - grep 'Loss is Nan|Loss was NaN' should return EMPTY"
echo "  - 5 'Epoch X average loss' lines, monotonically decreasing"
echo "=========================================="
