#!/usr/bin/env bash
# exp010 — HopCPT head-to-head, CAMELS-US 671 basins, single seed=42 (P0 v5.0)
#
# Usage:
#   bash experiments/exp010/run_hopcpt.sh sanity      # 5-basin × 50-epoch smoke (~5-15 min)
#   bash experiments/exp010/run_hopcpt.sh production  # 671-basin × 3000-epoch full run (~12-20h GPU)
#
# Both modes are designed to be wrapped in a tmux detached session per
# `feedback_long_running_use_tmux`:
#   tmux new -d -s exp010_hopcpt 'bash experiments/exp010/run_hopcpt.sh production'
#
# Logs land in experiments/exp010/logs/. Hydra job dir lands in
# experiments/exp010/results/run_<ts>/ (production) or experiments/exp010/_sanity/run_<ts>/.

set -euo pipefail

MODE="${1:-sanity}"
REPO_ROOT="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
PYTHON="/home/qingsong/miniconda3/envs/hscc-hydrology/bin/python"
HOPCPT_CODE="${REPO_ROOT}/libs/HopCPT/code"
LOG_DIR="${REPO_ROOT}/experiments/exp010/logs"
TS=$(date +'%Y%m%d_%H%M%S')

mkdir -p "${LOG_DIR}"

case "${MODE}" in
  sanity)
    CONFIG_NAME="exp010_hopcpt_sanity"
    LOG_FILE="${LOG_DIR}/sanity_${TS}.log"
    ;;
  production)
    CONFIG_NAME="exp010_hopcpt"
    LOG_FILE="${LOG_DIR}/production_${TS}.log"
    ;;
  *)
    echo "ERROR: unknown mode '${MODE}'. Use 'sanity' or 'production'." >&2
    exit 2
    ;;
esac

echo "[exp010] mode=${MODE} config=${CONFIG_NAME}"
echo "[exp010] log: ${LOG_FILE}"
echo "[exp010] start: $(date)"

cd "${HOPCPT_CODE}"
"${PYTHON}" main.py --config-name "${CONFIG_NAME}" 2>&1 | tee "${LOG_FILE}"
RC=${PIPESTATUS[0]}

echo "[exp010] done: $(date) (exit ${RC})"
exit "${RC}"
