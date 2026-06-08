#!/usr/bin/env bash
# exp010 Action 6 — HopCPT +2 seeds (137, 2024) sequential GPU runs.
# Tmux usage:
#   tmux new -d -s exp010_seeds 'bash experiments/exp010/run_hopcpt_multiseed.sh'
#
# Each seed: ~98 min wall clock (per exp010 production baseline). Total ~3.3h.

set -euo pipefail

REPO_ROOT="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
PYTHON="/home/qingsong/miniconda3/envs/hscc-hydrology/bin/python"
HOPCPT_CODE="${REPO_ROOT}/libs/HopCPT/code"
LOG_DIR="${REPO_ROOT}/experiments/exp010/logs"
mkdir -p "${LOG_DIR}"

run_seed() {
  local SEED=$1
  local CONFIG_NAME="exp010_hopcpt_seed${SEED}"
  local TS=$(date +'%Y%m%d_%H%M%S')
  local LOG_FILE="${LOG_DIR}/production_seed${SEED}_${TS}.log"

  echo "[exp010 multiseed] seed=${SEED}  start=$(date)  log=${LOG_FILE}"
  cd "${HOPCPT_CODE}"
  "${PYTHON}" main.py --config-name "${CONFIG_NAME}" 2>&1 | tee "${LOG_FILE}"
  local RC=${PIPESTATUS[0]}
  echo "[exp010 multiseed] seed=${SEED}  done=$(date)  exit=${RC}"
  if [ "${RC}" -ne 0 ]; then
    echo "[exp010 multiseed] seed=${SEED} FAILED, aborting"
    exit "${RC}"
  fi
}

echo "[exp010 multiseed] === A6 +2 seeds: 137 and 2024 ==="
echo "[exp010 multiseed] start=$(date)"

run_seed 137
run_seed 2024

echo "[exp010 multiseed] === ALL DONE ==="
echo "[exp010 multiseed] finish=$(date)"

# Post-run: aggregate 3-seed + closure evaluation
echo "[exp010 multiseed] === post-run aggregation ==="
cd "${REPO_ROOT}"
"${PYTHON}" experiments/exp010/scripts/aggregate_3seed.py 2>&1 | tee "${LOG_DIR}/aggregate_3seed_$(date +'%Y%m%d_%H%M%S').log"
"${PYTHON}" experiments/exp010/scripts/closure_evaluation_a10.py 2>&1 | tee "${LOG_DIR}/closure_a10_$(date +'%Y%m%d_%H%M%S').log"
echo "[exp010 multiseed] === post-run aggregation done ==="
