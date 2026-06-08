#!/usr/bin/env bash
# exp003 LORO 18-fold orchestration:
#   For each HUC k in 01..18 (or subset via --folds), execute:
#     1. nh-run train       (30 epochs on ~580-660 train_huc{k}.txt basins)
#     2. nh-run evaluate --period validation  (cal scores from train pool)
#     3. nh-run evaluate --period test         (held-out HUC test obs/pred)
#     4. python hscc_analysis_loro.py          (per-fold metrics.json)
#   Writes done.flag in fold result dir on success → resume skips completed folds.
#
# Designed for tmux detached sessions (per user preference for long jobs).
# All output tee'd to logs/loro_huc{k}_*.log.
#
# Usage:
#   bash experiments/exp003/run_loro_pipeline.sh [--folds k1,k2,...] [--epoch N] [--sanity]
#     --folds  comma-separated HUC codes (zero-padded), default = 01..18
#     --epoch  override evaluate epoch (default 30; A.6 sanity uses 5)
#     --sanity short-circuit train epochs to 5 (modifies config in place via sed)
#
# Examples:
#   bash experiments/exp003/run_loro_pipeline.sh                       # full 18-fold, 30 epoch
#   bash experiments/exp003/run_loro_pipeline.sh --folds 09 --sanity   # A.6 sanity on HUC-09

set -e
set -o pipefail
# NOTE: deliberately NOT using `set -u` because conda's activate.d scripts
# (libblas_mkl_activate.sh) reference unset MKL_INTERFACE_LAYER and would abort.

PROJ_DIR="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
LOG_DIR="$PROJ_DIR/experiments/exp003/logs"
CONFIG_DIR="$PROJ_DIR/experiments/exp003/configs"
RUN_PARENT="$PROJ_DIR/experiments/exp003/results"
ANALYSIS_SCRIPT="$PROJ_DIR/experiments/exp003/hscc_analysis_loro.py"

mkdir -p "$LOG_DIR" "$RUN_PARENT"
cd "$PROJ_DIR"

# Parse args
FOLDS_RAW=""
EPOCH=30
SANITY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --folds) FOLDS_RAW="$2"; shift 2 ;;
        --epoch) EPOCH="$2"; shift 2 ;;
        --sanity) SANITY=1; shift ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "$FOLDS_RAW" ]; then
    FOLDS=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18)
else
    IFS=',' read -ra FOLDS <<< "$FOLDS_RAW"
fi

# Activate conda environment
source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology

PIPELINE_LOG="$LOG_DIR/pipeline_$(date +'%y%m%d_%H%M%S').log"
echo "==========================================" | tee -a "$PIPELINE_LOG"
echo "exp003 LORO PIPELINE START — $(date +'%F %T')" | tee -a "$PIPELINE_LOG"
echo "folds: ${FOLDS[*]}" | tee -a "$PIPELINE_LOG"
echo "epoch: $EPOCH  sanity: $SANITY" | tee -a "$PIPELINE_LOG"
echo "==========================================" | tee -a "$PIPELINE_LOG"

n_done=0
n_skipped=0
n_failed=0

for HUC in "${FOLDS[@]}"; do
    CANONICAL_CONFIG="$CONFIG_DIR/loro_huc${HUC}.yml"
    if [ ! -f "$CANONICAL_CONFIG" ]; then
        echo "[$(date +'%F %T')] HUC-$HUC: missing config $CANONICAL_CONFIG, skip" | tee -a "$PIPELINE_LOG"
        n_failed=$((n_failed + 1))
        continue
    fi

    # Sanity mode: never touch the canonical config; copy to a tmp file and lower epochs there.
    if [ "$SANITY" -eq 1 ]; then
        TMP_CONFIG="$(mktemp -t exp003_sanity_huc${HUC}.XXXXXX.yml)"
        cp "$CANONICAL_CONFIG" "$TMP_CONFIG"
        sed -i 's/^epochs: .*/epochs: 5/' "$TMP_CONFIG"
        CONFIG="$TMP_CONFIG"
        EPOCH=5
        # Auto-cleanup tmp config no matter how this iteration ends
        trap "rm -f '$TMP_CONFIG'" EXIT
        echo "[$(date +'%F %T')] HUC-$HUC: SANITY mode → epochs=5 (tmp: $TMP_CONFIG)" \
            | tee -a "$PIPELINE_LOG"
    else
        CONFIG="$CANONICAL_CONFIG"
    fi

    FOLD_LOG="$LOG_DIR/loro_huc${HUC}_$(date +'%y%m%d_%H%M%S').log"
    echo "" | tee -a "$PIPELINE_LOG"
    echo "------------------------------------------" | tee -a "$PIPELINE_LOG"
    echo "[$(date +'%F %T')] HUC-$HUC start — log: $FOLD_LOG" | tee -a "$PIPELINE_LOG"

    # Resume check: skip if any prior run dir for this HUC already has done.flag
    EXISTING_DONE=""
    for d in $(compgen -G "$RUN_PARENT/exp003_loro_huc${HUC}_*" 2>/dev/null || true); do
        if [ -f "$d/done.flag" ]; then
            EXISTING_DONE="$d"
            break
        fi
    done
    if [ -n "$EXISTING_DONE" ]; then
        echo "[$(date +'%F %T')] HUC-$HUC: done.flag found in $EXISTING_DONE, skip" \
            | tee -a "$PIPELINE_LOG"
        n_skipped=$((n_skipped + 1))
        # tmp sanity config (if any) cleaned up by trap on iteration exit
        [ "$SANITY" -eq 1 ] && [ -f "${TMP_CONFIG:-}" ] && rm -f "$TMP_CONFIG"
        continue
    fi

    {
        echo "[$(date +'%F %T')] HUC-$HUC STEP 1/4 train"
        nh-run train --config-file "$CONFIG"

        # Discover the new fold run dir (latest matching name)
        RUN_DIR=$(ls -1dt "$RUN_PARENT"/exp003_loro_huc${HUC}_* 2>/dev/null | head -1)
        if [ -z "$RUN_DIR" ]; then
            echo "ERROR: cannot find run dir for HUC-$HUC under $RUN_PARENT"
            exit 11
        fi
        echo "[$(date +'%F %T')] HUC-$HUC run dir: $RUN_DIR"

        echo "[$(date +'%F %T')] HUC-$HUC STEP 2/4 evaluate validation"
        nh-run evaluate --run-dir "$RUN_DIR" --period validation --epoch "$EPOCH"

        echo "[$(date +'%F %T')] HUC-$HUC STEP 3/4 evaluate test"
        nh-run evaluate --run-dir "$RUN_DIR" --period test --epoch "$EPOCH"

        echo "[$(date +'%F %T')] HUC-$HUC STEP 4/4 HSCC LORO analysis"
        python "$ANALYSIS_SCRIPT" \
            --run_dir "$RUN_DIR" \
            --huc "$HUC" \
            --out_dir "$RUN_DIR/_analysis"

        # Mark done
        date -u +'%FT%TZ' > "$RUN_DIR/done.flag"
        echo "[$(date +'%F %T')] HUC-$HUC ✅ done.flag written"
    } 2>&1 | tee "$FOLD_LOG"
    fold_exit=${PIPESTATUS[0]}

    # Clean up sanity tmp config (canonical config never touched in sanity mode)
    [ "$SANITY" -eq 1 ] && [ -f "${TMP_CONFIG:-}" ] && rm -f "$TMP_CONFIG"

    if [ "$fold_exit" -ne 0 ]; then
        echo "[$(date +'%F %T')] HUC-$HUC ❌ FAIL (exit=$fold_exit)" | tee -a "$PIPELINE_LOG"
        n_failed=$((n_failed + 1))
    else
        n_done=$((n_done + 1))
    fi
done

echo "" | tee -a "$PIPELINE_LOG"
echo "==========================================" | tee -a "$PIPELINE_LOG"
echo "exp003 LORO PIPELINE DONE — $(date +'%F %T')" | tee -a "$PIPELINE_LOG"
echo "  ran: $n_done   skipped(done): $n_skipped   failed: $n_failed" | tee -a "$PIPELINE_LOG"
echo "==========================================" | tee -a "$PIPELINE_LOG"

if [ "$n_failed" -gt 0 ]; then
    exit 12
fi
