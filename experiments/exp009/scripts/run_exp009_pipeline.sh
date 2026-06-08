#!/usr/bin/env bash
# exp009 multi-seed reproducibility pipeline (4 NEW seeds: 137 / 2024 / 1337 / 7)
#
# Per plan §4 + §9 fallback rules + memory(tmux):
#   For each seed: train 30 epoch → eval validation → eval test → HSCC analysis → mechanism analysis
#   seed=42 reused via symlink (seed042_run -> exp002 archived run); not retrained.
#
# Run inside tmux detached session (memory: long jobs MUST use tmux):
#   tmux new -d -s exp009 "bash $0"
#   tmux attach -t exp009    # to view live progress
#
# Estimated wall clock: ~64h GPU (16h × 4 seeds, sequential — single RTX 4070 Ti).
#
# All output is tee'd to logs/seed_<S>_<step>.log so progress can be inspected
# from another terminal or via the Read tool.

set -e
set -o pipefail

PROJ_DIR="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
EXP_DIR="$PROJ_DIR/experiments/exp009"
LOG_DIR="$EXP_DIR/logs"
RESULTS_DIR="$EXP_DIR/results"
HSCC_SCRIPT="$PROJ_DIR/experiments/exp002/hscc_analysis_v2.py"
MECH_SCRIPT="$PROJ_DIR/experiments/exp007/run_mechanism_analysis.py"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR"

# Conda env (same as exp002)
# NOTE: r-reticulate env may already be on PATH (from user shell rc); generic `python`
#       resolves to r-reticulate's python (no pandas) even after `conda activate hscc-hydrology`.
#       We pin absolute paths via $CONDA_ENV_BIN to avoid this footgun.
source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology
CONDA_ENV_BIN="/home/qingsong/miniconda3/envs/hscc-hydrology/bin"
PY="$CONDA_ENV_BIN/python"
NH_RUN="$CONDA_ENV_BIN/nh-run"

log() { echo "[$(date +'%F %T')] $*" | tee -a "$PIPELINE_LOG"; }

log "=========================================="
log "exp009 v5.0 multi-seed pipeline START"
log "GPU: $(nvidia-smi -L | head -1)"
log "Conda: $CONDA_PREFIX"
log "Seeds: 137 2024 1337 7 (seed=42 reused via symlink)"
log "=========================================="

SEEDS=(137 2024 1337 7)

for SEED in "${SEEDS[@]}"; do
    PAD=$(printf "%03d" "$SEED")
    CONFIG="$EXP_DIR/configs/exp009_seed${PAD}.yml"
    EXP_NAME="exp009_seed${PAD}"
    LOG_TRAIN="$LOG_DIR/seed${PAD}_train.log"
    LOG_VAL="$LOG_DIR/seed${PAD}_eval_validation.log"
    LOG_TEST="$LOG_DIR/seed${PAD}_eval_test.log"
    LOG_HSCC="$LOG_DIR/seed${PAD}_hscc.log"
    LOG_MECH="$LOG_DIR/seed${PAD}_mechanism.log"

    log "------ SEED=$SEED START ------"

    # Idempotent run dir discovery: prefer canonical symlink (recovery), else latest match
    CANONICAL="$RESULTS_DIR/seed${PAD}_run"
    RUN_DIR=""
    if [ -L "$CANONICAL" ] || [ -d "$CANONICAL" ]; then
        RUN_DIR=$(readlink -f "$CANONICAL")
        log "  resume: existing run dir $RUN_DIR"
    fi

    # Step 1: Train (skip if model_epoch030.pt already exists)
    if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/model_epoch030.pt" ]; then
        log "  [1/5] train: SKIP (model_epoch030.pt exists)"
    else
        log "  [1/5] train: $CONFIG"
        "$NH_RUN" train --config-file "$CONFIG" 2>&1 | tee "$LOG_TRAIN"
        RUN_DIR=$(ls -1dt "$RESULTS_DIR/${EXP_NAME}_"* 2>/dev/null | head -1)
        if [ -z "$RUN_DIR" ]; then
            log "  ERROR: cannot find run dir for $EXP_NAME under $RESULTS_DIR — abort"
            exit 1
        fi
        log "  run dir: $RUN_DIR"
        if [ ! -e "$CANONICAL" ]; then
            ln -snf "$(basename "$RUN_DIR")" "$CANONICAL"
            log "  symlink: seed${PAD}_run -> $(basename "$RUN_DIR")"
        fi
    fi

    # Step 2: Evaluate validation (skip if validation_results.p at epoch 30 exists)
    if [ -f "$RUN_DIR/validation/model_epoch030/validation_results.p" ]; then
        log "  [2/5] eval validation: SKIP (already exists)"
    else
        log "  [2/5] eval validation"
        "$NH_RUN" evaluate --run-dir "$RUN_DIR" --period validation --epoch 30 2>&1 | tee "$LOG_VAL"
    fi

    # Step 3: Evaluate test (skip if test_results.p exists)
    if [ -f "$RUN_DIR/test/model_epoch030/test_results.p" ]; then
        log "  [3/5] eval test: SKIP (already exists)"
    else
        log "  [3/5] eval test"
        "$NH_RUN" evaluate --run-dir "$RUN_DIR" --period test --epoch 30 2>&1 | tee "$LOG_TEST"
    fi

    # Step 4: HSCC + Global CP analysis (skip if metrics.json exists)
    if [ -f "$RUN_DIR/_analysis/metrics.json" ]; then
        log "  [4/5] HSCC analysis: SKIP (metrics.json exists)"
    else
        log "  [4/5] HSCC analysis"
        "$PY" "$HSCC_SCRIPT" \
            --run_dir "$RUN_DIR" \
            --out_dir "$RUN_DIR/_analysis" 2>&1 | tee "$LOG_HSCC"
    fi

    # Step 5: Mechanism analysis (exp007 reuse) — skip if _mechanism dir non-empty
    if [ -d "$RUN_DIR/_mechanism" ] && [ "$(ls -A "$RUN_DIR/_mechanism" 2>/dev/null)" ]; then
        log "  [5/5] mechanism analysis: SKIP (_mechanism populated)"
    else
        log "  [5/5] mechanism analysis"
        "$PY" "$MECH_SCRIPT" \
            --exp002_run_dir "$RUN_DIR" \
            --out_dir "$RUN_DIR/_mechanism" 2>&1 | tee "$LOG_MECH"
    fi

    log "------ SEED=$SEED DONE ------"
done

log "=========================================="
log "All 4 seeds finished — running cross-seed aggregation..."
log "(aggregate script: experiments/exp009/scripts/aggregate_5seed.py — TBD by user / next session)"
log "=========================================="

# sha256 manifest of all 5 seeds (4 trained + 1 symlinked)
log "Computing sha256 manifest..."
MANIFEST="$RESULTS_DIR/sha256_manifest.txt"
{
    echo "# exp009 5-seed sha256 manifest — generated $(date -Iseconds)"
    for SEED in 042 137 2024 1337 007; do
        RUN="$RESULTS_DIR/seed${SEED}_run"
        if [ -e "$RUN" ]; then
            echo ""
            echo "## seed=$((10#$SEED))  run: $(readlink -f "$RUN" || echo "$RUN")"
            for f in config.yml model_epoch030.pt test/model_epoch030/test_results.p test/model_epoch030/test_metrics.csv; do
                if [ -f "$RUN/$f" ]; then
                    sha256sum "$RUN/$f" 2>/dev/null | sed "s|$RUN/||"
                fi
            done
        fi
    done
} > "$MANIFEST"
log "Manifest: $MANIFEST"

log "Pipeline complete — total wall clock: $(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m"
log "Next steps: cross-seed aggregation + plan §5 success criteria evaluation"
