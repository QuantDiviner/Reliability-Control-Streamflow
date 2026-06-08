#!/usr/bin/env bash
# exp003 Action A5 — 3-seed ensemble for the 4 most failed HUCs (10/12/14/18).
#
# Goal: Per R2 D-R2-exp003 §A5, generate 95% CI for HSCC coverage on the 4
# folds where seed=42 most badly violated PUB-relaxed criteria
# (HUC-10 humid 0.509, HUC-12 humid 0.551, HUC-14 dry n=1 cov=0.126,
# HUC-18 humid 0.753 + snow violation).
#
# Reuses exp003/run_loro_pipeline.sh logic but only for these 4 HUCs and
# 2 new seeds (137, 2024). seed=42 already exists from the original run.
# Total ETA: 4 HUCs × 2 seeds × ~70 min = ~9.5 h GPU.
#
# tmux new-session -d -s exp003_a5 "bash experiments/exp003/run_a5_3seed_failed_hucs.sh"

set -eo pipefail

PROJ="/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP"
LOG_DIR="$PROJ/experiments/exp003/logs"
RESULTS_ROOT="$PROJ/experiments/exp003/results"
CONFIG_DIR="$PROJ/experiments/exp003/configs"
mkdir -p "$LOG_DIR"
cd "$PROJ"

source /home/qingsong/miniconda3/etc/profile.d/conda.sh
conda activate hscc-hydrology

HUCS=(10 12 14 18)
NEW_SEEDS=(137 2024)

echo "=========================================="
echo "exp003 A5 3-seed pipeline START — $(date +'%F %T')"
echo "HUCs: ${HUCS[*]} ; new seeds: ${NEW_SEEDS[*]}"
echo "=========================================="

for HUC in "${HUCS[@]}"; do
    for SEED in "${NEW_SEEDS[@]}"; do
        TAG="huc${HUC}_seed${SEED}"
        CONFIG="$CONFIG_DIR/loro_${TAG}.yml"
        echo ""
        echo "[$(date +'%F %T')] === HUC-${HUC} seed=${SEED} ==="

        # Skip if already done (resume support; tolerate empty glob under set -e)
        EXISTING=$(compgen -G "$RESULTS_ROOT/exp003_loro_${TAG}_*" 2>/dev/null | tail -1 || true)
        if [ -n "$EXISTING" ] && [ -f "$EXISTING/_analysis/metrics.json" ]; then
            echo "  SKIP: already done at $EXISTING"
            continue
        fi

        # Step 1: train
        echo "[$(date +'%F %T')]   train (30 epoch) ..."
        nh-run train --config-file "$CONFIG" 2>&1 | tee "$LOG_DIR/a5_train_${TAG}.log"

        RUN_DIR=$(compgen -G "$RESULTS_ROOT/exp003_loro_${TAG}_*" | tail -1 || true)
        [ -z "$RUN_DIR" ] && { echo "[FAIL] no run_dir for ${TAG}"; exit 2; }
        echo "  run_dir: $RUN_DIR"

        # Step 2: evaluate validation (cal scores)
        echo "[$(date +'%F %T')]   evaluate validation ..."
        nh-run evaluate --run-dir "$RUN_DIR" --period validation --epoch 30 2>&1 \
            | tee "$LOG_DIR/a5_eval_val_${TAG}.log"

        # Step 3: evaluate test
        echo "[$(date +'%F %T')]   evaluate test ..."
        nh-run evaluate --run-dir "$RUN_DIR" --period test --epoch 30 2>&1 \
            | tee "$LOG_DIR/a5_eval_test_${TAG}.log"

        # Step 4: HSCC analysis (uses exp003/hscc_analysis_loro.py)
        echo "[$(date +'%F %T')]   HSCC analysis ..."
        OUT_DIR="$RUN_DIR/_analysis"
        mkdir -p "$OUT_DIR"
        python "$PROJ/experiments/exp003/hscc_analysis_loro.py" \
            --run_dir "$RUN_DIR" --huc "$HUC" --out_dir "$OUT_DIR" 2>&1 \
            | tee "$LOG_DIR/a5_hscc_${TAG}.log"

        echo "[$(date +'%F %T')]   ${TAG} done. metrics → $OUT_DIR/metrics.json"
    done
done

echo ""
echo "=========================================="
echo "[$(date +'%F %T')] all 8 (4 HUC × 2 new seeds) trains complete"
echo "running A5 cross-seed aggregator"
echo "=========================================="

python "$PROJ/experiments/exp003/aggregate_a5_3seed.py" 2>&1 \
    | tee "$LOG_DIR/a5_aggregate.log"

echo ""
echo "=========================================="
echo "exp003 A5 COMPLETE — $(date +'%F %T')"
echo "outputs: experiments/exp003/results/_a5_3seed/{summary.json, per_huc_table.csv}"
echo "=========================================="
