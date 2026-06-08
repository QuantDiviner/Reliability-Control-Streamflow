#!/usr/bin/env python
"""S1-O-R1-03: Cross-period predictive value of the calibration-period
residual-variance screen (sigma^2_b = log_resid_var) versus naive static
baselines (aridity, q_mean, frac_snow).

Goal: prove that the calibration-period signed-log-residual variance carries
NON-TRIVIAL information about which basins suffer SEVERE test-period
undercoverage under global split CP, beyond what cheap static catchment
attributes already encode.

Label:  SEVERE UNDERCOVERAGE = (coverage_global < 0.85)
        (0.85 is the project's UNDERCOVERAGE_FLOOR, run_post_review_enhancements.py:34)

Predictors compared as severe-undercoverage screens:
  (1) log_resid_var  (the diagnostic, sigma^2_b)  -- higher => more undercoverage
  (2) aridity        (naive static attribute)     -- higher => more undercoverage
  (3) q_mean         (naive static attribute)     -- LOWER  => more undercoverage (sign flipped)
  (4) frac_snow      (naive static attribute)     -- higher => more undercoverage

For each predictor:
  - ROC AUC (roc_auc_score, orientation chosen so the predictor scores POSITIVE
    class higher; q_mean is negated so "low flow" => high screen score).
  - Out-of-fold balanced accuracy / precision / recall via StratifiedKFold
    (n_splits=5, shuffle=True, random_state=0): on each TRAIN fold choose the
    threshold that maximizes balanced accuracy (Youden's J on the oriented
    predictor), apply to the held-out fold, pool OOF predictions, then compute
    metrics with the same classifier_metrics conventions as
    run_post_review_enhancements.py (balanced_accuracy = 0.5*(recall+specificity)).

CAVEAT (label leakage): the reviewer's suggested baseline "test-period marginal
coverage itself" is derived from the same coverage_global used to build the
label, so its AUC is trivially ~1.0. It is an INVALID baseline and is EXCLUDED.

CPU-only, self-contained, re-runnable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ---- Frozen project constants (mirror run_post_review_enhancements.py) -------
UNDERCOVERAGE_FLOOR = 0.85  # SEVERE undercoverage label floor
SEED_TARGET = 42            # per-basin CSV may hold a single seed; use 42
CV_FOLDS = 5
CV_RANDOM_STATE = 0

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
PER_BASIN_CSV = RESULTS / "f2_us_per_basin_residual_variance.csv"
HYDRO_TXT = (
    EXP.parent.parent
    / "data" / "raw" / "CAMELS_US" / "camels_attributes_v2.0" / "camels_hydro.txt"
)
OUT_JSON = RESULTS / "diagnostic_naive_baseline.json"

# Predictor definitions: name -> (column, orientation_sign)
# orientation_sign = +1 if higher predictor => more undercoverage (positive class),
#                    -1 if lower predictor => more undercoverage (we negate so that
#                       the oriented score is higher for the positive class).
PREDICTORS = {
    "log_resid_var": ("log_resid_var", +1.0),  # sigma^2_b, the diagnostic
    "aridity": ("aridity", +1.0),
    "q_mean": ("q_mean", -1.0),                 # low mean flow => more undercoverage
    "frac_snow": ("frac_snow", +1.0),
}


def classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Same conventions as run_post_review_enhancements.py:classifier_metrics."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    bal_acc = 0.5 * (recall + specificity)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(bal_acc),
    }


def choose_threshold_balacc(score: np.ndarray, y: np.ndarray) -> float:
    """Pick the threshold on an ORIENTED score (higher => positive class) that
    maximizes balanced accuracy (== Youden's J argmax). Predict positive when
    score >= threshold. Mirrors choose_threshold() but with balanced accuracy as
    the objective (per task spec) and works on any oriented predictor."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=bool)
    vals = np.sort(np.unique(score[~np.isnan(score)]))
    if len(vals) > 200:
        vals = np.quantile(vals, np.linspace(0.02, 0.98, 193))
    # candidate thresholds: each unique value plus +inf (predict all-negative)
    candidates = list(vals) + [np.inf]
    best_thr = None
    best_key = None
    for thr in candidates:
        pred = score >= thr
        m = classifier_metrics(y, pred)
        # primary: balanced accuracy; tie-breakers: recall, then lower threshold
        key = (m["balanced_accuracy"], m["recall"], -abs(float(thr)))
        if best_key is None or key > best_key:
            best_key = key
            best_thr = float(thr)
    assert best_thr is not None
    return best_thr


def oriented_score(df: pd.DataFrame, col: str, sign: float) -> np.ndarray:
    return sign * df[col].to_numpy(dtype=float)


def load_data() -> pd.DataFrame:
    if not PER_BASIN_CSV.exists():
        raise FileNotFoundError(f"missing per-basin CSV: {PER_BASIN_CSV}")
    df = pd.read_csv(PER_BASIN_CSV, dtype={"basin_id": str})
    seeds_present = sorted(df["seed"].unique().tolist())
    if SEED_TARGET in seeds_present:
        df = df[df["seed"] == SEED_TARGET].copy()
    else:  # fall back to the single present seed and record it
        only = seeds_present[0]
        df = df[df["seed"] == only].copy()
    df["basin_id"] = df["basin_id"].str.strip().str.zfill(8)

    # Merge q_mean from camels_hydro.txt (semicolon-delimited gauge_id).
    if not HYDRO_TXT.exists():
        raise FileNotFoundError(f"missing CAMELS hydro signatures: {HYDRO_TXT}")
    hydro = pd.read_csv(HYDRO_TXT, sep=";", dtype={"gauge_id": str})
    hydro["gauge_id"] = hydro["gauge_id"].str.strip().str.zfill(8)
    hydro = hydro[["gauge_id", "q_mean"]].rename(columns={"gauge_id": "basin_id"})

    merged = df.merge(hydro, on="basin_id", how="left", validate="one_to_one")
    n_missing_q = int(merged["q_mean"].isna().sum())
    if n_missing_q:
        # keep only basins with all predictors available for a fair comparison
        merged = merged.dropna(subset=["q_mean"]).copy()
    merged.attrs["n_missing_qmean"] = n_missing_q
    merged.attrs["seeds_present"] = seeds_present
    merged.attrs["seed_used"] = SEED_TARGET if SEED_TARGET in seeds_present else seeds_present[0]
    return merged


def evaluate_predictor(df: pd.DataFrame, col: str, sign: float,
                       y: np.ndarray) -> dict[str, float]:
    score = oriented_score(df, col, sign)
    valid = ~np.isnan(score)
    score_v = score[valid]
    y_v = y[valid]

    # ROC AUC on the oriented score (higher => positive class).
    auc = float(roc_auc_score(y_v, score_v))

    # Out-of-fold thresholded classification.
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    oof_pred = np.zeros(len(y_v), dtype=bool)
    idx = np.arange(len(y_v))
    for train_idx, test_idx in skf.split(idx, y_v):
        thr = choose_threshold_balacc(score_v[train_idx], y_v[train_idx])
        oof_pred[test_idx] = score_v[test_idx] >= thr
    m = classifier_metrics(y_v, oof_pred)
    return {
        "auc": round(auc, 4),
        "oos_balanced_accuracy": round(m["balanced_accuracy"], 4),
        "oos_precision": round(m["precision"], 4),
        "oos_recall": round(m["recall"], 4),
    }


def main() -> None:
    df = load_data()
    y = (df["coverage_global"].to_numpy(dtype=float) < UNDERCOVERAGE_FLOOR)
    n_basins = int(len(df))
    positive_rate = float(np.mean(y))

    predictors_out: dict[str, dict[str, float]] = {}
    for name, (col, sign) in PREDICTORS.items():
        predictors_out[name] = evaluate_predictor(df, col, sign, y)

    sig = predictors_out["log_resid_var"]
    delta_auc_vs_aridity = round(sig["auc"] - predictors_out["aridity"]["auc"], 4)
    delta_auc_vs_q_mean = round(sig["auc"] - predictors_out["q_mean"]["auc"], 4)
    delta_auc_vs_frac_snow = round(sig["auc"] - predictors_out["frac_snow"]["auc"], 4)
    delta_balacc_vs_aridity = round(
        sig["oos_balanced_accuracy"] - predictors_out["aridity"]["oos_balanced_accuracy"], 4
    )
    delta_balacc_vs_q_mean = round(
        sig["oos_balanced_accuracy"] - predictors_out["q_mean"]["oos_balanced_accuracy"], 4
    )
    delta_balacc_vs_frac_snow = round(
        sig["oos_balanced_accuracy"] - predictors_out["frac_snow"]["oos_balanced_accuracy"], 4
    )

    notes = (
        "S1-O-R1-03: calibration-period signed-log-residual variance (sigma^2_b = "
        "log_resid_var) is tested as a SEVERE-undercoverage screen (coverage_global "
        f"< {UNDERCOVERAGE_FLOOR}) against naive static catchment attributes (aridity, "
        "q_mean, frac_snow). Predictors are oriented so higher => positive class "
        "(q_mean is negated: low mean flow => more undercoverage). AUC via "
        "roc_auc_score; OOS metrics via StratifiedKFold(n_splits=5, shuffle=True, "
        "random_state=0) choosing the per-train-fold balanced-accuracy-maximizing "
        "threshold (Youden's J) and pooling held-out predictions; "
        "balanced_accuracy=0.5*(recall+specificity) per "
        "run_post_review_enhancements.py:classifier_metrics(). LABEL-LEAKAGE TRAP: "
        "the reviewer's suggested baseline 'test-period marginal coverage itself' is "
        "derived from the SAME coverage_global used to define the label, so its AUC is "
        "trivially ~1.0; it is an INVALID baseline and is EXCLUDED here. q_mean merged "
        "from camels_hydro.txt on 8-char zero-padded gauge id; aridity & frac_snow are "
        "from the per-basin CSV. Per-basin CSV contained only seed="
        f"{df.attrs['seeds_present']} (the 5-seed per-basin CSVs were pruned), so a "
        "single-seed StratifiedKFold over basins is used rather than the F2 "
        "leave-one-seed protocol."
    )
    if df.attrs.get("n_missing_qmean", 0):
        notes += (
            f" {df.attrs['n_missing_qmean']} basin(s) had a NaN q_mean signature in "
            "camels_hydro.txt (e.g. gauge 03281100) and were dropped so every "
            "predictor is scored on the identical basin set."
        )

    out = {
        "analysis": "S1-O-R1-03 diagnostic vs naive-baseline severe-undercoverage screen",
        "label_definition": f"severe_undercoverage = coverage_global < {UNDERCOVERAGE_FLOOR}",
        "label_floor": UNDERCOVERAGE_FLOOR,
        "seed_used": int(df.attrs["seed_used"]),
        "seeds_present_in_csv": [int(s) for s in df.attrs["seeds_present"]],
        "n_basins": n_basins,
        "positive_rate": round(positive_rate, 4),
        "cv_folds": CV_FOLDS,
        "cv_random_state": CV_RANDOM_STATE,
        "predictors": predictors_out,
        "delta_auc_vs_aridity": delta_auc_vs_aridity,
        "delta_auc_vs_q_mean": delta_auc_vs_q_mean,
        "delta_auc_vs_frac_snow": delta_auc_vs_frac_snow,
        "delta_balacc_vs_aridity": delta_balacc_vs_aridity,
        "delta_balacc_vs_q_mean": delta_balacc_vs_q_mean,
        "delta_balacc_vs_frac_snow": delta_balacc_vs_frac_snow,
        "excluded_invalid_baseline": {
            "name": "test_period_marginal_coverage",
            "reason": "label-leakage: label is derived from coverage_global, so AUC ~1.0 trivially",
        },
        "notes": notes,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")

    # ---- stdout report -------------------------------------------------------
    print(f"seed_used={out['seed_used']}  seeds_present={out['seeds_present_in_csv']}")
    print(f"n_basins={n_basins}  positive_rate={positive_rate:.4f}  "
          f"(severe undercoverage = coverage_global < {UNDERCOVERAGE_FLOOR})")
    print(f"n_positive={int(y.sum())}  n_negative={int((~y).sum())}")
    print("-" * 72)
    hdr = f"{'predictor':16s} {'AUC':>8s} {'oos_balacc':>11s} {'oos_prec':>9s} {'oos_rec':>8s}"
    print(hdr)
    for name, m in predictors_out.items():
        print(f"{name:16s} {m['auc']:8.4f} {m['oos_balanced_accuracy']:11.4f} "
              f"{m['oos_precision']:9.4f} {m['oos_recall']:8.4f}")
    print("-" * 72)
    print(f"delta_AUC sigma^2_b - aridity   = {delta_auc_vs_aridity:+.4f}")
    print(f"delta_AUC sigma^2_b - q_mean    = {delta_auc_vs_q_mean:+.4f}")
    print(f"delta_AUC sigma^2_b - frac_snow = {delta_auc_vs_frac_snow:+.4f}")
    print(f"delta_balacc sigma^2_b - aridity = {delta_balacc_vs_aridity:+.4f}")
    print("-" * 72)
    print("EXCLUDED invalid baseline: test-period marginal coverage "
          "(label-leakage, AUC ~1.0).")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
