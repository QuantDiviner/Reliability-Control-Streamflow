"""
exp003 LORO aggregate — combine 18 per-fold metrics.json into transfer matrix +
cross-fold summary, evaluate global PUB-success criteria.

Reads per-fold output written by hscc_analysis_loro.py:
  experiments/exp003/results/exp003_loro_huc{HUC}_<ts>/_analysis/metrics.json

Writes:
  experiments/exp003/transfer_matrix.csv
    one row per (HUC, tier) with: n_test, global_coverage, hscc_coverage,
    global_width, hscc_width, hscc_q, n_cal_basins, n_cal_pts, test_pred_min
  experiments/exp003/aggregated_metrics.json
    cross-fold marginals + S1'..S7' fold-level pass rates and gap-to-exp002

Optionally compares cross-fold HSCC averages with exp002 metrics
(experiments/exp002/results/<run>/metrics.json) to evaluate S7' (Tier-1 vs Tier-2 gap).

Usage:
    python experiments/exp003/aggregate_loro.py [--exp002_metrics PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "experiments" / "exp003" / "results"
OUT_TRANSFER = ROOT / "experiments" / "exp003" / "transfer_matrix.csv"
OUT_AGG = ROOT / "experiments" / "exp003" / "aggregated_metrics.json"

TIERS = ["dry", "semi_arid", "humid", "snow"]

# PUB fold-level S1'..S6' thresholds — must match hscc_analysis_loro.py
S1_GLOBAL_PASS_RATIO = 12 / 18  # ≥ 12/18 folds with all eligible tiers in [0.85, 0.92]
S2_MAX_SPREAD = 0.08
S3_MIN_GLOBAL_SPREAD = 0.10
S6_MIN_REDUCTION = 0.05
S6_FOLD_PASS_RATIO = 14 / 18
S7_GAP_LO = -0.05  # exp003 cov can be <= 5pp lower than exp002
S7_GAP_HI = 0.0

# n_min filter — must match hscc_analysis_loro.py TIER_MIN_TEST_BASINS
# Cells with n_test_basins < TIER_MIN ablate single-basin noise from cookbook
# labels and basin-weighted marginals. R3 (D-R3-exp003 Codex) requires this be
# applied uniformly across aggregate_loro.py and hscc_analysis_loro.py before A3.
TIER_MIN_TEST_BASINS = 3


def find_fold_metrics(huc: str) -> Path | None:
    pattern = f"exp003_loro_huc{huc}_*"
    candidates = sorted(RESULTS_ROOT.glob(pattern))
    for c in reversed(candidates):  # latest first
        m = c / "_analysis" / "metrics.json"
        if m.exists():
            return m
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp002_metrics",
        default=None,
        help="path to exp002 metrics.json for S7' gap evaluation",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    fold_metrics: dict[str, dict] = {}
    missing_folds: list[str] = []

    for k in range(1, 19):
        huc = f"{k:02d}"
        m_path = find_fold_metrics(huc)
        if m_path is None:
            missing_folds.append(huc)
            continue
        with open(m_path) as f:
            m = json.load(f)
        fold_metrics[huc] = m
        for t, td in m.get("per_tier", {}).items():
            rows.append(
                {
                    "huc": huc,
                    "tier": t,
                    "n_test_basins": td.get("n_basins"),
                    "n_cal_basins": td.get("n_cal_basins"),
                    "n_cal_pts": td.get("n_cal_pts"),
                    "global_coverage": td.get("global_coverage"),
                    "hscc_coverage": td.get("hscc_coverage"),
                    "global_width_mm_d": td.get("global_width_mm_d"),
                    "hscc_width_mm_d": td.get("hscc_width_mm_d"),
                    "hscc_q": td.get("hscc_q"),
                    "test_pred_min": td.get("test_pred_min"),
                    "global_q": m.get("global_q"),
                }
            )

    if not rows:
        raise SystemExit(
            "No fold metrics found. Run hscc_analysis_loro.py for at least one fold first."
        )

    df = pd.DataFrame(rows).sort_values(["huc", "tier"]).reset_index(drop=True)
    # Tag low-power cells (n_test < TIER_MIN_TEST_BASINS) so downstream cookbook
    # labelling (A3) can filter single-basin noise. These cells are still kept
    # in transfer_matrix.csv for transparency.
    df["low_power"] = df["n_test_basins"].astype(float) < TIER_MIN_TEST_BASINS
    df.to_csv(OUT_TRANSFER, index=False)

    # Cross-fold aggregates
    fold_spread_global = []
    fold_spread_hscc = []
    fold_pass_S1 = 0
    fold_pass_S6 = 0
    folds_evaluated = 0

    for huc, m in fold_metrics.items():
        sg = m.get("tier_coverage_spread_global_pp", 0) / 100.0
        sh = m.get("tier_coverage_spread_hscc_pp", 0) / 100.0
        fold_spread_global.append(sg)
        fold_spread_hscc.append(sh)
        sc = m.get("success_criteria_pub", {})
        if sc.get("S1_prime_per_tier"):
            fold_pass_S1 += 1
        if sc.get("S6_prime_spread_reduction"):
            fold_pass_S6 += 1
        folds_evaluated += 1

    avg_spread_global = float(np.mean(fold_spread_global)) if fold_spread_global else 0.0
    avg_spread_hscc = float(np.mean(fold_spread_hscc)) if fold_spread_hscc else 0.0

    # Per-tier marginal across folds (basin-weighted).
    # Two views are reported: unfiltered (all cells) and filtered (n_test ≥ TIER_MIN_TEST_BASINS).
    # The filtered view is the canonical one for cookbook labelling (R3 D-R3 §5).
    def _basin_weighted(sub: pd.DataFrame) -> dict | None:
        if sub.empty:
            return None
        n = sub["n_test_basins"].astype(float).values
        cg = sub["global_coverage"].astype(float).values
        ch = sub["hscc_coverage"].astype(float).values
        wg = sub["global_width_mm_d"].astype(float).values
        wh = sub["hscc_width_mm_d"].astype(float).values
        weights = n / n.sum() if n.sum() > 0 else np.ones_like(n) / len(n)
        return {
            "n_test_basins_total": int(n.sum()),
            "n_folds_with_tier": int(len(sub)),
            "global_coverage_basin_weighted": float(np.sum(cg * weights)),
            "hscc_coverage_basin_weighted": float(np.sum(ch * weights)),
            "global_width_basin_weighted": float(np.sum(wg * weights)),
            "hscc_width_basin_weighted": float(np.sum(wh * weights)),
        }

    per_tier_marginal: dict[str, dict] = {}
    per_tier_marginal_unfiltered: dict[str, dict] = {}
    low_power_dropped: dict[str, list[str]] = {}
    for t in TIERS:
        sub_all = df[df["tier"] == t]
        sub_full = sub_all[~sub_all["low_power"]]
        unfiltered = _basin_weighted(sub_all)
        filtered = _basin_weighted(sub_full)
        if unfiltered is not None:
            per_tier_marginal_unfiltered[t] = unfiltered
        if filtered is not None:
            per_tier_marginal[t] = filtered
        dropped = sub_all[sub_all["low_power"]]
        if not dropped.empty:
            low_power_dropped[t] = [
                f"HUC-{r.huc}(n={int(r.n_test_basins)})"
                for r in dropped.itertuples()
            ]

    # Compose criteria
    s1 = fold_pass_S1 / folds_evaluated >= S1_GLOBAL_PASS_RATIO if folds_evaluated else False
    s2 = avg_spread_hscc <= S2_MAX_SPREAD
    s3 = avg_spread_global >= S3_MIN_GLOBAL_SPREAD
    s6 = fold_pass_S6 / folds_evaluated >= S6_FOLD_PASS_RATIO if folds_evaluated else False

    # S5': humid HSCC ≥ 0.85 in ALL eligible folds (humid n_test ≥ TIER_MIN).
    # Folds with humid n=1/2 are excluded from the denominator (low statistical power).
    s5_total = 0
    s5_pass = 0
    s5_failures: list[str] = []
    s5_low_power_skipped: list[str] = []
    for huc, m in fold_metrics.items():
        humid_meta = m.get("per_tier", {}).get("humid")
        if not humid_meta:
            continue
        n_test = humid_meta.get("n_basins", 0)
        if n_test < TIER_MIN_TEST_BASINS:
            if n_test >= 1:
                s5_low_power_skipped.append(f"HUC-{huc} humid n={n_test}")
            continue
        s5_total += 1
        cov = humid_meta.get("hscc_coverage")
        if cov is not None and cov >= 0.85:
            s5_pass += 1
        else:
            s5_failures.append(f"HUC-{huc} humid={cov}")
    s5 = (s5_pass == s5_total) if s5_total else None

    # S7' gap to exp002
    s7 = None
    s7_detail = None
    if args.exp002_metrics:
        with open(args.exp002_metrics) as f:
            exp002_m = json.load(f)
        gaps = {}
        for t in TIERS:
            t1 = exp002_m.get("per_tier", {}).get(t, {}).get("hscc_coverage")
            t2 = per_tier_marginal.get(t, {}).get("hscc_coverage_basin_weighted")
            if t1 is None or t2 is None:
                continue
            gaps[t] = t2 - t1
        s7_detail = gaps
        if gaps:
            ok = all(S7_GAP_LO <= g <= S7_GAP_HI for g in gaps.values())
            s7 = bool(ok)

    out = {
        "n_folds_evaluated": folds_evaluated,
        "missing_folds": missing_folds,
        "tier_min_test_basins": TIER_MIN_TEST_BASINS,
        "low_power_cells_dropped": low_power_dropped,
        "avg_tier_coverage_spread_global_pp": avg_spread_global * 100,
        "avg_tier_coverage_spread_hscc_pp": avg_spread_hscc * 100,
        "fold_pass_S1_per_tier_count": fold_pass_S1,
        "fold_pass_S1_per_tier_ratio": fold_pass_S1 / folds_evaluated if folds_evaluated else 0,
        "fold_pass_S6_spread_reduction_count": fold_pass_S6,
        "fold_pass_S6_spread_reduction_ratio": (
            fold_pass_S6 / folds_evaluated if folds_evaluated else 0
        ),
        "fold_pass_S5_humid_count": s5_pass,
        "fold_pass_S5_humid_total": s5_total,
        "fold_pass_S5_humid_low_power_skipped": s5_low_power_skipped,
        "fold_S5_failures": s5_failures,
        "per_tier_marginal_basin_weighted": per_tier_marginal,
        "per_tier_marginal_basin_weighted_unfiltered": per_tier_marginal_unfiltered,
        "global_pub_success_criteria": {
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "S5": s5,
            "S6": s6,
            "S7_gap_to_exp002": s7,
        },
        "S7_gap_per_tier_pp": (
            None if s7_detail is None else {k: v * 100 for k, v in s7_detail.items()}
        ),
    }

    with open(OUT_AGG, "w") as f:
        json.dump(out, f, indent=2)

    print(f"transfer matrix      → {OUT_TRANSFER}")
    print(f"aggregated metrics   → {OUT_AGG}")
    print(f"folds evaluated:       {folds_evaluated}/18 (missing: {missing_folds or 'none'})")
    print(
        f"avg spread:            global = {avg_spread_global*100:.1f}pp, "
        f"HSCC = {avg_spread_hscc*100:.1f}pp"
    )
    print(
        f"fold pass S1' / S6' / S5' (n_test ≥ {TIER_MIN_TEST_BASINS}): "
        f"{fold_pass_S1}/{folds_evaluated} / "
        f"{fold_pass_S6}/{folds_evaluated} / {s5_pass}/{s5_total}"
    )
    if low_power_dropped:
        print("low-power cells (n_test < TIER_MIN), per tier:")
        for t, cells in low_power_dropped.items():
            print(f"  {t}: {', '.join(cells)}")
    if s7_detail is not None:
        print(f"S7' gap-to-exp002 per tier (pp): "
              + ", ".join(f"{k}={v*100:+.2f}" for k, v in s7_detail.items()))


if __name__ == "__main__":
    main()
