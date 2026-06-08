"""
Post-review item 1: connect the residual-variance mechanism to the snow transfer
boundary — source-to-target residual-variance KS distance vs snow coverage drop.

Reviewer concern (20250528 报告2, high-priority #1): the snow dominance violation is
not uniform; @fig-loro-heatmap shows it is driven by a subset of held-out regions. The
manuscript asserts in prose that these are the regions whose residual distributions are
least comparable to the source pool, but does not SHOW it. This script builds the
requested subplot data: per held-out region, the KS distance between the per-basin log
residual variance of the calibration source-pool snow basins and the held-out region's
snow basins, paired with that region's snow coverage drop (Global CP - HSCC).

This is pure post-processing of stored fold pickles (no retraining, no GPU). The
per-basin statistic is identical to the Fig 2 mechanism: the variance of signed log
residuals, var(log(obs+eps) - log(pred+eps)). Source residuals are calibration-period (the pool the HSCC snow
quantile is built from); target residuals are the held-out region's observed test-period
residuals (the held-out region has no calibration data in LORO). The asymmetry is
inherent to leave-one-region-out transfer and is stated in the figure caption.

Outputs:
  paper/data/tables/loro_snow_ks_vs_coverage.csv
  experiments/post_review_wrr_enhancements/results/item1_snow_ks_vs_coverage.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "exp003"))
from hscc_analysis_loro import (  # noqa: E402
    EPS,
    extract_basin_series,
    find_latest_epoch,
    load_attributes,
    load_nh_pickle,
    assign_tier,
)

RESULTS_ROOT = ROOT / "experiments" / "exp003" / "results"
OUT_CSV = ROOT / "paper" / "data" / "tables" / "loro_snow_ks_vs_coverage.csv"
OUT_JSON = ROOT / "experiments" / "post_review_wrr_enhancements" / "results" / "item1_snow_ks_vs_coverage.json"
SNOW_MIN_TARGET = 3  # match @fig-loro-heatmap minimum displayed-cell sample size


def find_fold_dir(huc: str) -> Path | None:
    """Return the anchor LORO fold used by the manuscript transfer heatmap."""
    anchor_candidates = sorted(
        c for c in RESULTS_ROOT.glob(f"exp003_loro_huc{huc}_*")
        if "_seed" not in c.name
    )
    fallback_candidates = sorted(RESULTS_ROOT.glob(f"exp003_loro_huc{huc}_*"))
    for c in reversed(anchor_candidates or fallback_candidates):
        if (c / "_analysis" / "metrics.json").exists():
            return c
    return None


def snow_basin_variances(results, attrs) -> np.ndarray:
    """Per-basin variance of signed log residuals for snow-tier basins."""
    out = []
    for basin_id in results.keys():
        try:
            obs, pred = extract_basin_series(results, basin_id)
        except Exception:
            continue
        if len(obs) < 100:
            continue
        gid = str(basin_id).zfill(8)
        a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0})
        if assign_tier(a["aridity"], a["frac_snow"]) != "snow":
            continue
        log_resid = np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6))
        out.append(float(np.var(log_resid)))
    return np.array(out)


def main() -> None:
    attrs = load_attributes()
    rows = []
    for k in range(1, 19):
        huc = f"{k:02d}"
        fold_dir = find_fold_dir(huc)
        if fold_dir is None:
            continue
        with open(fold_dir / "_analysis" / "metrics.json") as f:
            fm = json.load(f)
        snow_meta = fm.get("per_tier", {}).get("snow")
        if snow_meta is None:
            continue  # no snow tier in this held-out region

        val_epoch = find_latest_epoch(fold_dir, "validation")
        test_epoch = find_latest_epoch(fold_dir, "test")
        if val_epoch is None or test_epoch is None:
            continue
        val_res = load_nh_pickle(val_epoch, "validation")
        test_res = load_nh_pickle(test_epoch, "test")

        src_vars = snow_basin_variances(val_res, attrs)     # source-pool snow basins (cal period)
        tgt_vars = snow_basin_variances(test_res, attrs)    # held-out region snow basins (test period)
        n_tgt = int(tgt_vars.size)
        if n_tgt < SNOW_MIN_TARGET or src_vars.size < 2:
            continue

        ks = float(stats.ks_2samp(src_vars, tgt_vars).statistic)
        med_src = float(np.median(src_vars))
        med_tgt = float(np.median(tgt_vars))
        # Mechanism-faithful x-axis: source-to-target shift in residual-variance MAGNITUDE
        # (Fig 2 mechanism is about the level of per-basin signed-log-residual variance). The KS
        # statistic is also retained, but it discards the magnitude the mechanism predicts.
        var_log_ratio = float(np.log(med_tgt / med_src))
        gcov = float(snow_meta["global_coverage"])
        hcov = float(snow_meta["hscc_coverage"])
        rows.append({
            "huc": huc,
            "n_snow_source": int(src_vars.size),
            "n_snow_target": n_tgt,
            "ks_distance": ks,
            "median_source_log_resid_var": med_src,
            "median_target_log_resid_var": med_tgt,
            "var_shift_log_ratio": var_log_ratio,
            "global_snow_coverage": gcov,
            "hscc_snow_coverage": hcov,
            "coverage_drop_global_minus_hscc": gcov - hcov,
            "hscc_shortfall_below_nominal": 0.90 - hcov,
        })
        print(f"HUC-{huc}: n_snow={n_tgt:3d}  KS={ks:.3f}  var_shift(log)={var_log_ratio:+.2f}  "
              f"global={gcov:.3f}  hscc={hcov:.3f}  shortfall={0.90-hcov:+.3f}")

    df = pd.DataFrame(rows).sort_values("var_shift_log_ratio").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Spearman associations (small n, reported as diagnostic, not inferential)
    def _spear(x, y):
        if len(x) < 3:
            return None, None
        r = stats.spearmanr(x, y)
        return float(r.correlation), float(r.pvalue)

    rho_shift_short, p_shift_short = _spear(df["var_shift_log_ratio"], df["hscc_shortfall_below_nominal"])
    rho_shift_cov, p_shift_cov = _spear(df["var_shift_log_ratio"], df["hscc_snow_coverage"])
    rho_ks_short, p_ks_short = _spear(df["ks_distance"], df["hscc_shortfall_below_nominal"])
    summary = {
        "n_snow_regions": int(len(df)),
        "snow_min_target_basins": SNOW_MIN_TARGET,
        "spearman_var_shift_vs_shortfall_rho": rho_shift_short,
        "spearman_var_shift_vs_shortfall_p": p_shift_short,
        "spearman_var_shift_vs_hscc_coverage_rho": rho_shift_cov,
        "spearman_var_shift_vs_hscc_coverage_p": p_shift_cov,
        "spearman_ks_vs_shortfall_rho": rho_ks_short,
        "spearman_ks_vs_shortfall_p": p_ks_short,
        "var_shift_log_ratio_min": float(df["var_shift_log_ratio"].min()),
        "var_shift_log_ratio_max": float(df["var_shift_log_ratio"].max()),
        "ks_distance_min": float(df["ks_distance"].min()),
        "ks_distance_max": float(df["ks_distance"].max()),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote: {OUT_CSV}")
    print(f"wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
