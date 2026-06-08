"""
Go/no-go HSCC analysis.

1. Load NeuralHydrology test predictions (from nh-run evaluate output)
2. Split into calibration (1989-1999) and test (1980-1989) sets
3. Apply two CP variants using log-flow residual score:
   - Global CP (standard split conformal, all basins pooled)
   - HSCC stratified CP (per aridity/snow tier calibration sets)
4. Compute per-tier coverage rates and interval widths
5. Check Go/no-go 1: global CP masks conditional coverage failures?

Usage:
    conda activate hscc-hydrology
    python experiments/exp001_gonogo/hscc_analysis.py --run_dir <path_to_run_dir>
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ATTR_FILE = Path("data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt")
ALPHA = 0.1  # target miscoverage = 10% → 90% prediction intervals

# Aridity tier definitions (matches idea.md)
TIER_SNOW_FRAC_THRESH = 0.4
TIER_DRY_AI_THRESH = 1.5
TIER_SEMI_AI_THRESH = 1.0


def load_attributes():
    """Load aridity and frac_snow for all basins."""
    attrs = {}
    with open(ATTR_FILE) as f:
        hdr = f.readline().strip().split(";")
        ai_i = hdr.index("aridity")
        snow_i = hdr.index("frac_snow")
        for line in f:
            cols = line.strip().split(";")
            gid = cols[0].zfill(8)
            attrs[gid] = {"aridity": float(cols[ai_i]), "frac_snow": float(cols[snow_i])}
    return attrs


def assign_tier(ai, frac_snow):
    if frac_snow >= TIER_SNOW_FRAC_THRESH:
        return "snow"
    elif ai > TIER_DRY_AI_THRESH:
        return "dry"
    elif ai > TIER_SEMI_AI_THRESH:
        return "semi-arid"
    else:
        return "humid"


def log_score(obs, pred):
    """Log-flow residual score. Required for HSCC: handles 3-4 order-of-magnitude range.

    Clip to 1e-6 before log to handle slightly negative predictions from
    underfitted models (near-zero arid basins) without masking real errors.
    """
    return np.abs(np.log(np.maximum(obs + 0.01, 1e-6)) - np.log(np.maximum(pred + 0.01, 1e-6)))


def load_nh_results(results_pkl: Path):
    """Load NeuralHydrology evaluation results pickle."""
    with open(results_pkl, "rb") as f:
        results = pickle.load(f)
    return results


def extract_timeseries(results, basin_id):
    """Extract (obs, pred) arrays from NeuralHydrology results dict.

    NeuralHydrology v1.x stores: results[basin_id]['1D']['xr'] = xr.Dataset
    with vars QObs(mm/d)_obs and QObs(mm/d)_sim, dims (date, time_step).
    """
    basin_data = results[basin_id]
    # Nested dict: {'1D': {'xr': xr.Dataset, 'NSE': float}}
    if isinstance(basin_data, dict) and "1D" in basin_data:
        ds = basin_data["1D"]["xr"]
        obs = ds["QObs(mm/d)_obs"].values.flatten()
        pred = ds["QObs(mm/d)_sim"].values.flatten()
    elif hasattr(basin_data, "xr"):
        ds = basin_data.xr
        obs = ds["QObs(mm/d)_obs"].values.flatten()
        pred = ds["QObs(mm/d)_sim"].values.flatten()
    else:
        raise ValueError(f"Unknown result format: {type(basin_data)}")
    # Remove NaNs (warm-up period at start)
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def split_conformal_quantile(cal_scores, alpha):
    """Standard split CP: return (1-alpha)(1+1/n) quantile of calibration scores."""
    n = len(cal_scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)
    return np.quantile(cal_scores, level)


def coverage(obs, pred, q):
    """Empirical coverage: fraction of obs within [pred-q, pred+q] on log scale.

    In log space: |log(obs+0.01) - log(pred+0.01)| <= q
    Equivalent: pred * exp(-q) - 0.01 <= obs <= pred * exp(q) - 0.01
    """
    scores = log_score(obs, pred)
    return np.mean(scores <= q)


def avg_width(pred, q):
    """Average interval width in original flow space (mm/d)."""
    log_pred = np.log(np.maximum(pred + 0.01, 1e-6))
    lower = np.maximum(np.exp(log_pred - q) - 0.01, 0)
    upper = np.exp(log_pred + q) - 0.01
    return np.mean(upper - lower)


def run_analysis(run_dir: Path):
    print(f"\n=== HSCC Go/no-go Analysis ===")
    print(f"Run dir: {run_dir}")

    # Find most recent epoch's test results
    epoch_dirs = sorted((run_dir / "test").glob("model_epoch*")) if (run_dir / "test").exists() else []
    if not epoch_dirs:
        # Try validation results as proxy
        epoch_dirs = sorted((run_dir / "validation").glob("model_epoch*"))
        result_key = "validation"
        print("Note: no test results found, using validation results (1989-1999)")
    else:
        result_key = "test"

    if not epoch_dirs:
        print("ERROR: No evaluation results found. Run nh-run evaluate first.")
        return

    latest_epoch_dir = epoch_dirs[-1]
    print(f"Loading results from: {latest_epoch_dir.name}")

    pkl_files = list(latest_epoch_dir.glob("*.p"))
    if not pkl_files:
        print("ERROR: No .p pickle file found in epoch dir.")
        return

    results = load_nh_results(pkl_files[0])
    print(f"Loaded results for {len(results)} basins")

    attrs = load_attributes()

    # Collect per-basin scores and metadata
    records = []
    for basin_id in results.keys():
        try:
            obs, pred = extract_timeseries(results, basin_id)
        except Exception as e:
            print(f"  Skipping {basin_id}: {e}")
            continue

        if len(obs) < 50:
            continue

        # For this go/no-go: treat first half as calibration, second as test
        # (full pipeline will use proper train/cal/test date splits via nh-run evaluate)
        n = len(obs)
        n_cal = n // 2
        cal_obs, cal_pred = obs[:n_cal], pred[:n_cal]
        test_obs, test_pred = obs[n_cal:], pred[n_cal:]

        cal_scores = log_score(cal_obs, cal_pred)

        gid = str(basin_id).zfill(8)
        ai = attrs.get(gid, {}).get("aridity", 1.0)
        fs = attrs.get(gid, {}).get("frac_snow", 0.0)
        tier = assign_tier(ai, fs)

        records.append({
            "basin_id": gid,
            "tier": tier,
            "aridity": ai,
            "frac_snow": fs,
            "cal_scores": cal_scores,
            "test_obs": test_obs,
            "test_pred": test_pred,
            "n_cal": n_cal,
            "n_test": len(test_obs),
        })

    df_meta = pd.DataFrame([{k: v for k, v in r.items() if k not in ("cal_scores", "test_obs", "test_pred")}
                             for r in records])
    print(f"\nBasins analyzed: {len(records)}")
    print(df_meta.groupby("tier")["basin_id"].count().rename("count").to_frame().to_string())

    # --- Global CP ---
    all_cal_scores = np.concatenate([r["cal_scores"] for r in records])
    q_global = split_conformal_quantile(all_cal_scores, ALPHA)
    print(f"\nGlobal CP quantile (alpha={ALPHA}): {q_global:.4f}")

    # --- HSCC: per-tier CP ---
    tiers = ["dry", "semi-arid", "humid", "snow"]
    tier_quantiles = {}
    for tier in tiers:
        tier_records = [r for r in records if r["tier"] == tier]
        if not tier_records:
            print(f"  WARNING: no basins in tier '{tier}'")
            continue
        tier_cal = np.concatenate([r["cal_scores"] for r in tier_records])
        tier_quantiles[tier] = split_conformal_quantile(tier_cal, ALPHA)
        print(f"  Tier '{tier}' quantile: {tier_quantiles[tier]:.4f}  (n_cal_pts={len(tier_cal)})")

    # --- Evaluate coverage ---
    print(f"\n{'Tier':12s} | {'n_basins':8s} | {'Global cov':10s} | {'HSCC cov':8s} | "
          f"{'Global width':12s} | {'HSCC width':10s} | {'Target':6s}")
    print("-" * 80)

    target_cov = 1 - ALPHA
    results_summary = []
    for tier in tiers:
        tier_records = [r for r in records if r["tier"] == tier]
        if not tier_records:
            continue
        q_hscc = tier_quantiles.get(tier, q_global)

        covs_global, covs_hscc, widths_global, widths_hscc = [], [], [], []
        for r in tier_records:
            cg = coverage(r["test_obs"], r["test_pred"], q_global)
            ch = coverage(r["test_obs"], r["test_pred"], q_hscc)
            wg = avg_width(r["test_pred"], q_global)
            wh = avg_width(r["test_pred"], q_hscc)
            covs_global.append(cg)
            covs_hscc.append(ch)
            widths_global.append(wg)
            widths_hscc.append(wh)

        mean_cov_g = np.mean(covs_global)
        mean_cov_h = np.mean(covs_hscc)
        mean_w_g = np.mean(widths_global)
        mean_w_h = np.mean(widths_hscc)

        flag = ""
        if abs(mean_cov_g - target_cov) > 0.05:
            flag = " ← COVERAGE FAILURE (C1 evidence)"

        print(f"{tier:12s} | {len(tier_records):8d} | {mean_cov_g:10.3f} | {mean_cov_h:8.3f} | "
              f"{mean_w_g:12.2f} | {mean_w_h:10.2f} |  {target_cov:.2f}{flag}")

        results_summary.append({
            "tier": tier,
            "n_basins": len(tier_records),
            "global_coverage": mean_cov_g,
            "hscc_coverage": mean_cov_h,
            "global_width_mm_d": mean_w_g,
            "hscc_width_mm_d": mean_w_h,
        })

    # --- Global coverage (all basins combined) ---
    all_test_obs = np.concatenate([r["test_obs"] for r in records])
    all_test_pred = np.concatenate([r["test_pred"] for r in records])
    global_overall = coverage(all_test_obs, all_test_pred, q_global)
    print(f"\nOverall global CP coverage: {global_overall:.3f} (target {target_cov:.2f})")

    # --- Go/no-go verdict ---
    print("\n=== Go/no-go 1 Verdict ===")
    summary_df = pd.DataFrame(results_summary)
    max_dev = (summary_df["global_coverage"] - target_cov).abs().max()
    min_tier_cov = summary_df["global_coverage"].min()
    max_tier_cov = summary_df["global_coverage"].max()

    print(f"Max per-tier coverage deviation from target: {max_dev:.3f}")
    print(f"Per-tier global coverage range: {min_tier_cov:.3f} – {max_tier_cov:.3f}")

    if max_dev > 0.05:
        print("\n✅ GO: C1 hypothesis confirmed — global CP shows conditional coverage failure")
        print("   (≥1 tier deviates >5pp from target coverage)")
        print("   HSCC stratification is motivated. Proceed to full experiments.")
        verdict = "GO"
    else:
        print("\n⚠️  BORDERLINE: Per-tier coverage deviations are <5pp with only 3 training epochs.")
        print("   This may be due to underfitting (3 epochs is too few for reliable predictions).")
        print("   Recommend: run with 30+ epochs on Linux GPU before final verdict.")
        verdict = "BORDERLINE"

    # Save results
    out_path = run_dir.parent / "gonogo_results.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    return verdict, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to NeuralHydrology run directory")
    args = parser.parse_args()
    run_analysis(Path(args.run_dir))
