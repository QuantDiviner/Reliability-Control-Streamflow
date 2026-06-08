"""Analyze exp005 residuals: AR1, heteroscedasticity, per-basin/tier coverage,
and the H1 / H4 Wilcoxon tests across C0 vs C3 conditions.

Reads two NeuralHydrology run dirs (one per condition), pulls the test-period
predictions and observations, computes residuals, and produces:

- residual_diagnostics.csv : per-basin AR1 / heteroscedasticity for each condition
- condition_summary.json   : aggregate stats per condition + per-tier
- causal_test.json         : Wilcoxon paired test for H1 (AR1↑) and H4 (HSCC fails)
- hscc_per_tier_coverage.json : per-tier conformal coverage for each condition

Usage:
    python analyze_residuals.py \\
        --c0-run-dir <path-to-C0-run> \\
        --c3-run-dir <path-to-C3-run> \\
        --out-dir <path-to-write-outputs>
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path("/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP")
TIERS_CSV = PROJECT_ROOT / "experiments/exp002/basin_tiers.csv"
SYNTH_MANIFEST = PROJECT_ROOT / "experiments/exp005/data/synthetic_basins/c0/_manifest.json"
ALPHA = 0.10
TARGET_COVERAGE = 1.0 - ALPHA


def load_predictions(run_dir: Path, period: str = "test", epoch: int = 30) -> dict[str, dict[str, np.ndarray]]:
    """Load NH evaluation results pickle.

    Returns
    -------
    dict mapping basin_id -> {'obs': array, 'sim': array, 'date': array}
    """
    pkl = run_dir / period / f"model_epoch{epoch:03d}" / f"{period}_results.p"
    if not pkl.exists():
        raise FileNotFoundError(f"Missing: {pkl}")
    with open(pkl, "rb") as f:
        results = pickle.load(f)

    out: dict[str, dict[str, np.ndarray]] = {}
    for basin_id, freq_data in results.items():
        # NH stores per-frequency dict; default '1D'
        if "1D" in freq_data:
            xr_data = freq_data["1D"]["xr"]
        else:
            xr_data = list(freq_data.values())[0]["xr"]
        # Variables may be named 'QObs(mm/d)_obs' / 'QObs(mm/d)_sim'
        var_obs = next((v for v in xr_data.data_vars if "_obs" in v), None)
        var_sim = next((v for v in xr_data.data_vars if "_sim" in v), None)
        if var_obs is None or var_sim is None:
            continue
        obs = xr_data[var_obs].values.squeeze()
        sim = xr_data[var_sim].values.squeeze()
        date = xr_data.coords[next(iter(xr_data.coords))].values
        # Drop NaN obs (missing days)
        mask = ~np.isnan(obs)
        out[basin_id] = {"obs": obs[mask], "sim": sim[mask], "date": date[mask]}
    return out


def ar1(residuals: np.ndarray) -> float:
    """Lag-1 autocorrelation (Pearson)."""
    if len(residuals) < 3:
        return np.nan
    r = residuals - residuals.mean()
    denom = (r ** 2).sum()
    if denom < 1e-12:
        return np.nan
    return float((r[:-1] * r[1:]).sum() / denom)


def heteroscedasticity(obs: np.ndarray, residuals: np.ndarray) -> float:
    """Spearman correlation between |residual| and obs magnitude.

    Strong positive value indicates variance scales with obs magnitude.
    """
    if len(residuals) < 10:
        return np.nan
    rho, _ = stats.spearmanr(np.abs(residuals), obs)
    return float(rho)


def hscc_calibration(
    cal_obs: np.ndarray, cal_sim: np.ndarray,
    test_obs: np.ndarray, test_sim: np.ndarray,
    alpha: float = ALPHA, eps: float = 0.01,
) -> tuple[float, float]:
    """Apply per-basin split-conformal calibration on log_flow residual score.

    Returns
    -------
    coverage : float
    mean_width_log : float
        Mean interval width in log space (for sharpness).
    """
    if len(cal_obs) < 10 or len(test_obs) < 10:
        return np.nan, np.nan
    cal_scores = np.abs(np.log(cal_obs + eps) - np.log(np.maximum(cal_sim, 0.0) + eps))
    n_cal = len(cal_scores)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(cal_scores, q_level))
    # Test intervals in log space: log(sim) ± q_hat
    test_log_sim = np.log(np.maximum(test_sim, 0.0) + eps)
    test_log_obs = np.log(test_obs + eps)
    lower = test_log_sim - q_hat
    upper = test_log_sim + q_hat
    inside = (test_log_obs >= lower) & (test_log_obs <= upper)
    coverage = float(inside.mean())
    mean_width = float((upper - lower).mean())
    return coverage, mean_width


def split_basin(obs: np.ndarray, sim: np.ndarray, calib_frac: float = 0.5) -> tuple:
    n = len(obs)
    split = int(n * calib_frac)
    cal_obs, cal_sim = obs[:split], sim[:split]
    test_obs, test_sim = obs[split:], sim[split:]
    return cal_obs, cal_sim, test_obs, test_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c0-run-dir", required=True, type=Path)
    parser.add_argument("--c3-run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epoch", type=int, default=30)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading C0 predictions from {args.c0_run_dir}...")
    c0 = load_predictions(args.c0_run_dir, "test", args.epoch)
    print(f"Loading C3 predictions from {args.c3_run_dir}...")
    c3 = load_predictions(args.c3_run_dir, "test", args.epoch)

    common_basins = sorted(set(c0) & set(c3))
    print(f"Common basins: {len(common_basins)}")

    # Tiers
    tiers_df = pd.read_csv(TIERS_CSV, dtype={"gauge_id": str})
    tiers_df["gauge_id"] = tiers_df["gauge_id"].str.zfill(8)
    tier_map = dict(zip(tiers_df["gauge_id"], tiers_df["tier"]))
    # Synthetic manifest tier map (the 20 basins)
    manifest = json.loads(SYNTH_MANIFEST.read_text())
    synth_tier_map = {b["gauge_id"]: b["tier"] for b in manifest["basins"]}

    rows = []
    for bid in common_basins:
        # Normalise to zero-padded 8-char
        bid_norm = str(bid).zfill(8)
        tier = synth_tier_map.get(bid_norm, tier_map.get(bid_norm, "unknown"))

        for cond_name, data in [("c0", c0[bid]), ("c3", c3[bid])]:
            obs = data["obs"]
            sim = data["sim"]
            resid = obs - sim
            nse = 1.0 - (resid ** 2).sum() / ((obs - obs.mean()) ** 2).sum() if obs.var() > 0 else np.nan

            # HSCC calibration: split per-basin into half-half
            cal_obs, cal_sim, test_obs, test_sim = split_basin(obs, sim, 0.5)
            cov, width = hscc_calibration(cal_obs, cal_sim, test_obs, test_sim)

            rows.append({
                "basin": bid_norm,
                "tier": tier,
                "condition": cond_name,
                "n_days": len(obs),
                "nse": float(nse) if np.isfinite(nse) else np.nan,
                "ar1": ar1(resid),
                "het_spearman": heteroscedasticity(obs, resid),
                "mean_resid": float(resid.mean()),
                "mean_abs_resid": float(np.abs(resid).mean()),
                "hscc_coverage": cov,
                "hscc_mean_width_log": width,
            })

    df = pd.DataFrame(rows)
    csv_out = args.out_dir / "residual_diagnostics.csv"
    df.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out} ({len(df)} rows)")

    # Aggregate per condition
    summary = {}
    for cond in ("c0", "c3"):
        sub = df[df["condition"] == cond]
        summary[cond] = {
            "n_basins": int(len(sub)),
            "nse_mean": float(sub["nse"].mean()),
            "nse_median": float(sub["nse"].median()),
            "ar1_mean": float(sub["ar1"].mean()),
            "ar1_median": float(sub["ar1"].median()),
            "het_mean": float(sub["het_spearman"].mean()),
            "hscc_coverage_mean": float(sub["hscc_coverage"].mean()),
            "hscc_coverage_per_tier": {
                tier: float(sub[sub["tier"] == tier]["hscc_coverage"].mean())
                for tier in ["dry", "semi_arid", "humid", "snow"]
            },
            "hscc_mean_width_log_mean": float(sub["hscc_mean_width_log"].mean()),
        }
    summary_path = args.out_dir / "condition_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    # Causal tests (paired)
    pivot_ar1 = df.pivot_table(index="basin", columns="condition", values="ar1")
    pivot_cov = df.pivot_table(index="basin", columns="condition", values="hscc_coverage")
    pivot_nse = df.pivot_table(index="basin", columns="condition", values="nse")
    pivot_het = df.pivot_table(index="basin", columns="condition", values="het_spearman")

    def safe_wilcoxon(x, y, alternative="greater"):
        diff = (x - y).dropna()
        if len(diff) < 5:
            return None
        try:
            stat, pval = stats.wilcoxon(x.dropna().values, y.dropna().values, alternative=alternative)
            return {"statistic": float(stat), "pvalue": float(pval), "n_pairs": int(len(diff)),
                    "median_diff": float(diff.median())}
        except ValueError as e:
            return {"error": str(e)}

    causal = {
        "H1_ar1_increases_under_perturbation": safe_wilcoxon(pivot_ar1["c3"], pivot_ar1["c0"], "greater"),
        "H4_hscc_coverage_decreases_under_perturbation": safe_wilcoxon(pivot_cov["c3"], pivot_cov["c0"], "less"),
        "H_aux_nse_decreases": safe_wilcoxon(pivot_nse["c3"], pivot_nse["c0"], "less"),
        "H_aux_het_increases": safe_wilcoxon(pivot_het["c3"], pivot_het["c0"], "greater"),
        "alpha": ALPHA,
        "target_coverage": TARGET_COVERAGE,
    }
    causal_path = args.out_dir / "causal_test.json"
    with open(causal_path, "w") as f:
        json.dump(causal, f, indent=2)
    print(f"Wrote {causal_path}")

    # Per-tier HSCC coverage breakdown
    per_tier = {}
    for cond in ("c0", "c3"):
        sub = df[df["condition"] == cond]
        per_tier[cond] = {
            tier: {
                "n": int(len(sub[sub["tier"] == tier])),
                "coverage_mean": float(sub[sub["tier"] == tier]["hscc_coverage"].mean()),
                "coverage_std": float(sub[sub["tier"] == tier]["hscc_coverage"].std()),
                "deviation_pp": float(abs(sub[sub["tier"] == tier]["hscc_coverage"].mean() - TARGET_COVERAGE) * 100),
            }
            for tier in ["dry", "semi_arid", "humid", "snow"]
        }
    pt_path = args.out_dir / "hscc_per_tier_coverage.json"
    with open(pt_path, "w") as f:
        json.dump(per_tier, f, indent=2)
    print(f"Wrote {pt_path}")

    # Print compact summary
    print("\n=== exp005 Lean MVP — Causal Test Summary ===")
    print(f"  Common basins: {len(common_basins)}")
    print(f"  C0 mean NSE = {summary['c0']['nse_mean']:.3f}, AR1 = {summary['c0']['ar1_mean']:.3f}, HSCC cov = {summary['c0']['hscc_coverage_mean']:.3f}")
    print(f"  C3 mean NSE = {summary['c3']['nse_mean']:.3f}, AR1 = {summary['c3']['ar1_mean']:.3f}, HSCC cov = {summary['c3']['hscc_coverage_mean']:.3f}")
    print()
    print(f"  H1 (AR1 ↑ under perturbation, paired Wilcoxon greater): "
          f"p = {causal['H1_ar1_increases_under_perturbation'].get('pvalue', 'NA')}")
    print(f"  H4 (HSCC cov ↓ under perturbation, paired Wilcoxon less): "
          f"p = {causal['H4_hscc_coverage_decreases_under_perturbation'].get('pvalue', 'NA')}")
    print()
    print("  HSCC coverage per tier:")
    print(f"    {'tier':<10} {'C0':>6} {'C3':>6}")
    for tier in ["dry", "semi_arid", "humid", "snow"]:
        c0_cov = summary['c0']['hscc_coverage_per_tier'][tier]
        c3_cov = summary['c3']['hscc_coverage_per_tier'][tier]
        print(f"    {tier:<10} {c0_cov:>6.3f} {c3_cov:>6.3f}")


if __name__ == "__main__":
    main()
