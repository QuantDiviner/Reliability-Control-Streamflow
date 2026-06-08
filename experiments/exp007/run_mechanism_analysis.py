"""
exp007 — Residual Mechanism Analysis (C2.5 主证据)

复用 exp002 checkpoint 提取每 basin 的:
  - LSTM cal-period residuals (1990-2000)
  - 8 mechanism variables (AR1, het, NSE, KGE, log_resid_var, snow_seasonal,
                            low_flow_bias, high_flow_bias)
  - per-basin miscoverage (global CP and HSCC)

Compute:
  - Spearman correlation matrix (8 mechanism vars × 2 miscoverage targets)
  - Q1-Q4 quadrant analysis (AR1 × heteroscedasticity)
  - Scatter plot data tables

Gate G1: Spearman(AR1, miscoverage_global) >= 0.4
      OR Spearman(Het, miscoverage_global) >= 0.4
      → C2.5 主柱成立
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/exp002"))
from hscc_analysis_v2 import (
    ALPHA, EPS,
    load_attributes, assign_tier,
    log_score, split_cp_quantile,
    coverage_log,
    extract_basin_series, load_nh_pickle, find_latest_epoch,
)

TIERS = ["dry", "semi_arid", "humid", "snow"]


def lag1_autocorr(x):
    if len(x) < 3:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def heteroscedasticity_index(residuals, q):
    """std(|r| at Q90+) / std(|r| at Q10-) — high = heteroscedastic."""
    r_abs = np.abs(residuals)
    q90 = np.quantile(q, 0.90)
    q10 = np.quantile(q, 0.10)
    high = r_abs[q >= q90]
    low = r_abs[q <= q10]
    if len(high) < 5 or len(low) < 5:
        return np.nan
    s_low = float(np.std(low))
    s_high = float(np.std(high))
    if s_low < 1e-9:
        return np.nan
    return s_high / s_low


def nse(obs, sim):
    if len(obs) < 2:
        return np.nan
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom < 1e-9:
        return np.nan
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


def kge(obs, sim):
    if len(obs) < 2 or obs.std() < 1e-9 or sim.std() < 1e-9:
        return np.nan
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(sim.std() / obs.std())
    beta = float(sim.mean() / obs.mean()) if obs.mean() > 1e-9 else np.nan
    if not np.isfinite(beta):
        return np.nan
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def low_high_bias(obs, sim, lo_q=0.25, hi_q=0.75):
    if len(obs) < 4:
        return np.nan, np.nan
    qlo = np.quantile(obs, lo_q)
    qhi = np.quantile(obs, hi_q)
    low_mask = obs <= qlo
    high_mask = obs >= qhi
    low_bias = float((sim[low_mask] - obs[low_mask]).mean()) if low_mask.any() else np.nan
    high_bias = float((sim[high_mask] - obs[high_mask]).mean()) if high_mask.any() else np.nan
    return low_bias, high_bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp002_run_dir", default="experiments/exp002/results/exp002_camels_us_temporal_2504_155058", type=str)
    parser.add_argument("--out_dir", default="experiments/exp007/results/run_2604", type=str)
    args = parser.parse_args()

    run_dir = Path(args.exp002_run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== exp007 Mechanism Analysis ===")
    print(f"source: {run_dir}")
    print(f"out:    {out_dir}")

    val_epoch = find_latest_epoch(run_dir, "validation")
    test_epoch = find_latest_epoch(run_dir, "test")
    val_res = load_nh_pickle(val_epoch, "validation")
    test_res = load_nh_pickle(test_epoch, "test")
    attrs = load_attributes()

    # Step 1: Extract per-basin cal/test data + compute mechanism variables
    print("\n[step1] extracting per-basin residuals + mechanism variables")
    rows = []
    cal_data = {}  # for HSCC q computation
    test_data = {}

    for basin_id in val_res.keys():
        if basin_id not in test_res:
            continue
        try:
            cal_obs, cal_pred = extract_basin_series(val_res, basin_id)
            tst_obs, tst_pred = extract_basin_series(test_res, basin_id)
        except Exception:
            continue
        if len(cal_obs) < 100 or len(tst_obs) < 100:
            continue
        gid = str(basin_id).zfill(8)
        a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0})
        tier = assign_tier(a["aridity"], a["frac_snow"])

        cal_resid = cal_obs - cal_pred
        nse_test = nse(tst_obs, tst_pred)
        kge_test = kge(tst_obs, tst_pred)
        ar1 = lag1_autocorr(cal_resid)
        het = heteroscedasticity_index(cal_resid, cal_obs)
        log_resid = np.log(np.maximum(cal_obs + EPS, 1e-6)) - np.log(np.maximum(cal_pred + EPS, 1e-6))
        log_resid_var = float(np.var(log_resid))
        low_bias, high_bias = low_high_bias(cal_obs, cal_pred)

        rows.append({
            "basin_id": gid,
            "tier": tier,
            "aridity": a["aridity"],
            "frac_snow": a["frac_snow"],
            "n_cal": len(cal_obs),
            "n_test": len(tst_obs),
            # 8 mechanism variables
            "AR1": ar1,
            "Het": het,
            "NSE_test": nse_test,
            "KGE_test": kge_test,
            "log_resid_var": log_resid_var,
            "snow_seasonal": a["frac_snow"],  # placeholder: snow×seasonal_concentration; using frac_snow alone for v1
            "low_flow_bias": low_bias,
            "high_flow_bias": high_bias,
        })
        cal_data[gid] = log_score(cal_obs, cal_pred)
        test_data[gid] = (tst_obs, tst_pred, tier)

    print(f"[step1] {len(rows)} basins analyzed")

    # Step 2: Compute global CP q and HSCC q_tier
    all_cal = np.concatenate([cal_data[r["basin_id"]] for r in rows])
    q_global = split_cp_quantile(all_cal, ALPHA)
    q_tier = {}
    for t in TIERS:
        s_list = [cal_data[r["basin_id"]] for r in rows if r["tier"] == t]
        if s_list:
            q_tier[t] = split_cp_quantile(np.concatenate(s_list), ALPHA)
    print(f"[step2] q_global = {q_global:.4f}")
    print(f"[step2] q_tier = {{ {', '.join(f'{t}: {q_tier[t]:.4f}' for t in TIERS if t in q_tier)} }}")

    # Step 3: Per-basin miscoverage
    print("[step3] computing per-basin miscoverage")
    target = 1 - ALPHA
    for r in rows:
        gid = r["basin_id"]
        tst_obs, tst_pred, tier = test_data[gid]
        cov_g = coverage_log(tst_obs, tst_pred, q_global)
        cov_h = coverage_log(tst_obs, tst_pred, q_tier.get(tier, np.nan))
        r["coverage_global"] = float(cov_g)
        r["coverage_hscc"] = float(cov_h)
        r["miscoverage_global"] = float(abs(cov_g - target))
        r["miscoverage_hscc"] = float(abs(cov_h - target))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "mechanism_metrics.csv", index=False)
    print(f"[step3] wrote: {out_dir/'mechanism_metrics.csv'} ({len(df)} basins, {len(df.columns)} cols)")

    # Step 4: Spearman correlation matrix
    print("\n[step4] Spearman correlation matrix")
    mech_vars = ["AR1", "Het", "NSE_test", "KGE_test", "log_resid_var",
                 "snow_seasonal", "low_flow_bias", "high_flow_bias"]
    targets = ["miscoverage_global", "miscoverage_hscc", "coverage_global", "coverage_hscc"]
    spearman_rows = []
    for var in mech_vars:
        for tgt in targets:
            x = df[var].values
            y = df[tgt].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 30:
                spearman_rows.append({"var": var, "target": tgt, "n": int(mask.sum()),
                                       "spearman_rho": np.nan, "spearman_p": np.nan})
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            spearman_rows.append({
                "var": var, "target": tgt, "n": int(mask.sum()),
                "spearman_rho": float(rho), "spearman_p": float(p),
                "abs_rho": float(abs(rho)),
            })
    spearman_df = pd.DataFrame(spearman_rows)
    spearman_df.to_csv(out_dir / "spearman_matrix.csv", index=False)

    # Highlight Gate G1
    g1_pairs = spearman_df[(spearman_df["var"].isin(["AR1", "Het"])) &
                            (spearman_df["target"] == "miscoverage_global")]
    print("\n[Gate G1] Spearman of {AR1, Het} vs miscoverage_global:")
    for _, r in g1_pairs.iterrows():
        ok = "✅ PASS" if abs(r["spearman_rho"]) >= 0.4 else "❌"
        print(f"  ρ({r['var']:5s}, miscov_global) = {r['spearman_rho']:+.3f} (p={r['spearman_p']:.4f}, n={r['n']}) {ok}")
    g1_pass = bool((g1_pairs["abs_rho"] >= 0.4).any())
    print(f"\n[Gate G1] C2.5 主柱: {'✅ 成立' if g1_pass else '❌ 触发 R3 重审'}")

    # Step 5: Q1-Q4 quadrant analysis (AR1 × Het)
    print("\n[step5] Q1-Q4 quadrant analysis")
    valid = df.dropna(subset=["AR1", "Het", "miscoverage_global"])
    ar1_med = valid["AR1"].median()
    het_med = valid["Het"].median()

    def quadrant(row):
        hi_ar1 = row["AR1"] >= ar1_med
        hi_het = row["Het"] >= het_med
        if not hi_ar1 and not hi_het:
            return "Q1_low_low_reliable"
        if not hi_ar1 and hi_het:
            return "Q2_low_high_marginal_het"
        if hi_ar1 and not hi_het:
            return "Q3_high_low_marginal_ar1"
        return "Q4_high_high_unreliable"

    valid = valid.copy()
    valid["quadrant"] = valid.apply(quadrant, axis=1)
    quad_summary = valid.groupby("quadrant").agg(
        n_basins=("basin_id", "count"),
        mean_miscov_global=("miscoverage_global", "mean"),
        mean_miscov_hscc=("miscoverage_hscc", "mean"),
        mean_NSE=("NSE_test", "mean"),
    ).reset_index()
    quad_summary.to_csv(out_dir / "quadrant_summary.csv", index=False)
    print("[step5] quadrant means:")
    print(quad_summary.to_string(index=False))

    q1_miscov = float(quad_summary.loc[quad_summary["quadrant"] == "Q1_low_low_reliable", "mean_miscov_global"].iloc[0])
    q4_miscov = float(quad_summary.loc[quad_summary["quadrant"] == "Q4_high_high_unreliable", "mean_miscov_global"].iloc[0])
    q4_q1_diff = q4_miscov - q1_miscov

    # Step 6: Save final results.json
    results = {
        "exp": "exp007_mechanism_analysis",
        "n_basins": int(len(df)),
        "alpha": ALPHA,
        "q_global": float(q_global),
        "q_tier": {t: float(v) for t, v in q_tier.items()},
        "median_AR1": float(ar1_med),
        "median_Het": float(het_med),
        "gate_G1": {
            "criterion": "Spearman(AR1 or Het, miscoverage_global) >= 0.4",
            "spearman_AR1_miscov": float(g1_pairs.loc[g1_pairs["var"] == "AR1", "spearman_rho"].iloc[0]),
            "spearman_Het_miscov": float(g1_pairs.loc[g1_pairs["var"] == "Het", "spearman_rho"].iloc[0]),
            "passed": g1_pass,
        },
        "quadrant_analysis": {
            "Q1_reliable_mean_miscov": q1_miscov,
            "Q4_unreliable_mean_miscov": q4_miscov,
            "Q4_Q1_diff": q4_q1_diff,
            "S2_pass_diff_ge_005": bool(q4_q1_diff >= 0.05),
        },
        "spearman_top5_abs": spearman_df.dropna().nlargest(5, "abs_rho")[["var", "target", "spearman_rho", "spearman_p", "n"]].to_dict(orient="records"),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote: {out_dir/'results.json'}")
    print(f"[done] Q4-Q1 miscov diff = {q4_q1_diff:.3f} (S2: {results['quadrant_analysis']['S2_pass_diff_ge_005']})")


if __name__ == "__main__":
    main()
