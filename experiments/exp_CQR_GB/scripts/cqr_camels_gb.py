"""
exp_CQR_GB — residual-CQR on CAMELS-GB native 50-basin prediction set.

Implements two bounded methods authorized by D-050:
  1. vanilla Global residual-CQR with one pooled conformity correction
  2. Mondrian residual-CQR with GB-tier-specific models and corrections

The metric calculations mirror exp011 Phase 2 where possible.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import QuantileRegressor

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)

ROOT = Path(__file__).resolve().parents[3]
ALPHA = 0.10
EPS = 0.01

GB_TIERS = ["gb_drier_q4", "gb_mid", "gb_wet", "gb_montane"]
EXP004_RUN = ROOT / "experiments/exp004/results/exp004_camels_gb_native_2604_214507"
GB_TIERS_CSV = ROOT / "experiments/exp004/basin_lists/gb_basin_tiers.csv"
OUT_DIR = ROOT / "experiments/exp_CQR_GB/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_epoch(run_dir, kind):
    return sorted((Path(run_dir) / kind).glob("model_epoch*"))[-1]


def extract(results, basin):
    bd = results[basin]
    ds = bd["1D"]["xr"]
    obs_key = "discharge_spec_obs" if "discharge_spec_obs" in ds.data_vars else "QObs(mm/d)_obs"
    sim_key = "discharge_spec_sim" if "discharge_spec_sim" in ds.data_vars else "QObs(mm/d)_sim"
    obs = ds[obs_key].values.flatten()
    pred = ds[sim_key].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def split_cp_quantile(scores, alpha):
    n = len(scores)
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


def make_features(pred, cal_mean, cal_sd):
    n = len(pred)
    return np.column_stack([
        pred,
        np.log(np.maximum(pred + EPS, 1e-6)),
        np.full(n, cal_mean),
        np.full(n, cal_sd),
    ])


def pit_value(obs, lo, hi, alpha=ALPHA):
    obs = np.asarray(obs)
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    pit = np.empty_like(obs, dtype=float)
    width = np.maximum(hi - lo, 1e-12)
    inside = (obs >= lo) & (obs <= hi)
    below = obs < lo
    above = obs > hi
    pit[inside] = alpha / 2 + (1 - alpha) * (obs[inside] - lo[inside]) / width[inside]
    pit[below] = np.clip((alpha / 2) * np.maximum(obs[below], 0) / np.maximum(lo[below], 1e-6), 0, alpha / 2)
    pit[above] = np.clip(
        1 - alpha / 2 + (alpha / 2) * (obs[above] - hi[above]) / np.maximum(hi[above], 1e-6),
        1 - alpha / 2,
        1.0,
    )
    return pit


def crps_pwl_vectorized(obs, lo, hi, alpha=ALPHA):
    obs = np.asarray(obs, dtype=float)
    lo = np.maximum(np.asarray(lo, dtype=float), 1e-9)
    hi = np.maximum(np.asarray(hi, dtype=float), lo + 1e-9)
    y = obs
    bp0 = np.zeros_like(y)
    bp1 = lo
    bp2 = hi
    bp3 = 2 * hi
    bp4 = np.maximum(np.maximum(y, bp3) * 1.5, bp3 + 1.0)

    a1 = np.zeros_like(lo); b1 = (alpha / 2) / lo; c1 = np.zeros_like(lo)
    a2 = np.full_like(lo, alpha / 2); b2 = (1 - alpha) / (hi - lo); c2 = lo
    a3 = np.full_like(lo, 1 - alpha / 2); b3 = (alpha / 2) / hi; c3 = hi
    a4 = np.full_like(lo, 1.0); b4 = np.zeros_like(lo); c4 = 2 * hi

    def integral_f_sq(a, b, c, z0, z1):
        valid = z1 > z0
        out = np.zeros_like(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            nonzero = (b != 0) & valid
            v1 = a + b * (z1 - c)
            v0 = a + b * (z0 - c)
            out = np.where(nonzero, (v1 ** 3 - v0 ** 3) / (3 * b), out)
        return np.where((b == 0) & valid, a ** 2 * (z1 - z0), out)

    def integral_1mf_sq(a, b, c, z0, z1):
        return integral_f_sq(1 - a, -b, c, z0, z1)

    def integrate_below(a, b, c, z_lo, z_hi):
        upper = np.minimum(z_hi, y)
        z0 = np.minimum(z_lo, upper)
        return integral_f_sq(a, b, c, z0, upper)

    def integrate_above(a, b, c, z_lo, z_hi):
        lower = np.maximum(z_lo, y)
        z1 = np.maximum(z_hi, lower)
        return integral_1mf_sq(a, b, c, lower, z1)

    crps_below = (
        integrate_below(a1, b1, c1, bp0, bp1)
        + integrate_below(a2, b2, c2, bp1, bp2)
        + integrate_below(a3, b3, c3, bp2, bp3)
        + integrate_below(a4, b4, c4, bp3, bp4)
    )
    crps_above = (
        integrate_above(a1, b1, c1, bp0, bp1)
        + integrate_above(a2, b2, c2, bp1, bp2)
        + integrate_above(a3, b3, c3, bp2, bp3)
        + integrate_above(a4, b4, c4, bp3, bp4)
    )
    return crps_below + crps_above


def winkler_score_per_point(obs, lo, hi, alpha):
    width = hi - lo
    below = obs < lo
    above = obs > hi
    pen = np.zeros_like(obs, dtype=float)
    pen[below] = (2.0 / alpha) * (lo[below] - obs[below])
    pen[above] = (2.0 / alpha) * (obs[above] - hi[above])
    return width + pen


def reliability_dev(pit_vals, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    emp = np.array([np.mean(pit_vals <= edges[i + 1]) for i in range(n_bins)])
    return float(np.mean(np.abs(emp - centers)))


def train_qr(x_cal, r_cal, tag):
    fit_x, fit_r = x_cal, r_cal
    fit_cap = 10_000
    if len(fit_r) > fit_cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(fit_r), fit_cap, replace=False)
        fit_x, fit_r = fit_x[idx], fit_r[idx]
    print(f"  training {tag}: n_cal={len(r_cal)} fit_n={len(fit_r)}")
    lo = QuantileRegressor(quantile=ALPHA / 2, alpha=1e-3, solver="highs").fit(fit_x, fit_r)
    hi = QuantileRegressor(quantile=1 - ALPHA / 2, alpha=1e-3, solver="highs").fit(fit_x, fit_r)
    e = np.maximum(lo.predict(x_cal) - r_cal, r_cal - hi.predict(x_cal))
    return lo, hi, split_cp_quantile(e, ALPHA)


def main():
    print("=== exp_CQR_GB: CAMELS-GB Global and Mondrian residual-CQR ===\n")
    tier_df = pd.read_csv(GB_TIERS_CSV, dtype={"gauge_id": str})
    tier_df["gauge_id"] = tier_df["gauge_id"].astype(str)
    tier_map = dict(zip(tier_df["gauge_id"], tier_df["tier"]))

    val_p = find_latest_epoch(EXP004_RUN, "validation") / "validation_results.p"
    test_p = find_latest_epoch(EXP004_RUN, "test") / "test_results.p"
    print(f"loading val={val_p} test={test_p}")
    val = pickle.load(open(val_p, "rb"))
    test = pickle.load(open(test_p, "rb"))

    records = []
    for basin in val.keys():
        if basin not in test:
            continue
        b_str = str(basin).lstrip("0") if str(basin).startswith("0") else str(basin)
        if b_str not in tier_map:
            continue
        cal_obs, cal_pred = extract(val, basin)
        tst_obs, tst_pred = extract(test, basin)
        if len(cal_obs) < 100 or len(tst_obs) < 100:
            continue
        cal_mean = float(np.mean(cal_obs))
        cal_sd = float(np.std(cal_obs, ddof=1))
        records.append({
            "basin": b_str,
            "tier": tier_map[b_str],
            "x_cal": make_features(cal_pred, cal_mean, cal_sd),
            "r_cal": cal_obs - cal_pred,
            "x_test": make_features(tst_pred, cal_mean, cal_sd),
            "obs_test": tst_obs,
            "pred_test": tst_pred,
        })
    print(f"basins analyzed: {len(records)}")
    print(f"tier counts: {pd.Series([r['tier'] for r in records]).value_counts().to_dict()}")

    x_all = np.vstack([r["x_cal"] for r in records])
    r_all = np.concatenate([r["r_cal"] for r in records])
    g_lo, g_hi, g_qhat = train_qr(x_all, r_all, "Global_CQR")
    print(f"  Global_CQR Q_hat={g_qhat:.6f}")

    tier_models = {}
    for tier in GB_TIERS:
        subset = [r for r in records if r["tier"] == tier]
        if not subset:
            continue
        x_t = np.vstack([r["x_cal"] for r in subset])
        r_t = np.concatenate([r["r_cal"] for r in subset])
        lo, hi, qhat = train_qr(x_t, r_t, f"Mondrian_CQR[{tier}]")
        tier_models[tier] = (lo, hi, qhat)
        print(f"  Mondrian_CQR[{tier}] Q_hat={qhat:.6f}")

    rows = []
    pit_store = {m: {t: [] for t in GB_TIERS} for m in ["Global_CQR", "Mondrian_CQR"]}
    for rec in records:
        tier = rec["tier"]
        obs = rec["obs_test"]
        pred = rec["pred_test"]
        x = rec["x_test"]
        candidates = [("Global_CQR", g_lo, g_hi, g_qhat)]
        if tier in tier_models:
            candidates.append(("Mondrian_CQR", *tier_models[tier]))
        for method, lo_model, hi_model, qhat in candidates:
            ql = lo_model.predict(x)
            qh = hi_model.predict(x)
            lo = np.maximum(pred + ql - qhat, 0.0)
            hi = pred + qh + qhat
            coverage = float(np.mean((obs >= lo) & (obs <= hi)))
            pit = pit_value(obs, lo, hi)
            pit_store[method][tier].append(pit)
            rows.append({
                "basin": rec["basin"],
                "tier": tier,
                "method": method,
                "n_test": int(len(obs)),
                "coverage": coverage,
                "coverage_eps": coverage - (1 - ALPHA),
                "mean_pi_width": float(np.mean(hi - lo)),
                "winkler_score": float(np.mean(winkler_score_per_point(obs, lo, hi, ALPHA))),
                "crps": float(np.mean(crps_pwl_vectorized(obs, lo, hi))),
                "Q_hat": float(qhat),
            })

    per_basin = pd.DataFrame(rows)
    per_basin.to_csv(OUT_DIR / "cqr_gb_per_basin.csv", index=False)

    tier_rows = []
    for method in ["Global_CQR", "Mondrian_CQR"]:
        for tier in GB_TIERS:
            sub = per_basin[(per_basin["method"] == method) & (per_basin["tier"] == tier)]
            if sub.empty:
                continue
            pit = np.concatenate(pit_store[method][tier])
            ks = stats.kstest(pit, "uniform")
            tier_rows.append({
                "method": method,
                "tier": tier,
                "n_basins": int(len(sub)),
                "mean_coverage": float(sub["coverage"].mean()),
                "std_coverage": float(sub["coverage"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "mean_pi_width": float(sub["mean_pi_width"].mean()),
                "mean_winkler": float(sub["winkler_score"].mean()),
                "mean_crps": float(sub["crps"].mean()),
                "pit_ks_stat": float(ks.statistic),
                "pit_ks_pvalue": float(ks.pvalue),
                "reliability_dev_mean": reliability_dev(pit),
                "pit_n_points": int(len(pit)),
                "Q_hat": float(sub["Q_hat"].iloc[0]),
            })
    per_tier = pd.DataFrame(tier_rows)
    per_tier.to_csv(OUT_DIR / "cqr_gb_per_tier.csv", index=False)

    marginal_rows = []
    for method in ["Global_CQR", "Mondrian_CQR"]:
        sub = per_basin[per_basin["method"] == method]
        tsub = per_tier[per_tier["method"] == method]
        all_pit = np.concatenate([np.concatenate(pit_store[method][t]) for t in GB_TIERS if pit_store[method][t]])
        ks = stats.kstest(all_pit, "uniform")
        marginal_rows.append({
            "method": method,
            "n_basins": int(sub["basin"].nunique()),
            "marginal_coverage": float(sub["coverage"].mean()),
            "marginal_width": float(sub["mean_pi_width"].mean()),
            "marginal_winkler": float(sub["winkler_score"].mean()),
            "marginal_crps": float(sub["crps"].mean()),
            "spread_pp": float((tsub["mean_coverage"].max() - tsub["mean_coverage"].min()) * 100),
            "pit_ks_stat_marginal": float(ks.statistic),
            "pit_ks_pvalue_marginal": float(ks.pvalue),
            "reliability_dev_marginal": reliability_dev(all_pit),
        })
    marginal = pd.DataFrame(marginal_rows)
    marginal.to_csv(OUT_DIR / "cqr_gb_marginal.csv", index=False)

    summary = {
        "experiment": "exp_CQR_GB",
        "alpha": ALPHA,
        "dataset": "CAMELS-GB native exp004 seed=42, 50 basins",
        "methods": ["Global_CQR", "Mondrian_CQR"],
        "n_basins": int(per_basin["basin"].nunique()),
        "marginal": marginal.to_dict(orient="records"),
        "per_tier": per_tier.to_dict(orient="records"),
        "scope_note": "Bounded D-050 repair; no HopCPT-GB or exp012 executed.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Marginal summary ===")
    print(marginal.to_string(index=False))
    print("\n=== Per-tier summary ===")
    print(per_tier.to_string(index=False))
    print(f"\nwrote outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
