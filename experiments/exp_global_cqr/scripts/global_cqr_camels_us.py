"""
exp_global_cqr — Vanilla Romano 2019 Global CQR on CAMELS-US gauged 671 basins.

Vectorized CRPS and PIT calculation matching exp011 Phase 2.
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
TIERS = ["dry", "semi_arid", "humid", "snow"]
TIER_PRETTY = {"dry": "Dry", "semi_arid": "Semi-arid", "humid": "Humid", "snow": "Snow"}

EXP002_RUN = ROOT / "experiments/exp002/results/exp002_camels_us_temporal_2504_155058"
ATTR_FILE = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
OUT_DIR = ROOT / "experiments/exp_global_cqr/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log_print(msg):
    print(msg, flush=True)


def load_attrs():
    attrs = {}
    with open(ATTR_FILE) as f:
        hdr = f.readline().strip().split(";")
        ai_i, sn_i = hdr.index("aridity"), hdr.index("frac_snow")
        pm_i = hdr.index("p_mean")
        for line in f:
            cols = line.strip().split(";")
            if not cols or not cols[0]:
                continue
            attrs[cols[0].zfill(8)] = {"aridity": float(cols[ai_i]),
                                        "frac_snow": float(cols[sn_i]),
                                        "p_mean": float(cols[pm_i])}
    return attrs


def assign_tier(ai, fs):
    if fs >= 0.40: return "snow"
    if ai > 1.5: return "dry"
    if ai > 1.0: return "semi_arid"
    return "humid"


def find_latest_epoch(rd, k):
    return sorted((Path(rd) / k).glob("model_epoch*"))[-1]


def extract(results, b):
    bd = results[b]
    ds = bd["1D"]["xr"]
    obs = ds["QObs(mm/d)_obs"].values.flatten()
    pred = ds["QObs(mm/d)_sim"].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def split_cp_quantile(scores, alpha):
    n = len(scores)
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


def make_features(pred, ai, fs, pm):
    n = len(pred)
    return np.column_stack([
        pred,
        np.log(np.maximum(pred + EPS, 1e-6)),
        np.full(n, ai),
        np.full(n, fs),
        np.full(n, pm),
    ])


def pit_value(obs, lo, hi, alpha=ALPHA):
    obs = np.asarray(obs); lo = np.asarray(lo); hi = np.asarray(hi)
    pit = np.empty_like(obs, dtype=float)
    width = np.maximum(hi - lo, 1e-12)
    inside = (obs >= lo) & (obs <= hi)
    below = obs < lo
    above = obs > hi
    pit[inside] = alpha / 2 + (1 - alpha) * (obs[inside] - lo[inside]) / width[inside]
    pit[below] = np.clip((alpha / 2) * np.maximum(obs[below], 0) / np.maximum(lo[below], 1e-6),
                          0, alpha / 2)
    pit[above] = np.clip(1 - alpha / 2 + (alpha / 2) * (obs[above] - hi[above]) / np.maximum(hi[above], 1e-6),
                          1 - alpha / 2, 1.0)
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

    def integral_F_sq(a, b, c, z0, z1):
        z1c = z1; z0c = z0
        valid = z1c > z0c
        out = np.zeros_like(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            nonzero = (b != 0) & valid
            v1 = a + b * (z1c - c); v0 = a + b * (z0c - c)
            out_nonzero = (v1 ** 3 - v0 ** 3) / (3 * b)
            out = np.where(nonzero, out_nonzero, out)
        zero = (b == 0) & valid
        out = np.where(zero, a ** 2 * (z1c - z0c), out)
        return out

    def integral_1mF_sq(a, b, c, z0, z1):
        return integral_F_sq(1 - a, -b, c, z0, z1)

    def integrate_below(a, b, c, z_lo, z_hi):
        upper = np.minimum(z_hi, y)
        z0 = np.minimum(z_lo, upper)
        return integral_F_sq(a, b, c, z0, upper)

    def integrate_above(a, b, c, z_lo, z_hi):
        lower = np.maximum(z_lo, y)
        z1 = np.maximum(z_hi, lower)
        return integral_1mF_sq(a, b, c, lower, z1)

    crps_below = (integrate_below(a1, b1, c1, bp0, bp1) + integrate_below(a2, b2, c2, bp1, bp2) +
                  integrate_below(a3, b3, c3, bp2, bp3) + integrate_below(a4, b4, c4, bp3, bp4))
    crps_above = (integrate_above(a1, b1, c1, bp0, bp1) + integrate_above(a2, b2, c2, bp1, bp2) +
                  integrate_above(a3, b3, c3, bp2, bp3) + integrate_above(a4, b4, c4, bp3, bp4))
    return crps_below + crps_above


def reliability_dev(pit_vals, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    nominal = bin_centers
    emp = np.array([np.mean(pit_vals <= bin_edges[i + 1]) for i in range(n_bins)])
    return float(np.mean(np.abs(emp - nominal))), nominal, emp


def winkler_score_per_point(obs, lo, hi, alpha):
    width = hi - lo
    below = obs < lo
    above = obs > hi
    pen = np.zeros_like(obs, dtype=float)
    pen[below] = (2.0 / alpha) * (lo[below] - obs[below])
    pen[above] = (2.0 / alpha) * (obs[above] - hi[above])
    return width + pen


def main():
    log_print("=== exp_global_cqr: Vanilla Global CQR on CAMELS-US gauged ===\n")
    attrs = load_attrs()

    val_p = find_latest_epoch(EXP002_RUN, "validation") / "validation_results.p"
    test_p = find_latest_epoch(EXP002_RUN, "test") / "test_results.p"
    log_print(f"loading val={val_p.name} test={test_p.name}")
    val = pickle.load(open(val_p, "rb"))
    test = pickle.load(open(test_p, "rb"))

    per_basin = []
    for b in val.keys():
        if b not in test: continue
        try:
            cal_obs, cal_pred = extract(val, b)
            tst_obs, tst_pred = extract(test, b)
        except Exception:
            continue
        if len(cal_obs) < 100 or len(tst_obs) < 100:
            continue
        gid = str(b).zfill(8)
        a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0, "p_mean": 3.0})
        per_basin.append({
            "basin": gid,
            "tier": assign_tier(a["aridity"], a["frac_snow"]),
            "aridity": a["aridity"],
            "frac_snow": a["frac_snow"],
            "p_mean": a["p_mean"],
            "cal_obs": cal_obs,
            "cal_pred": cal_pred,
            "tst_obs": tst_obs,
            "tst_pred": tst_pred,
        })
    log_print(f"basins: {len(per_basin)}")

    log_print("\nTraining GLOBAL residual-CQR...")
    all_X_cal = []
    all_r_cal = []
    for r in per_basin:
        Xc = make_features(r["cal_pred"], r["aridity"], r["frac_snow"], r["p_mean"])
        rc = r["cal_obs"] - r["cal_pred"]
        all_X_cal.append(Xc)
        all_r_cal.append(rc)
        
    Xc_full = np.vstack(all_X_cal)
    rc_full = np.concatenate(all_r_cal)
    n_full = len(rc_full)
    
    if n_full > 200_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_full, 200_000, replace=False)
        Xc, rc = Xc_full[idx], rc_full[idx]
    else:
        Xc, rc = Xc_full, rc_full
        
    qr_lo_global = QuantileRegressor(quantile=ALPHA / 2, alpha=1e-3, solver="highs").fit(Xc, rc)
    qr_hi_global = QuantileRegressor(quantile=1 - ALPHA / 2, alpha=1e-3, solver="highs").fit(Xc, rc)
    
    # Predict on full cal set to get conformity scores
    ql = qr_lo_global.predict(Xc_full)
    qh = qr_hi_global.predict(Xc_full)
    E = np.maximum(ql - rc_full, rc_full - qh)
    q_hat_global = split_cp_quantile(E, ALPHA)
    log_print(f"  Global CQR Q̂ = {q_hat_global:.6f} (n_cal_full={n_full}, fit_n={len(rc)})")

    log_print("\nComputing PIT / CRPS / Winkler / coverage per basin (vectorized)...")
    pb_records = []
    pit_tier = {t: [] for t in TIERS}
    for i, r in enumerate(per_basin):
        if i % 100 == 0:
            log_print(f"  basin {i}/{len(per_basin)}")
        t = r["tier"]
        obs, pred = r["tst_obs"], r["tst_pred"]
        n = len(obs)
        Xt = make_features(pred, r["aridity"], r["frac_snow"], r["p_mean"])
        ql = qr_lo_global.predict(Xt)
        qh = qr_hi_global.predict(Xt)
        
        lo_c = np.maximum(pred + ql - q_hat_global, 0.0)
        hi_c = pred + qh + q_hat_global

        cov = float(np.mean((obs >= lo_c) & (obs <= hi_c)))
        width = float(np.mean(hi_c - lo_c))
        win = float(np.mean(winkler_score_per_point(obs, lo_c, hi_c, ALPHA)))
        pit = pit_value(obs, lo_c, hi_c)
        pit_tier[t].append(pit)
        crps = float(np.mean(crps_pwl_vectorized(obs, lo_c, hi_c)))
        
        pb_records.append({
            "basin": r["basin"], "tier": t, "method": "Global_CQR",
            "n_test": int(n),
            "coverage": cov, "mean_pi_width": width,
            "winkler_score": win, "crps": crps,
        })

    pb_df = pd.DataFrame(pb_records)
    pb_df.to_csv(OUT_DIR / "per_basin_metrics.csv", index=False)
    log_print(f"\nwrote: {OUT_DIR / 'per_basin_metrics.csv'}")

    # 4. Per-tier aggregates
    rows = []
    for t in TIERS:
        sub = pb_df[pb_df["tier"] == t]
        if sub.empty:
            continue
        pit_pool = np.concatenate(pit_tier[t]) if pit_tier[t] else np.array([])
        ks = stats.kstest(pit_pool, "uniform") if len(pit_pool) else None
        rd_dev, nom, emp = reliability_dev(pit_pool, 10) if len(pit_pool) else (np.nan, None, None)
        rows.append({
            "method": "Global_CQR", "tier": t,
            "n_basins": int(len(sub)),
            "mean_coverage": float(sub["coverage"].mean()),
            "std_coverage": float(sub["coverage"].std(ddof=1)),
            "mean_pi_width": float(sub["mean_pi_width"].mean()),
            "mean_winkler": float(sub["winkler_score"].mean()),
            "mean_crps": float(sub["crps"].mean()),
            "pit_ks_stat": float(ks.statistic) if ks else np.nan,
            "pit_ks_pvalue": float(ks.pvalue) if ks else np.nan,
            "reliability_dev_mean": float(rd_dev),
            "pit_n_points": int(len(pit_pool)),
        })
    tier_df = pd.DataFrame(rows)
    tier_df.to_csv(OUT_DIR / "uq_per_method_per_tier.csv", index=False)
    log_print(f"wrote: {OUT_DIR / 'uq_per_method_per_tier.csv'}")
    log_print("\n=== Per-tier summary ===")
    log_print(tier_df.to_string(index=False))

    # 5. Marginal aggregates
    all_pit = np.concatenate([np.concatenate(pit_tier[t]) for t in TIERS if pit_tier[t]])
    ks = stats.kstest(all_pit, "uniform")
    rd, _, _ = reliability_dev(all_pit, 10)
    marg_dict = {
        "method": "Global_CQR",
        "marginal_coverage": float(pb_df["coverage"].mean()),
        "marginal_width": float(pb_df["mean_pi_width"].mean()),
        "marginal_winkler": float(pb_df["winkler_score"].mean()),
        "marginal_crps": float(pb_df["crps"].mean()),
        "pit_ks_stat_marginal": float(ks.statistic),
        "pit_ks_pvalue_marginal": float(ks.pvalue),
        "reliability_dev_marginal": float(rd),
        "spread_pp": float((tier_df["mean_coverage"].max() - tier_df["mean_coverage"].min()) * 100),
    }
    marg_df = pd.DataFrame([marg_dict])
    marg_df.to_csv(OUT_DIR / "uq_marginal_per_method.csv", index=False)
    log_print("\n=== Marginal ===")
    log_print(marg_df.to_string(index=False))

    summary = {
        "experiment": "exp_global_cqr",
        "dataset": "CAMELS-US gauged 671 basins",
        "alpha": ALPHA,
        "n_basins": int(pb_df["basin"].nunique()),
        "marginal": marg_dict,
        "per_tier": tier_df.to_dict(orient="records"),
        "q_hat_global": q_hat_global,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log_print(f"wrote: {OUT_DIR / 'summary.json'}")
    log_print("\n✅ exp_global_cqr done")


if __name__ == "__main__":
    main()
