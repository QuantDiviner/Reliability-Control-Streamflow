"""
exp011 Phase 1 — Residual-CQR (Romano 2019) on CAMELS-US gauged 671 basins.

CQR algorithm (Romano 2019 Algorithm 1) — residual variant:
  1. Use exp002 LSTM as fixed point predictor f̂(x_t)
  2. Compute residuals r_t = y_t - f̂(x_t)
  3. Train two QuantileRegressors on calibration period:
       q̂_lo(x_t) = QR(quantile=α/2).fit(features_cal, r_cal)   # 5%-quantile of residual
       q̂_hi(x_t) = QR(quantile=1-α/2).fit(features_cal, r_cal) # 95%-quantile of residual
  4. CQR conformal correction:
       E_t = max(q̂_lo(x_t) - r_t, r_t - q̂_hi(x_t))   on cal
       Q̂ = ceil((n+1)(1-α))/n quantile of {E_t}
  5. Test interval: [f̂(x_t) + q̂_lo(x_t) - Q̂, f̂(x_t) + q̂_hi(x_t) + Q̂]

Features for the quantile regressors (per-timestep, per-basin):
  - f̂(x_t)              — LSTM point prediction
  - log(f̂(x_t) + ε)     — log-flow proxy (residual heteroscedasticity)
  - aridity              — static
  - frac_snow            — static
  - p_mean               — static (climatology proxy)

Quantile regressor: sklearn LinearQuantileRegressor (regularization α_reg=1e-3 to handle
high-D feature × large-n residual training). Per-tier model (4 tiers × 2 quantiles = 8 models)
to capture per-regime heteroscedasticity.

α = 0.10 (target 90% PI coverage). Tier scheme = exp002 PROJECT_CHARTER §2.

Output:
  experiments/exp011/results/_cqr_baseline/
    cqr_per_basin.csv             # per-basin coverage / width / winkler
    cqr_per_tier.csv              # per-tier mean ± std
    cqr_metrics.json              # full summary
    comparison_matrix.csv         # CQR vs HSCC vs Global CP vs HopCPT (574-subset)
    figure_cqr_vs_others.png      # 4-panel head-to-head

Compute: ~30-60 min CPU (4 tiers × 2 QR × ~600k cal points).
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor

ROOT = Path(__file__).resolve().parents[3]
ALPHA = 0.10
EPS = 0.01
TIERS = ["dry", "semi_arid", "humid", "snow"]

EXP002_RUN = ROOT / "experiments/exp002/results/exp002_camels_us_temporal_2504_155058"
ATTR_FILE = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
OUT_DIR = ROOT / "experiments/exp011/results/_cqr_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
            attrs[cols[0].zfill(8)] = {
                "aridity": float(cols[ai_i]),
                "frac_snow": float(cols[sn_i]),
                "p_mean": float(cols[pm_i]),
            }
    return attrs


def assign_tier(ai, fs):
    if fs >= 0.40:
        return "snow"
    if ai > 1.5:
        return "dry"
    if ai > 1.0:
        return "semi_arid"
    return "humid"


def find_latest_epoch(run_dir, kind):
    sub = Path(run_dir) / kind
    return sorted(sub.glob("model_epoch*"))[-1]


def extract(results, b):
    bd = results[b]
    ds = bd["1D"]["xr"]
    obs = ds["QObs(mm/d)_obs"].values.flatten()
    pred = ds["QObs(mm/d)_sim"].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def winkler_score(obs, lo, hi, alpha):
    width = hi - lo
    below = obs < lo
    above = obs > hi
    pen = np.zeros_like(obs)
    pen[below] = (2.0 / alpha) * (lo[below] - obs[below])
    pen[above] = (2.0 / alpha) * (obs[above] - hi[above])
    return float(np.mean(width + pen))


def make_features(pred, ai, fs, pm):
    """Per-basin feature matrix: [pred, log(pred+ε), aridity, frac_snow, p_mean].
    Static features broadcast across timesteps.
    """
    n = len(pred)
    return np.column_stack([
        pred,
        np.log(np.maximum(pred + EPS, 1e-6)),
        np.full(n, ai),
        np.full(n, fs),
        np.full(n, pm),
    ])


def cqr_per_tier(records_by_tier, tier_features_cal, tier_resid_cal,
                 tier_features_test, tier_resid_test, tier_pred_test):
    """Train per-tier CQR and apply conformal correction.

    Returns: per_tier_dict {tier: {q_lo, q_hi, Q_hat, q_lo_pred_test, q_hi_pred_test}}
    """
    qr_lo, qr_hi, q_hat_per_tier = {}, {}, {}
    for t in TIERS:
        if t not in tier_features_cal:
            continue
        Xc, rc = tier_features_cal[t], tier_resid_cal[t]
        n_cal = len(rc)
        # sklearn QuantileRegressor scales poorly above ~1M points; subsample if needed
        if n_cal > 200_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_cal, 200_000, replace=False)
            Xc, rc = Xc[idx], rc[idx]
        print(f"  [tier {t}] n_cal={len(rc)} (subsampled from {n_cal} if >200k)")
        qr_lo[t] = QuantileRegressor(quantile=ALPHA / 2, alpha=1e-3, solver="highs").fit(Xc, rc)
        qr_hi[t] = QuantileRegressor(quantile=1 - ALPHA / 2, alpha=1e-3, solver="highs").fit(Xc, rc)
        # CQR conformity scores E on FULL cal (not subsampled, for accurate Q̂)
        Xc_full = tier_features_cal[t]
        rc_full = tier_resid_cal[t]
        ql = qr_lo[t].predict(Xc_full)
        qh = qr_hi[t].predict(Xc_full)
        E = np.maximum(ql - rc_full, rc_full - qh)
        n = len(E)
        level = min(np.ceil((1 - ALPHA) * (n + 1)) / n, 1.0)
        q_hat_per_tier[t] = float(np.quantile(E, level))
        print(f"  [tier {t}] q̂_lo coef shape {qr_lo[t].coef_.shape}; Q̂={q_hat_per_tier[t]:.4f}")
    return qr_lo, qr_hi, q_hat_per_tier


def main():
    print("=== exp011 Phase 1: residual-CQR on CAMELS-US gauged ===\n")
    attrs = load_attrs()

    # Load exp002 NH val/test
    val_p = find_latest_epoch(EXP002_RUN, "validation") / "validation_results.p"
    test_p = find_latest_epoch(EXP002_RUN, "test") / "test_results.p"
    print(f"loading {val_p.name} + {test_p.name}")
    val = pickle.load(open(val_p, "rb"))
    test = pickle.load(open(test_p, "rb"))

    # Per-basin features + residuals
    per_basin = []
    skipped = []
    for b in val.keys():
        if b not in test:
            continue
        try:
            cal_obs, cal_pred = extract(val, b)
            tst_obs, tst_pred = extract(test, b)
        except Exception as e:
            skipped.append((b, str(e)))
            continue
        if len(cal_obs) < 100 or len(tst_obs) < 100:
            skipped.append((b, "too few"))
            continue
        gid = str(b).zfill(8)
        a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0, "p_mean": 3.0})
        per_basin.append({
            "basin": gid,
            "tier": assign_tier(a["aridity"], a["frac_snow"]),
            "aridity": a["aridity"],
            "frac_snow": a["frac_snow"],
            "p_mean": a["p_mean"],
            "cal_features": make_features(cal_pred, a["aridity"], a["frac_snow"], a["p_mean"]),
            "cal_resid": cal_obs - cal_pred,
            "tst_features": make_features(tst_pred, a["aridity"], a["frac_snow"], a["p_mean"]),
            "tst_obs": tst_obs,
            "tst_pred": tst_pred,
        })
    print(f"basins analyzed: {len(per_basin)} (skipped {len(skipped)})")

    # Pool per tier for QR training
    tier_features_cal, tier_resid_cal = {}, {}
    tier_features_test, tier_resid_test, tier_pred_test = {}, {}, {}
    for r in per_basin:
        t = r["tier"]
        tier_features_cal.setdefault(t, []).append(r["cal_features"])
        tier_resid_cal.setdefault(t, []).append(r["cal_resid"])
        tier_features_test.setdefault(t, []).append(r["tst_features"])
        tier_resid_test.setdefault(t, []).append(r["tst_obs"] - r["tst_pred"])
        tier_pred_test.setdefault(t, []).append(r["tst_pred"])
    for t in TIERS:
        if t in tier_features_cal:
            tier_features_cal[t] = np.vstack(tier_features_cal[t])
            tier_resid_cal[t] = np.concatenate(tier_resid_cal[t])

    print(f"\nTier sizes (cal):")
    for t in TIERS:
        if t in tier_features_cal:
            print(f"  {t:10s}: {len(tier_resid_cal[t]):>8d} cal points")

    # Train per-tier CQR
    print("\nTraining per-tier QuantileRegressors...")
    qr_lo, qr_hi, q_hat_per_tier = cqr_per_tier(
        per_basin, tier_features_cal, tier_resid_cal,
        tier_features_test, tier_resid_test, tier_pred_test
    )

    # Apply CQR to test
    print("\nApplying CQR to test period...")
    pb_rows = []
    for r in per_basin:
        t = r["tier"]
        if t not in qr_lo:
            continue
        Xt = r["tst_features"]
        ql = qr_lo[t].predict(Xt)
        qh = qr_hi[t].predict(Xt)
        Q_hat = q_hat_per_tier[t]
        pred = r["tst_pred"]
        obs = r["tst_obs"]
        # CQR test interval (in residual space) → translate to flow space
        lo = np.maximum(pred + ql - Q_hat, 0.0)
        hi = pred + qh + Q_hat
        cov = float(np.mean((obs >= lo) & (obs <= hi)))
        width = float(np.mean(hi - lo))
        win = winkler_score(obs, lo, hi, ALPHA)
        pb_rows.append({
            "basin": r["basin"],
            "tier": t,
            "aridity": r["aridity"],
            "frac_snow": r["frac_snow"],
            "n_test": len(obs),
            "coverage": cov,
            "coverage_eps": cov - (1 - ALPHA),
            "mean_pi_width": width,
            "winkler_score": win,
            "Q_hat_tier": Q_hat,
        })
    pb_df = pd.DataFrame(pb_rows)
    pb_csv = OUT_DIR / "cqr_per_basin.csv"
    pb_df.to_csv(pb_csv, index=False)
    print(f"wrote: {pb_csv}")

    # Per-tier aggregate
    tier_rows = []
    for t in TIERS:
        sub = pb_df[pb_df["tier"] == t]
        if sub.empty:
            continue
        tier_rows.append({
            "tier": t,
            "n_basins": int(len(sub)),
            "mean_coverage": float(sub["coverage"].mean()),
            "median_coverage": float(sub["coverage"].median()),
            "std_coverage": float(sub["coverage"].std(ddof=1)),
            "mean_pi_width": float(sub["mean_pi_width"].mean()),
            "median_pi_width": float(sub["mean_pi_width"].median()),
            "mean_winkler_score": float(sub["winkler_score"].mean()),
            "Q_hat_tier": float(sub["Q_hat_tier"].iloc[0]),
        })
    tier_df = pd.DataFrame(tier_rows)
    tier_csv = OUT_DIR / "cqr_per_tier.csv"
    tier_df.to_csv(tier_csv, index=False)
    print(f"wrote: {tier_csv}")

    spread_pp = (tier_df["mean_coverage"].max() - tier_df["mean_coverage"].min()) * 100
    overall_cov = float(pb_df["coverage"].mean())
    overall_width = float(pb_df["mean_pi_width"].mean())
    overall_winkler = float(pb_df["winkler_score"].mean())

    summary = {
        "experiment": "exp011_phase_1",
        "method": "residual-CQR (Romano 2019, sklearn QuantileRegressor on exp002 LSTM residuals)",
        "alpha": ALPHA,
        "n_basins_analyzed": int(len(pb_df)),
        "per_tier": tier_df.to_dict(orient="records"),
        "overall": {
            "mean_coverage": overall_cov,
            "mean_pi_width": overall_width,
            "mean_winkler_score": overall_winkler,
            "tier_coverage_spread_pp": float(spread_pp),
        },
        "Q_hat_per_tier": q_hat_per_tier,
        "S1_marginal_in_band": bool(0.88 <= overall_cov <= 0.92),
        "S2_per_tier_reported": True,
    }
    out_json = OUT_DIR / "cqr_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote: {out_json}")

    print("\n=== HEADLINE ===")
    print(f"  marginal coverage: {overall_cov:.4f}  (target 0.90)")
    print(f"  spread:           {spread_pp:.2f}pp")
    print(f"  per-tier:")
    for r in tier_rows:
        print(f"    {r['tier']:10s}: cov={r['mean_coverage']:.4f}  width={r['mean_pi_width']:.3f}  "
              f"winkler={r['mean_winkler_score']:.3f}  n={r['n_basins']}  Q̂={r['Q_hat_tier']:.4f}")


if __name__ == "__main__":
    main()
