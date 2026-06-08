"""
exp011 Phase 2 — Standard UQ metrics scan on CAMELS-US gauged 671 basins.

VECTORIZED VERSION (v2, 2026-05-05): CRPS computed via closed-form integration
of piecewise-linear F̂; 10× speedup over the v1 Python-loop version.

Computes PIT, CRPS, reliability deviation, Winkler, coverage, width for 3 methods:
  - HSCC (per-tier conformal log-flow score quantile)
  - Global split CP (single conformal log-flow quantile)
  - residual-CQR (per-tier QuantileRegressor on LSTM residuals)

Predictive CDF F̂(y|x) for interval methods uses piecewise-linear ramp:
  F(z) = (α/2)·z/lo                    for 0 ≤ z < lo
  F(z) = α/2 + (1-α)·(z-lo)/(hi-lo)    for lo ≤ z ≤ hi
  F(z) = (1-α/2) + (α/2)·(z-hi)/hi     for hi < z < 2·hi
  F(z) = 1                             for z ≥ 2·hi

CRPS closed form for piecewise-linear F (Hersbach 2000-style):
  CRPS = ∫₀^∞ (F(z) - 𝟙{z≥y})² dz
       = ∫₀^y F(z)² dz + ∫_y^∞ (1-F(z))² dz
The piecewise-linear integral is computed analytically per region — see _crps_pwl.

Outputs:
  experiments/exp011/results/_uq_metrics/camels_us/
    per_basin_metrics.csv
    uq_per_method_per_tier.csv
    uq_marginal_per_method.csv
    summary.json
    figures/{pit_histograms,reliability_diagrams,metrics_comparison}.png
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import QuantileRegressor

# CRITICAL: force unbuffered stdout for piped runs
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)

ROOT = Path(__file__).resolve().parents[3]
ALPHA = 0.10
EPS = 0.01
TIERS = ["dry", "semi_arid", "humid", "snow"]
TIER_PRETTY = {"dry": "Dry", "semi_arid": "Semi-arid", "humid": "Humid", "snow": "Snow"}

EXP002_RUN = ROOT / "experiments/exp002/results/exp002_camels_us_temporal_2504_155058"
ATTR_FILE = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
OUT_DIR = ROOT / "experiments/exp011/results/_uq_metrics/camels_us"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


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


def log_score(obs, pred):
    return np.abs(np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6)))


def interval_log(pred, q):
    log_p = np.log(np.maximum(pred + EPS, 1e-6))
    lo = np.maximum(np.exp(log_p - q) - EPS, 0.0)
    hi = np.exp(log_p + q) - EPS
    return lo, hi


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
    """Vectorized PIT for piecewise-linear interval CDF.

    F(y < lo)   = (α/2) * y / lo                         (clipped to [0, α/2])
    F(lo≤y≤hi) = α/2 + (1-α) * (y-lo)/(hi-lo)
    F(y > hi)  = (1-α/2) + (α/2) * (y-hi)/hi             (clipped to [1-α/2, 1])
    """
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
    """Closed-form CRPS for piecewise-linear interval CDF, fully vectorized.

    F is defined as:
      Region 1: [0, lo)        — F(z) = (α/2) z/lo            (slope a = α/(2 lo))
      Region 2: [lo, hi]       — F(z) = α/2 + (1-α)(z-lo)/(hi-lo)
      Region 3: (hi, 2 hi]     — F(z) = (1-α/2) + (α/2)(z-hi)/hi
      Region 4: (2 hi, ∞)      — F(z) = 1

    CRPS = ∫₀^y F²dz + ∫_y^∞ (1-F)² dz.

    For a linear segment F(z) = a + b(z - c) on [z0, z1] (with b > 0), we have:
      ∫ F² dz = ((a + b(z-c))³ / (3b))           if b ≠ 0
      ∫ (1-F)² dz = (-((1-a) - b(z-c))³ / (3b))  if b ≠ 0

    We split the [0, ∞) integration domain at y and at the breakpoints
    {0, lo, hi, 2hi}. For each segment we compute closed-form ∫F² (z<y) +
    ∫(1-F)² (z>y). All operations broadcast across the input arrays.
    """
    obs = np.asarray(obs, dtype=float)
    lo = np.maximum(np.asarray(lo, dtype=float), 1e-9)  # avoid div0
    hi = np.maximum(np.asarray(hi, dtype=float), lo + 1e-9)
    y = obs

    # Define segment boundaries
    bp0 = np.zeros_like(y)
    bp1 = lo
    bp2 = hi
    bp3 = 2 * hi
    bp4 = np.maximum(np.maximum(y, bp3) * 1.5, bp3 + 1.0)  # upper integration bound

    # Segment slopes/intercepts: F(z) = a_i + b_i * (z - c_i)
    # Seg 1: a=0, b=α/(2lo), c=0     (F = (α/2)z/lo)
    # Seg 2: a=α/2, b=(1-α)/(hi-lo), c=lo
    # Seg 3: a=1-α/2, b=(α/2)/hi, c=hi
    # Seg 4: a=1, b=0, c=2hi
    a1 = np.zeros_like(lo); b1 = (alpha / 2) / lo; c1 = np.zeros_like(lo)
    a2 = np.full_like(lo, alpha / 2); b2 = (1 - alpha) / (hi - lo); c2 = lo
    a3 = np.full_like(lo, 1 - alpha / 2); b3 = (alpha / 2) / hi; c3 = hi
    a4 = np.full_like(lo, 1.0); b4 = np.zeros_like(lo); c4 = 2 * hi

    # Helper: ∫_{z0}^{z1} F² dz where F(z) = a + b(z - c)
    # = (a + b(z1-c))³ / (3b) - (a + b(z0-c))³ / (3b)   if b != 0
    # = a² (z1 - z0)                                      if b == 0
    def integral_F_sq(a, b, c, z0, z1):
        z1c = z1
        z0c = z0
        # Mask z0 > z1 → 0
        valid = z1c > z0c
        out = np.zeros_like(a)
        # b != 0 case
        with np.errstate(divide="ignore", invalid="ignore"):
            nonzero = (b != 0) & valid
            v1 = a + b * (z1c - c)
            v0 = a + b * (z0c - c)
            out_nonzero = (v1 ** 3 - v0 ** 3) / (3 * b)
            out = np.where(nonzero, out_nonzero, out)
        zero = (b == 0) & valid
        out = np.where(zero, a ** 2 * (z1c - z0c), out)
        return out

    def integral_1mF_sq(a, b, c, z0, z1):
        # 1 - F(z) = (1 - a) - b(z - c) = a' + b'(z - c) with a' = 1 - a, b' = -b
        return integral_F_sq(1 - a, -b, c, z0, z1)

    # Compute ∫₀^y F² dz (for region overlapping z<y) per segment
    # For each segment [bp_i, bp_{i+1}], integrate F² over [bp_i, min(bp_{i+1}, y)]
    def integrate_below(a, b, c, z_lo, z_hi):
        # ∫_{z_lo}^{min(z_hi, y)} F²
        upper = np.minimum(z_hi, y)
        z0 = np.minimum(z_lo, upper)  # ensure z0 <= upper
        return integral_F_sq(a, b, c, z0, upper)

    def integrate_above(a, b, c, z_lo, z_hi):
        # ∫_{max(z_lo, y)}^{z_hi} (1-F)²
        lower = np.maximum(z_lo, y)
        z1 = np.maximum(z_hi, lower)
        return integral_1mF_sq(a, b, c, lower, z1)

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
    log_print("=== exp011 Phase 2 (v2 vectorized): UQ metrics on CAMELS-US gauged ===\n")
    attrs = load_attrs()

    val_p = find_latest_epoch(EXP002_RUN, "validation") / "validation_results.p"
    test_p = find_latest_epoch(EXP002_RUN, "test") / "test_results.p"
    log_print(f"loading val={val_p.name} test={test_p.name}")
    val = pickle.load(open(val_p, "rb"))
    test = pickle.load(open(test_p, "rb"))

    per_basin = []
    for b in val.keys():
        if b not in test:
            continue
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

    # 1. Compute Global CP + HSCC quantiles
    all_cal = np.concatenate([log_score(r["cal_obs"], r["cal_pred"]) for r in per_basin])
    q_global = split_cp_quantile(all_cal, ALPHA)
    q_tier = {}
    for t in TIERS:
        s = [log_score(r["cal_obs"], r["cal_pred"]) for r in per_basin if r["tier"] == t]
        s = np.concatenate(s) if s else np.array([])
        q_tier[t] = split_cp_quantile(s, ALPHA) if len(s) else np.nan
    log_print(f"q_global (log-flow): {q_global:.4f}")
    for t in TIERS:
        log_print(f"q_HSCC[{t}]: {q_tier[t]:.4f}")

    # 2. Train residual-CQR per tier
    log_print("\nTraining residual-CQR per tier...")
    tier_X_cal, tier_r_cal = {}, {}
    for r in per_basin:
        t = r["tier"]
        Xc = make_features(r["cal_pred"], r["aridity"], r["frac_snow"], r["p_mean"])
        rc = r["cal_obs"] - r["cal_pred"]
        tier_X_cal.setdefault(t, []).append(Xc)
        tier_r_cal.setdefault(t, []).append(rc)
    qr_lo, qr_hi, q_hat_cqr = {}, {}, {}
    for t in TIERS:
        if t not in tier_X_cal:
            continue
        Xc_full = np.vstack(tier_X_cal[t])
        rc_full = np.concatenate(tier_r_cal[t])
        n_full = len(rc_full)
        if n_full > 200_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_full, 200_000, replace=False)
            Xc, rc = Xc_full[idx], rc_full[idx]
        else:
            Xc, rc = Xc_full, rc_full
        qr_lo[t] = QuantileRegressor(quantile=ALPHA / 2, alpha=1e-3, solver="highs").fit(Xc, rc)
        qr_hi[t] = QuantileRegressor(quantile=1 - ALPHA / 2, alpha=1e-3, solver="highs").fit(Xc, rc)
        ql = qr_lo[t].predict(Xc_full)
        qh = qr_hi[t].predict(Xc_full)
        E = np.maximum(ql - rc_full, rc_full - qh)
        q_hat_cqr[t] = split_cp_quantile(E, ALPHA)
        log_print(f"  CQR Q̂[{t}] = {q_hat_cqr[t]:.6f} (n_cal_full={n_full}, fit_n={len(rc)})")

    # 3. Per-basin: vectorized intervals + metrics for each method
    log_print("\nComputing PIT / CRPS / Winkler / coverage per basin per method (vectorized)...")
    pb_records = []
    method_pit_tier = {m: {t: [] for t in TIERS} for m in ["Global_CP", "HSCC", "CQR"]}
    for i, r in enumerate(per_basin):
        if i % 100 == 0:
            log_print(f"  basin {i}/{len(per_basin)}")
        t = r["tier"]
        obs, pred = r["tst_obs"], r["tst_pred"]
        n = len(obs)
        # Intervals
        lo_g, hi_g = interval_log(pred, q_global)
        lo_h, hi_h = interval_log(pred, q_tier[t])
        Xt = make_features(pred, r["aridity"], r["frac_snow"], r["p_mean"])
        ql = qr_lo[t].predict(Xt); qh = qr_hi[t].predict(Xt)
        Q_hat = q_hat_cqr[t]
        lo_c = np.maximum(pred + ql - Q_hat, 0.0)
        hi_c = pred + qh + Q_hat

        for method, lo, hi in [("Global_CP", lo_g, hi_g), ("HSCC", lo_h, hi_h), ("CQR", lo_c, hi_c)]:
            cov = float(np.mean((obs >= lo) & (obs <= hi)))
            width = float(np.mean(hi - lo))
            win = float(np.mean(winkler_score_per_point(obs, lo, hi, ALPHA)))
            pit = pit_value(obs, lo, hi)
            method_pit_tier[method][t].append(pit)
            crps = float(np.mean(crps_pwl_vectorized(obs, lo, hi)))
            pb_records.append({
                "basin": r["basin"], "tier": t, "method": method,
                "n_test": int(n),
                "coverage": cov, "mean_pi_width": width,
                "winkler_score": win, "crps": crps,
            })

    pb_df = pd.DataFrame(pb_records)
    pb_df.to_csv(OUT_DIR / "per_basin_metrics.csv", index=False)
    log_print(f"\nwrote: {OUT_DIR / 'per_basin_metrics.csv'}")

    # 4. Per-tier aggregates
    rows = []
    for method in ["Global_CP", "HSCC", "CQR"]:
        for t in TIERS:
            sub = pb_df[(pb_df["method"] == method) & (pb_df["tier"] == t)]
            if sub.empty:
                continue
            pit_pool = np.concatenate(method_pit_tier[method][t]) if method_pit_tier[method][t] else np.array([])
            ks = stats.kstest(pit_pool, "uniform") if len(pit_pool) else None
            rd_dev, nom, emp = reliability_dev(pit_pool, 10) if len(pit_pool) else (np.nan, None, None)
            rows.append({
                "method": method, "tier": t,
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
    log_print("\n=== Per-tier per-method summary ===")
    log_print(tier_df.to_string(index=False))

    # 5. Marginal aggregates
    marg_rows = []
    for method in ["Global_CP", "HSCC", "CQR"]:
        sub = pb_df[pb_df["method"] == method]
        all_pit = np.concatenate([np.concatenate(method_pit_tier[method][t])
                                   for t in TIERS if method_pit_tier[method][t]])
        ks = stats.kstest(all_pit, "uniform")
        rd, _, _ = reliability_dev(all_pit, 10)
        marg_rows.append({
            "method": method,
            "marginal_coverage": float(sub["coverage"].mean()),
            "marginal_width": float(sub["mean_pi_width"].mean()),
            "marginal_winkler": float(sub["winkler_score"].mean()),
            "marginal_crps": float(sub["crps"].mean()),
            "pit_ks_stat_marginal": float(ks.statistic),
            "pit_ks_pvalue_marginal": float(ks.pvalue),
            "reliability_dev_marginal": float(rd),
            "spread_pp": float((tier_df[tier_df["method"] == method]["mean_coverage"].max() -
                                tier_df[tier_df["method"] == method]["mean_coverage"].min()) * 100),
        })
    marg_df = pd.DataFrame(marg_rows)
    marg_df.to_csv(OUT_DIR / "uq_marginal_per_method.csv", index=False)
    log_print("\n=== Marginal per-method ===")
    log_print(marg_df.to_string(index=False))

    summary = {
        "experiment": "exp011_phase_2",
        "version": "v2_vectorized",
        "dataset": "CAMELS-US gauged 671 basins",
        "alpha": ALPHA,
        "n_basins": int(pb_df["basin"].nunique()),
        "methods": ["Global_CP", "HSCC", "CQR"],
        "marginal_per_method": marg_df.to_dict(orient="records"),
        "per_method_per_tier": tier_df.to_dict(orient="records"),
        "q_global": q_global,
        "q_hscc_per_tier": q_tier,
        "q_hat_cqr_per_tier": q_hat_cqr,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log_print(f"wrote: {OUT_DIR / 'summary.json'}")

    # 6. Figures
    log_print("\nGenerating figures...")

    # PIT histograms
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    for j, method in enumerate(["Global_CP", "HSCC", "CQR"]):
        for k, t in enumerate(TIERS):
            ax = axes[j, k]
            if method_pit_tier[method][t]:
                pit_pool = np.concatenate(method_pit_tier[method][t])
                ax.hist(pit_pool, bins=20, color={"Global_CP": "#888", "HSCC": "#1f77b4", "CQR": "#d62728"}[method],
                        edgecolor="k", alpha=0.7)
                ks = stats.kstest(pit_pool, "uniform")
                ax.set_title(f"{method} — {TIER_PRETTY[t]}\nKS={ks.statistic:.3f} p={ks.pvalue:.1e}",
                             fontsize=9)
            ax.axhline(len(pit_pool) / 20, color="k", ls="--", lw=0.5)
            if k == 0: ax.set_ylabel("Count")
            if j == 2: ax.set_xlabel("PIT")
            ax.grid(alpha=0.3)
    fig.suptitle("PIT histograms × method × tier (CAMELS-US gauged 671 basins)", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pit_histograms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log_print(f"wrote: {FIG_DIR / 'pit_histograms.png'}")

    # Reliability diagrams
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for k, t in enumerate(TIERS):
        ax = axes[k]
        for method, color in [("Global_CP", "#888"), ("HSCC", "#1f77b4"), ("CQR", "#d62728")]:
            if method_pit_tier[method][t]:
                pit_pool = np.concatenate(method_pit_tier[method][t])
                _, nom, emp = reliability_dev(pit_pool, 10)
                ax.plot(nom, emp, "o-", color=color, label=method, markersize=6, lw=1.5)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.set_title(TIER_PRETTY[t])
        ax.set_xlabel("Nominal CDF")
        if k == 0: ax.set_ylabel("Empirical CDF")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Reliability diagrams per tier (CAMELS-US gauged 671 basins)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliability_diagrams.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log_print(f"wrote: {FIG_DIR / 'reliability_diagrams.png'}")

    # Metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics_to_plot = [("mean_coverage", "Coverage", 0.90, axes[0, 0]),
                       ("mean_pi_width", "PI Width (mm/d)", None, axes[0, 1]),
                       ("mean_winkler", "Winkler Score (mm/d)", None, axes[1, 0]),
                       ("mean_crps", "CRPS (mm/d)", None, axes[1, 1])]
    method_colors = {"Global_CP": "#888", "HSCC": "#1f77b4", "CQR": "#d62728"}
    x = np.arange(len(TIERS))
    bw = 0.27
    for col, label, target, ax in metrics_to_plot:
        for i, method in enumerate(["Global_CP", "HSCC", "CQR"]):
            vals = []
            for t in TIERS:
                row = tier_df[(tier_df["method"] == method) & (tier_df["tier"] == t)]
                vals.append(float(row[col].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 1) * bw, vals, bw, label=method, color=method_colors[method])
        ax.set_xticks(x); ax.set_xticklabels([TIER_PRETTY[t] for t in TIERS])
        ax.set_ylabel(label); ax.set_title(label)
        if target: ax.axhline(target, color="k", ls="--", lw=1, label="Target")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Per-tier UQ metrics: Global CP vs HSCC vs CQR (CAMELS-US 671 basins)", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log_print(f"wrote: {FIG_DIR / 'metrics_comparison.png'}")

    log_print("\n✅ Phase 2 done")


if __name__ == "__main__":
    main()
