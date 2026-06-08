"""
exp011 Phase 2 GB extension — PIT/CRPS/reliability/Winkler on CAMELS-GB.

Reuses the vectorized core from phase2_uq_metrics_camels_us.py, adapted to:
  - GB tier scheme (D-022): gb_drier_q4 / gb_mid / gb_wet / gb_montane
  - exp004 seed=42 native LSTM checkpoint (50 basins after PCR-004 rename)
  - 2 methods (HSCC + Global CP); CQR skipped on GB (see plan §3.1, similar to LORO)

Output:
  experiments/exp011/results/_uq_metrics/camels_gb/
    per_basin_metrics.csv
    uq_per_method_per_tier.csv
    uq_marginal_per_method.csv
    summary.json
    figures/{pit_histograms,reliability_diagrams}.png
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

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)

ROOT = Path(__file__).resolve().parents[3]
ALPHA = 0.10
EPS = 0.01

GB_TIERS = ["gb_drier_q4", "gb_mid", "gb_wet", "gb_montane"]
GB_TIER_PRETTY = {
    "gb_drier_q4": "GB Dry-Q4", "gb_mid": "GB Mid", "gb_wet": "GB Wet", "gb_montane": "GB Montane"
}

EXP004_SEED42_RUN = ROOT / "experiments/exp004/results/exp004_camels_gb_native_2604_214507"
GB_TIERS_CSV = ROOT / "experiments/exp004/basin_lists/gb_basin_tiers.csv"
OUT_DIR = ROOT / "experiments/exp011/results/_uq_metrics/camels_gb"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log_print(msg):
    print(msg, flush=True)


def find_latest_epoch(rd, k):
    return sorted((Path(rd) / k).glob("model_epoch*"))[-1]


def extract(results, b):
    bd = results[b]
    ds = bd["1D"]["xr"]
    # CAMELS-GB uses discharge_spec_obs/sim; CAMELS-US uses QObs(mm/d)_obs/sim
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


def log_score(obs, pred):
    return np.abs(np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6)))


def interval_log(pred, q):
    log_p = np.log(np.maximum(pred + EPS, 1e-6))
    lo = np.maximum(np.exp(log_p - q) - EPS, 0.0)
    hi = np.exp(log_p + q) - EPS
    return lo, hi


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
        valid = z1 > z0
        out = np.zeros_like(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            nonzero = (b != 0) & valid
            v1 = a + b * (z1 - c)
            v0 = a + b * (z0 - c)
            out_nonzero = (v1 ** 3 - v0 ** 3) / (3 * b)
            out = np.where(nonzero, out_nonzero, out)
        zero = (b == 0) & valid
        out = np.where(zero, a ** 2 * (z1 - z0), out)
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
    log_print("=== exp011 Phase 2 (GB extension): UQ metrics on CAMELS-GB ===\n")

    # Load GB tier assignments
    tier_df = pd.read_csv(GB_TIERS_CSV, dtype={"gauge_id": str})
    tier_df["gauge_id"] = tier_df["gauge_id"].astype(str)
    tier_map = dict(zip(tier_df["gauge_id"], tier_df["tier"]))
    log_print(f"GB tier assignments: {len(tier_map)} basins from {GB_TIERS_CSV.name}")
    log_print(f"  tier counts: {tier_df['tier'].value_counts().to_dict()}")

    val_p = find_latest_epoch(EXP004_SEED42_RUN, "validation") / "validation_results.p"
    test_p = find_latest_epoch(EXP004_SEED42_RUN, "test") / "test_results.p"
    log_print(f"loading val={val_p.name} test={test_p.name}")
    val = pickle.load(open(val_p, "rb"))
    test = pickle.load(open(test_p, "rb"))

    per_basin = []
    for b in val.keys():
        if b not in test:
            continue
        b_str = str(b).lstrip("0") if str(b).startswith("0") else str(b)
        if b_str not in tier_map:
            continue
        try:
            cal_obs, cal_pred = extract(val, b)
            tst_obs, tst_pred = extract(test, b)
        except Exception:
            continue
        if len(cal_obs) < 100 or len(tst_obs) < 100:
            continue
        per_basin.append({
            "basin": b_str,
            "tier": tier_map[b_str],
            "cal_obs": cal_obs,
            "cal_pred": cal_pred,
            "tst_obs": tst_obs,
            "tst_pred": tst_pred,
        })
    log_print(f"basins analyzed: {len(per_basin)}")
    log_print(f"  per-tier counts: {pd.Series([r['tier'] for r in per_basin]).value_counts().to_dict()}")

    # 1. Compute Global CP + HSCC quantiles (GB tier-specific)
    all_cal = np.concatenate([log_score(r["cal_obs"], r["cal_pred"]) for r in per_basin])
    q_global = split_cp_quantile(all_cal, ALPHA)
    q_tier = {}
    for t in GB_TIERS:
        s = [log_score(r["cal_obs"], r["cal_pred"]) for r in per_basin if r["tier"] == t]
        s = np.concatenate(s) if s else np.array([])
        q_tier[t] = split_cp_quantile(s, ALPHA) if len(s) else np.nan
    log_print(f"\nq_global (log-flow): {q_global:.4f}")
    for t in GB_TIERS:
        log_print(f"q_HSCC[{t}]: {q_tier[t]:.4f}")

    # 2. Compute UQ metrics per basin per method
    log_print("\nComputing PIT / CRPS / Winkler / coverage per basin per method (vectorized)...")
    pb_records = []
    method_pit_tier = {m: {t: [] for t in GB_TIERS} for m in ["Global_CP", "HSCC"]}
    for r in per_basin:
        t = r["tier"]
        obs, pred = r["tst_obs"], r["tst_pred"]
        n = len(obs)
        lo_g, hi_g = interval_log(pred, q_global)
        lo_h, hi_h = interval_log(pred, q_tier[t])
        for method, lo, hi in [("Global_CP", lo_g, hi_g), ("HSCC", lo_h, hi_h)]:
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

    # 3. Per-tier aggregates
    rows = []
    for method in ["Global_CP", "HSCC"]:
        for t in GB_TIERS:
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
                "std_coverage": float(sub["coverage"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "mean_pi_width": float(sub["mean_pi_width"].mean()),
                "mean_winkler": float(sub["winkler_score"].mean()),
                "mean_crps": float(sub["crps"].mean()),
                "pit_ks_stat": float(ks.statistic) if ks else np.nan,
                "pit_ks_pvalue": float(ks.pvalue) if ks else np.nan,
                "reliability_dev_mean": float(rd_dev),
                "pit_n_points": int(len(pit_pool)),
            })
    tier_summary = pd.DataFrame(rows)
    tier_summary.to_csv(OUT_DIR / "uq_per_method_per_tier.csv", index=False)
    log_print(f"wrote: {OUT_DIR / 'uq_per_method_per_tier.csv'}")
    log_print("\n=== Per-tier per-method summary ===")
    log_print(tier_summary.to_string(index=False))

    # 4. Marginal aggregates
    marg_rows = []
    for method in ["Global_CP", "HSCC"]:
        sub = pb_df[pb_df["method"] == method]
        all_pit = np.concatenate([np.concatenate(method_pit_tier[method][t])
                                   for t in GB_TIERS if method_pit_tier[method][t]])
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
            "spread_pp": float((tier_summary[tier_summary["method"] == method]["mean_coverage"].max() -
                                tier_summary[tier_summary["method"] == method]["mean_coverage"].min()) * 100),
        })
    marg_df = pd.DataFrame(marg_rows)
    marg_df.to_csv(OUT_DIR / "uq_marginal_per_method.csv", index=False)
    log_print("\n=== Marginal per-method ===")
    log_print(marg_df.to_string(index=False))

    summary = {
        "experiment": "exp011_phase_2_gb",
        "version": "v2_vectorized",
        "dataset": "CAMELS-GB native (50 basins, exp004 seed=42)",
        "alpha": ALPHA,
        "n_basins": int(pb_df["basin"].nunique()),
        "tier_scheme": "GB_tier_scheme (D-022; gb_drier_q4/gb_mid/gb_wet/gb_montane)",
        "methods": ["Global_CP", "HSCC"],
        "marginal_per_method": marg_df.to_dict(orient="records"),
        "per_method_per_tier": tier_summary.to_dict(orient="records"),
        "q_global": q_global,
        "q_hscc_per_tier": q_tier,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log_print(f"wrote: {OUT_DIR / 'summary.json'}")

    # 5. PIT histograms
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=False)
    for j, method in enumerate(["Global_CP", "HSCC"]):
        for k, t in enumerate(GB_TIERS):
            ax = axes[j, k]
            if method_pit_tier[method][t]:
                pit_pool = np.concatenate(method_pit_tier[method][t])
                ax.hist(pit_pool, bins=20, color={"Global_CP": "#888", "HSCC": "#1f77b4"}[method],
                        edgecolor="k", alpha=0.7)
                ks = stats.kstest(pit_pool, "uniform")
                ax.set_title(f"{method} — {GB_TIER_PRETTY[t]}\nKS={ks.statistic:.3f} p={ks.pvalue:.1e}",
                             fontsize=9)
                ax.axhline(len(pit_pool) / 20, color="k", ls="--", lw=0.5)
            if k == 0: ax.set_ylabel("Count")
            if j == 1: ax.set_xlabel("PIT")
            ax.grid(alpha=0.3)
    fig.suptitle("PIT histograms × method × GB tier (CAMELS-GB 50 basins, exp004 seed=42)", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pit_histograms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log_print(f"wrote: {FIG_DIR / 'pit_histograms.png'}")

    # Reliability diagrams
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for k, t in enumerate(GB_TIERS):
        ax = axes[k]
        for method, color in [("Global_CP", "#888"), ("HSCC", "#1f77b4")]:
            if method_pit_tier[method][t]:
                pit_pool = np.concatenate(method_pit_tier[method][t])
                _, nom, emp = reliability_dev(pit_pool, 10)
                ax.plot(nom, emp, "o-", color=color, label=method, markersize=6, lw=1.5)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.set_title(GB_TIER_PRETTY[t])
        ax.set_xlabel("Nominal CDF")
        if k == 0: ax.set_ylabel("Empirical CDF")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Reliability diagrams per GB tier (CAMELS-GB 50 basins)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliability_diagrams.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log_print(f"wrote: {FIG_DIR / 'reliability_diagrams.png'}")

    log_print("\n✅ Phase 2 GB extension done")


if __name__ == "__main__":
    main()
