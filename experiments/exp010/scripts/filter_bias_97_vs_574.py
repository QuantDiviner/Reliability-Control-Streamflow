"""
exp010 Action 7 (R2 P1-3): 97-basin filter bias analysis.

Compare attribute distributions of dropped (97) vs kept (574) basins:
  - frac_snow, aridity, p_mean (precip), pet_mean, area, gauge_lon, gauge_lat
  - log_residual_variance per basin (from exp002 LSTM val period preds)

KS test for each attribute. Output:
  experiments/exp010/results/_analysis/filter_bias_97vs574.csv
  experiments/exp010/results/_analysis/filter_bias_97vs574.json
  experiments/exp010/results/_analysis/filter_bias_97vs574.png
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
EPS = 0.01

LIST_671 = ROOT / "data/raw/exp002_basin_list_671.txt"
LIST_574 = ROOT / "data/raw/exp010_basin_list_clean.txt"
EXP002_RUN = ROOT / "experiments/exp002/results/exp002_camels_us_temporal_2504_155058"
ATTR_CLIM = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
ATTR_TOPO = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_topo.txt"
OUT_DIR = ROOT / "experiments/exp010/results/_analysis"


def load_attr_file(path, cols):
    df = pd.read_csv(path, sep=";", dtype={"gauge_id": str})
    df["gauge_id"] = df["gauge_id"].str.zfill(8)
    return df[["gauge_id"] + cols]


def main():
    print("=== exp010 Action 7: 97-basin filter bias analysis ===\n")
    b671 = [b.strip().zfill(8) for b in open(LIST_671) if b.strip()]
    b574 = [b.strip().zfill(8) for b in open(LIST_574) if b.strip()]
    b97 = sorted(set(b671) - set(b574))
    print(f"671 list: {len(b671)} basins")
    print(f"574 list (clean): {len(b574)} basins")
    print(f"dropped (97): {len(b97)} basins")
    if len(b97) != 97:
        print(f"WARNING: dropped count = {len(b97)}, expected 97")

    # Load attributes
    clim = load_attr_file(ATTR_CLIM, ["aridity", "frac_snow", "p_mean", "pet_mean"])
    topo = load_attr_file(ATTR_TOPO, ["area_gages2", "gauge_lon", "gauge_lat", "elev_mean"])
    attr = clim.merge(topo, on="gauge_id", how="inner")
    attr["kept"] = attr["gauge_id"].isin(b574).astype(int)
    attr["dropped"] = attr["gauge_id"].isin(b97).astype(int)

    # Compute log-residual-variance from exp002 validation period
    val_p = sorted((EXP002_RUN / "validation").glob("model_epoch*"))[-1] / "validation_results.p"
    print(f"loading {val_p}...")
    val = pickle.load(open(val_p, "rb"))
    log_resid_var = {}
    for b, bd in val.items():
        try:
            ds = bd["1D"]["xr"] if isinstance(bd, dict) and "1D" in bd else bd.xr
            obs = ds["QObs(mm/d)_obs"].values.flatten()
            pred = ds["QObs(mm/d)_sim"].values.flatten()
            valid = ~(np.isnan(obs) | np.isnan(pred))
            obs, pred = obs[valid], pred[valid]
            if len(obs) < 100:
                continue
            log_score = np.abs(np.log(np.maximum(obs + EPS, 1e-6)) -
                               np.log(np.maximum(pred + EPS, 1e-6)))
            log_resid_var[str(b).zfill(8)] = float(np.var(log_score))
        except Exception as e:
            log_resid_var[str(b).zfill(8)] = np.nan
    lrv_df = pd.DataFrame(
        [{"gauge_id": k, "log_resid_var": v} for k, v in log_resid_var.items()]
    )
    attr = attr.merge(lrv_df, on="gauge_id", how="left")

    # KS tests per attribute, kept (n=574) vs dropped (n=97)
    test_cols = ["aridity", "frac_snow", "p_mean", "pet_mean",
                 "area_gages2", "gauge_lon", "gauge_lat", "elev_mean", "log_resid_var"]
    rows = []
    print(f"\n{'Attribute':18s} {'kept_mean':>10s} {'dropped_mean':>13s} {'kept_med':>10s} "
          f"{'dropped_med':>12s} {'KS_stat':>8s} {'KS_p':>9s}")
    for c in test_cols:
        kept = attr.loc[attr["kept"] == 1, c].dropna().values
        dropped = attr.loc[attr["dropped"] == 1, c].dropna().values
        if len(dropped) == 0 or len(kept) == 0:
            continue
        ks = stats.ks_2samp(kept, dropped)
        rows.append({
            "attribute": c,
            "n_kept": int(len(kept)),
            "n_dropped": int(len(dropped)),
            "kept_mean": float(np.mean(kept)),
            "dropped_mean": float(np.mean(dropped)),
            "kept_median": float(np.median(kept)),
            "dropped_median": float(np.median(dropped)),
            "kept_std": float(np.std(kept)),
            "dropped_std": float(np.std(dropped)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "biased": bool(ks.pvalue < 0.05),
        })
        print(f"{c:18s} {np.mean(kept):>10.3f} {np.mean(dropped):>13.3f} "
              f"{np.median(kept):>10.3f} {np.median(dropped):>12.3f} "
              f"{ks.statistic:>8.3f} {ks.pvalue:>9.2e}{'  *' if ks.pvalue < 0.05 else ''}")

    # Tier composition of dropped vs kept
    def assign_tier(ai, fs):
        if fs >= 0.40:
            return "snow"
        if ai > 1.5:
            return "dry"
        if ai > 1.0:
            return "semi_arid"
        return "humid"
    attr["tier"] = [assign_tier(a, f) for a, f in zip(attr["aridity"], attr["frac_snow"])]
    print(f"\n{'Tier':10s} {'kept_n':>8s} {'kept_%':>8s} {'dropped_n':>10s} {'dropped_%':>10s}")
    tier_rows = []
    for t in ["dry", "semi_arid", "humid", "snow"]:
        nk = int(((attr["kept"] == 1) & (attr["tier"] == t)).sum())
        nd = int(((attr["dropped"] == 1) & (attr["tier"] == t)).sum())
        pct_k = nk / len(b574) * 100
        pct_d = nd / len(b97) * 100 if len(b97) > 0 else 0
        tier_rows.append({"tier": t, "kept_n": nk, "dropped_n": nd,
                          "kept_pct": pct_k, "dropped_pct": pct_d,
                          "diff_pp": pct_d - pct_k})
        print(f"{t:10s} {nk:>8d} {pct_k:>7.1f}% {nd:>10d} {pct_d:>9.1f}%")

    # Output table
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "filter_bias_97vs574.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote: {out_csv}")

    out_json = OUT_DIR / "filter_bias_97vs574.json"
    with open(out_json, "w") as f:
        json.dump({
            "n_dropped": len(b97),
            "n_kept": len(b574),
            "drop_reason": "HopCPT loader/hydrology.py:88-98 strict NaN check on 1980-2014 full period; NH NaN-safe loss tolerated. ELOG-006.",
            "attribute_ks_tests": rows,
            "tier_composition": tier_rows,
        }, f, indent=2)
    print(f"wrote: {out_json}")

    # Figure: 9-panel attribute distribution comparison
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, c in zip(axes.flat, test_cols):
        kept = attr.loc[attr["kept"] == 1, c].dropna().values
        dropped = attr.loc[attr["dropped"] == 1, c].dropna().values
        bins = np.histogram_bin_edges(np.concatenate([kept, dropped]), bins=30)
        ax.hist(kept, bins=bins, alpha=0.6, label=f"kept (n={len(kept)})",
                color="#1f77b4", density=True)
        ax.hist(dropped, bins=bins, alpha=0.6, label=f"dropped (n={len(dropped)})",
                color="#d62728", density=True)
        ks = stats.ks_2samp(kept, dropped)
        sig = "*" if ks.pvalue < 0.05 else ""
        ax.set_title(f"{c}\nKS={ks.statistic:.3f} p={ks.pvalue:.2e}{sig}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Filter bias: kept (574) vs dropped (97) attribute distributions",
                 y=0.995)
    fig.tight_layout()
    out_png = OUT_DIR / "filter_bias_97vs574.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote: {out_png}")


if __name__ == "__main__":
    main()
