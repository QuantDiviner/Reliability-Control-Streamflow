"""
exp010 Action 3 (R2 Route B): Official 574-subset comparison matrix.

Recomputes HSCC + Global CP for exp002 (single seed=42) and exp009 (5 seeds:
[42, 137, 2024, 1337, 7]) restricted to the **same 574-basin subset** that
HopCPT (exp010) ran on. exp010 metrics are already on this subset and are
loaded from tier_aggregate.csv directly.

Output: 4 tier × 3 method × {coverage, width_mm_d, winkler_mm_d}
  - experiments/exp010/results/_analysis/comparison_matrix_574subset.csv
  - experiments/exp010/results/_analysis/comparison_matrix_574subset.json
  - experiments/exp010/results/_analysis/comparison_matrix_574subset_panel.png

Reference: docs/reports/20260504_084658_D-R2-exp010_opus.md §4 Action 3
           experiments/exp002/hscc_analysis_v2.py (HSCC reference impl)
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
ALPHA = 0.10
EPS = 0.01

TIERS = ["dry", "semi_arid", "humid", "snow"]
TIER_PRETTY = {"dry": "Dry", "semi_arid": "Semi-arid", "humid": "Humid", "snow": "Snow"}

EXP002_RUN = ROOT / "experiments/exp002/results/exp002_camels_us_temporal_2504_155058"
EXP009_SEED_DIRS = {
    42:   ROOT / "experiments/exp009/results/seed042_run",
    137:  ROOT / "experiments/exp009/results/seed137_run",
    2024: ROOT / "experiments/exp009/results/seed2024_run",
    1337: ROOT / "experiments/exp009/results/seed1337_run",
    7:    ROOT / "experiments/exp009/results/seed007_run",
}
EXP010_BASIN_CSV = ROOT / "experiments/exp010/results/_analysis/per_basin_metrics.csv"
EXP010_TIER_CSV  = ROOT / "experiments/exp010/results/_analysis/tier_aggregate.csv"
EXP010_METRICS   = ROOT / "experiments/exp010/results/_analysis/metrics.json"

OUT_DIR = ROOT / "experiments/exp010/results/_analysis"
ATTR_FILE = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"

# -------- helpers --------

def load_attributes():
    attrs = {}
    with open(ATTR_FILE) as f:
        hdr = f.readline().strip().split(";")
        ai_i, sn_i = hdr.index("aridity"), hdr.index("frac_snow")
        for line in f:
            cols = line.strip().split(";")
            if not cols or not cols[0]:
                continue
            attrs[cols[0].zfill(8)] = {
                "aridity": float(cols[ai_i]),
                "frac_snow": float(cols[sn_i]),
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


def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def find_latest_epoch(run_dir, kind):
    sub = Path(run_dir) / kind
    if not sub.exists():
        return None
    epochs = sorted(sub.glob("model_epoch*"))
    return epochs[-1] if epochs else None


def extract_basin_series(results, basin_id):
    bd = results[basin_id]
    if isinstance(bd, dict) and "1D" in bd:
        ds = bd["1D"]["xr"]
    elif hasattr(bd, "xr"):
        ds = bd.xr
    else:
        raise ValueError(f"unknown result format for {basin_id}: {type(bd)}")
    obs = ds["QObs(mm/d)_obs"].values.flatten()
    pred = ds["QObs(mm/d)_sim"].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def log_score(obs, pred):
    return np.abs(
        np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6))
    )


def split_cp_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.nan
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


def coverage_log(obs, pred, q):
    return float(np.mean(log_score(obs, pred) <= q))


def interval_orig(pred, q):
    log_p = np.log(np.maximum(pred + EPS, 1e-6))
    lo = np.maximum(np.exp(log_p - q) - EPS, 0.0)
    hi = np.exp(log_p + q) - EPS
    return lo, hi


def winkler_score(obs, lo, hi, alpha):
    """
    Winkler interval score (Gneiting & Raftery 2007, Eq. 43):
      WS = (U - L)
           + (2/alpha)*(L - y)  if y < L
           + (2/alpha)*(y - U)  if y > U
    """
    width = hi - lo
    below = obs < lo
    above = obs > hi
    pen = np.zeros_like(obs)
    pen[below] = (2.0 / alpha) * (lo[below] - obs[below])
    pen[above] = (2.0 / alpha) * (obs[above] - hi[above])
    return float(np.mean(width + pen))


# -------- core: per-run computation on 574-subset --------

def compute_run_metrics(run_dir, basin_set, attrs, tag):
    """Compute HSCC + Global CP for one NH run dir restricted to basin_set."""
    val_ep = find_latest_epoch(run_dir, "validation")
    test_ep = find_latest_epoch(run_dir, "test")
    if val_ep is None or test_ep is None:
        raise RuntimeError(f"{tag}: missing val/test under {run_dir}")
    val_res = load_pickle(val_ep / "validation_results.p")
    test_res = load_pickle(test_ep / "test_results.p")

    records = []
    skipped = []
    for b in basin_set:
        if b not in val_res or b not in test_res:
            skipped.append((b, "missing in val/test"))
            continue
        try:
            cal_obs, cal_pred = extract_basin_series(val_res, b)
            tst_obs, tst_pred = extract_basin_series(test_res, b)
        except Exception as e:
            skipped.append((b, str(e)))
            continue
        if len(cal_obs) < 100 or len(tst_obs) < 100:
            skipped.append((b, f"too few days cal={len(cal_obs)} test={len(tst_obs)}"))
            continue
        gid = str(b).zfill(8)
        a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0})
        records.append({
            "basin_id": gid,
            "tier": assign_tier(a["aridity"], a["frac_snow"]),
            "cal_scores": log_score(cal_obs, cal_pred),
            "tst_obs": tst_obs,
            "tst_pred": tst_pred,
        })
    n = len(records)
    if skipped:
        print(f"  [{tag}] skipped {len(skipped)} (first 5: {skipped[:5]})")
    print(f"  [{tag}] basins analyzed: {n}")

    # Global q
    all_cal = np.concatenate([r["cal_scores"] for r in records])
    q_global = split_cp_quantile(all_cal, ALPHA)
    # HSCC q per tier
    q_tier = {}
    for t in TIERS:
        s_t = [r["cal_scores"] for r in records if r["tier"] == t]
        s_t = np.concatenate(s_t) if s_t else np.array([])
        q_tier[t] = split_cp_quantile(s_t, ALPHA) if len(s_t) else np.nan

    # per-tier metrics (basin-level mean of coverage/width/winkler)
    tier_rows = []
    for t in TIERS:
        rs = [r for r in records if r["tier"] == t]
        if not rs:
            continue
        gcov, hcov, gwid, hwid, gwin, hwin = [], [], [], [], [], []
        for r in rs:
            obs, pred = r["tst_obs"], r["tst_pred"]
            # global
            gcov.append(coverage_log(obs, pred, q_global))
            lo_g, hi_g = interval_orig(pred, q_global)
            gwid.append(float(np.mean(hi_g - lo_g)))
            gwin.append(winkler_score(obs, lo_g, hi_g, ALPHA))
            # hscc
            hcov.append(coverage_log(obs, pred, q_tier[t]))
            lo_h, hi_h = interval_orig(pred, q_tier[t])
            hwid.append(float(np.mean(hi_h - lo_h)))
            hwin.append(winkler_score(obs, lo_h, hi_h, ALPHA))
        tier_rows.append({
            "tier": t,
            "n_basins": len(rs),
            "global_coverage": float(np.mean(gcov)),
            "global_width_mm_d": float(np.mean(gwid)),
            "global_winkler_mm_d": float(np.mean(gwin)),
            "hscc_coverage": float(np.mean(hcov)),
            "hscc_width_mm_d": float(np.mean(hwid)),
            "hscc_winkler_mm_d": float(np.mean(hwin)),
            "global_q": float(q_global),
            "hscc_q": float(q_tier[t]),
        })
    return {
        "n_basins": n,
        "q_global": float(q_global),
        "q_tier": {t: float(q_tier[t]) for t in TIERS},
        "tier_rows": tier_rows,
    }


# -------- main --------

def main():
    print("=== exp010 Action 3: 574-subset comparison matrix ===\n")

    # 1. 574 basin list
    df574 = pd.read_csv(EXP010_BASIN_CSV, dtype={"basin": str})
    df574["basin"] = df574["basin"].str.zfill(8)
    basin_574 = list(df574["basin"].values)
    print(f"574-subset size (from exp010 per_basin_metrics.csv): {len(basin_574)}")

    attrs = load_attributes()

    # 2. exp002 single seed=42 on 574-subset
    print("\n--- exp002 (seed=42) on 574-subset ---")
    e2 = compute_run_metrics(EXP002_RUN, basin_574, attrs, "exp002")

    # 3. exp009 5 seeds on 574-subset
    print("\n--- exp009 (5 seeds) on 574-subset ---")
    e9_per_seed = {}
    for seed, rd in EXP009_SEED_DIRS.items():
        print(f"\n  seed={seed} run={rd.name}")
        e9_per_seed[seed] = compute_run_metrics(rd, basin_574, attrs, f"exp009_s{seed}")

    # aggregate exp009: mean ± std across seeds, per tier
    e9_tier_agg = []
    for t in TIERS:
        rows = []
        for seed, m in e9_per_seed.items():
            r = next((x for x in m["tier_rows"] if x["tier"] == t), None)
            if r is not None:
                rows.append(r)
        if not rows:
            continue
        arr = pd.DataFrame(rows)
        e9_tier_agg.append({
            "tier": t,
            "n_basins": int(arr["n_basins"].iloc[0]),
            "global_coverage_mean": float(arr["global_coverage"].mean()),
            "global_coverage_std": float(arr["global_coverage"].std(ddof=1)),
            "global_width_mean": float(arr["global_width_mm_d"].mean()),
            "global_width_std": float(arr["global_width_mm_d"].std(ddof=1)),
            "global_winkler_mean": float(arr["global_winkler_mm_d"].mean()),
            "global_winkler_std": float(arr["global_winkler_mm_d"].std(ddof=1)),
            "hscc_coverage_mean": float(arr["hscc_coverage"].mean()),
            "hscc_coverage_std": float(arr["hscc_coverage"].std(ddof=1)),
            "hscc_width_mean": float(arr["hscc_width_mm_d"].mean()),
            "hscc_width_std": float(arr["hscc_width_mm_d"].std(ddof=1)),
            "hscc_winkler_mean": float(arr["hscc_winkler_mm_d"].mean()),
            "hscc_winkler_std": float(arr["hscc_winkler_mm_d"].std(ddof=1)),
        })

    # 4. exp010 HopCPT (already 574-subset)
    print("\n--- exp010 HopCPT (already 574-subset, loaded from tier_aggregate.csv) ---")
    h_tier = pd.read_csv(EXP010_TIER_CSV)
    print(h_tier.to_string(index=False))

    # 5. Build long-format comparison matrix
    rows = []
    # per tier
    for t in TIERS:
        # exp002 single seed (Global CP)
        e2r = next((x for x in e2["tier_rows"] if x["tier"] == t), None)
        if e2r:
            rows.append({"tier": t, "method": "Global_CP_exp002_seed42",
                         "n_basins": e2r["n_basins"],
                         "coverage": e2r["global_coverage"],
                         "width_mm_d": e2r["global_width_mm_d"],
                         "winkler_mm_d": e2r["global_winkler_mm_d"],
                         "coverage_std": np.nan, "width_std": np.nan, "winkler_std": np.nan})
            rows.append({"tier": t, "method": "HSCC_exp002_seed42",
                         "n_basins": e2r["n_basins"],
                         "coverage": e2r["hscc_coverage"],
                         "width_mm_d": e2r["hscc_width_mm_d"],
                         "winkler_mm_d": e2r["hscc_winkler_mm_d"],
                         "coverage_std": np.nan, "width_std": np.nan, "winkler_std": np.nan})
        # exp009 5-seed mean ± std
        e9r = next((x for x in e9_tier_agg if x["tier"] == t), None)
        if e9r:
            rows.append({"tier": t, "method": "Global_CP_exp009_5seed",
                         "n_basins": e9r["n_basins"],
                         "coverage": e9r["global_coverage_mean"],
                         "coverage_std": e9r["global_coverage_std"],
                         "width_mm_d": e9r["global_width_mean"],
                         "width_std": e9r["global_width_std"],
                         "winkler_mm_d": e9r["global_winkler_mean"],
                         "winkler_std": e9r["global_winkler_std"]})
            rows.append({"tier": t, "method": "HSCC_exp009_5seed",
                         "n_basins": e9r["n_basins"],
                         "coverage": e9r["hscc_coverage_mean"],
                         "coverage_std": e9r["hscc_coverage_std"],
                         "width_mm_d": e9r["hscc_width_mean"],
                         "width_std": e9r["hscc_width_std"],
                         "winkler_mm_d": e9r["hscc_winkler_mean"],
                         "winkler_std": e9r["hscc_winkler_std"]})
        # exp010 HopCPT
        hr = h_tier[h_tier["tier"] == t]
        if not hr.empty:
            rows.append({"tier": t, "method": "HopCPT_exp010_seed42",
                         "n_basins": int(hr["n_basins"].iloc[0]),
                         "coverage": float(hr["mean_coverage"].iloc[0]),
                         "coverage_std": float(hr["std_coverage"].iloc[0]),
                         "width_mm_d": float(hr["mean_pi_width"].iloc[0]),
                         "width_std": np.nan,
                         "winkler_mm_d": float(hr["mean_winkler_score"].iloc[0]),
                         "winkler_std": np.nan})

    df = pd.DataFrame(rows)
    df = df[["tier", "method", "n_basins",
             "coverage", "coverage_std",
             "width_mm_d", "width_std",
             "winkler_mm_d", "winkler_std"]]
    out_csv = OUT_DIR / "comparison_matrix_574subset.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote: {out_csv}")
    print("\nComparison matrix (long-format):")
    print(df.to_string(index=False))

    # 6. Spread summary
    spread_summary = {}
    for tag, df_method in [
        ("Global_CP_exp002_seed42", df[df["method"] == "Global_CP_exp002_seed42"]),
        ("HSCC_exp002_seed42",       df[df["method"] == "HSCC_exp002_seed42"]),
        ("Global_CP_exp009_5seed",   df[df["method"] == "Global_CP_exp009_5seed"]),
        ("HSCC_exp009_5seed",        df[df["method"] == "HSCC_exp009_5seed"]),
        ("HopCPT_exp010_seed42",     df[df["method"] == "HopCPT_exp010_seed42"]),
    ]:
        if df_method.empty:
            continue
        spread_pp = (df_method["coverage"].max() - df_method["coverage"].min()) * 100
        spread_summary[tag] = {
            "spread_pp": float(spread_pp),
            "min_tier": str(df_method.loc[df_method["coverage"].idxmin(), "tier"]),
            "max_tier": str(df_method.loc[df_method["coverage"].idxmax(), "tier"]),
            "min_coverage": float(df_method["coverage"].min()),
            "max_coverage": float(df_method["coverage"].max()),
            "mean_width_mm_d": float(df_method["width_mm_d"].mean()),
            "mean_winkler_mm_d": float(df_method["winkler_mm_d"].mean()),
        }
    print("\nCross-tier spread summary:")
    for k, v in spread_summary.items():
        print(f"  {k:30s} spread={v['spread_pp']:5.2f}pp  "
              f"min={v['min_tier']:9s} ({v['min_coverage']:.3f})  "
              f"max={v['max_tier']:9s} ({v['max_coverage']:.3f})  "
              f"width={v['mean_width_mm_d']:.2f}  winkler={v['mean_winkler_mm_d']:.2f}")

    # 7. JSON dump
    out_json = OUT_DIR / "comparison_matrix_574subset.json"
    with open(out_json, "w") as f:
        json.dump({
            "alpha": ALPHA,
            "n_basins_subset": len(basin_574),
            "exp002": e2,
            "exp009_per_seed": {str(k): v for k, v in e9_per_seed.items()},
            "exp009_tier_5seed_aggregate": e9_tier_agg,
            "exp010_HopCPT_tier_aggregate": h_tier.to_dict(orient="records"),
            "comparison_long_format": df.to_dict(orient="records"),
            "spread_summary": spread_summary,
        }, f, indent=2, default=str)
    print(f"\nwrote: {out_json}")

    # 8. 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panel_methods = [
        ("Global_CP_exp009_5seed",  "Global CP (5-seed)", "#888888", "o"),
        ("HSCC_exp009_5seed",       "HSCC (5-seed)",      "#1f77b4", "s"),
        ("HopCPT_exp010_seed42",    "HopCPT (1 seed)",    "#d62728", "^"),
    ]
    x = np.arange(len(TIERS))

    # panel 1: coverage
    ax = axes[0, 0]
    for m, lbl, col, mk in panel_methods:
        sub = df[df["method"] == m].set_index("tier").reindex(TIERS)
        yerr = sub["coverage_std"].fillna(0).values * 1.96  # 95% CI ish
        ax.errorbar(x, sub["coverage"], yerr=yerr, fmt=mk, color=col, label=lbl,
                    capsize=3, markersize=7, lw=1.5)
    ax.axhline(0.90, color="k", ls="--", lw=1, label="Target 0.90")
    ax.set_xticks(x); ax.set_xticklabels([TIER_PRETTY[t] for t in TIERS])
    ax.set_ylabel("Mean coverage (basin-level)")
    ax.set_title("Per-tier coverage on 574-basin subset")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # panel 2: PI width
    ax = axes[0, 1]
    for m, lbl, col, mk in panel_methods:
        sub = df[df["method"] == m].set_index("tier").reindex(TIERS)
        yerr = sub["width_std"].fillna(0).values * 1.96
        ax.errorbar(x, sub["width_mm_d"], yerr=yerr, fmt=mk, color=col, label=lbl,
                    capsize=3, markersize=7, lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels([TIER_PRETTY[t] for t in TIERS])
    ax.set_ylabel("Mean PI width (mm/d)")
    ax.set_title("Per-tier interval width")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # panel 3: Winkler
    ax = axes[1, 0]
    for m, lbl, col, mk in panel_methods:
        sub = df[df["method"] == m].set_index("tier").reindex(TIERS)
        yerr = sub["winkler_std"].fillna(0).values * 1.96
        ax.errorbar(x, sub["winkler_mm_d"], yerr=yerr, fmt=mk, color=col, label=lbl,
                    capsize=3, markersize=7, lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels([TIER_PRETTY[t] for t in TIERS])
    ax.set_ylabel("Winkler score (mm/d)")
    ax.set_title("Per-tier Winkler score (lower = better)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # panel 4: cross-tier spread bar
    ax = axes[1, 1]
    bar_methods = [
        ("Global_CP_exp002_seed42", "Global CP\n(exp002 s42)"),
        ("HSCC_exp002_seed42",       "HSCC\n(exp002 s42)"),
        ("Global_CP_exp009_5seed",   "Global CP\n(5-seed mean)"),
        ("HSCC_exp009_5seed",        "HSCC\n(5-seed mean)"),
        ("HopCPT_exp010_seed42",     "HopCPT\n(s42)"),
    ]
    bx = np.arange(len(bar_methods))
    bv = [spread_summary.get(m, {}).get("spread_pp", 0) for m, _ in bar_methods]
    bcols = ["#888", "#1f77b4", "#888", "#1f77b4", "#d62728"]
    bars = ax.bar(bx, bv, color=bcols, edgecolor="k")
    for bi, v in zip(bars, bv):
        ax.annotate(f"{v:.1f}", xy=(bi.get_x() + bi.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(bx); ax.set_xticklabels([l for _, l in bar_methods], fontsize=8)
    ax.set_ylabel("Cross-tier coverage spread (pp)")
    ax.set_title("Reliability headline (lower = more uniform across tiers)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"574-basin subset head-to-head: HSCC vs Global CP vs HopCPT (α={ALPHA})",
        y=0.995, fontsize=12,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "comparison_matrix_574subset_panel.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote: {out_png}")


if __name__ == "__main__":
    main()
