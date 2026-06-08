"""
exp004 HSCC analysis — CAMELS-GB version (adapted from experiments/exp002/hscc_analysis_v2.py).

Differences vs exp002:
  - tier metadata source: experiments/exp004/basin_lists/gb_basin_tiers.csv
    (produced by experiments/exp004/scripts/select_gb_basins.py)
  - NH result column names: discharge_spec_obs / discharge_spec_sim (instead of QObs(mm/d)_*)
  - basin_id format: CAMELS-GB uses non-zero-padded gauge IDs (varying length)

Pipeline mirrors exp002:
  1. Load NH validation_results.p (calibration period 2000-2010)
  2. Load NH test_results.p (test period 2010-2015)
  3. Compute log-flow residual scores (PROJECT_CHARTER §2.2 frozen, ε=0.01)
  4. Global split CP: pool all cal scores → q_global
  5. HSCC: pool cal scores per tier → q_tier
  6. Apply each q to test → per-tier coverage + interval width
  7. Output: hscc_results.csv + metrics.json + summary verdict
  8. Comparison vs exp002 US results (read from experiments/exp002/.../metrics.json)

Usage:
    python experiments/exp004/hscc_analysis_gb.py --run_dir <run_dir>
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TIERS_FILE = ROOT / "experiments" / "exp004" / "basin_lists" / "gb_basin_tiers.csv"
EXP002_METRICS = ROOT / "experiments" / "exp002" / "results" / "exp002_camels_us_temporal_2504_155058" / "_analysis" / "metrics.json"

ALPHA = 0.10
EPS = 0.01

TARGET_OBS_COL = "discharge_spec_obs"
TARGET_SIM_COL = "discharge_spec_sim"


def load_basin_tiers():
    """Read basin → tier mapping produced by select_gb_basins.py."""
    if not TIERS_FILE.exists():
        raise FileNotFoundError(f"missing {TIERS_FILE}; run select_gb_basins.py first.")
    df = pd.read_csv(TIERS_FILE, dtype={"gauge_id": str})
    return {
        str(row.gauge_id): {
            "tier": row.tier,
            "aridity": float(row.aridity),
            "frac_snow": float(row.frac_snow),
        }
        for row in df.itertuples()
    }


def log_score(obs, pred):
    return np.abs(np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6)))


def split_cp_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.nan
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return np.quantile(scores, level)


def coverage_log(obs, pred, q):
    s = log_score(obs, pred)
    return float(np.mean(s <= q))


def width_orig(pred, q):
    log_p = np.log(np.maximum(pred + EPS, 1e-6))
    lo = np.maximum(np.exp(log_p - q) - EPS, 0)
    hi = np.exp(log_p + q) - EPS
    return float(np.mean(hi - lo))


def extract_basin_series(results, basin_id):
    bd = results[basin_id]
    if isinstance(bd, dict) and "1D" in bd:
        ds = bd["1D"]["xr"]
    elif hasattr(bd, "xr"):
        ds = bd.xr
    else:
        raise ValueError(f"unknown result format for basin {basin_id}: {type(bd)}")
    if TARGET_OBS_COL not in ds or TARGET_SIM_COL not in ds:
        # Fallback: NH sometimes preserves original column names with parentheses
        avail = list(ds.data_vars)
        raise ValueError(f"basin {basin_id}: missing {TARGET_OBS_COL}/{TARGET_SIM_COL}; available={avail[:6]}")
    obs = ds[TARGET_OBS_COL].values.flatten()
    pred = ds[TARGET_SIM_COL].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def load_nh_pickle(epoch_dir, kind):
    p = epoch_dir / f"{kind}_results.p"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def find_latest_epoch(run_dir, kind):
    sub = run_dir / kind
    if not sub.exists():
        return None
    epochs = sorted(sub.glob("model_epoch*"))
    return epochs[-1] if epochs else None


def run_analysis(run_dir, out_dir):
    print(f"\n=== exp004 HSCC Analysis (CAMELS-GB) ===")
    print(f"run_dir: {run_dir}")

    val_epoch = find_latest_epoch(run_dir, "validation")
    test_epoch = find_latest_epoch(run_dir, "test")
    if val_epoch is None or test_epoch is None:
        print(f"ERROR: missing val/test outputs (val={val_epoch}, test={test_epoch})")
        return None

    print(f"calibration source: {val_epoch.relative_to(run_dir)}")
    print(f"test source       : {test_epoch.relative_to(run_dir)}")

    val_res = load_nh_pickle(val_epoch, "validation")
    test_res = load_nh_pickle(test_epoch, "test")
    tiers = load_basin_tiers()

    records = []
    for basin_id in val_res.keys():
        if basin_id not in test_res:
            continue
        try:
            cal_obs, cal_pred = extract_basin_series(val_res, basin_id)
            tst_obs, tst_pred = extract_basin_series(test_res, basin_id)
        except Exception as e:
            print(f"  skip {basin_id}: {e}")
            continue
        if len(cal_obs) < 100 or len(tst_obs) < 100:
            print(f"  skip {basin_id}: too few valid days (cal={len(cal_obs)}, test={len(tst_obs)})")
            continue

        meta = tiers.get(str(basin_id))
        if meta is None:
            print(f"  skip {basin_id}: no tier metadata")
            continue

        records.append({
            "basin_id": str(basin_id),
            "tier": meta["tier"],
            "aridity": meta["aridity"],
            "frac_snow": meta["frac_snow"],
            "cal_scores": log_score(cal_obs, cal_pred),
            "test_obs": tst_obs,
            "test_pred": tst_pred,
            "n_cal": len(cal_obs),
            "n_test": len(tst_obs),
            "test_pred_min": float(np.min(tst_pred)),
            "nse_test": _nse(tst_obs, tst_pred),
        })

    print(f"\nbasins analyzed: {len(records)}")
    if not records:
        print("FAIL: no usable basins.")
        return None

    # Per-tier min(Q̂)
    tier_min_q = {}
    print("\n--- per-tier min(Q̂) (S5 sanity, mirror exp002 NOTE-001) ---")
    for tier in ["gb_drier_q4", "gb_mid", "gb_wet", "gb_montane"]:
        rs = [r["test_pred_min"] for r in records if r["tier"] == tier]
        if not rs:
            continue
        m = min(rs)
        tier_min_q[tier] = m
        flag = " ✅" if m >= -5 else " ⚠️ alarm"  # S5: log-clipping magnitude ≤ 5
        print(f"  {tier:10s}: min(Q̂) = {m:+.4f}{flag}")

    # Global CP
    all_cal = np.concatenate([r["cal_scores"] for r in records])
    q_global = split_cp_quantile(all_cal, ALPHA)
    print(f"\nGlobal CP quantile (α={ALPHA}): {q_global:.4f}  (n_cal_pts={len(all_cal)})")

    # HSCC per-tier (GB-specific labels per D-022)
    tier_list = ["gb_drier_q4", "gb_mid", "gb_wet", "gb_montane"]
    q_tier = {}
    for t in tier_list:
        s = np.concatenate([r["cal_scores"] for r in records if r["tier"] == t]) \
            if any(r["tier"] == t for r in records) else np.array([])
        q_tier[t] = split_cp_quantile(s, ALPHA) if len(s) else np.nan
        print(f"  HSCC q[{t:10s}] = {q_tier[t]:.4f}  (n_cal_pts={len(s)})")

    # Per-tier coverage + width
    print(f"\n{'Tier':10s} | {'n':4s} | {'global_cov':10s} | {'HSCC_cov':9s} | "
          f"{'global_w':9s} | {'HSCC_w':8s} | {'NSE':6s} | {'target':6s}")
    print("-" * 88)
    summary = []
    target = 1 - ALPHA
    for t in tier_list:
        rs = [r for r in records if r["tier"] == t]
        if not rs:
            continue
        cgs, chs, wgs, whs, nses = [], [], [], [], []
        for r in rs:
            cgs.append(coverage_log(r["test_obs"], r["test_pred"], q_global))
            chs.append(coverage_log(r["test_obs"], r["test_pred"], q_tier[t]))
            wgs.append(width_orig(r["test_pred"], q_global))
            whs.append(width_orig(r["test_pred"], q_tier[t]))
            nses.append(r["nse_test"])
        cg, ch = np.mean(cgs), np.mean(chs)
        wg, wh = np.mean(wgs), np.mean(whs)
        nse = np.mean(nses)
        flag = " ← C1 evidence" if abs(cg - target) > 0.05 else ""
        print(f"{t:10s} | {len(rs):4d} | {cg:10.3f} | {ch:9.3f} | {wg:9.2f} | {wh:8.2f} | "
              f"{nse:6.3f} | {target:.2f}{flag}")
        summary.append({
            "tier": t, "n_basins": len(rs),
            "global_coverage": cg, "hscc_coverage": ch,
            "global_width_mm_d": wg, "hscc_width_mm_d": wh,
            "global_q": q_global, "hscc_q": q_tier[t],
            "nse_mean": nse,
            "test_pred_min_in_tier": tier_min_q.get(t),
        })

    df = pd.DataFrame(summary)
    spread_global = df["global_coverage"].max() - df["global_coverage"].min()
    spread_hscc = df["hscc_coverage"].max() - df["hscc_coverage"].min()
    nse_overall = float(df["nse_mean"].mean())

    print(f"\nTier coverage spread: global = {spread_global*100:.1f}pp, HSCC = {spread_hscc*100:.1f}pp")
    print(f"Mean NSE (test): {nse_overall:.3f}")

    # Frame decision (plan §1)
    if spread_global >= 0.10 and (spread_global - spread_hscc) >= 0.10:
        frame = "consistent"
        frame_note = "GB results consistent with US — main-text §5.6 positive frame"
    elif spread_global >= 0.10 and (spread_global - spread_hscc) >= 0.08:
        frame = "partial"
        frame_note = "Partial replication — qualitative success; quantitative weaker"
    else:
        frame = "inconsistent"
        frame_note = "Push to §6.4 Discussion as applicability boundary"

    # Plan §4 success criteria
    s1_pass = spread_global >= 0.10
    s2_pass = ((df["hscc_coverage"] >= 0.85) & (df["hscc_coverage"] <= 0.93)).sum() >= 3
    s3_pass = (spread_global - spread_hscc) >= 0.08
    s4_pass = nse_overall >= 0.4
    s5_pass = all(abs(tier_min_q.get(t, 0)) <= 5 for t in tier_min_q)

    print("\n--- Success criteria (plan §4) ---")
    print(f"  S1 (GB global spread ≥ 10pp = C1)     : {'✅' if s1_pass else '❌'} ({spread_global*100:.1f}pp)")
    print(f"  S2 (HSCC ≥3/4 tier ∈ [0.85, 0.93])    : {'✅' if s2_pass else '❌'}")
    print(f"  S3 (HSCC spread reduction ≥ 8pp)      : {'✅' if s3_pass else '❌'} ({(spread_global-spread_hscc)*100:.1f}pp)")
    print(f"  S4 (mean NSE ≥ 0.4)                   : {'✅' if s4_pass else '❌'} ({nse_overall:.3f})")
    print(f"  S5 (|min(Q̂)| ≤ 5)                     : {'✅' if s5_pass else '⚠️'}")
    print(f"\n→ Frame: {frame} — {frame_note}")

    # exp002 comparison
    us_cmp = None
    if EXP002_METRICS.exists():
        with open(EXP002_METRICS) as f:
            us = json.load(f)
        us_spread_g = us.get("tier_coverage_spread_global_pp")
        us_spread_h = us.get("tier_coverage_spread_hscc_pp")
        us_cmp = {
            "us_global_spread_pp": us_spread_g,
            "us_hscc_spread_pp": us_spread_h,
            "gb_global_spread_pp": spread_global * 100,
            "gb_hscc_spread_pp": spread_hscc * 100,
        }
        print(f"\nUS (exp002) vs GB (exp004):")
        print(f"  global spread: US {us_spread_g:.2f}pp → GB {spread_global*100:.2f}pp")
        print(f"  HSCC   spread: US {us_spread_h:.2f}pp → GB {spread_hscc*100:.2f}pp")

    # Outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "hscc_results.csv", index=False)
    metrics = {
        "alpha": ALPHA,
        "n_basins_analyzed": len(records),
        "global_q": float(q_global),
        "tier_coverage_spread_global_pp": float(spread_global * 100),
        "tier_coverage_spread_hscc_pp": float(spread_hscc * 100),
        "nse_overall": nse_overall,
        "per_tier": {r["tier"]: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                   for k, v in r.items() if k != "tier"}
                       for r in summary},
        "success_criteria": {
            "S1": bool(s1_pass), "S2": bool(s2_pass), "S3": bool(s3_pass),
            "S4": bool(s4_pass), "S5": bool(s5_pass),
        },
        "frame": frame,
        "frame_note": frame_note,
        "us_comparison": us_cmp,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"\nwrote: {out_dir/'hscc_results.csv'}")
    print(f"wrote: {out_dir/'metrics.json'}")
    return metrics


def _nse(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    den = np.sum((obs - obs.mean()) ** 2)
    if den <= 0:
        return float("nan")
    return 1.0 - np.sum((obs - pred) ** 2) / den


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    rd = Path(args.run_dir).resolve()
    od = Path(args.out_dir).resolve() if args.out_dir else (rd / "_analysis")
    run_analysis(rd, od)
