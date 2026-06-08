"""
exp002 HSCC analysis (adapted from exp001_gonogo/hscc_analysis.py).

Key change vs gonogo (NOTE-001 修复):
  gonogo: array halving (n_cal = n // 2) on a single test_results.p time series
  exp002: read NeuralHydrology validation_results.p (calibration period 1990-2000)
          and test_results.p (test period 2000-2014) separately — proper date-period splits.

Pipeline:
  1. Load NH validation_results.p (calibration period) → per-basin (obs, pred) on cal period
  2. Load NH test_results.p (test period) → per-basin (obs, pred) on test period
  3. Compute log-flow residual scores
  4. Global split CP: pool all cal scores → q_global
  5. HSCC: pool cal scores per tier (snow > dry > semi_arid > humid) → q_tier
  6. Apply each q to test (obs, pred) → per-tier coverage + interval width
  7. Output: hscc_results.csv + global_cp_results.csv + summary verdict

Usage:
    /home/qingsong/miniconda3/envs/hscc-hydrology/bin/python \
      experiments/exp002/hscc_analysis_v2.py \
      --run_dir experiments/exp002/results/<run_name>
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ATTR_FILE = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
ALPHA = 0.10
EPS = 0.01

TIER_SNOW_FRAC = 0.40
TIER_DRY_AI = 1.5
TIER_SEMI_AI = 1.0


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
    if fs >= TIER_SNOW_FRAC:
        return "snow"
    if ai > TIER_DRY_AI:
        return "dry"
    if ai > TIER_SEMI_AI:
        return "semi_arid"
    return "humid"


def log_score(obs, pred):
    return np.abs(np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6)))


def split_cp_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.nan
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return np.quantile(scores, level)


def coverage_log(obs, pred, q):
    """In log-flow space: |log(obs+eps) - log(pred+eps)| <= q."""
    s = log_score(obs, pred)
    return float(np.mean(s <= q))


def width_orig(pred, q):
    """Mean interval width in original flow units (mm/d), translated from log-space band."""
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
    obs = ds["QObs(mm/d)_obs"].values.flatten()
    pred = ds["QObs(mm/d)_sim"].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def load_nh_pickle(epoch_dir: Path, kind: str):
    """kind in {'validation', 'test'} — NH stores results as kind+'_results.p' under model_epochN/."""
    p = epoch_dir / f"{kind}_results.p"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def find_latest_epoch(run_dir: Path, kind: str):
    sub = run_dir / kind
    if not sub.exists():
        return None
    epochs = sorted(sub.glob("model_epoch*"))
    return epochs[-1] if epochs else None


def run_analysis(run_dir: Path, out_dir: Path):
    print(f"\n=== exp002 HSCC Analysis ===")
    print(f"run_dir: {run_dir}")

    val_epoch = find_latest_epoch(run_dir, "validation")
    test_epoch = find_latest_epoch(run_dir, "test")
    if val_epoch is None or test_epoch is None:
        print(f"ERROR: missing validation or test outputs (val={val_epoch}, test={test_epoch})")
        return None

    print(f"calibration source: {val_epoch.relative_to(run_dir)}")
    print(f"test source       : {test_epoch.relative_to(run_dir)}")

    val_res = load_nh_pickle(val_epoch, "validation")
    test_res = load_nh_pickle(test_epoch, "test")
    attrs = load_attributes()

    # Per-basin extraction
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

        gid = str(basin_id).zfill(8)
        a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0})
        records.append({
            "basin_id": gid,
            "tier": assign_tier(a["aridity"], a["frac_snow"]),
            "aridity": a["aridity"],
            "frac_snow": a["frac_snow"],
            "cal_scores": log_score(cal_obs, cal_pred),
            "test_obs": tst_obs,
            "test_pred": tst_pred,
            "n_cal": len(cal_obs),
            "n_test": len(tst_obs),
            # NOTE-001 第一项验证：每 basin 训练后预测下界
            "test_pred_min": float(np.min(tst_pred)),
        })

    print(f"\nbasins analyzed: {len(records)}")

    # NOTE-001 第一项: per-tier min(Q̂)
    print("\n--- NOTE-001 check: per-tier min(Q̂) (must be >= 0 for S4) ---")
    tier_min_q = {}
    for tier in ["dry", "semi_arid", "humid", "snow"]:
        rs = [r["test_pred_min"] for r in records if r["tier"] == tier]
        if not rs:
            continue
        m = min(rs)
        tier_min_q[tier] = m
        flag = " ✅" if m >= 0 else " ❌ FAIL S4"
        print(f"  {tier:10s}: min(Q̂) = {m:+.4f}{flag}")

    # Global CP
    all_cal = np.concatenate([r["cal_scores"] for r in records])
    q_global = split_cp_quantile(all_cal, ALPHA)
    print(f"\nGlobal CP quantile (α={ALPHA}): {q_global:.4f}  (n_cal_pts={len(all_cal)})")

    # HSCC per-tier
    tiers = ["dry", "semi_arid", "humid", "snow"]
    q_tier = {}
    for t in tiers:
        s = np.concatenate([r["cal_scores"] for r in records if r["tier"] == t]) \
            if any(r["tier"] == t for r in records) else np.array([])
        q_tier[t] = split_cp_quantile(s, ALPHA) if len(s) else np.nan
        print(f"  HSCC q[{t}] = {q_tier[t]:.4f}  (n_cal_pts={len(s)})")

    # Per-tier coverage + width
    print(f"\n{'Tier':10s} | {'n':4s} | {'global_cov':10s} | {'HSCC_cov':9s} | "
          f"{'global_w':9s} | {'HSCC_w':8s} | {'target':6s}")
    print("-" * 78)
    summary = []
    target = 1 - ALPHA
    for t in tiers:
        rs = [r for r in records if r["tier"] == t]
        if not rs:
            continue
        cgs, chs, wgs, whs = [], [], [], []
        for r in rs:
            cgs.append(coverage_log(r["test_obs"], r["test_pred"], q_global))
            chs.append(coverage_log(r["test_obs"], r["test_pred"], q_tier[t]))
            wgs.append(width_orig(r["test_pred"], q_global))
            whs.append(width_orig(r["test_pred"], q_tier[t]))
        cg, ch = np.mean(cgs), np.mean(chs)
        wg, wh = np.mean(wgs), np.mean(whs)
        flag = ""
        if abs(cg - target) > 0.05:
            flag = " ← C1 evidence"
        print(f"{t:10s} | {len(rs):4d} | {cg:10.3f} | {ch:9.3f} | {wg:9.2f} | {wh:8.2f} | "
              f"{target:.2f}{flag}")
        summary.append({
            "tier": t, "n_basins": len(rs),
            "global_coverage": cg, "hscc_coverage": ch,
            "global_width_mm_d": wg, "hscc_width_mm_d": wh,
            "global_q": q_global, "hscc_q": q_tier[t],
            "test_pred_min_in_tier": tier_min_q.get(t),
        })

    # Aggregate
    all_test_obs = np.concatenate([r["test_obs"] for r in records])
    all_test_pred = np.concatenate([r["test_pred"] for r in records])
    overall_global = coverage_log(all_test_obs, all_test_pred, q_global)
    print(f"\nOverall global CP coverage: {overall_global:.3f} (target {target:.2f})")

    df = pd.DataFrame(summary)
    spread_global = df["global_coverage"].max() - df["global_coverage"].min()
    spread_hscc = df["hscc_coverage"].max() - df["hscc_coverage"].min()
    print(f"\nTier coverage spread: global = {spread_global*100:.1f}pp, HSCC = {spread_hscc*100:.1f}pp")

    # Success criteria S1-S6
    print("\n--- Success criteria (PROJECT_CHARTER §2.1) ---")
    s1 = ((df["hscc_coverage"] >= 0.88) & (df["hscc_coverage"] <= 0.92)).all()
    s2 = spread_hscc <= 0.05
    s3 = spread_global >= 0.10
    s4 = all((tier_min_q.get(t, 0) >= 0) for t in tiers if t in tier_min_q)
    s5 = bool((df.loc[df["tier"] == "humid", "hscc_coverage"] >= 0.88).all()) if (df["tier"] == "humid").any() else None
    print(f"  S1 (all tiers HSCC ∈ [0.88, 0.92])  : {'✅' if s1 else '❌'}")
    print(f"  S2 (HSCC spread ≤ 5pp)              : {'✅' if s2 else '❌'} ({spread_hscc*100:.1f}pp)")
    print(f"  S3 (global spread ≥ 10pp = C1)      : {'✅' if s3 else '❌'} ({spread_global*100:.1f}pp)")
    print(f"  S4 (per-tier min(Q̂) ≥ 0)           : {'✅' if s4 else '❌'}")
    print(f"  S5 (humid HSCC ≥ 0.88)              : {'✅' if s5 else '❌' if s5 is False else 'N/A'}")

    # Outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "hscc_results.csv", index=False)
    pd.DataFrame([{
        "global_q": q_global,
        "global_overall_coverage": overall_global,
        "n_basins_analyzed": len(records),
    }]).to_csv(out_dir / "global_cp_results.csv", index=False)

    metrics = {
        "alpha": ALPHA,
        "n_basins_analyzed": len(records),
        "global_q": float(q_global),
        "overall_global_coverage": float(overall_global),
        "tier_coverage_spread_global_pp": float(spread_global * 100),
        "tier_coverage_spread_hscc_pp": float(spread_hscc * 100),
        "per_tier": {
            r["tier"]: {
                "n_basins": int(r["n_basins"]),
                "global_coverage": float(r["global_coverage"]),
                "hscc_coverage": float(r["hscc_coverage"]),
                "global_width_mm_d": float(r["global_width_mm_d"]),
                "hscc_width_mm_d": float(r["hscc_width_mm_d"]),
                "test_pred_min": float(r["test_pred_min_in_tier"]) if r["test_pred_min_in_tier"] is not None else None,
            } for r in summary
        },
        "success_criteria": {"S1": bool(s1), "S2": bool(s2), "S3": bool(s3), "S4": bool(s4),
                             "S5": (None if s5 is None else bool(s5))},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nwrote: {out_dir/'hscc_results.csv'}")
    print(f"wrote: {out_dir/'global_cp_results.csv'}")
    print(f"wrote: {out_dir/'metrics.json'}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str,
                        help="path to NH run dir (contains validation/ and test/)")
    parser.add_argument("--out_dir", default=None, type=str,
                        help="output dir (default: <run_dir>/../<analysis>)")
    args = parser.parse_args()
    rd = Path(args.run_dir).resolve()
    od = Path(args.out_dir).resolve() if args.out_dir else rd.parent
    run_analysis(rd, od)
