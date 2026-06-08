"""
exp003 HSCC LORO analysis — per-fold version.

Adapted from experiments/exp002/hscc_analysis_v2.py with the following PUB-specific
changes (per exp003/plan.md §方法 + §成功标准):

  - Cal scores come from `train_basins` (the 17-HUC training pool's validation period
    1990-2000). These basins are NOT in the held-out HUC.
  - Test (obs, pred) come from `test_basins` (basins located inside held-out HUC-k,
    test period 2000-2014).
  - HSCC quantile per tier (snow > dry > semi_arid > humid) is built from the
    cal_scores of train_basins, then applied to the held-out HUC test set.
  - PUB-relaxed success criteria: S1' (per-fold per-tier in [0.85, 0.92]; tiers with
    n_test < 3 are skipped from S1' denominator), S2' (HSCC spread ≤ 8pp on tiers
    that have test basins), S3' (global spread ≥ 10pp), S4' (per-tier min(Q̂) only
    alarm if max neg > 5.0 per D-016 v1.1), S5' (humid HSCC ≥ 0.85 if humid present),
    S6' (HSCC reduces global spread by ≥ 5pp).

Outputs (under out_dir):
  - metrics.json        per-fold complete summary (consumed by aggregate_loro.py)
  - hscc_results.csv    per-tier rows
  - global_cp_results.csv

Usage:
    python experiments/exp003/hscc_analysis_loro.py \
      --run_dir experiments/exp003/results/<fold_run> \
      --huc 09 \
      --out_dir experiments/exp003/results/<fold_run>/_analysis
"""
from __future__ import annotations

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

# PUB-relaxed thresholds (exp003 plan §成功标准)
PUB_TIER_LO = 0.85
PUB_TIER_HI = 0.92
PUB_HUMID_MIN = 0.85
PUB_S2_MAX_SPREAD = 0.08
PUB_S4_NEG_ALARM = 5.0
PUB_S6_MIN_SPREAD_REDUCTION = 0.05
TIER_MIN_TEST_BASINS = 3  # tiers with fewer test basins skipped from S1' denominator


def load_attributes() -> dict[str, dict[str, float]]:
    attrs: dict[str, dict[str, float]] = {}
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


def assign_tier(ai: float, fs: float) -> str:
    if fs >= TIER_SNOW_FRAC:
        return "snow"
    if ai > TIER_DRY_AI:
        return "dry"
    if ai > TIER_SEMI_AI:
        return "semi_arid"
    return "humid"


def log_score(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6)))


def split_cp_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return float("nan")
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


def coverage_log(obs: np.ndarray, pred: np.ndarray, q: float) -> float:
    s = log_score(obs, pred)
    return float(np.mean(s <= q))


def width_orig(pred: np.ndarray, q: float) -> float:
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


def build_basin_records(results, attrs, *, kind: str):
    records = []
    skipped = 0
    for basin_id in results.keys():
        try:
            obs, pred = extract_basin_series(results, basin_id)
        except Exception as exc:
            print(f"  skip {basin_id} ({kind}): {exc}")
            skipped += 1
            continue
        if len(obs) < 100:
            skipped += 1
            continue
        gid = str(basin_id).zfill(8)
        a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0})
        records.append(
            {
                "basin_id": gid,
                "tier": assign_tier(a["aridity"], a["frac_snow"]),
                "aridity": a["aridity"],
                "frac_snow": a["frac_snow"],
                "obs": obs,
                "pred": pred,
                "n": len(obs),
                "pred_min": float(np.min(pred)),
            }
        )
    return records, skipped


def run_fold_analysis(run_dir: Path, huc: str, out_dir: Path) -> dict | None:
    print(f"\n=== exp003 LORO fold HUC-{huc} ===")
    print(f"run_dir: {run_dir}")

    val_epoch = find_latest_epoch(run_dir, "validation")
    test_epoch = find_latest_epoch(run_dir, "test")
    if val_epoch is None or test_epoch is None:
        print(f"ERROR: missing validation or test outputs (val={val_epoch}, test={test_epoch})")
        return None

    print(f"calibration source: {val_epoch.relative_to(run_dir)}")
    print(f"test source       : {test_epoch.relative_to(run_dir)}")

    val_res = load_nh_pickle(val_epoch, "validation")  # train_basins on cal period
    test_res = load_nh_pickle(test_epoch, "test")  # held-out HUC basins on test period
    attrs = load_attributes()

    cal_records, cal_skipped = build_basin_records(val_res, attrs, kind="cal")
    tst_records, tst_skipped = build_basin_records(test_res, attrs, kind="test")

    print(f"cal basins (train pool):  {len(cal_records)} (skipped {cal_skipped})")
    print(f"test basins (HUC-{huc}):    {len(tst_records)} (skipped {tst_skipped})")

    # PUB integrity: held-out HUC basins must NOT appear in cal pool
    cal_ids = {r["basin_id"] for r in cal_records}
    tst_ids = {r["basin_id"] for r in tst_records}
    leakage = cal_ids & tst_ids
    if leakage:
        print(f"FATAL: PUB leakage — {len(leakage)} basins appear in BOTH cal and test")
        print(f"  examples: {sorted(leakage)[:10]}")
        return None

    # Build cal scores per tier (HSCC) and pooled (Global CP)
    for r in cal_records:
        r["scores"] = log_score(r["obs"], r["pred"])

    all_cal_scores = np.concatenate([r["scores"] for r in cal_records]) if cal_records else np.array([])
    q_global = split_cp_quantile(all_cal_scores, ALPHA)

    tiers = ["dry", "semi_arid", "humid", "snow"]
    q_tier: dict[str, float] = {}
    n_cal_tier: dict[str, int] = {}
    for t in tiers:
        s_list = [r["scores"] for r in cal_records if r["tier"] == t]
        s = np.concatenate(s_list) if s_list else np.array([])
        n_cal_tier[t] = len(s)
        q_tier[t] = split_cp_quantile(s, ALPHA) if len(s) else float("nan")

    print(f"\nGlobal CP q (α={ALPHA}): {q_global:.4f}  (cal_pts={len(all_cal_scores)})")
    for t in tiers:
        nb = sum(1 for r in cal_records if r["tier"] == t)
        print(f"  HSCC q[{t}]={q_tier[t]:.4f}  (cal_basins={nb}, cal_pts={n_cal_tier[t]})")

    # Per-tier eval on held-out HUC test set
    summary: list[dict] = []
    tier_pred_min: dict[str, float] = {}
    print(
        f"\n{'Tier':10s} | {'n':3s} | {'cov_glob':9s} | {'cov_HSCC':9s} | "
        f"{'w_glob':8s} | {'w_HSCC':8s} | min(Q̂)"
    )
    print("-" * 78)
    for t in tiers:
        rs = [r for r in tst_records if r["tier"] == t]
        if not rs:
            continue
        cgs, chs, wgs, whs = [], [], [], []
        mins = []
        for r in rs:
            cgs.append(coverage_log(r["obs"], r["pred"], q_global))
            wgs.append(width_orig(r["pred"], q_global))
            if not np.isnan(q_tier[t]):
                chs.append(coverage_log(r["obs"], r["pred"], q_tier[t]))
                whs.append(width_orig(r["pred"], q_tier[t]))
            mins.append(r["pred_min"])
        cg = float(np.mean(cgs))
        wg = float(np.mean(wgs))
        ch = float(np.mean(chs)) if chs else float("nan")
        wh = float(np.mean(whs)) if whs else float("nan")
        tmin = float(np.min(mins))
        tier_pred_min[t] = tmin
        print(
            f"{t:10s} | {len(rs):3d} | {cg:9.3f} | {ch:9.3f} | {wg:8.2f} | "
            f"{wh:8.2f} | {tmin:+.3f}"
        )
        summary.append(
            {
                "tier": t,
                "n_basins": len(rs),
                "global_coverage": cg,
                "hscc_coverage": ch,
                "global_width_mm_d": wg,
                "hscc_width_mm_d": wh,
                "global_q": q_global,
                "hscc_q": q_tier[t],
                "test_pred_min": tmin,
            }
        )

    if not summary:
        print(f"WARN: HUC-{huc} fold yielded zero per-tier eval rows (no test basins?)")
        return None

    df = pd.DataFrame(summary)
    spread_global = float(df["global_coverage"].max() - df["global_coverage"].min())
    spread_hscc = float(df["hscc_coverage"].max() - df["hscc_coverage"].min())
    print(
        f"\nTier coverage spread (HUC-{huc}): global = {spread_global*100:.1f}pp, "
        f"HSCC = {spread_hscc*100:.1f}pp"
    )

    # PUB success criteria (exp003 plan §成功标准)
    eligible_tiers = df[df["n_basins"] >= TIER_MIN_TEST_BASINS]
    s1_pass_per_tier = bool(
        (eligible_tiers["hscc_coverage"] >= PUB_TIER_LO).all()
        and (eligible_tiers["hscc_coverage"] <= PUB_TIER_HI).all()
    ) if len(eligible_tiers) else False
    s2 = bool(spread_hscc <= PUB_S2_MAX_SPREAD)
    s3 = bool(spread_global >= 0.10)
    max_neg = -min(min(tier_pred_min.values()), 0.0) if tier_pred_min else 0.0
    s4 = bool(max_neg <= PUB_S4_NEG_ALARM)
    if "humid" in df["tier"].values:
        humid_cov = float(df.loc[df["tier"] == "humid", "hscc_coverage"].iloc[0])
        s5 = bool(humid_cov >= PUB_HUMID_MIN)
    else:
        s5 = None
    s6 = bool((spread_global - spread_hscc) >= PUB_S6_MIN_SPREAD_REDUCTION)

    print("\n--- PUB-relaxed success criteria (per-fold view) ---")
    print(f"  S1' eligible-tier HSCC ∈ [{PUB_TIER_LO}, {PUB_TIER_HI}]: "
          f"{'✅' if s1_pass_per_tier else '❌'} (eligible n_tiers={len(eligible_tiers)}/{len(df)})")
    print(f"  S2' HSCC spread ≤ {PUB_S2_MAX_SPREAD*100:.0f}pp           : "
          f"{'✅' if s2 else '❌'} ({spread_hscc*100:.1f}pp)")
    print(f"  S3' global spread ≥ 10pp                : "
          f"{'✅' if s3 else '❌'} ({spread_global*100:.1f}pp)")
    print(f"  S4' max neg(Q̂) ≤ {PUB_S4_NEG_ALARM}              : "
          f"{'✅' if s4 else '❌ ALARM'} (max_neg={max_neg:.2f})")
    print(f"  S5' humid HSCC ≥ {PUB_HUMID_MIN}              : "
          f"{'✅' if s5 else '❌' if s5 is False else 'N/A'}")
    print(f"  S6' HSCC reduces spread ≥ "
          f"{PUB_S6_MIN_SPREAD_REDUCTION*100:.0f}pp : {'✅' if s6 else '❌'} "
          f"(Δ={(spread_global-spread_hscc)*100:.1f}pp)")

    # Outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "hscc_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "huc": huc,
                "global_q": q_global,
                "n_cal_basins": len(cal_records),
                "n_test_basins": len(tst_records),
            }
        ]
    ).to_csv(out_dir / "global_cp_results.csv", index=False)

    metrics = {
        "huc": huc,
        "alpha": ALPHA,
        "n_cal_basins": len(cal_records),
        "n_test_basins": len(tst_records),
        "global_q": float(q_global),
        "tier_coverage_spread_global_pp": spread_global * 100,
        "tier_coverage_spread_hscc_pp": spread_hscc * 100,
        "per_tier": {
            row["tier"]: {
                "n_basins": int(row["n_basins"]),
                "global_coverage": float(row["global_coverage"]),
                "hscc_coverage": float(row["hscc_coverage"]),
                "global_width_mm_d": float(row["global_width_mm_d"]),
                "hscc_width_mm_d": float(row["hscc_width_mm_d"]),
                "hscc_q": (
                    None if np.isnan(row["hscc_q"]) else float(row["hscc_q"])
                ),
                "test_pred_min": float(row["test_pred_min"]),
                "n_cal_basins": sum(1 for r in cal_records if r["tier"] == row["tier"]),
                "n_cal_pts": int(n_cal_tier[row["tier"]]),
            }
            for _, row in df.iterrows()
        },
        "success_criteria_pub": {
            "S1_prime_per_tier": bool(s1_pass_per_tier),
            "S2_prime_spread": s2,
            "S3_prime_global": s3,
            "S4_prime_neg_alarm": s4,
            "S5_prime_humid": (None if s5 is None else bool(s5)),
            "S6_prime_spread_reduction": s6,
            "max_neg_pred": float(max_neg),
            "eligible_tier_count": int(len(eligible_tiers)),
            "tier_count_with_test": int(len(df)),
        },
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
                        help="path to NH run dir for this fold (contains validation/ + test/)")
    parser.add_argument("--huc", required=True, type=str,
                        help="held-out HUC code, zero-padded e.g. '09'")
    parser.add_argument("--out_dir", default=None, type=str,
                        help="output dir (default: <run_dir>/_analysis)")
    args = parser.parse_args()
    rd = Path(args.run_dir).resolve()
    od = Path(args.out_dir).resolve() if args.out_dir else rd / "_analysis"
    run_fold_analysis(rd, args.huc.zfill(2), od)
