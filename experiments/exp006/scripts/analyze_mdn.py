"""exp006 MDN baseline analysis.

Reads NeuralHydrology UncertaintyTester output (test_results.p, validation_results.p)
and computes per-tier reliability + sharpness metrics for head-to-head comparison
against HSCC (exp002 results) and global split-CP.

Pipeline (per period, default test):
  1. Load <run_dir>/<period>/model_epoch030/<period>_results.p
     Schema: { basin: { '1D': xr.Dataset(QObs(mm/d)_obs, QObs(mm/d)_sim) } }
     where _sim has dims (date, time_step, samples) with n_samples=200.
  2. For each basin × timestep:
       lower = np.quantile(samples, alpha/2)        # alpha=0.10 → 5% quantile
       upper = np.quantile(samples, 1 - alpha/2)    # 95% quantile
       width = upper - lower
       hit   = lower <= obs <= upper
  3. Join with basin_tiers.csv (gauge_id → tier).
  4. Aggregate per tier: coverage, mean width, NSE (point estimate from samples mean).
  5. Outputs:
       metrics_per_tier.json   — full per-tier metrics
       mdn_intervals.csv       — per-basin per-timestep (basin, date, obs, lower, upper, mean, hit)
       pareto_data.csv         — 4-tier Pareto points (reliability_dev, mean_width)

Usage:
  python analyze_mdn.py --run_dir <run_dir> --tiers <basin_tiers.csv> --period test \\
                        --epoch 30 --alpha 0.10 --out_dir <run_dir>/_analysis

Authority: PROJECT_CHARTER §2.2 (alpha=0.10), exp006/plan.md §2.3 + §3.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(run_dir: Path, period: str, epoch: int) -> dict:
    p = run_dir / period / f"model_epoch{epoch:03d}" / f"{period}_results.p"
    if not p.exists():
        raise FileNotFoundError(f"Results pickle not found: {p}")
    with p.open("rb") as fp:
        return pickle.load(fp)


def basin_intervals(basin_id: str, ds_dict: dict, alpha: float) -> pd.DataFrame:
    """Extract obs + 90% interval from a basin's xarray Dataset."""
    if "1D" not in ds_dict:
        return pd.DataFrame()
    ds = ds_dict["1D"]["xr"]
    target = "QObs(mm/d)"
    obs_name = f"{target}_obs"
    sim_name = f"{target}_sim"
    if obs_name not in ds or sim_name not in ds:
        return pd.DataFrame()

    # obs: (date, time_step) → predict_last_n=1 so time_step=1; squeeze
    obs = ds[obs_name].isel(time_step=-1).values
    sim = ds[sim_name].isel(time_step=-1).values  # (date, samples)
    dates = ds["date"].values

    lower = np.nanquantile(sim, alpha / 2.0, axis=-1)
    upper = np.nanquantile(sim, 1.0 - alpha / 2.0, axis=-1)
    mean_pred = np.nanmean(sim, axis=-1)
    width = upper - lower

    df = pd.DataFrame({
        "basin": basin_id,
        "date": dates,
        "obs": obs,
        "mean": mean_pred,
        "lower": lower,
        "upper": upper,
        "width": width,
    })
    df = df.dropna(subset=["obs"])
    df["hit"] = ((df["obs"] >= df["lower"]) & (df["obs"] <= df["upper"])).astype(int)
    return df


def basin_nse(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    obs = df["obs"].to_numpy()
    sim = df["mean"].to_numpy()
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--tiers", required=True, type=Path,
                    help="basin_tiers.csv with columns gauge_id,tier")
    ap.add_argument("--period", default="test", choices=["validation", "test"])
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp006-analysis] loading {args.period}_results.p (epoch={args.epoch})")
    results = load_results(args.run_dir, args.period, args.epoch)
    print(f"[exp006-analysis] basins in pickle: {len(results)}")

    tiers = pd.read_csv(args.tiers, dtype={"gauge_id": str})
    tiers["gauge_id"] = tiers["gauge_id"].str.zfill(8)
    tier_map = dict(zip(tiers["gauge_id"], tiers["tier"]))

    rows = []
    nse_rows = []
    for basin, ds_dict in results.items():
        bid = str(basin).zfill(8)
        df = basin_intervals(bid, ds_dict, args.alpha)
        if df.empty:
            continue
        df["tier"] = tier_map.get(bid, "unknown")
        rows.append(df)
        nse_rows.append({"basin": bid, "tier": df["tier"].iloc[0], "nse": basin_nse(df)})

    if not rows:
        raise RuntimeError("No basins yielded interval data — check pickle schema.")

    full = pd.concat(rows, ignore_index=True)
    full.to_csv(args.out_dir / "mdn_intervals.csv", index=False)
    print(f"[exp006-analysis] mdn_intervals.csv  rows={len(full)}  basins={full['basin'].nunique()}")

    nse_df = pd.DataFrame(nse_rows)
    nse_df.to_csv(args.out_dir / "basin_nse.csv", index=False)

    target_cov = 1.0 - args.alpha
    per_tier = (
        full.groupby("tier")
        .agg(
            n_basins=("basin", "nunique"),
            n_timesteps=("hit", "size"),
            coverage=("hit", "mean"),
            mean_width=("width", "mean"),
            median_width=("width", "median"),
        )
        .reset_index()
    )
    per_tier["coverage_dev"] = (per_tier["coverage"] - target_cov).abs()
    per_tier["target_coverage"] = target_cov

    nse_per_tier = nse_df.groupby("tier")["nse"].mean().reset_index().rename(columns={"nse": "mean_basin_nse"})
    per_tier = per_tier.merge(nse_per_tier, on="tier", how="left")

    overall = pd.DataFrame([{
        "tier": "ALL",
        "n_basins": full["basin"].nunique(),
        "n_timesteps": len(full),
        "coverage": full["hit"].mean(),
        "mean_width": full["width"].mean(),
        "median_width": full["width"].median(),
        "coverage_dev": abs(full["hit"].mean() - target_cov),
        "target_coverage": target_cov,
        "mean_basin_nse": nse_df["nse"].mean(),
    }])
    summary = pd.concat([per_tier, overall], ignore_index=True)
    summary.to_csv(args.out_dir / "pareto_data.csv", index=False)

    spread_global = (per_tier["coverage"].max() - per_tier["coverage"].min()) * 100.0
    metrics = {
        "alpha": args.alpha,
        "target_coverage": target_cov,
        "n_samples_per_step": int(np.unique(
            np.array([list(v["1D"]["xr"].dims.values()) for _, v in results.items() if "1D" in v])[..., -1]
        )[0]) if results else None,
        "epoch": args.epoch,
        "period": args.period,
        "per_tier": {
            row["tier"]: {
                "n_basins": int(row["n_basins"]),
                "n_timesteps": int(row["n_timesteps"]),
                "coverage": float(row["coverage"]),
                "mean_width": float(row["mean_width"]),
                "median_width": float(row["median_width"]),
                "coverage_deviation_pp": float(row["coverage_dev"] * 100.0),
                "mean_basin_nse": float(row["mean_basin_nse"]) if pd.notna(row["mean_basin_nse"]) else None,
            }
            for _, row in per_tier.iterrows()
        },
        "overall": {
            "coverage": float(overall["coverage"].iloc[0]),
            "mean_width": float(overall["mean_width"].iloc[0]),
            "mean_basin_nse": float(overall["mean_basin_nse"].iloc[0]) if pd.notna(overall["mean_basin_nse"].iloc[0]) else None,
            "tier_coverage_spread_pp": float(spread_global),
        },
    }

    out_json = args.out_dir / "metrics_per_tier.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"[exp006-analysis] metrics_per_tier.json  spread={spread_global:.2f}pp")
    print(f"[exp006-analysis] outputs: {args.out_dir}/{{metrics_per_tier.json, mdn_intervals.csv, pareto_data.csv, basin_nse.csv}}")


if __name__ == "__main__":
    main()
