#!/usr/bin/env python3
"""Run vanilla global residual-CQR on CAMELS-US across exp009 seeds.

The critical distinction from exp011 is that this script computes one pooled
conformal correction, Q_hat_global, per seed. It does not compute per-tier
corrections.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor


ROOT = Path(__file__).resolve().parents[3]
ALPHA = 0.10
EPS = 0.01
TIERS = ["dry", "semi_arid", "humid", "snow"]
ATTR_FILE = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
EXP009_RESULTS = ROOT / "experiments/exp009/results"
OUT_ROOT = ROOT / "experiments/exp_global_cqr/results"

SEED_RUNS = {
    42: EXP009_RESULTS / "seed042_run",
    137: EXP009_RESULTS / "seed137_run",
    2024: EXP009_RESULTS / "seed2024_run",
    1337: EXP009_RESULTS / "seed1337_run",
    7: EXP009_RESULTS / "seed007_run",
}


def load_attrs() -> dict[str, dict[str, float]]:
    attrs: dict[str, dict[str, float]] = {}
    with open(ATTR_FILE) as f:
        header = f.readline().strip().split(";")
        aridity_i = header.index("aridity")
        snow_i = header.index("frac_snow")
        p_mean_i = header.index("p_mean")
        for line in f:
            cols = line.strip().split(";")
            if not cols or not cols[0]:
                continue
            attrs[cols[0].zfill(8)] = {
                "aridity": float(cols[aridity_i]),
                "frac_snow": float(cols[snow_i]),
                "p_mean": float(cols[p_mean_i]),
            }
    return attrs


def assign_tier(aridity: float, frac_snow: float) -> str:
    if frac_snow >= 0.40:
        return "snow"
    if aridity > 1.5:
        return "dry"
    if aridity > 1.0:
        return "semi_arid"
    return "humid"


def latest_result_file(run_dir: Path, split: str, filename: str) -> Path:
    candidates = sorted((run_dir / split).glob(f"model_epoch*/{filename}"))
    if not candidates:
        raise FileNotFoundError(f"No {filename} under {run_dir / split}")
    return candidates[-1]


def extract(results: dict, basin: str) -> tuple[np.ndarray, np.ndarray]:
    ds = results[basin]["1D"]["xr"]
    obs = ds["QObs(mm/d)_obs"].values.flatten()
    pred = ds["QObs(mm/d)_sim"].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def make_features(pred: np.ndarray, aridity: float, frac_snow: float, p_mean: float) -> np.ndarray:
    n = len(pred)
    return np.column_stack(
        [
            pred,
            np.log(np.maximum(pred + EPS, 1e-6)),
            np.full(n, aridity),
            np.full(n, frac_snow),
            np.full(n, p_mean),
        ]
    )


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


def winkler_score(obs: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    width = hi - lo
    below = obs < lo
    above = obs > hi
    penalty = np.zeros_like(obs, dtype=float)
    penalty[below] = (2.0 / alpha) * (lo[below] - obs[below])
    penalty[above] = (2.0 / alpha) * (obs[above] - hi[above])
    return float(np.mean(width + penalty))


def load_seed_records(seed: int, attrs: dict[str, dict[str, float]]) -> list[dict]:
    run_dir = SEED_RUNS[seed]
    val_path = latest_result_file(run_dir, "validation", "validation_results.p")
    test_path = latest_result_file(run_dir, "test", "test_results.p")
    val = pickle.load(open(val_path, "rb"))
    test = pickle.load(open(test_path, "rb"))

    records = []
    for basin in val.keys():
        if basin not in test:
            continue
        basin_id = str(basin).zfill(8)
        if basin_id not in attrs:
            continue
        cal_obs, cal_pred = extract(val, basin)
        test_obs, test_pred = extract(test, basin)
        if len(cal_obs) < 100 or len(test_obs) < 100:
            continue
        attr = attrs[basin_id]
        records.append(
            {
                "basin": basin_id,
                "tier": assign_tier(attr["aridity"], attr["frac_snow"]),
                "aridity": attr["aridity"],
                "frac_snow": attr["frac_snow"],
                "p_mean": attr["p_mean"],
                "cal_features": make_features(
                    cal_pred, attr["aridity"], attr["frac_snow"], attr["p_mean"]
                ),
                "cal_resid": cal_obs - cal_pred,
                "test_features": make_features(
                    test_pred, attr["aridity"], attr["frac_snow"], attr["p_mean"]
                ),
                "test_obs": test_obs,
                "test_pred": test_pred,
            }
        )
    return records


def fit_global_quantile_regressors(
    records: list[dict], max_train_points: int, random_seed: int
) -> tuple[QuantileRegressor, QuantileRegressor, float]:
    x_cal = np.vstack([record["cal_features"] for record in records])
    resid_cal = np.concatenate([record["cal_resid"] for record in records])

    x_train = x_cal
    resid_train = resid_cal
    if len(resid_train) > max_train_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(resid_train), max_train_points, replace=False)
        x_train = x_train[idx]
        resid_train = resid_train[idx]

    qr_lo = QuantileRegressor(quantile=ALPHA / 2, alpha=1e-3, solver="highs").fit(
        x_train, resid_train
    )
    qr_hi = QuantileRegressor(quantile=1 - ALPHA / 2, alpha=1e-3, solver="highs").fit(
        x_train, resid_train
    )

    lo_cal = qr_lo.predict(x_cal)
    hi_cal = qr_hi.predict(x_cal)
    conformity = np.maximum(lo_cal - resid_cal, resid_cal - hi_cal)
    q_hat_global = finite_sample_quantile(conformity, ALPHA)
    return qr_lo, qr_hi, q_hat_global


def evaluate_seed(seed: int, max_train_points: int) -> dict:
    attrs = load_attrs()
    records = load_seed_records(seed, attrs)
    if not records:
        raise RuntimeError(f"No usable records for seed {seed}")

    out_dir = OUT_ROOT / f"seed{seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    qr_lo, qr_hi, q_hat_global = fit_global_quantile_regressors(
        records, max_train_points=max_train_points, random_seed=seed
    )

    rows = []
    for record in records:
        q_lo = qr_lo.predict(record["test_features"])
        q_hi = qr_hi.predict(record["test_features"])
        pred = record["test_pred"]
        obs = record["test_obs"]
        lo = np.maximum(pred + q_lo - q_hat_global, 0.0)
        hi = pred + q_hi + q_hat_global
        coverage = float(np.mean((obs >= lo) & (obs <= hi)))
        rows.append(
            {
                "seed": seed,
                "basin": record["basin"],
                "tier": record["tier"],
                "aridity": record["aridity"],
                "frac_snow": record["frac_snow"],
                "n_test": int(len(obs)),
                "coverage": coverage,
                "coverage_eps": coverage - (1 - ALPHA),
                "mean_pi_width": float(np.mean(hi - lo)),
                "winkler_score": winkler_score(obs, lo, hi, ALPHA),
                "Q_hat_global": q_hat_global,
            }
        )

    per_basin = pd.DataFrame(rows)
    per_tier_rows = []
    for tier in TIERS:
        sub = per_basin[per_basin["tier"] == tier]
        if sub.empty:
            continue
        per_tier_rows.append(
            {
                "seed": seed,
                "tier": tier,
                "n_basins": int(len(sub)),
                "mean_coverage": float(sub["coverage"].mean()),
                "median_coverage": float(sub["coverage"].median()),
                "std_coverage": float(sub["coverage"].std(ddof=1)),
                "mean_pi_width": float(sub["mean_pi_width"].mean()),
                "median_pi_width": float(sub["mean_pi_width"].median()),
                "mean_winkler_score": float(sub["winkler_score"].mean()),
                "Q_hat_global": q_hat_global,
            }
        )
    per_tier = pd.DataFrame(per_tier_rows)

    spread_pp = float((per_tier["mean_coverage"].max() - per_tier["mean_coverage"].min()) * 100)
    summary = {
        "experiment": "exp_global_cqr",
        "method": "vanilla global residual-CQR (single pooled Q_hat_global)",
        "seed": seed,
        "alpha": ALPHA,
        "n_basins_analyzed": int(len(per_basin)),
        "n_tiers_reported": int(len(per_tier)),
        "Q_hat_global": q_hat_global,
        "overall": {
            "mean_coverage": float(per_basin["coverage"].mean()),
            "mean_pi_width": float(per_basin["mean_pi_width"].mean()),
            "mean_winkler_score": float(per_basin["winkler_score"].mean()),
            "tier_coverage_spread_pp": spread_pp,
        },
        "per_tier": per_tier.to_dict(orient="records"),
        "success_criteria": {
            "S1_single_global_qhat": True,
            "S2_all_tiers_reported": bool(set(TIERS).issubset(set(per_tier["tier"]))),
        },
    }

    per_basin.to_csv(out_dir / "global_cqr_per_basin.csv", index=False)
    per_tier.to_csv(out_dir / "global_cqr_per_tier.csv", index=False)
    (out_dir / "global_cqr_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def aggregate(summaries: list[dict]) -> dict:
    agg_dir = OUT_ROOT / "_5seed_aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    tier_rows = []
    for summary in summaries:
        seed = summary["seed"]
        seed_rows.append(
            {
                "seed": seed,
                "n_basins": summary["n_basins_analyzed"],
                "Q_hat_global": summary["Q_hat_global"],
                "mean_coverage": summary["overall"]["mean_coverage"],
                "tier_coverage_spread_pp": summary["overall"]["tier_coverage_spread_pp"],
                "mean_pi_width": summary["overall"]["mean_pi_width"],
                "mean_winkler_score": summary["overall"]["mean_winkler_score"],
            }
        )
        tier_rows.extend(summary["per_tier"])

    seed_df = pd.DataFrame(seed_rows)
    tier_df = pd.DataFrame(tier_rows)
    seed_df.to_csv(agg_dir / "per_seed_summary.csv", index=False)
    tier_df.to_csv(agg_dir / "per_seed_tier_summary.csv", index=False)

    tier_summary = (
        tier_df.groupby("tier")
        .agg(
            n_basins=("n_basins", "mean"),
            mean_coverage=("mean_coverage", "mean"),
            std_coverage=("mean_coverage", "std"),
            mean_pi_width=("mean_pi_width", "mean"),
            mean_winkler_score=("mean_winkler_score", "mean"),
        )
        .reset_index()
    )
    tier_summary.to_csv(agg_dir / "tier_summary.csv", index=False)

    aggregate_json = {
        "experiment": "exp_global_cqr",
        "method": "vanilla global residual-CQR (single pooled Q_hat_global)",
        "alpha": ALPHA,
        "seeds": [int(summary["seed"]) for summary in summaries],
        "n_seeds": len(summaries),
        "overall_cross_seed": {
            "mean_coverage": float(seed_df["mean_coverage"].mean()),
            "std_coverage": float(seed_df["mean_coverage"].std(ddof=1))
            if len(seed_df) > 1
            else 0.0,
            "tier_coverage_spread_pp": float(seed_df["tier_coverage_spread_pp"].mean()),
            "tier_coverage_spread_pp_std": float(seed_df["tier_coverage_spread_pp"].std(ddof=1))
            if len(seed_df) > 1
            else 0.0,
            "mean_pi_width": float(seed_df["mean_pi_width"].mean()),
            "mean_winkler_score": float(seed_df["mean_winkler_score"].mean()),
        },
        "per_tier_cross_seed": tier_summary.to_dict(orient="records"),
        "success_criteria": {
            "S1_single_global_qhat_per_seed": True,
            "S2_all_tiers_reported": bool(set(TIERS).issubset(set(tier_summary["tier"]))),
            "S3_seed_runs_completed": len(summaries),
            "S4_cross_seed_summary_complete": True,
            "S5_ready_for_pareto_row": len(summaries) >= 1,
        },
    }
    (agg_dir / "aggregate.json").write_text(json.dumps(aggregate_json, indent=2))
    return aggregate_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2024, 1337, 7])
    parser.add_argument("--max-train-points", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in args.seeds:
        if seed not in SEED_RUNS:
            raise ValueError(f"Unknown seed {seed}; expected one of {sorted(SEED_RUNS)}")
        print(f"=== exp_global_cqr seed {seed} ===", flush=True)
        summary = evaluate_seed(seed, max_train_points=args.max_train_points)
        overall = summary["overall"]
        print(
            f"seed {seed}: coverage={overall['mean_coverage']:.4f}, "
            f"spread={overall['tier_coverage_spread_pp']:.2f} pp, "
            f"width={overall['mean_pi_width']:.3f}, Qhat={summary['Q_hat_global']:.4f}",
            flush=True,
        )
        summaries.append(summary)

    headline = aggregate(summaries)["overall_cross_seed"]
    print("=== aggregate ===", flush=True)
    print(
        f"coverage={headline['mean_coverage']:.4f}, "
        f"spread={headline['tier_coverage_spread_pp']:.2f} pp, "
        f"width={headline['mean_pi_width']:.3f}, "
        f"winkler={headline['mean_winkler_score']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
