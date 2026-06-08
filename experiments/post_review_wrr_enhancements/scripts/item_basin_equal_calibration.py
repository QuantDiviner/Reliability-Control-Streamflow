"""Seed-42 basin-equal calibration sanity check for Global CP.

This addresses the review concern that Global CP estimates its calibration
quantile by pooling basin-days, while tier coverage is reported basin-equal.
The check recomputes the Global CP quantile with equal total calibration weight
per basin and compares tier coverages against the original day-pooled quantile.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "experiments/exp002/results/exp002_camels_us_temporal_2504_155058"
ATTR_FILE = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
OUT = ROOT / "experiments/post_review_wrr_enhancements/results/item_basin_equal_calibration.json"

ALPHA = 0.10
EPS = 0.01


def load_attributes() -> dict[str, dict[str, float]]:
    attrs: dict[str, dict[str, float]] = {}
    with ATTR_FILE.open() as f:
        hdr = f.readline().strip().split(";")
        ai_i, sn_i = hdr.index("aridity"), hdr.index("frac_snow")
        for line in f:
            cols = line.strip().split(";")
            if cols and cols[0]:
                attrs[cols[0].zfill(8)] = {
                    "aridity": float(cols[ai_i]),
                    "frac_snow": float(cols[sn_i]),
                }
    return attrs


def assign_tier(aridity: float, frac_snow: float) -> str:
    if frac_snow >= 0.40:
        return "snow"
    if aridity > 1.50:
        return "dry"
    if aridity > 1.00:
        return "semi_arid"
    return "humid"


def extract_series(results: dict, basin: str) -> tuple[np.ndarray, np.ndarray]:
    ds = results[basin]["1D"]["xr"]
    obs = ds["QObs(mm/d)_obs"].values.flatten()
    pred = ds["QObs(mm/d)_sim"].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def log_score(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(
        np.log(np.maximum(obs + EPS, 1e-6))
        - np.log(np.maximum(pred + EPS, 1e-6))
    )


def split_cp_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order] / weights.sum()
    cdf = np.cumsum(weights)
    return float(values[np.searchsorted(cdf, q, side="left")])


def coverage(obs: np.ndarray, pred: np.ndarray, qhat: float) -> float:
    return float(np.mean(log_score(obs, pred) <= qhat))


def tier_coverage(records: list[dict], qhat: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for tier in ["dry", "semi_arid", "humid", "snow"]:
        tier_records = [r for r in records if r["tier"] == tier]
        out[tier] = float(
            np.mean([coverage(r["test_obs"], r["test_pred"], qhat) for r in tier_records])
        )
    return out


def main() -> None:
    with (RUN_DIR / "validation/model_epoch030/validation_results.p").open("rb") as f:
        val = pickle.load(f)
    with (RUN_DIR / "test/model_epoch030/test_results.p").open("rb") as f:
        test = pickle.load(f)

    attrs = load_attributes()
    records = []
    all_scores = []
    basin_equal_weights = []
    for basin in val:
        if basin not in test:
            continue
        cal_obs, cal_pred = extract_series(val, basin)
        test_obs, test_pred = extract_series(test, basin)
        if len(cal_obs) < 100 or len(test_obs) < 100:
            continue
        gid = str(basin).zfill(8)
        attr = attrs[gid]
        scores = log_score(cal_obs, cal_pred)
        records.append({
            "basin": gid,
            "tier": assign_tier(attr["aridity"], attr["frac_snow"]),
            "test_obs": test_obs,
            "test_pred": test_pred,
        })
        all_scores.append(scores)
        basin_equal_weights.append(np.full(len(scores), 1.0 / len(scores)))

    scores = np.concatenate(all_scores)
    weights = np.concatenate(basin_equal_weights)
    q_day = split_cp_quantile(scores)
    q_basin_equal = weighted_quantile(scores, weights, 1 - ALPHA)

    day_cov = tier_coverage(records, q_day)
    be_cov = tier_coverage(records, q_basin_equal)

    result = {
        "analysis": "seed42_basin_equal_calibration_global_cp",
        "n_basins": len(records),
        "day_pooled_q": q_day,
        "basin_equal_q": q_basin_equal,
        "q_ratio_basin_equal_to_day_pooled": q_basin_equal / q_day,
        "day_pooled_tier_coverage": day_cov,
        "basin_equal_tier_coverage": be_cov,
        "day_pooled_spread_pp": (max(day_cov.values()) - min(day_cov.values())) * 100,
        "basin_equal_spread_pp": (max(be_cov.values()) - min(be_cov.values())) * 100,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
