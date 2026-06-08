"""
Post-review item 8: UMAL marginal CRPS from the stored predictive samples.

Reviewer concern (20250528 报告2, #8): UMAL is a density model whose most native
score is CRPS, yet Table 3 leaves it "not reported". The test_results.p pickles store
200 predictive samples per basin-day in mm/d, so CRPS is a pure post-processing read
of existing outputs (no retraining, no GPU).

CRPS is computed on UMAL's native 200-sample predictive distribution, in mm/d, and
aggregated the same way exp011 reports marginal CRPS for the interval methods:
per-basin mean over test days, then an unweighted (basin-equal) mean across basins.
This keeps the UMAL cell comparable to the Global CP / HSCC / CQR cells in Table 3.

Outputs:
  experiments/post_review_wrr_enhancements/results/item8_umal_crps.json
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP006 = ROOT / "experiments" / "exp006" / "results"
OUT = ROOT / "experiments" / "post_review_wrr_enhancements" / "results" / "item8_umal_crps.json"

# Same 3 seed runs that feed the UMAL aggregate (exp006/results/_3seed_aggregate/aggregate.json)
SEED_RUNS = {
    42: "exp006_mdn_camels_us_2904_155205",
    1337: "exp006_mdn_camels_us_seed1337_3004_223135",
    2024: "exp006_mdn_camels_us_seed2024_0105_123958",
}
TARGET = "QObs(mm/d)"


def crps_ensemble(sim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-timestep empirical CRPS for an ensemble, vectorized, O(T m log m).

    sim: (T, m) predictive samples (mm/d). obs: (T,) observations (mm/d).
    Uses CRPS = E|X-y| - 0.5 E|X-X'| with the sorted-sample closed form for the
    second term:  E|X-X'| = (2/m^2) * sum_i (2i - m - 1) x_(i).
    """
    m = sim.shape[1]
    s = np.sort(sim, axis=1)
    i = np.arange(1, m + 1)
    coef = (2 * i - m - 1).astype(float)  # (m,)
    mean_abs_diff = (2.0 / m**2) * (s * coef).sum(axis=1)  # (T,)
    term1 = np.abs(s - obs[:, None]).mean(axis=1)  # (T,)
    return term1 - 0.5 * mean_abs_diff


def basin_series(ds):
    obs = ds[f"{TARGET}_obs"].values.squeeze()        # (T,)
    sim = ds[f"{TARGET}_sim"].values.squeeze()         # (T, m)
    if sim.ndim == 1:
        sim = sim[:, None]
    return obs, sim


def run_seed(run_dir: Path) -> dict:
    pkl = run_dir / "test" / "model_epoch030" / "test_results.p"
    with open(pkl, "rb") as f:
        res = pickle.load(f)
    per_basin = []
    for basin, freqs in res.items():
        ds = freqs["1D"]["xr"]
        obs, sim = basin_series(ds)
        valid = np.isfinite(obs) & np.isfinite(sim).all(axis=1)
        if valid.sum() < 10:
            continue
        crps_t = crps_ensemble(sim[valid], obs[valid])
        per_basin.append(float(np.mean(crps_t)))
    arr = np.array(per_basin)
    return {
        "n_basins": int(arr.size),
        "marginal_crps": float(arr.mean()),       # basin-equal mean of per-basin mean CRPS, mm/d
        "median_basin_crps": float(np.median(arr)),
    }


def main() -> None:
    per_seed = {}
    for seed, name in SEED_RUNS.items():
        rd = EXP006 / name
        per_seed[seed] = run_seed(rd)
        print(f"seed {seed}: n={per_seed[seed]['n_basins']}  marginal_crps={per_seed[seed]['marginal_crps']:.4f} mm/d")
    vals = np.array([per_seed[s]["marginal_crps"] for s in SEED_RUNS])
    out = {
        "units": "mm/d",
        "predictive_object": "UMAL native 200-sample predictive distribution",
        "aggregation": "per-basin mean over test days, then basin-equal mean across basins (matches exp011 marginal CRPS)",
        "n_seeds": len(SEED_RUNS),
        "per_seed": {str(s): per_seed[s] for s in SEED_RUNS},
        "umal_marginal_crps_seed42": per_seed[42]["marginal_crps"],
        "umal_marginal_crps_mean": float(vals.mean()),
        "umal_marginal_crps_std": float(vals.std(ddof=1)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n3-seed UMAL marginal CRPS = {vals.mean():.4f} ± {vals.std(ddof=1):.4f} mm/d")
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
