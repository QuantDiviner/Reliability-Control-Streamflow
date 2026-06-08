"""
exp004 Action 6 — Ensemble Gaussian baseline (head-to-head with HSCC).

Reuses the 3 trained LSTM checkpoints from Action 4 (seeds 42 / 137 / 2024)
to construct two parametric baselines, then evaluates per-tier coverage at
nominal 90% level on the test period:

  1. Ensemble Gaussian (per-day):
       μ(x) = mean of 3-seed LSTM predictions
       σ(x) = std  of 3-seed LSTM predictions   (epistemic uncertainty)
       PI = [μ - 1.645·σ, μ + 1.645·σ]
       — A bare-bones ensemble UQ baseline (Lakshminarayanan 2017 deep ensembles
         + Gaussian assumption).

  2. Residual-σ Gaussian (homoscedastic):
       μ(x) = mean of 3-seed LSTM predictions
       σ̂   = std of (obs - μ) over the calibration period (single number per basin)
       PI = [μ - 1.645·σ̂, μ + 1.645·σ̂]
       — A simple post-hoc residual baseline (Klotz 2022 §3.3 reference style).

Comparison to Section 5.4 / 5.6:
       HSCC         (proposed)         — per-tier conformal quantile, log-flow score
       Global CP    (Vovk 2005)        — single global conformal quantile
       Ensemble σ   (this script)      — epistemic UQ via 3-seed ensemble
       Residual σ̂   (this script)      — residual-based parametric baseline

Outputs:
  experiments/exp004/results/_3seed_aggregate/baseline_per_tier.csv
    per-tier coverage + width for all 4 methods (HSCC + Global + Ensemble + Residual)
  experiments/exp004/results/_3seed_aggregate/baseline_summary.json
    cross-tier spread + paragraph-ready numbers for §5.6 head-to-head paragraph

Usage:
  python experiments/exp004/scripts/ensemble_baseline_action6.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = ROOT / "experiments" / "exp004" / "results"
TIERS_FILE = ROOT / "experiments" / "exp004" / "basin_lists" / "gb_basin_tiers.csv"
OUT_DIR = RESULTS_ROOT / "_3seed_aggregate"

ALPHA = 0.10
Z_SCORE = 1.6448536269514722  # 90% two-sided ⇒ 1.645·σ each side

# 3-seed run dirs from Action 4
SEED_RUN_DIRS = {
    42: RESULTS_ROOT / "exp004_camels_gb_native_2604_214507",
    137: None,   # auto-detect
    2024: None,  # auto-detect
}


def find_run_dir(seed: int) -> Path:
    if SEED_RUN_DIRS[seed] is not None:
        return SEED_RUN_DIRS[seed]
    candidates = sorted(RESULTS_ROOT.glob(f"exp004_camels_gb_native_seed{seed}_*"))
    if not candidates:
        raise FileNotFoundError(f"no run_dir for seed={seed}")
    return candidates[-1]


def load_basin_series(run_dir: Path, period: str) -> dict:
    """Returns {basin: (obs, pred)} arrays from NH pickle."""
    sub = run_dir / period / "model_epoch030"
    p = sub / f"{period}_results.p"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    with open(p, "rb") as f:
        res = pickle.load(f)
    out = {}
    for bid, bd in res.items():
        if isinstance(bd, dict) and "1D" in bd:
            ds = bd["1D"]["xr"]
        elif hasattr(bd, "xr"):
            ds = bd.xr
        else:
            continue
        obs = ds["discharge_spec_obs"].values.flatten()
        pred = ds["discharge_spec_sim"].values.flatten()
        valid = ~(np.isnan(obs) | np.isnan(pred))
        out[str(bid).zfill(8)] = (obs[valid], pred[valid])
    return out


def coverage_width(obs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    cov = float(np.mean((obs >= lo) & (obs <= hi)))
    width = float(np.mean(np.maximum(hi - lo, 0.0)))
    return cov, width


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load 3-seed predictions (cal + test periods)
    seeds = [42, 137, 2024]
    cal_data: dict[int, dict] = {}
    test_data: dict[int, dict] = {}
    for s in seeds:
        rd = find_run_dir(s)
        cal_data[s] = load_basin_series(rd, "validation")
        test_data[s] = load_basin_series(rd, "test")
        print(f"  seed={s}: cal {len(cal_data[s])} basins, test {len(test_data[s])} basins")

    # Common basin set
    common_basins = sorted(set(cal_data[42]).intersection(*(set(d) for d in test_data.values())))
    print(f"\ncommon basins across 3 seeds (cal & test): {len(common_basins)}")

    # Load tier mapping (50-basin default; matches Action 4 3-seed)
    tiers_df = pd.read_csv(TIERS_FILE, dtype={"gauge_id": str})
    tiers_df["gauge_id"] = tiers_df["gauge_id"].str.zfill(8)
    tier_of = dict(zip(tiers_df["gauge_id"], tiers_df["tier"]))

    rows_per_basin = []
    for bid in common_basins:
        # Get arrays from each seed (truncate to common min length to align)
        obs_cal_seeds = [cal_data[s][bid][0] for s in seeds]
        pred_cal_seeds = [cal_data[s][bid][1] for s in seeds]
        obs_test_seeds = [test_data[s][bid][0] for s in seeds]
        pred_test_seeds = [test_data[s][bid][1] for s in seeds]

        n_cal = min(len(a) for a in obs_cal_seeds)
        n_test = min(len(a) for a in obs_test_seeds)
        if n_cal < 50 or n_test < 50:
            continue

        # Align by truncating each seed's series to common length
        obs_cal = obs_cal_seeds[0][:n_cal]
        pred_cal_stack = np.stack([p[:n_cal] for p in pred_cal_seeds])  # (3, n_cal)
        obs_test = obs_test_seeds[0][:n_test]
        pred_test_stack = np.stack([p[:n_test] for p in pred_test_seeds])  # (3, n_test)

        # === Method 1: Ensemble Gaussian ===
        mu_test = pred_test_stack.mean(axis=0)
        sig_test = pred_test_stack.std(axis=0, ddof=1)
        # Some basins may have near-zero ensemble σ (3 seeds agree); add tiny floor for stability
        sig_test = np.maximum(sig_test, 1e-3)
        lo_ens = mu_test - Z_SCORE * sig_test
        hi_ens = mu_test + Z_SCORE * sig_test
        cov_ens, w_ens = coverage_width(obs_test, lo_ens, hi_ens)

        # === Method 2: Residual-σ Gaussian (homoscedastic, per-basin) ===
        mu_cal = pred_cal_stack.mean(axis=0)
        residuals_cal = obs_cal - mu_cal
        sig_resid = float(np.std(residuals_cal, ddof=1))
        sig_resid = max(sig_resid, 1e-3)
        lo_res = mu_test - Z_SCORE * sig_resid
        hi_res = mu_test + Z_SCORE * sig_resid
        cov_res, w_res = coverage_width(obs_test, lo_res, hi_res)

        rows_per_basin.append({
            "basin_id": bid,
            "tier": tier_of.get(bid, "unknown"),
            "n_test": n_test,
            "ensemble_coverage": cov_ens,
            "ensemble_width": w_ens,
            "residual_coverage": cov_res,
            "residual_width": w_res,
            "ensemble_sigma_mean": float(sig_test.mean()),
            "residual_sigma": sig_resid,
        })

    df = pd.DataFrame(rows_per_basin)
    print(f"\nbaselines computed for {len(df)} basins; tier coverage:")

    tier_table = []
    for t in ["gb_drier_q4", "gb_mid", "gb_wet", "gb_montane"]:
        sub = df[df["tier"] == t]
        if sub.empty:
            continue
        tier_table.append({
            "tier": t,
            "n_basins": int(len(sub)),
            "ensemble_coverage_mean": float(sub["ensemble_coverage"].mean()),
            "ensemble_width_mean": float(sub["ensemble_width"].mean()),
            "residual_coverage_mean": float(sub["residual_coverage"].mean()),
            "residual_width_mean": float(sub["residual_width"].mean()),
        })
    tier_df = pd.DataFrame(tier_table)
    tier_df.to_csv(OUT_DIR / "baseline_per_tier.csv", index=False)
    print(tier_df.to_string(index=False))

    # Cross-tier spread (max-min) for both baselines
    ens_cov_arr = tier_df["ensemble_coverage_mean"].values
    res_cov_arr = tier_df["residual_coverage_mean"].values
    ens_spread = float(ens_cov_arr.max() - ens_cov_arr.min()) if len(ens_cov_arr) > 1 else 0.0
    res_spread = float(res_cov_arr.max() - res_cov_arr.min()) if len(res_cov_arr) > 1 else 0.0

    # Compare to HSCC + Global CP (from 3-seed aggregate)
    summary_3seed_path = OUT_DIR / "summary.json"
    hscc_spread, global_spread = None, None
    if summary_3seed_path.exists():
        with open(summary_3seed_path) as f:
            agg = json.load(f)
        hscc_spread = agg["spread_hscc_pp"]["mean"] / 100
        global_spread = agg["spread_global_pp"]["mean"] / 100

    paragraph_summary = {
        "n_basins_evaluated": int(len(df)),
        "tier_coverage_table": tier_df.to_dict(orient="records"),
        "cross_tier_spread_pp": {
            "global_cp": (global_spread * 100) if global_spread is not None else None,
            "hscc_3seed_mean": (hscc_spread * 100) if hscc_spread is not None else None,
            "ensemble_gaussian": ens_spread * 100,
            "residual_gaussian": res_spread * 100,
        },
        "method_comparison": (
            "Cross-tier coverage spread (lower = more uniform per-tier reliability):\n"
            f"  Global CP        : {global_spread*100:.2f}pp\n" if global_spread else ""
        ) + (
            f"  HSCC (3-seed)    : {hscc_spread*100:.2f}pp\n" if hscc_spread else ""
        ) + (
            f"  Ensemble Gaussian: {ens_spread*100:.2f}pp\n"
            f"  Residual Gaussian: {res_spread*100:.2f}pp\n"
        ),
        "interpretation_notes": [
            "Ensemble Gaussian uses 3-seed epistemic σ; tends to be very narrow "
            "(seeds agree on most days), so intervals can under-cover.",
            "Residual Gaussian uses fixed per-basin residual σ from cal period; "
            "ignores heteroscedasticity but is a fair single-distribution baseline.",
            "HSCC's advantage over both Gaussian baselines is distribution-free "
            "calibration via conformal quantiles on log-flow score.",
        ],
    }
    with open(OUT_DIR / "baseline_summary.json", "w") as f:
        json.dump(paragraph_summary, f, indent=2)

    print(f"\n=== HEAD-TO-HEAD CROSS-TIER SPREAD ===")
    print(paragraph_summary["method_comparison"])

    print(f"wrote {OUT_DIR}/baseline_per_tier.csv")
    print(f"wrote {OUT_DIR}/baseline_summary.json")


if __name__ == "__main__":
    main()
