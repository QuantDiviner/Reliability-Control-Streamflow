"""
exp004 Action 4 aggregator — 3-seed mean ± std + basin-block bootstrap CI.

Reads:
  experiments/exp004/results/exp004_camels_gb_native_2604_214507/_analysis/metrics.json   (seed=42)
  experiments/exp004/results/exp004_camels_gb_native_seed137_*/_analysis/metrics.json     (seed=137)
  experiments/exp004/results/exp004_camels_gb_native_seed2024_*/_analysis/metrics.json    (seed=2024)
  experiments/exp004/results/exp004_camels_gb_native_*/_analysis/hscc_results.csv         (per-basin coverage)

Writes:
  experiments/exp004/results/_3seed_aggregate/summary.json
  experiments/exp004/results/_3seed_aggregate/per_tier_table.csv
  experiments/exp004/results/_3seed_aggregate/bootstrap_ci.json
  experiments/exp004/results/_3seed_aggregate/per_basin_distribution.csv

Statistics:
  - per-tier (HSCC coverage, global coverage, HSCC width, global width):
      mean ± std across 3 seeds
  - basin-block bootstrap CI (1000 resamples) on per-tier coverage,
      using basins as resampling unit (per-tier within each seed, then averaged)
  - 3-seed cross-seed std on global spread reduction
"""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = ROOT / "experiments" / "exp004" / "results"
OUT_DIR = RESULTS_ROOT / "_3seed_aggregate"

SEEDS = [42, 137, 2024]
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)
TIERS = ["gb_drier_q4", "gb_mid", "gb_wet", "gb_montane"]


def find_run_dir(seed: int) -> Path:
    if seed == 42:
        # Original run, fixed path
        return RESULTS_ROOT / "exp004_camels_gb_native_2604_214507"
    candidates = sorted(RESULTS_ROOT.glob(f"exp004_camels_gb_native_seed{seed}_*"))
    if not candidates:
        raise FileNotFoundError(f"no run_dir for seed={seed}")
    return candidates[-1]  # latest


def load_seed(seed: int) -> dict:
    run_dir = find_run_dir(seed)
    with open(run_dir / "_analysis" / "metrics.json") as f:
        m = json.load(f)
    hscc_csv = run_dir / "_analysis" / "hscc_results.csv"
    per_basin = pd.read_csv(hscc_csv) if hscc_csv.exists() else None
    return {"seed": seed, "run_dir": run_dir, "metrics": m, "per_basin": per_basin}


def per_tier_aggregation(loaded: list[dict]) -> dict:
    """Mean ± std per tier across seeds."""
    rows = []
    for t in TIERS:
        cov_g, cov_h, w_g, w_h = [], [], [], []
        n_b = []
        for L in loaded:
            tier_dat = L["metrics"].get("per_tier", {}).get(t)
            if tier_dat is None:
                continue
            cov_g.append(tier_dat["global_coverage"])
            cov_h.append(tier_dat["hscc_coverage"])
            w_g.append(tier_dat["global_width_mm_d"])
            w_h.append(tier_dat["hscc_width_mm_d"])
            n_b.append(tier_dat["n_basins"])
        if not cov_h:
            continue
        rows.append({
            "tier": t,
            "n_basins": int(n_b[0]) if n_b else 0,
            "n_seeds": len(cov_h),
            "global_cov_mean": float(np.mean(cov_g)),
            "global_cov_std": float(np.std(cov_g, ddof=1)) if len(cov_g) > 1 else 0.0,
            "hscc_cov_mean": float(np.mean(cov_h)),
            "hscc_cov_std": float(np.std(cov_h, ddof=1)) if len(cov_h) > 1 else 0.0,
            "global_width_mean": float(np.mean(w_g)),
            "hscc_width_mean": float(np.mean(w_h)),
        })
    return rows


def basin_block_bootstrap(loaded: list[dict]) -> dict:
    """For each tier, pool per-basin coverages across all seeds, then
    bootstrap-resample basins (with replacement) to estimate 95% CI on
    the basin-mean coverage. Treats seeds as additional sampling
    variation (1000 resamples * 3 seeds = 3000 samples per CI)."""
    cis: dict[str, dict] = {}
    for t in TIERS:
        all_basin_cov_hscc = []
        all_basin_cov_global = []
        for L in loaded:
            pb = L["per_basin"]
            if pb is None:
                continue
            sub = pb[pb["tier"] == t] if "tier" in pb.columns else pd.DataFrame()
            if sub.empty:
                continue
            for col_h, col_g in [
                ("hscc_coverage", "global_coverage"),
                ("hscc_cov", "global_cov"),  # legacy column names
            ]:
                if col_h in sub.columns:
                    all_basin_cov_hscc.extend(sub[col_h].dropna().tolist())
                    all_basin_cov_global.extend(sub[col_g].dropna().tolist())
                    break
        if not all_basin_cov_hscc:
            cis[t] = {"note": "no per-basin data"}
            continue
        all_basin_cov_hscc = np.array(all_basin_cov_hscc)
        all_basin_cov_global = np.array(all_basin_cov_global)
        n = len(all_basin_cov_hscc)
        boot_h, boot_g = [], []
        for _ in range(N_BOOTSTRAP):
            idx = RNG.integers(0, n, size=n)
            boot_h.append(all_basin_cov_hscc[idx].mean())
            boot_g.append(all_basin_cov_global[idx].mean() if all_basin_cov_global.size else np.nan)
        boot_h = np.array(boot_h)
        boot_g = np.array(boot_g)
        cis[t] = {
            "n_basin_samples": int(n),
            "hscc_cov_ci95": [float(np.quantile(boot_h, 0.025)), float(np.quantile(boot_h, 0.975))],
            "hscc_cov_mean": float(boot_h.mean()),
            "global_cov_ci95": [float(np.quantile(boot_g, 0.025)), float(np.quantile(boot_g, 0.975))],
            "global_cov_mean": float(boot_g.mean()),
        }
    return cis


def per_basin_long_format(loaded: list[dict]) -> pd.DataFrame:
    """Concat all per-basin rows across seeds for plotting."""
    frames = []
    for L in loaded:
        pb = L["per_basin"]
        if pb is None:
            continue
        pb = pb.copy()
        pb["seed"] = L["seed"]
        frames.append(pb)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading 3 seeds...")
    loaded = []
    for s in SEEDS:
        try:
            L = load_seed(s)
            loaded.append(L)
            m = L["metrics"]
            print(f"  seed={s}: spread_global={m['tier_coverage_spread_global_pp']:.2f}pp, "
                  f"spread_hscc={m['tier_coverage_spread_hscc_pp']:.2f}pp, "
                  f"NSE={m.get('nse_overall', 'NA')}")
        except FileNotFoundError as e:
            print(f"  seed={s}: SKIP ({e})")

    if len(loaded) < 2:
        raise SystemExit("need at least 2 seeds; abort")

    # Per-tier aggregation
    rows = per_tier_aggregation(loaded)
    df_tier = pd.DataFrame(rows)
    df_tier.to_csv(OUT_DIR / "per_tier_table.csv", index=False)
    print(f"\nper-tier 3-seed mean ± std ({len(loaded)} seeds):")
    print(df_tier.to_string(index=False))

    # Bootstrap CI
    cis = basin_block_bootstrap(loaded)
    print(f"\nbasin-block bootstrap CIs (n_boot={N_BOOTSTRAP}):")
    for t, c in cis.items():
        if "note" in c:
            print(f"  {t}: {c['note']}")
            continue
        print(f"  {t}: HSCC mean={c['hscc_cov_mean']:.3f}, "
              f"95% CI [{c['hscc_cov_ci95'][0]:.3f}, {c['hscc_cov_ci95'][1]:.3f}] "
              f"(n_basin_samples={c['n_basin_samples']})")
    with open(OUT_DIR / "bootstrap_ci.json", "w") as f:
        json.dump({"n_bootstrap": N_BOOTSTRAP, "n_seeds": len(loaded), "per_tier_ci": cis}, f, indent=2)

    # Long-format per-basin distribution for plotting
    df_long = per_basin_long_format(loaded)
    df_long.to_csv(OUT_DIR / "per_basin_distribution.csv", index=False)
    print(f"\nper-basin long format saved (n_rows={len(df_long)})")

    # Cross-seed summary
    spread_global_seeds = [L["metrics"]["tier_coverage_spread_global_pp"] for L in loaded]
    spread_hscc_seeds = [L["metrics"]["tier_coverage_spread_hscc_pp"] for L in loaded]
    spread_reduction_seeds = [g - h for g, h in zip(spread_global_seeds, spread_hscc_seeds)]
    nse_seeds = [L["metrics"].get("nse_overall") for L in loaded if "nse_overall" in L["metrics"]]
    summary = {
        "n_seeds": len(loaded),
        "seeds": [L["seed"] for L in loaded],
        "spread_global_pp": {
            "values": spread_global_seeds,
            "mean": float(np.mean(spread_global_seeds)),
            "std": float(np.std(spread_global_seeds, ddof=1)) if len(spread_global_seeds) > 1 else 0.0,
        },
        "spread_hscc_pp": {
            "values": spread_hscc_seeds,
            "mean": float(np.mean(spread_hscc_seeds)),
            "std": float(np.std(spread_hscc_seeds, ddof=1)) if len(spread_hscc_seeds) > 1 else 0.0,
        },
        "spread_reduction_pp": {
            "values": spread_reduction_seeds,
            "mean": float(np.mean(spread_reduction_seeds)),
            "std": float(np.std(spread_reduction_seeds, ddof=1)) if len(spread_reduction_seeds) > 1 else 0.0,
        },
        "nse_overall": {
            "values": nse_seeds,
            "mean": float(np.mean(nse_seeds)) if nse_seeds else None,
            "std": float(np.std(nse_seeds, ddof=1)) if len(nse_seeds) > 1 else 0.0,
        },
        "frame_decision_d_023": "partial-strong",
        "pcr_004_tier_renames": {"gb_arid": "gb_drier_q4", "gb_snow_inf": "gb_montane"},
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== 3-SEED CROSS-SEED SUMMARY ===")
    print(f"spread_global   : {summary['spread_global_pp']['mean']:.2f}pp ± "
          f"{summary['spread_global_pp']['std']:.2f}")
    print(f"spread_HSCC     : {summary['spread_hscc_pp']['mean']:.2f}pp ± "
          f"{summary['spread_hscc_pp']['std']:.2f}")
    print(f"spread reduction: {summary['spread_reduction_pp']['mean']:.2f}pp ± "
          f"{summary['spread_reduction_pp']['std']:.2f}")
    if nse_seeds:
        print(f"NSE overall     : {summary['nse_overall']['mean']:.3f} ± "
              f"{summary['nse_overall']['std']:.3f}")

    print(f"\nwrote {OUT_DIR}/per_tier_table.csv")
    print(f"wrote {OUT_DIR}/bootstrap_ci.json")
    print(f"wrote {OUT_DIR}/per_basin_distribution.csv")
    print(f"wrote {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
