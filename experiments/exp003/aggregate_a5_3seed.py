"""
exp003 A5 cross-seed aggregator — 3-seed mean ± std for HUC 10/12/14/18.

Reads:
  experiments/exp003/results/exp003_loro_huc{10,12,14,18}_*/_analysis/metrics.json   (seed=42)
  experiments/exp003/results/exp003_loro_huc{10,12,14,18}_seed{137,2024}_*/_analysis/metrics.json

Writes:
  experiments/exp003/results/_a5_3seed/summary.json
  experiments/exp003/results/_a5_3seed/per_huc_table.csv

Per HUC per tier: mean ± std of HSCC coverage across 3 seeds.
This generates the 95% CI claimed in R2 D-R2-exp003 A5 P1-1 mitigation.
"""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "experiments" / "exp003" / "results"
OUT_DIR = RESULTS_ROOT / "_a5_3seed"

HUCS = ["10", "12", "14", "18"]
SEEDS = [42, 137, 2024]
TIERS = ["dry", "semi_arid", "humid", "snow"]


def find_run_dir(huc: str, seed: int) -> Path | None:
    if seed == 42:
        # Original seed=42 run pattern: exp003_loro_huc{HUC}_<ts> (no seed suffix)
        candidates = sorted(RESULTS_ROOT.glob(f"exp003_loro_huc{huc}_2*"))
        # Filter out any with seed suffix (only original)
        candidates = [c for c in candidates if "_seed" not in c.name]
    else:
        candidates = sorted(RESULTS_ROOT.glob(f"exp003_loro_huc{huc}_seed{seed}_*"))
    if not candidates:
        return None
    return candidates[-1]


def load_metrics(huc: str, seed: int) -> dict | None:
    rd = find_run_dir(huc, seed)
    if rd is None:
        return None
    p = rd / "_analysis" / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading metrics for HUC × seed grid:")
    rows = []
    per_huc_per_seed = {}
    for huc in HUCS:
        per_huc_per_seed[huc] = {}
        for s in SEEDS:
            m = load_metrics(huc, s)
            status = "✅" if m else "❌"
            print(f"  HUC-{huc} seed={s}: {status}")
            if m is None:
                continue
            per_huc_per_seed[huc][s] = m
            for tier, td in m.get("per_tier", {}).items():
                rows.append({
                    "huc": huc,
                    "seed": s,
                    "tier": tier,
                    "n_test_basins": td["n_basins"],
                    "global_coverage": td["global_coverage"],
                    "hscc_coverage": td["hscc_coverage"],
                    "global_width_mm_d": td["global_width_mm_d"],
                    "hscc_width_mm_d": td["hscc_width_mm_d"],
                })

    if not rows:
        raise SystemExit("no metrics found; abort")

    long_df = pd.DataFrame(rows)

    # Mean ± std across seeds for each (huc, tier)
    agg = long_df.groupby(["huc", "tier"]).agg(
        n_seeds=("seed", "nunique"),
        n_test_basins=("n_test_basins", "first"),
        global_cov_mean=("global_coverage", "mean"),
        global_cov_std=("global_coverage", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        hscc_cov_mean=("hscc_coverage", "mean"),
        hscc_cov_std=("hscc_coverage", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        hscc_width_mean=("hscc_width_mm_d", "mean"),
        global_width_mean=("global_width_mm_d", "mean"),
    ).reset_index()
    agg["hscc_cov_ci95_lo"] = agg["hscc_cov_mean"] - 1.96 * agg["hscc_cov_std"]
    agg["hscc_cov_ci95_hi"] = agg["hscc_cov_mean"] + 1.96 * agg["hscc_cov_std"]
    agg.to_csv(OUT_DIR / "per_huc_table.csv", index=False)

    print(f"\n=== per (HUC, tier) 3-seed table ({agg['n_seeds'].max()} seeds) ===")
    print(agg.to_string(index=False))

    # Per-HUC overall (spread reduction etc.) per seed
    huc_summary = {}
    for huc in HUCS:
        s_data = []
        for seed in SEEDS:
            m = per_huc_per_seed.get(huc, {}).get(seed)
            if not m:
                continue
            s_data.append({
                "seed": seed,
                "spread_global_pp": m.get("tier_coverage_spread_global_pp"),
                "spread_hscc_pp": m.get("tier_coverage_spread_hscc_pp"),
            })
        if not s_data:
            continue
        sg = [d["spread_global_pp"] for d in s_data if d["spread_global_pp"] is not None]
        sh = [d["spread_hscc_pp"] for d in s_data if d["spread_hscc_pp"] is not None]
        huc_summary[huc] = {
            "n_seeds": len(s_data),
            "spread_global_pp_mean": float(np.mean(sg)) if sg else None,
            "spread_global_pp_std": float(np.std(sg, ddof=1)) if len(sg) > 1 else 0.0,
            "spread_hscc_pp_mean": float(np.mean(sh)) if sh else None,
            "spread_hscc_pp_std": float(np.std(sh, ddof=1)) if len(sh) > 1 else 0.0,
        }

    summary = {
        "n_hucs": len(HUCS),
        "seeds": SEEDS,
        "per_huc": huc_summary,
        "narrative_hooks": [
            "R2 D-R2-exp003 A5 mitigation: 95% CI on per-tier HSCC coverage for 4 most failed HUCs (10, 12, 14, 18).",
            "HUC-10 humid 0.509 (single seed) → 3-seed mean ± std with CI exposes seed-variability of within-tier exchangeability failure.",
            "Narrative §5.4 transfer reliability map can cite per-cell ±std as uncertainty bands instead of point estimates.",
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== per-HUC cross-seed summary ===")
    for huc, s in huc_summary.items():
        print(f"  HUC-{huc}: spread_global={s['spread_global_pp_mean']:.2f}±{s['spread_global_pp_std']:.2f}pp, "
              f"spread_hscc={s['spread_hscc_pp_mean']:.2f}±{s['spread_hscc_pp_std']:.2f}pp")

    print(f"\nwrote {OUT_DIR}/per_huc_table.csv")
    print(f"wrote {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
