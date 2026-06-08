"""
Select 50 CAMELS-GB basins, balanced across a GB-specific 4-tier scheme.

Tier scheme (per decision-log D-022):
    GB aridity range [0.12, 0.96] / frac_snow ∈ [0, 0.17] — US thresholds (aridity > 1.5 → dry,
    frac_snow ≥ 0.4 → snow) collapse all 671 GB basins to a single humid tier and break HSCC
    stratification. Plan §7 anticipated this; the methodology (stratify-by-regime) is portable
    while threshold values must be dataset-specific.

    Priority (snow first, then aridity quantiles on remaining basins):
        gb_montane : frac_snow >= 0.05            (top ~5%, snow-influenced highlands)
        gb_drier_q4     : aridity   > 0.70             (driest UK quartile, eastern England)
        gb_mid      : 0.50 < aridity <= 0.70       (mid-aridity quartile)
        gb_wet      : aridity   <= 0.50            (wet half — Wales/Scotland)

Filters (plan §2.2 + benchmark catchment):
    - benchmark_catch == 'Y' (Coxon 2020 curated set, 137 basins)
    - area >= 10 km²
    - streamflow completeness >= 90% in [cal_start, test_end]

Outputs:
    experiments/exp004/basin_lists/gb_50.txt              (chosen 50 basins, one per line)
    experiments/exp004/basin_lists/gb_basin_tiers.csv     (basin, tier, aridity, frac_snow)
    experiments/exp004/basin_lists/gb_selection_log.txt   (counts + audit)

Run AFTER CAMELS-GB extraction:
    python experiments/exp004/scripts/select_gb_basins.py [--n 50] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
GB_DIR = ROOT / "data" / "raw" / "CAMELS_GB"
ATTR_DIR = GB_DIR / "data" / "attributes"  # may be GB_DIR/attributes (post-extract layout TBD)
TIMESERIES_DIR = GB_DIR / "data" / "timeseries"  # same caveat
OUT_DIR = ROOT / "experiments" / "exp004" / "basin_lists"


def find_attribute_dir() -> Path:
    """Probe known CAMELS-GB layouts. After running pipeline `mv` step
    files live in <root>/data/attributes/ (NH-compatible)."""
    for cand in [
        GB_DIR / "data" / "attributes",
        GB_DIR / "attributes",
        GB_DIR / "data" / "CAMELS_GB_DATASET" / "attributes",
    ]:
        if cand.is_dir() and any(cand.glob("*_attributes.csv")):
            return cand
    # Fallback: CSVs may sit at <root>/data/ (raw zip layout, before the pipeline mv)
    for cand in [GB_DIR / "data", GB_DIR]:
        if cand.is_dir() and any(cand.glob("CAMELS_GB_*_attributes.csv")):
            return cand
    raise FileNotFoundError(
        f"Could not locate CAMELS-GB attributes under {GB_DIR}. "
        "Ensure the zip has been extracted and CSVs are reachable."
    )


def find_timeseries_dir() -> Path:
    for cand in [
        GB_DIR / "data" / "timeseries",
        GB_DIR / "timeseries",
        GB_DIR / "data" / "CAMELS_GB_DATASET" / "timeseries",
    ]:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Could not locate CAMELS-GB timeseries folder under {GB_DIR}.")


def load_attributes(attr_dir: Path) -> pd.DataFrame:
    """Concatenate all *_attributes.csv files into one basin-indexed dataframe."""
    csvs = sorted(attr_dir.glob("*_attributes.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_attributes.csv files in {attr_dir}")
    frames = []
    for f in csvs:
        df = pd.read_csv(f, dtype={"gauge_id": str})
        df = df.set_index("gauge_id")
        frames.append(df)
    return pd.concat(frames, axis=1)


def assign_tier(aridity: float, frac_snow: float) -> str:
    """GB-specific tier assignment (D-022). snow priority, then aridity quantiles."""
    if frac_snow >= 0.05:
        return "gb_montane"
    if aridity > 0.70:
        return "gb_drier_q4"
    if aridity > 0.50:
        return "gb_mid"
    return "gb_wet"


def streamflow_completeness(ts_dir: Path, basin: str, start: str, end: str) -> float:
    """Return fraction of non-NaN discharge days in [start, end]. -1 if file missing."""
    files = list(ts_dir.glob(f"**/CAMELS_GB_hydromet_timeseries_{basin}_*.csv"))
    if not files:
        return -1.0
    df = pd.read_csv(files[0], parse_dates=["date"], dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    mask = (df["date"] >= start) & (df["date"] <= end)
    sub = df.loc[mask]
    if sub.empty:
        return 0.0
    # CAMELS-GB target is 'discharge_spec' (mm/d) or 'discharge_vol' (m3/s)
    tgt_col = next((c for c in ("discharge_spec", "discharge_vol", "Q") if c in sub.columns), None)
    if tgt_col is None:
        return 0.0
    return float(sub[tgt_col].notna().mean())


def balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Pick n basins balanced across tiers. Allocation roughly:
        humid: 40%   semi_arid: 25%   dry: 20%   snow: 15%
    (humid dominates UK climate; cap others by available counts)."""
    rng = np.random.default_rng(seed)
    # Tier targets reflect the GB distribution (most basins are wet),
    # but oversample arid/mid/snow_inf to ensure each tier has enough basins for HSCC.
    targets = {
        "gb_wet":      int(round(n * 0.40)),
        "gb_mid":      int(round(n * 0.25)),
        "gb_drier_q4":     int(round(n * 0.25)),
        "gb_montane": int(round(n * 0.10)),
    }
    diff = n - sum(targets.values())
    targets["gb_wet"] += diff

    chosen = []
    actual = {tier: 0 for tier in targets}
    for tier, k in targets.items():
        pool = df[df["tier"] == tier]
        take = min(k, len(pool))
        if take > 0:
            picks = pool.sample(n=take, random_state=int(rng.integers(0, 1_000_000)))
            chosen.append(picks)
        actual[tier] = take

    # Fill remaining shortfall from the most-abundant tier (gb_wet > gb_mid > gb_drier_q4 > gb_montane)
    short = n - sum(actual.values())
    if short > 0 and chosen:
        remaining = df.drop(index=pd.concat(chosen).index)
        for tier in ["gb_wet", "gb_mid", "gb_drier_q4", "gb_montane"]:
            pool = remaining[remaining["tier"] == tier]
            take = min(short, len(pool))
            if take > 0:
                picks = pool.sample(n=take, random_state=int(rng.integers(0, 1_000_000)))
                chosen.append(picks)
                actual[tier] += take
                short -= take
                if short == 0:
                    break

    out = pd.concat(chosen).sort_index() if chosen else df.iloc[0:0]
    return out, actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of basins to select (default 50)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cal-start", default="1995-10-01")
    ap.add_argument("--test-end", default="2015-09-30")
    ap.add_argument("--min-area-km2", type=float, default=10.0)
    ap.add_argument("--max-missing-frac", type=float, default=0.10)
    args = ap.parse_args()

    attr_dir = find_attribute_dir()
    ts_dir = find_timeseries_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] attributes: {attr_dir}")
    print(f"[INFO] timeseries: {ts_dir}")

    attrs = load_attributes(attr_dir)
    print(f"[INFO] attribute rows: {len(attrs)}")

    # Required columns. Coxon 2020 uses these exact names.
    required = ["aridity", "frac_snow", "area"]
    missing = [c for c in required if c not in attrs.columns]
    if missing:
        # Print available column hints to help user diagnose CAMELS-GB column naming if it has shifted.
        print(f"[FAIL] missing required columns: {missing}")
        print(f"[INFO] available columns sample: {list(attrs.columns)[:30]}")
        sys.exit(2)

    df = attrs.copy()
    df["tier"] = [assign_tier(a, s) for a, s in zip(df["aridity"], df["frac_snow"])]
    print("[INFO] tier counts (raw):", Counter(df["tier"]))

    # Filter: area
    df = df[df["area"] >= args.min_area_km2]

    # Filter: benchmark catchment ('Y'/'N' string per Coxon 2020)
    if "benchmark_catch" in df.columns:
        before = len(df)
        df = df[df["benchmark_catch"].astype(str).str.upper() == "Y"]
        print(f"[INFO] benchmark_catch=='Y' filter kept {len(df)}/{before}")
    elif "num_reservoir" in df.columns:
        before = len(df)
        df = df[df["num_reservoir"].fillna(0) == 0]
        print(f"[INFO] num_reservoir==0 filter kept {len(df)}/{before}")

    # Filter: streamflow completeness in cal+test span
    print("[INFO] streamflow completeness check (this may take a minute)...")
    completeness = []
    for basin in df.index:
        completeness.append(streamflow_completeness(ts_dir, basin, args.cal_start, args.test_end))
    df["sf_completeness"] = completeness
    before = len(df)
    df = df[df["sf_completeness"] >= (1.0 - args.max_missing_frac)]
    print(f"[INFO] completeness filter kept {len(df)}/{before}")

    print("[INFO] tier counts (post-filter):", Counter(df["tier"]))

    chosen, actual = balanced_sample(df, args.n, args.seed)
    print(f"[INFO] chosen {len(chosen)} basins; per-tier: {actual}")

    # Write outputs
    list_path = OUT_DIR / f"gb_{args.n}.txt"
    list_path.write_text("\n".join(chosen.index) + "\n")

    tiers_path = OUT_DIR / "gb_basin_tiers.csv"
    chosen.reset_index()[["gauge_id", "tier", "aridity", "frac_snow", "area", "sf_completeness"]].to_csv(
        tiers_path, index=False
    )

    log_path = OUT_DIR / "gb_selection_log.txt"
    log_path.write_text(
        json.dumps(
            {
                "n_target": args.n,
                "n_actual": len(chosen),
                "tier_counts": dict(actual),
                "seed": args.seed,
                "filters": {
                    "min_area_km2": args.min_area_km2,
                    "max_missing_frac": args.max_missing_frac,
                    "cal_start": args.cal_start,
                    "test_end": args.test_end,
                },
            },
            indent=2,
        )
    )

    print(f"[OK] wrote {list_path}")
    print(f"[OK] wrote {tiers_path}")
    print(f"[OK] wrote {log_path}")


if __name__ == "__main__":
    main()
