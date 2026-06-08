"""Generate exp005 synthetic data.

For each of 20 representative CAMELS-US basins (5 per tier centroid):
1. Read real Daymet forcings (P, T_avg, PET via Hamon).
2. Derive HBV-Lite parameters from CAMELS attributes.
3. Run HBV-Lite to produce synthetic Q + true SM/SWE.
4. Save per-basin parquet/CSV with [date, P, T, PET, Q_synth, SM_true, SWE_true].

This produces the C0 (clean) condition forcings.
The C3 (perturbed) condition is derived later by `perturb_forcing.py`.

Usage:
    python generate_synthetic.py --out-dir experiments/exp005/data/synthetic_basins/c0
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts dir to path so we can import hbv_lite
sys.path.insert(0, str(Path(__file__).parent))
from hbv_lite import HBVParameters, params_from_camels, simulate, estimate_pet_hamon  # noqa: E402

PROJECT_ROOT = Path("/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP")
CAMELS_ROOT = PROJECT_ROOT / "data/raw/CAMELS_US"
ATTR_DIR = CAMELS_ROOT / "camels_attributes_v2.0"
FORCING_ROOT = CAMELS_ROOT / "basin_mean_forcing/daymet"
TIERS_CSV = PROJECT_ROOT / "experiments/exp002/basin_tiers.csv"


def load_camels_attributes() -> pd.DataFrame:
    """Join soil + topo + clim into a single per-basin attribute frame."""
    soil = pd.read_csv(ATTR_DIR / "camels_soil.txt", sep=";", dtype={"gauge_id": str})
    topo = pd.read_csv(ATTR_DIR / "camels_topo.txt", sep=";", dtype={"gauge_id": str})
    clim = pd.read_csv(ATTR_DIR / "camels_clim.txt", sep=";", dtype={"gauge_id": str})
    df = soil.merge(topo, on="gauge_id").merge(clim, on="gauge_id")
    df["gauge_id"] = df["gauge_id"].str.zfill(8)
    return df


def select_basins_per_tier(tiers_df: pd.DataFrame, attrs: pd.DataFrame, n_per_tier: int = 5) -> pd.DataFrame:
    """Pick n_per_tier basins closest to each tier centroid in (aridity, frac_snow) space.

    Uses standardised distance to be robust to scale.
    """
    rng = np.random.default_rng(42)  # for reproducible tie-breaking
    selected = []

    tiers_df = tiers_df.copy()
    tiers_df["gauge_id"] = tiers_df["gauge_id"].astype(str).str.zfill(8)

    for tier_name, group in tiers_df.groupby("tier"):
        # Compute centroid in original units
        cent_arid = group["aridity"].median()
        cent_fs = group["frac_snow"].median()
        # Standardise globally for distance
        sigma_arid = max(tiers_df["aridity"].std(ddof=0), 1e-6)
        sigma_fs = max(tiers_df["frac_snow"].std(ddof=0), 1e-6)
        d = (
            ((group["aridity"] - cent_arid) / sigma_arid) ** 2
            + ((group["frac_snow"] - cent_fs) / sigma_fs) ** 2
        ) ** 0.5
        # Tiny noise to break ties deterministically
        d = d + rng.normal(0, 1e-9, len(d))
        picked = group.assign(d_to_centroid=d).nsmallest(n_per_tier, "d_to_centroid")
        selected.append(picked)
    out = pd.concat(selected).reset_index(drop=True)
    out = out.merge(attrs[["gauge_id", "p_mean", "pet_mean", "soil_porosity", "soil_depth_pelletier", "gauge_lat"]], on="gauge_id", how="left")
    return out


def load_forcing(gauge_id: str) -> pd.DataFrame:
    """Load CAMELS-US Daymet forcing for one basin."""
    huc2 = gauge_id[:2]
    f = FORCING_ROOT / huc2 / f"{gauge_id}_lump_cida_forcing_leap.txt"
    if not f.exists():
        # Some basins are under different HUC; scan all
        for huc_dir in FORCING_ROOT.iterdir():
            cand = huc_dir / f"{gauge_id}_lump_cida_forcing_leap.txt"
            if cand.exists():
                f = cand
                break
    df = pd.read_csv(f, sep=r"\s+", skiprows=4, names=[
        "year", "month", "day", "hour", "dayl_s", "prcp", "srad", "swe_camels", "tmax", "tmin", "vp"
    ])
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["temp"] = (df["tmax"] + df["tmin"]) / 2.0
    df["doy"] = df["date"].dt.dayofyear
    return df[["date", "doy", "prcp", "temp", "tmax", "tmin", "srad", "vp"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="Multiplicative white noise added to synthetic Q (relative to mean Q). 0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir = args.out_dir.resolve()  # ensure absolute for relative_to() later
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CAMELS-US attributes...")
    attrs = load_camels_attributes()
    tiers = pd.read_csv(TIERS_CSV, dtype={"gauge_id": str})
    selected = select_basins_per_tier(tiers, attrs, n_per_tier=5)
    print(f"Selected {len(selected)} basins ({selected['tier'].value_counts().to_dict()}):")
    print(selected[["gauge_id", "tier", "aridity", "frac_snow", "p_mean"]].to_string())

    manifest = []
    for _, row in selected.iterrows():
        gid = row["gauge_id"]
        forcing = load_forcing(gid)

        # Hamon PET from temp + latitude
        pet = estimate_pet_hamon(
            forcing["temp"].to_numpy(),
            forcing["doy"].to_numpy(),
            float(row["gauge_lat"]),
        )

        params = params_from_camels({
            "soil_porosity": row["soil_porosity"],
            "soil_depth_pelletier": row["soil_depth_pelletier"],
            "frac_snow": row["frac_snow"],
            "aridity": row["aridity"],
            "p_mean": row["p_mean"],
        })

        out = simulate(
            forcing["prcp"].to_numpy(),
            forcing["temp"].to_numpy(),
            pet,
            params,
        )

        q_clean = out["q"]
        # Optional Gaussian noise on Q (so LSTM has something to learn)
        if args.noise_std > 0:
            mean_q = max(q_clean.mean(), 1e-3)
            q_noisy = q_clean + rng.normal(0, args.noise_std * mean_q, len(q_clean))
            q_noisy = np.maximum(q_noisy, 0.0)
        else:
            q_noisy = q_clean.copy()

        df_out = pd.DataFrame({
            "date": forcing["date"],
            "prcp": forcing["prcp"],
            "tmax": forcing["tmax"],
            "tmin": forcing["tmin"],
            "temp_avg": forcing["temp"],
            "srad": forcing["srad"],
            "vp": forcing["vp"],
            "pet": pet,
            "q_synth_clean": q_clean,
            "q_synth_noisy": q_noisy,
            "sm_true": out["sm"],
            "swe_true": out["swe"],
            "uz_true": out["uz"],
            "lz_true": out["lz"],
            "actual_et": out["actual_et"],
        })
        out_csv = args.out_dir / f"{gid}.csv"
        df_out.to_csv(out_csv, index=False)

        # Quick water balance check
        p_total = forcing["prcp"].sum()
        q_total = q_clean.sum()
        et_total = out["actual_et"].sum()
        runoff_ratio = q_total / max(p_total, 1e-9)

        manifest.append({
            "gauge_id": gid,
            "tier": row["tier"],
            "aridity": float(row["aridity"]),
            "frac_snow": float(row["frac_snow"]),
            "n_days": int(len(df_out)),
            "p_mean_synth": float(forcing["prcp"].mean()),
            "q_mean_synth_clean": float(q_clean.mean()),
            "q_mean_synth_noisy": float(q_noisy.mean()),
            "runoff_ratio_synth": float(runoff_ratio),
            "et_mean_synth": float(out["actual_et"].mean()),
            "swe_max": float(out["swe"].max()),
            "sm_mean": float(out["sm"].mean()),
            "params": {k: float(v) for k, v in params.__dict__.items() if isinstance(v, (int, float))},
            "csv": str(out_csv.relative_to(PROJECT_ROOT)),
        })
        print(f"  {gid} ({row['tier']}): P={p_total/365.25:.2f} Q={q_clean.mean():.3f} runoff_ratio={runoff_ratio:.2f}")

    # Save manifest
    manifest_path = args.out_dir / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "n_basins": len(manifest),
            "noise_std": args.noise_std,
            "seed": args.seed,
            "tier_counts": pd.DataFrame(manifest)["tier"].value_counts().to_dict(),
            "basins": manifest,
        }, f, indent=2)
    print(f"\nWrote {len(manifest)} basin CSVs + manifest at {manifest_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
