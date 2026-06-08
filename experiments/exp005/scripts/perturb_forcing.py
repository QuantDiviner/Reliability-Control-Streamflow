"""Perturb CAMELS-US forcings for exp005 C3 condition.

Generates two synthetic CAMELS-US-compatible dataset trees:

    experiments/exp005/data/synthetic_camels/c0/
        basin_mean_forcing/daymet/{huc2}/{gid}_lump_cida_forcing_leap.txt  ← original CAMELS forcing
        usgs_streamflow/{huc2}/{gid}_streamflow_qc.txt                    ← synthetic Q from HBV (in cfs)
        camels_attributes_v2.0  → symlink to real attributes

    experiments/exp005/data/synthetic_camels/c3/
        basin_mean_forcing/daymet/{huc2}/{gid}_lump_cida_forcing_leap.txt  ← P + 20% white noise, T ± 2°C white noise
        usgs_streamflow/{huc2}/{gid}_streamflow_qc.txt                    ← SAME synthetic Q as c0
        camels_attributes_v2.0  → symlink to real attributes

The two conditions share the same target Q (synthetic from clean HBV + 5% noise).
Only the LSTM's input forcings differ. This is the causal design: in c0 the LSTM
sees clean forcings and can reconstruct the HBV transfer function; in c3 the
forcings are corrupted, so the LSTM cannot infer true SM/SWE, producing
systematic residual autocorrelation (H1) and HSCC failure to recover (H4).

Outputs match exactly the directory structure required by NeuralHydrology's
camels_us dataset loader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP")
CAMELS_ROOT = PROJECT_ROOT / "data/raw/CAMELS_US"
SRC_FORCING_ROOT = CAMELS_ROOT / "basin_mean_forcing/daymet"
ATTR_DIR = CAMELS_ROOT / "camels_attributes_v2.0"
SYNTH_C0 = PROJECT_ROOT / "experiments/exp005/data/synthetic_basins/c0"
OUT_ROOT = PROJECT_ROOT / "experiments/exp005/data/synthetic_camels"


def load_basin_areas() -> dict[str, float]:
    """Read basin areas (km^2) from CAMELS topo attributes."""
    topo = pd.read_csv(ATTR_DIR / "camels_topo.txt", sep=";", dtype={"gauge_id": str})
    topo["gauge_id"] = topo["gauge_id"].str.zfill(8)
    # area_gages2 is the USGS-reported area; fall back to area_geospa_fabric
    area_col = "area_gages2" if "area_gages2" in topo.columns else "area_geospa_fabric"
    return dict(zip(topo["gauge_id"], topo[area_col].astype(float)))


def mm_per_day_to_cfs(q_mm_day: np.ndarray, area_km2: float) -> np.ndarray:
    """Convert discharge from mm/day (catchment-averaged) to cfs (cubic feet per sec)."""
    # Q (m^3/s) = Q (mm/day) × 1e-3 m × area_km2 × 1e6 m^2 / 86400 s
    q_m3s = q_mm_day * 1e-3 * area_km2 * 1e6 / 86400.0
    q_cfs = q_m3s / 0.0283168  # 1 cfs = 0.0283168 m^3/s
    return q_cfs


def read_forcing_file(path: Path) -> tuple[list[str], pd.DataFrame]:
    """Read a CAMELS-US lump_cida_forcing_leap.txt; return (header_lines, dataframe)."""
    with open(path) as f:
        lines = f.readlines()
    header = lines[:4]  # latitude, elev, area, column-names
    data = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=4,
        names=[
            "year", "month", "day", "hour",
            "dayl_s", "prcp", "srad", "swe_camels", "tmax", "tmin", "vp",
        ],
        engine="python",
    )
    return header, data


def write_forcing_file(path: Path, header_lines: list[str], df: pd.DataFrame) -> None:
    """Write a CAMELS-US-format forcing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(header_lines)
        for _, row in df.iterrows():
            f.write(
                f"{int(row['year']):4d} {int(row['month']):02d} {int(row['day']):02d} "
                f"{int(row['hour']):02d}\t"
                f"{row['dayl_s']:.2f}\t{row['prcp']:.2f}\t{row['srad']:.2f}\t"
                f"{row['swe_camels']:.2f}\t{row['tmax']:.2f}\t{row['tmin']:.2f}\t"
                f"{row['vp']:.2f}\n"
            )


def write_streamflow_file(path: Path, gauge_id: str, dates: pd.Series, q_cfs: np.ndarray) -> None:
    """Write a CAMELS-US-format streamflow_qc.txt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for date, q in zip(dates, q_cfs):
            f.write(
                f"{gauge_id} {date.year:4d} {date.month:02d} {date.day:02d} "
                f"{q:9.2f} A\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prcp-noise-frac", type=float, default=0.20,
                        help="Std of P perturbation as fraction of basin mean P (C3 only).")
    parser.add_argument("--temp-noise-c", type=float, default=2.0,
                        help="Std of T perturbation in degC (C3 only).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng_c3 = np.random.default_rng(args.seed)
    areas = load_basin_areas()

    manifest_c0 = json.loads((SYNTH_C0 / "_manifest.json").read_text())
    basins = manifest_c0["basins"]
    print(f"Processing {len(basins)} basins...")

    perturbation_log = []

    for c_name in ("c0", "c3"):
        out_root = OUT_ROOT / c_name
        out_root.mkdir(parents=True, exist_ok=True)
        # Symlink camels_attributes_v2.0
        attr_link = out_root / "camels_attributes_v2.0"
        if attr_link.exists() or attr_link.is_symlink():
            attr_link.unlink()
        attr_link.symlink_to(ATTR_DIR)

    for b in basins:
        gid = b["gauge_id"]
        huc2 = gid[:2]
        area = areas[gid]
        synth_csv = pd.read_csv(SYNTH_C0 / f"{gid}.csv", parse_dates=["date"])

        # Source forcing (we'll modify for c3)
        src_forcing_path = SRC_FORCING_ROOT / huc2 / f"{gid}_lump_cida_forcing_leap.txt"
        if not src_forcing_path.exists():
            for huc_dir in SRC_FORCING_ROOT.iterdir():
                cand = huc_dir / f"{gid}_lump_cida_forcing_leap.txt"
                if cand.exists():
                    src_forcing_path = cand
                    huc2 = huc_dir.name
                    break
        header, forcing_df = read_forcing_file(src_forcing_path)

        # === C0 condition ===
        # Forcing: copy original CAMELS daymet forcing as-is.
        # Streamflow: synthetic Q (mm/day) → cfs.
        q_cfs_synth = mm_per_day_to_cfs(synth_csv["q_synth_noisy"].to_numpy(), area)
        c0_forcing_out = OUT_ROOT / "c0" / "basin_mean_forcing/daymet" / huc2 / f"{gid}_lump_cida_forcing_leap.txt"
        c0_streamflow_out = OUT_ROOT / "c0" / "usgs_streamflow" / huc2 / f"{gid}_streamflow_qc.txt"
        write_forcing_file(c0_forcing_out, header, forcing_df)
        write_streamflow_file(c0_streamflow_out, gid, synth_csv["date"], q_cfs_synth)

        # === C3 condition ===
        # Perturb P (additive Gaussian, sigma = frac × basin mean P)
        # Perturb T_max, T_min by SAME daily Gaussian shift (sigma = temp_noise_c)
        # Keep srad, vp untouched (per plan §2.3 'P and T' only)
        forcing_c3 = forcing_df.copy()
        n = len(forcing_c3)
        mean_p = float(forcing_c3["prcp"].mean())
        p_noise = rng_c3.normal(0.0, args.prcp_noise_frac * mean_p, n)
        t_noise = rng_c3.normal(0.0, args.temp_noise_c, n)
        forcing_c3["prcp"] = np.maximum(forcing_c3["prcp"] + p_noise, 0.0)
        forcing_c3["tmax"] = forcing_c3["tmax"] + t_noise
        forcing_c3["tmin"] = forcing_c3["tmin"] + t_noise

        c3_forcing_out = OUT_ROOT / "c3" / "basin_mean_forcing/daymet" / huc2 / f"{gid}_lump_cida_forcing_leap.txt"
        c3_streamflow_out = OUT_ROOT / "c3" / "usgs_streamflow" / huc2 / f"{gid}_streamflow_qc.txt"
        write_forcing_file(c3_forcing_out, header, forcing_c3)
        # Same target Q
        write_streamflow_file(c3_streamflow_out, gid, synth_csv["date"], q_cfs_synth)

        perturbation_log.append({
            "gauge_id": gid,
            "tier": b["tier"],
            "area_km2": area,
            "n_days": n,
            "mean_p_clean": mean_p,
            "p_noise_std": float(args.prcp_noise_frac * mean_p),
            "t_noise_std": float(args.temp_noise_c),
            "q_mean_cfs": float(np.mean(q_cfs_synth)),
            "q_mean_mm_day": float(synth_csv["q_synth_noisy"].mean()),
        })
        print(f"  {gid} ({b['tier']:>9}, area={area:7.1f} km²): Q_mean = {synth_csv['q_synth_noisy'].mean():.3f} mm/d → {np.mean(q_cfs_synth):.1f} cfs")

    # Save perturbation log
    out_manifest = OUT_ROOT / "_manifest.json"
    with open(out_manifest, "w") as f:
        json.dump({
            "n_basins": len(perturbation_log),
            "prcp_noise_frac": args.prcp_noise_frac,
            "temp_noise_c": args.temp_noise_c,
            "seed": args.seed,
            "basins": perturbation_log,
        }, f, indent=2)

    # Write basin list (for NH configs)
    basin_list_path = OUT_ROOT / "basin_list_20.txt"
    with open(basin_list_path, "w") as f:
        for b in basins:
            f.write(f"{b['gauge_id']}\n")

    print(f"\nWrote synthetic CAMELS dataset trees:")
    print(f"  {OUT_ROOT}/c0/  (clean)")
    print(f"  {OUT_ROOT}/c3/  (P+T perturbed)")
    print(f"  Basin list: {basin_list_path}")
    print(f"  Manifest:   {out_manifest}")


if __name__ == "__main__":
    main()
