"""
Build exp002 basin list = full CAMELS-US 671 set intersected with streamflow + daymet availability.

Output: experiments/exp002/basin_list_671.txt
        + per-tier breakdown (snow priority over aridity tiers)
"""
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ATTR_FILE = os.path.join(ROOT, "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt")
STREAMFLOW_DIR = os.path.join(ROOT, "data/raw/CAMELS_US/usgs_streamflow")
DAYMET_DIR = os.path.join(ROOT, "data/raw/CAMELS_US/basin_mean_forcing/daymet")
OUT_FILE = os.path.join(os.path.dirname(__file__), "basin_list_671.txt")
TIER_OUT = os.path.join(os.path.dirname(__file__), "basin_tiers.csv")


def load_clim_basins(path):
    basins = {}
    with open(path) as f:
        header = f.readline().strip().split(";")
        ai_idx = header.index("aridity")
        snow_idx = header.index("frac_snow")
        for line in f:
            cols = line.strip().split(";")
            if not cols or not cols[0]:
                continue
            gid = cols[0].zfill(8)
            basins[gid] = {
                "aridity": float(cols[ai_idx]),
                "frac_snow": float(cols[snow_idx]),
            }
    return basins


def list_files_under(root, suffix):
    """Walk root/HUC2/*<suffix>, return set of gauge IDs (first underscore-separated token)."""
    found = set()
    for huc2 in sorted(os.listdir(root)):
        sub = os.path.join(root, huc2)
        if not os.path.isdir(sub):
            continue
        for fn in os.listdir(sub):
            if fn.endswith(suffix):
                gid = fn.split("_")[0]
                found.add(gid)
    return found


def assign_tier(ai, fs):
    if fs >= 0.4:
        return "snow"
    if ai > 1.5:
        return "dry"
    if ai > 1.0:
        return "semi_arid"
    return "humid"


def main():
    clim = load_clim_basins(ATTR_FILE)
    print(f"camels_clim.txt: {len(clim)} basins")

    sf = list_files_under(STREAMFLOW_DIR, "_streamflow_qc.txt")
    print(f"streamflow files: {len(sf)} unique gauges")

    dm = list_files_under(DAYMET_DIR, "_lump_cida_forcing_leap.txt")
    print(f"daymet files: {len(dm)} unique gauges")

    final = sorted(set(clim.keys()) & sf & dm)
    print(f"\nintersection (attr ∩ streamflow ∩ daymet): {len(final)} basins")

    missing_sf = set(clim.keys()) - sf
    missing_dm = set(clim.keys()) - dm
    if missing_sf:
        print(f"  WARN: {len(missing_sf)} attr-listed basins missing streamflow: {sorted(missing_sf)[:5]}...")
    if missing_dm:
        print(f"  WARN: {len(missing_dm)} attr-listed basins missing daymet: {sorted(missing_dm)[:5]}...")

    with open(OUT_FILE, "w") as f:
        for g in final:
            f.write(g + "\n")
    print(f"\nwrote: {OUT_FILE}")

    tier_counts = Counter()
    with open(TIER_OUT, "w") as f:
        f.write("gauge_id,aridity,frac_snow,tier\n")
        for g in final:
            ai = clim[g]["aridity"]
            fs = clim[g]["frac_snow"]
            t = assign_tier(ai, fs)
            tier_counts[t] += 1
            f.write(f"{g},{ai:.4f},{fs:.4f},{t}\n")
    print(f"wrote: {TIER_OUT}")

    print("\nTier breakdown:")
    for t in ["dry", "semi_arid", "humid", "snow"]:
        print(f"  {t:10s}: {tier_counts[t]}")
    print(f"  total      : {sum(tier_counts.values())}")


if __name__ == "__main__":
    main()
