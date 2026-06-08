"""
Select 100 basins from CAMELS-US balanced across aridity tiers for Go/no-go exp.

Tiers (from idea.md):
  dry:       AI > 1.5          (water-limited, arid/semi-arid)
  semi-arid: 1.0 < AI <= 1.5
  humid:     0.5 < AI <= 1.0
  snow:      frac_snow >= 0.4  (dominates across AI tiers — checked last)

Strategy: 25 basins per tier, stratified random sample.
Snow tier takes basins not already selected by aridity tiers.
"""

import os
import random
import numpy as np

SEED = 42
N_PER_TIER = 25
ATTR_FILE = os.path.join(
    os.path.dirname(__file__), "../../data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
)
STREAMFLOW_DIR = os.path.join(
    os.path.dirname(__file__), "../../data/raw/CAMELS_US/usgs_streamflow"
)
OUT_FILE = os.path.join(os.path.dirname(__file__), "basin_list_100.txt")

random.seed(SEED)
np.random.seed(SEED)


def load_camels_clim(attr_file):
    basins = {}
    with open(attr_file) as f:
        header = f.readline().strip().split(";")
        ai_idx = header.index("aridity")
        snow_idx = header.index("frac_snow")
        for line in f:
            cols = line.strip().split(";")
            gauge_id = cols[0].zfill(8)
            ai = float(cols[ai_idx])
            frac_snow = float(cols[snow_idx])
            basins[gauge_id] = {"aridity": ai, "frac_snow": frac_snow}
    return basins


def build_streamflow_index(streamflow_dir):
    """Build set of gauge IDs that have streamflow files (gauge IDs ≠ HUC-2 prefix)."""
    available = set()
    for huc2 in os.listdir(streamflow_dir):
        folder = os.path.join(streamflow_dir, huc2)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith("_streamflow_qc.txt"):
                gauge_id = fname.split("_")[0]
                available.add(gauge_id)
    return available


def main():
    basins = load_camels_clim(os.path.abspath(ATTR_FILE))
    sf_dir = os.path.abspath(STREAMFLOW_DIR)

    # Build index of all basins with streamflow files
    sf_index = build_streamflow_index(sf_dir)
    print(f"Streamflow files found: {len(sf_index)} basins")

    # Filter to basins that have streamflow files
    available = {k: v for k, v in basins.items() if k in sf_index}
    print(f"Total basins with streamflow data: {len(available)}")

    # Assign primary tiers (snow takes priority for marginal AI basins)
    dry, semi_arid, humid, snow = [], [], [], []
    for gid, attrs in available.items():
        ai = attrs["aridity"]
        fs = attrs["frac_snow"]
        if fs >= 0.4:
            snow.append(gid)
        elif ai > 1.5:
            dry.append(gid)
        elif ai > 1.0:
            semi_arid.append(gid)
        else:
            humid.append(gid)

    print(f"  dry={len(dry)}, semi-arid={len(semi_arid)}, humid={len(humid)}, snow={len(snow)}")

    def sample(lst, n):
        if len(lst) < n:
            print(f"  WARNING: only {len(lst)} basins in tier, using all")
            return lst[:]
        return random.sample(lst, n)

    selected_dry = sample(dry, N_PER_TIER)
    selected_semi = sample(semi_arid, N_PER_TIER)
    selected_humid = sample(humid, N_PER_TIER)
    selected_snow = sample(snow, N_PER_TIER)

    all_selected = sorted(set(selected_dry + selected_semi + selected_humid + selected_snow))
    print(f"\nSelected {len(all_selected)} basins (may be <100 due to snow overlap)")

    with open(OUT_FILE, "w") as f:
        for gid in all_selected:
            f.write(gid + "\n")
    print(f"Saved to: {OUT_FILE}")

    # Print tier summary
    print("\nTier breakdown:")
    print(f"  dry (AI>1.5):          {len(selected_dry)} basins")
    print(f"  semi-arid (1-1.5):     {len(selected_semi)} basins")
    print(f"  humid (AI<1.0):        {len(selected_humid)} basins")
    print(f"  snow (frac_snow>=0.4): {len(selected_snow)} basins")

    # Print aridity stats per tier
    print("\nAridity index stats per tier:")
    for name, ids in [("dry", selected_dry), ("semi-arid", selected_semi),
                      ("humid", selected_humid), ("snow", selected_snow)]:
        ai_vals = [available[g]["aridity"] for g in ids]
        snow_vals = [available[g]["frac_snow"] for g in ids]
        print(f"  {name:10s}: AI={np.mean(ai_vals):.2f}±{np.std(ai_vals):.2f}, "
              f"frac_snow={np.mean(snow_vals):.2f}±{np.std(snow_vals):.2f}")


if __name__ == "__main__":
    main()
