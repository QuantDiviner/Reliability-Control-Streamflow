"""
Audit streamflow + daymet completeness per basin in the three sub-periods used by exp002:
  train: 1980-10-01 .. 1990-09-30
  cal  : 1990-10-01 .. 2000-09-30
  test : 2000-10-01 .. 2014-09-30

Flag basins with > 5% missing in any sub-period (NOTE-001 -> exp002 plan risk row).

Output: experiments/exp002/data_completeness_report.csv
"""
import os
import sys
from datetime import date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SF_DIR = os.path.join(ROOT, "data/raw/CAMELS_US/usgs_streamflow")
DM_DIR = os.path.join(ROOT, "data/raw/CAMELS_US/basin_mean_forcing/daymet")
BASIN_LIST = os.path.join(os.path.dirname(__file__), "basin_list_671.txt")
OUT_CSV = os.path.join(os.path.dirname(__file__), "data_completeness_report.csv")
THRESHOLD = 0.05  # 5% missing tolerance per period

PERIODS = {
    "train": (date(1980, 10, 1), date(1990, 9, 30)),
    "cal":   (date(1990, 10, 1), date(2000, 9, 30)),
    "test":  (date(2000, 10, 1), date(2014, 9, 30)),
}


def expected_days(p_start, p_end):
    return (p_end - p_start).days + 1


def find_file(root, gauge_id, suffix):
    for huc2 in os.listdir(root):
        sub = os.path.join(root, huc2)
        if not os.path.isdir(sub):
            continue
        f = os.path.join(sub, f"{gauge_id}{suffix}")
        if os.path.exists(f):
            return f
    return None


def audit_streamflow(path):
    """Returns: dict period -> (n_present, n_missing) where missing = -999 or out-of-range."""
    counts = {p: {"days_in_range": 0, "missing": 0, "total_lines": 0} for p in PERIODS}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                y, m, d = int(parts[1]), int(parts[2]), int(parts[3])
                val = float(parts[4])
                dt = date(y, m, d)
            except (ValueError, IndexError):
                continue
            for p, (s, e) in PERIODS.items():
                if s <= dt <= e:
                    counts[p]["days_in_range"] += 1
                    if val == -999.0 or val < 0:
                        counts[p]["missing"] += 1
                    counts[p]["total_lines"] += 1
                    break
    return counts


def audit_daymet(path):
    counts = {p: {"days_in_range": 0, "missing": 0, "total_lines": 0} for p in PERIODS}
    with open(path) as f:
        for _ in range(4):
            f.readline()  # skip header (3 metadata + 1 column header)
        for line in f:
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                dt = date(y, m, d)
            except (ValueError, IndexError):
                continue
            # daymet rarely has explicit missing markers; we just count rows in range
            for p, (s, e) in PERIODS.items():
                if s <= dt <= e:
                    counts[p]["days_in_range"] += 1
                    counts[p]["total_lines"] += 1
                    break
    return counts


def main():
    with open(BASIN_LIST) as f:
        basins = [b.strip() for b in f if b.strip()]
    print(f"auditing {len(basins)} basins...")

    rows = []
    flagged = []
    for i, gid in enumerate(basins):
        sf_path = find_file(SF_DIR, gid, "_streamflow_qc.txt")
        dm_path = find_file(DM_DIR, gid, "_lump_cida_forcing_leap.txt")
        if not sf_path or not dm_path:
            print(f"[{i+1}/{len(basins)}] {gid}: MISSING FILE  sf={bool(sf_path)} dm={bool(dm_path)}")
            flagged.append(gid)
            continue

        sf = audit_streamflow(sf_path)
        dm = audit_daymet(dm_path)

        bad_periods = []
        row = {"gauge_id": gid}
        for p in PERIODS:
            exp = expected_days(*PERIODS[p])
            sf_p = sf[p]["days_in_range"]
            sf_miss = sf[p]["missing"]
            dm_p = dm[p]["days_in_range"]
            sf_miss_frac = sf_miss / max(sf_p, 1)
            dm_cov_frac = dm_p / exp
            row[f"sf_{p}_days"] = sf_p
            row[f"sf_{p}_missing_pct"] = round(sf_miss_frac * 100, 2)
            row[f"dm_{p}_days"] = dm_p
            row[f"dm_{p}_cov_pct"] = round(dm_cov_frac * 100, 2)
            if sf_miss_frac > THRESHOLD or dm_cov_frac < (1 - THRESHOLD):
                bad_periods.append(p)
        row["flagged_periods"] = ";".join(bad_periods)
        if bad_periods:
            flagged.append(gid)
        rows.append(row)

        if (i + 1) % 100 == 0 or (i + 1) == len(basins):
            print(f"[{i+1}/{len(basins)}] done; flagged so far: {len(flagged)}")

    # write CSV
    if rows:
        keys = list(rows[0].keys())
        with open(OUT_CSV, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"\nwrote: {OUT_CSV}")

    print(f"\nSummary:")
    print(f"  audited: {len(rows)}")
    print(f"  flagged: {len(flagged)} basins ({len(flagged)/max(len(basins),1)*100:.1f}%)")
    if flagged:
        print(f"  first 10 flagged: {flagged[:10]}")


if __name__ == "__main__":
    main()
