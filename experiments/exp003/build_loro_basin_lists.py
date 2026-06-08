#!/usr/bin/env python
"""Build 18-fold leave-one-region-out (LORO) basin lists for exp003.

For each HUC-2 region k in 01..18:
  - test_huc{k}.txt  = basins located in HUC-k that are also in the 671-basin canonical set
  - train_huc{k}.txt = the remaining basins in the 671 set (NOT in HUC-k)

Also emits per_huc_tier_distribution.csv (A.2 deliverable) so we can flag
folds whose held-out HUC has 0 or very few basins in some tier (S1' denominator
adjustment per plan §成功标准).

Reads:
  - data/raw/CAMELS_US/usgs_streamflow/{HUC}/<gauge_id>_streamflow_qc.txt
  - experiments/exp002/basin_list_671.txt
  - experiments/exp002/basin_tiers.csv

Writes (all under experiments/exp003/basin_lists/):
  - train_huc{k:02d}.txt          (one gauge_id per line)
  - test_huc{k:02d}.txt
  - per_huc_tier_distribution.csv (k, n_total, n_dry, n_semi_arid, n_humid, n_snow)
  - build_summary.txt             (audit trail)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMFLOW_DIR = REPO_ROOT / "data" / "raw" / "CAMELS_US" / "usgs_streamflow"
CANONICAL_LIST = REPO_ROOT / "experiments" / "exp002" / "basin_list_671.txt"
TIER_CSV = REPO_ROOT / "experiments" / "exp002" / "basin_tiers.csv"
OUT_DIR = REPO_ROOT / "experiments" / "exp003" / "basin_lists"

MIN_BASINS_PER_HUC = 5  # warn threshold per plan §appendix A.1

TIERS = ("dry", "semi_arid", "humid", "snow")


def load_canonical_basins() -> set[str]:
    return {line.strip() for line in CANONICAL_LIST.read_text().splitlines() if line.strip()}


def load_tier_map() -> dict[str, str]:
    tier_map: dict[str, str] = {}
    with TIER_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tier_map[row["gauge_id"]] = row["tier"]
    return tier_map


def map_basin_to_huc() -> dict[str, str]:
    """Return {gauge_id: HUC-2 (zero-padded string '01'..'18')}.

    File naming convention: <gauge_id>_streamflow_qc.txt under HUC-k subdir.
    """
    basin_to_huc: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    if not STREAMFLOW_DIR.exists():
        sys.exit(f"FATAL: streamflow dir not found at {STREAMFLOW_DIR}")
    for huc_dir in sorted(STREAMFLOW_DIR.iterdir()):
        if not huc_dir.is_dir():
            continue
        huc = huc_dir.name  # '01'..'18'
        for f in huc_dir.iterdir():
            if not f.name.endswith("_streamflow_qc.txt"):
                continue
            gauge_id = f.name.split("_")[0]
            if gauge_id in basin_to_huc and basin_to_huc[gauge_id] != huc:
                duplicates.append((gauge_id, basin_to_huc[gauge_id], huc))
            basin_to_huc[gauge_id] = huc
    if duplicates:
        for gid, prev, cur in duplicates:
            print(f"WARN: gauge {gid} appears in HUC-{prev} and HUC-{cur}; using HUC-{cur}")
    return basin_to_huc


def main() -> None:
    canonical = load_canonical_basins()
    tier_map = load_tier_map()
    basin_to_huc = map_basin_to_huc()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Diagnostic: canonical basins not found in any HUC dir
    missing_in_streamflow = canonical - set(basin_to_huc.keys())
    if missing_in_streamflow:
        print(
            f"WARN: {len(missing_in_streamflow)} canonical basins missing in streamflow dir: "
            f"{sorted(missing_in_streamflow)[:10]}{'...' if len(missing_in_streamflow) > 10 else ''}"
        )

    # Group canonical basins by HUC
    huc_to_basins: dict[str, list[str]] = defaultdict(list)
    for gid in sorted(canonical):
        huc = basin_to_huc.get(gid)
        if huc is None:
            continue
        huc_to_basins[huc].append(gid)

    summary_lines: list[str] = []
    summary_lines.append("# exp003 LORO basin list build summary")
    summary_lines.append(f"canonical_basins\t{len(canonical)}")
    summary_lines.append(f"missing_in_streamflow\t{len(missing_in_streamflow)}")
    summary_lines.append("")
    summary_lines.append("HUC\tn_test\tn_train\tlow_count_warn")

    # Tier distribution table
    dist_rows: list[dict[str, int | str]] = []

    folds_written = 0
    coverage_check_total = 0

    for k in range(1, 19):
        huc = f"{k:02d}"
        test_basins = sorted(huc_to_basins.get(huc, []))
        train_basins = sorted(canonical - set(test_basins))

        coverage_check_total += len(test_basins)

        test_path = OUT_DIR / f"test_huc{huc}.txt"
        train_path = OUT_DIR / f"train_huc{huc}.txt"
        test_path.write_text("\n".join(test_basins) + ("\n" if test_basins else ""))
        train_path.write_text("\n".join(train_basins) + "\n")
        folds_written += 1

        warn = "LOW" if len(test_basins) < MIN_BASINS_PER_HUC else ""
        summary_lines.append(f"{huc}\t{len(test_basins)}\t{len(train_basins)}\t{warn}")

        # Tier counts
        counts = {t: 0 for t in TIERS}
        for gid in test_basins:
            t = tier_map.get(gid)
            if t in counts:
                counts[t] += 1
        dist_rows.append(
            {
                "huc": huc,
                "n_total": len(test_basins),
                **{f"n_{t}": counts[t] for t in TIERS},
            }
        )

    # Write tier distribution CSV (A.2 deliverable)
    dist_csv = OUT_DIR / "per_huc_tier_distribution.csv"
    with dist_csv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["huc", "n_total"] + [f"n_{t}" for t in TIERS]
        )
        w.writeheader()
        w.writerows(dist_rows)

    summary_lines.append("")
    summary_lines.append(
        f"coverage_check\ttest_union={coverage_check_total}\tcanonical={len(canonical)}\t"
        f"identical={'OK' if coverage_check_total == len(canonical) else 'MISMATCH'}"
    )

    (OUT_DIR / "build_summary.txt").write_text("\n".join(summary_lines) + "\n")

    print(f"Wrote {folds_written} fold pairs (train_huc01..18.txt + test_huc01..18.txt) to {OUT_DIR}")
    print(f"Tier distribution: {dist_csv}")
    print(f"Audit summary:    {OUT_DIR / 'build_summary.txt'}")
    if coverage_check_total != len(canonical):
        sys.exit(
            f"FATAL: union of test sets ({coverage_check_total}) != canonical ({len(canonical)}). "
            "Some basins may be missing from streamflow dir or duplicated across HUCs."
        )


if __name__ == "__main__":
    main()
