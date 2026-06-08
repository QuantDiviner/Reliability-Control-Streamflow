"""
exp010 Stage 1 — per-seed analysis (parameterized version of stage1_data_precheck.py).

Used by Action 6 multi-seed aggregator: one log file per seed → per_basin_metrics.csv
+ tier_aggregate.csv + metrics.json under <run_dir>/_analysis/.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP")
NH_LIB = REPO / "libs/neuralhydrology"
CAMELS_DIR = REPO / "data/raw/CAMELS_US"

sys.path.insert(0, str(NH_LIB))
from neuralhydrology.datasetzoo.camelsus import load_camels_us_attributes


def parse_log(log_path):
    text = log_path.read_text()
    start_re = re.compile(r"Start evaluation for TS (\d+)_[^ ]+ with alpha 0\.1")
    metrics_re = re.compile(r"Evaluation metrics \(alpha 0\.1: (\{.*\})\s*$", re.MULTILINE)
    starts = [(m.start(), m.group(1)) for m in start_re.finditer(text)]
    metrics_iter = list(metrics_re.finditer(text))
    rows = []
    for s_pos, basin in starts:
        m = next((mm for mm in metrics_iter if mm.start() > s_pos), None)
        if m is None:
            continue
        try:
            d = ast.literal_eval(m.group(1))
        except Exception:
            continue
        rows.append({"basin": basin, **d})
    return pd.DataFrame(rows)


def assign_tier(row, snow_thresh=0.4, dry_thresh=1.5, semi_arid_thresh=1.0):
    if row["frac_snow"] >= snow_thresh:
        return "snow"
    a = row["aridity"]
    if a > dry_thresh:
        return "dry"
    if a > semi_arid_thresh:
        return "semi_arid"
    return "humid"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_path", required=True)
    p.add_argument("--basin_list", default=str(REPO / "data/raw/exp010_basin_list_clean.txt"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()

    log_path = Path(args.log_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basins = Path(args.basin_list).read_text().strip().splitlines()

    print(f"[stage1 seed={args.seed}] log: {log_path}")
    df = parse_log(log_path)
    print(f"[stage1 seed={args.seed}] parsed {len(df)} basins")

    missing = set(basins) - set(df["basin"])
    extra = set(df["basin"]) - set(basins)
    if missing:
        print(f"WARN seed={args.seed}: {len(missing)} missing")
    if extra:
        print(f"WARN seed={args.seed}: {len(extra)} extra")

    attrs = load_camels_us_attributes(CAMELS_DIR, basins)[
        ["frac_snow", "aridity", "p_mean", "pet_mean"]
    ]
    df = df.merge(attrs, left_on="basin", right_index=True, how="left")
    df["tier"] = df.apply(assign_tier, axis=1)

    pb = out_dir / "per_basin_metrics.csv"
    df.to_csv(pb, index=False)

    tier_groups = df.groupby("tier")
    tier_agg = pd.DataFrame({
        "n_basins": tier_groups.size(),
        "mean_coverage": tier_groups["mean_coverage"].mean(),
        "median_coverage": tier_groups["mean_coverage"].median(),
        "std_coverage": tier_groups["mean_coverage"].std(),
        "mean_pi_width": tier_groups["mean_pi_width"].mean(),
        "median_pi_width": tier_groups["mean_pi_width"].median(),
        "mean_winkler_score": tier_groups["winkler_score"].mean(),
        "mean_eps": tier_groups["mean_coverage_eps"].mean(),
    })
    tier_order = ["dry", "semi_arid", "humid", "snow"]
    tier_agg = tier_agg.reindex(tier_order)
    ta = out_dir / "tier_aggregate.csv"
    tier_agg.to_csv(ta)

    spread_pp = (tier_agg["mean_coverage"].dropna().max() -
                 tier_agg["mean_coverage"].dropna().min()) * 100
    summary = {
        "experiment": "exp010",
        "action": "6_multiseed",
        "alpha": 0.1,
        "seed": args.seed,
        "basin_count_planned": 574,
        "basin_count_actual": int(len(df)),
        "basin_count_missing": int(len(missing)),
        "basin_count_extra": int(len(extra)),
        "tier_counts": df["tier"].value_counts().to_dict(),
        "overall": {
            "mean_coverage": float(df["mean_coverage"].mean()),
            "mean_coverage_eps_vs_0_90": float(df["mean_coverage_eps"].mean()),
            "mean_pi_width": float(df["mean_pi_width"].mean()),
            "mean_winkler_score": float(df["winkler_score"].mean()),
        },
        "per_tier_mean_coverage": {
            t: float(v) for t, v in tier_agg["mean_coverage"].dropna().items()
        },
        "per_tier_n_basins": {
            t: int(v) for t, v in tier_agg["n_basins"].dropna().items()
        },
        "tier_coverage_spread_pp": float(spread_pp),
        "log_path": str(log_path),
        "run_dir": args.run_dir,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[stage1 seed={args.seed}] === HEADLINE ===")
    print(f"  marginal coverage: {summary['overall']['mean_coverage']:.4f}")
    print(f"  spread:           {spread_pp:.2f}pp")
    print(f"  tiers: {summary['per_tier_mean_coverage']}")
    print(f"  wrote: {pb} {ta} {(out_dir / 'metrics.json')}")


if __name__ == "__main__":
    main()
