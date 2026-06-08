"""
exp010 Stage 1 — Data precheck.

Parse 574 per-basin evaluation metrics from production log, join with
CAMELS-US attributes (frac_snow, aridity), assign exp002 tier_scheme tiers,
and aggregate per-tier coverage / width / winkler. Reports cross-tier spread
(the §5.7 head-to-head core number).

Pure mechanical parsing — no judgement, per Step 07 Stage 1 protocol.
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/qingsong/桌面/代码仓库/Reliability-Control-Streamflow-CP")

LOG_PATH = REPO / "experiments/exp010/logs/production_20260503_120656.log"
NH_LIB = REPO / "libs/neuralhydrology"
CAMELS_DIR = REPO / "data/raw/CAMELS_US"
BASIN_LIST = REPO / "data/raw/exp010_basin_list_clean.txt"
OUT_DIR = REPO / "experiments/exp010/results/_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(NH_LIB))
from neuralhydrology.datasetzoo.camelsus import load_camels_us_attributes


def parse_log(log_path: Path) -> pd.DataFrame:
    """Walk log; pair each 'Start evaluation for TS <basin>...' with the
    NEXT 'Evaluation metrics (alpha 0.1: {...})' line."""
    text = log_path.read_text()
    start_re = re.compile(r"Start evaluation for TS (\d+)_[^ ]+ with alpha 0\.1")
    metrics_re = re.compile(r"Evaluation metrics \(alpha 0\.1: (\{.*\})\s*$", re.MULTILINE)

    starts = [(m.start(), m.group(1)) for m in start_re.finditer(text)]
    metrics_iter = list(metrics_re.finditer(text))

    rows = []
    for s_pos, basin in starts:
        m = next((mm for mm in metrics_iter if mm.start() > s_pos), None)
        if m is None:
            print(f"WARN: no metrics found after basin {basin}", file=sys.stderr)
            continue
        try:
            d = ast.literal_eval(m.group(1))
        except Exception as e:
            print(f"WARN: parse failed for basin {basin}: {e}", file=sys.stderr)
            continue
        rows.append({"basin": basin, **d})
    return pd.DataFrame(rows)


def assign_tier(row, snow_thresh=0.4, dry_thresh=1.5, semi_arid_thresh=1.0):
    """exp002/PROJECT_CHARTER tier_scheme: snow优先, then aridity bands."""
    if row["frac_snow"] >= snow_thresh:
        return "snow"
    a = row["aridity"]
    if a > dry_thresh:
        return "dry"
    if a > semi_arid_thresh:
        return "semi_arid"
    return "humid"


def main():
    print(f"[stage1] parsing log: {LOG_PATH}")
    df = parse_log(LOG_PATH)
    print(f"[stage1] parsed per-basin metrics: {len(df)} rows")

    basins = BASIN_LIST.read_text().strip().splitlines()
    print(f"[stage1] basin_list count: {len(basins)}")

    missing = set(basins) - set(df["basin"])
    extra = set(df["basin"]) - set(basins)
    if missing:
        print(f"WARN: {len(missing)} basins in list but missing from log: {sorted(missing)[:10]}")
    if extra:
        print(f"WARN: {len(extra)} basins in log but not in list: {sorted(extra)[:10]}")

    # NaN check on coverage / width
    for col in ["mean_coverage", "mean_pi_width", "winkler_score"]:
        n_nan = df[col].isna().sum()
        if n_nan:
            print(f"WARN: {col} has {n_nan} NaN values")

    # Load attributes
    attrs = load_camels_us_attributes(CAMELS_DIR, basins)[["frac_snow", "aridity", "p_mean", "pet_mean"]]
    df = df.merge(attrs, left_on="basin", right_index=True, how="left")
    n_attr_nan = df[["frac_snow", "aridity"]].isna().any(axis=1).sum()
    if n_attr_nan:
        print(f"WARN: {n_attr_nan} basins missing frac_snow/aridity")

    # Assign tier
    df["tier"] = df.apply(assign_tier, axis=1)
    print("[stage1] tier counts:")
    print(df["tier"].value_counts().to_string())

    # Save per-basin csv
    per_basin_path = OUT_DIR / "per_basin_metrics.csv"
    df.to_csv(per_basin_path, index=False)
    print(f"[stage1] wrote {per_basin_path} ({len(df)} rows × {df.shape[1]} cols)")

    # Per-tier aggregate
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
    print("\n[stage1] per-tier aggregate:")
    print(tier_agg.to_string())

    tier_agg_path = OUT_DIR / "tier_aggregate.csv"
    tier_agg.to_csv(tier_agg_path)
    print(f"[stage1] wrote {tier_agg_path}")

    # Cross-tier spread (the headline number for §5.7 head-to-head)
    tier_means = tier_agg["mean_coverage"].dropna()
    spread_pp = (tier_means.max() - tier_means.min()) * 100
    overall_coverage = df["mean_coverage"].mean()
    overall_eps = df["mean_coverage_eps"].mean()
    overall_width = df["mean_pi_width"].mean()
    overall_winkler = df["winkler_score"].mean()

    summary = {
        "experiment": "exp010",
        "method": "HopCPT (Auer 2024) — EpsPredictionHopfield + precomputed_nh on exp002 LSTM",
        "alpha": 0.1,
        "seed": 42,
        "basin_count_planned": 671,
        "basin_count_actual": int(len(df)),
        "basin_count_missing": int(len(missing)),
        "basin_count_extra": int(len(extra)),
        "tier_scheme": "snow_first then aridity (>1.5 dry / 1.0-1.5 semi_arid / <=1.0 humid); from PROJECT_CHARTER §2",
        "tier_counts": df["tier"].value_counts().to_dict(),
        "overall": {
            "mean_coverage": float(overall_coverage),
            "mean_coverage_eps_vs_0_90": float(overall_eps),
            "mean_pi_width": float(overall_width),
            "mean_winkler_score": float(overall_winkler),
        },
        "per_tier_mean_coverage": {tier: float(v) for tier, v in tier_agg["mean_coverage"].dropna().items()},
        "per_tier_n_basins": {tier: int(v) for tier, v in tier_agg["n_basins"].dropna().items()},
        "tier_coverage_spread_pp": float(spread_pp),
        "headline_comparison_NOTE": (
            "exp002 671-basin spread 19.86pp / HSCC 1.01pp; exp009 5-seed "
            "spread 20.97±1.22pp / HSCC 1.53±0.51pp. exp010 spread above is "
            "574-basin subset, NOT directly comparable until exp002/exp009 "
            "are recomputed on same 574-basin subset (CPU-only follow-up)."
        ),
        "production_run_dir": "experiments/exp010/results/run_030526_120659/",
        "wall_clock_min": 98,
        "log_path": str(LOG_PATH.relative_to(REPO)),
        "per_basin_csv": str(per_basin_path.relative_to(REPO)),
        "tier_aggregate_csv": str(tier_agg_path.relative_to(REPO)),
    }
    summary_path = OUT_DIR / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[stage1] wrote {summary_path}")

    print("\n=== HEADLINE NUMBERS ===")
    print(f"  Overall coverage:        {overall_coverage:.4f}  (target 0.90)")
    print(f"  Overall coverage eps:    {overall_eps:+.4f}")
    print(f"  Overall PI width:        {overall_width:.3f}")
    print(f"  Overall winkler score:   {overall_winkler:.3f}")
    print(f"  Per-tier coverage:")
    for tier in tier_order:
        v = tier_agg.loc[tier, "mean_coverage"] if tier in tier_agg.index else None
        n = tier_agg.loc[tier, "n_basins"] if tier in tier_agg.index else 0
        if v is not None and not pd.isna(v):
            print(f"    {tier:10s}: {v:.4f}  (n={int(n)})")
    print(f"  Cross-tier coverage spread: {spread_pp:.2f} pp")
    print("\n  vs. exp002 671-basin (Global CP 19.86pp / HSCC 1.01pp)")
    print("  vs. exp009 5-seed (Global CP 20.97±1.22pp / HSCC 1.53±0.51pp)")
    print("  NOTE: cross-method comparison requires same 574-basin subset (deferred).")


if __name__ == "__main__":
    main()
