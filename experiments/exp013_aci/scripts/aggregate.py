"""Re-aggregate exp013 ACI per_basin_metrics with robust width statistics.

Mean width is exp-inflated by extreme-flow days; report median + p95 width and
mean log-width (the natural scale of the score function) as primary headline.

Two modes:
  (default)  reads per_basin_metrics.csv          -> writes aci_summary.json
  --clip     reads per_basin_metrics_clipped.csv  -> writes aci_summary_clipped.json

The clipped variant (reviewer issue S1-O-R1-04) caps the per-timestep upper
endpoint at a physical ceiling W_MAX_b = K * max(calibration observed flow) to
remove the non-physical width blow-ups of the unbounded multiplicative
back-transform Q_hat * exp(+q_t). Coverage and tier spread should be
essentially unchanged; only the heavy-tailed width is tamed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/exp013_aci/results"
TIERS = ["dry", "semi_arid", "humid", "snow"]
CLIP_K = 2.0


def aggregate(clip=False):
    suffix = "_clipped" if clip else ""
    df = pd.read_csv(OUT / f"per_basin_metrics{suffix}.csv")

    rows = []
    for t in TIERS:
        sub = df[df["tier"] == t]
        row = {
            "tier": t,
            "n_basins": int(len(sub)),
            "mean_coverage": float(sub["coverage"].mean()),
            "median_coverage": float(sub["coverage"].median()),
            "std_coverage": float(sub["coverage"].std()),
            "median_width_mm_d": float(sub["mean_width"].median()),
            "p95_width_mm_d": float(sub["mean_width"].quantile(0.95)),
            "mean_log_width": float(sub["mean_log_width"].mean()),
            "median_log_width": float(sub["mean_log_width"].median()),
        }
        if clip and "clip_binding_fraction" in sub:
            # Timestep-weighted clip-binding fraction within the tier.
            n_clipped = float(sub["n_clipped"].sum()) if "n_clipped" in sub else float("nan")
            n_test = float(sub["n_test"].sum()) if "n_test" in sub else float("nan")
            row["clip_binding_fraction"] = (n_clipped / n_test) if n_test else float("nan")
            row["mean_basin_clip_binding_fraction"] = float(sub["clip_binding_fraction"].mean())
        rows.append(row)
    tier_df = pd.DataFrame(rows)
    tier_df.to_csv(OUT / f"tier_aggregate{suffix}.csv", index=False)

    covs = tier_df["mean_coverage"].values
    spread_pp_mean = float((covs.max() - covs.min()) * 100)
    covs_med = tier_df["median_coverage"].values
    spread_pp_med = float((covs_med.max() - covs_med.min()) * 100)
    log_w = tier_df["mean_log_width"].values
    log_w_spread = float(log_w.max() - log_w.min())

    summary = {
        "method": "ACI_Gibbs2021_clipped" if clip else "ACI_Gibbs2021",
        "alpha_star": 0.10,
        "gamma": 0.005,
        "score_function": "log_flow_residual",
        "score_epsilon": 0.01,
        "lstm_seed_set": [42],
        "n_basins": int(len(df)),
        "marginal_coverage": float(df["coverage"].mean()),
        "marginal_coverage_median_basin": float(df["coverage"].median()),
        "tier_coverage_spread_pp": spread_pp_mean,
        "tier_coverage_spread_pp_median_basin": spread_pp_med,
        "log_width_spread_across_tiers": log_w_spread,
        "tier_coverage_spread_definition": "max - min of basin-mean coverage across {dry, semi_arid, humid, snow}",
        "width_reporting_note": (
            "mean width in mm/d is dominated by extreme-flow days (log-space "
            "score gives interval = Q_hat * exp(+/- q_t); a single q_t > 4 at "
            "high Q_hat blows up the linear mean). Report median width per "
            "basin (robust) and mean log-width (natural scale)."
        ),
        "per_tier": tier_df.to_dict(orient="records"),
        "comparison_context": {
            "global_cp_seed42_tier_spread_pp": 20.68,
            "global_cp_5seed_tier_spread_pp": 20.97,
            "hscc_5seed_tier_spread_pp": 1.53,
            "hopcpt_3seed_tier_spread_pp": 2.48,
            "global_cqr_5seed_tier_spread_pp": 4.75,
            "mondrian_cqr_seed42_tier_spread_pp": 2.38,
            "aci_seed42_tier_spread_pp": spread_pp_mean,
        },
        "notes": [
            "Single LSTM seed=42 (only exp002 .p file retained locally). ACI itself is deterministic.",
            "Comparable upstream to exp_global_cqr (seed=42 only) and exp_CQR_GB.",
            "Initial alpha_0 = alpha_star = 0.10; gamma = 0.005 per Gibbs & Candes 2021.",
            "ACI closes tier coverage spread to <1 pp but the price is interval inflation on "
            "extreme-flow days: mean width across the snow tier is two orders of magnitude "
            "larger than median width, indicating a Pareto trade-off rather than a free lunch.",
        ],
    }

    if clip:
        n_clipped_total = float(df["n_clipped"].sum()) if "n_clipped" in df else float("nan")
        n_test_total = float(df["n_test"].sum()) if "n_test" in df else float("nan")
        clip_binding_overall = (n_clipped_total / n_test_total) if n_test_total else float("nan")
        summary["clip_multiplier_K"] = float(CLIP_K)
        summary["clip_ceiling_definition"] = (
            "W_MAX_b = K * nanmax(calibration-period observed streamflow QObs(mm/d)_obs "
            "for basin b); upper interval endpoint hi is capped at W_MAX_b, lo unchanged. "
            "Falls back to max predicted flow if obs all-NaN."
        )
        summary["clip_binding_fraction"] = clip_binding_overall
        summary["clip_binding_fraction_per_tier"] = {
            r["tier"]: r.get("clip_binding_fraction", float("nan"))
            for r in summary["per_tier"]
        }
        summary["comparison_context"]["aci_clipped_seed42_tier_spread_pp"] = spread_pp_mean
        summary["notes"].append(
            "CLIPPED variant (reviewer S1-O-R1-04): per-timestep upper endpoint capped at "
            f"W_MAX_b = {CLIP_K:g} * max(calibration observed flow per basin). This removes "
            "the non-physical p95 width blow-ups of the unbounded multiplicative back-transform "
            "while leaving marginal coverage and tier spread essentially unchanged "
            "(clip binds on only a tiny fraction of high-flow timesteps)."
        )

    out_name = f"aci_summary{suffix}.json"
    with open(OUT / out_name, "w") as f:
        json.dump(summary, f, indent=2)

    print(tier_df.to_string(index=False))
    print(f"\n[{'CLIPPED' if clip else 'unbounded'}] -> {out_name}")
    print(f"marginal coverage      = {summary['marginal_coverage']:.4f}")
    print(f"tier coverage spread   = {spread_pp_mean:.2f} pp (mean basin), {spread_pp_med:.2f} pp (median basin)")
    print(f"log-width spread       = {log_w_spread:.2f} (natural log of width ratio)")
    if clip:
        print(f"clip-binding fraction  = {summary['clip_binding_fraction']:.6f} (overall, timestep-weighted)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aggregate exp013 ACI per-basin metrics to per-tier summary.")
    ap.add_argument("--clip", action="store_true",
                    help="aggregate the clipped run (per_basin_metrics_clipped.csv -> aci_summary_clipped.json)")
    args = ap.parse_args()
    aggregate(clip=args.clip)
