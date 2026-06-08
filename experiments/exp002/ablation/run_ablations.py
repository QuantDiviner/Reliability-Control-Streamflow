"""
exp002 Ablation Suite Runner — 4 项 ablation 统一执行器

加载 exp002 完整 NH validation_results.p + test_results.p 一次到内存，
随后 4 项 ablation 直接在内存中操作 tier 标签 / basin 子集，避免重复加载。

Ablations:
  A1: random-tier permutation (N=50)
  A2: boundary sensitivity (5 aridity × 3 snow combos)
  A3: leave-one-regime-out (4 leave-out variants)
  A4: min-cal-size stress (sizes={25,50,100,200} × N=20 reps × 4 tier)

Usage:
  /home/qingsong/miniconda3/envs/hscc-hydrology/bin/python \
    experiments/exp002/ablation/run_ablations.py \
    --run_dir experiments/exp002/results/exp002_camels_us_temporal_2504_155058 \
    --ablations A1,A2,A3,A4 \
    --out_root experiments/exp002/ablation
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/exp002"))
from hscc_analysis_v2 import (
    ALPHA, EPS,
    load_attributes, assign_tier,
    log_score, split_cp_quantile,
    coverage_log, width_orig,
    extract_basin_series, load_nh_pickle, find_latest_epoch,
)

TIERS = ["dry", "semi_arid", "humid", "snow"]


# -----------------------------------------------------------------------------
# Data loading (一次性，所有 ablation 共用)
# -----------------------------------------------------------------------------

def load_records(run_dir: Path):
    """Load NH val/test pickle and extract per-basin (cal_scores, test_obs, test_pred, tier)."""
    val_epoch = find_latest_epoch(run_dir, "validation")
    test_epoch = find_latest_epoch(run_dir, "test")
    print(f"[load] val: {val_epoch.name}, test: {test_epoch.name}")
    val_res = load_nh_pickle(val_epoch, "validation")
    test_res = load_nh_pickle(test_epoch, "test")
    attrs = load_attributes()

    records = []
    for basin_id in val_res.keys():
        if basin_id not in test_res:
            continue
        try:
            cal_obs, cal_pred = extract_basin_series(val_res, basin_id)
            tst_obs, tst_pred = extract_basin_series(test_res, basin_id)
        except Exception:
            continue
        if len(cal_obs) < 100 or len(tst_obs) < 100:
            continue
        gid = str(basin_id).zfill(8)
        a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0})
        records.append({
            "basin_id": gid,
            "tier": assign_tier(a["aridity"], a["frac_snow"]),
            "aridity": a["aridity"],
            "frac_snow": a["frac_snow"],
            "cal_scores": log_score(cal_obs, cal_pred),
            "test_obs": tst_obs,
            "test_pred": tst_pred,
        })
    print(f"[load] {len(records)} basins loaded")
    return records


# -----------------------------------------------------------------------------
# Common HSCC computation given a per-basin tier assignment
# -----------------------------------------------------------------------------

def compute_hscc_metrics(records, tier_field="tier", q_global=None):
    """Compute per-tier coverage given tier_field on each record.
    If q_global given, also compute global coverage in same set; else recompute it.
    Returns: dict {per_tier: {tier: {n, hscc_cov, global_cov, hscc_q}}, spread_global, spread_hscc}
    """
    if q_global is None:
        all_cal = np.concatenate([r["cal_scores"] for r in records])
        q_global = split_cp_quantile(all_cal, ALPHA)

    # per-tier q
    q_tier = {}
    for t in TIERS:
        s_list = [r["cal_scores"] for r in records if r[tier_field] == t]
        if s_list:
            s = np.concatenate(s_list)
            q_tier[t] = split_cp_quantile(s, ALPHA)
        else:
            q_tier[t] = np.nan

    # per-tier coverage
    per_tier = {}
    for t in TIERS:
        rs = [r for r in records if r[tier_field] == t]
        if not rs:
            continue
        cgs = [coverage_log(r["test_obs"], r["test_pred"], q_global) for r in rs]
        chs = [coverage_log(r["test_obs"], r["test_pred"], q_tier[t]) for r in rs]
        per_tier[t] = {
            "n_basins": len(rs),
            "global_coverage": float(np.mean(cgs)),
            "hscc_coverage": float(np.mean(chs)),
            "hscc_q": float(q_tier[t]),
        }

    cov_g = [v["global_coverage"] for v in per_tier.values()]
    cov_h = [v["hscc_coverage"] for v in per_tier.values()]
    spread_g = (max(cov_g) - min(cov_g)) if cov_g else 0
    spread_h = (max(cov_h) - min(cov_h)) if cov_h else 0
    return {
        "per_tier": per_tier,
        "spread_global_pp": float(spread_g * 100),
        "spread_hscc_pp": float(spread_h * 100),
        "spread_reduction_pp": float((spread_g - spread_h) * 100),
        "q_global": float(q_global),
    }


# -----------------------------------------------------------------------------
# A1 — Random-Tier Permutation (N=50)
# -----------------------------------------------------------------------------

def ablation_A1(records, out_dir, N=50, seed=42):
    print(f"\n=== A1: Random-Tier Permutation (N={N}) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Real baseline (exp002 actual tier)
    real = compute_hscc_metrics(records, tier_field="tier")
    print(f"[A1] real spread_reduction = {real['spread_reduction_pp']:.2f}pp "
          f"(global {real['spread_global_pp']:.2f} → HSCC {real['spread_hscc_pp']:.2f})")

    # Real per-tier sizes (preserve)
    tier_counts = {t: sum(1 for r in records if r["tier"] == t) for t in TIERS}
    print(f"[A1] preserve per-tier sizes: {tier_counts}")

    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(N):
        # generate random tier assignment with same per-tier counts
        labels = []
        for t, n in tier_counts.items():
            labels.extend([t] * n)
        labels = np.array(labels)
        rng.shuffle(labels)
        # assign to records
        for r, lab in zip(records, labels):
            r["_random_tier"] = lab
        m = compute_hscc_metrics(records, tier_field="_random_tier", q_global=real["q_global"])
        rows.append({
            "rep": rep,
            "spread_global_pp": m["spread_global_pp"],
            "spread_hscc_pp": m["spread_hscc_pp"],
            "spread_reduction_pp": m["spread_reduction_pp"],
        })
        if (rep + 1) % 10 == 0:
            print(f"  [A1] rep {rep+1}/{N} done")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "random_tier_distribution.csv", index=False)

    rand_mean = df["spread_reduction_pp"].mean()
    rand_sd = df["spread_reduction_pp"].std()
    real_red = real["spread_reduction_pp"]
    z = (real_red - rand_mean) / rand_sd if rand_sd > 0 else float("inf")
    p_one_sided = float(np.mean(df["spread_reduction_pp"].values >= real_red))

    results = {
        "ablation": "A1_random_tier_permutation",
        "N": N,
        "seed": seed,
        "real_spread_reduction_pp": real_red,
        "real_spread_global_pp": real["spread_global_pp"],
        "real_spread_hscc_pp": real["spread_hscc_pp"],
        "random_mean_pp": float(rand_mean),
        "random_sd_pp": float(rand_sd),
        "random_min_pp": float(df["spread_reduction_pp"].min()),
        "random_max_pp": float(df["spread_reduction_pp"].max()),
        "z_score_real_vs_random": float(z),
        "empirical_p_value_one_sided": p_one_sided,  # P(random >= real)
        "success_A1_1_real_gt_mean_3sd": bool(real_red > rand_mean + 3 * rand_sd),
        "success_A1_2_random_mean_near_zero": bool(abs(rand_mean) <= 2.0),
        "success_A1_3_p_lt_001": bool(p_one_sided < 0.001),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[A1] real spread_reduction = {real_red:.2f}pp")
    print(f"[A1] random distribution: mean = {rand_mean:.2f}pp, sd = {rand_sd:.2f}pp")
    print(f"[A1] z-score = {z:.2f} | p (one-sided) = {p_one_sided:.4f}")
    print(f"[A1] A1.1 (real > mean + 3sd): {results['success_A1_1_real_gt_mean_3sd']}")
    print(f"[A1] A1.2 (random mean ≈ 0): {results['success_A1_2_random_mean_near_zero']}")
    print(f"[A1] A1.3 (p < 0.001): {results['success_A1_3_p_lt_001']}")
    print(f"[A1] wrote: {out_dir/'random_tier_distribution.csv'}, results.json")
    return results


# -----------------------------------------------------------------------------
# A2 — Tier Boundary Sensitivity (5 aridity × 3 snow = 15 combos)
# -----------------------------------------------------------------------------

def assign_tier_custom(ai, fs, snow_th, dry_th, semi_th):
    if fs >= snow_th:
        return "snow"
    if ai > dry_th:
        return "dry"
    if ai > semi_th:
        return "semi_arid"
    return "humid"


def ablation_A2(records, out_dir):
    print(f"\n=== A2: Tier Boundary Sensitivity ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    # baseline boundary
    baseline = (0.40, 1.5, 1.0)  # snow, dry, semi
    # perturbations: aridity (dry, semi) jointly slide; snow alone
    aridity_pairs = [
        (1.5, 1.0),    # baseline
        (1.45, 0.95),  # -0.05
        (1.55, 1.05),  # +0.05
        (1.40, 0.90),  # -0.10
        (1.60, 1.10),  # +0.10
    ]
    snow_thresholds = [0.30, 0.40, 0.50]

    real = compute_hscc_metrics(records, tier_field="tier")

    rows = []
    for snow_th in snow_thresholds:
        for dry_th, semi_th in aridity_pairs:
            for r in records:
                r["_tier_var"] = assign_tier_custom(r["aridity"], r["frac_snow"],
                                                    snow_th, dry_th, semi_th)
            m = compute_hscc_metrics(records, tier_field="_tier_var", q_global=real["q_global"])
            tier_counts = {t: sum(1 for r in records if r["_tier_var"] == t) for t in TIERS}
            rows.append({
                "snow_th": snow_th, "dry_th": dry_th, "semi_th": semi_th,
                "is_baseline": (snow_th == 0.40 and dry_th == 1.5 and semi_th == 1.0),
                "n_dry": tier_counts["dry"],
                "n_semi": tier_counts["semi_arid"],
                "n_humid": tier_counts["humid"],
                "n_snow": tier_counts["snow"],
                "spread_global_pp": m["spread_global_pp"],
                "spread_hscc_pp": m["spread_hscc_pp"],
                "spread_reduction_pp": m["spread_reduction_pp"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "boundary_sweep.csv", index=False)

    # success: at least 13/15 combos with spread_reduction >= 15pp
    n_pass_15 = int((df["spread_reduction_pp"] >= 15.0).sum())
    n_total = len(df)
    spread_min = float(df["spread_reduction_pp"].min())
    spread_max = float(df["spread_reduction_pp"].max())
    spread_range = spread_max - spread_min

    results = {
        "ablation": "A2_boundary_sensitivity",
        "n_combos": n_total,
        "spread_reduction_min_pp": spread_min,
        "spread_reduction_max_pp": spread_max,
        "spread_reduction_range_pp": float(spread_range),
        "n_combos_pass_15pp": n_pass_15,
        "success_A2_1_at_least_13_of_15_pass": bool(n_pass_15 >= 13),
        "success_A2_3_min_above_12pp": bool(spread_min >= 12.0),
        "baseline_spread_reduction_pp": float(df.loc[df["is_baseline"], "spread_reduction_pp"].iloc[0]),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[A2] {n_total} combos, spread_reduction range [{spread_min:.2f}, {spread_max:.2f}]pp")
    print(f"[A2] {n_pass_15}/{n_total} combos with spread_reduction ≥ 15pp")
    print(f"[A2] A2.1 (≥ 13/15 pass): {results['success_A2_1_at_least_13_of_15_pass']}")
    print(f"[A2] A2.3 (min ≥ 12pp): {results['success_A2_3_min_above_12pp']}")
    print(f"[A2] wrote: {out_dir/'boundary_sweep.csv'}, results.json")
    return results


# -----------------------------------------------------------------------------
# A3 — Leave-One-Regime-Out (4 variants)
# -----------------------------------------------------------------------------

def ablation_A3(records, out_dir):
    print(f"\n=== A3: Leave-One-Regime-Out ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    real = compute_hscc_metrics(records, tier_field="tier")
    rows = []
    for hold_out in TIERS:
        sub = [r for r in records if r["tier"] != hold_out]
        if not sub:
            continue
        m = compute_hscc_metrics(sub, tier_field="tier")
        rows.append({
            "hold_out_tier": hold_out,
            "n_remaining_basins": len(sub),
            "spread_global_pp": m["spread_global_pp"],
            "spread_hscc_pp": m["spread_hscc_pp"],
            "spread_reduction_pp": m["spread_reduction_pp"],
            "remaining_tier_dry_cov": m["per_tier"].get("dry", {}).get("hscc_coverage"),
            "remaining_tier_semi_cov": m["per_tier"].get("semi_arid", {}).get("hscc_coverage"),
            "remaining_tier_humid_cov": m["per_tier"].get("humid", {}).get("hscc_coverage"),
            "remaining_tier_snow_cov": m["per_tier"].get("snow", {}).get("hscc_coverage"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "leave_one_out.csv", index=False)

    n_pass_12 = int((df["spread_reduction_pp"] >= 12.0).sum())
    humid_holdout_spread = float(df.loc[df["hold_out_tier"] == "humid", "spread_reduction_pp"].iloc[0])

    results = {
        "ablation": "A3_leave_one_regime_out",
        "real_spread_reduction_pp": real["spread_reduction_pp"],
        "leave_out_results": df.to_dict(orient="records"),
        "n_pass_12pp": n_pass_12,
        "humid_holdout_spread_reduction_pp": humid_holdout_spread,
        "success_A3_1_at_least_3_of_4_pass": bool(n_pass_12 >= 3),
        "success_A3_2_humid_holdout_above_8pp": bool(humid_holdout_spread >= 8.0),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[A3] real (full) spread_reduction = {real['spread_reduction_pp']:.2f}pp")
    for r in rows:
        print(f"  hold-out {r['hold_out_tier']:10s}: spread_reduction = {r['spread_reduction_pp']:.2f}pp")
    print(f"[A3] A3.1 (≥ 3/4 pass 12pp): {results['success_A3_1_at_least_3_of_4_pass']}")
    print(f"[A3] A3.2 (humid hold-out ≥ 8pp): {results['success_A3_2_humid_holdout_above_8pp']}")
    print(f"[A3] wrote: {out_dir/'leave_one_out.csv'}, results.json")
    return results


# -----------------------------------------------------------------------------
# A4 — Min-Cal-Size Stress
# -----------------------------------------------------------------------------

def ablation_A4(records, out_dir, sizes=(25, 50, 100, 200), N=20, seed=2026):
    print(f"\n=== A4: Min-Cal-Size Stress (sizes={sizes}, N={N} per size per tier) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    # We sub-sample basins (not days) within each tier for cal scores; test set kept whole
    rng = np.random.default_rng(seed)
    real = compute_hscc_metrics(records, tier_field="tier")
    q_global = real["q_global"]

    # Group basin records by tier
    by_tier = {t: [r for r in records if r["tier"] == t] for t in TIERS}
    actual_n = {t: len(by_tier[t]) for t in TIERS}
    print(f"[A4] actual per-tier basin counts: {actual_n}")

    rows = []
    for cal_size in sizes:
        for t in TIERS:
            n_avail = actual_n[t]
            if cal_size > n_avail:
                # Skip if requested cal_size > available basins
                continue
            for rep in range(N):
                idx = rng.choice(n_avail, size=cal_size, replace=False)
                cal_basins = [by_tier[t][i] for i in idx]
                cal_scores_pool = np.concatenate([r["cal_scores"] for r in cal_basins])
                q_t = split_cp_quantile(cal_scores_pool, ALPHA)
                # apply q_t to all test basins of this tier
                covs = [coverage_log(r["test_obs"], r["test_pred"], q_t) for r in by_tier[t]]
                rows.append({
                    "tier": t,
                    "cal_size_basins": cal_size,
                    "rep": rep,
                    "hscc_q": float(q_t),
                    "hscc_coverage_mean": float(np.mean(covs)),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "min_cal_size_sweep.csv", index=False)

    # Aggregate: mean ± sd of coverage per (tier, cal_size)
    agg = df.groupby(["tier", "cal_size_basins"])["hscc_coverage_mean"].agg(["mean", "std", "min", "max"]).reset_index()
    agg.to_csv(out_dir / "min_cal_size_aggregate.csv", index=False)

    # Success criteria
    a4_1_pass = True  # min_cal_size=100: all 4 tiers in [0.86, 0.92]
    a4_2_pass = True  # min_cal_size=50: all 4 tiers in [0.83, 0.93]
    a4_3_count = 0    # min_cal_size=25: at least 3/4 in [0.80, 0.95]

    for t in TIERS:
        for size, lo, hi, criterion in [(100, 0.86, 0.92, "A4.1"),
                                          (50, 0.83, 0.93, "A4.2"),
                                          (25, 0.80, 0.95, "A4.3")]:
            sub = agg[(agg["tier"] == t) & (agg["cal_size_basins"] == size)]
            if len(sub) == 0:
                continue
            mean_cov = float(sub["mean"].iloc[0])
            in_range = (lo <= mean_cov <= hi)
            if criterion == "A4.1" and not in_range:
                a4_1_pass = False
            if criterion == "A4.2" and not in_range:
                a4_2_pass = False
            if criterion == "A4.3" and in_range:
                a4_3_count += 1

    results = {
        "ablation": "A4_min_cal_size_stress",
        "sizes_tested": list(sizes),
        "N_reps": N,
        "actual_n_basins_per_tier": actual_n,
        "aggregate_summary": agg.to_dict(orient="records"),
        "success_A4_1_size100_in_086_092_all_tiers": a4_1_pass,
        "success_A4_2_size50_in_083_093_all_tiers": a4_2_pass,
        "success_A4_3_size25_in_080_095_at_least_3_of_4": bool(a4_3_count >= 3),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[A4] Coverage by (tier, cal_size):")
    for _, row in agg.iterrows():
        print(f"  {row['tier']:10s} | size={int(row['cal_size_basins']):3d} | "
              f"mean={row['mean']:.3f} ± {row['std']:.3f} (n={int(row['cal_size_basins'])})")
    print(f"[A4] A4.1 (size=100, ∈ [0.86, 0.92]): {a4_1_pass}")
    print(f"[A4] A4.2 (size=50,  ∈ [0.83, 0.93]): {a4_2_pass}")
    print(f"[A4] A4.3 (size=25,  ≥ 3/4 ∈ [0.80, 0.95]): {a4_3_count}/4")
    print(f"[A4] wrote: {out_dir/'min_cal_size_sweep.csv'}, aggregate, results.json")
    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--out_root", default="experiments/exp002/ablation", type=str)
    parser.add_argument("--ablations", default="A1,A2,A3,A4", type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_root = Path(args.out_root).resolve()
    to_run = set(a.strip() for a in args.ablations.split(","))

    records = load_records(run_dir)

    summary = {"run_dir": str(run_dir), "n_basins": len(records), "ablations_run": list(to_run)}
    if "A1" in to_run:
        summary["A1"] = ablation_A1(records, out_root / "01_random_tier_permutation_results")
    if "A2" in to_run:
        summary["A2"] = ablation_A2(records, out_root / "02_boundary_sensitivity_results")
    if "A3" in to_run:
        summary["A3"] = ablation_A3(records, out_root / "03_leave_one_regime_out_results")
    if "A4" in to_run:
        summary["A4"] = ablation_A4(records, out_root / "04_min_cal_size_stress_results")

    with open(out_root / "ABLATION_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[main] wrote: {out_root/'ABLATION_SUMMARY.json'}")


if __name__ == "__main__":
    main()
