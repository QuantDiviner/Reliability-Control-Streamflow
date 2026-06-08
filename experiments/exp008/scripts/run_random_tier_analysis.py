"""
exp008 — Random-Tier Control (combined script).

Reuses exp003's 18 LORO checkpoints + cal/test predictions; only the *tier
assignment* layer is randomized. This is a pure CPU analysis (no model
training).

Per fold (HUC-{01..18}):
  Load exp003 NH validation/test pickle files for that fold
  Build cal_records / tst_records (basin, obs, pred, scores, tier_physical)

Per repetition (rep ∈ 0..N_REPEATS-1):
  Generate two independent random tier label vectors (cal & test) that
    PRESERVE the per-tier basin counts of the physical labels in that fold
    (matched-margins null model — the strongest reasonable null).
  Build random per-tier q_tier from cal scores within each random group.
  Apply random per-tier q to test basins by their random tier label.
  Compute per-(random)tier coverage and spread (max-min over 4 random groups).

Per fold aggregate:
  Random distribution: mean ± std of HSCC spread, per tier coverage, etc.
  Compare to physical exp003 results (same fold).

Cross-fold:
  18 physical values vs 18×N random values per metric.
  Welch t-test + Cohen's d per tier.
  Welch t-test on cross-fold spread reduction.

Outputs:
  experiments/exp008/results/_analysis/random_distribution.json
    18-fold random vs physical comparison.
  experiments/exp008/results/_analysis/comparison_to_exp003.csv
    long-form table for plotting.
  experiments/exp008/results/_analysis/effect_size_table.csv
    per-tier t-stat + p-value + Cohen's d.
  experiments/exp008/results/_analysis/per_fold_random.json
    detailed per-fold random repetitions.

Usage:
  python experiments/exp008/scripts/run_random_tier_analysis.py [--n_repeats 20]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
EXP003_RESULTS = ROOT / "experiments" / "exp003" / "results"
EXP003_AGG = ROOT / "experiments" / "exp003" / "aggregated_metrics.json"
ATTR_FILE = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
OUT_DIR = ROOT / "experiments" / "exp008" / "results" / "_analysis"

ALPHA = 0.10
EPS = 0.01
TIERS = ["dry", "semi_arid", "humid", "snow"]

TIER_SNOW_FRAC = 0.40
TIER_DRY_AI = 1.5
TIER_SEMI_AI = 1.0


def assign_tier(ai: float, fs: float) -> str:
    if fs >= TIER_SNOW_FRAC:
        return "snow"
    if ai > TIER_DRY_AI:
        return "dry"
    if ai > TIER_SEMI_AI:
        return "semi_arid"
    return "humid"


def load_attributes() -> dict[str, dict[str, float]]:
    attrs: dict[str, dict[str, float]] = {}
    with open(ATTR_FILE) as f:
        hdr = f.readline().strip().split(";")
        ai_i, sn_i = hdr.index("aridity"), hdr.index("frac_snow")
        for line in f:
            cols = line.strip().split(";")
            if not cols or not cols[0]:
                continue
            attrs[cols[0].zfill(8)] = {
                "aridity": float(cols[ai_i]),
                "frac_snow": float(cols[sn_i]),
            }
    return attrs


def log_score(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6)))


def split_cp_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return float("nan")
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


def coverage_log(obs: np.ndarray, pred: np.ndarray, q: float) -> float:
    s = log_score(obs, pred)
    return float(np.mean(s <= q))


def extract_basin_series(results, basin_id):
    bd = results[basin_id]
    if isinstance(bd, dict) and "1D" in bd:
        ds = bd["1D"]["xr"]
    elif hasattr(bd, "xr"):
        ds = bd.xr
    else:
        raise ValueError(f"unknown result format for basin {basin_id}: {type(bd)}")
    obs = ds["QObs(mm/d)_obs"].values.flatten()
    pred = ds["QObs(mm/d)_sim"].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def find_latest_epoch(run_dir: Path, kind: str) -> Path | None:
    sub = run_dir / kind
    if not sub.exists():
        return None
    epochs = sorted(sub.glob("model_epoch*"))
    return epochs[-1] if epochs else None


def load_fold(huc: str, attrs: dict) -> dict | None:
    """Returns dict with cal_records (validation period) and tst_records
    (test period) for one fold, with physical tier labels preserved."""
    candidates = sorted(EXP003_RESULTS.glob(f"exp003_loro_huc{huc}_*"))
    if not candidates:
        return None
    run_dir = candidates[-1]
    val_epoch = find_latest_epoch(run_dir, "validation")
    test_epoch = find_latest_epoch(run_dir, "test")
    if val_epoch is None or test_epoch is None:
        return None
    with open(val_epoch / "validation_results.p", "rb") as f:
        val_res = pickle.load(f)
    with open(test_epoch / "test_results.p", "rb") as f:
        test_res = pickle.load(f)

    def make_records(res, kind: str) -> list[dict]:
        out = []
        for bid in res.keys():
            try:
                obs, pred = extract_basin_series(res, bid)
            except Exception:
                continue
            if len(obs) < 100:
                continue
            gid = str(bid).zfill(8)
            a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0})
            scores = log_score(obs, pred)
            out.append({
                "basin_id": gid,
                "tier_physical": assign_tier(a["aridity"], a["frac_snow"]),
                "obs": obs,
                "pred": pred,
                "scores": scores,
            })
        return out

    cal_records = make_records(val_res, "cal")
    tst_records = make_records(test_res, "test")
    return {"huc": huc, "run_dir": run_dir, "cal": cal_records, "tst": tst_records}


def assign_random_tiers(records: list[dict], rng: np.random.Generator) -> np.ndarray:
    """Permute physical tier labels among the records.

    This preserves the per-tier basin count exactly (matched-margins null),
    which is the strongest possible null model for detecting hydrological
    signal: any spread reduction beyond what permutation gives is purely
    structural."""
    n = len(records)
    physical = np.array([r["tier_physical"] for r in records])
    perm = rng.permutation(n)
    return physical[perm]


def per_tier_coverage_with_labels(
    cal_records: list[dict],
    tst_records: list[dict],
    cal_tier_labels: np.ndarray,
    tst_tier_labels: np.ndarray,
) -> dict:
    """Compute per-tier HSCC coverage given arbitrary tier label vectors.

    Returns dict with per_tier coverage + spread + global q + global cov for
    bookkeeping (global stats are independent of labels)."""
    # Global cal pool
    all_cal = np.concatenate([r["scores"] for r in cal_records]) if cal_records else np.array([])
    q_global = split_cp_quantile(all_cal, ALPHA)

    # Per-tier q from cal pool
    q_tier: dict[str, float] = {}
    for t in TIERS:
        mask = cal_tier_labels == t
        if not mask.any():
            q_tier[t] = float("nan")
            continue
        s = np.concatenate([cal_records[i]["scores"] for i in np.where(mask)[0]])
        q_tier[t] = split_cp_quantile(s, ALPHA)

    # Per-tier cov on test (basin-mean within tier)
    per_tier: dict[str, dict] = {}
    for t in TIERS:
        mask = tst_tier_labels == t
        if not mask.any() or np.isnan(q_tier[t]):
            continue
        idxs = np.where(mask)[0]
        cgs, chs = [], []
        for i in idxs:
            r = tst_records[i]
            cgs.append(coverage_log(r["obs"], r["pred"], q_global))
            chs.append(coverage_log(r["obs"], r["pred"], q_tier[t]))
        per_tier[t] = {
            "n_basins": int(len(idxs)),
            "global_coverage": float(np.mean(cgs)),
            "hscc_coverage": float(np.mean(chs)),
        }

    # Spread (basin-weighted across tiers if all 4 present)
    coverages_global = [per_tier[t]["global_coverage"] for t in TIERS if t in per_tier]
    coverages_hscc = [per_tier[t]["hscc_coverage"] for t in TIERS if t in per_tier]
    spread_global = (max(coverages_global) - min(coverages_global)
                     if len(coverages_global) >= 2 else 0.0)
    spread_hscc = (max(coverages_hscc) - min(coverages_hscc)
                   if len(coverages_hscc) >= 2 else 0.0)

    return {
        "q_global": q_global,
        "per_tier": per_tier,
        "spread_global_pp": spread_global * 100,
        "spread_hscc_pp": spread_hscc * 100,
        "spread_reduction_pp": (spread_global - spread_hscc) * 100,
    }


def run_fold(fold: dict, n_repeats: int, base_seed: int = 42) -> dict:
    huc = fold["huc"]
    cal_records = fold["cal"]
    tst_records = fold["tst"]
    cal_phys = np.array([r["tier_physical"] for r in cal_records])
    tst_phys = np.array([r["tier_physical"] for r in tst_records])

    # Physical baseline
    physical = per_tier_coverage_with_labels(cal_records, tst_records, cal_phys, tst_phys)

    # Random repetitions
    random_runs = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(base_seed + int(huc) * 1000 + rep)
        cal_rand = assign_random_tiers(cal_records, rng)
        tst_rand = assign_random_tiers(tst_records, rng)
        random_runs.append(
            per_tier_coverage_with_labels(cal_records, tst_records, cal_rand, tst_rand)
        )

    # Aggregate random distribution
    random_summary = {
        "n_repeats": n_repeats,
        "spread_hscc_pp_mean": float(np.mean([r["spread_hscc_pp"] for r in random_runs])),
        "spread_hscc_pp_std": float(np.std([r["spread_hscc_pp"] for r in random_runs])),
        "spread_reduction_pp_mean": float(np.mean([r["spread_reduction_pp"] for r in random_runs])),
        "spread_reduction_pp_std": float(np.std([r["spread_reduction_pp"] for r in random_runs])),
        "per_tier_hscc_cov_mean": {
            t: float(np.mean([r["per_tier"].get(t, {}).get("hscc_coverage", np.nan)
                              for r in random_runs])) for t in TIERS
        },
        "per_tier_hscc_cov_std": {
            t: float(np.std([r["per_tier"].get(t, {}).get("hscc_coverage", np.nan)
                             for r in random_runs])) for t in TIERS
        },
    }

    return {
        "huc": huc,
        "physical": physical,
        "random": random_summary,
        "random_raw": [
            {"spread_hscc_pp": r["spread_hscc_pp"],
             "spread_reduction_pp": r["spread_reduction_pp"]}
            for r in random_runs
        ],
    }


def cross_fold_analysis(fold_results: list[dict]) -> dict:
    """Welch t-test + Cohen's d for physical vs random across 18 folds."""
    physical_spread_reduction = np.array([f["physical"]["spread_reduction_pp"] for f in fold_results])
    physical_spread_hscc = np.array([f["physical"]["spread_hscc_pp"] for f in fold_results])

    # Random pool: 18 folds × N reps reductions, flattened
    random_spread_reduction_all = []
    random_spread_hscc_all = []
    for f in fold_results:
        for r in f["random_raw"]:
            random_spread_reduction_all.append(r["spread_reduction_pp"])
            random_spread_hscc_all.append(r["spread_hscc_pp"])
    random_spread_reduction_all = np.array(random_spread_reduction_all)
    random_spread_hscc_all = np.array(random_spread_hscc_all)

    # Welch t-test on cross-fold spread reduction (one-sided: physical > random)
    t_stat, p_two_sided = stats.ttest_ind(
        physical_spread_reduction, random_spread_reduction_all,
        equal_var=False, alternative="greater"
    )
    pooled_std = np.sqrt(0.5 * (physical_spread_reduction.var(ddof=1)
                                + random_spread_reduction_all.var(ddof=1)))
    cohens_d = ((physical_spread_reduction.mean() - random_spread_reduction_all.mean())
                / pooled_std) if pooled_std > 0 else float("nan")

    # Per-tier comparison: physical per-tier coverage vs random per-tier coverage mean
    per_tier_test = {}
    for t in TIERS:
        phys_vals = []
        rand_vals = []
        for f in fold_results:
            p_cov = f["physical"]["per_tier"].get(t, {}).get("hscc_coverage")
            if p_cov is not None:
                phys_vals.append(p_cov)
            r_cov_mean = f["random"]["per_tier_hscc_cov_mean"].get(t)
            if r_cov_mean is not None and not np.isnan(r_cov_mean):
                rand_vals.append(r_cov_mean)
        if not phys_vals or not rand_vals:
            per_tier_test[t] = {"note": "insufficient data"}
            continue
        phys_arr = np.array(phys_vals)
        rand_arr = np.array(rand_vals)
        t_t, p_t = stats.ttest_ind(phys_arr, rand_arr, equal_var=False)
        per_tier_test[t] = {
            "n_folds_with_tier": int(len(phys_arr)),
            "physical_cov_mean": float(phys_arr.mean()),
            "physical_cov_std": float(phys_arr.std(ddof=1)) if len(phys_arr) > 1 else 0.0,
            "random_cov_mean": float(rand_arr.mean()),
            "random_cov_std": float(rand_arr.std(ddof=1)) if len(rand_arr) > 1 else 0.0,
            "abs_diff_pp": float(abs(phys_arr.mean() - rand_arr.mean()) * 100),
            "t_statistic": float(t_t),
            "p_value_two_sided": float(p_t),
        }

    # Plan §3 success criteria
    s1_per_tier_pass = {}
    for t, td in per_tier_test.items():
        if "note" in td:
            s1_per_tier_pass[t] = None
            continue
        # S1 interpretation: physical |cov - 0.9| < random |cov - 0.9| - 2σ_random?
        # Re-cast as "physical's deviation from 0.9 is smaller than random by ≥ 2 sd"
        phys_dev = abs(td["physical_cov_mean"] - 0.90)
        rand_dev = abs(td["random_cov_mean"] - 0.90)
        rand_sd = td["random_cov_std"]
        s1_per_tier_pass[t] = bool(phys_dev < rand_dev - 2 * rand_sd) if rand_sd > 0 else None

    # S3: random mean spread reduction ≤ 50% of physical mean
    s3_pass = bool(random_spread_reduction_all.mean()
                   <= 0.5 * physical_spread_reduction.mean())

    # S4: random spread reduction not significantly different from 0 (one-sample)
    t4, p4 = stats.ttest_1samp(random_spread_reduction_all, 0.0)
    s4_pass = bool(p4 > 0.05)

    return {
        "physical_spread_reduction_pp_mean": float(physical_spread_reduction.mean()),
        "physical_spread_reduction_pp_std": float(physical_spread_reduction.std(ddof=1)),
        "random_spread_reduction_pp_mean": float(random_spread_reduction_all.mean()),
        "random_spread_reduction_pp_std": float(random_spread_reduction_all.std(ddof=1)),
        "welch_t_stat": float(t_stat),
        "welch_p_value_one_sided_phys_gt_rand": float(p_two_sided),  # already one-sided greater
        "cohens_d_phys_vs_rand": float(cohens_d),
        "per_tier_comparison": per_tier_test,
        "plan_s1_per_tier_pass": s1_per_tier_pass,
        "plan_s2_overall_p_lt_0_01": bool(p_two_sided < 0.01),
        "plan_s3_random_le_50pct_physical": s3_pass,
        "plan_s4_random_not_diff_from_zero": s4_pass,
        "n_random_total": int(len(random_spread_reduction_all)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_repeats", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading attributes from {ATTR_FILE}")
    attrs = load_attributes()
    print(f"loaded {len(attrs)} basins")

    fold_results = []
    t0 = time.time()
    for k in range(1, 19):
        huc = f"{k:02d}"
        print(f"\n[{k}/18] HUC-{huc}", flush=True)
        fold = load_fold(huc, attrs)
        if fold is None:
            print(f"  WARN: no exp003 results found, skipping")
            continue
        print(f"  cal {len(fold['cal'])} basins, test {len(fold['tst'])} basins")
        fr = run_fold(fold, args.n_repeats, args.base_seed)
        # Free pickles after fold run
        del fold
        phys = fr["physical"]
        rnd = fr["random"]
        print(f"  physical: spread_HSCC={phys['spread_hscc_pp']:.2f}pp, "
              f"reduction={phys['spread_reduction_pp']:.2f}pp")
        print(f"  random  : spread_HSCC={rnd['spread_hscc_pp_mean']:.2f}±{rnd['spread_hscc_pp_std']:.2f}pp, "
              f"reduction={rnd['spread_reduction_pp_mean']:.2f}±{rnd['spread_reduction_pp_std']:.2f}pp")
        fold_results.append(fr)
    elapsed = time.time() - t0
    print(f"\nfolds processed: {len(fold_results)}/18 in {elapsed:.0f}s")

    if not fold_results:
        sys.exit("no folds processed — abort")

    # Cross-fold statistics
    print("\n=== cross-fold analysis ===")
    cross = cross_fold_analysis(fold_results)
    print(f"physical mean spread reduction: {cross['physical_spread_reduction_pp_mean']:.2f}pp "
          f"± {cross['physical_spread_reduction_pp_std']:.2f}")
    print(f"random   mean spread reduction: {cross['random_spread_reduction_pp_mean']:.2f}pp "
          f"± {cross['random_spread_reduction_pp_std']:.2f}")
    print(f"Welch t = {cross['welch_t_stat']:.3f}, p (one-sided phys > rand) = "
          f"{cross['welch_p_value_one_sided_phys_gt_rand']:.4f}, "
          f"Cohen's d = {cross['cohens_d_phys_vs_rand']:.2f}")
    print(f"Plan §3 S2 (overall p < 0.01)        : {'✅' if cross['plan_s2_overall_p_lt_0_01'] else '❌'}")
    print(f"Plan §3 S3 (random ≤ 50% physical)   : {'✅' if cross['plan_s3_random_le_50pct_physical'] else '❌'}")
    print(f"Plan §3 S4 (random not diff from 0)  : {'✅' if cross['plan_s4_random_not_diff_from_zero'] else '❌'}")
    print(f"Plan §3 S1 per-tier (phys dev < rand dev - 2σ):")
    for t, p in cross["plan_s1_per_tier_pass"].items():
        print(f"  {t:10s} : {'✅' if p else '❌' if p is False else 'N/A'}")

    # Save outputs
    out_distribution = {
        "n_repeats": args.n_repeats,
        "base_seed": args.base_seed,
        "n_folds_processed": len(fold_results),
        "elapsed_seconds": elapsed,
        "cross_fold": cross,
        "per_fold": [
            {"huc": f["huc"], "physical": f["physical"], "random": f["random"]}
            for f in fold_results
        ],
    }
    with open(OUT_DIR / "random_distribution.json", "w") as f:
        json.dump(out_distribution, f, indent=2)

    # Long-form CSV for plotting
    rows = []
    for f in fold_results:
        rows.append({"huc": f["huc"], "kind": "physical",
                     "spread_hscc_pp": f["physical"]["spread_hscc_pp"],
                     "spread_reduction_pp": f["physical"]["spread_reduction_pp"]})
        for rep_idx, r in enumerate(f["random_raw"]):
            rows.append({"huc": f["huc"], "kind": f"random_rep_{rep_idx}",
                         "spread_hscc_pp": r["spread_hscc_pp"],
                         "spread_reduction_pp": r["spread_reduction_pp"]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "comparison_to_exp003.csv", index=False)

    # Effect size table per tier
    rows = []
    for t, td in cross["per_tier_comparison"].items():
        if "note" in td:
            rows.append({"tier": t, "note": td["note"]})
            continue
        rows.append({
            "tier": t,
            "n_folds_with_tier": td["n_folds_with_tier"],
            "physical_cov_mean": td["physical_cov_mean"],
            "physical_cov_std": td["physical_cov_std"],
            "random_cov_mean": td["random_cov_mean"],
            "random_cov_std": td["random_cov_std"],
            "abs_diff_pp": td["abs_diff_pp"],
            "t_statistic": td["t_statistic"],
            "p_value_two_sided": td["p_value_two_sided"],
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "effect_size_table.csv", index=False)

    print(f"\nwrote {OUT_DIR/'random_distribution.json'}")
    print(f"wrote {OUT_DIR/'comparison_to_exp003.csv'}")
    print(f"wrote {OUT_DIR/'effect_size_table.csv'}")


if __name__ == "__main__":
    main()
