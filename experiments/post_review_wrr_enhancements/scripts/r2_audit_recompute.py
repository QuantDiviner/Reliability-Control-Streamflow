#!/usr/bin/env python3
"""
R2 reviewer-audit recomputations (2026-06-04).

Addresses the WRR internal-review report items that need new numbers, all from
RETAINED data (no LSTM retraining):

  M1  Boundary-sensitivity failing-combo facts (from regenerated boundary_sweep.csv).
  M2  Out-of-sample screening: basin-level k-fold CV (threshold chosen out-of-fold)
      removes the in-sample/resubstitution optimism of the pooled estimate.
  M3  Basin-level screening (671 US basins / 50 GB basins as independent units),
      replacing the pseudo-replicated 3355=671x5 / 150=50x3 basin-seed pooling.
  M6  Per-tier Spearman(log_resid_var, signed coverage error) for BOTH Global-CP
      and HSCC residuals at seed 42 (the §4.5 values were HSCC; Figure 2 is Global-CP).
  m2  Fold-level (n=18) paired / Wilcoxon test for the LORO random-tier control,
      replacing the pseudo-replicated n=360 Welch test.

Writes experiments/post_review_wrr_enhancements/results/r2_audit_addons.json and
retains the per-basin residual-variance CSVs (report item m8).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/qingsongshan/Desktop/代码仓库/Reliability-Control-Streamflow-CP")
sys.path.insert(0, str(ROOT / "experiments/exp002"))
from hscc_analysis_v2 import (  # noqa: E402
    ALPHA, EPS, load_attributes, assign_tier,
    log_score, split_cp_quantile, coverage_log,
    extract_basin_series, load_nh_pickle, find_latest_epoch,
)

sys.path.insert(0, str(ROOT / "experiments/post_review_wrr_enhancements/scripts"))
import run_post_review_enhancements as pr  # noqa: E402

OUT = ROOT / "experiments/post_review_wrr_enhancements/results"
TIERS = ["dry", "semi_arid", "humid", "snow"]
FLOOR = 0.85
N_FOLDS = 5


def signed_log_resid(obs, pred):
    # identical to the original mechanism pipeline (guards against negative LSTM sim)
    return np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6))


# ---------- per-basin regeneration (seed-42 US, from retained pickle) ----------
def build_us_seed42_per_basin() -> pd.DataFrame:
    run_dir = ROOT / "experiments/exp002/results/exp002_camels_us_temporal_2504_155058"
    val = load_nh_pickle(find_latest_epoch(run_dir, "validation"), "validation")
    test = load_nh_pickle(find_latest_epoch(run_dir, "test"), "test")
    attrs = load_attributes()
    recs = []
    for b in val.keys():
        if b not in test:
            continue
        try:
            co, cp = extract_basin_series(val, b)
            to, tp = extract_basin_series(test, b)
        except Exception:
            continue
        if len(co) < 100 or len(to) < 100:
            continue
        gid = str(b).zfill(8)
        a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0})
        recs.append(dict(basin_id=gid, tier=assign_tier(a["aridity"], a["frac_snow"]),
                         aridity=a["aridity"], frac_snow=a["frac_snow"],
                         cal_scores=log_score(co, cp),
                         log_resid_var=float(np.var(signed_log_resid(co, cp))),
                         test_obs=to, test_pred=tp))
    q_global = split_cp_quantile(np.concatenate([r["cal_scores"] for r in recs]), ALPHA)
    q_tier = {t: split_cp_quantile(np.concatenate([r["cal_scores"] for r in recs if r["tier"] == t]), ALPHA)
              for t in TIERS}
    rows = [dict(seed=42, basin_id=r["basin_id"], tier=r["tier"], aridity=r["aridity"],
                 frac_snow=r["frac_snow"], log_resid_var=r["log_resid_var"],
                 coverage_global=coverage_log(r["test_obs"], r["test_pred"], q_global),
                 coverage_hscc=coverage_log(r["test_obs"], r["test_pred"], q_tier[r["tier"]]))
            for r in recs]
    return pd.DataFrame(rows)


# ---------- classifier metrics + threshold ----------
def cls(y, p):
    return pr.classifier_metrics(np.asarray(y, bool), np.asarray(p, bool))


def best_threshold(df, col):
    return pr.choose_threshold(df, col)["threshold"]


def kfold_oos(df, col, k=N_FOLDS):
    """Out-of-sample: choose threshold on train folds, predict held-out fold; pool predictions."""
    d = df.dropna(subset=["log_resid_var", col]).sort_values("basin_id").reset_index(drop=True)
    fold = np.arange(len(d)) % k
    yhat = np.zeros(len(d), bool)
    for f in range(k):
        tr, te = d[fold != f], d.index[fold == f]
        thr = best_threshold(tr, col)
        yhat[fold == f] = d.loc[te, "log_resid_var"].values >= thr
    y = d[col].values < FLOOR
    m = cls(y, yhat)
    m["n_folds"] = k
    return m


def basin_insample(df, col):
    d = df.dropna(subset=["log_resid_var", col])
    thr = best_threshold(d, col)
    m = cls(d[col].values < FLOOR, d["log_resid_var"].values >= thr)
    m["threshold"] = float(thr)
    return m


def per_tier_rho(df):
    out = {"global": {}, "hscc": {}}
    for t in TIERS:
        s = df[df["tier"] == t]
        for tgt, key in [("coverage_global", "global"), ("coverage_hscc", "hscc")]:
            rho, p = stats.spearmanr(s["log_resid_var"], s[tgt])
            out[key][t] = {"rho": float(rho), "p": float(p), "n": int(len(s))}
    # whole-sample sanity vs exp007 (-0.8926 global)
    rho_g, _ = stats.spearmanr(df["log_resid_var"], df["coverage_global"])
    rho_h, _ = stats.spearmanr(df["log_resid_var"], df["coverage_hscc"])
    out["overall_global_rho"] = float(rho_g)
    out["overall_hscc_rho"] = float(rho_h)
    return out


def gb_basin_level():
    try:
        gb = pr.load_gb_per_basin()
    except Exception as e:
        print(f"[R2] GB skipped (tier file not retained): {e}")
        return None, None
    if gb.empty:
        return None, None
    gb.to_csv(OUT / "f2_gb_per_basin_residual_variance.csv", index=False)
    basin = gb.groupby("basin_id").agg(
        log_resid_var=("log_resid_var", "mean"),
        coverage_global=("coverage_global", "mean"),
        tier=("tier", "first")).reset_index()
    return gb, basin


def m2_foldlevel_randomtier():
    d = json.loads((ROOT / "experiments/exp008/results/_analysis/random_distribution.json").read_text())
    phys = np.array([f["physical"]["spread_reduction_pp"] for f in d["per_fold"]])
    rand = np.array([f["random"]["spread_reduction_pp_mean"] for f in d["per_fold"]])
    diff = phys - rand
    paired_t, paired_p2 = stats.ttest_rel(phys, rand)
    welch_t, welch_p2 = stats.ttest_ind(phys, rand, equal_var=False)
    try:
        w_stat, w_p2 = stats.wilcoxon(phys, rand)
    except Exception:
        w_stat, w_p2 = float("nan"), float("nan")
    dz = float(np.mean(diff) / np.std(diff, ddof=1))
    informative = int(np.sum((phys != 0) | (rand != 0)))
    return dict(
        n_folds=int(len(phys)), n_informative_folds=informative,
        physical_mean_pp=float(np.mean(phys)), random_mean_pp=float(np.mean(rand)),
        paired_t=float(paired_t), paired_p_one_sided=float(paired_p2 / 2),
        welch_foldmean_t=float(welch_t), welch_foldmean_p_one_sided=float(welch_p2 / 2),
        wilcoxon_stat=float(w_stat), wilcoxon_p_one_sided=float(w_p2 / 2),
        cohens_dz=dz)


def m1_boundary_facts():
    csv = OUT.parent.parent / "exp002/ablation/02_boundary_sensitivity_results/boundary_sweep.csv"
    csv = ROOT / "experiments/exp002/ablation/02_boundary_sensitivity_results/boundary_sweep.csv"
    df = pd.read_csv(csv)
    df["pass15"] = df["spread_reduction_pp"] >= 15.0
    fail = df[~df["pass15"]]
    snow_counts = {float(s): int(df[df["snow_th"] == s]["n_snow"].iloc[0]) for s in sorted(df["snow_th"].unique())}
    worst = df.sort_values("spread_reduction_pp").iloc[0]
    return dict(
        n_combos=int(len(df)), n_pass_15=int(df["pass15"].sum()),
        preregistered_pass_target=13, preregistered_met=bool(df["pass15"].sum() >= 13),
        failing_snow_cutoffs=sorted(set(float(x) for x in fail["snow_th"])),
        loose_snow_cutoff=float(min(df["snow_th"])), restrictive_snow_cutoff=float(max(df["snow_th"])),
        n_snow_by_cutoff=snow_counts,
        worst_combo_snow_th=float(worst["snow_th"]), worst_combo_dry_th=float(worst["dry_th"]),
        worst_combo_semi_th=float(worst["semi_th"]), worst_combo_reduction_pp=float(worst["spread_reduction_pp"]),
        worst_combo_uses_baseline_aridity=bool((worst["dry_th"] == 1.5) and (worst["semi_th"] == 1.0)),
        min_reduction_pp=float(df["spread_reduction_pp"].min()), max_reduction_pp=float(df["spread_reduction_pp"].max()))


def main():
    print("[R2] building seed-42 US per-basin ...")
    us = build_us_seed42_per_basin()
    us.to_csv(OUT / "f2_us_per_basin_residual_variance.csv", index=False)
    rho = per_tier_rho(us)
    print(f"[R2] sanity: US seed-42 overall global rho={rho['overall_global_rho']:.4f} (exp007=-0.8926)")

    us_in = basin_insample(us, "coverage_global")
    us_oos = kfold_oos(us, "coverage_global")
    print(f"[R2] US basin in-sample: P={us_in['precision']:.3f} R={us_in['recall']:.3f} balacc={us_in['balanced_accuracy']:.3f} thr={us_in['threshold']:.3f} n={us_in['n']}")
    print(f"[R2] US basin OOS(5-fold): P={us_oos['precision']:.3f} R={us_oos['recall']:.3f} balacc={us_oos['balanced_accuracy']:.3f} n={us_oos['n']}")

    gb_raw, gb_basin = gb_basin_level()
    gb_block = None
    if gb_basin is not None:
        gb_in = basin_insample(gb_basin, "coverage_global")
        gb_oos = kfold_oos(gb_basin, "coverage_global")
        # GB leave-one-seed CV (3 seeds) from the existing code path, retains CSV
        cv, _pool = pr.threshold_cv(gb_raw, "CAMELS-GB", "coverage_global")
        los = {f"los_{m}": float(cv[f"test_{m}"].mean()) for m in ["precision", "recall", "specificity", "balanced_accuracy"]}
        gb_block = dict(n_basins=int(len(gb_basin)), in_sample=gb_in, oos_5fold=gb_oos,
                        leave_one_seed_n_seeds=int(gb_raw["seed"].nunique()), **los)
        print(f"[R2] GB basin in-sample: P={gb_in['precision']:.3f} R={gb_in['recall']:.3f} balacc={gb_in['balanced_accuracy']:.3f} n={gb_in['n']}")
        print(f"[R2] GB basin OOS(5-fold): P={gb_oos['precision']:.3f} R={gb_oos['recall']:.3f} balacc={gb_oos['balanced_accuracy']:.3f}")

    addons = dict(
        _generated="2026-06-04 R2 reviewer-audit recompute (retained-data, no retraining)",
        m1_boundary=m1_boundary_facts(),
        m2m3_us=dict(seed=42, n_basins=int(us_in["n"]), in_sample=us_in, oos_5fold=us_oos,
                     note="seed-42 basin-level; replaces 3355=671x5 pseudo-replicated pooled estimate"),
        m2m3_gb=gb_block,
        m6_per_tier_rho_seed42=rho,
        m2_foldlevel_randomtier=m2_foldlevel_randomtier(),
    )
    (OUT / "r2_audit_addons.json").write_text(json.dumps(addons, indent=2))
    print(f"[R2] wrote {OUT/'r2_audit_addons.json'}")
    print("[R2] m6 per-tier global rho:", {t: round(rho['global'][t]['rho'], 3) for t in TIERS})
    print("[R2] m6 per-tier hscc   rho:", {t: round(rho['hscc'][t]['rho'], 3) for t in TIERS})
    print("[R2] m2 fold-level:", {k: round(v, 4) if isinstance(v, float) else v for k, v in addons["m2_foldlevel_randomtier"].items()})


if __name__ == "__main__":
    main()
