"""
Post-review WRR enhancement analyses.

This package runs three bounded, non-retraining analyses requested after the
2026-05-28 reviewer-risk audit:

F2  Residual-variance threshold calibration, reported as a dataset-calibrated
    exploratory evaluation protocol rather than a universal deployment rule.
G6  LORO weighted/localized conformal rescue attempt using static-attribute
    density-ratio weights only; no held-out target discharge or residuals enter
    the weights.
B2  5-seed basin-block bootstrap from retained per-basin mechanism summaries.

Outputs are written under experiments/post_review_wrr_enhancements/results.
"""
from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/post_review_wrr_enhancements/results"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
NOMINAL = 1.0 - ALPHA
EPS = 0.01
UNDERCOVERAGE_FLOOR = 0.85

US_TIERS = ["dry", "semi_arid", "humid", "snow"]
GB_TIERS = ["gb_drier_q4", "gb_mid", "gb_wet", "gb_montane"]

US_ATTR_FILE = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
GB_TIERS_FILE = ROOT / "experiments/exp004/basin_lists/gb_basin_tiers.csv"

US_SEED_SOURCES = [
    (42, ROOT / "experiments/exp007/results/run_2604/mechanism_metrics.csv"),
    (7, ROOT / "experiments/exp009/results/exp009_seed007_0205_215826/_mechanism/mechanism_metrics.csv"),
    (137, ROOT / "experiments/exp009/results/exp009_seed137_0205_180701/_mechanism/mechanism_metrics.csv"),
    (1337, ROOT / "experiments/exp009/results/exp009_seed1337_0205_204444/_mechanism/mechanism_metrics.csv"),
    (2024, ROOT / "experiments/exp009/results/exp009_seed2024_0205_193110/_mechanism/mechanism_metrics.csv"),
]

GB_RUNS = [
    (42, ROOT / "experiments/exp004/results/exp004_camels_gb_native_2604_214507"),
    (137, ROOT / "experiments/exp004/results/exp004_camels_gb_native_seed137_2704_010950"),
    (2024, ROOT / "experiments/exp004/results/exp004_camels_gb_native_seed2024_2704_012056"),
]


def load_us_attrs() -> dict[str, dict[str, float]]:
    attrs: dict[str, dict[str, float]] = {}
    with open(US_ATTR_FILE) as f:
        hdr = f.readline().strip().split(";")
        ai_i = hdr.index("aridity")
        snow_i = hdr.index("frac_snow")
        p_i = hdr.index("p_mean") if "p_mean" in hdr else None
        pet_i = hdr.index("pet_mean") if "pet_mean" in hdr else None
        for line in f:
            cols = line.strip().split(";")
            if not cols or not cols[0]:
                continue
            attrs[cols[0].zfill(8)] = {
                "aridity": float(cols[ai_i]),
                "frac_snow": float(cols[snow_i]),
                "p_mean": float(cols[p_i]) if p_i is not None else np.nan,
                "pet_mean": float(cols[pet_i]) if pet_i is not None else np.nan,
            }
    return attrs


def load_gb_tiers() -> dict[str, dict[str, float | str]]:
    df = pd.read_csv(GB_TIERS_FILE, dtype={"gauge_id": str})
    return {
        str(r.gauge_id): {
            "tier": str(r.tier),
            "aridity": float(r.aridity),
            "frac_snow": float(r.frac_snow),
        }
        for r in df.itertuples()
    }


def log_score(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6)))


def signed_log_residual(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(obs + EPS, 1e-6)) - np.log(np.maximum(pred + EPS, 1e-6))


def split_cp_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    if n == 0:
        return float("nan")
    level = min(math.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, alpha: float = ALPHA) -> float:
    valid = ~(np.isnan(scores) | np.isnan(weights))
    scores = scores[valid]
    weights = weights[valid]
    if len(scores) == 0 or float(weights.sum()) <= 0:
        return float("nan")
    order = np.argsort(scores)
    s = scores[order]
    w = weights[order]
    cum = np.cumsum(w)
    cutoff = (1 - alpha) * float(w.sum())
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    idx = min(idx, len(s) - 1)
    return float(s[idx])


def coverage_log(obs: np.ndarray, pred: np.ndarray, q: float) -> float:
    return float(np.mean(log_score(obs, pred) <= q))


def width_orig(pred: np.ndarray, q: float) -> float:
    log_p = np.log(np.maximum(pred + EPS, 1e-6))
    lo = np.maximum(np.exp(log_p - q) - EPS, 0)
    hi = np.exp(log_p + q) - EPS
    return float(np.mean(hi - lo))


def find_latest_epoch_pickle(run_dir: Path, kind: str) -> Path | None:
    sub = run_dir / kind
    if not sub.exists():
        return None
    epochs = sorted(sub.glob("model_epoch*"))
    for ep in reversed(epochs):
        p = ep / f"{kind}_results.p"
        if p.exists():
            return p
    return None


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_series(results, basin_id, obs_col: str, sim_col: str):
    bd = results[basin_id]
    if isinstance(bd, dict) and "1D" in bd:
        ds = bd["1D"]["xr"]
    elif hasattr(bd, "xr"):
        ds = bd.xr
    else:
        raise ValueError(f"unknown result format for basin {basin_id}: {type(bd)}")
    obs = ds[obs_col].values.flatten()
    pred = ds[sim_col].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def load_us_per_basin() -> pd.DataFrame:
    frames = []
    for seed, path in US_SEED_SOURCES:
        df = pd.read_csv(path, dtype={"basin_id": str})
        df["basin_id"] = df["basin_id"].str.zfill(8)
        df["seed"] = seed
        df["dataset"] = "CAMELS-US"
        frames.append(df)
    cols = [
        "dataset", "seed", "basin_id", "tier", "aridity", "frac_snow",
        "log_resid_var", "coverage_global", "coverage_hscc",
    ]
    out = pd.concat(frames, ignore_index=True)
    return out[cols].copy()


def load_gb_per_basin() -> pd.DataFrame:
    tiers = load_gb_tiers()
    frames = []
    for seed, run_dir in GB_RUNS:
        val_p = find_latest_epoch_pickle(run_dir, "validation")
        test_p = find_latest_epoch_pickle(run_dir, "test")
        if val_p is None or test_p is None:
            continue
        val = load_pickle(val_p)
        test = load_pickle(test_p)
        records = []
        for basin_id in val.keys():
            sid = str(basin_id)
            if basin_id not in test or sid not in tiers:
                continue
            try:
                cal_obs, cal_pred = extract_series(val, basin_id, "discharge_spec_obs", "discharge_spec_sim")
                tst_obs, tst_pred = extract_series(test, basin_id, "discharge_spec_obs", "discharge_spec_sim")
            except Exception:
                continue
            if len(cal_obs) < 100 or len(tst_obs) < 100:
                continue
            meta = tiers[sid]
            records.append({
                "seed": seed,
                "basin_id": sid,
                "tier": meta["tier"],
                "aridity": meta["aridity"],
                "frac_snow": meta["frac_snow"],
                "cal_scores": log_score(cal_obs, cal_pred),
                "log_resid_var": float(np.var(signed_log_residual(cal_obs, cal_pred))),
                "test_obs": tst_obs,
                "test_pred": tst_pred,
            })
        if not records:
            continue
        q_global = split_cp_quantile(np.concatenate([r["cal_scores"] for r in records]))
        q_tier = {}
        for tier in GB_TIERS:
            ss = [r["cal_scores"] for r in records if r["tier"] == tier]
            q_tier[tier] = split_cp_quantile(np.concatenate(ss)) if ss else float("nan")
        rows = []
        for r in records:
            rows.append({
                "dataset": "CAMELS-GB",
                "seed": seed,
                "basin_id": r["basin_id"],
                "tier": r["tier"],
                "aridity": r["aridity"],
                "frac_snow": r["frac_snow"],
                "log_resid_var": r["log_resid_var"],
                "coverage_global": coverage_log(r["test_obs"], r["test_pred"], q_global),
                "coverage_hscc": coverage_log(r["test_obs"], r["test_pred"], q_tier[r["tier"]]),
            })
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bal_acc = 0.5 * (recall + specificity)
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
    }


def choose_threshold(train: pd.DataFrame, coverage_col: str) -> dict[str, float]:
    y = train[coverage_col].values < UNDERCOVERAGE_FLOOR
    vals = np.sort(train["log_resid_var"].dropna().unique())
    if len(vals) > 200:
        vals = np.quantile(vals, np.linspace(0.02, 0.98, 193))
    best = None
    for thr in vals:
        pred = train["log_resid_var"].values >= thr
        m = classifier_metrics(y, pred)
        # F1 is primary; balanced accuracy breaks ties for broader screening value.
        score = (m["f1"], m["balanced_accuracy"], m["recall"], -abs(float(thr)))
        if best is None or score > best["score"]:
            best = {"threshold": float(thr), "score": score, **m}
    assert best is not None
    best.pop("score", None)
    return best


def threshold_cv(df: pd.DataFrame, dataset: str, coverage_col: str) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for held_seed in sorted(df["seed"].unique()):
        train = df[df["seed"] != held_seed]
        test = df[df["seed"] == held_seed]
        chosen = choose_threshold(train, coverage_col)
        pred = test["log_resid_var"].values >= chosen["threshold"]
        y = test[coverage_col].values < UNDERCOVERAGE_FLOOR
        metrics = classifier_metrics(y, pred)
        rows.append({
            "dataset": dataset,
            "coverage_target": coverage_col,
            "heldout_seed": int(held_seed),
            "threshold_from_training": chosen["threshold"],
            **{f"train_{k}": v for k, v in chosen.items() if k != "threshold"},
            **{f"test_{k}": v for k, v in metrics.items()},
        })
    cv = pd.DataFrame(rows)
    pooled = choose_threshold(df, coverage_col)
    pred = df["log_resid_var"].values >= pooled["threshold"]
    y = df[coverage_col].values < UNDERCOVERAGE_FLOOR
    pooled_metrics = {"dataset": dataset, "coverage_target": coverage_col, **pooled}
    pooled_metrics.update({f"pooled_{k}": v for k, v in classifier_metrics(y, pred).items()})
    return cv, pooled_metrics


def run_f2() -> dict:
    us = load_us_per_basin()
    gb = load_gb_per_basin()
    us.to_csv(OUT / "f2_us_per_basin_residual_variance.csv", index=False)
    if len(gb):
        gb.to_csv(OUT / "f2_gb_per_basin_residual_variance.csv", index=False)

    cv_frames = []
    pooled = []
    for df, name in [(us, "CAMELS-US"), (gb, "CAMELS-GB")]:
        if df.empty:
            continue
        for target in ["coverage_global", "coverage_hscc"]:
            cv, pool = threshold_cv(df, name, target)
            cv_frames.append(cv)
            pooled.append(pool)
    cv_all = pd.concat(cv_frames, ignore_index=True)
    pooled_df = pd.DataFrame(pooled)
    cv_all.to_csv(OUT / "f2_threshold_leave_one_seed_cv.csv", index=False)
    pooled_df.to_csv(OUT / "f2_threshold_pooled_summary.csv", index=False)

    primary = pooled_df[
        (pooled_df["dataset"] == "CAMELS-US") & (pooled_df["coverage_target"] == "coverage_global")
    ].iloc[0].to_dict()
    gb_primary = None
    if not gb.empty:
        gb_primary = pooled_df[
            (pooled_df["dataset"] == "CAMELS-GB") & (pooled_df["coverage_target"] == "coverage_global")
        ].iloc[0].to_dict()

    summary = {
        "analysis": "F2 residual-variance threshold calibration",
        "risk_label": f"per-basin coverage < {UNDERCOVERAGE_FLOOR:.2f}",
        "primary_recommended_protocol": [
            "Compute per-basin calibration-period log-flow residual variance using only observed calibration data.",
            "Calibrate the numeric threshold within the target dataset or a representative historical basin pool.",
            "Flag basins above the calibrated threshold as residual-variance risk basins.",
            "Treat the flag as an exploratory evaluation protocol: it triggers a conditional-coverage audit, not automatic deployment.",
        ],
        "primary_us_global_threshold": primary,
        "gb_external_threshold": gb_primary,
        "limitations": [
            "Numeric thresholds are dataset-calibrated and should not be transplanted as universal constants.",
            "The target is severe undercoverage screening, not causal mechanism proof.",
            "GB validation is based on the bounded 50-basin exp004/3-seed package.",
        ],
    }
    (OUT / "f2_threshold_protocol_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_loro_fold_records(run_dir: Path, huc: str, attrs: dict[str, dict[str, float]]):
    val_p = find_latest_epoch_pickle(run_dir, "validation")
    test_p = find_latest_epoch_pickle(run_dir, "test")
    if val_p is None or test_p is None:
        return None
    val = load_pickle(val_p)
    test = load_pickle(test_p)

    def build(results, kind: str):
        rows = []
        for basin_id in results.keys():
            gid = str(basin_id).zfill(8)
            try:
                obs, pred = extract_series(results, basin_id, "QObs(mm/d)_obs", "QObs(mm/d)_sim")
            except Exception:
                continue
            if len(obs) < 100:
                continue
            a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0, "p_mean": np.nan, "pet_mean": np.nan})
            tier = "snow" if a["frac_snow"] >= 0.40 else "dry" if a["aridity"] > 1.5 else "semi_arid" if a["aridity"] > 1.0 else "humid"
            rows.append({
                "basin_id": gid,
                "kind": kind,
                "huc": huc,
                "tier": tier,
                "aridity": a["aridity"],
                "frac_snow": a["frac_snow"],
                "p_mean": a.get("p_mean", np.nan),
                "pet_mean": a.get("pet_mean", np.nan),
                "scores": log_score(obs, pred),
                "obs": obs,
                "pred": pred,
            })
        return rows

    cal = build(val, "cal")
    test_rows = build(test, "test")
    return cal, test_rows


def density_ratio_weights(source: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    features = ["aridity", "frac_snow"]
    src = source[features].astype(float).values
    tgt = target[features].astype(float).values
    if len(source) < 10 or len(target) < 3:
        return np.ones(len(source))
    # Use a robust diagonal Gaussian density-ratio estimator. It is intentionally
    # simpler than a high-capacity classifier to avoid overfitting small HUC pools.
    mu_s = np.nanmean(src, axis=0)
    mu_t = np.nanmean(tgt, axis=0)
    var_s = np.nanvar(src, axis=0) + 1e-4
    var_t = np.nanvar(tgt, axis=0) + 1e-4
    log_src = -0.5 * np.sum(np.log(2 * np.pi * var_s) + ((src - mu_s) ** 2) / var_s, axis=1)
    log_tgt = -0.5 * np.sum(np.log(2 * np.pi * var_t) + ((src - mu_t) ** 2) / var_t, axis=1)
    raw = np.exp(np.clip(log_tgt - log_src, -6.0, 6.0))
    raw = np.clip(raw, 0.05, 20.0)
    return raw / np.mean(raw)


def eval_records(records: list[dict], q: float) -> tuple[float, float]:
    cov = [coverage_log(r["obs"], r["pred"], q) for r in records]
    wid = [width_orig(r["pred"], q) for r in records]
    return float(np.mean(cov)), float(np.mean(wid))


def run_g6() -> dict:
    attrs = load_us_attrs()
    fold_dirs = []
    for p in sorted((ROOT / "experiments/exp003/results").glob("exp003_loro_huc??_*")):
        if re.search(r"_seed\d+", p.name):
            continue
        m = re.search(r"huc(\d\d)", p.name)
        if m:
            fold_dirs.append((m.group(1), p))

    cell_rows = []
    diagnostics = []
    for huc, run_dir in fold_dirs:
        loaded = load_loro_fold_records(run_dir, huc, attrs)
        if loaded is None:
            continue
        cal_records, test_records = loaded
        if not cal_records or not test_records:
            continue
        cal_df = pd.DataFrame([{k: v for k, v in r.items() if k not in {"scores", "obs", "pred"}} for r in cal_records])
        test_df = pd.DataFrame([{k: v for k, v in r.items() if k not in {"scores", "obs", "pred"}} for r in test_records])
        weights = density_ratio_weights(cal_df, test_df)
        for r, w in zip(cal_records, weights):
            r["weight"] = float(w)

        all_scores = np.concatenate([r["scores"] for r in cal_records])
        q_global = split_cp_quantile(all_scores)
        w_all_scores = np.concatenate([r["scores"] for r in cal_records])
        w_all_weights = np.concatenate([np.full(len(r["scores"]), r["weight"]) for r in cal_records])
        q_weighted_global = weighted_quantile(w_all_scores, w_all_weights)

        diagnostics.append({
            "huc": huc,
            "n_source_basins": len(cal_records),
            "n_target_basins": len(test_records),
            "weight_min": float(np.min(weights)),
            "weight_median": float(np.median(weights)),
            "weight_max": float(np.max(weights)),
            "weight_cv": float(np.std(weights) / np.mean(weights)),
            "q_global": q_global,
            "q_weighted_global": q_weighted_global,
        })

        for tier in US_TIERS:
            cal_t = [r for r in cal_records if r["tier"] == tier]
            test_t = [r for r in test_records if r["tier"] == tier]
            if not test_t or not cal_t:
                continue
            scores_t = np.concatenate([r["scores"] for r in cal_t])
            weights_t = np.concatenate([np.full(len(r["scores"]), r["weight"]) for r in cal_t])
            q_hscc = split_cp_quantile(scores_t)
            q_weighted_hscc = weighted_quantile(scores_t, weights_t)
            for method, q in [
                ("Global_CP", q_global),
                ("HSCC", q_hscc),
                ("Weighted_Global_CP", q_weighted_global),
                ("Weighted_HSCC", q_weighted_hscc),
            ]:
                cov, width = eval_records(test_t, q)
                cell_rows.append({
                    "huc": huc,
                    "tier": tier,
                    "method": method,
                    "n_test_basins": len(test_t),
                    "coverage": cov,
                    "width_mm_d": width,
                    "q": q,
                    "weighted": method.startswith("Weighted"),
                })

    cells = pd.DataFrame(cell_rows)
    diag = pd.DataFrame(diagnostics)
    cells.to_csv(OUT / "g6_loro_weighted_cells.csv", index=False)
    diag.to_csv(OUT / "g6_loro_weight_diagnostics.csv", index=False)

    agg_rows = []
    for method, mdf in cells.groupby("method"):
        for tier, tdf in mdf.groupby("tier"):
            w = tdf["n_test_basins"].values.astype(float)
            agg_rows.append({
                "method": method,
                "tier": tier,
                "n_cells": int(len(tdf)),
                "n_test_basins_weight": int(w.sum()),
                "coverage": float(np.average(tdf["coverage"], weights=w)),
                "width_mm_d": float(np.average(tdf["width_mm_d"], weights=w)),
            })
    agg = pd.DataFrame(agg_rows)
    spread_rows = []
    for method, mdf in agg.groupby("method"):
        spread_rows.append({
            "method": method,
            "mean_coverage": float(mdf["coverage"].mean()),
            "spread_pp": float((mdf["coverage"].max() - mdf["coverage"].min()) * 100),
            "mean_width_mm_d": float(mdf["width_mm_d"].mean()),
            "min_tier": str(mdf.loc[mdf["coverage"].idxmin(), "tier"]),
            "max_tier": str(mdf.loc[mdf["coverage"].idxmax(), "tier"]),
        })
    spread = pd.DataFrame(spread_rows).sort_values("spread_pp")
    agg.to_csv(OUT / "g6_loro_weighted_tier_aggregate.csv", index=False)
    spread.to_csv(OUT / "g6_loro_weighted_spread_summary.csv", index=False)

    base = spread[spread["method"] == "HSCC"].iloc[0].to_dict()
    weighted = spread[spread["method"] == "Weighted_HSCC"].iloc[0].to_dict()
    width_ratio = (
        float(weighted["mean_width_mm_d"]) / float(base["mean_width_mm_d"])
        if float(base["mean_width_mm_d"]) > 0 else float("inf")
    )
    if width_ratio > 5:
        interpretation = "limited_rescue_with_unacceptable_width_cost"
    elif weighted["spread_pp"] >= base["spread_pp"] - 1.0:
        interpretation = "negative_or_limited_rescue"
    else:
        interpretation = "material_rescue_candidate"
    summary = {
        "analysis": "G6 static-attribute weighted conformal on exp003 LORO",
        "weighting": "diagonal-Gaussian density-ratio estimate using source/target aridity and frac_snow only",
        "no_target_flow_leakage": True,
        "baseline_hscc": base,
        "weighted_hscc": weighted,
        "delta_weighted_minus_baseline_spread_pp": float(weighted["spread_pp"] - base["spread_pp"]),
        "weighted_to_baseline_width_ratio": width_ratio,
        "interpretation": interpretation,
        "claim_boundary": "Empirical rescue attempt only; not a formal Tibshirani weighted-CP coverage guarantee.",
    }
    (OUT / "g6_loro_weighted_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_b2_light(us: pd.DataFrame | None = None) -> dict:
    if us is None:
        us = load_us_per_basin()
    rng = np.random.default_rng(20260528)
    per_tier_rows = []
    spread_rows = []
    boot_spread = []
    n_boot = 5000

    matrices: dict[str, dict[str, np.ndarray]] = {}
    for tier in US_TIERS:
        sub = us[us["tier"] == tier].copy()
        global_mat = sub.pivot(index="basin_id", columns="seed", values="coverage_global")
        hscc_mat = sub.pivot(index="basin_id", columns="seed", values="coverage_hscc")
        # Keep only complete basin x seed blocks so the bootstrap unit is stable.
        keep = global_mat.notna().all(axis=1) & hscc_mat.notna().all(axis=1)
        global_mat = global_mat.loc[keep]
        hscc_mat = hscc_mat.loc[keep]
        matrices[tier] = {
            "global": global_mat.to_numpy(dtype=float),
            "hscc": hscc_mat.to_numpy(dtype=float),
        }

    for _ in range(n_boot):
        sampled_seed = {}
        for tier in US_TIERS:
            gmat = matrices[tier]["global"]
            hmat = matrices[tier]["hscc"]
            n_basin, n_seed = gmat.shape
            b_idx = rng.integers(0, n_basin, n_basin)
            s_idx = rng.integers(0, n_seed, n_seed)
            sampled_seed[tier] = {
                "global": float(gmat[np.ix_(b_idx, s_idx)].mean()),
                "hscc": float(hmat[np.ix_(b_idx, s_idx)].mean()),
            }
        boot_spread.append({
            "global_spread_pp": (max(v["global"] for v in sampled_seed.values()) - min(v["global"] for v in sampled_seed.values())) * 100,
            "hscc_spread_pp": (max(v["hscc"] for v in sampled_seed.values()) - min(v["hscc"] for v in sampled_seed.values())) * 100,
            **{f"{tier}_global": sampled_seed[tier]["global"] for tier in US_TIERS},
            **{f"{tier}_hscc": sampled_seed[tier]["hscc"] for tier in US_TIERS},
        })

    boot = pd.DataFrame(boot_spread)
    boot.to_csv(OUT / "b2_light_bootstrap_samples.csv", index=False)

    for tier in US_TIERS:
        for method, col in [("Global_CP", f"{tier}_global"), ("HSCC", f"{tier}_hscc")]:
            vals = boot[col]
            per_tier_rows.append({
                "tier": tier,
                "method": method,
                "mean": float(vals.mean()),
                "ci95_lo": float(vals.quantile(0.025)),
                "ci95_hi": float(vals.quantile(0.975)),
                "p10": float(vals.quantile(0.10)),
                "p90": float(vals.quantile(0.90)),
            })
    for method, col in [("Global_CP", "global_spread_pp"), ("HSCC", "hscc_spread_pp")]:
        vals = boot[col]
        spread_rows.append({
            "method": method,
            "spread_pp_mean": float(vals.mean()),
            "spread_pp_ci95_lo": float(vals.quantile(0.025)),
            "spread_pp_ci95_hi": float(vals.quantile(0.975)),
            "spread_pp_p10": float(vals.quantile(0.10)),
            "spread_pp_p90": float(vals.quantile(0.90)),
        })

    per_tier = pd.DataFrame(per_tier_rows)
    spread = pd.DataFrame(spread_rows)
    per_tier.to_csv(OUT / "b2_light_tier_coverage_ci.csv", index=False)
    spread.to_csv(OUT / "b2_light_spread_ci.csv", index=False)
    summary = {
        "analysis": "B2-light 5-seed basin-block bootstrap",
        "n_bootstrap": n_boot,
        "unit": "basin block with seed resampling from retained per-basin mechanism summaries",
        "not_in_scope": "daily/year-block bootstrap because four non-seed42 test_results.p files are not retained",
        "spread_ci": spread.to_dict(orient="records"),
    }
    (OUT / "b2_light_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_report(f2: dict, g6: dict, b2: dict):
    f2_pool = pd.read_csv(OUT / "f2_threshold_pooled_summary.csv")
    g6_spread = pd.read_csv(OUT / "g6_loro_weighted_spread_summary.csv")
    b2_spread = pd.read_csv(OUT / "b2_light_spread_ci.csv")

    def row_value(df, **where):
        sub = df.copy()
        for k, v in where.items():
            sub = sub[sub[k] == v]
        return sub.iloc[0].to_dict()

    us_global = row_value(f2_pool, dataset="CAMELS-US", coverage_target="coverage_global")
    us_hscc = row_value(f2_pool, dataset="CAMELS-US", coverage_target="coverage_hscc")
    hscc = row_value(g6_spread, method="HSCC")
    whscc = row_value(g6_spread, method="Weighted_HSCC")
    b2_g = row_value(b2_spread, method="Global_CP")
    b2_h = row_value(b2_spread, method="HSCC")

    lines = [
        "# Post-Review WRR Enhancement Analyses",
        "",
        "**Scope**: F2 residual-variance threshold protocol, G6 static-attribute weighted conformal LORO rescue attempt, and B2-light 5-seed basin-block bootstrap. No model retraining was performed.",
        "",
        "## F2 Residual-Variance Threshold Protocol",
        "",
        f"- Primary US threshold for flagging Global CP per-basin coverage < 0.85: `log_resid_var >= {us_global['threshold']:.6f}`.",
        f"- Pooled US screening performance: precision {us_global['pooled_precision']:.3f}, recall {us_global['pooled_recall']:.3f}, F1 {us_global['pooled_f1']:.3f}, balanced accuracy {us_global['pooled_balanced_accuracy']:.3f}.",
        f"- HSCC residual-risk threshold: `log_resid_var >= {us_hscc['threshold']:.6f}`; precision {us_hscc['pooled_precision']:.3f}, recall {us_hscc['pooled_recall']:.3f}, F1 {us_hscc['pooled_f1']:.3f}.",
        "- Recommended wording: dataset-calibrated exploratory evaluation protocol, not universal deployment rule and not causal mechanism proof.",
        "",
        "## G6 Weighted/Localized Conformal LORO Attempt",
        "",
        "- Weights use only static target/source attributes (`aridity`, `frac_snow`) through a diagonal-Gaussian density-ratio estimate.",
        f"- Baseline HSCC LORO spread: {hscc['spread_pp']:.2f} pp; Weighted HSCC spread: {whscc['spread_pp']:.2f} pp.",
        f"- Baseline HSCC mean coverage: {hscc['mean_coverage']:.3f}; Weighted HSCC mean coverage: {whscc['mean_coverage']:.3f}.",
        f"- Baseline HSCC mean width: {hscc['mean_width_mm_d']:.2f} mm/d; Weighted HSCC mean width: {whscc['mean_width_mm_d']:.2f} mm/d.",
        f"- Interpretation: `{g6['interpretation']}`. This is an empirical rescue attempt, not a formal Tibshirani weighted-CP guarantee.",
        "",
        "## B2-Light 5-Seed Basin-Block Bootstrap",
        "",
        f"- Global CP spread bootstrap mean {b2_g['spread_pp_mean']:.2f} pp, 95% CI [{b2_g['spread_pp_ci95_lo']:.2f}, {b2_g['spread_pp_ci95_hi']:.2f}].",
        f"- HSCC spread bootstrap mean {b2_h['spread_pp_mean']:.2f} pp, 95% CI [{b2_h['spread_pp_ci95_lo']:.2f}, {b2_h['spread_pp_ci95_hi']:.2f}].",
        "- This closes the basin-block reviewer concern with retained per-basin summaries. It does not replace a daily/year-block bootstrap.",
        "",
        "## Output Files",
        "",
        "- `f2_threshold_protocol_summary.json`",
        "- `f2_threshold_leave_one_seed_cv.csv`",
        "- `f2_threshold_pooled_summary.csv`",
        "- `g6_loro_weighted_cells.csv`",
        "- `g6_loro_weighted_spread_summary.csv`",
        "- `g6_loro_weighted_summary.json`",
        "- `b2_light_tier_coverage_ci.csv`",
        "- `b2_light_spread_ci.csv`",
        "- `b2_light_summary.json`",
    ]
    report = "\n".join(lines) + "\n"
    (ROOT / "experiments/post_review_wrr_enhancements/report.md").write_text(report)
    (OUT / "summary.json").write_text(json.dumps({"f2": f2, "g6": g6, "b2": b2}, indent=2))


def main():
    print("=== F2 residual-variance threshold protocol ===")
    f2 = run_f2()
    print(json.dumps(f2["primary_us_global_threshold"], indent=2))

    print("\n=== G6 weighted conformal LORO attempt ===")
    g6 = run_g6()
    print(json.dumps(g6, indent=2))

    print("\n=== B2-light 5-seed basin-block bootstrap ===")
    us = load_us_per_basin()
    b2 = run_b2_light(us)
    print(json.dumps(b2["spread_ci"], indent=2))

    write_report(f2, g6, b2)
    print(f"\nWrote report: {ROOT / 'experiments/post_review_wrr_enhancements/report.md'}")


if __name__ == "__main__":
    main()
