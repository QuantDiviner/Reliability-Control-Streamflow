"""
exp013_aci — Adaptive Conformal Inference (Gibbs & Candes 2021) on CAMELS-US.

Online, deterministic alpha-update conformal prediction on top of the exp002
LSTM (seed=42) predictions. Score function is the project-standard log-flow
residual s_t = |log(Q + eps) - log(Q_hat + eps)|, eps = 0.01.

Algorithm (Gibbs & Candes 2021):
  alpha_{t+1} = alpha_t + gamma * (alpha_star - err_t)
  err_t = 1 if y_t not in C_t else 0
  C_t = [Q_hat_t * exp(-q_t) - eps_adj, Q_hat_t * exp(+q_t) - eps_adj]
  q_t = empirical (1 - alpha_t) quantile of calibration scores

Initial calibration scores come from the validation period (1990-10 to
2000-09). Test period (2000-10 to 2014-09) drives the online update.

Only one LSTM seed (42) is available locally; ACI itself is deterministic
so the reported numbers are a single-LSTM-seed point estimate. The 5-seed
HSCC/Global CP entries in metrics.json are not reproducible here because
the other four LSTM .p files have been pruned.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ALPHA_STAR = 0.10
EPS = 0.01
GAMMA = 0.005                     # Gibbs 2021 recommended default
ALPHA_FLOOR, ALPHA_CEIL = 1e-4, 0.5
MIN_CAL = 100
MIN_TEST = 100
CLIP_K = 2.0                      # physical ceiling multiplier: W_MAX_b = K * max(cal obs)

TIERS = ["dry", "semi_arid", "humid", "snow"]
EXP002_RUN = ROOT / "experiments/exp002/results/exp002_camels_us_temporal_2504_155058"
ATTR_FILE = ROOT / "data/raw/CAMELS_US/camels_attributes_v2.0/camels_clim.txt"
OUT_DIR = ROOT / "experiments/exp013_aci/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def load_attrs():
    attrs = {}
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


def assign_tier(ai, fs):
    if fs >= 0.40:
        return "snow"
    if ai > 1.5:
        return "dry"
    if ai > 1.0:
        return "semi_arid"
    return "humid"


def find_latest(rd, k):
    return sorted((rd / k).glob("model_epoch*"))[-1]


def extract(results, b):
    ds = results[b]["1D"]["xr"]
    obs = ds["QObs(mm/d)_obs"].values.flatten()
    pred = ds["QObs(mm/d)_sim"].values.flatten()
    valid = ~(np.isnan(obs) | np.isnan(pred))
    return obs[valid], pred[valid]


def log_score(obs, pred):
    # s_t = |log(Q + eps) - log(Q_hat + eps)|; clip to keep log defined
    a = np.log(np.maximum(obs + EPS, 1e-9))
    b = np.log(np.maximum(pred + EPS, 1e-9))
    return np.abs(a - b)


def aci_basin(cal_scores, tst_obs, tst_pred, w_max=None):
    """Run Gibbs 2021 ACI on one basin.

    cal_scores: 1D array of log-residual scores from validation period.
    tst_obs / tst_pred: 1D arrays of observed and predicted flow in mm/d.
    w_max: optional physical upper-bound ceiling on the interval's upper
        endpoint ``hi`` (mm/d). When set, ``hi`` is capped at ``w_max`` AFTER
        the multiplicative back-transform. This tames the heavy right tail of
        the log-space score (Q_hat * exp(+q_t) explodes when q_t hits the
        score distribution's tail); ``lo`` is left unchanged. For ordinary
        timesteps ``w_max`` is far above ``hi`` so nothing changes.

    Returns dict with coverage, mean_width, mean log-interval, n_test, etc.
    When clipping is active, also returns clip-binding counts.
    """
    sorted_cal = np.sort(cal_scores)
    n_cal = len(sorted_cal)

    alpha = ALPHA_STAR
    n_test = len(tst_obs)
    covered = np.empty(n_test, dtype=bool)
    width = np.empty(n_test, dtype=float)
    log_width = np.empty(n_test, dtype=float)
    n_clipped = 0

    for t in range(n_test):
        # Empirical (1-alpha) quantile of cal scores at current alpha.
        # Use the standard split-CP correction: level = ceil((1-alpha)(n+1))/n.
        a_clip = float(np.clip(alpha, ALPHA_FLOOR, ALPHA_CEIL))
        level = min(np.ceil((1.0 - a_clip) * (n_cal + 1)) / n_cal, 1.0)
        q_t = float(np.quantile(sorted_cal, level))

        ph = tst_pred[t]
        # Symmetric in log space.
        log_lo = np.log(max(ph + EPS, 1e-9)) - q_t
        log_hi = np.log(max(ph + EPS, 1e-9)) + q_t
        lo = max(np.exp(log_lo) - EPS, 0.0)
        hi = max(np.exp(log_hi) - EPS, lo)

        # Physically-motivated clip on the upper endpoint only.
        if w_max is not None and hi > w_max:
            hi = max(w_max, lo)  # never let the cap fall below lo
            n_clipped += 1

        obs_t = tst_obs[t]
        covered[t] = (obs_t >= lo) and (obs_t <= hi)
        width[t] = hi - lo
        log_width[t] = 2.0 * q_t  # exact in log space

        err = 0.0 if covered[t] else 1.0
        alpha = float(np.clip(alpha + GAMMA * (ALPHA_STAR - err), ALPHA_FLOOR, ALPHA_CEIL))

    out = {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "mean_log_width": float(np.mean(log_width)),
        "final_alpha": float(alpha),
        "n_test": int(n_test),
        "n_cal": int(n_cal),
    }
    if w_max is not None:
        out["w_max"] = float(w_max)
        out["n_clipped"] = int(n_clipped)
        out["clip_binding_fraction"] = float(n_clipped) / float(n_test) if n_test else 0.0
    return out


def main(clip=False, clip_k=CLIP_K):
    t0 = time.time()
    mode = f"CLIPPED (K={clip_k})" if clip else "unbounded"
    log(f"=== exp013_aci: Gibbs 2021 ACI on CAMELS-US (LSTM seed=42) — {mode} ===\n")

    attrs = load_attrs()
    val_p = find_latest(EXP002_RUN, "validation") / "validation_results.p"
    test_p = find_latest(EXP002_RUN, "test") / "test_results.p"
    log(f"loading val={val_p}")
    log(f"loading test={test_p}")
    val = pickle.load(open(val_p, "rb"))
    test = pickle.load(open(test_p, "rb"))

    rows = []
    skipped = []
    for i, b in enumerate(sorted(val.keys())):
        if b not in test:
            skipped.append((b, "no_test"))
            continue
        try:
            cal_obs, cal_pred = extract(val, b)
            tst_obs, tst_pred = extract(test, b)
        except Exception as e:
            skipped.append((b, f"extract:{e}"))
            continue
        if len(cal_obs) < MIN_CAL or len(tst_obs) < MIN_TEST:
            skipped.append((b, "too_few_samples"))
            continue
        gid = str(b).zfill(8)
        a = attrs.get(gid, {"aridity": 1.0, "frac_snow": 0.0})
        tier = assign_tier(a["aridity"], a["frac_snow"])

        cal_scores = log_score(cal_obs, cal_pred)

        w_max = None
        if clip:
            # Physical ceiling = K * largest calibration-period observed flow.
            obs_max = float(np.nanmax(cal_obs)) if cal_obs.size else np.nan
            if not np.isfinite(obs_max):
                # Fall back to max predicted flow if obs unavailable / all-NaN.
                pred_max = float(np.nanmax(cal_pred)) if cal_pred.size else np.nan
                obs_max = pred_max
            w_max = clip_k * obs_max if np.isfinite(obs_max) else None

        res = aci_basin(cal_scores, tst_obs, tst_pred, w_max=w_max)
        res.update({"basin": gid, "tier": tier,
                    "aridity": a["aridity"], "frac_snow": a["frac_snow"]})
        rows.append(res)
        if (i + 1) % 100 == 0 or i == 0:
            log(f"  basin {i+1}/{len(val)}: {gid} tier={tier} "
                f"cov={res['coverage']:.3f} width={res['mean_width']:.3f} "
                f"alpha_final={res['final_alpha']:.4f}")

    df = pd.DataFrame(rows)
    log(f"\nbasins processed: {len(df)}; skipped: {len(skipped)}")

    suffix = "_clipped" if clip else ""
    pb_path = OUT_DIR / f"per_basin_metrics{suffix}.csv"
    df.to_csv(pb_path, index=False)
    log(f"wrote: {pb_path}")

    if clip:
        # Clipped mode only writes the per-basin CSV here; the full robust
        # summary (median/p95 width + clip-binding stats, matching the
        # unbounded aci_summary.json schema) is produced by aggregate.py
        # --clip, which writes aci_summary_clipped.json. The trailing coarse
        # summary below is reserved for the unbounded run so the clipped run
        # never overwrites the original tier_aggregate.csv / aci_summary.json.
        binding = float(df["clip_binding_fraction"].mean()) if "clip_binding_fraction" in df else float("nan")
        log(f"\nmean per-basin clip-binding fraction = {binding:.6f}")
        log(f"elapsed = {round(time.time() - t0, 2)}s")
        return

    # Per-tier aggregate (basin-mean, equal-weight per basin to match HSCC/Global CP).
    tier_rows = []
    for t in TIERS:
        sub = df[df["tier"] == t]
        tier_rows.append({
            "tier": t,
            "n_basins": int(len(sub)),
            "mean_coverage": float(sub["coverage"].mean()) if len(sub) else float("nan"),
            "mean_width": float(sub["mean_width"].mean()) if len(sub) else float("nan"),
            "median_coverage": float(sub["coverage"].median()) if len(sub) else float("nan"),
        })
    tier_df = pd.DataFrame(tier_rows)
    tier_path = OUT_DIR / "tier_aggregate.csv"
    tier_df.to_csv(tier_path, index=False)
    log("\n=== Per-tier summary ===")
    log(tier_df.to_string(index=False))

    covs = tier_df["mean_coverage"].dropna().values
    spread_pp = float((covs.max() - covs.min()) * 100) if len(covs) >= 2 else float("nan")
    marginal = float(df["coverage"].mean())
    summary = {
        "method": "ACI_Gibbs2021",
        "alpha_star": ALPHA_STAR,
        "gamma": GAMMA,
        "score_function": "log_flow_residual",
        "score_epsilon": EPS,
        "lstm_seed_set": [42],
        "n_basins": int(len(df)),
        "n_basins_skipped": int(len(skipped)),
        "marginal_coverage": marginal,
        "tier_coverage_spread_pp": spread_pp,
        "tier_coverage_spread_definition": "max - min of basin-mean coverage across {dry, semi_arid, humid, snow}",
        "per_tier": tier_df.to_dict(orient="records"),
        "notes": [
            "Single LSTM seed=42 (only exp002 .p files retained locally).",
            "ACI is deterministic given fixed LSTM predictions; no CP-level seed variability to report.",
            "Initial alpha_0 = alpha_star = 0.10; gamma = 0.005 per Gibbs & Candes 2021.",
            "Calibration scores fixed from validation (1990-10..2000-09); only alpha updates online.",
            "Comparable upstream LSTM to exp_global_cqr (also seed=42 only) and exp010 (HopCPT).",
        ],
        "skipped_examples": skipped[:10],
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    sm_path = OUT_DIR / "aci_summary.json"
    with open(sm_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nwrote: {sm_path}")
    log(f"\nmarginal coverage = {marginal:.4f}")
    log(f"tier spread (pp)  = {spread_pp:.2f}")
    log(f"elapsed = {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="exp013 ACI (Gibbs 2021) on CAMELS-US.")
    ap.add_argument("--clip", action="store_true",
                    help="cap per-timestep upper endpoint at W_MAX_b = K * max(cal obs); "
                         "writes per_basin_metrics_clipped.csv (summary via aggregate.py --clip)")
    ap.add_argument("--clip-k", type=float, default=CLIP_K,
                    help=f"physical ceiling multiplier K (default {CLIP_K})")
    args = ap.parse_args()
    main(clip=args.clip, clip_k=args.clip_k)
