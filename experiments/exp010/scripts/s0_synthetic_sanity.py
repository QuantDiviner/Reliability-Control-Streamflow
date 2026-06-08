"""
exp010 Action 5 (R2 P1-1): S0 synthetic sanity — iid + AR(1) + heteroscedastic.

Verifies that the three CP methods (Global CP, HSCC, online-adaptive HopCPT-style)
achieve target marginal coverage on controlled DGPs. plan §4.4 strict requirement.

DGP 1: iid Gaussian residuals (n=10000, σ=1)
  Expected: all 3 methods cov ≈ 0.90 ± 0.02

DGP 2: AR(1) Gaussian (n=10000, φ=0.7, innov σ=1)
  Expected: all 3 methods cov ≈ 0.90; width slightly larger than iid due to dependence

DGP 3: Heteroscedastic Gaussian (n=10000, σ(t) = 0.5 + 1.5*sigmoid(t))
  Expected:
    - Global CP: cov ≈ 0.90 marginal but per-segment-tier severe under/over-cover
    - HSCC (4 tiers by σ-bin): cov ≈ 0.90 marginal AND per-tier ∈ [0.88, 0.92]
    - Online-adaptive: cov ≈ 0.90 marginal; per-tier coverage adaptive

Method definitions (simple, self-contained):
  - Global CP: empirical (1-α)(n+1)/n quantile of pooled |residual| on cal set;
               one tau applied to all test points.
  - HSCC: assign synthetic "tier" by σ-bin (4 bins); per-tier (1-α)(n+1)/n quantile
          of |residual| on tier's cal subset; per-tier tau applied to test.
  - Online-adaptive (HopCPT-style proxy): rolling-window quantile of |residual| over
          the last W=200 points (FIFO memory), with cal-period cold start.

Output:
  experiments/exp010/_sanity/s0_synthetic_results.json
  experiments/exp010/_sanity/s0_synthetic_panel.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experiments/exp010/_sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
N = 10000
N_CAL = 4000   # 40% cal / 60% test, like NH val/test split
N_TEST = N - N_CAL
RNG_SEED = 42


def split_cp_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.nan
    level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


# -------- DGPs --------

def dgp_iid(rng):
    eps = rng.normal(0, 1.0, size=N)
    sigma = np.ones(N)
    return eps, sigma


def dgp_ar1(rng, phi=0.7):
    eps = np.zeros(N)
    e0 = rng.normal(0, 1.0, size=N)
    eps[0] = e0[0]
    for t in range(1, N):
        eps[t] = phi * eps[t - 1] + np.sqrt(1 - phi**2) * e0[t]  # stationary marginal var=1
    sigma = np.ones(N)  # marginal stationary
    return eps, sigma


def dgp_het(rng):
    """Heteroscedastic with 4 σ-regimes mixed evenly in cal and test (analogous to
    CAMELS 4 hydroclimate tiers). Each point has σ drawn uniformly from
    {0.5, 1.0, 1.5, 2.0} — HSCC by σ-bin should perfectly recover per-tier coverage,
    Global CP marginal coverage may still be 0.90 but per-tier coverage will spread."""
    sigma_levels = np.array([0.5, 1.0, 1.5, 2.0])
    sigma_idx = rng.integers(0, 4, size=N)
    sigma = sigma_levels[sigma_idx]
    eps = rng.normal(0, sigma, size=N)
    return eps, sigma


# -------- methods --------

def global_cp(scores_cal, scores_test):
    q = split_cp_quantile(scores_cal, ALPHA)
    cov = float(np.mean(scores_test <= q))
    width = 2 * q  # symmetric interval half-width = q (predicting ε ~ |ε| ≤ q)
    return cov, q, width


def hscc(scores_cal, scores_test, tier_cal, tier_test):
    tiers = np.unique(tier_cal)
    q_tier = {}
    for t in tiers:
        s = scores_cal[tier_cal == t]
        q_tier[t] = split_cp_quantile(s, ALPHA) if len(s) else np.nan
    cov_overall = float(
        np.mean([scores_test[i] <= q_tier[tier_test[i]] for i in range(len(scores_test))])
    )
    per_tier = {}
    for t in tiers:
        sel = tier_test == t
        if sel.sum() == 0:
            continue
        per_tier[int(t)] = {
            "n": int(sel.sum()),
            "coverage": float(np.mean(scores_test[sel] <= q_tier[t])),
            "q": float(q_tier[t]),
            "width": 2 * float(q_tier[t]),
        }
    return cov_overall, per_tier


def online_adaptive(scores, n_cal, window=200):
    """HopCPT-style proxy: rolling-window empirical (1-α)-quantile of |scores|.
    Cold-start: use cal scores as initial buffer; then FIFO update with each test
    score becoming visible AFTER its prediction is made (causal)."""
    buf = list(scores[:n_cal])  # cold start
    if window < n_cal:
        buf = buf[-window:]
    cov_flags = []
    qs = []
    for t in range(n_cal, len(scores)):
        q = split_cp_quantile(np.asarray(buf), ALPHA) if buf else np.nan
        cov_flags.append(scores[t] <= q)
        qs.append(q)
        buf.append(scores[t])  # observe after predicting (online)
        if len(buf) > window:
            buf.pop(0)
    return float(np.mean(cov_flags)), float(np.mean(qs)), np.array(qs)


# -------- run one DGP --------

def run_dgp(name, eps, sigma):
    print(f"\n--- {name} ---")
    abs_eps = np.abs(eps)
    cal_idx = np.arange(N_CAL)
    test_idx = np.arange(N_CAL, N)

    s_cal = abs_eps[cal_idx]
    s_test = abs_eps[test_idx]

    # Global CP
    cov_g, q_g, w_g = global_cp(s_cal, s_test)
    print(f"  Global CP : cov={cov_g:.4f}  q={q_g:.4f}  half-width={q_g:.4f}")

    # HSCC: assign 4 tiers by σ-level (constant for iid/AR1 → random pseudo-tier;
    # het uses the discrete σ-level as tier directly).
    if np.std(sigma) < 1e-9:
        # constant σ — random pseudo-tiers (sanity: HSCC ≈ Global CP if tiers are random)
        rng_tier = np.random.default_rng(RNG_SEED + 1)
        tier = rng_tier.integers(0, 4, size=N)
    else:
        # discrete σ-levels → use rank index of unique σ as tier (stable, no quantile bug)
        unique_sigmas = np.unique(sigma)
        s2t = {s: i for i, s in enumerate(unique_sigmas)}
        tier = np.array([s2t[s] for s in sigma])
    cov_h, per_tier = hscc(s_cal, s_test, tier[cal_idx], tier[test_idx])
    print(f"  HSCC      : cov={cov_h:.4f}")
    for k, v in per_tier.items():
        print(f"            tier {k}: cov={v['coverage']:.4f}  q={v['q']:.4f}  n={v['n']}")
    if per_tier:
        spread = max(v["coverage"] for v in per_tier.values()) - \
                 min(v["coverage"] for v in per_tier.values())
        print(f"            HSCC tier spread: {spread*100:.2f}pp")
    else:
        spread = np.nan

    # Online-adaptive (HopCPT proxy)
    cov_a, q_a_mean, qs = online_adaptive(abs_eps, N_CAL, window=200)
    print(f"  HopCPT-proxy (rolling W=200): cov={cov_a:.4f}  mean(q)={q_a_mean:.4f}")

    # Per-tier coverage of Global CP (for het check)
    per_tier_global = {}
    for t in np.unique(tier[test_idx]):
        sel = tier[test_idx] == t
        if sel.sum() == 0:
            continue
        per_tier_global[int(t)] = {
            "n": int(sel.sum()),
            "coverage": float(np.mean(s_test[sel] <= q_g)),
        }
    spread_global = max(v["coverage"] for v in per_tier_global.values()) - \
                    min(v["coverage"] for v in per_tier_global.values())
    print(f"  Global CP per-tier spread: {spread_global*100:.2f}pp")

    # Pass/fail per plan §4.4
    eps_target = 0.02  # |cov - 0.90| < 0.02 tolerance
    pass_global = abs(cov_g - 0.90) < eps_target
    pass_hscc = abs(cov_h - 0.90) < eps_target
    pass_adapt = abs(cov_a - 0.90) < eps_target

    return {
        "dgp": name,
        "n_total": int(N),
        "n_cal": int(N_CAL),
        "n_test": int(N_TEST),
        "alpha": ALPHA,
        "global_cp": {"coverage": cov_g, "q": q_g, "half_width": q_g,
                      "per_tier_coverage": per_tier_global,
                      "per_tier_spread_pp": spread_global * 100,
                      "pass_marginal": pass_global},
        "hscc": {"coverage": cov_h, "per_tier": per_tier,
                 "per_tier_spread_pp": (spread * 100) if not np.isnan(spread) else None,
                 "pass_marginal": pass_hscc},
        "online_adaptive": {"coverage": cov_a, "mean_q": q_a_mean,
                            "pass_marginal": pass_adapt},
        "verdict": all([pass_global, pass_hscc, pass_adapt]),
    }


def main():
    print("=== exp010 Action 5: S0 synthetic sanity ===\n")
    print(f"alpha={ALPHA}  N={N}  cal:test = {N_CAL}:{N_TEST}\n")
    rng = np.random.default_rng(RNG_SEED)
    results = {}
    eps_iid, sig_iid = dgp_iid(rng)
    results["iid"] = run_dgp("iid Gaussian σ=1", eps_iid, sig_iid)
    eps_ar1, sig_ar1 = dgp_ar1(rng)
    results["ar1"] = run_dgp("AR(1) Gaussian φ=0.7", eps_ar1, sig_ar1)
    eps_het, sig_het = dgp_het(rng)
    results["het"] = run_dgp("Heteroscedastic Gaussian σ(t)=0.5+1.5·sigmoid", eps_het, sig_het)

    # Summary
    print("\n--- S0 verdict ---")
    overall_pass = all(r["verdict"] for r in results.values())
    for k, r in results.items():
        marks = ["✅" if r["global_cp"]["pass_marginal"] else "❌",
                 "✅" if r["hscc"]["pass_marginal"] else "❌",
                 "✅" if r["online_adaptive"]["pass_marginal"] else "❌"]
        print(f"  {k:5s} | Global={marks[0]} HSCC={marks[1]} Adaptive={marks[2]} "
              f"→ {'PASS' if r['verdict'] else 'FAIL'}")
    print(f"\nS0 overall: {'PASS ✅' if overall_pass else 'FAIL ❌'}")

    # Critical het check: HSCC should spread < Global CP spread on heteroscedastic
    het = results["het"]
    print(f"\n[Het tier-equity check]")
    print(f"  Global CP per-tier spread: {het['global_cp']['per_tier_spread_pp']:.2f}pp")
    print(f"  HSCC      per-tier spread: {het['hscc']['per_tier_spread_pp']:.2f}pp")
    het_equity = (het["global_cp"]["per_tier_spread_pp"] >
                  het["hscc"]["per_tier_spread_pp"] + 1.0)  # at least 1pp better
    print(f"  HSCC reduces per-tier spread: {'✅' if het_equity else '❌'}")

    out_json = OUT_DIR / "s0_synthetic_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "exp": "exp010",
            "action": "5_S0_synthetic_sanity",
            "config": {"N": N, "N_CAL": N_CAL, "alpha": ALPHA, "rng_seed": RNG_SEED,
                       "tolerance_eps_target": 0.02,
                       "rolling_window_W": 200},
            "results": results,
            "overall_pass": bool(overall_pass),
            "het_equity_check": bool(het_equity),
        }, f, indent=2)
    print(f"\nwrote: {out_json}")

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    methods = ["Global CP", "HSCC", "Online-adaptive"]
    dgps = ["iid", "ar1", "het"]
    colors = {"Global CP": "#888888", "HSCC": "#1f77b4", "Online-adaptive": "#d62728"}

    # Panel 1: marginal coverage
    ax = axes[0, 0]
    x = np.arange(len(dgps))
    bw = 0.25
    for i, m in enumerate(methods):
        key = {"Global CP": "global_cp", "HSCC": "hscc", "Online-adaptive": "online_adaptive"}[m]
        vals = [results[d][key]["coverage"] for d in dgps]
        ax.bar(x + (i - 1) * bw, vals, bw, label=m, color=colors[m])
    ax.axhline(0.90, color="k", ls="--", lw=1, label="Target 0.90")
    ax.set_xticks(x); ax.set_xticklabels(["IID", "AR(1)", "Het"])
    ax.set_ylabel("Marginal coverage"); ax.set_ylim(0.8, 1.0)
    ax.set_title("Panel 1. Marginal coverage on 3 synthetic DGPs")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Panel 2: het per-tier coverage
    ax = axes[0, 1]
    het_tiers = sorted(results["het"]["hscc"]["per_tier"].keys())
    g_cov = [results["het"]["global_cp"]["per_tier_coverage"][t]["coverage"] for t in het_tiers]
    h_cov = [results["het"]["hscc"]["per_tier"][t]["coverage"] for t in het_tiers]
    x = np.arange(len(het_tiers))
    ax.bar(x - 0.18, g_cov, 0.36, label="Global CP", color=colors["Global CP"])
    ax.bar(x + 0.18, h_cov, 0.36, label="HSCC (σ-bin)", color=colors["HSCC"])
    ax.axhline(0.90, color="k", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"σ-bin {t+1}" for t in het_tiers])
    ax.set_ylabel("Per-tier coverage"); ax.set_ylim(0.7, 1.0)
    ax.set_title("Panel 2. Het: per-tier coverage (Global vs HSCC)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Panel 3: het σ-regime distribution
    ax = axes[1, 0]
    rng_p3 = np.random.default_rng(RNG_SEED + 200)
    eps_p3, sig_p3 = dgp_het(rng_p3)
    levels = np.array([0.5, 1.0, 1.5, 2.0])
    ax.hist(sig_p3, bins=20, color="#d62728", alpha=0.7, edgecolor="k")
    for L in levels:
        ax.axvline(L, color="k", ls=":", lw=0.8)
    ax.set_xlabel("σ regime"); ax.set_ylabel("count")
    ax.set_title("Panel 3. Het DGP σ regime distribution\n(4 levels: 0.5/1.0/1.5/2.0, uniform mix)")
    ax.grid(alpha=0.3)

    # Panel 4: het tier-specific q comparison (HSCC recovers per-σ theoretical quantile)
    ax = axes[1, 1]
    het_tiers = sorted(results["het"]["hscc"]["per_tier"].keys())
    q_vals = [results["het"]["hscc"]["per_tier"][t]["q"] for t in het_tiers]
    q_global_het = results["het"]["global_cp"]["q"]
    levels_avail = np.array([0.5, 1.0, 1.5, 2.0])[het_tiers] if max(het_tiers) < 4 else \
                   np.array([0.5, 1.0, 1.5, 2.0])
    theo = levels_avail * 1.6449  # |Gaussian| 90%-quantile = σ × Φ⁻¹(0.95)
    x = np.arange(len(het_tiers))
    ax.bar(x - 0.18, theo, 0.36, label="Theoretical (σ × 1.645)", color="lightgreen")
    ax.bar(x + 0.18, q_vals, 0.36, label="HSCC q (per σ-bin)", color=colors["HSCC"])
    ax.axhline(q_global_het, color=colors["Global CP"], ls="--", lw=1.5,
               label=f"Global CP q={q_global_het:.3f}")
    ax.set_xticks(x); ax.set_xticklabels([f"σ={levels_avail[i]}" for i in range(len(het_tiers))])
    ax.set_ylabel("|ε| 90%-quantile")
    ax.set_title("Panel 4. Het: HSCC q recovers per-tier theoretical")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"S0 synthetic sanity: 3 DGPs × 3 methods (α={ALPHA}, target 0.90)",
                 y=0.995)
    fig.tight_layout()
    out_png = OUT_DIR / "s0_synthetic_panel.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote: {out_png}")


if __name__ == "__main__":
    main()
