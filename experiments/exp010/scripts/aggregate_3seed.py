"""
exp010 Action 6 aggregator — combine seeds [42, 137, 2024] for HopCPT 3-seed mean ± std.

Reads each seed's `_analysis/metrics.json` + `tier_aggregate.csv` + `per_basin_metrics.csv`,
aggregates per-tier (n_basins, mean_coverage, mean_pi_width, mean_winkler_score) across
seeds, and emits 3-seed mean ± std + per-basin coverage variability.

Output:
  experiments/exp010/results/_3seed_aggregate/
    aggregate.json                  # full per-seed + per-tier mean ± std
    tier_3seed_summary.csv          # tier × {coverage_mean, coverage_std, ...}
    per_basin_3seed.csv             # basin × seed coverage matrix
    figure_3seed_panel.png          # 4-panel (cov / width / winkler per tier + per-basin std hist)

Usage (called automatically by post-multiseed hook, but also manual):
  /home/qingsong/miniconda3/envs/hscc-hydrology/bin/python \
    experiments/exp010/scripts/aggregate_3seed.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
TIERS = ["dry", "semi_arid", "humid", "snow"]
TIER_PRETTY = {"dry": "Dry", "semi_arid": "Semi-arid", "humid": "Humid", "snow": "Snow"}

# seed=42 reuses original exp010 production analysis;
# seed=137/2024 use new production_seed*_*.log → run_*/_analysis/
SEED_RUN_GLOBS = {
    42:   "experiments/exp010/results/run_030526_120659",  # original production
    137:  None,  # filled at runtime via glob
    2024: None,  # filled at runtime via glob
}
OUT_DIR = ROOT / "experiments/exp010/results/_3seed_aggregate"


def find_log_for_seed(seed):
    """Find latest production log for given seed."""
    if seed == 42:
        return ROOT / "experiments/exp010/logs/production_20260503_120656.log"
    pattern = f"experiments/exp010/logs/production_seed{seed}_*.log"
    matches = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def find_run_for_seed(seed):
    """Find run dir for given seed by checking which run_*/hydra_run dir exists with seed name."""
    if seed == 42:
        return ROOT / "experiments/exp010/results/run_030526_120659"
    candidates = sorted(
        (ROOT / "experiments/exp010/results").glob("run_*"),
        key=lambda p: p.stat().st_mtime,
    )
    for c in candidates:
        for hd in c.glob("hydra_run/hopcpt_camels_us_574_seed*_*"):
            if f"_seed{seed}_" in hd.name:
                return c
    # Fallback: find run dir whose mtime is just after seed log mtime
    log = find_log_for_seed(seed)
    if log:
        log_mtime = log.stat().st_mtime
        for c in candidates:
            if abs(c.stat().st_mtime - log_mtime) < 7200:  # within 2h
                return c
    return None


def ensure_analysis_for_seed(seed):
    """Run stage1_per_seed.py if metrics.json missing for this seed's run dir."""
    import subprocess
    log = find_log_for_seed(seed)
    rd = find_run_for_seed(seed)
    if log is None:
        print(f"  [seed={seed}] no log file found")
        return None
    if rd is None:
        # Use a flat analysis dir under exp010 for runs we couldn't locate
        rd = ROOT / f"experiments/exp010/results/_seed{seed}_analysis"
    analysis = rd / "_analysis"
    if (analysis / "metrics.json").exists():
        return analysis
    print(f"  [seed={seed}] running stage1_per_seed.py on log={log.name}")
    analysis.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/home/qingsong/miniconda3/envs/hscc-hydrology/bin/python",
        str(ROOT / "experiments/exp010/scripts/stage1_per_seed.py"),
        "--log_path", str(log),
        "--out_dir", str(analysis),
        "--seed", str(seed),
        "--run_dir", str(rd),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [seed={seed}] stage1 failed (exit {res.returncode}):")
        print(res.stderr[-2000:])
        print(res.stdout[-500:])
        return None
    print(res.stdout[-500:])
    return analysis


def load_seed_metrics(seed):
    analysis = ensure_analysis_for_seed(seed)
    if analysis is None:
        print(f"  [seed={seed}] no analysis available")
        return None
    metrics = json.load(open(analysis / "metrics.json"))
    tier_df = pd.read_csv(analysis / "tier_aggregate.csv")
    pb_df = pd.read_csv(analysis / "per_basin_metrics.csv", dtype={"basin": str})
    pb_df["basin"] = pb_df["basin"].str.zfill(8)
    return {
        "seed": seed,
        "analysis_dir": str(analysis),
        "metrics": metrics,
        "tier_df": tier_df,
        "per_basin_df": pb_df,
    }


def main():
    print("=== exp010 Action 6: 3-seed aggregate (HopCPT) ===\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_data = {}
    for s in [42, 137, 2024]:
        d = load_seed_metrics(s)
        if d is not None:
            seed_data[s] = d
    print(f"\nLoaded {len(seed_data)} seeds: {list(seed_data.keys())}")
    if len(seed_data) < 2:
        print("ERROR: need at least 2 seeds")
        return

    # Per-tier 3-seed mean ± std
    rows = []
    for t in TIERS:
        per_seed_rows = []
        for s, d in seed_data.items():
            tdf = d["tier_df"]
            sub = tdf[tdf["tier"] == t]
            if sub.empty:
                continue
            per_seed_rows.append({
                "seed": s,
                "n_basins": int(sub["n_basins"].iloc[0]),
                "mean_coverage": float(sub["mean_coverage"].iloc[0]),
                "mean_pi_width": float(sub["mean_pi_width"].iloc[0]),
                "mean_winkler": float(sub["mean_winkler_score"].iloc[0]),
            })
        if not per_seed_rows:
            continue
        arr = pd.DataFrame(per_seed_rows)
        rows.append({
            "tier": t,
            "n_basins": int(arr["n_basins"].iloc[0]),
            "n_seeds": len(arr),
            "coverage_mean": float(arr["mean_coverage"].mean()),
            "coverage_std":  float(arr["mean_coverage"].std(ddof=1)),
            "coverage_min":  float(arr["mean_coverage"].min()),
            "coverage_max":  float(arr["mean_coverage"].max()),
            "width_mean":    float(arr["mean_pi_width"].mean()),
            "width_std":     float(arr["mean_pi_width"].std(ddof=1)),
            "winkler_mean":  float(arr["mean_winkler"].mean()),
            "winkler_std":   float(arr["mean_winkler"].std(ddof=1)),
        })
    df_summary = pd.DataFrame(rows)
    print("\n=== 3-seed per-tier summary ===")
    print(df_summary.to_string(index=False))

    out_csv = OUT_DIR / "tier_3seed_summary.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nwrote: {out_csv}")

    # Cross-tier spread per seed
    spread_per_seed = []
    for s, d in seed_data.items():
        tdf = d["tier_df"]
        spread = (tdf["mean_coverage"].max() - tdf["mean_coverage"].min()) * 100
        spread_per_seed.append({"seed": s, "spread_pp": float(spread),
                                "marginal": float(d["metrics"]["overall"]["mean_coverage"]),
                                "n_basins": int(d["metrics"].get("basin_count_actual",
                                                                  d["metrics"].get("basin_count_planned", 574)))})
    df_spread = pd.DataFrame(spread_per_seed)
    print("\n=== Per-seed spread + marginal ===")
    print(df_spread.to_string(index=False))
    print(f"\n3-seed spread: mean={df_spread['spread_pp'].mean():.2f}pp std={df_spread['spread_pp'].std(ddof=1):.2f}pp")
    print(f"3-seed marginal: mean={df_spread['marginal'].mean():.4f} std={df_spread['marginal'].std(ddof=1):.4f}")

    # Per-basin coverage matrix
    pb_seed_dfs = {s: d["per_basin_df"][["basin", "tier", "mean_coverage"]].rename(
                       columns={"mean_coverage": f"cov_seed{s}"})
                   for s, d in seed_data.items()}
    pb_matrix = list(pb_seed_dfs.values())[0]
    for s in list(seed_data.keys())[1:]:
        pb_matrix = pb_matrix.merge(pb_seed_dfs[s][["basin", f"cov_seed{s}"]], on="basin")
    cov_cols = [c for c in pb_matrix.columns if c.startswith("cov_seed")]
    pb_matrix["cov_mean"] = pb_matrix[cov_cols].mean(axis=1)
    pb_matrix["cov_std"] = pb_matrix[cov_cols].std(axis=1, ddof=1)
    out_pb = OUT_DIR / "per_basin_3seed.csv"
    pb_matrix.to_csv(out_pb, index=False)
    print(f"\nwrote: {out_pb}")

    # JSON dump
    out_json = OUT_DIR / "aggregate.json"
    with open(out_json, "w") as f:
        json.dump({
            "exp": "exp010",
            "action": "6_HopCPT_+2_seeds_aggregate",
            "seeds": list(seed_data.keys()),
            "tier_3seed_summary": df_summary.to_dict(orient="records"),
            "spread_per_seed": spread_per_seed,
            "spread_3seed_mean_pp": float(df_spread["spread_pp"].mean()),
            "spread_3seed_std_pp": float(df_spread["spread_pp"].std(ddof=1)),
            "marginal_3seed_mean": float(df_spread["marginal"].mean()),
            "marginal_3seed_std": float(df_spread["marginal"].std(ddof=1)),
            "per_basin_cov_std_mean": float(pb_matrix["cov_std"].mean()),
            "per_basin_cov_std_max": float(pb_matrix["cov_std"].max()),
        }, f, indent=2)
    print(f"wrote: {out_json}")

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(TIERS))

    # Panel 1: per-tier coverage with std bars
    ax = axes[0, 0]
    sub = df_summary.set_index("tier").reindex(TIERS)
    ax.errorbar(x, sub["coverage_mean"], yerr=sub["coverage_std"] * 1.96,
                fmt="s", color="#d62728", capsize=4, markersize=8, lw=1.5,
                label="HopCPT 3-seed mean ± 95% CI")
    ax.axhline(0.90, color="k", ls="--", lw=1, label="Target 0.90")
    ax.set_xticks(x); ax.set_xticklabels([TIER_PRETTY[t] for t in TIERS])
    ax.set_ylabel("Mean coverage")
    ax.set_title("Panel 1. Per-tier coverage (3-seed)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: per-tier width + winkler
    ax = axes[0, 1]
    ax.errorbar(x - 0.1, sub["width_mean"], yerr=sub["width_std"] * 1.96,
                fmt="o", color="#1f77b4", capsize=4, markersize=7, label="Mean PI width (mm/d)")
    ax2 = ax.twinx()
    ax2.errorbar(x + 0.1, sub["winkler_mean"], yerr=sub["winkler_std"] * 1.96,
                 fmt="^", color="#2ca02c", capsize=4, markersize=7, label="Mean Winkler (mm/d)")
    ax.set_xticks(x); ax.set_xticklabels([TIER_PRETTY[t] for t in TIERS])
    ax.set_ylabel("PI width (mm/d)", color="#1f77b4")
    ax2.set_ylabel("Winkler (mm/d)", color="#2ca02c")
    ax.set_title("Panel 2. Per-tier width + Winkler (3-seed mean)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 3: spread per seed
    ax = axes[1, 0]
    bx = np.arange(len(spread_per_seed))
    ax.bar(bx, [r["spread_pp"] for r in spread_per_seed], color="#d62728", edgecolor="k")
    for bi, r in enumerate(spread_per_seed):
        ax.text(bi, r["spread_pp"] + 0.1, f"{r['spread_pp']:.2f}pp", ha="center", fontsize=9)
    ax.axhline(df_spread["spread_pp"].mean(), color="k", ls="--", lw=1,
               label=f"3-seed mean {df_spread['spread_pp'].mean():.2f}pp")
    ax.set_xticks(bx); ax.set_xticklabels([f"seed={r['seed']}" for r in spread_per_seed])
    ax.set_ylabel("Cross-tier coverage spread (pp)")
    ax.set_title("Panel 3. Per-seed cross-tier spread")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Panel 4: per-basin coverage std distribution
    ax = axes[1, 1]
    bins = np.linspace(0, max(0.05, pb_matrix["cov_std"].max() * 1.05), 40)
    ax.hist(pb_matrix["cov_std"], bins=bins, color="#888", edgecolor="k")
    ax.axvline(pb_matrix["cov_std"].mean(), color="k", ls="--", lw=1.5,
               label=f"mean σ_cov={pb_matrix['cov_std'].mean():.4f}")
    ax.set_xlabel("Per-basin coverage σ across 3 seeds")
    ax.set_ylabel("Number of basins")
    ax.set_title("Panel 4. Per-basin coverage variability across 3 seeds")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"exp010 Action 6: HopCPT 3-seed aggregate (seeds {list(seed_data.keys())})",
                 y=0.995, fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / "figure_3seed_panel.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote: {out_png}")


if __name__ == "__main__":
    main()
