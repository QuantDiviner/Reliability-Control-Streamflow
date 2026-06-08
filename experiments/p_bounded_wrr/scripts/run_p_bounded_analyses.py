"""
Run D-050 P-Bounded CPU-only analyses:
  - HopCPT 574-subset fair matrix with Global CP, HSCC, Global CQR, Mondrian-CQR, HopCPT
  - exp003 deployment fallback rule quantification
  - multiple-testing correction table
  - mechanism/causal claim audit over controlled non-manuscript docs
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/p_bounded_wrr/results"
REPORT_DIR = ROOT / "docs/reports"
OUT.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TIERS = ["dry", "semi_arid", "humid", "snow"]


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        running = min(running, p[idx] * n / rank)
        adj[idx] = running
    return np.minimum(adj, 1.0)


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        val = (n - rank + 1) * p[idx]
        running = max(running, val)
        adj_sorted[rank - 1] = min(running, 1.0)
    adj = np.empty(n, dtype=float)
    for pos, idx in enumerate(order):
        adj[idx] = adj_sorted[pos]
    return adj


def coverage_spread(rows):
    cov = [r["coverage"] for r in rows]
    return float((max(cov) - min(cov)) * 100)


def build_574_matrix():
    basin_574 = pd.read_csv(
        ROOT / "experiments/exp010/results/_analysis/per_basin_metrics.csv",
        dtype={"basin": str},
    )
    basin_574["basin"] = basin_574["basin"].str.zfill(8)
    subset = set(basin_574["basin"])

    base = pd.read_csv(ROOT / "experiments/exp010/results/_analysis/comparison_matrix_574subset.csv")
    rows = []
    for _, r in base[base["method"].isin(["Global_CP_exp009_5seed", "HSCC_exp009_5seed"])].iterrows():
        rows.append({
            "tier": r["tier"],
            "method": {"Global_CP_exp009_5seed": "Global_CP", "HSCC_exp009_5seed": "HSCC"}[r["method"]],
            "n_basins": int(r["n_basins"]),
            "coverage": float(r["coverage"]),
            "coverage_std": float(r["coverage_std"]) if pd.notna(r["coverage_std"]) else np.nan,
            "width_mm_d": float(r["width_mm_d"]),
            "width_std": float(r["width_std"]) if pd.notna(r["width_std"]) else np.nan,
            "winkler_mm_d": float(r["winkler_mm_d"]),
            "winkler_std": float(r["winkler_std"]) if pd.notna(r["winkler_std"]) else np.nan,
            "source": "exp010 comparison_matrix_574subset; exp009 5-seed",
        })

    hop = pd.read_csv(ROOT / "experiments/exp010/results/_3seed_aggregate/tier_3seed_summary.csv")
    for _, r in hop.iterrows():
        rows.append({
            "tier": r["tier"],
            "method": "HopCPT",
            "n_basins": int(r["n_basins"]),
            "coverage": float(r["coverage_mean"]),
            "coverage_std": float(r["coverage_std"]),
            "width_mm_d": float(r["width_mean"]),
            "width_std": float(r["width_std"]),
            "winkler_mm_d": float(r["winkler_mean"]),
            "winkler_std": float(r["winkler_std"]),
            "source": "exp010 3-seed aggregate on same 574 basins",
        })

    seed_rows = []
    for p in sorted((ROOT / "experiments/exp_global_cqr/results").glob("seed*/global_cqr_per_basin.csv")):
        df = pd.read_csv(p, dtype={"basin": str})
        df["basin"] = df["basin"].str.zfill(8)
        df = df[df["basin"].isin(subset)]
        seed = int(df["seed"].iloc[0])
        for tier in TIERS:
            sub = df[df["tier"] == tier]
            seed_rows.append({
                "seed": seed,
                "tier": tier,
                "n_basins": int(len(sub)),
                "coverage": float(sub["coverage"].mean()),
                "width_mm_d": float(sub["mean_pi_width"].mean()),
                "winkler_mm_d": float(sub["winkler_score"].mean()),
            })
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT / "global_cqr_574_per_seed_tier.csv", index=False)
    for tier in TIERS:
        sub = seed_df[seed_df["tier"] == tier]
        rows.append({
            "tier": tier,
            "method": "Global_CQR",
            "n_basins": int(sub["n_basins"].iloc[0]),
            "coverage": float(sub["coverage"].mean()),
            "coverage_std": float(sub["coverage"].std(ddof=1)),
            "width_mm_d": float(sub["width_mm_d"].mean()),
            "width_std": float(sub["width_mm_d"].std(ddof=1)),
            "winkler_mm_d": float(sub["winkler_mm_d"].mean()),
            "winkler_std": float(sub["winkler_mm_d"].std(ddof=1)),
            "source": "exp_global_cqr 5 seeds restricted to HopCPT 574 subset",
        })

    mondrian = pd.read_csv(ROOT / "experiments/exp011/results/_cqr_baseline/cqr_per_basin.csv", dtype={"basin": str})
    mondrian["basin"] = mondrian["basin"].str.zfill(8)
    mondrian = mondrian[mondrian["basin"].isin(subset)]
    for tier in TIERS:
        sub = mondrian[mondrian["tier"] == tier]
        rows.append({
            "tier": tier,
            "method": "Mondrian_CQR",
            "n_basins": int(len(sub)),
            "coverage": float(sub["coverage"].mean()),
            "coverage_std": float(sub["coverage"].std(ddof=1)),
            "width_mm_d": float(sub["mean_pi_width"].mean()),
            "width_std": np.nan,
            "winkler_mm_d": float(sub["winkler_score"].mean()),
            "winkler_std": np.nan,
            "source": "exp011 residual-CQR restricted to HopCPT 574 subset",
        })

    matrix = pd.DataFrame(rows)
    matrix = matrix[["tier", "method", "n_basins", "coverage", "coverage_std", "width_mm_d", "width_std", "winkler_mm_d", "winkler_std", "source"]]
    matrix.to_csv(OUT / "fair_matrix_574_with_cqr.csv", index=False)

    spread = []
    for method, sub in matrix.groupby("method"):
        spread.append({
            "method": method,
            "spread_pp": float((sub["coverage"].max() - sub["coverage"].min()) * 100),
            "mean_coverage": float(sub["coverage"].mean()),
            "mean_width_mm_d": float(sub["width_mm_d"].mean()),
            "mean_winkler_mm_d": float(sub["winkler_mm_d"].mean()),
            "min_tier": str(sub.loc[sub["coverage"].idxmin(), "tier"]),
            "max_tier": str(sub.loc[sub["coverage"].idxmax(), "tier"]),
        })
    spread_df = pd.DataFrame(spread).sort_values("spread_pp")
    spread_df.to_csv(OUT / "fair_matrix_574_spread_summary.csv", index=False)
    (OUT / "fair_matrix_574_with_cqr.json").write_text(json.dumps({
        "n_basins_subset": len(subset),
        "matrix": matrix.to_dict(orient="records"),
        "spread_summary": spread_df.to_dict(orient="records"),
        "note": "All methods restricted to the same HopCPT-eligible 574-basin subset; HopCPT uses existing 3-seed aggregate.",
    }, indent=2))
    return matrix, spread_df


def fallback_rule():
    df = pd.read_csv(ROOT / "experiments/exp003/transfer_matrix.csv")
    df["dominance_gap"] = df["global_coverage"] - df["hscc_coverage"]
    df["width_ratio_hscc_global"] = df["hscc_width_mm_d"] / df["global_width_mm_d"]

    def evaluate(floor=0.85, margin=0.02, use_low_power=True):
        trigger = (df["hscc_coverage"] < floor) | (df["dominance_gap"] > margin)
        if use_low_power:
            trigger = trigger | df["low_power"].astype(bool)
        selected_cov = np.where(trigger, df["global_coverage"], df["hscc_coverage"])
        selected_width = np.where(trigger, df["global_width_mm_d"], df["hscc_width_mm_d"])
        return pd.DataFrame({
            "trigger": trigger,
            "selected_coverage": selected_cov,
            "selected_width": selected_width,
        })

    default = evaluate()
    out_cells = df.copy()
    out_cells["fallback_trigger"] = default["trigger"]
    out_cells["selected_method"] = np.where(out_cells["fallback_trigger"], "Global_CP", "HSCC")
    out_cells["selected_coverage"] = default["selected_coverage"]
    out_cells["selected_width_mm_d"] = default["selected_width"]
    out_cells.to_csv(OUT / "fallback_rule_cells.csv", index=False)

    rng = np.random.default_rng(42)
    boot = []
    n = len(df)
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        b = out_cells.iloc[idx]
        boot.append({
            "fallback_rate": float(b["fallback_trigger"].mean()),
            "mean_selected_coverage": float(b["selected_coverage"].mean()),
            "mean_selected_width": float(b["selected_width_mm_d"].mean()),
            "selected_spread_pp": float((b["selected_coverage"].max() - b["selected_coverage"].min()) * 100),
        })
    boot_df = pd.DataFrame(boot)
    ci = {}
    for col in boot_df.columns:
        ci[col] = {
            "mean": float(boot_df[col].mean()),
            "ci95_lo": float(boot_df[col].quantile(0.025)),
            "ci95_hi": float(boot_df[col].quantile(0.975)),
        }

    sens_rows = []
    for floor in [0.80, 0.85, 0.88]:
        for margin in [0.00, 0.02, 0.05]:
            for use_low_power in [False, True]:
                ev = evaluate(floor, margin, use_low_power)
                sens_rows.append({
                    "floor": floor,
                    "dominance_margin": margin,
                    "use_low_power": use_low_power,
                    "fallback_rate": float(ev["trigger"].mean()),
                    "mean_selected_coverage": float(ev["selected_coverage"].mean()),
                    "mean_selected_width": float(ev["selected_width"].mean()),
                    "selected_spread_pp": float((ev["selected_coverage"].max() - ev["selected_coverage"].min()) * 100),
                })
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "fallback_rule_sensitivity.csv", index=False)

    exploratory = (
        ci["fallback_rate"]["ci95_hi"] - ci["fallback_rate"]["ci95_lo"] > 0.30
        or ci["mean_selected_coverage"]["ci95_lo"] < 0.85
        or ci["selected_spread_pp"]["mean"] > 30.0
    )
    summary = {
        "default_trigger": "fallback to Global CP if low_power OR HSCC coverage < 0.85 OR Global coverage exceeds HSCC by > 0.02 in the validation cell",
        "n_cells": int(n),
        "default": {
            "fallback_cells": int(out_cells["fallback_trigger"].sum()),
            "fallback_rate": float(out_cells["fallback_trigger"].mean()),
            "mean_selected_coverage": float(out_cells["selected_coverage"].mean()),
            "mean_selected_width": float(out_cells["selected_width_mm_d"].mean()),
            "selected_spread_pp": float((out_cells["selected_coverage"].max() - out_cells["selected_coverage"].min()) * 100),
            "hscc_only_mean_coverage": float(df["hscc_coverage"].mean()),
            "global_only_mean_coverage": float(df["global_coverage"].mean()),
        },
        "bootstrap_ci": ci,
        "stability_call": "exploratory_diagnostic" if exploratory else "operational_candidate",
        "stability_reason": (
            "downgraded because bootstrap lower coverage is below 0.85 and/or selected spread remains large"
            if exploratory else
            "bootstrap stability and selected coverage satisfy the operational-candidate screen"
        ),
    }
    (OUT / "fallback_rule_summary.json").write_text(json.dumps(summary, indent=2))
    return out_cells, sens, summary


def multiple_testing():
    tests = []
    exp008 = json.load(open(ROOT / "experiments/exp008/results/_analysis/random_distribution.json"))
    tests.append({
        "claim_family": "exp008 physical-vs-random spread",
        "claim": "physical tiers reduce cross-fold spread more than random tiers",
        "p_value": exp008["cross_fold"]["welch_p_value_one_sided_phys_gt_rand"],
        "scope": "primary",
    })
    et = pd.read_csv(ROOT / "experiments/exp008/results/_analysis/effect_size_table.csv")
    for _, r in et.iterrows():
        tests.append({
            "claim_family": "exp008 per-tier absolute coverage physical-vs-random",
            "claim": f"{r['tier']} per-tier coverage differs from random tiering",
            "p_value": float(r["p_value_two_sided"]),
            "scope": "secondary",
        })
    tests.append({
        "claim_family": "exp003 cookbook predictability",
        "claim": "cookbook predictor outperforms tier-only baseline",
        "p_value": float(json.load(open(ROOT / "experiments/exp003/cookbook_predictability_summary.json"))["paired_bootstrap_pvalue"]),
        "scope": "secondary",
    })
    causal = json.load(open(ROOT / "experiments/exp005/results/_analysis/causal_test.json"))
    for key in ["h1_ar1_increase_pvalue", "h4_hscc_drop_pvalue"]:
        if key in causal:
            tests.append({
                "claim_family": "exp005 lean causal tests",
                "claim": key,
                "p_value": float(causal[key]),
                "scope": "secondary",
            })
    sfull = pd.read_csv(ROOT / "experiments/exp007/results/run_2604/spearman_signed_full.csv")
    for var in ["log_resid_var", "Het", "AR1"]:
        cand = sfull[(sfull["var"].astype(str) == var) & (sfull["target"] == "cov_err_global_signed")]
        for _, r in cand.iterrows():
            tests.append({
                "claim_family": "exp007 residual diagnostic",
                "claim": f"{var} vs global signed coverage error",
                "p_value": float(r["p"]),
                "scope": "diagnostic",
            })

    df = pd.DataFrame(tests)
    df["holm_p"] = holm_adjust(df["p_value"])
    df["bh_fdr_p"] = bh_adjust(df["p_value"])
    df["classification"] = np.where(
        df["holm_p"] < 0.05,
        "corrected-significant",
        np.where(df["p_value"] < 0.05, "nominal-only", "descriptive"),
    )
    df.to_csv(OUT / "multiple_testing_correction.csv", index=False)
    (OUT / "multiple_testing_correction.json").write_text(json.dumps({
        "n_tests": int(len(df)),
        "method": "Holm family-wise correction plus Benjamini-Hochberg FDR diagnostic",
        "table": df.to_dict(orient="records"),
    }, indent=2))
    return df


def claim_audit():
    controlled = [
        ROOT / "PROJECT_CHARTER.md",
        ROOT / "EXPERIMENTS_INDEX.md",
        ROOT / "RESEARCH_STATE.md",
        ROOT / "docs/narrative-framework.md",
        ROOT / "docs/experiment-review.md",
        ROOT / "docs/fpa-results-digest.md",
        ROOT / "docs/research-summary.md",
    ]
    keywords = re.compile(r"\b(causal|causality|causally|mechanism|mechanistic|explain|explains|driver|drives)\b|机制|因果|解释|驱动", re.I)
    rows = []
    for path in controlled:
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            if keywords.search(line):
                lower = line.lower()
                if any(x in lower for x in [
                    "not causal", "not causal proof", "not claimed as causal", "not a causal",
                    "non-causal", "diagnostic", "correlational", "correlation",
                    "deferred", "future work", "null causal", "inconclusive", "acknowledg",
                    "causal proof or", "causal tests",
                    "不作 causal", "不是因果", "非因果", "未证明", "诊断", "相关", "降级",
                    "未来", "null", "不显著",
                ]):
                    status = "acceptable_diagnostic_or_disclaimer"
                elif any(x in lower for x in ["causal", "causality", "因果"]):
                    status = "needs_manual_downgrade"
                elif any(x in lower for x in ["mechanism", "mechanistic", "机制"]):
                    status = "review_for_diagnostic_framing"
                else:
                    status = "context_review"
                rows.append({
                    "file": str(path.relative_to(ROOT)),
                    "line": i,
                    "status": status,
                    "text": line.strip()[:260],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mechanism_claim_audit.csv", index=False)
    return df


def write_report(matrix, spread, fallback_summary, mt, audit):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = REPORT_DIR / f"{ts}_p_bounded_wrr_cpu_analyses.md"
    corrected = mt["classification"].value_counts().to_dict()
    audit_counts = audit["status"].value_counts().to_dict() if not audit.empty else {}
    spread_table = spread[[
        "method", "spread_pp", "mean_coverage", "mean_width_mm_d", "mean_winkler_mm_d", "min_tier", "max_tier"
    ]].copy()
    spread_table = spread_table.round({
        "spread_pp": 3,
        "mean_coverage": 4,
        "mean_width_mm_d": 3,
        "mean_winkler_mm_d": 3,
    })
    spread_md = [
        "| method | spread_pp | mean_coverage | mean_width_mm_d | mean_winkler_mm_d | min_tier | max_tier |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for _, r in spread_table.iterrows():
        spread_md.append(
            f"| {r['method']} | {r['spread_pp']} | {r['mean_coverage']} | "
            f"{r['mean_width_mm_d']} | {r['mean_winkler_mm_d']} | {r['min_tier']} | {r['max_tier']} |"
        )
    lines = [
        "# P-Bounded WRR CPU Analyses",
        "",
        f"Time: {ts}",
        "",
        "## 574-Subset Matrix",
        "",
        "All comparison rows are restricted to the HopCPT-eligible 574-basin subset.",
        "",
        "\n".join(spread_md),
        "",
        "Output: `experiments/p_bounded_wrr/results/fair_matrix_574_with_cqr.csv`.",
        "",
        "## Deployment Fallback Rule",
        "",
        f"Default trigger: {fallback_summary['default_trigger']}.",
        "",
        f"- Fallback cells: {fallback_summary['default']['fallback_cells']} / {fallback_summary['n_cells']}",
        f"- Mean selected coverage: {fallback_summary['default']['mean_selected_coverage']:.3f}",
        f"- Selected spread: {fallback_summary['default']['selected_spread_pp']:.2f} pp",
        f"- Stability call: {fallback_summary['stability_call']}",
        f"- Stability reason: {fallback_summary['stability_reason']}",
        "",
        "Output: `experiments/p_bounded_wrr/results/fallback_rule_summary.json`.",
        "",
        "## Multiple Testing",
        "",
        f"Classification counts: {corrected}.",
        "",
        "Output: `experiments/p_bounded_wrr/results/multiple_testing_correction.csv`.",
        "",
        "## Mechanism / Causal Claim Audit",
        "",
        f"Audit status counts: {audit_counts}.",
        "",
        "Output: `experiments/p_bounded_wrr/results/mechanism_claim_audit.csv`.",
        "",
        "## LORO Trigger Check",
        "",
        "No immediate LORO spot-check is triggered by this CPU-only pass alone. The fallback rule is marked according to its bootstrap stability field; if FPA Round 12 still blocks on C1.STOCH_ROBUST, the D-050 pre-registered limited LORO trigger remains active.",
        "",
    ]
    report.write_text("\n".join(lines))
    return report


def main():
    matrix, spread = build_574_matrix()
    _, _, fallback_summary = fallback_rule()
    mt = multiple_testing()
    audit = claim_audit()
    report = write_report(matrix, spread, fallback_summary, mt, audit)
    print(f"wrote report: {report}")


if __name__ == "__main__":
    main()
