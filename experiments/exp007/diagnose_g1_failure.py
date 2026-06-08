"""
exp007 Gate G1 失败诊断：
  原 spearman 用 |cov − 0.9| 把 dry under-coverage 和 snow over-coverage 混合。
  这里：
    1. 加 signed coverage error (cov − 0.9) — 带方向
    2. per-tier Spearman 拆分（AR1/Het vs cov_signed within each tier）
    3. AR1 distribution diagnostics (是否 671 basins AR1 差异太小)
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/exp002"))

OUT = Path("experiments/exp007/results/run_2604")
df = pd.read_csv(OUT / "mechanism_metrics.csv")

print(f"\n=== exp007 G1 Failure Diagnosis (n_basins={len(df)}) ===")

# Add signed coverage errors
df["cov_err_global_signed"] = df["coverage_global"] - 0.90
df["cov_err_hscc_signed"] = df["coverage_hscc"] - 0.90
df.to_csv(OUT / "mechanism_metrics.csv", index=False)

# Step 1: AR1 distribution diagnostics
print("\n[1] AR1 distribution across 671 basins:")
print(f"  min/median/max: {df['AR1'].min():.3f} / {df['AR1'].median():.3f} / {df['AR1'].max():.3f}")
print(f"  IQR: [{df['AR1'].quantile(0.25):.3f}, {df['AR1'].quantile(0.75):.3f}]")
print(f"  std: {df['AR1'].std():.3f}")

print("\n[1] Het distribution across 671 basins:")
print(f"  min/median/max: {df['Het'].min():.3f} / {df['Het'].median():.3f} / {df['Het'].max():.3f}")
print(f"  IQR: [{df['Het'].quantile(0.25):.3f}, {df['Het'].quantile(0.75):.3f}]")
print(f"  finite n: {df['Het'].notna().sum()}/{len(df)}")

# Per-tier AR1 / Het
print("\n[1] per-tier AR1 / Het / cov_signed:")
for t, sub in df.groupby("tier"):
    print(f"  {t:10s} n={len(sub):3d}  AR1={sub['AR1'].median():.3f}  "
          f"Het={sub['Het'].median():.3f}  cov_signed_mean={sub['cov_err_global_signed'].mean():+.3f}")

# Step 2: Signed coverage Spearman (vs |.|)
print("\n[2] Spearman with signed cov_err (full pool, n=671):")
mech_vars = ["AR1", "Het", "NSE_test", "KGE_test", "log_resid_var", "low_flow_bias", "high_flow_bias"]
targets_compare = [
    ("miscoverage_global", "abs"),
    ("cov_err_global_signed", "signed"),
    ("miscoverage_hscc", "abs"),
    ("cov_err_hscc_signed", "signed"),
]
sig_rows = []
for var in mech_vars:
    for tgt, kind in targets_compare:
        x = df[var].values
        y = df[tgt].values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 30:
            continue
        rho, p = stats.spearmanr(x[m], y[m])
        sig_rows.append({"var": var, "target": tgt, "kind": kind,
                         "spearman_rho": float(rho), "p": float(p), "n": int(m.sum())})

sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(OUT / "spearman_signed_full.csv", index=False)

# Print top picks
print("\n[2] Spearman vs signed cov_err_global_signed (top |rho|):")
sub = sig_df[sig_df["target"] == "cov_err_global_signed"].sort_values("spearman_rho", key=abs, ascending=False)
for _, r in sub.iterrows():
    flag = " ✅" if abs(r["spearman_rho"]) >= 0.4 else ""
    print(f"  {r['var']:18s} rho={r['spearman_rho']:+.3f}  p={r['p']:.4g}  n={r['n']}{flag}")

print("\n[2] Spearman vs |miscov_global| (top |rho|, original definition):")
sub = sig_df[sig_df["target"] == "miscoverage_global"].sort_values("spearman_rho", key=abs, ascending=False)
for _, r in sub.iterrows():
    flag = " ✅" if abs(r["spearman_rho"]) >= 0.4 else ""
    print(f"  {r['var']:18s} rho={r['spearman_rho']:+.3f}  p={r['p']:.4g}  n={r['n']}{flag}")

# Step 3: per-tier Spearman
print("\n[3] per-tier Spearman (AR1, Het vs signed cov_err):")
per_tier_rows = []
for t in ["dry", "semi_arid", "humid", "snow"]:
    sub = df[df["tier"] == t]
    print(f"\n  tier={t:10s} (n={len(sub)})")
    for var in ["AR1", "Het", "NSE_test", "KGE_test", "log_resid_var"]:
        for tgt in ["cov_err_global_signed", "cov_err_hscc_signed", "miscoverage_global"]:
            x = sub[var].values
            y = sub[tgt].values
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 20:
                continue
            rho, p = stats.spearmanr(x[m], y[m])
            per_tier_rows.append({"tier": t, "var": var, "target": tgt,
                                   "spearman_rho": float(rho), "p": float(p), "n": int(m.sum())})
            if abs(rho) >= 0.3:
                tag = " ✅ ≥0.3" if abs(rho) >= 0.3 else ""
                print(f"    {var:15s} vs {tgt:25s} rho={rho:+.3f} p={p:.3f}{tag}")

per_tier_df = pd.DataFrame(per_tier_rows)
per_tier_df.to_csv(OUT / "spearman_per_tier.csv", index=False)

# Step 4: Aridity / frac_snow direct correlation with miscoverage
print("\n[4] Direct attribute correlation with cov_err_global_signed (full pool):")
for var in ["aridity", "frac_snow"]:
    x = df[var].values
    y = df["cov_err_global_signed"].values
    rho, p = stats.spearmanr(x, y)
    flag = " ✅" if abs(rho) >= 0.4 else ""
    print(f"  {var:15s} rho={rho:+.3f}  p={p:.4g}  n={len(x)}{flag}")

# Final summary: which signal is actually present?
print("\n=== Diagnosis Summary ===")
strongest = sig_df.iloc[sig_df["spearman_rho"].abs().idxmax()]
print(f"Strongest mech var ↔ miscov signal (full pool): "
      f"{strongest['var']} vs {strongest['target']} rho={strongest['spearman_rho']:+.3f}")

# Check whether direct attribute is the actually strong correlation
arid_rho, _ = stats.spearmanr(df["aridity"].values, df["cov_err_global_signed"].values)
fs_rho, _ = stats.spearmanr(df["frac_snow"].values, df["cov_err_global_signed"].values)
print(f"Aridity vs signed_cov_err:    rho={arid_rho:+.3f}")
print(f"frac_snow vs signed_cov_err:  rho={fs_rho:+.3f}")

# Check per-tier strongest signals
top_per_tier = per_tier_df.iloc[per_tier_df["spearman_rho"].abs().idxmax()]
print(f"Strongest per-tier signal: {top_per_tier['tier']}/{top_per_tier['var']} vs "
      f"{top_per_tier['target']} rho={top_per_tier['spearman_rho']:+.3f}")

# Save diagnosis json
diag = {
    "g1_original_failed": True,
    "ar1_distribution": {
        "min": float(df["AR1"].min()), "median": float(df["AR1"].median()),
        "max": float(df["AR1"].max()), "iqr": [float(df["AR1"].quantile(0.25)),
                                                  float(df["AR1"].quantile(0.75))],
    },
    "het_distribution": {
        "min": float(df["Het"].min()), "median": float(df["Het"].median()),
        "max": float(df["Het"].max()),
        "n_finite": int(df["Het"].notna().sum()),
    },
    "strongest_mechvar_pool": {
        "var": str(strongest["var"]), "target": str(strongest["target"]),
        "rho": float(strongest["spearman_rho"]), "p": float(strongest["p"]),
    },
    "aridity_signed_cov_err_rho": float(arid_rho),
    "frac_snow_signed_cov_err_rho": float(fs_rho),
    "strongest_per_tier": {
        "tier": str(top_per_tier["tier"]), "var": str(top_per_tier["var"]),
        "target": str(top_per_tier["target"]), "rho": float(top_per_tier["spearman_rho"]),
    },
    "g1_passed_with_signed_cov": bool(sig_df.loc[sig_df["target"] == "cov_err_global_signed", "spearman_rho"].abs().max() >= 0.4),
    "interpretation": "see ABLATION_REPORT for narrative",
}
with open(OUT / "g1_diagnosis.json", "w") as f:
    json.dump(diag, f, indent=2)
print(f"\nwrote: {OUT/'g1_diagnosis.json'} / spearman_signed_full.csv / spearman_per_tier.csv")
