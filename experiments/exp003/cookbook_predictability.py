"""
exp003 A3 — Cookbook predictability study (R3-mandated design).

For C3 PUB cookbook to be "actionable", the expected reliability class of a
(HUC, tier) cell must be predictable from the cell's static attributes alone
(without observing test coverage). This script tests that hypothesis under
the strict design dictated by R3 (D-R3-exp003 codex):

  - Group leave-one-HUC-out (NOT cell-level random CV). The cookbook deploys
    on unseen HUCs, so train/test split must mirror that.
  - Compare against three baselines: majority-class, tier-only, geographic
    1-NN-HUC. Cookbook only counts as actionable if it outperforms ALL
    three baselines.
  - Report per-class precision/recall + confusion matrix + bootstrap CI.
  - Hard gate: cross-HUC accuracy ≥ 0.60 AND statistically significantly
    better than the strongest baseline (paired bootstrap p < 0.05).

Inputs:
  experiments/exp003/transfer_matrix.csv (40 cells, 8 low-power flagged)
  data/raw/CAMELS_US/camels_attributes_v2.0/camels_{clim,topo}.txt
  experiments/exp003/basin_lists/{test_huc{01..18}.txt}

Outputs:
  experiments/exp003/cookbook_predictability.csv
    one row per cell with: huc, tier, true_class, predicted_class
    (RandomForest), held-out fold's HUC, plus baseline predictions.
  experiments/exp003/cookbook_predictability_summary.json
    overall accuracy + per-class metrics + baseline accuracies +
    bootstrap CI + statistical test result (R3 gate).

Usage:
  python experiments/exp003/cookbook_predictability.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
TRANSFER = ROOT / "experiments" / "exp003" / "transfer_matrix.csv"
CAMELS = ROOT / "data" / "raw" / "CAMELS_US" / "camels_attributes_v2.0"
BASIN_LISTS = ROOT / "experiments" / "exp003" / "basin_lists"
OUT_CSV = ROOT / "experiments" / "exp003" / "cookbook_predictability.csv"
OUT_JSON = ROOT / "experiments" / "exp003" / "cookbook_predictability_summary.json"

# Class boundaries (R3-aligned with deployment fallback rule D-024)
RELIABLE_THRESHOLD = 0.85   # HSCC ≥ 0.85 → "Reliable"
MARGINAL_THRESHOLD = 0.70   # 0.70 ≤ HSCC < 0.85 → "Marginal"; <0.70 → "Unreliable"

CLASSES = ["Reliable", "Marginal", "Unreliable"]
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)


def label_class(hscc_cov: float) -> str:
    if hscc_cov >= RELIABLE_THRESHOLD:
        return "Reliable"
    if hscc_cov >= MARGINAL_THRESHOLD:
        return "Marginal"
    return "Unreliable"


def load_features() -> pd.DataFrame:
    clim = pd.read_csv(CAMELS / "camels_clim.txt", sep=";", dtype={"gauge_id": str})
    topo = pd.read_csv(CAMELS / "camels_topo.txt", sep=";", dtype={"gauge_id": str})
    df = clim.merge(topo, on="gauge_id", how="inner")
    df["gauge_id"] = df["gauge_id"].str.zfill(8)
    return df


def build_huc_basin_map() -> dict[str, str]:
    """basin_id → huc_id ('01'..'18') from test_huc{XX}.txt files."""
    mapping: dict[str, str] = {}
    for k in range(1, 19):
        huc = f"{k:02d}"
        path = BASIN_LISTS / f"test_huc{huc}.txt"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                mapping[line.zfill(8)] = huc
    return mapping


def build_cell_features(transfer: pd.DataFrame, attrs: pd.DataFrame,
                        basin_huc: dict[str, str]) -> pd.DataFrame:
    """For each (huc, tier) cell, compute mean static attributes of test basins."""
    # Load tier assignments — recompute from camels attrs to be deterministic.
    # tier scheme matches PROJECT_CHARTER §2.2 (snow takes priority over aridity).
    def assign_tier(row):
        if row["frac_snow"] >= 0.4:
            return "snow"
        if row["aridity"] > 1.5:
            return "dry"
        if row["aridity"] > 1.0:
            return "semi_arid"
        return "humid"

    attrs = attrs.copy()
    attrs["tier"] = attrs.apply(assign_tier, axis=1)
    attrs["huc"] = attrs["gauge_id"].map(basin_huc)
    attrs = attrs.dropna(subset=["huc"])

    feature_cols = ["aridity", "frac_snow", "p_mean", "p_seasonality",
                    "high_prec_freq", "low_prec_freq",
                    "elev_mean", "slope_mean", "area_gages2",
                    "gauge_lat", "gauge_lon"]

    rows = []
    for _, cell in transfer.iterrows():
        huc, tier = cell["huc"], cell["tier"]
        sub = attrs[(attrs["huc"] == huc) & (attrs["tier"] == tier)]
        if sub.empty:
            continue
        feat = {f"{c}_mean": float(sub[c].mean()) for c in feature_cols}
        feat.update({
            "huc": huc,
            "tier": tier,
            "n_test_basins": cell["n_test_basins"],
            "low_power": cell["low_power"],
            "hscc_coverage": cell["hscc_coverage"],
            "global_coverage": cell["global_coverage"],
            "true_class": label_class(cell["hscc_coverage"]),
        })
        rows.append(feat)
    return pd.DataFrame(rows)


def baseline_majority(y_train: np.ndarray, n_test: int) -> np.ndarray:
    cls, counts = np.unique(y_train, return_counts=True)
    majority = cls[np.argmax(counts)]
    return np.full(n_test, majority)


def baseline_tier_only(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Predict each test cell's class as the modal class for its tier in training."""
    tier_mode: dict[str, str] = {}
    for t in CLASSES:
        pass  # placeholder
    for tier in train_df["tier"].unique():
        sub = train_df[train_df["tier"] == tier]
        if sub.empty:
            continue
        cls, counts = np.unique(sub["true_class"], return_counts=True)
        tier_mode[tier] = cls[np.argmax(counts)]
    # Fallback: global majority if a tier never appeared in training
    cls, counts = np.unique(train_df["true_class"], return_counts=True)
    fallback = cls[np.argmax(counts)]
    return np.array([tier_mode.get(t, fallback) for t in test_df["tier"]])


def baseline_nearest_huc(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Predict each test cell's class as the class of the geographically
    nearest training cell (same tier preferred; otherwise nearest tier-agnostic).
    Distance: Euclidean on (lat, lon) of cell-mean."""
    preds = []
    for _, test_cell in test_df.iterrows():
        # Prefer same-tier neighbours
        same = train_df[train_df["tier"] == test_cell["tier"]]
        candidate = same if not same.empty else train_df
        d2 = ((candidate["gauge_lat_mean"] - test_cell["gauge_lat_mean"]) ** 2
              + (candidate["gauge_lon_mean"] - test_cell["gauge_lon_mean"]) ** 2)
        idx = d2.idxmin()
        preds.append(candidate.loc[idx, "true_class"])
    return np.array(preds)


def paired_bootstrap_pvalue(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray,
                            n_boot: int = N_BOOTSTRAP) -> float:
    """One-sided paired bootstrap test: H0: acc_a ≤ acc_b, H1: acc_a > acc_b."""
    correct_a = (y_a == y_true).astype(int)
    correct_b = (y_b == y_true).astype(int)
    diff = correct_a - correct_b
    obs = diff.mean()
    null = []
    for _ in range(n_boot):
        sign = RNG.choice([-1, 1], size=len(diff))
        null.append((diff * sign).mean())
    null = np.array(null)
    p = float((null >= obs).mean())
    return p


def bootstrap_acc_ci(y_true: np.ndarray, y_pred: np.ndarray,
                     n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple[float, float]:
    correct = (y_pred == y_true).astype(int)
    n = len(correct)
    boots = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boots.append(correct[idx].mean())
    boots = np.array(boots)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main() -> None:
    transfer = pd.read_csv(TRANSFER, dtype={"huc": str})
    transfer["huc"] = transfer["huc"].str.zfill(2)
    print(f"loaded {len(transfer)} cells, of which {transfer['low_power'].sum()} low-power")

    attrs = load_features()
    basin_huc = build_huc_basin_map()
    print(f"loaded {len(attrs)} basins, mapped {len(basin_huc)} to 18 HUCs")

    df = build_cell_features(transfer, attrs, basin_huc)
    print(f"feature matrix: {len(df)} cells (before low-power filter)")

    # R3 mandate: cookbook labels must use n_test ≥ TIER_MIN. Filter low-power.
    df_full = df[~df["low_power"]].reset_index(drop=True)
    print(f"feature matrix: {len(df_full)} cells (high-power, n_test ≥ 3)")

    # Distribution check
    cls_counts = df_full["true_class"].value_counts()
    print(f"class distribution: {cls_counts.to_dict()}")

    feature_cols = [c for c in df_full.columns if c.endswith("_mean")
                    and c not in {"gauge_lat_mean", "gauge_lon_mean"}]
    X = df_full[feature_cols].values
    y = df_full["true_class"].values
    groups = df_full["huc"].values

    # Group LOHO CV
    logo = LeaveOneGroupOut()
    rf_preds = np.empty(len(y), dtype=object)
    lr_preds = np.empty(len(y), dtype=object)
    maj_preds = np.empty(len(y), dtype=object)
    tier_preds = np.empty(len(y), dtype=object)
    nn_preds = np.empty(len(y), dtype=object)
    held_out_huc = np.empty(len(y), dtype=object)

    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        rf = RandomForestClassifier(n_estimators=200, max_depth=4,
                                    class_weight="balanced", random_state=42).fit(Xtr, y[train_idx])
        lr = LogisticRegression(max_iter=2000, multi_class="auto",
                                class_weight="balanced", random_state=42).fit(Xtr, y[train_idx])
        rf_preds[test_idx] = rf.predict(Xte)
        lr_preds[test_idx] = lr.predict(Xte)
        maj_preds[test_idx] = baseline_majority(y[train_idx], len(test_idx))
        tier_preds[test_idx] = baseline_tier_only(df_full.iloc[train_idx], df_full.iloc[test_idx])
        nn_preds[test_idx] = baseline_nearest_huc(df_full.iloc[train_idx], df_full.iloc[test_idx])
        held_out_huc[test_idx] = groups[test_idx]

    # Overall accuracies
    accs = {
        "rf": accuracy_score(y, rf_preds),
        "lr": accuracy_score(y, lr_preds),
        "majority": accuracy_score(y, maj_preds),
        "tier_only": accuracy_score(y, tier_preds),
        "nearest_huc": accuracy_score(y, nn_preds),
    }
    cis = {k: bootstrap_acc_ci(y, v) for k, v in {
        "rf": rf_preds, "lr": lr_preds, "majority": maj_preds,
        "tier_only": tier_preds, "nearest_huc": nn_preds,
    }.items()}

    print("\nOverall LOHO accuracies (95% bootstrap CI):")
    for k, a in accs.items():
        lo, hi = cis[k]
        print(f"  {k:14s}: {a:.3f}  [{lo:.3f}, {hi:.3f}]")

    # Pick best model = max(rf, lr); strongest baseline = max(majority, tier, nn)
    best_model_name = "rf" if accs["rf"] >= accs["lr"] else "lr"
    best_model_preds = rf_preds if best_model_name == "rf" else lr_preds
    baseline_accs = {k: accs[k] for k in ["majority", "tier_only", "nearest_huc"]}
    strongest_baseline_name = max(baseline_accs, key=baseline_accs.get)
    strongest_baseline_preds = {
        "majority": maj_preds, "tier_only": tier_preds, "nearest_huc": nn_preds
    }[strongest_baseline_name]

    p = paired_bootstrap_pvalue(y, best_model_preds, strongest_baseline_preds)

    # R3 hard gate
    gate_acc = accs[best_model_name] >= 0.60
    gate_pvalue = p < 0.05
    gate_pass = gate_acc and gate_pvalue
    if gate_pass:
        verdict = "PASS"
    elif accs[best_model_name] >= 0.50:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    print(f"\nR3 gate: best model = {best_model_name} ({accs[best_model_name]:.3f})")
    print(f"         strongest baseline = {strongest_baseline_name} ({accs[strongest_baseline_name]:.3f})")
    print(f"         paired bootstrap p = {p:.4f} (need < 0.05)")
    print(f"         accuracy ≥ 0.60: {gate_acc}")
    print(f"         significantly > baseline: {gate_pvalue}")
    print(f"         VERDICT: {verdict}")

    # Confusion matrices
    cm_rf = confusion_matrix(y, rf_preds, labels=CLASSES).tolist()

    # Per-class metrics
    def per_class(yt, yp):
        out = {}
        for c in CLASSES:
            tp = int(((yp == c) & (yt == c)).sum())
            fp = int(((yp == c) & (yt != c)).sum())
            fn = int(((yp != c) & (yt == c)).sum())
            prec = tp / (tp + fp) if (tp + fp) else None
            rec = tp / (tp + fn) if (tp + fn) else None
            out[c] = {"tp": tp, "fp": fp, "fn": fn,
                      "precision": prec, "recall": rec}
        return out

    # Save per-cell predictions
    out_df = df_full[["huc", "tier", "n_test_basins", "hscc_coverage", "true_class"]].copy()
    out_df["rf_pred"] = rf_preds
    out_df["lr_pred"] = lr_preds
    out_df["majority_pred"] = maj_preds
    out_df["tier_only_pred"] = tier_preds
    out_df["nearest_huc_pred"] = nn_preds
    out_df["held_out_huc"] = held_out_huc
    out_df.to_csv(OUT_CSV, index=False)

    summary = {
        "n_cells_used": int(len(df_full)),
        "n_low_power_excluded": int((~transfer["low_power"]).sum()),
        "class_distribution": cls_counts.to_dict(),
        "feature_columns": feature_cols,
        "accuracies_loho": accs,
        "bootstrap_ci_95": cis,
        "best_model": best_model_name,
        "strongest_baseline": strongest_baseline_name,
        "paired_bootstrap_pvalue": p,
        "r3_gate": {
            "accuracy_threshold_pass": bool(gate_acc),
            "significance_threshold_pass": bool(gate_pvalue),
            "overall_pass": bool(gate_pass),
            "verdict": verdict,
        },
        "confusion_matrix_rf": {
            "labels": CLASSES,
            "matrix_rows_true_cols_pred": cm_rf,
        },
        "per_class_metrics_rf": per_class(y, rf_preds),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
