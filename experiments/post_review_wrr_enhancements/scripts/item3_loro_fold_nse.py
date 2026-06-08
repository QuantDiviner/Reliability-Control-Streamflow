"""
Post-review item 3: per-fold held-out test NSE distribution for the 18 HUC-2 LORO folds.

Reviewer concern (20250528 报告2, #3): the LORO claim retrains 18 LSTMs, so the
reviewer wants visible evidence of point-prediction skill per fold. If some folds have
poor point skill, their coverage failure would conflate "point-skill transfer failure"
with "calibration transfer failure". NeuralHydrology already wrote per-basin test NSE
for every fold (test/model_epoch030/test_metrics.csv), so this is a pure read+aggregate
of existing outputs (no retraining, no GPU).

Fold selection uses the anchor run dir per HUC, matching
paper/scripts/extract_figure_tables.py and the manuscript transfer heatmap. If
NeuralHydrology did not retain test_metrics.csv for an anchor fold, NSE is computed
directly from test_results.p.

Outputs:
  paper/data/tables/loro_fold_nse.csv      (one row per fold: huc, run_dir, n_basins, NSE stats)
  experiments/post_review_wrr_enhancements/results/item3_loro_fold_nse.json (summary for SSOT)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments" / "exp003"))
from hscc_analysis_loro import extract_basin_series  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = ROOT / "experiments" / "exp003" / "results"
OUT_CSV = ROOT / "paper" / "data" / "tables" / "loro_fold_nse.csv"
OUT_JSON = ROOT / "experiments" / "post_review_wrr_enhancements" / "results" / "item3_loro_fold_nse.json"


def find_fold_dir(huc: str) -> Path | None:
    """Return the anchor LORO fold used by the manuscript transfer heatmap."""
    anchor_candidates = sorted(
        c for c in RESULTS_ROOT.glob(f"exp003_loro_huc{huc}_*")
        if "_seed" not in c.name
    )
    fallback_candidates = sorted(RESULTS_ROOT.glob(f"exp003_loro_huc{huc}_*"))
    for c in reversed(anchor_candidates or fallback_candidates):
        if (c / "_analysis" / "metrics.json").exists():
            return c
    return None


def _nse(obs: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)


def load_fold_nse(fold_dir: Path) -> np.ndarray | None:
    mcsv = fold_dir / "test" / "model_epoch030" / "test_metrics.csv"
    if mcsv.exists():
        df = pd.read_csv(mcsv)
        return df["NSE"].astype(float).values

    results_p = fold_dir / "test" / "model_epoch030" / "test_results.p"
    if not results_p.exists():
        return None

    import pickle

    with open(results_p, "rb") as f:
        results = pickle.load(f)
    vals = []
    for basin_id in results:
        try:
            obs, pred = extract_basin_series(results, basin_id)
        except Exception:
            continue
        vals.append(_nse(obs, pred))
    return np.asarray(vals, dtype=float)


def main() -> None:
    rows = []
    for k in range(1, 19):
        huc = f"{k:02d}"
        fold_dir = find_fold_dir(huc)
        if fold_dir is None:
            print(f"HUC-{huc}: no fold dir found")
            continue
        nse = load_fold_nse(fold_dir)
        if nse is None:
            print(f"HUC-{huc}: missing retained test metrics/results")
            continue
        nse = nse[np.isfinite(nse)]
        rows.append({
            "huc": huc,
            "run_dir": fold_dir.name,
            "n_basins": int(nse.size),
            "median_nse": float(np.median(nse)),
            "mean_nse": float(np.mean(nse)),
            "p25_nse": float(np.percentile(nse, 25)),
            "p75_nse": float(np.percentile(nse, 75)),
            "min_nse": float(np.min(nse)),
            "max_nse": float(np.max(nse)),
            "frac_nse_below_0": float(np.mean(nse < 0.0)),
            "frac_nse_below_0p3": float(np.mean(nse < 0.3)),
        })
        print(f"HUC-{huc}: n={nse.size:3d}  median NSE={np.median(nse):.3f}  "
              f"[{np.percentile(nse,25):.3f}, {np.percentile(nse,75):.3f}]  "
              f"min={np.min(nse):.3f}  ({fold_dir.name})")

    fold_df = pd.DataFrame(rows).sort_values("huc").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(OUT_CSV, index=False)

    fold_medians = fold_df["median_nse"].values
    summary = {
        "n_folds": int(len(fold_df)),
        "fold_median_nse_min": float(fold_medians.min()),
        "fold_median_nse_max": float(fold_medians.max()),
        "fold_median_nse_median": float(np.median(fold_medians)),
        "fold_median_nse_mean": float(fold_medians.mean()),
        "n_folds_median_nse_below_0p3": int(np.sum(fold_medians < 0.3)),
        "n_folds_median_nse_below_0": int(np.sum(fold_medians < 0.0)),
        "worst_fold_huc": fold_df.loc[fold_df["median_nse"].idxmin(), "huc"],
        "worst_fold_median_nse": float(fold_medians.min()),
        "best_fold_huc": fold_df.loc[fold_df["median_nse"].idxmax(), "huc"],
        "best_fold_median_nse": float(fold_medians.max()),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote: {OUT_CSV}")
    print(f"wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
