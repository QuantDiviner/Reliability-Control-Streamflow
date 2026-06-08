"""
Post-review item 4: collect the larger-backbone robustness summary into a single JSON
for the SSOT (collect_results.py reads it like the item1/3/8 summaries).

Reviewer concern (20250528 报告2, #4): hidden=64/30ep is small/short and under-training
could inflate residual variance and amplify the tier spread, so the conditional-coverage
failure might be a weak-model artifact. This collects the hidden=256/50ep single-seed
re-run on the SAME CAMELS-US seed-42 temporal split so the manuscript can state whether
the failure persists on a 4x-wider, longer-trained backbone.

Output:
  experiments/post_review_wrr_enhancements/results/item4_larger_backbone_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = (ROOT / "experiments/post_review_wrr_enhancements/item4_larger_backbone/results/"
       "item4_larger_backbone_h256_2805_213024")
OUT = ROOT / "experiments/post_review_wrr_enhancements/results/item4_larger_backbone_summary.json"


def main() -> None:
    m = json.loads((RUN / "_analysis" / "metrics.json").read_text())
    nse = pd.read_csv(RUN / "test" / "model_epoch050" / "test_metrics.csv")["NSE"].astype(float).values
    nse = nse[np.isfinite(nse)]
    pt = m["per_tier"]
    out = {
        "config": "cudalstm hidden=256, 50 epochs, seed=42, CAMELS-US temporal split (671 basins)",
        "hidden_size": 256,
        "epochs": 50,
        "seed": 42,
        "median_test_nse": float(np.median(nse)),
        "mean_test_nse": float(np.mean(nse)),
        "global_spread_pp": float(m["tier_coverage_spread_global_pp"]),
        "hscc_spread_pp": float(m["tier_coverage_spread_hscc_pp"]),
        "global_dry_coverage": float(pt["dry"]["global_coverage"]),
        "global_semi_arid_coverage": float(pt["semi_arid"]["global_coverage"]),
        "global_humid_coverage": float(pt["humid"]["global_coverage"]),
        "global_snow_coverage": float(pt["snow"]["global_coverage"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote: {OUT}")


if __name__ == "__main__":
    main()
