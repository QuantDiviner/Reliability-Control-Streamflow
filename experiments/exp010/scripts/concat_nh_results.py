"""
Concat NH validation_results.p (1990-10-01 ~ 2000-09-30, calibration period for HopCPT)
with test_results.p (2000-10-01 ~ 2014-09-30, test period) along 'date' dim,
producing a single .p file consumed by HopCPT's `precomputed_nh` model_fc.

HopCPT's PrecomputedNeuralHydrologyForcast (libs/HopCPT/code/models/forcast/precomputed_nh.py)
requires a single nh_results_path whose date length == X_full - train_steps. We do NOT
provide nh_train_results_path: that path prepends predictions and zeros prediction_offset,
which would only be correct if we had train-period (1980-1990) predictions. Without them,
HopCPT sets prediction_offset = train_offset and predictions[0] must align with calibration
day 0. Concatenating val + test gives exactly that.

See experiments/exp010/execution-log.md "HopCPT 接口关键发现" §B and Task #6 alignment proof.
"""
import argparse
import pickle
import sys
from pathlib import Path

import xarray as xr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--val",
        default="experiments/exp002/results/exp002_camels_us_temporal_2504_155058/validation/model_epoch030/validation_results.p",
        help="NH validation_results.p (1990-10-01 ~ 2000-09-30)",
    )
    ap.add_argument(
        "--test",
        default="experiments/exp002/results/exp002_camels_us_temporal_2504_155058/test/model_epoch030/test_results.p",
        help="NH test_results.p (2000-10-01 ~ 2014-09-30)",
    )
    ap.add_argument(
        "--out",
        default="experiments/exp010/cal_test_results.p",
        help="Output: concatenated calibration+test predictions",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    val_p = (repo_root / args.val).resolve()
    test_p = (repo_root / args.test).resolve()
    out_p = (repo_root / args.out).resolve()

    for p in (val_p, test_p):
        if not p.is_file():
            sys.exit(f"ERROR: missing {p}")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"[concat] loading val: {val_p}")
    with val_p.open("rb") as f:
        val = pickle.load(f)
    print(f"[concat] loading test: {test_p}")
    with test_p.open("rb") as f:
        test = pickle.load(f)

    val_basins = set(val.keys())
    test_basins = set(test.keys())
    if val_basins != test_basins:
        only_val = val_basins - test_basins
        only_test = test_basins - val_basins
        sys.exit(f"ERROR: basin mismatch — only_val={len(only_val)} only_test={len(only_test)}")
    print(f"[concat] basin count: {len(val_basins)}")

    out = {}
    expected_len = None
    for basin in sorted(val_basins):
        val_xr = val[basin]["1D"]["xr"]
        test_xr = test[basin]["1D"]["xr"]

        v_end = val_xr.date.values[-1]
        t_start = test_xr.date.values[0]
        gap_days = (t_start - v_end).astype("timedelta64[D]").astype(int)
        if gap_days != 1:
            sys.exit(
                f"ERROR: basin {basin} val end {v_end} → test start {t_start} gap={gap_days} days "
                f"(expected 1 day for contiguous concat)"
            )

        cat = xr.concat([val_xr, test_xr], dim="date")
        n = cat.sizes["date"]
        v_n = val_xr.sizes["date"]
        t_n = test_xr.sizes["date"]
        assert n == v_n + t_n, f"basin {basin}: concat len {n} != {v_n}+{t_n}"

        # NaN check on the variable HopCPT will consume
        sim = cat["QObs(mm/d)_sim"].sel(time_step=0).values
        n_nan = int((sim != sim).sum())
        if n_nan > 0:
            sys.exit(f"ERROR: basin {basin} has {n_nan} NaN(s) in QObs(mm/d)_sim — HopCPT requires NaN-free")

        if expected_len is None:
            expected_len = n
        elif n != expected_len:
            sys.exit(f"ERROR: basin {basin} length {n} != expected {expected_len} (uneven across basins)")

        out[basin] = {"1D": {"xr": cat}}

    print(f"[concat] all 671 basins — date count {expected_len} per basin "
          f"(val 3653 + test 5113 = 8766 expected)")

    print(f"[concat] writing: {out_p}")
    with out_p.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_p.stat().st_size / (1024 * 1024)
    print(f"[concat] done — output size {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
