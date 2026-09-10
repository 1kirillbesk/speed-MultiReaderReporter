# speed_MultiReaderReporter/extract_curve_features.py
"""Pull the stored peak points and delta-dQ/dV statistics out of cell_feature/.

Reads every cell_feature/<cell>.csv written by the pipeline and emits two tidy
tables:

  curve_features_scalars.csv
      one row per (cell, checkup): CU_time, weeks, capacities, throughput and
      every mean_/var_ statistic already computed by the pipeline.

  curve_features_peaks.csv
      one row per (cell, checkup, curve, direction, kind, peak): the peak
      coordinates unpacked from the peak* array columns.
          curve      ICA (dQ/dV over V) | DVA (dV/dQ over Q)
          direction  cha | dis
          kind       max | min
          x, y       peak position and height in that curve's own units

Note on the mean_/var_ columns: the pipeline computes them as a difference
against the FIRST checkup of each cell (baseline_idx=0 in
core.capacity.window_delta_mean_var), so checkup_index 0 is always exactly 0.0
by construction, not because nothing changed.

Usage:
    python speed_MultiReaderReporter/extract_curve_features.py
    python speed_MultiReaderReporter/extract_curve_features.py --cells "SPEED_LWHK_*"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path("E:/download/jgne/out_jgne")

# (curve, direction, kind, x_column, y_column)
PEAK_SPECS = [
    ("ICA", "cha", "max", "peakV_max_cha", "peakICA_max_cha"),
    ("ICA", "cha", "min", "peakV_min_cha", "peakICA_min_cha"),
    ("ICA", "dis", "max", "peakV_max_dis", "peakICA_max_dis"),
    ("ICA", "dis", "min", "peakV_min_dis", "peakICA_min_dis"),
    ("DVA", "cha", "max", "peakQ_max_cha", "peakDVA_max_cha"),
    ("DVA", "cha", "min", "peakQ_min_cha", "peakDVA_min_cha"),
    ("DVA", "dis", "max", "peakQ_max_dis", "peakDVA_max_dis"),
    ("DVA", "dis", "min", "peakQ_min_dis", "peakDVA_min_dis"),
]

# Curve arrays that stay in the CSV; excluded from the scalar table.
CURVE_COLS = ["Vcha", "dQdVcha", "Q_intVcha"]

def parse_array(value) -> np.ndarray:
    """Parse a numpy-repr string ('[1.0 2.0 ...]') back into an array.

    The pipeline writes arrays via DataFrame.to_csv, so they are whitespace
    separated with no commas - json.loads and ast.literal_eval both fail on them.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return np.array([], dtype=float)
    if "..." in s:
        raise ValueError("array was truncated by numpy print options when saved")
    out = []
    for tok in s.strip("[]").replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return np.asarray(out, dtype=float)

def extract_cell(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = csv_path.stem
    df = pd.read_csv(csv_path)

    peak_cols = [c for spec in PEAK_SPECS for c in spec[3:]]
    scalar_cols = [c for c in df.columns if c not in peak_cols + CURVE_COLS]

    scalars = df[scalar_cols].copy()
    scalars.insert(0, "cell", cell)
    scalars.insert(1, "checkup_index", range(len(df)))
    if "CU_time" in scalars.columns:
        t = pd.to_datetime(scalars["CU_time"], errors="coerce")
        scalars.insert(3, "weeks", (t - t.iloc[0]).dt.total_seconds() / (7 * 24 * 3600))

    cu_time = df["CU_time"] if "CU_time" in df.columns else pd.Series([None] * len(df))
    peak_rows = []
    for i in range(len(df)):
        for curve, direction, kind, x_col, y_col in PEAK_SPECS:
            if x_col not in df.columns or y_col not in df.columns:
                continue
            try:
                xs = parse_array(df[x_col].iloc[i])
                ys = parse_array(df[y_col].iloc[i])
            except ValueError as e:
                print(f"[WARN] {cell} checkup {i} {x_col}: {e}")
                continue
            n = min(len(xs), len(ys))
            for k in range(n):
                peak_rows.append({
                    "cell": cell,
                    "checkup_index": i,
                    "CU_time": cu_time.iloc[i],
                    "curve": curve,
                    "direction": direction,
                    "kind": kind,
                    "peak_index": k,
                    "x": xs[k],
                    "y": ys[k],
                })

    return scalars, pd.DataFrame(peak_rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"pipeline output root holding cell_feature/ (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--cells", default="*",
                    help="glob over cell names, e.g. \"SPEED_LWHK_*\" (default: all)")
    ap.add_argument("--dest", type=Path, default=None,
                    help="where to write the two csv files (default: --out-dir)")
    args = ap.parse_args()

    feature_dir = args.out_dir / "cell_feature"
    if not feature_dir.is_dir():
        print(f"[ERROR] no cell_feature folder under {args.out_dir}")
        return

    files = sorted(feature_dir.glob(f"{args.cells}.csv"))
    if not files:
        print(f"[INFO] no cell csv matching {args.cells!r} in {feature_dir}")
        return

    all_scalars, all_peaks = [], []
    for path in files:
        try:
            scalars, peaks = extract_cell(path)
        except Exception as e:
            print(f"[WARN] {path.stem}: {e}")
            continue
        all_scalars.append(scalars)
        if not peaks.empty:
            all_peaks.append(peaks)
        print(f"  [read] {path.stem:24} {len(scalars):3} checkup(s), {len(peaks):5} peak point(s)")

    dest = args.dest or args.out_dir
    dest.mkdir(parents=True, exist_ok=True)

    scalars = pd.concat(all_scalars, ignore_index=True) if all_scalars else pd.DataFrame()
    peaks = pd.concat(all_peaks, ignore_index=True) if all_peaks else pd.DataFrame()

    scalars_path = dest / "curve_features_scalars.csv"
    peaks_path = dest / "curve_features_peaks.csv"
    scalars.to_csv(scalars_path, index=False)
    peaks.to_csv(peaks_path, index=False)

    print()
    print(f"[OK] {len(scalars):5} row(s) -> {scalars_path}")
    print(f"[OK] {len(peaks):5} row(s) -> {peaks_path}")
    if not peaks.empty:
        print()
        print("peak points per curve/direction/kind:")
        print(peaks.groupby(["curve", "direction", "kind"]).size().to_string())

if __name__ == "__main__":
    main()
