# speed_MultiReaderReporter/add_delta_features_cu_c2.py
"""Add the mean_/var_ delta-curve statistics to the LWHK C/2 discharge data.

LWHK ONLY. The C/2 dataset comes from rul_eka_CU inside the eka_test_tkammer_*
files, which exist only for the LWHK cells - there is no JGNE equivalent and this
must not be pointed at JGNE data.

extract_lwhk_cu_discharge.py stores the curves per checkup but not the
mean_d_dqdv_* / var_d_dqdv_* statistics, because those are defined ACROSS
checkups: for each checkup i, take (y_i - y_baseline) inside a voltage window and
report its mean and variance. This script computes them with the pipeline's own
core.capacity.window_delta_mean_var, so the definition matches cell_feature/
exactly, and writes the columns back into cell_feature_cu_c2/<cell>.csv.

Windows come from the LWHK voltage config (low 2.0, lowm 3.35, highm 3.45,
high 3.6):
    _l_d   low  .. lowm     2.00 - 3.35 V
    _m_d   lowm .. highm    3.35 - 3.45 V
    _h_d   highm.. high     3.45 - 3.60 V
    dQ_d   low  .. high     2.00 - 3.60 V

NOTE ON THE HIGH WINDOW. This discharge starts around 3.466 V, so the _h_d window
(3.45-3.60 V) is covered only from 3.45 to ~3.466 - a sliver. Those columns are
computed for completeness but rest on very few points; prefer _m_d and _l_d.

Usage:
    python speed_MultiReaderReporter/add_delta_features_cu_c2.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE, HERE / "core", HERE / "loaders", HERE / "utils"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.capacity import window_delta_mean_var        # noqa: E402

DEFAULT_OUT_DIR = Path("E:/download/jgne/out_jgne")
DEFAULT_SUBDIR = "cell_feature_cu_c2"

VOLT = {"high": 3.6, "highm": 3.45, "lowm": 3.35, "low": 2.0}

# (x_col, y_col, window_lo, window_hi, mean_name, var_name)
SPECS = [
    ("Vdis", "dQdVdis", VOLT["lowm"], VOLT["highm"], "mean_d_dqdv_m_d", "var_d_dqdv_m_d"),
    ("Vdis", "dQdVdis", VOLT["low"], VOLT["lowm"], "mean_d_dqdv_l_d", "var_d_dqdv_l_d"),
    ("Vdis", "dQdVdis", VOLT["highm"], VOLT["high"], "mean_d_dqdv_h_d", "var_d_dqdv_h_d"),
    ("Vdis", "Q_intVdis", VOLT["low"], VOLT["high"], "mean_dQ_d", "var_dQ_d"),
    ("Vdis", "dTdV", VOLT["lowm"], VOLT["highm"], "mean_dqdv_mt", "var_d_dqdv_mt"),
    ("Vdis", "dTdV", VOLT["low"], VOLT["lowm"], "mean_dqdv_lt", "var_d_dqdv_lt"),
    ("Vdis", "dTdV", VOLT["highm"], VOLT["high"], "mean_dqdv_ht", "var_d_dqdv_ht"),
    ("Vdis", "T_intV", VOLT["low"], VOLT["high"], "mean_d_Qt", "var_d_Qt"),
]

def parse_array(value) -> np.ndarray:
    """numpy-repr string -> array (whitespace separated, no commas)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return np.array([], dtype=float)
    out = []
    for tok in s.strip("[]").replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return np.asarray(out, dtype=float)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--subdir", default=DEFAULT_SUBDIR,
                    help=f"per-cell folder to update (default: {DEFAULT_SUBDIR})")
    args = ap.parse_args()

    folder = args.out_dir / args.subdir
    if not folder.is_dir():
        print(f"[ERROR] {folder} not found - run extract_lwhk_cu_discharge.py --per-cell first")
        return

    files = sorted(folder.glob("*.csv"))
    if any("JGNE" in f.stem.upper() for f in files):
        print("[ERROR] JGNE cells found in the C/2 folder; this analysis is LWHK-only.")
        return

    combined = []
    for path in files:
        df = pd.read_csv(path)
        # rebuild the array columns as real arrays for window_delta_mean_var
        work = pd.DataFrame(index=range(len(df)))
        needed = {c for spec in SPECS for c in spec[:2]}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"  [skip] {path.stem}: missing {missing}")
            continue
        for col in needed:
            work[col] = [parse_array(v) for v in df[col]]

        added = 0
        for x_col, y_col, lo, hi, mean_name, var_name in SPECS:
            try:
                mean, var = window_delta_mean_var(work, x_col=x_col, y_col=y_col,
                                                  x_lo=lo, x_hi=hi)
                df[mean_name], df[var_name] = mean, var
                added += 2
            except Exception as e:
                print(f"  [warn] {path.stem} {mean_name}: {e}")
        df.to_csv(path, index=False)
        combined.append(df)
        print(f"  [ok]   {path.stem:24} {len(df):2} checkup(s), +{added} column(s)")

    if combined:
        allrows = pd.concat(combined, ignore_index=True)
        out = args.out_dir / "lwhk_cu_discharge_features.csv"
        allrows.to_csv(out, index=False)
        print(f"\n[OK] {len(allrows)} row(s) across {allrows.cell.nunique()} cell(s) -> {out}")
        cols = ["cell", "CU_time", "SOH_c2", "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
                "mean_d_dqdv_l_d", "var_d_dqdv_l_d"]
        print()
        print(allrows[[c for c in cols if c in allrows.columns]].head(12).to_string(index=False))

if __name__ == "__main__":
    main()
