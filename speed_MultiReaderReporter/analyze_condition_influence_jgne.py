# speed_MultiReaderReporter/analyze_condition_influence_jgne.py
"""Influence of the experimental conditions on the features, at a common SoH.

For every JGNE cell:
  1. SOH = cap_ocv_dis / cap_ocv_dis[0]   (same definition as analyze3dim.py)
  2. every feature is interpolated onto the requested SOH level
  3. the result is joined with the condition table from document_conditions.py
  4. each condition is correlated against each feature across cells

Interpolation only, never extrapolation: a cell whose measured SOH range does
not contain the target is skipped and listed with the reason. This matters here,
because only some cells age far enough - run with --report-only to see the
reachability table before committing to a level.

Usage:
    python speed_MultiReaderReporter/analyze_condition_influence_jgne.py
    python speed_MultiReaderReporter/analyze_condition_influence_jgne.py --target-soh 0.99
    python speed_MultiReaderReporter/analyze_condition_influence_jgne.py --report-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path("E:/download/jgne/out_jgne")
DEFAULT_TARGET_SOH = 0.98
SOH_COL = "cap_ocv_dis"

# Columns that are bookkeeping rather than features. The pipeline copies the
# parsed experimental condition into cell_feature/ too (soc_start, pause, ...);
# those are predictors, not responses, so they must not be correlated against
# themselves.
NON_FEATURE = {"CU_time", "cell", "checkup_index",
               "soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "dyn", "pause"}
# Array columns the pipeline leaves in the csv.
ARRAY_COLS = {"Vcha", "dQdVcha", "Q_intVcha",
              "peakV_max_cha", "peakICA_max_cha", "peakQ_max_cha", "peakDVA_max_cha",
              "peakV_min_cha", "peakICA_min_cha", "peakQ_min_cha", "peakDVA_min_cha",
              "peakV_max_dis", "peakICA_max_dis", "peakQ_max_dis", "peakDVA_max_dis",
              "peakV_min_dis", "peakICA_min_dis", "peakQ_min_dis", "peakDVA_min_dis"}

JGNE_CONDITIONS = ["soc_start", "soc_end", "pause_h", "has_pulse", "low_soc", "temp_C", "c_rate_chg", "c_rate_dchg"]

# ---------------------------------------------------------------- shared core

def load_cell(csv_path: Path, soh_col: str = SOH_COL) -> pd.DataFrame | None:
    """Feature table for one cell, with SOH and weeks added."""
    df = pd.read_csv(csv_path)
    if soh_col not in df.columns or df.empty:
        return None
    df = df.drop(columns=[c for c in ARRAY_COLS if c in df.columns], errors="ignore")
    df["SOH"] = df[soh_col] / df[soh_col].iloc[0]
    if "CU_time" in df.columns:
        t = pd.to_datetime(df["CU_time"], errors="coerce")
        df["weeks"] = (t - t.iloc[0]).dt.total_seconds() / (7 * 24 * 3600)
    return df

def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in NON_FEATURE and c != "SOH"
            and pd.api.types.is_numeric_dtype(df[c])]

def interp_at_soh(df: pd.DataFrame, target: float) -> tuple[dict | None, str]:
    """Linearly interpolate every feature at `target` SOH.

    Returns (values, reason). values is None when the target is not bracketed by
    the measured SOH range - we refuse to extrapolate.
    """
    d = df.dropna(subset=["SOH"])
    if len(d) < 2:
        return None, f"only {len(d)} checkup(s)"

    lo, hi = float(d["SOH"].min()), float(d["SOH"].max())
    if not (lo <= target <= hi):
        return None, f"target outside measured SOH range [{lo:.4f}, {hi:.4f}]"

    # np.interp needs strictly increasing x
    d = d.sort_values("SOH").drop_duplicates(subset="SOH", keep="first")
    if len(d) < 2:
        return None, "SOH values not distinct"

    out = {"n_checkups": len(df), "soh_min": lo, "soh_max": hi}
    for col in feature_columns(df):
        y = pd.to_numeric(d[col], errors="coerce")
        m = y.notna()
        out[col] = float(np.interp(target, d.loc[m, "SOH"], y[m])) if m.sum() >= 2 else np.nan
    return out, "ok"

def reachability(feature_dir: Path, pattern: str, soh_col: str = SOH_COL) -> pd.DataFrame:
    rows = []
    for path in sorted(feature_dir.glob(f"{pattern}.csv")):
        df = load_cell(path, soh_col)
        if df is None:
            rows.append({"cell": path.stem, "n_checkups": 0,
                         "soh_min": np.nan, "soh_max": np.nan})
            continue
        rows.append({"cell": path.stem, "n_checkups": len(df),
                     "soh_min": float(df["SOH"].min()),
                     "soh_max": float(df["SOH"].max())})
    return pd.DataFrame(rows)

def features_at_soh(feature_dir: Path, pattern: str, target: float,
                    verbose: bool = True, soh_col: str = SOH_COL) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept, skipped = [], []
    for path in sorted(feature_dir.glob(f"{pattern}.csv")):
        df = load_cell(path, soh_col)
        if df is None:
            skipped.append({"cell": path.stem, "reason": f"no {soh_col} column"})
            continue
        values, reason = interp_at_soh(df, target)
        if values is None:
            skipped.append({"cell": path.stem, "reason": reason})
            if verbose:
                print(f"  [skip] {path.stem:24} {reason}")
            continue
        kept.append({"cell": path.stem, **values})
        if verbose:
            print(f"  [ok]   {path.stem:24} {values['n_checkups']:2} checkup(s)")
    return pd.DataFrame(kept), pd.DataFrame(skipped)

def join_conditions(features: pd.DataFrame, cond_csv: Path,
                    prefer_col: str | None) -> pd.DataFrame:
    """Attach one condition row per cell.

    A cell may appear under more than one label (a programme switch); keep the
    row carrying `prefer_col`, otherwise the one covering the most files.
    """
    if not cond_csv.exists():
        print(f"[WARN] condition table not found: {cond_csv}")
        return features
    cond = pd.read_csv(cond_csv)
    # Rank WITHIN each cell, never across cells: a global filter on prefer_col
    # would delete every cell whose only label lacks it (the _40T, _ref and
    # _1C2C cells), leaving their conditions NaN and silently excluding them.
    cond = cond.copy()
    cond["_pref"] = (cond[prefer_col].notna().astype(int)
                     if prefer_col and prefer_col in cond.columns else 0)
    cond = (cond.sort_values(["_pref", "n_files"], ascending=[False, False])
                .drop_duplicates(subset="cell", keep="first")
                .drop(columns="_pref"))
    return features.merge(cond, on="cell", how="left", suffixes=("", "_cond"))

def condition_influence(df: pd.DataFrame, conditions: list[str],
                        features: list[str]) -> pd.DataFrame:
    """Pearson and Spearman of each condition against each feature, across cells."""
    from scipy.stats import pearsonr, spearmanr

    rows = []
    for cond in conditions:
        if cond not in df.columns:
            continue
        x_all = pd.to_numeric(df[cond], errors="coerce")
        if x_all.notna().sum() < 3 or x_all.nunique(dropna=True) < 2:
            continue  # constant or too sparse to say anything
        for feat in features:
            if feat not in df.columns:
                continue
            y_all = pd.to_numeric(df[feat], errors="coerce")
            m = x_all.notna() & y_all.notna()
            x, y = x_all[m], y_all[m]
            if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
                continue
            try:
                pr, pp = pearsonr(x, y)
                sr, sp = spearmanr(x, y)
            except Exception:
                continue
            rows.append({"condition": cond, "feature": feat, "n_cells": int(len(x)),
                         "pearson_r": pr, "pearson_p": pp,
                         "spearman_r": sr, "spearman_p": sp,
                         "abs_spearman": abs(sr)})
    out = pd.DataFrame(rows)
    return out.sort_values("abs_spearman", ascending=False).reset_index(drop=True) if not out.empty else out

def run(family: str, pattern: str, cond_csv_name: str, conditions: list[str],
        prefer_col: str | None, argv_desc: str):
    ap = argparse.ArgumentParser(description=argv_desc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"pipeline output root holding cell_feature/ (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--target-soh", type=float, default=DEFAULT_TARGET_SOH,
                    help=f"common SoH level to compare at (default: {DEFAULT_TARGET_SOH})")
    ap.add_argument("--report-only", action="store_true",
                    help="just print which cells reach which SoH, write nothing")
    ap.add_argument("--dest", type=Path, default=None, help="output folder (default: --out-dir)")
    ap.add_argument("--feature-dir", default="cell_feature",
                    help="subfolder of --out-dir holding the per-cell csv "
                         "(default: cell_feature; use cell_feature_cu_c2 for the LWHK C/2 data)")
    ap.add_argument("--soh-col", default=SOH_COL,
                    help=f"capacity column defining SOH (default: {SOH_COL}; "
                         f"use cap_dis for the C/2 discharge data)")
    ap.add_argument("--prefix", default=None,
                    help="output filename prefix (default: the family name)")
    args = ap.parse_args()

    feature_dir = args.out_dir / args.feature_dir
    if not feature_dir.is_dir():
        print(f"[ERROR] no cell_feature folder under {args.out_dir}")
        return

    reach = reachability(feature_dir, pattern, args.soh_col)
    if reach.empty:
        print(f"[INFO] no cell csv matching {pattern!r} in {feature_dir}")
        return

    print(f"=== {family} [{args.feature_dir}, SOH from {args.soh_col}]: {len(reach)} cell(s), "
          f"checkups {int(reach.n_checkups.min())}-{int(reach.n_checkups.max())}")
    print(f"    deepest SoH reached: {reach.soh_min.min():.4f}")
    for thr in (0.995, 0.99, 0.98, 0.97, 0.95):
        print(f"    cells reaching {thr:.3f}: {int((reach.soh_min <= thr).sum()):3}/{len(reach)}")
    print()

    if args.report_only:
        print(reach.sort_values("soh_min").to_string(index=False))
        return

    print(f"--- interpolating at SoH = {args.target_soh}")
    features, skipped = features_at_soh(feature_dir, pattern, args.target_soh,
                                        soh_col=args.soh_col)
    print()
    if features.empty:
        print(f"[ERROR] no cell brackets SoH {args.target_soh}; nothing to analyse.")
        if not skipped.empty:
            print(skipped.to_string(index=False))
        print("\nPick a level inside the measured range (see the table above), "
              "or run with --report-only.")
        return

    merged = join_conditions(features, args.out_dir / cond_csv_name, prefer_col)
    feat_cols = [c for c in features.columns
                 if c not in ("cell", "n_checkups", "soh_min", "soh_max")]
    influence = condition_influence(merged, conditions, feat_cols)

    dest = args.dest or args.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    tag = f"{args.target_soh:.3f}".replace("0.", "").rstrip("0") or "0"
    prefix = args.prefix or family.lower()
    f_path = dest / f"{prefix}_features_at_soh{tag}.csv"
    i_path = dest / f"{prefix}_condition_influence_soh{tag}.csv"
    merged.to_csv(f_path, index=False)
    influence.to_csv(i_path, index=False)

    print(f"[OK] {len(merged):3} cell(s)      -> {f_path}")
    print(f"[OK] {len(influence):3} pair(s)      -> {i_path}")
    if not skipped.empty:
        print(f"[--] {len(skipped):3} cell(s) skipped (target outside their range)")

    if not influence.empty:
        print(f"\nstrongest condition/feature relations at SoH {args.target_soh} "
              f"(n={len(merged)} cells):")
        print(influence.head(15).drop(columns="abs_spearman").to_string(index=False))

def main():
    run(family="JGNE",
        pattern="SPEED_JGNE_*",
        cond_csv_name="condition_overview_homocomp_jgne.csv",
        conditions=JGNE_CONDITIONS,
        prefer_col="pause_h",
        argv_desc=__doc__)

if __name__ == "__main__":
    main()
