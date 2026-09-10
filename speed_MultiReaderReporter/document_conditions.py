# speed_MultiReaderReporter/document_conditions.py
"""Document the experimental condition of every cell from its program label.

Scans the raw-data folder, pulls the cell id and the program label out of each
filename, decodes the condition encoded in that label, and writes one row per
(cell, label) to a CSV (optionally .xlsx).

Label grammar, e.g. rul_homocomp_JGNE_40SOC100_220h_40h_pulse:
    <prefix>_<soc_start>SOC<soc_end>[_<n>h]...[_suffix]...
      <a>SOC<b>   SOC window the cell is cycled between
      _<n>h       pause time in hours; if a label has several
                  (e.g. _220h_40h) the LAST one is the pause
      _pulse      the program contains relaxation pulses
      _lowSOC     low-SOC variant
      _<n>T       ambient temperature in degC (default 25 when absent)
      _<x>C<y>C   charge / discharge C-rate

Usage:
    python speed_MultiReaderReporter/document_conditions.py
    python speed_MultiReaderReporter/document_conditions.py --contains homocomp_jgne --excel
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_RAW_DIR = Path("E:/download/jgne/raw_data_csv")
DEFAULT_OUT_DIR = Path("E:/download/jgne/out_jgne")
DEFAULT_CONTAINS = "homocomp_jgne"
DEFAULT_TEMP_C = 25

# ---------- filename helpers (same convention as loaders/csvzip_loader.py) ----------

def cell_from_name(fname: str) -> str:
    """Cell id sits between the first and second '=' of the filename."""
    base = Path(fname).name
    eqs = [m.start() for m in re.finditer("=", base)]
    return base[eqs[0] + 1:eqs[1]].strip() if len(eqs) >= 2 else Path(base).stem

def program_from_name(fname: str) -> str:
    """Program label is the 6th '='-separated field."""
    parts = Path(fname).name.split("=")
    return parts[5].strip() if len(parts) >= 7 else Path(fname).stem

def date_from_name(fname: str) -> str:
    """Test start stamp is the 5th field, 'YYYY-MM-DD HHMMSS'."""
    parts = Path(fname).name.split("=")
    return parts[4].strip() if len(parts) >= 7 else ""

# ---------- label parsing ----------

def _c_rate(val: str) -> float:
    """'1' -> 1.0, '15' -> 1.5, '05' -> 0.5 (two or more digits read as tenths)."""
    return float(val) / 10 if len(val) > 1 else float(val)

def parse_condition(label: str) -> dict:
    lab = (label or "").lower()
    out = {
        "soc_start": None, "soc_end": None,
        "pause_h": None,
        "has_pulse": False, "low_soc": False,
        "temp_C": DEFAULT_TEMP_C,
        "c_rate_chg": None, "c_rate_dchg": None,
    }

    m = re.search(r"(\d+)soc(\d+)", lab)
    if m:
        out["soc_start"] = int(m.group(1))
        out["soc_end"] = int(m.group(2))

    # A label can carry more than one _<n>h token (e.g. _220h_40h). The LAST one
    # is the pause; any earlier number is something else and is ignored.
    hours = re.findall(r"_(\d+)h(?=_|$)", lab)
    if hours:
        out["pause_h"] = int(hours[-1])

    out["has_pulse"] = "_pulse" in lab
    out["low_soc"] = "lowsoc" in lab

    m = re.search(r"_(\d+)t(?=_|$)", lab)
    if m:
        out["temp_C"] = int(m.group(1))

    m = re.search(r"_(\d+)c(\d+)c(?=_|$)", lab)
    if m:
        out["c_rate_chg"] = _c_rate(m.group(1))
        out["c_rate_dchg"] = _c_rate(m.group(2))

    return out

# ---------- table build ----------

def build_table(raw_dir: Path, contains: str, feature_dir: Path | None,
                parser=parse_condition) -> pd.DataFrame:
    """One row per (cell, label). `parser` maps a label to its condition dict,
    so a different naming scheme only needs its own parser (see
    document_conditions_lwhk.py)."""
    needle = contains.lower()
    rows = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or needle not in path.name.lower():
            continue
        rows.append({
            "cell": cell_from_name(path.name),
            "label": program_from_name(path.name),
            "test_date": date_from_name(path.name),
            **parser(program_from_name(path.name)),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cond_cols = [c for c in df.columns if c not in ("cell", "label", "test_date")]

    grouped = (
        df.groupby(["cell", "label"], as_index=False)
          .agg(n_files=("test_date", "size"),
               first_test=("test_date", "min"),
               last_test=("test_date", "max"),
               **{c: (c, "first") for c in cond_cols})
    )

    if feature_dir is not None:
        grouped["has_feature_csv"] = [
            (feature_dir / f"{c}.csv").exists() for c in grouped["cell"]
        ]

    # Flags are documented as 1/0 rather than True/False so the table drops
    # straight into numeric downstream use.
    for col in grouped.columns:
        if grouped[col].dtype == bool:
            grouped[col] = grouped[col].astype(int)

    # cell, label, then the parser's own condition columns in declaration order,
    # then the bookkeeping columns.
    lead = ["cell", "label"] + cond_cols + ["n_files", "first_test", "last_test"]
    ordered = [c for c in lead if c in grouped.columns]
    ordered += [c for c in grouped.columns if c not in ordered]
    return grouped[ordered].sort_values(["cell", "label"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                    help=f"folder holding the raw zip/csv files (default: {DEFAULT_RAW_DIR})")
    ap.add_argument("--contains", default=DEFAULT_CONTAINS,
                    help=f"only files whose name contains this (default: {DEFAULT_CONTAINS})")
    ap.add_argument("--out", type=Path, default=None,
                    help="output csv path (default: <out-dir>/condition_overview_<contains>.csv)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"pipeline output root, used for the default --out and the "
                         f"has_feature_csv column (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--excel", action="store_true",
                    help="also write .xlsx next to the csv (needs openpyxl)")
    args = ap.parse_args()

    feature_dir = args.out_dir / "cell_feature"
    df = build_table(args.raw_dir, args.contains, feature_dir if feature_dir.exists() else None)
    if df.empty:
        print(f"[INFO] no files containing {args.contains!r} under {args.raw_dir}")
        return

    out_csv = args.out or (args.out_dir / f"condition_overview_{args.contains}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] {len(df)} row(s) -> {out_csv}")

    if args.excel:
        out_xlsx = out_csv.with_suffix(".xlsx")
        try:
            df.to_excel(out_xlsx, index=False)
            print(f"[OK] {len(df)} row(s) -> {out_xlsx}")
        except ImportError as e:
            print(f"[WARN] xlsx skipped ({e}); pip install openpyxl")

    print()
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
