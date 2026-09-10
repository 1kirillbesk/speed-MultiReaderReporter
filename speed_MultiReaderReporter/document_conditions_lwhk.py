# speed_MultiReaderReporter/document_conditions_lwhk.py
"""Document the experimental condition of the LWHK cells from their program label.

The LWHK temperature-chamber programs use a different grammar from the JGNE
homocomp ones, e.g.

    J8026_BMBF_SPEED=SPEED_LWHK_cycle_47=LWHK cycle=Ageing_V2_homo=
    2026-07-15 230637=eka_test_tkammer_200fec_30m=TS088500  Format01=Kreis M23-030

Two conditions are extracted from the label:
    _<n>fec   full equivalent cycles between rests   -> fec
    _<n>m     rest / relaxation time in MINUTES      -> rest_min

Also kept, since the labels carry it: a trailing _ref<n> marks a reference cell
(ref = 1, ref_index = n).

Labels that are not tkammer programs (the rul_eka_LW_CU_varCapa_relax checkups,
rul_LWcp_*, rul_drive_to50, rul_overhang_0) have no fec/rest tokens and are
reported with empty values.

Usage:
    python speed_MultiReaderReporter/document_conditions_lwhk.py
    python speed_MultiReaderReporter/document_conditions_lwhk.py --excel
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from document_conditions import (
    DEFAULT_OUT_DIR,
    DEFAULT_RAW_DIR,
    build_table,
)

DEFAULT_CONTAINS = "SPEED_LWHK"

def parse_condition_lwhk(label: str) -> dict:
    lab = (label or "").lower()
    out = {"fec": None, "rest_min": None, "ref": False, "ref_index": None}

    m = re.search(r"_(\d+)fec(?=_|$)", lab)
    if m:
        out["fec"] = int(m.group(1))

    m = re.search(r"_(\d+)m(?=_|$)", lab)
    if m:
        out["rest_min"] = int(m.group(1))

    m = re.search(r"_ref(\d*)(?=_|$)", lab)
    if m:
        out["ref"] = True
        out["ref_index"] = int(m.group(1)) if m.group(1) else None

    return out

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
    df = build_table(args.raw_dir, args.contains,
                     feature_dir if feature_dir.exists() else None,
                     parser=parse_condition_lwhk)
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
