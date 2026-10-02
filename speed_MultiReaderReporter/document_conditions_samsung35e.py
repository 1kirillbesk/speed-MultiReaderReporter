# speed_MultiReaderReporter/document_conditions_samsung35e.py
"""Document the experimental condition of the Model2Life + MobilEM Samsung 35E cells.

Fourth label grammar in this code base, after JGNE homocomp
(document_conditions.py), LWHK tkammer (document_conditions_lwhk.py) and BALD
(document_conditions_bald.py). Unlike those three, the condition here is NOT
encoded in a long flag chain - it is the cycling programme name itself:

    MobilEM     pde_SA35E_Cyc<DOD>P<profile>[b]   e.g. pde_SA35E_Cyc50P3
                pde_SA35E_CycV P4 / CycAhP4       variable / Ah-based variants
    Model2Life  wli_SA_E35_Cyc_<nn>[_<date>]      e.g. wli_SA_E35_Cyc_03

Extracted:
    Cyc<n>          cycling depth of discharge in %   -> dod_pct
    P<n>            profile index (meaning not
                    documented in the file names)     -> profile
    trailing 'b'    repeat/variant of that profile    -> variant_b
    CycV / CycAh    non-constant-DOD variants         -> cyc_var / cyc_ah
    wli_..._Cyc_<n> Model2Life programme index        -> prog_idx
    Std_CUp,
    StdCUpEOL,
    Kap/qOCV/Puls   checkup, not a cycling condition  -> is_checkup
    TestPAU         pure pause segment                -> is_pause

`study` is taken from the raw folder the file came from, not from the label, so
it stays right even for the handful of mso_/juh_ one-off programmes.

The cells do not collide between the two studies (MobilEM 101-196, Model2Life
443-489), which is why both feed one table and one output root.

Usage:
    python speed_MultiReaderReporter/document_conditions_samsung35e.py
    python speed_MultiReaderReporter/document_conditions_samsung35e.py --excel
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from document_conditions import build_table

# raw folder -> study name, matching config_samsung35e.yaml's input.path list
DEFAULT_RAW_DIRS = {
    Path("E:/download/MobilEM/raw_data_csv"): "MobilEM",
    Path("E:/download/Model2Life/raw_data_csv"): "Model2Life",
}
DEFAULT_OUT_DIR = Path("E:/download/samsung35e_combined")
DEFAULT_CONTAINS = "LBE_Samsung_35E"

# Substrings that mark a label as a checkup rather than a cycling condition.
_CHECKUP_TOKENS = ("cup", "kap", "qocv", "puls", "kaptest")


def parse_condition_samsung35e(label: str) -> dict:
    lab = (label or "").lower()
    out = {
        "dod_pct": None, "profile": None, "prog_idx": None,
        "variant_b": False, "cyc_var": False, "cyc_ah": False,
        "is_checkup": False, "is_pause": False, "is_cycling": False,
    }

    # MobilEM: ...Cyc50P3 / Cyc20P5b / CycVP4 / CycAhP4
    m = re.search(r"cyc(\d+|v|ah)(?:p(\d+))?(b?)\b", lab)
    if m:
        head, prof, suffix = m.group(1), m.group(2), m.group(3)
        if head == "v":
            out["cyc_var"] = True
        elif head == "ah":
            out["cyc_ah"] = True
        else:
            out["dod_pct"] = int(head)
        if prof is not None:
            out["profile"] = int(prof)
        out["variant_b"] = suffix == "b"
        out["is_cycling"] = True

    # Model2Life: wli_SA_E35_Cyc_03 (a trailing _<date> is ignored)
    m = re.search(r"cyc_(\d+)", lab)
    if m:
        out["prog_idx"] = int(m.group(1))
        out["is_cycling"] = True

    out["is_pause"] = "pau" in lab and not out["is_cycling"]
    out["is_checkup"] = (not out["is_cycling"]
                         and any(t in lab for t in _CHECKUP_TOKENS))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, action="append", default=None,
                    help="repeatable; defaults to the MobilEM and Model2Life folders")
    ap.add_argument("--contains", default=DEFAULT_CONTAINS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"pipeline output root (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--excel", action="store_true")
    args = ap.parse_args()

    raw_dirs = ({d: d.parent.name for d in args.raw_dir} if args.raw_dir
                else DEFAULT_RAW_DIRS)

    feature_dir = args.out_dir / "cell_feature"
    parts = []
    for raw_dir, study in raw_dirs.items():
        if not raw_dir.is_dir():
            print(f"[WARN] raw folder not found, skipped: {raw_dir}")
            continue
        part = build_table(raw_dir, args.contains,
                           feature_dir if feature_dir.exists() else None,
                           parser=parse_condition_samsung35e)
        if part.empty:
            print(f"[INFO] no files containing {args.contains!r} under {raw_dir}")
            continue
        part.insert(2, "study", study)
        print(f"[ok] {study}: {len(part)} (cell, label) row(s) "
              f"over {part['cell'].nunique()} cell(s)")
        parts.append(part)

    if not parts:
        return
    df = pd.concat(parts, ignore_index=True).sort_values(["cell", "label"])
    df = df.reset_index(drop=True)

    out_csv = args.out or (args.out_dir / "condition_overview_samsung35e.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] {len(df)} row(s), {df['cell'].nunique()} cell(s) -> {out_csv}")

    if args.excel:
        try:
            df.to_excel(out_csv.with_suffix(".xlsx"), index=False)
            print(f"[OK] -> {out_csv.with_suffix('.xlsx')}")
        except ImportError as e:
            print(f"[WARN] xlsx skipped ({e}); pip install openpyxl")

    cyc = df[df["is_cycling"] == 1]
    print()
    print(f"cycling labels only ({len(cyc)} rows):")
    print(cyc.groupby(["study", "dod_pct", "profile"], dropna=False)["cell"]
             .nunique().to_string())


if __name__ == "__main__":
    main()
