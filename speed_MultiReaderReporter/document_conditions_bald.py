# speed_MultiReaderReporter/document_conditions_bald.py
"""Document the experimental condition of the BALD / Samsung 35E cells.

Third label grammar in this code base, after the JGNE homocomp one
(document_conditions.py) and the LWHK tkammer one (document_conditions_lwhk.py):

    rul_inhomoSAM_cyc_<crate>_<a>SOC<b>[_flags...]
    e.g. rul_inhomoSAM_cyc_0C25_20SOC80_dyn_sdod

Extracted:
    <a>SOC<b>       SOC window                       -> soc_start, soc_end
    <n>C / <n>C<m>  C-rate: 1C -> 1.0, 0C25 -> 0.25  -> c_rate
    _dyn            dynamic profile                  -> dyn
    _sdod           stepped depth of discharge       -> sdod
    _scur           stepped current                  -> scur
    _pau            extra pause                      -> pau
    _short          shortened programme              -> short
    _pulse...       any pulse programme              -> pulse
    _pulsecyc[_<n>] pulse cycling                    -> pulsecyc
    _pulse<n>h<m>min pause h + pulse period min      -> pause_h, pulse_min
    _<n>h           plain pause in hours             -> pause_h
    _<n>h<m>dod     pause hours + DOD, e.g. 20h40dod -> pause_h, dod
    _<n>cyc         cycle count, e.g. 20cyc          -> n_cyc
    _<n>T           ambient temperature degC (def 25) -> temp_C
    init/inithomo   initial-condition programme      -> is_init

Only flags that actually occur in E:/download/BALD/raw_data_csv are emitted:
dyn (988 files), sdod (413), scur (299), pau (102), pulse (101), pulsecyc (41),
short (56). `ssoc` and `long` appear in the tester's programme list but have no
measurement files anywhere under E:/download, so they are not parsed.

Flags are 1/0, as in the other two tables. Reuses build_table() from
document_conditions.py, so only the parser differs.

Usage:
    python speed_MultiReaderReporter/document_conditions_bald.py
    python speed_MultiReaderReporter/document_conditions_bald.py --contains SPEED_BALD
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from document_conditions import DEFAULT_OUT_DIR, build_table

DEFAULT_RAW_DIR = Path("E:/download/BALD/raw_data_csv")
DEFAULT_CONTAINS = "BALD_35E"

def _c_rate(token: str) -> float | None:
    """'1C' -> 1.0, '0C25' -> 0.25, '0C5' -> 0.5, '2C' -> 2.0."""
    m = re.fullmatch(r"(\d+)c(\d*)", token)
    if not m:
        return None
    whole, frac = m.group(1), m.group(2)
    return float(f"{whole}.{frac}") if frac else float(whole)

def parse_condition_bald(label: str) -> dict:
    lab = (label or "").lower()
    out = {
        "soc_start": None, "soc_end": None, "c_rate": None,
        "pause_h": None, "dod": None, "n_cyc": None, "pulse_min": None,
        "dyn": False, "sdod": False, "scur": False, "pau": False,
        "short": False, "pulse": False, "pulsecyc": False,
        "is_init": False,
        "temp_C": 25,        # baseline; _<n>T overrides it below
    }

    m = re.search(r"(\d+)soc(\d+)", lab)
    if m:
        out["soc_start"] = int(m.group(1))
        out["soc_end"] = int(m.group(2))

    # C-rate token sits between the programme name and the SOC window
    for tok in lab.split("_"):
        if re.fullmatch(r"\d+c\d*", tok):
            r = _c_rate(tok)
            if r is not None:
                out["c_rate"] = r
                break

    # compound pause+DOD, e.g. _20h40dod
    m = re.search(r"_(\d+)h(\d+)dod(?=_|$)", lab)
    if m:
        out["pause_h"] = int(m.group(1))
        out["dod"] = int(m.group(2))

    # _40T -> 40 degC; absent means the 25 degC baseline
    m = re.search(r"_(\d+)t(?=_|$)", lab)
    if m:
        out["temp_C"] = int(m.group(1))

    m = re.search(r"_(\d+)cyc(?=_|$)", lab)
    if m:
        out["n_cyc"] = int(m.group(1))

    # plain pause token, e.g. _10h / _40h / _20h_pulse. The dod form (_10h20dod)
    # is handled above and does not match here, because a digit follows the h.
    if out["pause_h"] is None:
        m = re.search(r"_(\d+)h(?=_|$)", lab)
        if m:
            out["pause_h"] = int(m.group(1))

    # _pulse10h10min -> pause 10 h, pulse every 10 min
    m = re.search(r"pulse(\d+)h(\d+)min", lab)
    if m:
        out["pause_h"] = out["pause_h"] or int(m.group(1))
        out["pulse_min"] = int(m.group(2))

    # `dynsdod` appears written both as _dyn_sdod and _dynsdod
    out["dyn"] = bool(re.search(r"_dyn", lab))
    out["sdod"] = "sdod" in lab
    out["scur"] = "scur" in lab
    out["pau"] = bool(re.search(r"_pau(?=_|$)", lab))
    out["short"] = bool(re.search(r"_short(?=_|$)", lab))
    out["pulse"] = "pulse" in lab
    out["pulsecyc"] = "pulsecyc" in lab
    out["is_init"] = bool(re.search(r"init", lab))
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--contains", default=DEFAULT_CONTAINS,
                    help=f"only files whose name contains this (default: {DEFAULT_CONTAINS})")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("E:/download/BALD/out_bald"),
                    help="pipeline output root (default: E:/download/BALD/out_bald)")
    ap.add_argument("--excel", action="store_true")
    args = ap.parse_args()

    feature_dir = args.out_dir / "cell_feature"
    df = build_table(args.raw_dir, args.contains,
                     feature_dir if feature_dir.exists() else None,
                     parser=parse_condition_bald)
    if df.empty:
        print(f"[INFO] no files containing {args.contains!r} under {args.raw_dir}")
        return

    out_csv = args.out or (args.out_dir / f"condition_overview_{args.contains}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] {len(df)} row(s) -> {out_csv}")

    if args.excel:
        try:
            df.to_excel(out_csv.with_suffix(".xlsx"), index=False)
            print(f"[OK] -> {out_csv.with_suffix('.xlsx')}")
        except ImportError as e:
            print(f"[WARN] xlsx skipped ({e}); pip install openpyxl")

    print()
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
