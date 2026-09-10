# speed_MultiReaderReporter/extract_lwhk_cu_discharge.py
"""Extract the CC discharge that opens every LWHK cycling file, as its own feature set.

Each eka_test_tkammer_* file carries a second procedure, `rul_eka_CU`, before the
cycling starts:

    step 2  CC charge    +0.900 A -> 3.600 V
    step 3  CV taper at 3.600 V
    step 4  rest
    step 6  CC discharge -0.900 A, 3.466 -> 2.000 V     <- extracted here
    step 7  rest

MEASURED RATE. CNom in these files is 1.8 Ah and the discharge runs at 0.900 A,
so this is C/2 (0.5 C), not C/3 - C/3 would be 0.600 A. Verified identical on all
20 LWHK cells. The cycling proper runs at 1.800 A = 1 C.

The pipeline never sees these: the lwhk family matches checkups on "lw_cu", which
does not match "rul_eka_cu", so they are classified as cycling and dropped. All 47
tkammer files contain one, against only 2 full checkups per cell today.

Because the sweep is C/2 rather than the C/10 of the full checkup, the dQ/dV here
is polarisation-broadened - useful as a trend against itself over life, NOT
comparable with the ICA from rul_eka_LW_CU_varCapa_relax.

Output mirrors cell_feature/<cell>.csv column names, so extract_curve_features.py
and the SoH tooling work on it unchanged.

Usage:
    python speed_MultiReaderReporter/extract_lwhk_cu_discharge.py
    python speed_MultiReaderReporter/extract_lwhk_cu_discharge.py --no-curves
    python speed_MultiReaderReporter/extract_lwhk_cu_discharge.py --cells SPEED_LWHK_cycle_47
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE, HERE / "core", HERE / "loaders", HERE / "utils"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from loaders import csvzip_loader                      # noqa: E402
from core.capacity import (                            # noqa: E402
    extract_ICA, extract_DVA, extract_ITA, get_peaks, get_minima,
)

DEFAULT_RAW_DIR = Path("E:/download/jgne/raw_data_csv")
DEFAULT_OUT_DIR = Path("E:/download/jgne/out_jgne")

CU_PROCEDURE = "rul_eka_cu"      # matched case-insensitively
STEP_DISCHARGE = 6
STEP_CHARGE = 2

# Voltage window of this discharge; also the ICA/DVA resample range.
CFG = {"voltage": {"high": 3.6, "highm": 3.45, "lowm": 3.35, "low": 2.0}}

def _scalars(dis: pd.DataFrame, cha: pd.DataFrame) -> dict:
    t = pd.to_datetime(dis["abs_time"])
    dt_h = (t.iloc[-1] - t.iloc[0]).total_seconds() / 3600.0
    v = pd.to_numeric(dis["voltage_V"], errors="coerce")
    i = pd.to_numeric(dis["current_A"], errors="coerce")
    dt_s = t.diff().dt.total_seconds().fillna(0.0)

    out = {
        "cap_dis": float(pd.to_numeric(dis["qstep"], errors="coerce").iloc[-1]),
        "energy_dis_Wh": float((v * i.abs() * dt_s).sum() / 3600.0),
        "I_dis_A": float(i.abs().median()),
        "v_start": float(v.iloc[0]),
        "v_end": float(v.iloc[-1]),
        "v_mean": float(v.mean()),
        "duration_h": dt_h,
        "n_points": int(len(dis)),
    }
    if cha is not None and not cha.empty:
        out["cap_cha"] = float(pd.to_numeric(cha["qstep"], errors="coerce").iloc[-1])
    for col, name in (("T1", "T_cell"), ("Tenv", "T_env")):
        if col in dis.columns:
            s = pd.to_numeric(dis[col], errors="coerce")
            if s.notna().any():
                out[f"{name}_mean"] = float(s.mean())
                out[f"{name}_max"] = float(s.max())
                out[f"{name}_rise"] = float(s.max() - s.iloc[0])
    return out

def _curves(dis: pd.DataFrame, cell: str) -> dict:
    feats = {}
    V, dQdV, Q_intV = extract_ICA(dis, cell, CFG)
    feats["Vdis"], feats["dQdVdis"], feats["Q_intVdis"] = V, dQdV, Q_intV
    Q, dVdQ, V_intQ = extract_DVA(dis, cell, CFG)
    feats["Qdis"], feats["dVdQdis"], feats["V_intQdis"] = Q, dVdQ, V_intQ
    try:
        VT, dTdV, T_intV = extract_ITA(dis, cell, CFG)
        feats["VdisT"], feats["dTdV"], feats["T_intV"] = VT, dTdV, T_intV
    except Exception as e:
        print(f"    [warn] {cell}: ITA failed ({e})")

    pmax_i, pmin_i = get_peaks(V, dQdV, distance=50), get_minima(V, dQdV, distance=50)
    pmax_d, pmin_d = get_peaks(Q, dVdQ, distance=50), get_minima(Q, dVdQ, distance=50)
    feats["peakV_max_dis"], feats["peakICA_max_dis"] = pmax_i["x_peaks"], pmax_i["y_peaks"]
    feats["peakV_min_dis"], feats["peakICA_min_dis"] = pmin_i["x_peaks"], pmin_i["y_peaks"]
    feats["peakQ_max_dis"], feats["peakDVA_max_dis"] = pmax_d["x_peaks"], pmax_d["y_peaks"]
    feats["peakQ_min_dis"], feats["peakDVA_min_dis"] = pmin_d["x_peaks"], pmin_d["y_peaks"]
    return feats

def process_file(path: Path, out_root: Path, with_curves: bool) -> list[dict]:
    cell = csvzip_loader.infer_cell_from_path(path)
    runs = csvzip_loader.load(path, CFG, out_root) or []
    rows = []
    for rec in runs:
        if CU_PROCEDURE not in (rec.program or "").lower():
            continue
        df = rec.df
        if "step_int" not in df.columns:
            continue
        dis = df[df["step_int"] == STEP_DISCHARGE].reset_index(drop=True)
        cha = df[df["step_int"] == STEP_CHARGE].reset_index(drop=True)
        if dis.empty:
            print(f"    [warn] {cell}: no step {STEP_DISCHARGE} in {rec.program}")
            continue
        row = {"cell": cell, "CU_time": pd.to_datetime(dis["abs_time"]).iloc[0],
               "source_file": path.name, "procedure": rec.program}
        row.update(_scalars(dis, cha))
        if with_curves:
            try:
                row.update(_curves(dis, cell))
            except Exception as e:
                print(f"    [warn] {cell}: curve extraction failed ({e})")
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--cells", default="SPEED_LWHK",
                    help="substring the filename must contain (default: SPEED_LWHK)")
    ap.add_argument("--contains", default="tkammer",
                    help="programme substring carrying the CU (default: tkammer)")
    ap.add_argument("--no-curves", action="store_true",
                    help="scalars only; skip ICA/DVA/ITA and peaks (much faster)")
    ap.add_argument("--per-cell", action="store_true",
                    help="also write one csv per cell, like cell_feature/")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    files = sorted(p for p in args.raw_dir.iterdir()
                   if p.is_file() and args.cells.lower() in p.name.lower()
                   and args.contains.lower() in p.name.lower())
    if not files:
        print(f"[INFO] nothing matching {args.cells!r} + {args.contains!r} in {args.raw_dir}")
        return
    print(f"[detector] {len(files)} file(s)")

    rows = []
    for p in files:
        print(f"  [read] {p.name[:88]}")
        try:
            rows.extend(process_file(p, args.out_dir, not args.no_curves))
        except Exception as e:
            print(f"    [warn] failed: {e}")

    if not rows:
        print("[ERROR] no rul_eka_CU discharge found.")
        return

    df = pd.DataFrame(rows).sort_values(["cell", "CU_time"]).reset_index(drop=True)
    # SoH per cell against that cell's own first CU discharge
    df["SOH_c2"] = df.groupby("cell")["cap_dis"].transform(lambda s: s / s.iloc[0])

    dest = args.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "lwhk_cu_discharge_features.csv"
    df.to_csv(out, index=False)
    print(f"\n[OK] {len(df)} discharge(s) across {df.cell.nunique()} cell(s) -> {out}")

    if args.per_cell:
        pc = dest / "cell_feature_cu_c2"
        pc.mkdir(parents=True, exist_ok=True)
        for cell, g in df.groupby("cell"):
            g.to_csv(pc / f"{cell}.csv", index=False)
        print(f"[OK] per-cell csv -> {pc}")

    cols = ["cell", "CU_time", "I_dis_A", "cap_dis", "SOH_c2", "energy_dis_Wh", "duration_h"]
    print()
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

if __name__ == "__main__":
    main()
