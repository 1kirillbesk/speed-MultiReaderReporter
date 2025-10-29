# speed_MultiReaderReporter/core/reports.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from scipy.io import savemat
from .metrics import run_metrics

ReportFormat = Literal["csv", "mat", "both"]

def _build_dataframe(series_list) -> pd.DataFrame:
    """Build the per-run rows + TOTAL row, identical to your current CSV shape."""
    rows = [run_metrics(df, label) for df, label in series_list]
    total = {
        "program": "TOTAL", "start_time": "", "end_time": "",
        "duration_h": round(sum(r["duration_h"] for r in rows), 6),
        "throughput_Ah": round(sum(r["throughput_Ah"] for r in rows), 6),
        "net_Ah": round(sum(r["net_Ah"] for r in rows), 6),
        "avg_abs_current_A": "", "avg_current_A": "", "max_abs_current_A": "",
        "n_points": sum(r["n_points"] for r in rows),
    }
    df_out = pd.DataFrame(rows + [total], columns=[
        "program","start_time","end_time","duration_h","throughput_Ah","net_Ah",
        "avg_abs_current_A","avg_current_A","max_abs_current_A","n_points"
    ])
    return df_out

def _write_csv(df_out: pd.DataFrame, out_csv: Path, title: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] wrote report: {title} → {out_csv}")

def _to_mat_cellstr(seq: list[str]) -> np.ndarray:
    """Make a MATLAB column cell array from a list of strings."""
    seq2 = [("" if s is None else str(s)) for s in seq]
    arr = np.empty((len(seq2), 1), dtype=object)
    arr[:, 0] = seq2
    return arr

def _write_mat(df_out: pd.DataFrame, out_mat: Path, varname: str, title: str) -> None:
    """
    Save a MATLAB struct with fields matching the CSV columns.
    Strings become cell arrays (Nx1), numerics become double (Nx1).
    """
    out_mat.parent.mkdir(parents=True, exist_ok=True)

    def numcol(name: str) -> np.ndarray:
        return df_out[name].to_numpy(dtype=float).reshape(-1, 1)

    def strcol(name: str) -> np.ndarray:
        return _to_mat_cellstr(df_out[name].astype(str).replace("nan", "", regex=False).tolist())

    mat_struct = {
        "program":              strcol("program"),
        "start_time":           strcol("start_time"),
        "end_time":             strcol("end_time"),
        "duration_h":           numcol("duration_h"),
        "throughput_Ah":        numcol("throughput_Ah"),
        "net_Ah":               numcol("net_Ah"),
        "avg_abs_current_A":    _to_mat_cellstr([""] * len(df_out)),  # kept blank like CSV TOTAL logic
        "avg_current_A":        _to_mat_cellstr([""] * len(df_out)),
        "max_abs_current_A":    _to_mat_cellstr([""] * len(df_out)),
        "n_points":             df_out["n_points"].to_numpy(dtype=float).reshape(-1, 1),
    }

    savemat(out_mat, {varname: mat_struct})
    print(f"[OK] wrote report: {title} → {out_mat}")

def write_report(series_list,
                 out_base: Path,
                 title: str,
                 fmt: ReportFormat = "csv",
                 mat_variable: str = "report") -> None:
    """
    Write report(s) in the requested format.
    - out_base is a *base path without extension* (e.g., .../report)
    - fmt: "csv" | "mat" | "both"
    - mat_variable: MATLAB variable name of the struct
    """
    if not series_list:
        return
    df_out = _build_dataframe(series_list)

    if fmt in ("csv", "both"):
        _write_csv(df_out, out_base.with_suffix(".csv"), title)
    if fmt in ("mat", "both"):
        _write_mat(df_out, out_base.with_suffix(".mat"), mat_variable, title)
