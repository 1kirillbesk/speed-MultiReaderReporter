# speed_MultiReaderReporter/core/reports.py
from __future__ import annotations
from pathlib import Path
from typing import Literal, Sequence
import numpy as np
import pandas as pd
from scipy.io import savemat
from .metrics import run_metrics

ReportFormat = Literal["csv", "mat", "both"]

def _build_dataframe(series_list,
                     grouped_runs: Sequence[tuple[str, Sequence[tuple[pd.DataFrame, str]]]] | None = None
                     ) -> pd.DataFrame:
    """Build the per-run rows + TOTAL row.

    When ``grouped_runs`` is provided (same ordering/length as ``series_list``),
    the resulting dataframe also includes segment-level rows and a ``group``
    column.
    """

    use_grouped = bool(grouped_runs) and len(series_list) == len(grouped_runs or [])

    if use_grouped:
        rows: list[dict] = []
        total_duration = 0.0
        total_throughput = 0.0
        total_net = 0.0
        total_points = 0

        for (df_run, _), (run_label, segments) in zip(series_list, grouped_runs or []):
            for df_seg, grp_label in segments:
                metrics = run_metrics(df_seg, run_label)
                metrics["group"] = grp_label
                rows.append(metrics)

            run_total = run_metrics(df_run, run_label)
            run_total["group"] = "TOTAL"
            rows.append(run_total)

            total_duration += run_total["duration_h"]
            total_throughput += run_total["throughput_Ah"]
            total_net += run_total["net_Ah"]
            total_points += run_total["n_points"]

        overall_total = {
            "program": "TOTAL",
            "group": "TOTAL",
            "start_time": "",
            "end_time": "",
            "duration_h": round(total_duration, 6),
            "throughput_Ah": round(total_throughput, 6),
            "net_Ah": round(total_net, 6),
            "avg_abs_current_A": "",
            "avg_current_A": "",
            "max_abs_current_A": "",
            "n_points": int(total_points),
        }
        rows.append(overall_total)

        cols = [
            "program", "group", "start_time", "end_time", "duration_h",
            "throughput_Ah", "net_Ah", "avg_abs_current_A", "avg_current_A",
            "max_abs_current_A", "n_points",
        ]
        return pd.DataFrame(rows, columns=cols)

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

    if "group" in df_out.columns:
        mat_struct["group"] = strcol("group")

    savemat(out_mat, {varname: mat_struct})
    print(f"[OK] wrote report: {title} → {out_mat}")

def write_report(series_list,
                 out_base: Path,
                 title: str,
                 fmt: ReportFormat = "csv",
                 mat_variable: str = "report",
                 grouped_runs: Sequence[tuple[str, Sequence[tuple[pd.DataFrame, str]]]] | None = None) -> None:
    """
    Write report(s) in the requested format.
    - out_base is a *base path without extension* (e.g., .../report)
    - fmt: "csv" | "mat" | "both"
    - mat_variable: MATLAB variable name of the struct
    """
    if not series_list:
        return
    df_out = _build_dataframe(series_list, grouped_runs=grouped_runs)

    if fmt in ("csv", "both"):
        _write_csv(df_out, out_base.with_suffix(".csv"), title)
    if fmt in ("mat", "both"):
        _write_mat(df_out, out_base.with_suffix(".mat"), mat_variable, title)

def _build_dataframe_grouped(segments) -> pd.DataFrame:
    """
    segments: list of tuples (df_seg, program_label, group_label)
    """
    from .metrics import run_metrics
    rows = []
    for df_seg, prog, grp in segments:
        r = run_metrics(df_seg, prog)
        r["group"] = grp
        rows.append(r)
    cols = ["program","group","start_time","end_time","duration_h","throughput_Ah","net_Ah",
            "avg_abs_current_A","avg_current_A","max_abs_current_A","n_points"]
    return pd.DataFrame(rows, columns=cols)

def write_grouped_report(segments, out_base: Path, title: str,
                         fmt: ReportFormat = "csv", mat_variable: str = "report_grouped") -> None:
    if not segments:
        return
    df_out = _build_dataframe_grouped(segments)
    if fmt in ("csv","both"):
        _write_csv(df_out, out_base.with_suffix(".csv"), title)
    if fmt in ("mat","both"):
        _write_mat(df_out, out_base.with_suffix(".mat"), mat_variable, title)
