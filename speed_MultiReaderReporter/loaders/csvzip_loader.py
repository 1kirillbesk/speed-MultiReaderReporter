# speed_MultiReaderReporter/loaders/csvzip_loader.py
from __future__ import annotations
from pathlib import Path
import zipfile, io, re, logging
import pandas as pd

from core.normalize import parse_columns, to_abs_time, to_float, to_str
from core.model import RunRecord

_LOG = logging.getLogger(__name__)

# ---------- filename helpers ----------
def _cell_from_name(fname: str) -> str:
    base = Path(fname).name
    eqs = [m.start() for m in re.finditer(r"=", base)]
    return base[eqs[0] + 1:eqs[1]].strip() if len(eqs) >= 2 else Path(base).stem


def infer_cell_from_path(path: Path) -> str:
    """Public helper to infer the cell id from a CSV/ZIP path."""
    return _cell_from_name(path.name)

def _program_from_name(fname: str) -> str:
    parts = Path(fname).name.split("=")
    return parts[5].strip() if len(parts) >= 7 else Path(fname).stem

# ---------- CSV normalization ----------
def _df_from_csv_bytes(buff: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(buff), sep=",", decimal=".", low_memory=False)
    zeit, strom, spannung, schritt, zustand = parse_columns(df)
    prozedur_col = None
    for c in df.columns:
        if str(c).strip().lower() == "prozedur":
            prozedur_col = c
            break
    if zeit is None or strom is None:
        raise ValueError("CSV missing required columns (Zeit/Strom).")

    cols = {
        "abs_time":  to_abs_time(df[zeit]),
        "current_A": to_float(df[strom]),
    }
    if spannung is not None:
        cols["voltage_V"] = to_float(df[spannung])
    if schritt is not None:
        cols["step_int"] = pd.to_numeric(df[schritt], errors="coerce").round().astype("Int64")
    if zustand is not None:
        cols["state"] = to_str(df[zustand])
    if prozedur_col is not None:
        cols["procedure"] = to_str(df[prozedur_col])
    out = pd.DataFrame(cols).dropna(subset=["abs_time", "current_A"]).sort_values("abs_time")
    return out.reset_index(drop=True)


def _segment_by_procedure(df: pd.DataFrame, cell: str, program: str,
                          source_path: Path, loader: str) -> list[RunRecord]:
    """
    Some CSVs contain nested procedures in a ``Prozedur`` column. When multiple
    contiguous procedure blocks are present, emit one RunRecord per block so the
    classifier can act on the inner procedure names.
    """
    proc_col = "procedure" if "procedure" in df.columns else None
    if proc_col is None:
        return [RunRecord(cell=cell, program=program, df=df, source_path=source_path, loader=loader)]

    unique_proc = df[proc_col].dropna().unique()
    if unique_proc.size <= 1:
        return [RunRecord(cell=cell, program=program, df=df, source_path=source_path, loader=loader)]

    segments: list[RunRecord] = []
    current_proc = None
    start_idx = None
    proc_series = df[proc_col]

    for idx, proc in enumerate(proc_series):
        if pd.isna(proc):
            if current_proc is not None and start_idx is not None:
                block = df.iloc[start_idx:idx].copy()
                if not block.empty:
                    segments.append(RunRecord(cell=cell, program=str(current_proc),
                                              df=block, source_path=source_path, loader=loader))
                current_proc = None
                start_idx = None
            continue

        if current_proc is None:
            current_proc = proc
            start_idx = idx
            continue

        if proc != current_proc:
            block = df.iloc[start_idx:idx].copy()
            if not block.empty:
                segments.append(RunRecord(cell=cell, program=str(current_proc),
                                          df=block, source_path=source_path, loader=loader))
            current_proc = proc
            start_idx = idx

    if current_proc is not None and start_idx is not None:
        block = df.iloc[start_idx:].copy()
        if not block.empty:
            segments.append(RunRecord(cell=cell, program=str(current_proc),
                                      df=block, source_path=source_path, loader=loader))

    if segments:
        _LOG.info("SPLIT %s into %d Prozedur segments for cell %s", source_path.name, len(segments), cell)
        return segments

    return [RunRecord(cell=cell, program=program, df=df, source_path=source_path, loader=loader)]

# ---------- public loader ----------
def load(path: Path, cfg: dict, out_root: Path) -> list[RunRecord]:
    """
    Accepts: a .zip with CSV members, or a loose .csv file.
    Returns: list[RunRecord] with canonical dataframe per CSV.
    """
    records: list[RunRecord] = []

    if path.suffix.lower() == ".csv":
        # loose CSV
        df = _df_from_csv_bytes(path.read_bytes())
        cell = _cell_from_name(path.name)
        program = _program_from_name(path.name)
        records.extend(_segment_by_procedure(df, cell, program, path, "csvzip"))
        return records

    # zip with CSVs
    with zipfile.ZipFile(path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            return records
        cell = _cell_from_name(path.name)
        program = _program_from_name(path.name)  # program taken from ZIP name (consistent with earlier)
        for member in members:
            raw = zf.read(member)
            try:
                df = _df_from_csv_bytes(raw)
            except Exception:
                continue
            records.extend(_segment_by_procedure(df, cell, program, path, "csvzip"))
    return records
