# speed_MultiReaderReporter/loaders/csvzip_loader.py
from __future__ import annotations
from pathlib import Path
import zipfile, io, re, logging
from typing import Iterable
import pandas as pd

from ..core.normalize import parse_columns, to_abs_time, to_float, to_str
from ..core.model import RunRecord

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


def _pulse_keywords(cfg: dict | None) -> tuple[str, ...]:
    cls = (cfg or {}).get("classification", {}) if cfg else {}
    kws = cls.get("procedure_pulse_keywords")
    if isinstance(kws, Iterable) and not isinstance(kws, (str, bytes)):
        lowered = tuple(str(k).lower() for k in kws if str(k).strip())
        if lowered:
            return lowered
    return ("pulse",)


def _segment_by_procedure(df: pd.DataFrame, cell: str, program: str,
                          source_path: Path, loader: str, cfg: dict | None) -> list[RunRecord]:
    """
    Some CSVs contain nested procedures in a ``Prozedur`` column. When multiple
    contiguous procedure blocks are present, emit one RunRecord per block so the
    classifier can act on the inner procedure names.
    """
    proc_col = "procedure" if "procedure" in df.columns else None
    if proc_col is None:
        return [RunRecord(cell=cell, program=program, df=df, source_path=source_path, loader=loader)]

    proc_series_raw = df[proc_col].apply(lambda v: "" if pd.isna(v) else str(v).strip())
    proc_series_raw = proc_series_raw.replace({"nan": ""})

    unique_proc = pd.Series([p for p in proc_series_raw if p]).unique()
    if unique_proc.size <= 1:
        return [RunRecord(cell=cell, program=program, df=df, source_path=source_path, loader=loader)]

    pulse_kws = _pulse_keywords(cfg)
    effective_proc = pd.Series(pd.NA, index=df.index, dtype="object")
    active_proc: str | None = None
    pulse_attached = False

    for idx, proc in proc_series_raw.items():
        proc_lower = proc.lower()
        is_pulse = any(k in proc_lower for k in pulse_kws) if proc else False

        if proc and not is_pulse:
            active_proc = proc
            effective_proc.iloc[idx] = active_proc
        elif is_pulse and active_proc is not None:
            effective_proc.iloc[idx] = active_proc
            pulse_attached = True
        else:
            active_proc = None

    effective_df = df.copy()
    effective_df["effective_procedure"] = effective_proc
    effective_df = effective_df.dropna(subset=["effective_procedure"]).reset_index(drop=True)

    if effective_df.empty:
        return [RunRecord(cell=cell, program=program, df=df, source_path=source_path, loader=loader)]

    segments: list[RunRecord] = []
    current_proc = None
    start_idx = None
    proc_series = effective_df["effective_procedure"]

    for idx, proc in proc_series.items():
        if current_proc is None:
            current_proc = proc
            start_idx = idx
            continue

        if proc != current_proc:
            block = effective_df.iloc[start_idx:idx].copy().drop(columns=["effective_procedure"], errors="ignore")
            if not block.empty:
                segments.append(RunRecord(cell=cell, program=str(current_proc),
                                          df=block, source_path=source_path, loader=loader))
            current_proc = proc
            start_idx = idx

    if current_proc is not None and start_idx is not None:
        block = effective_df.iloc[start_idx:].copy().drop(columns=["effective_procedure"], errors="ignore")
        if not block.empty:
            segments.append(RunRecord(cell=cell, program=str(current_proc),
                                      df=block, source_path=source_path, loader=loader))

    if segments:
        _LOG.info("SPLIT %s into %d Prozedur segments for cell %s", source_path.name, len(segments), cell)
        if pulse_attached:
            _LOG.info("[SPLIT] merged pulse procedures into main procedures for file %s, produced %d runs.",
                      source_path.name, len(segments))
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
        records.extend(_segment_by_procedure(df, cell, program, path, "csvzip", cfg))
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
            records.extend(_segment_by_procedure(df, cell, program, path, "csvzip", cfg))
    return records
