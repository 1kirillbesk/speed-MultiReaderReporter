# speed_MultiReaderReporter/loaders/csvzip_loader.py
from __future__ import annotations
from pathlib import Path
import zipfile, io, re
import pandas as pd

from core.normalize import parse_columns, to_abs_time, to_float, to_str
from core.model import RunRecord

# ---------- filename helpers ----------
def _cell_from_name(fname: str) -> str:
    base = Path(fname).name
    eqs = [m.start() for m in re.finditer(r"=", base)]
    return base[eqs[0] + 1:eqs[1]].strip() if len(eqs) >= 2 else Path(base).stem

def _program_from_name(fname: str) -> str:
    parts = Path(fname).name.split("=")
    return parts[5].strip() if len(parts) >= 7 else Path(fname).stem

# ---------- CSV normalization ----------
def _df_from_csv_bytes(buff: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(buff), sep=",", decimal=".", low_memory=False)
    zeit, strom, spannung, schritt, zustand = parse_columns(df)
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
    out = pd.DataFrame(cols).dropna(subset=["abs_time", "current_A"]).sort_values("abs_time")
    return out.reset_index(drop=True)

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
        rec = RunRecord(
            cell=_cell_from_name(path.name),
            program=_program_from_name(path.name),
            df=df, source_path=path, loader="csvzip",
        )
        records.append(rec)
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
            records.append(RunRecord(cell=cell, program=program, df=df,
                                     source_path=path, loader="csvzip"))
    return records
