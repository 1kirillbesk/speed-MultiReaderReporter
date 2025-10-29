# speed_MultiReaderReporter/loaders/csvzip_loader.py
from __future__ import annotations
from pathlib import Path
import zipfile, io, re
import pandas as pd


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
def _parse_columns(df: pd.DataFrame):
    cmap = {str(c).strip().lower(): c for c in df.columns}
    zeit = cmap.get("zeit")
    strom = cmap.get("strom")
    spannung = cmap.get("spannung")            # optional
    schritt = None                             # exact 'Schritt' only (avoid Schrittdauer)
    for k, v in cmap.items():
        if k == "schritt":
            schritt = v
            break
    return zeit, strom, spannung, schritt

def _to_abs_time(series: pd.Series) -> pd.Series:
    z = pd.to_datetime(series, errors="coerce")
    if z.notna().any():
        return z
    # common fallbacks
    fmts = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S"]
    for fmt in fmts:
        z = pd.to_datetime(series, format=fmt, errors="coerce")
        if z.notna().any():
            return z
    return z  # all NaT (will be dropped)

def _to_float(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def _df_from_csv_bytes(buff: bytes) -> pd.DataFrame:
    # machine export: comma + '.' decimals
    df = pd.read_csv(io.BytesIO(buff), sep=",", decimal=".", low_memory=False)
    zeit, strom, spannung, schritt = _parse_columns(df)
    if zeit is None or strom is None:
        raise ValueError("CSV missing required columns (Zeit/Strom).")
    cols = {
        "abs_time":  _to_abs_time(df[zeit]),
        "current_A": _to_float(df[strom]),
    }
    if spannung is not None:
        cols["voltage_V"] = _to_float(df[spannung])
    if schritt is not None:
        cols["step_int"] = pd.to_numeric(df[schritt], errors="coerce").round().astype("Int64")
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
