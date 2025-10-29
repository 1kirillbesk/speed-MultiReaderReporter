# speed_MultiReaderReporter/core/normalize.py
from __future__ import annotations
import pandas as pd

def parse_columns(df: pd.DataFrame):
    cmap = {str(c).strip().lower(): c for c in df.columns}
    zeit = cmap.get("zeit")
    strom = cmap.get("strom")
    spannung = cmap.get("spannung")
    zustand = cmap.get("zustand")
    schritt = None                           # exact 'Schritt' only (avoid Schrittdauer)
    for k, v in cmap.items():
        if k == "schritt":
            schritt = v
            break
    return zeit, strom, spannung, schritt, zustand

def to_abs_time(series) -> pd.Series:
    z = pd.to_datetime(series, errors="coerce")
    if z.notna().sum() == 0:
        fmts = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S"]
        for fmt in fmts:
            z = pd.to_datetime(series, format=fmt, errors="coerce")
            if z.notna().any():
                break
    return z

def to_float(s) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def to_str(s) -> pd.Series:
    return s.astype(str)
