# speed_MultiReaderReporter/loaders/pkl_loader.py
from __future__ import annotations
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from core.model import RunRecord

def _load_pickle_any(path: Path):
    """Try pandas-aware unpickling, fallback to raw pickle."""
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def _extract_df(obj, fname: str) -> pd.DataFrame:
    """Extract DataFrame from a pickle object that might be DF or dict of DFs."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v
    raise TypeError(f"{fname}: unsupported pickle content ({type(obj)})")

def _df_from_pkl(path: Path) -> pd.DataFrame:
    """Normalize pickle content into the common format."""
    df_raw = _extract_df(_load_pickle_any(path), path.name).copy()

    # Expected columns
    for col in ["abs_time", "volt", "curr"]:
        if col not in df_raw.columns:
            raise KeyError(f"{path.name}: missing column {col}")

    df = pd.DataFrame({
        "abs_time": pd.to_datetime(df_raw["abs_time"], errors="coerce"),
        "voltage_V": pd.to_numeric(df_raw["volt"], errors="coerce"),
        "current_A": pd.to_numeric(df_raw["curr"], errors="coerce"),
    }).dropna(subset=["abs_time", "current_A"])

    df = df.sort_values("abs_time").reset_index(drop=True)
    return df

def _cell_from_path(p: Path) -> str:
    """Extract cell name between first and second '=' if present, else folder name."""
    parts = p.stem.split("=")
    if len(parts) >= 2:
        return parts[1].strip()
    return p.parent.name

def _program_from_path(p: Path) -> str:
    """Simplified program name extraction for .pkl (usually rpt id)."""
    return p.stem


def infer_cell_from_path(path: Path) -> str:
    """Public helper to infer the cell id from a PKL path."""
    return _cell_from_path(path)

def load(path: Path, cfg: dict, out_root: Path) -> list[RunRecord]:
    df = _df_from_pkl(path)
    return [RunRecord(
        cell=_cell_from_path(path),
        program=_program_from_path(path),
        df=df,
        source_path=path,
        loader="pkl",
    )]
