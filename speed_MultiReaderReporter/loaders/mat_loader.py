# speed_MultiReaderReporter/loaders/mat_loader.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

from core.model import RunRecord

# --- small helpers ---
def _to_abs_time_from_secs_or_datenum(t: np.ndarray) -> pd.Series:
    t = np.asarray(t, float).ravel()
    if t.size == 0:
        return pd.to_datetime([], utc=False)

    # MATLAB datenum ranges around 7xxxxx–8xxxxx (days since 0000-01-00)
    looks_like_datenum = (np.nanmedian(t) > 1e5) and (np.nanmedian(t) < 1e7)
    if looks_like_datenum:
        # convert datenum to datetime: days offset from 1970-01-01 is 719529
        return pd.to_datetime(t - 719529.0, unit="D", origin="unix")
    else:
        return pd.to_datetime(t, unit="s")

def _get_field(obj, name):
    try: return getattr(obj, name)
    except Exception: pass
    try: return obj[name]
    except Exception: return None

def _try_fast_diga_arrays(path: Path) -> dict[str, np.ndarray]:
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    diga = m.get("diga");  daten = _get_field(diga, "daten") if diga is not None else None
    if daten is None: raise RuntimeError("No diga.daten")
    def arr(x): return np.squeeze(np.asarray(x))
    t = _get_field(daten, "Zeit")
    v = _get_field(daten, "Spannung")
    i = _get_field(daten, "Strom")
    if t is None or v is None or i is None:
        raise RuntimeError("Missing Zeit/Spannung/Strom")
    out = {"t": arr(t).astype(float),
           "v": arr(v).astype(float),
           "i": arr(i).astype(float)}
    s = _get_field(daten, "Schritt")
    if s is not None:
        out["step"] = arr(s).astype(float)
    return out

# ---- filename parsing like your MAT code ----
def _cell_from_path(p: Path) -> str:
    parts = p.stem.split("=")
    if len(parts) >= 2:
        return parts[1].strip()
    return p.stem  # fallback

def _program_from_path(p: Path) -> str:
    parts = p.stem.split("=")
    return parts[5].strip() if len(parts) > 5 else ""

# ---- normalize to canonical dataframe ----
def _df_from_mat(path: Path) -> pd.DataFrame:
    a = _try_fast_diga_arrays(path)

    t = np.asarray(a["t"], float)
    v = np.asarray(a["v"], float)
    i = np.asarray(a["i"], float)
    if not (t.ndim == v.ndim == i.ndim == 1 and len(t) == len(v) == len(i) and len(t) > 1):
        raise RuntimeError(f"shape mismatch in {path.name}")

    abs_time = _to_abs_time_from_secs_or_datenum(t)

    data = {
        "abs_time":  abs_time,              # REQUIRED
        "current_A": i.astype(float),       # REQUIRED
        "voltage_V": v.astype(float),       # OPTIONAL but present for MATs
    }

    df = pd.DataFrame(data).dropna(subset=["abs_time", "current_A"])
    df = df.sort_values("abs_time").reset_index(drop=True)
    if "step" in a and len(a["step"]) == len(t):
        step_ser = pd.Series(a["step"])
        df["step_int"] = pd.to_numeric(step_ser, errors="coerce").round().astype("Int64")
    try:
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        diga = m.get("diga")
        daten = getattr(diga, "daten", None) if diga is not None else None
        zst = getattr(daten, "Zustand", None) if daten is not None else None
        if zst is not None:
            zst = np.asarray(zst).astype(object).ravel()
            # make length agree and convert to str series
            if len(zst) == len(df):
                df["state"] = pd.Series(zst).astype(str)
    except Exception:
        pass

    return df


# ---- public loader ----
def load(path: Path, cfg: dict, out_root: Path) -> list[RunRecord]:
    df = _df_from_mat(path)
    return [RunRecord(
        cell=_cell_from_path(path),
        program=_program_from_path(path),
        df=df,
        source_path=path,
        loader="mat",
    )]
