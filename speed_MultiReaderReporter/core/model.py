# speed_MultiReaderReporter/core/model.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class RunRecord:
    cell: str                 # e.g. SPEED_LW_reference_4
    program: str              # program/test name (short)
    df: pd.DataFrame          # canonical columns: abs_time, current_A, Zustand, (voltage_V)?, (step_int)?
    source_path: Path         # file on disk (.zip/.csv/.mat)
    loader: str               # "csvzip" or "mat"
