# speed_MultiReaderReporter/core/soh.py
from __future__ import annotations
import pandas as pd
from .metrics import integrate_ah

def cumulative_throughput_until(series_list: list[tuple[pd.DataFrame, str]],
                                t_end: pd.Timestamp) -> float:
    total_Ah = 0.0
    for df, _ in series_list:
        if df.empty: continue
        part = df[df["abs_time"] <= t_end]
        if part.shape[0] >= 2:
            thru, _ = integrate_ah(part)
            total_Ah += thru
    return total_Ah
