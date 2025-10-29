# speed_MultiReaderReporter/core/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def integrate_ah(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty:
        return 0.0, 0.0
    t = (df["abs_time"] - df["abs_time"].iloc[0]).dt.total_seconds().to_numpy()
    i = df["current_A"].to_numpy()
    net_As  = np.trapezoid(i, x=t)
    thru_As = np.trapezoid(np.abs(i), x=t)
    return thru_As / 3600.0, net_As / 3600.0

def run_metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"program": label, "start_time": "", "end_time": "", "duration_h": 0.0,
                "throughput_Ah": 0.0, "net_Ah": 0.0,
                "avg_abs_current_A": 0.0, "avg_current_A": 0.0,
                "max_abs_current_A": 0.0, "n_points": 0}
    df = df.sort_values("abs_time")
    start, end = df["abs_time"].iloc[0], df["abs_time"].iloc[-1]
    dur_s = (end - start).total_seconds()
    thru_Ah, net_Ah = integrate_ah(df)
    import numpy as np
    return {
        "program": label,
        "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time":   end.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_h": round(dur_s / 3600.0, 6),
        "throughput_Ah": round(thru_Ah, 6),
        "net_Ah": round(net_Ah, 6),
        "avg_abs_current_A": round(float(np.nanmean(np.abs(df["current_A"]))), 6),
        "avg_current_A":     round(float(np.nanmean(df["current_A"])), 6),
        "max_abs_current_A": round(float(np.nanmax(np.abs(df["current_A"]))), 6),
        "n_points": int(df.shape[0]),
    }
