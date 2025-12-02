# speed_MultiReaderReporter/core/capacity.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Step19Result:
    capacity_Ah: float
    discharge_end_time: pd.Timestamp
    index_start: int
    index_end: int
    discharge_start_time: pd.Timestamp
    min_voltage_V: float | None

def capacity_for_step_Ah(g: pd.DataFrame, step_target: int, *,
                         want_negative: bool = True,
                         eod_v_cut: float | None = None,
                         i_thresh: float = 0.0) -> tuple[float, int | None, int | None]:
    if g.empty or "step_int" not in g.columns:
        return 0.0, None, None
    gg = g.sort_values("abs_time").copy()
    steps = gg["step_int"].to_numpy()
    if not np.any(steps == step_target):
        return 0.0, None, None

    dt_h = gg["abs_time"].diff().dt.total_seconds().fillna(0.0).to_numpy() / 3600.0
    I    = gg["current_A"].to_numpy(float)
    V    = gg["voltage_V"].to_numpy(float) if "voltage_V" in gg.columns else np.full_like(I, np.nan)

    mask = (steps == step_target)
    segs, s = [], None
    for k, flag in enumerate(mask):
        if flag and s is None: s = k
        elif not flag and s is not None: segs.append((s, k-1)); s = None
    if s is not None: segs.append((s, len(mask)-1))

    best = (0.0, None, None)
    for a, b in segs:
        end = b
        if eod_v_cut is not None and np.isfinite(V).any():
            rel = np.nonzero(V[a:b+1] <= eod_v_cut)[0]
            if rel.size: end = a + int(rel[0])
        if end < a: continue

        Iseg  = I[a:end+1]; Iprev = np.concatenate(([Iseg[0]], Iseg[:-1])); dt = dt_h[a:end+1]
        if want_negative:
            Iseg  = np.where(Iseg  <= -i_thresh, Iseg, 0.0)
            Iprev = np.where(Iprev <= -i_thresh, Iprev, 0.0)
            dQ = -0.5 * (Iseg + Iprev) * dt
        else:
            Iseg  = np.where(Iseg  >=  i_thresh, Iseg, 0.0)
            Iprev = np.where(Iprev >=  i_thresh, Iprev, 0.0)
            dQ =  0.5 * (Iseg + Iprev) * dt
        cap = float(np.sum(dQ))
        if abs(cap) > abs(best[0]): best = (cap, a, end)
    return best

def compute_checkup_point_step19(g: pd.DataFrame, *,
                                 min_step_required: int = 20,
                                 eod_v_cut: float | None = None,
                                 i_thresh: float = 0.0) -> Step19Result | None:
    if g.empty or "step_int" not in g.columns:
        return None
    gg = g.sort_values("abs_time").copy()
    steps_raw = gg["step_int"].to_numpy()
    steps = np.where(steps_raw == 9999, np.nan, steps_raw)
    finite = steps[np.isfinite(steps)]
    if finite.size == 0 or int(np.nanmax(finite)) < min_step_required:
        return None
    cap_ah, a19, b19 = capacity_for_step_Ah(gg, 19, want_negative=True,
                                            eod_v_cut=eod_v_cut, i_thresh=i_thresh)
    if a19 is None or b19 is None or cap_ah <= 1e-4:
        return None
    has22_after = np.any((steps == 22) & (np.arange(len(steps)) > b19))
    if not has22_after:
        return None
    t_end = gg["abs_time"].iloc[b19]
    t_start = gg["abs_time"].iloc[a19]

    min_v = None
    if "voltage_V" in gg.columns:
        vseg = gg["voltage_V"].to_numpy(float)[a19:b19+1]
        finite = vseg[np.isfinite(vseg)]
        if finite.size:
            min_v = float(np.min(finite))

    return Step19Result(
        capacity_Ah=cap_ah,
        discharge_end_time=t_end,
        index_start=a19,
        index_end=b19,
        discharge_start_time=t_start,
        min_voltage_V=min_v,
    )
