# speed_MultiReaderReporter/core/capacity.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class StepCapacityResult:
    capacity_Ah: float
    discharge_end_time: pd.Timestamp
    index_start: int
    index_end: int
    discharge_start_time: pd.Timestamp
    min_voltage_V: float | None
    step_id: int

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

def capacity_for_mask_Ah(g: pd.DataFrame, mask: np.ndarray, *,
                         want_negative: bool = True,
                         eod_v_cut: float | None = None,
                         i_thresh: float = 0.0) -> tuple[float, int | None, int | None]:
    if g.empty or mask is None or len(mask) != len(g):
        return 0.0, None, None

    gg = g
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.sum() == 0:
        return 0.0, None, None

    dt_h = gg["abs_time"].diff().dt.total_seconds().fillna(0.0).to_numpy() / 3600.0
    I = gg["current_A"].to_numpy(float)
    V = gg["voltage_V"].to_numpy(float) if "voltage_V" in gg.columns else np.full_like(I, np.nan)

    segs, s = [], None
    for k, flag in enumerate(mask_arr):
        if flag and s is None:
            s = k
        elif not flag and s is not None:
            segs.append((s, k - 1))
            s = None
    if s is not None:
        segs.append((s, len(mask_arr) - 1))

    best = (0.0, None, None)
    for a, b in segs:
        end = b
        if eod_v_cut is not None and np.isfinite(V).any():
            rel = np.nonzero(V[a:b + 1] <= eod_v_cut)[0]
            if rel.size:
                end = a + int(rel[0])
        if end < a:
            continue

        Iseg = I[a:end + 1]
        Iprev = np.concatenate(([Iseg[0]], Iseg[:-1]))
        dt = dt_h[a:end + 1]
        if want_negative:
            Iseg = np.where(Iseg <= -i_thresh, Iseg, 0.0)
            Iprev = np.where(Iprev <= -i_thresh, Iprev, 0.0)
            dQ = -0.5 * (Iseg + Iprev) * dt
        else:
            Iseg = np.where(Iseg >= i_thresh, Iseg, 0.0)
            Iprev = np.where(Iprev >= i_thresh, Iprev, 0.0)
            dQ = 0.5 * (Iseg + Iprev) * dt
        cap = float(np.sum(dQ))
        if abs(cap) > abs(best[0]):
            best = (cap, a, end)
    return best

def compute_checkup_point_step(g: pd.DataFrame, step_target: int, *,
                               min_step_required: int | None = 20,
                               eod_v_cut: float | None = None,
                               i_thresh: float = 0.0,
                               trailing_step_id: int | None = None,
                               require_trailing_step: bool = False) -> StepCapacityResult | None:
    if g.empty or "step_int" not in g.columns:
        return None
    gg = g.sort_values("abs_time").copy()
    steps_raw = gg["step_int"].to_numpy()
    steps = np.where(steps_raw == 9999, np.nan, steps_raw)
    finite = steps[np.isfinite(steps)]
    if min_step_required is not None:
        if finite.size == 0 or int(np.nanmax(finite)) < min_step_required:
            return None
    cap_ah, a_step, b_step = capacity_for_step_Ah(
        gg,
        step_target,
        want_negative=True,
        eod_v_cut=eod_v_cut,
        i_thresh=i_thresh,
    )
    if a_step is None or b_step is None or cap_ah <= 1e-4:
        return None
    if require_trailing_step and trailing_step_id is not None:
        has_trailing_after = np.any((steps == trailing_step_id) & (np.arange(len(steps)) > b_step))
        if not has_trailing_after:
            return None
    t_end = gg["abs_time"].iloc[b_step]
    t_start = gg["abs_time"].iloc[a_step]

    min_v = None
    if "voltage_V" in gg.columns:
        vseg = gg["voltage_V"].to_numpy(float)[a_step:b_step+1]
        finite = vseg[np.isfinite(vseg)]
        if finite.size:
            min_v = float(np.min(finite))

    return StepCapacityResult(
        capacity_Ah=cap_ah,
        discharge_end_time=t_end,
        index_start=a_step,
        index_end=b_step,
        discharge_start_time=t_start,
        min_voltage_V=min_v,
        step_id=step_target,
    )


def compute_checkup_point_step19(g: pd.DataFrame, *,
                                 min_step_required: int = 20,
                                 eod_v_cut: float | None = None,
                                 i_thresh: float = 0.0) -> StepCapacityResult | None:
    return compute_checkup_point_step(
        g,
        19,
        min_step_required=min_step_required,
        eod_v_cut=eod_v_cut,
        i_thresh=i_thresh,
        trailing_step_id=22,
        require_trailing_step=True,
    )


def compute_checkup_point_step6(g: pd.DataFrame, *,
                                min_step_required: int | None = None,
                                eod_v_cut: float | None = None,
                                i_thresh: float = 0.0,
                                trailing_step_id: int | None = None,
                                require_trailing_step: bool = False) -> StepCapacityResult | None:
    return compute_checkup_point_step(
        g,
        6,
        min_step_required=min_step_required,
        eod_v_cut=eod_v_cut,
        i_thresh=i_thresh,
        trailing_step_id=trailing_step_id,
        require_trailing_step=require_trailing_step,
    )


def compute_checkup_point_rpt_by_state(g: pd.DataFrame, *,
                                       discharge_state_keywords: list[str],
                                       min_segment_duration_s: float = 60.0,
                                       min_capacity_Ah: float = 1e-4,
                                       min_voltage_span_V: float | None = None,
                                       min_step_required: int | None = None,
                                       eod_v_cut: float | None = None,
                                       i_thresh: float = 0.0,
                                       trailing_step_id: int | None = None,
                                       require_trailing_step: bool = False) -> StepCapacityResult | None:
    if g.empty or "abs_time" not in g.columns or "current_A" not in g.columns:
        return None
    gg = g.sort_values("abs_time").copy()

    if min_step_required is not None and "step_int" in gg.columns:
        steps_raw = pd.to_numeric(gg["step_int"], errors="coerce").to_numpy()
        steps = np.where(steps_raw == 9999, np.nan, steps_raw)
        finite = steps[np.isfinite(steps)]
        if finite.size == 0 or int(np.nanmax(finite)) < min_step_required:
            return None

    if "state" not in gg.columns:
        return None

    kws = [str(k).strip().lower() for k in (discharge_state_keywords or []) if str(k).strip()]
    if not kws:
        return None

    state_vals = gg["state"].apply(lambda v: "" if pd.isna(v) else str(v)).str.lower()
    mask = state_vals.apply(lambda s: any(k in s for k in kws)).to_numpy()
    if not mask.any():
        return None

    segs, s = [], None
    for k, flag in enumerate(mask):
        if flag and s is None:
            s = k
        elif not flag and s is not None:
            segs.append((s, k - 1))
            s = None
    if s is not None:
        segs.append((s, len(mask) - 1))

    if not segs:
        return None

    dt_s = gg["abs_time"].diff().dt.total_seconds().fillna(0.0).to_numpy()
    keep = []
    for a, b in segs:
        duration = float(np.sum(dt_s[a:b + 1]))
        if duration >= min_segment_duration_s:
            keep.append((a, b))

    if not keep:
        return None

    best = (0.0, None, None)
    for a, b in keep:
        seg_mask = np.zeros(len(gg), dtype=bool)
        seg_mask[a:b + 1] = True
        cap_ah, a_seg, b_seg = capacity_for_mask_Ah(
            gg,
            seg_mask,
            want_negative=True,
            eod_v_cut=eod_v_cut,
            i_thresh=i_thresh,
        )
        if a_seg is None or b_seg is None or cap_ah <= min_capacity_Ah:
            continue
        if abs(cap_ah) > abs(best[0]):
            best = (cap_ah, a_seg, b_seg)

    cap_ah, a_step, b_step = best
    if a_step is None or b_step is None:
        return None

    if min_voltage_span_V is not None and "voltage_V" in gg.columns:
        vseg = pd.to_numeric(gg["voltage_V"].iloc[a_step:b_step + 1], errors="coerce").to_numpy()
        finite = vseg[np.isfinite(vseg)]
        if finite.size:
            span = float(np.nanmax(finite) - np.nanmin(finite))
            if span < min_voltage_span_V:
                return None

    if require_trailing_step and trailing_step_id is not None and "step_int" in gg.columns:
        steps = pd.to_numeric(gg["step_int"], errors="coerce").to_numpy()
        has_trailing_after = np.any((steps == trailing_step_id) & (np.arange(len(steps)) > b_step))
        if not has_trailing_after:
            return None

    t_end = gg["abs_time"].iloc[b_step]
    t_start = gg["abs_time"].iloc[a_step]
    min_v = None
    if "voltage_V" in gg.columns:
        vseg = gg["voltage_V"].to_numpy(float)[a_step:b_step + 1]
        finite = vseg[np.isfinite(vseg)]
        if finite.size:
            min_v = float(np.min(finite))

    return StepCapacityResult(
        capacity_Ah=cap_ah,
        discharge_end_time=t_end,
        index_start=a_step,
        index_end=b_step,
        discharge_start_time=t_start,
        min_voltage_V=min_v,
        step_id=-1,
    )
