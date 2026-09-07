# speed_MultiReaderReporter/core/capacity.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from csaps import csaps
from scipy.signal import find_peaks, peak_prominences

import matplotlib.pyplot as plt
from matplotlib import cm
import os


def hampel_filter(x, k=11, t0=3.0):
    """
    Simple 1D Hampel filter (despike), similar to MATLAB hampel.
    k  = window size (odd)
    t0 = threshold in MAD units
    """
    x = np.asarray(x, float)
    n = len(x)
    x_filtered = x.copy()
    k2 = (k - 1) // 2

    for i in range(n):
        i_min = max(i - k2, 0)
        i_max = min(i + k2 + 1, n)
        window = x[i_min:i_max]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        if mad == 0:
            continue
        sigma = 1.4826 * mad
        if np.abs(x[i] - median) > t0 * sigma:
            x_filtered[i] = median
    return x_filtered

def resample_to_n(
    x,y,y2,
    n=100,kind="linear",
    x_lo=None,x_hi=None,
):
    """
    Resample y(x) and y2(x) onto n uniformly spaced x points using interpolation.

    Args:
        x   : x-values
        y   : first y-series
        y2  : second y-series (mandatory)
        n   : number of output points
        kind: kept for API compatibility (linear interpolation)
        x_lo, x_hi: optional bounds; if None, use data min/max

    Returns:
        x_new, y_new, y2_new
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    y2 = np.asarray(y2, float).ravel()

    # finite filtering
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(y2)
    x, y, y2 = x[ok], y[ok], y2[ok]
    if x.size < 2:
        return np.array([]), np.array([]), np.array([])

    # sort by x
    order = np.argsort(x)
    x, y, y2 = x[order], y[order], y2[order]

    # make x strictly increasing
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    y2 = y2[idx]

    # bounds (default to full range)
    lo = x[0] if x_lo is None else max(float(x_lo), float(x[0]))
    hi = x[-1] if x_hi is None else min(float(x_hi), float(x[-1]))
    if hi <= lo:
        return np.array([]), np.array([]), np.array([])

    # restrict to window, include neighbors if needed
    m = (x >= lo) & (x <= hi)
    if np.count_nonzero(m) < 2:
        left = np.searchsorted(x, lo, side="left")
        right = np.searchsorted(x, hi, side="right") - 1
        i0 = max(left - 1, 0)
        i1 = min(right + 1, len(x) - 1)
        xw = x[i0:i1 + 1]
        yw = y[i0:i1 + 1]
        y2w = y2[i0:i1 + 1]
    else:
        xw = x[m]
        yw = y[m]
        y2w = y2[m]

    if xw.size < 2:
        return np.array([]), np.array([]), np.array([])

    x_new = np.linspace(lo, hi, n)
    y_new = np.interp(x_new, xw, yw)
    y2_new = np.interp(x_new, xw, y2w)

    return x_new, y_new, y2_new


def smooth_derivative_csaps(x, y,fine_mult=1.0,          # 1.0 -> Nfine=max(1000, len(x)); 3.0 -> denser
    min_fine=1000,p_max=0.99999,hampel_k=None,          # e.g. 5 or 11 to despike y; None disables
    do_raw_derivative=True, # return raw dy/dx on original points
    negate=False            # if you want -dy/dx
):
    """
    General smoothing-derivative function (MATLAB csaps+fnder style) for ICA/DVA.
    Inputs:
      x, y : array-like
    Returns:
      x_fine, y_fine, dydx_smooth, (x_raw, dydx_raw)  [last two only if do_raw_derivative]
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # 0) hygiene
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 5:
        raise ValueError("Not enough valid points after removing NaNs/Infs.")

    # sort by x (required for monotone scaling)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # optional raw derivative before forcing uniqueness (keeps same length as original)
    dydx_raw = None
    x_raw = None
    if do_raw_derivative:
        x_raw = x.copy()
        dydx_raw = np.gradient(y, x)
        if negate:
            dydx_raw = -dydx_raw

    # make x strictly monotone for spline: keep first occurrence of each x
    x_u, iu = np.unique(x, return_index=True)
    y_u = y[iu]
    if len(x_u) < 5:
        raise ValueError("Not enough unique x values to build a spline.")

    # optional Hampel despike on y
    if hampel_k is not None:
        y_u = hampel_filter(y_u, k=hampel_k)

    # 1) scale x to [0,1]
    xmin, xmax = x_u[0], x_u[-1]
    dx = max(xmax - xmin, np.finfo(float).eps)
    xs = (x_u - xmin) / dx

    # 2) define fine grid in scaled x
    Nfine = int(max(min_fine, fine_mult * len(x_u)))
    xs_fine = np.linspace(0.0, 1.0, Nfine)

    # 3) GCV smoothing + clamp
    auto_res = csaps(xs, y_u, xs_fine)
    y_fine = auto_res.values
    p_used = min(auto_res.smooth, p_max)

    # 4) build final spline with p_used
    sp = csaps(xs, y_u, smooth=p_used)

    # analytic derivative wrt scaled x
    dy_dxs = sp(xs_fine, nu=1)

    # chain rule: dy/dx = (dy/dxs) * (dxs/dx) = (dy/dxs) * (1/dx)
    dydx_smooth = dy_dxs * (1.0 / dx)

    if negate:
        dydx_smooth = -dydx_smooth

    # rescale fine grid back to original x-units
    x_fine = xs_fine * dx + xmin

    if do_raw_derivative:
        return x_fine, y_fine, dydx_smooth, x_raw, dydx_raw, p_used
    else:
        return x_fine, y_fine, dydx_smooth, p_used


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

# here, analyze the checkup part
def extract_features(df,cell,cfg):
    def _keywords(value) -> tuple[str, ...]:
        if isinstance(value, (list, tuple, set)):
            return tuple(str(v).strip().lower() for v in value if str(v).strip())
        if isinstance(value, str):
            text = value.strip().lower()
            return (text,) if text else ()
        return ()

    def _contains_keywords(series: pd.Series, keywords: tuple[str, ...]) -> pd.Series:
        if not keywords:
            return pd.Series(False, index=series.index)
        out = pd.Series(False, index=series.index)
        lowered = series.astype(str).str.lower()
        for kw in keywords:
            out = out | lowered.str.contains(kw, case=False, na=False, regex=False)
        return out

    def _step_choice(values, choice: int, fallback: list[int]) -> int:
        if isinstance(values, (list, tuple)):
            if not values:
                values = fallback
            if len(values) > choice:
                return int(values[choice])
            return int(values[-1])
        return int(values)

    def _last_qstep(frame: pd.DataFrame) -> float:
        if frame.empty or "qstep" not in frame.columns:
            return np.nan
        return frame["qstep"].iloc[-1]

    features = {}
    enable_discharge_ocv = bool(cfg.get("analysis", {}).get("enable_discharge_ocv", False))
    # dt = df["relative_time_s"].diff()
    # CU_num = [17, 2, 6, 9]
    proc = df["procedure"].astype(str)
    analysis_cfg = cfg.get("analysis", {}) if isinstance(cfg, dict) else {}
    main_proc_keywords = _keywords(analysis_cfg.get("main_procedure_keywords"))
    charge_proc_keywords = _keywords(analysis_cfg.get("charge_procedure_keywords"))
    discharge_proc_keywords = _keywords(analysis_cfg.get("discharge_procedure_keywords"))

    if main_proc_keywords:
        choice = 0
        rpt_mask = _contains_keywords(proc, main_proc_keywords)
        charge_mask = _contains_keywords(proc, charge_proc_keywords) if charge_proc_keywords else rpt_mask
        rpt_cyc_mask = _contains_keywords(proc, discharge_proc_keywords) if discharge_proc_keywords else rpt_mask
    elif df["procedure"].str.contains("lw_cu", case=False, na=False).any():
        choice = 1
        rpt_mask = (df["procedure"].str.contains("lw_cu", case=False, na=False))
        charge_mask = rpt_mask
        rpt_cyc_mask = (df["procedure"].str.contains("lw_cu", case=False, na=False))
    else:
        choice = 0
        rpt_mask = (df["procedure"].str.contains("rpt", case=False, na=False))
        charge_mask = rpt_mask
        rpt_cyc_mask = (df["procedure"].str.contains("LWcp", case=False, na=False))
        lwcp_rows = proc[proc.str.contains("LWcp", case=False, na=False)]

        if not lwcp_rows.empty:
            last_token = lwcp_rows.iloc[0].rsplit("_", 1)[-1]
            if last_token == "25":
                choice = 2

    cu_steps_cfg = cfg.get("CU_steps", {})
    ocv_cha_steps = cu_steps_cfg.get("ocv_cha", [9, 22, 9])
    ocv_dis_steps = cu_steps_cfg.get("ocv_dis", [6, 19, 6])
    capa_cha_steps = cu_steps_cfg.get("capa_cha", [2, 15, 2])
    capa_dis_steps = cu_steps_cfg.get("capa_dis", [11, 9, 12])

    ocv_cha_step = _step_choice(ocv_cha_steps, choice, [9, 22, 9])
    ocv_dis_step = _step_choice(ocv_dis_steps, choice, [6, 19, 6])
    capa_cha_step = _step_choice(capa_cha_steps, choice, [2, 15, 2])
    capa_dis_step = _step_choice(capa_dis_steps, choice, [11, 9, 12])
    mask_ocv_cha = ((df["step_int"] == ocv_cha_step) & rpt_mask)
    mask_ocv_dis = ((df["step_int"] == ocv_dis_step) & rpt_mask)
    mask_capa_cha = ((df["step_int"] == capa_cha_step) & charge_mask)
    mask_capa_dis = ((df["step_int"] == capa_dis_step) & rpt_cyc_mask)

    df_ocv_cha = df.loc[mask_ocv_cha]; df_ocv_dis = df.loc[mask_ocv_dis]
    df_capa_cha = df.loc[mask_capa_cha]; df_capa_dis = df.loc[mask_capa_dis]

    features['CU_time'] = df["abs_time"].iloc[0]
    features['cap_dis'] = _last_qstep(df_capa_dis); features['cap_cha'] = _last_qstep(df_capa_cha)
    features['cap_ocv_cha'] = _last_qstep(df_ocv_cha)
    features['cap_ocv_dis'] = _last_qstep(df_ocv_dis) if enable_discharge_ocv else np.nan
    # todo: make the input a list for shorter code
    # get the ICA info for feature extraction
    V_cha, dQdV_cha, Q_intcha = extract_ICA(df_ocv_cha, cell, cfg)
    features["Vcha"] = V_cha
    features["dQdVcha"] = dQdV_cha
    features["Q_intVcha"] = Q_intcha
    V_dis = dQdV_dis = Q_intdis = None
    if enable_discharge_ocv and (not df_ocv_dis.empty):
        V_dis, dQdV_dis, Q_intdis = extract_ICA(df_ocv_dis, cell, cfg)
        features["Vdis"] = V_dis
        features["dQdVdis"] = dQdV_dis
        features["Q_intVdis"] = Q_intdis
    # get DVA peaks
    Q_cha, dVdQ_cha, V_intcha = extract_DVA(df_ocv_cha, cell, cfg)
    features["Qcha"] = Q_cha
    features["dVdQcha"] = dVdQ_cha
    features["V_intQcha"] = V_intcha
    Q_dis = dVdQ_dis = V_intdis = None
    if enable_discharge_ocv and (not df_ocv_dis.empty):
        Q_dis, dVdQ_dis, V_intdis = extract_DVA(df_ocv_dis, cell, cfg)
        features["Qdis"] = Q_dis
        features["dVdQdis"] = dVdQ_dis
        features["V_intQdis"] = V_intdis

    peak_cfg = cfg.get("peak_selection", {}) if isinstance(cfg, dict) else {}
    dist_pron = peak_cfg.get("distance_pronounced", 50)
    prom_pron = peak_cfg.get("min_prominence", None)
    width_pron = peak_cfg.get("min_width", None)
    three_k = int(peak_cfg.get("window_top_k", 3))

    cha_ica_ext = extrema_in_three_peak_window(
        V_cha, dQdV_cha, distance=dist_pron, top_k_window=three_k, prominence=prom_pron, width=width_pron
    )
    cha_dva_ext = extrema_in_three_peak_window(
        Q_cha, dVdQ_cha, distance=dist_pron, top_k_window=three_k, prominence=prom_pron, width=width_pron
    )

    features["peakV_max_cha"] = cha_ica_ext["max"]["x_peaks"]; features["peakICA_max_cha"] = cha_ica_ext["max"]["y_peaks"]
    features["peakQ_max_cha"] = cha_dva_ext["max"]["x_peaks"]; features["peakDVA_max_cha"] = cha_dva_ext["max"]["y_peaks"]
    features["peakV_min_cha"] = cha_ica_ext["min"]["x_peaks"]; features["peakICA_min_cha"] = cha_ica_ext["min"]["y_peaks"]
    features["peakQ_min_cha"] = cha_dva_ext["min"]["x_peaks"]; features["peakDVA_min_cha"] = cha_dva_ext["min"]["y_peaks"]

    # Keep compatibility fields but enforce the same single max/min rule.
    features["pr_peakV_max_cha"] = cha_ica_ext["max"]["x_peaks"]; features["pr_peakICA_max_cha"] = cha_ica_ext["max"]["y_peaks"]
    features["pr_peakV_min_cha"] = cha_ica_ext["min"]["x_peaks"]; features["pr_peakICA_min_cha"] = cha_ica_ext["min"]["y_peaks"]
    features["pr_peakQ_max_cha"] = cha_dva_ext["max"]["x_peaks"]; features["pr_peakDVA_max_cha"] = cha_dva_ext["max"]["y_peaks"]
    features["pr_peakQ_min_cha"] = cha_dva_ext["min"]["x_peaks"]; features["pr_peakDVA_min_cha"] = cha_dva_ext["min"]["y_peaks"]

    if enable_discharge_ocv and V_dis is not None and Q_dis is not None:
        dis_ica_ext = extrema_in_three_peak_window(
            V_dis, dQdV_dis, distance=dist_pron, top_k_window=three_k, prominence=prom_pron, width=width_pron
        )
        dis_dva_ext = extrema_in_three_peak_window(
            Q_dis, dVdQ_dis, distance=dist_pron, top_k_window=three_k, prominence=prom_pron, width=width_pron
        )
        features["peakV_max_dis"] = dis_ica_ext["max"]["x_peaks"]; features["peakICA_max_dis"] = dis_ica_ext["max"]["y_peaks"]
        features["peakQ_max_dis"] = dis_dva_ext["max"]["x_peaks"]; features["peakDVA_max_dis"] = dis_dva_ext["max"]["y_peaks"]
        features["peakV_min_dis"] = dis_ica_ext["min"]["x_peaks"]; features["peakICA_min_dis"] = dis_ica_ext["min"]["y_peaks"]
        features["peakQ_min_dis"] = dis_dva_ext["min"]["x_peaks"]; features["peakDVA_min_dis"] = dis_dva_ext["min"]["y_peaks"]

        features["pr_peakV_max_dis"] = dis_ica_ext["max"]["x_peaks"]; features["pr_peakICA_max_dis"] = dis_ica_ext["max"]["y_peaks"]
        features["pr_peakV_min_dis"] = dis_ica_ext["min"]["x_peaks"]; features["pr_peakICA_min_dis"] = dis_ica_ext["min"]["y_peaks"]
        features["pr_peakQ_max_dis"] = dis_dva_ext["max"]["x_peaks"]; features["pr_peakDVA_max_dis"] = dis_dva_ext["max"]["y_peaks"]
        features["pr_peakQ_min_dis"] = dis_dva_ext["min"]["x_peaks"]; features["pr_peakDVA_min_dis"] = dis_dva_ext["min"]["y_peaks"]

    # get the thermal features
    # V_disT, dTdV_dis, T_intV = extract_ITA(df_capa_dis, cell, cfg)
    # features["VdisT"] = V_disT
    # features["dTdV"] = dTdV_dis
    # features["T_intV"] = T_intV

    return features


def extract_ICA(df,cell,cfg):

    V = np.asarray(df["voltage_V"], dtype=float)
    q = np.asarray(df["qstep"], dtype=float)
    current_rate = df["current_A"].iloc[10]

    # Use separate voltage limits for discharge when available.
    vol_cha = cfg.get("voltage", {})
    vol_dis = cfg.get("voltage_dis", vol_cha)
    lim_cfg = vol_cha if current_rate > 0 else vol_dis

    dV = np.diff(V)
    not_increasing = dV <= 0; not_decreasing = dV >= 0
    above_threshold = V[1:] > lim_cfg['uplim']; below_threshold = V[1:] < lim_cfg['downlim']
    if current_rate > 0:
        stop_idx = np.where(not_increasing & above_threshold)[0]
    else:
        stop_idx = np.where(not_decreasing & below_threshold)[0]
    if len(stop_idx) > 0:
        cut = stop_idx[0] + 1  # +1 because diff shifts index
        V = V[:cut]
        q = q[:cut]
    try:
        V_fine_ica, Q_fine_ica, dQdV_smooth, V_raw, dQdV_orig, _ = smooth_derivative_csaps(
            V, q,
            fine_mult=1.0,
            min_fine=1000,
            p_max=0.9999,
            hampel_k=5,
            do_raw_derivative=True,
            negate=True
        )
    except ValueError as e:
        print(f"[WARN] {cell} RPT skipped (DVA): {e}")

    V_100, dQdV_100, Q_intV = resample_to_n(
        V_fine_ica,
        dQdV_smooth,
        Q_fine_ica,
        x_lo=lim_cfg['downlim'],
        x_hi=lim_cfg['uplim'],
        n=1000,
    )

    return V_100, dQdV_100, Q_intV


def extract_DVA(df,cell,cfg):
    V = np.asarray(df["voltage_V"], dtype=float)
    q = np.asarray(df["qstep"], dtype=float)
    current_rate = df["current_A"].iloc[10]

    vol_cha = cfg.get("voltage", {})
    vol_dis = cfg.get("voltage_dis", vol_cha)
    lim_cfg = vol_cha if current_rate > 0 else vol_dis

    dV = np.diff(V)
    not_increasing = dV <= 0; not_decreasing = dV >= 0
    above_threshold = V[1:] > lim_cfg['uplim']; below_threshold = V[1:] < lim_cfg['downlim']
    if current_rate > 0:
        stop_idx = np.where(not_increasing & above_threshold)[0]
    else:
        stop_idx = np.where(not_decreasing & below_threshold)[0]
    if len(stop_idx) > 0:
        cut = stop_idx[0] + 1  # +1 because diff shifts index
        V = V[:cut]
        q = q[:cut]
    try:
        Q_fine, V_fine, dVdQ_smooth, Q_raw, dVdQ_orig, _ = smooth_derivative_csaps(
            q, V,
            fine_mult=1.0,
            min_fine=1000,
            p_max=0.9999,
            hampel_k=5,
            do_raw_derivative=True,
            negate=True
        )
    except ValueError as e:
        print(f"[WARN] {cell} RPT skipped (DVA): {e}")

    Q_100, dVdQ_100, V_intQ = resample_to_n(Q_fine, dVdQ_smooth, V_fine,n=1000)

    return Q_100, dVdQ_100, V_intQ


def extract_ITA(df,cell,cfg):
    V = np.asarray(df["voltage_V"], dtype=float)
    T = np.asarray(df["T1"], dtype=float)
    current_rate = df["current_A"].iloc[10]

    vol_cha = cfg.get("voltage", {})
    vol_dis = cfg.get("voltage_dis", vol_cha)
    lim_cfg = vol_cha if current_rate > 0 else vol_dis

    dV = np.diff(V)
    not_increasing = dV <= 0; not_decreasing = dV >= 0
    above_threshold = V[1:] > lim_cfg['uplim']; below_threshold = V[1:] < lim_cfg['downlim']
    if current_rate > 0:
        stop_idx = np.where(not_increasing & above_threshold)[0]
    else:
        stop_idx = np.where(not_decreasing & below_threshold)[0]
    if len(stop_idx) > 0:
        cut = stop_idx[0] + 1  # +1 because diff shifts index
        V = V[:cut]
        T = T[:cut]
    try:
        V_fine_ica, T_fine_ica, dTdV_smooth, V_raw, dTdV_orig, _ = smooth_derivative_csaps(
            V, T,
            fine_mult=1.0,
            min_fine=1000,
            p_max=0.9999,
            hampel_k=5,
            do_raw_derivative=True,
            negate=True
        )
    except ValueError as e:
        print(f"[WARN] {cell} RPT skipped (DVA): {e}")

    V_100, dTdV_100, T_intV = resample_to_n(
        V_fine_ica,
        dTdV_smooth,
        T_fine_ica,
        x_lo=lim_cfg['downlim'],
        x_hi=lim_cfg['uplim'],
        n=1000,
    )

    return V_100, dTdV_100, T_intV


def window_delta_mean_var(
    df: pd.DataFrame, x_col: str, y_col: str,
    x_lo: float, x_hi: float, baseline_idx=0,
):
    """
    For each row i:
      mask = x_i in [x_lo, x_hi]
      diff = (y_i - y_baseline)[mask]
      store mean(diff), var(diff)
    """
    mean = []
    var = []

    y0 = np.asarray(df.loc[baseline_idx, y_col], dtype=float)
    y0 = np.nan_to_num(y0, nan=0.0)

    for i in range(len(df)):
        if i == baseline_idx:
            mean.append(0.0)
            var.append(0.0)
            continue

        y_i = np.asarray(df.loc[i, y_col], dtype=float)
        x_i = np.asarray(df.loc[i, x_col], dtype=float)

        # 🔹 replace NaNs with 0
        y_i = np.nan_to_num(y_i, nan=0.0)
        x_i = np.nan_to_num(x_i, nan=0.0)

        mask = (x_i >= x_lo) & (x_i <= x_hi)

        if not np.any(mask):
            mean.append(np.nan)
            var.append(np.nan)
            continue

        n = min(len(y0), len(y_i), len(x_i))
        diff = (y_i[:n] - y0[:n])[mask[:n]]

        mean.append(float(np.mean(diff)))
        var.append(float(np.var(diff, ddof=0)))

    return mean, var


def compute_window_feature_set(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    windows: dict,
    prefix: str,
    baseline_idx: int = 0,
):
    """
    Compute mean/var delta features for multiple windows at once.
    windows: {"tag": (x_lo, x_hi), ...}
    output keys: mean_{prefix}_{tag}, var_{prefix}_{tag}
    """
    out = {}
    for tag, bounds in windows.items():
        x_lo, x_hi = bounds
        m, v = window_delta_mean_var(
            df,
            x_col=x_col,
            y_col=y_col,
            x_lo=float(x_lo),
            x_hi=float(x_hi),
            baseline_idx=baseline_idx,
        )
        out[f"mean_{prefix}_{tag}"] = m
        out[f"var_{prefix}_{tag}"] = v
    return out


def get_peaks(x, y, distance=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"V and dQdV must have the same length. Got {len(x)} and {len(y)}")
    # Find peak indices
    idx, properties = find_peaks(y, distance=distance)

    return {
        "idx": idx,
        "x_peaks": x[idx],
        "y_peaks": y[idx],
        "properties": properties
    }

def get_minima(x, y, distance=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y must have the same length. Got {len(x)} and {len(y)}"
        )
    # Minima = peaks of -y
    idx, properties = find_peaks(-y, distance=distance)

    return {
        "idx": idx,
        "x_peaks": x[idx],
        "y_peaks": y[idx],
        "properties": properties
    }


def _most_pronounced_core(
    x,
    y,
    *,
    distance=None,
    top_k=3,
    minima=False,
    prominence=None,
    width=None,
):
    """
    Peak picker ranked by prominence (robust when early peaks fade/shift).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have same length. Got {len(x)} and {len(y)}")

    y_work = -y if minima else y
    idx, props = find_peaks(y_work, distance=distance, prominence=prominence, width=width)
    if idx.size == 0:
        return {
            "idx": idx,
            "x_peaks": x[idx],
            "y_peaks": y[idx],
            "prominence": np.array([], dtype=float),
            "properties": props,
        }

    prom = peak_prominences(y_work, idx)[0]
    order = np.argsort(prom)[::-1]
    if top_k is not None and int(top_k) > 0:
        order = order[: int(top_k)]
    idx_sel = idx[order]
    prom_sel = prom[order]

    # Keep selected peaks ordered by x for readability
    ord_x = np.argsort(x[idx_sel])
    idx_sel = idx_sel[ord_x]
    prom_sel = prom_sel[ord_x]

    return {
        "idx": idx_sel,
        "x_peaks": x[idx_sel],
        "y_peaks": y[idx_sel],
        "prominence": prom_sel,
        "properties": props,
    }


def get_most_pronounced_peaks(x, y, distance=None, top_k=3, prominence=None, width=None):
    return _most_pronounced_core(
        x,
        y,
        distance=distance,
        top_k=top_k,
        minima=False,
        prominence=prominence,
        width=width,
    )


def get_most_pronounced_minima(x, y, distance=None, top_k=3, prominence=None, width=None):
    return _most_pronounced_core(
        x,
        y,
        distance=distance,
        top_k=top_k,
        minima=True,
        prominence=prominence,
        width=width,
    )

def topk_by_y(xp, yp, k, largest=True):
    """
    Select top-k peaks by y value.
    largest=True  -> take k largest y (maxima)
    largest=False -> take k smallest y (minima; most negative / lowest)
    Returns (xp_sel, yp_sel)
    """
    xp = np.asarray(xp).ravel()
    yp = np.asarray(yp).ravel()

    m = np.isfinite(xp) & np.isfinite(yp)
    xp, yp = xp[m], yp[m]
    if xp.size == 0:
        return xp, yp

    if k is None or k <= 0 or xp.size <= k:
        return xp, yp

    if largest:
        idx = np.argsort(yp)[-k:]   # biggest y
    else:
        idx = np.argsort(yp)[:k]    # smallest y

    # sort selected points by x (prettier)
    idx = idx[np.argsort(xp[idx])]
    return xp[idx], yp[idx]

def extrema_in_three_peak_window(
    x,
    y,
    *,
    distance=50,
    top_k_window=3,
    prominence=None,
    width=None,
):
    """
    Keep only ONE maximum and ONE minimum inside the x-window spanned by
    the most pronounced `top_k_window` maxima.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape[0] != y.shape[0] or x.size == 0:
        empty = np.array([], dtype=float)
        return {"max": {"x_peaks": empty, "y_peaks": empty}, "min": {"x_peaks": empty, "y_peaks": empty}}

    core = get_most_pronounced_peaks(
        x,
        y,
        distance=distance,
        top_k=top_k_window,
        prominence=prominence,
        width=width,
    )
    if core["x_peaks"].size >= 2:
        lo = float(np.min(core["x_peaks"]))
        hi = float(np.max(core["x_peaks"]))
    else:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))

    pmax = get_peaks(x, y, distance=distance)
    pmin = get_minima(x, y, distance=distance)

    max_mask = (pmax["x_peaks"] >= lo) & (pmax["x_peaks"] <= hi)
    min_mask = (pmin["x_peaks"] >= lo) & (pmin["x_peaks"] <= hi)

    x_max = pmax["x_peaks"][max_mask]
    y_max = pmax["y_peaks"][max_mask]
    x_min = pmin["x_peaks"][min_mask]
    y_min = pmin["y_peaks"][min_mask]

    if y_max.size > 0:
        i_max = int(np.argmax(y_max))
        x_max = np.array([x_max[i_max]], dtype=float)
        y_max = np.array([y_max[i_max]], dtype=float)
    else:
        x_max = np.array([], dtype=float)
        y_max = np.array([], dtype=float)

    if y_min.size > 0:
        i_min = int(np.argmin(y_min))
        x_min = np.array([x_min[i_min]], dtype=float)
        y_min = np.array([y_min[i_min]], dtype=float)
    else:
        x_min = np.array([], dtype=float)
        y_min = np.array([], dtype=float)

    return {
        "max": {"x_peaks": x_max, "y_peaks": y_max},
        "min": {"x_peaks": x_min, "y_peaks": y_min},
    }

# here, analyze the throughput part
def compute_cum_abs_charge(df, time_col="abs_time", current_col="current_A"):
    """
    Convert absolute timestamp to relative time,
    compute dt, multiply by |current_A|,
    take cumulative sum, and return the final value.

    Returns:
      final_cum_value (float)
    """

    # ensure datetime
    t = pd.to_datetime(df[time_col])

    # store first absolute time
    first_abs_time = t.iloc[0]

    # relative time in seconds
    t_rel = (t - first_abs_time).dt.total_seconds()

    # time difference
    dt = t_rel.diff().fillna(0.0)

    # elementwise |I| * dt
    dt_abs_I = dt * df[current_col].abs()

    # cumulative sum
    cum_abs_I = dt_abs_I.cumsum()

    return {
        "abs_time": first_abs_time,
        "throughput": float(cum_abs_I.iloc[-1]),
    }
