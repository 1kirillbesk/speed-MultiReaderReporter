from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from csaps import csaps
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib import cm
import os

def _to_rel_seconds(abs_time: pd.Series) -> pd.Series:
    """
    Convert abs_time column to float seconds relative to the start of the provided Series.
    Handles datetime-like or numeric time.
    """
    s = abs_time.copy()

    # If it's already numeric-ish, try numeric conversion first
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.9:
        rel = s_num - s_num.iloc[0]
        return rel.astype(float)

    # Otherwise parse as datetime
    t = pd.to_datetime(s, errors="coerce", utc=True)
    if t.isna().all():
        raise ValueError("abs_time could not be parsed as numeric or datetime.")
    rel = (t - t.iloc[0]).dt.total_seconds()
    return rel.astype(float)

def _value_at_time(t: np.ndarray, y: np.ndarray, target: float):
    """
    Return y at the sample closest to target time using searchsorted (right).
    """
    if target <= t[0]:
        return y[0]
    if target >= t[-1]:
        return y[-1]
    j = np.searchsorted(t, target, side="left")
    # pick nearer of j-1 and j
    j0 = max(j - 1, 0)
    j1 = min(j, len(t) - 1)
    return y[j0] if abs(t[j0] - target) <= abs(t[j1] - target) else y[j1]

def split_into_3_sections_by_step_reset(df: pd.DataFrame, step_col="step_int"):
    """
    Split df into sections whenever step_int resets (diff < 0).
    Returns a list of 3 dataframes (or fewer if not enough resets).
    """
    df = df.copy().reset_index(drop=True)

    # Find reset points where step goes down (e.g., 12 -> 1)
    step = pd.to_numeric(df[step_col], errors="coerce")
    resets = np.where(step.diff().fillna(0).to_numpy() < 0)[0]  # indices where a new section starts

    # Build section boundaries
    starts = np.r_[0, resets]
    ends   = np.r_[resets - 1, len(df) - 1]

    sections = [df.iloc[s:e+1].reset_index(drop=True) for s, e in zip(starts, ends)]

    # You said there are 3 SOC blocks (80/50/20) -> take first 3
    return sections[:3]


def _soc_prefix(soc_label) -> str:
    try:
        return f"pulse{int(round(float(soc_label) * 100.0))}"
    except Exception:
        return f"pulse_{str(soc_label)}"

def analyze_section_pulses(
    sec: pd.DataFrame,
    voltage_col="voltage_V",
    current_col="current_A",      # or "currentrate" if that's your column name
    time_col="abs_time",
    near_zero_A=0.05,             # current magnitude below this is considered "rest"
    pulse_A=0.2,                  # current magnitude above this is considered "pulse"
    min_gap_s=2.0,                # minimum time between pulse starts (debounce)
    pre_window_s=0.3,             # voltage baseline window before pulse start
    I_est_window_s=0.2,           # window after start to estimate pulse current level
    eval_times=(0.2, 1.0, 10.0),  # seconds after pulse start
):
    """
    Detect pulses and compute R at eval_times.
    Returns list of dicts (one per pulse).
    """
    sec = sec.copy().reset_index(drop=True)

    # Build relative time in seconds
    sec["_t"] = (
            pd.to_datetime(sec["abs_time"])
            - pd.to_datetime(sec["abs_time"]).iloc[0]
    ).dt.total_seconds()

    t = sec["_t"].to_numpy(dtype=float)
    V = pd.to_numeric(sec[voltage_col], errors="coerce").to_numpy(dtype=float)
    I = pd.to_numeric(sec[current_col], errors="coerce").to_numpy(dtype=float)

    # Drop rows with NaNs in any key signal
    good = np.isfinite(t) & np.isfinite(V) & np.isfinite(I)
    t, V, I = t[good], V[good], I[good]
    if len(t) < 10:
        return []

    # Pulse start condition: from rest to pulse
    rest = np.abs(I) <= near_zero_A
    pul  = np.abs(I) >= pulse_A
    starts = np.where(rest[:-1] & pul[1:])[0] + 1

    # Debounce using min_gap_s
    pulse_starts = []
    last_t = -np.inf
    for idx in starts:
        if t[idx] - last_t >= min_gap_s:
            pulse_starts.append(idx)
            last_t = t[idx]
    if not pulse_starts:
        return []

    pulses = []
    for p_idx, idx0 in enumerate(pulse_starts, start=1):
        t0 = t[idx0]

        # Baseline voltage just before pulse: mean in [t0-pre_window_s, t0)
        pre_mask = (t >= (t0 - pre_window_s)) & (t < t0)
        if pre_mask.sum() >= 3:
            V_pre = float(np.nanmean(V[pre_mask]))
            I_pre = float(np.nanmean(I[pre_mask]))
        else:
            # fallback: use the immediate previous sample
            j = max(idx0 - 1, 0)
            V_pre = float(V[j])
            I_pre = float(I[j])

        # Estimate pulse current level using window right after start
        post_mask = (t >= t0) & (t <= (t0 + I_est_window_s))
        if post_mask.sum() >= 3:
            I_pulse_level = float(np.nanmedian(I[post_mask]))
        else:
            I_pulse_level = float(I[idx0])

        # Step current (relative to rest current)
        I_step = I_pulse_level - I_pre
        if abs(I_step) < 1e-6:
            # avoid division by ~0; skip this pulse
            continue

        # Evaluate resistances
        R = {}
        V_at = {}
        for dt_eval in eval_times:
            V_eval = float(_value_at_time(t, V, t0 + dt_eval))
            dV = V_eval - V_pre
            R[f"R_{dt_eval:g}s_ohm"] = dV / I_step
            V_at[f"V_{dt_eval:g}s_V"] = V_eval

        pulse_info = {
            "pulse_index": p_idx,
            "t0_s": float(t0),
            "V_pre_V": V_pre,
            "I_pre_A": I_pre,
            "I_pulse_A": I_pulse_level,
            "I_step_A": I_step,
            **V_at,
            **R,
            "summary": (
                f"I_step={I_step:+.3f} A, "
                f"R0.2={R.get('R_0.2s_ohm', np.nan):+.5f} Ω, "
                f"R1={R.get('R_1s_ohm', np.nan):+.5f} Ω, "
                f"R10={R.get('R_10s_ohm', np.nan):+.5f} Ω"
            ),
        }
        pulses.append(pulse_info)

    return pulses

def analyze_df_pulse(
    df_filtered: pd.DataFrame,
    soc_labels=(0.8, 0.5, 0.2),
    step_col="step_int",
    voltage_col="voltage_V",
    current_col="current_A",  # change to "currentrate" if needed
    time_col="abs_time",
):
    sections = split_into_3_sections_by_step_reset(df_filtered, step_col=step_col)

    out = {}
    for i, sec in enumerate(sections):
        soc = soc_labels[i] if i < len(soc_labels) else f"section_{i+1}"
        pulses = analyze_section_pulses(
            sec,
            voltage_col=voltage_col,
            current_col=current_col,
            time_col=time_col,
        )
        out[str(soc)] = {
            "n_rows": int(len(sec)),
            "n_pulses": int(len(pulses)),
            "pulses": pulses,
        }

    pulse_lists = {
        "R_0.2s_ohm": [],
        "R_1s_ohm": [],
        "R_10s_ohm": [],
    }

    for i, soc in enumerate(soc_labels):
        soc_key = str(soc)
        prefix = _soc_prefix(soc)
        section_info = out.get(soc_key, {"n_rows": 0, "n_pulses": 0, "pulses": []})
        pulses = section_info["pulses"]

        pulse_lists[f"{prefix}_n_rows"] = int(section_info["n_rows"])
        pulse_lists[f"{prefix}_n_pulses"] = int(section_info["n_pulses"])
        pulse_lists[f"{prefix}_R_0.2s_ohm"] = [p["R_0.2s_ohm"] for p in pulses]
        pulse_lists[f"{prefix}_R_1s_ohm"] = [p["R_1s_ohm"] for p in pulses]
        pulse_lists[f"{prefix}_R_10s_ohm"] = [p["R_10s_ohm"] for p in pulses]
        pulse_lists[f"{prefix}_t0_s"] = [p["t0_s"] for p in pulses]
        pulse_lists[f"{prefix}_I_step_A"] = [p["I_step_A"] for p in pulses]

        pulse_lists["R_0.2s_ohm"].extend(pulse_lists[f"{prefix}_R_0.2s_ohm"])
        pulse_lists["R_1s_ohm"].extend(pulse_lists[f"{prefix}_R_1s_ohm"])
        pulse_lists["R_10s_ohm"].extend(pulse_lists[f"{prefix}_R_10s_ohm"])

    return pulse_lists


def extract_pulse_sections(
    df_filtered: pd.DataFrame,
    soc_labels=(0.8, 0.5, 0.2),
    step_col="step_int",
    voltage_col="voltage_V",
    current_col="current_A",
    time_col="abs_time",
):
    """
    Extract the full pulse sections from the checkup, similar to how OCV arrays are stored.
    Each SOC block is exported as list-like arrays for time, voltage, current, and step.
    """
    sections = split_into_3_sections_by_step_reset(df_filtered, step_col=step_col)
    pulse_sections = {}

    for i, soc in enumerate(soc_labels):
        prefix = _soc_prefix(soc)
        if i >= len(sections) or sections[i].empty:
            pulse_sections[f"{prefix}_time_s"] = []
            pulse_sections[f"{prefix}_voltage_V"] = []
            pulse_sections[f"{prefix}_current_A"] = []
            pulse_sections[f"{prefix}_step_int"] = []
            continue

        sec = sections[i].copy().reset_index(drop=True)
        pulse_sections[f"{prefix}_time_s"] = _to_rel_seconds(sec[time_col]).tolist() if time_col in sec.columns else []
        pulse_sections[f"{prefix}_voltage_V"] = pd.to_numeric(sec[voltage_col], errors="coerce").tolist() if voltage_col in sec.columns else []
        pulse_sections[f"{prefix}_current_A"] = pd.to_numeric(sec[current_col], errors="coerce").tolist() if current_col in sec.columns else []
        pulse_sections[f"{prefix}_step_int"] = pd.to_numeric(sec[step_col], errors="coerce").tolist() if step_col in sec.columns else []

    return pulse_sections

# --- Example usage ---
# results = analyze_df_filtered(
#     df_filtered,
#     soc_labels=(0.8, 0.5, 0.2),
#     step_col="step_int",
#     voltage_col="voltage_V",
#     current_col="currentrate",   # <- if that's your actual column name
#     time_col="abs_time",
# )
# results
