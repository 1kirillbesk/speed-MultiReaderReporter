from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

DEFAULT_VOLTAGE_WINDOWS: tuple[tuple[float, float], ...] = ((1.9, 2.1), (3.55, 3.65))

@dataclass
class GroupCfg:
    pause_label: str = "PAU"
    min_points: int = 5
    require_step_change: bool = True
    voltage_low: float | None = None
    voltage_high: float | None = None
    voltage_windows: tuple[tuple[float, float], ...] = DEFAULT_VOLTAGE_WINDOWS

def _where_step_changes(step: pd.Series) -> np.ndarray:
    s = pd.to_numeric(step, errors="coerce")
    s = s.fillna(np.nan)
    idx = np.where(s.shift(-1).to_numpy() != s.to_numpy())[0]  # indices where value changes at i->i+1
    return idx

def split_checkup_into_groups(df: pd.DataFrame, cfg: GroupCfg) -> list[tuple[pd.DataFrame, str]]:
    if "state" not in df.columns or "step_int" not in df.columns:
        return [(df, "G1 (full)")]   # cannot segment robustly

    N = len(df)
    if N == 0:
        return []

    # candidate starts: positions with pause label that coincide with a step change boundary
    state_vals_raw = df["state"].astype(str).fillna("")
    state_vals = np.array([s.strip().upper() for s in state_vals_raw])
    pause_label_norm = str(cfg.pause_label).strip().upper()

    if cfg.require_step_change:
        # boundaries indicate the first index of a new step (including 0)
        change_idx = _where_step_changes(df["step_int"])
        boundaries = np.concatenate(([0], change_idx + 1)) if change_idx.size else np.array([0])
        boundaries = np.unique(np.clip(boundaries, 0, N - 1))
    else:
        # when step enforcement is disabled, allow any row as potential start
        boundaries = np.arange(N, dtype=int)

    if "voltage_V" in df.columns:
        voltage_vals = pd.to_numeric(df["voltage_V"], errors="coerce").to_numpy()
    else:
        voltage_vals = np.full(N, np.nan)

    def _is_valid_voltage(v: float) -> bool:
        if not np.isfinite(v):
            return False
        windows = getattr(cfg, "voltage_windows", ()) or ()
        if windows:
            for low, high in windows:
                lo = -np.inf if low is None else float(low)
                hi = np.inf if high is None else float(high)
                if lo > hi:
                    lo, hi = hi, lo
                if lo <= v <= hi:
                    return True
            return False

        low_set = cfg.voltage_low is not None
        high_set = cfg.voltage_high is not None
        if not low_set and not high_set:
            return True

        low = cfg.voltage_low if low_set else -np.inf
        high = cfg.voltage_high if high_set else np.inf
        return v < low or v > high

    starts = []
    for idx in boundaries:
        if idx >= N:
            continue
        if state_vals[idx] == pause_label_norm:
            v = voltage_vals[idx] if idx < len(voltage_vals) else np.nan
            if not _is_valid_voltage(v):
                continue
            if not starts or idx != starts[-1]:
                starts.append(int(idx))

    if not starts:
        return [(df.reset_index(drop=True), "G1 (full)")]

    # ensure coverage from the first valid pause onwards
    ends = starts[1:] + [N]

    # build segments and clean up tiny ones
    segs = []
    buffered = None
    for k, (a, b) in enumerate(zip(starts, ends), start=1):
        a = max(0, min(a, N))
        b = max(a+1, min(b, N))
        seg = df.iloc[a:b].copy()
        if buffered is not None:
            seg = pd.concat([buffered, seg])
            buffered = None

        if len(seg) < cfg.min_points:
            buffered = seg
            continue

        step_vals = pd.to_numeric(seg["step_int"], errors="coerce")
        finite = step_vals.dropna()
        if finite.empty:
            label = f"G{k}"
        else:
            label = f"G{k} (steps {int(finite.min())}–{int(finite.max())})"
        segs.append((seg.reset_index(drop=True), label))

    if buffered is not None:
        if segs:
            last_df, last_label = segs[-1]
            merged = pd.concat([last_df, buffered]).reset_index(drop=True)
            segs[-1] = (merged, last_label)
        else:
            seg = buffered.reset_index(drop=True)
            step_vals = pd.to_numeric(seg.get("step_int"), errors="coerce")
            finite = step_vals.dropna()
            label = "G1" if finite.empty else f"G1 (steps {int(finite.min())}–{int(finite.max())})"
            segs.append((seg, label))

    if not segs:
        return [(df.reset_index(drop=True), "G1 (full)")]
    return segs
# --- config-driven wrapper helpers ---

@dataclass
class PreparedGrouping:
    mode: str                      # off | report | plot | both
    do_report: bool
    do_plots: bool
    cfg: GroupCfg
    max_points_per_segment: int

def _parse_voltage_windows(value) -> tuple[tuple[float, float], ...]:
    if value is None:
        return DEFAULT_VOLTAGE_WINDOWS

    windows: list[tuple[float, float]] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            low_raw, high_raw = item[0], item[1]
            try:
                low = float(low_raw) if low_raw is not None else None
            except (TypeError, ValueError):
                low = None
            try:
                high = float(high_raw) if high_raw is not None else None
            except (TypeError, ValueError):
                high = None

            if low is None and high is None:
                continue

            if low is not None and high is not None and high < low:
                low, high = high, low

            windows.append((low, high))

    return tuple(windows)


def prepare_grouping(global_cfg: dict) -> PreparedGrouping | None:
    """
    Read checkup_grouping section from config and return a PreparedGrouping.
    Returns None when grouping is fully disabled.
    """
    grp = (global_cfg or {}).get("checkup_grouping", {}) or {}

    # Unified mode; backward-compat with old keys
    mode = str(grp.get("mode", "")).lower().strip()
    if mode not in ("off", "report", "plot", "both"):
        enabled = bool(grp.get("enabled", False) or grp.get("plot_segments", False))
        mode = "both" if grp.get("plot_segments", False) and enabled else ("report" if enabled else "off")

    if mode == "off":
        return None

    voltage_low = grp.get("voltage_low")
    voltage_high = grp.get("voltage_high")
    def _to_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    voltage_windows = _parse_voltage_windows(grp.get("voltage_windows"))
    if not voltage_windows and grp.get("voltage_windows") is None:
        voltage_windows = DEFAULT_VOLTAGE_WINDOWS

    groupcfg_kwargs = dict(
        pause_label=str(grp.get("pause_label", "PAU")),
        min_points=int(grp.get("min_points", 5)),
        require_step_change=bool(grp.get("require_step_change", True)),
        voltage_low=_to_float(voltage_low),
        voltage_high=_to_float(voltage_high),
    )

    gcfg = GroupCfg(**groupcfg_kwargs)
    try:
        setattr(gcfg, "voltage_windows", voltage_windows)
    except Exception:
        # Some legacy implementations of GroupCfg might not allow attribute mutation,
        # in which case we leave the default behavior untouched.
        pass
    return PreparedGrouping(
        mode=mode,
        do_report=(mode in ("report", "both")),
        do_plots=(mode in ("plot", "both")),
        cfg=gcfg,
        max_points_per_segment=int(grp.get("plot_max_points_per_segment", 5000)),
    )

def compute_grouped_segments(
    checkup_list: list[tuple[pd.DataFrame, str]],
    gcfg: GroupCfg,
) -> tuple[list[tuple[str, list[tuple[pd.DataFrame, str]]]], list[tuple[pd.DataFrame, str, str]]]:
    """
    Compute segments for all checkups.
    Returns:
      per_run: [(run_label, [(df_seg, group_label), ...]), ...]
      flat:    [(df_seg, run_label, group_label), ...]   # useful for reports
    """
    per_run: list[tuple[str, list[tuple[pd.DataFrame, str]]]] = []
    flat: list[tuple[pd.DataFrame, str, str]] = []

    for df_chk, lbl_chk in checkup_list:
        label_norm = str(lbl_chk).lower()
        if "glu" in label_norm or "aer" in label_norm:
            segs = [(df_chk.reset_index(drop=True), "G1 (full)")]
        else:
            segs = split_checkup_into_groups(df_chk, gcfg)
        per_run.append((lbl_chk, segs))
        for df_seg, grp_label in segs:
            flat.append((df_seg, lbl_chk, grp_label))

    return per_run, flat
