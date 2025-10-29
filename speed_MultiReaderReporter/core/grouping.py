from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class GroupCfg:
    pause_label: str = "PAU"
    min_points: int = 5
    require_step_change: bool = True
    voltage_low: float = 2.0
    voltage_high: float = 3.6

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
    state_vals = df["state"].astype(str).fillna("").to_numpy()

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

    starts = []
    for idx in boundaries:
        if idx >= N:
            continue
        if state_vals[idx] == cfg.pause_label:
            v = voltage_vals[idx] if idx < len(voltage_vals) else np.nan
            if not np.isfinite(v) or not (v < cfg.voltage_low or v > cfg.voltage_high):
                continue
            if not starts or idx != starts[-1]:
                starts.append(int(idx))

    if not starts:
        return [(df.reset_index(drop=True), "G1 (full)")]

    # ensure coverage from the first valid pause onwards
    ends = starts[1:] + [N]

    # build segments and clean up tiny ones
    segs = []
    for k, (a, b) in enumerate(zip(starts, ends), start=1):
        a = max(0, min(a, N))
        b = max(a+1, min(b, N))
        seg = df.iloc[a:b].copy()
        if len(seg) < cfg.min_points:
            # merge forward if possible, else skip
            if segs:
                # append to previous
                prev = segs[-1][0]
                segs[-1] = (pd.concat([prev, seg]).reset_index(drop=True), segs[-1][1])
            else:
                # if first is too short, just skip
                continue
        else:
            step_vals = pd.to_numeric(seg["step_int"], errors="coerce")
            finite = step_vals.dropna()
            if finite.empty:
                label = f"G{k}"
            else:
                label = f"G{k} (steps {int(finite.min())}–{int(finite.max())})"
            segs.append((seg.reset_index(drop=True), label))

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

    gcfg = GroupCfg(
        pause_label=str(grp.get("pause_label", "PAU")),
        min_points=int(grp.get("min_points", 5)),
        require_step_change=bool(grp.get("require_step_change", True)),
    )
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
