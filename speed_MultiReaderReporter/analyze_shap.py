from __future__ import annotations
from pathlib import Path
import argparse
import ast
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import xgboost as xgb
import shap
import matplotlib.pyplot as plt


DEFAULT_OUT_DIR = Path("E:/download/jgne/out_jgne")
NOMINAL_CAPACITY_AH = 1.8


def _parse_feature_array(value) -> np.ndarray:
    """Parse a curve/list cell from a pipeline CSV into a flat float array."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float).ravel()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return np.array([], dtype=float)
    # New pipeline exports use Python-list syntax; accept older NumPy whitespace
    # array strings too.
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return np.asarray(parsed, dtype=float).ravel()
        if np.isscalar(parsed):
            return np.asarray([float(parsed)])
    except (ValueError, SyntaxError, TypeError):
        pass
    text = text.strip("[]").replace(",", " ")
    return np.fromstring(text, sep=" ", dtype=float)


def _row_label(row: pd.Series, index: int) -> str:
    time = pd.to_datetime(row.get("CU_time"), errors="coerce")
    return time.strftime("%Y-%m-%d") if not pd.isna(time) else f"checkup {index}"


def _current_rate_label(samples) -> str:
    """Label the active measured current and approximate rate at 1.8 Ah."""
    current = np.asarray(samples, dtype=float).ravel()
    active = current[np.isfinite(current) & (np.abs(current) > 0.05)]
    if not active.size:
        return "current unavailable"
    amperes = float(np.median(active))
    return f"{amperes:+.2f} A (≈{abs(amperes) / NOMINAL_CAPACITY_AH:.2g} C)"


def _limit_dva_axis(ax, values: list[np.ndarray]) -> None:
    """Cap DVA y limits at -1 and 2 when the plotted data reach them."""
    finite = [array[np.isfinite(array)] for array in values if array.size]
    finite = [array for array in finite if array.size]
    if not finite:
        return
    lower, upper = ax.get_ylim()
    if any(np.min(array) <= -1 for array in finite):
        lower = -1
    if any(np.max(array) >= 2 for array in finite):
        upper = 2
    ax.set_ylim(lower, upper)


def _ocv_dva_from_raw(cell: str, cu_time: pd.Timestamp, raw_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """Recompute only the selected checkup's saved-method OCV DVA from its ZIP."""
    if pd.isna(cu_time):
        raise ValueError("Selected checkup has no valid CU_time to locate its raw ZIP")
    from core.capacity import extract_DVA
    from core.family import resolve_family_cfg
    from loaders.csvzip_loader import infer_cell_from_path, load
    import yaml

    config_path = Path(__file__).resolve().parent / "config_homo_JGNE.yaml"
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    family_cfg = resolve_family_cfg(cfg, cell)
    steps = family_cfg["CU_steps"]
    matches = []
    for path in raw_dir.glob("*.zip"):
        if infer_cell_from_path(path) != cell:
            continue
        parts = path.name.split("=")
        stamp = pd.to_datetime(parts[4], format="%Y-%m-%d %H%M%S", errors="coerce") if len(parts) > 4 else pd.NaT
        if pd.notna(stamp):
            matches.append((stamp, path))
    matches.sort(key=lambda item: item[0])
    eligible = [(stamp, path) for stamp, path in matches if stamp <= cu_time]
    if not eligible:
        raise FileNotFoundError(f"No raw ZIP starting at or before {cu_time} for {cell} in {raw_dir}")
    zip_path = eligible[-1][1]
    records = load(zip_path, cfg, raw_dir)
    proc = str(family_cfg["procedures"]["rpt"])
    results = {}
    for direction, key in (("cha", "ocv_cha"), ("dis", "ocv_dis")):
        step = steps[key]
        step = int(step[0] if isinstance(step, (list, tuple)) else step)
        segments = []
        for record in records:
            frame = record.df
            if not {"procedure", "step_int", "abs_time", "qstep", "voltage_V", "current_A"} <= set(frame.columns):
                continue
            mask = (frame["procedure"].astype(str).str.contains(proc, case=False, regex=False, na=False)
                    & pd.to_numeric(frame["step_int"], errors="coerce").eq(step))
            segment = frame.loc[mask].sort_values("abs_time")
            if len(segment) >= 12:
                distance = abs((segment["abs_time"].iloc[0] - cu_time).total_seconds())
                segments.append((distance, segment))
        if not segments:
            print(f"[WARN] {cell}: no OCV {direction} step {step} found in {zip_path.name}")
            continue
        segment = min(segments, key=lambda item: item[0])[1]
        capacity, dva, _ = extract_DVA(segment, cell, family_cfg)
        results[direction] = (
            np.asarray(capacity).ravel(), np.asarray(dva).ravel(),
            _current_rate_label(segment["current_A"]),
        )
    return results


def visualize_one_cell(cell: str, feature_dir: Path, trace_dir: Path,
                       dest: Path, show: bool = False,
                       checkup_index: int = 0, raw_dir: Path | None = None) -> list[Path]:
    """Plot one checkup's extracted features and raw traces for one cell."""
    feature_path = feature_dir / f"{cell}.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"No feature table for {cell}: {feature_path}")
    df = pd.read_csv(feature_path)
    if df.empty:
        raise ValueError(f"Feature table is empty: {feature_path}")
    if not 0 <= checkup_index < len(df):
        raise ValueError(
            f"Checkup index {checkup_index} is out of range for {cell} "
            f"(available: 0 through {len(df) - 1})"
        )
    df = df.iloc[[checkup_index]].reset_index(drop=True)

    family = "LWHK" if cell.upper().startswith("SPEED_LWHK_") else (
        "JGNE" if cell.upper().startswith("SPEED_JGNE_") else "Unknown family"
    )
    title_prefix = f"{family} | {cell} | checkup {checkup_index} ({_row_label(df.iloc[0], checkup_index)})"

    # Isolate the selected checkup from previously generated all-checkup plots.
    dest = dest / cell / f"checkup_{checkup_index:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    labels = [_row_label(row, checkup_index) for _, row in df.iterrows()]
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, max(len(df), 1)))
    outputs: list[Path] = []

    cu_time = pd.to_datetime(df.loc[0, "CU_time"], errors="coerce")
    stamp = cu_time.strftime("%Y%m%dT%H%M%S") if pd.notna(cu_time) else f"{checkup_index:04d}"
    npz_path = trace_dir / f"{cell}_{stamp}.npz"
    trace_currents = {}
    if npz_path.exists():
        with np.load(npz_path) as traces:
            trace_currents = {
                direction: _current_rate_label(traces[key])
                for direction, key in (("cha", "current_fast_raw_cha_A"),
                                       ("dis", "current_fast_raw_dis_A"))
                if key in traces
            }

    # The OCV curves are not saved in full in the feature CSV. Loading the
    # selected checkup once also gives the measured OCV current for both ICA
    # and DVA labels.
    raw_dir = raw_dir or Path("E:/download/jgne/raw_data_csv")
    try:
        ocv_dva = _ocv_dva_from_raw(cell, cu_time, raw_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARN] {cell}: OCV DVA curve unavailable: {exc}")
        ocv_dva = {}

    def save(fig, name: str) -> None:
        path = dest / f"{cell}_{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        outputs.append(path)
        if show:
            plt.show()
        plt.close(fig)

    # Scalar capacity and derived values for the selected checkup only.
    capacity_cols = [c for c in ("cap_ocv_dis", "cap_ocv_cha", "cap_dis",
                                  "cap_cha", "cap_fast_dis", "cap_fast_cha")
                     if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    derived_cols = [c for c in ("mean_d_dqdv_m_c", "mean_d_dqdv_m_d",
                                  "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
                                  "mean_d_dqdv_h_d", "mean_d_dqdv_l_d",
                                  "throughput_cum", "soh_ocv_dis", "soh_fast_dis",
                                  "cap_ocv_minus_fast_dis_Ah", "soh_ocv_minus_fast_dis")
                    if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    if capacity_cols or derived_cols:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        for ax, cols, title, units in (
            (axes[0], capacity_cols, "Capacity", "Ah"),
            (axes[1], derived_cols, "Derived features", "Feature value"),
        ):
            values = pd.to_numeric(df.loc[0, cols], errors="coerce") if cols else pd.Series(dtype=float)
            ax.barh(cols, values, color="steelblue")
            ax.set_title(title)
            ax.set_xlabel(units)
            ax.grid(axis="x", alpha=0.25)
            ax.invert_yaxis()
        fig.suptitle(f"{title_prefix}: extracted scalar features")
        fig.tight_layout()
        save(fig, "scalar_features")

    # Show each checkup's OCV ICA curve, where the full curves are in the CSV.
    if {"Vcha", "dQdVcha"}.issubset(df.columns):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        curve_specs = [("Vcha", "dQdVcha", "Charge OCV ICA"),
                       ("Vdis", "dQdVdis", "Discharge OCV ICA")]
        plotted = False
        for ax, (vcol, icol, title) in zip(axes, curve_specs):
            if vcol not in df.columns or icol not in df.columns:
                ax.set_visible(False)
                continue
            for i, (_, row) in enumerate(df.iterrows()):
                voltage = _parse_feature_array(row.get(vcol))
                ica = _parse_feature_array(row.get(icol))
                n = min(len(voltage), len(ica))
                if n:
                    ax.plot(voltage[:n], ica[:n], color=colors[i],
                            label=labels[i], alpha=0.85)
                    direction = "cha" if vcol == "Vcha" else "dis"
                    for kind, marker in (("max", "x"), ("min", "o")):
                        px = _parse_feature_array(row.get(f"peakV_{kind}_{direction}"))
                        py = _parse_feature_array(row.get(f"peakICA_{kind}_{direction}"))
                        m = min(len(px), len(py))
                        if m:
                            ax.scatter(px[:m], py[:m], color=[colors[i]], marker=marker,
                                       s=25, alpha=0.8)
                    plotted = True
            direction = "cha" if vcol == "Vcha" else "dis"
            rate = ocv_dva[direction][2] if direction in ocv_dva else "current unavailable"
            ax.set_title(f"{title}\n{rate}")
            ax.set_xlabel("Voltage [V]")
            ax.set_ylabel("dQ/dV [Ah/V]")
            ax.grid(alpha=0.25)
        if plotted:
            axes[0].legend(fontsize=7, title="Checkup", loc="best")
            fig.suptitle(f"{title_prefix}: OCV incremental-capacity curves")
            fig.tight_layout()
            save(fig, "ocv_ica")
        else:
            plt.close(fig)

    # High-rate ICA curves are stored in the per-cell feature CSV.
    fast_specs = [("Vfast_cha", "dQdVfast_cha", "High-rate charge ICA"),
                  ("Vfast_dis", "dQdVfast_dis", "High-rate discharge ICA")]
    if any(v in df.columns and q in df.columns for v, q, _ in fast_specs):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plotted = False
        for ax, (vcol, qcol, title) in zip(axes, fast_specs):
            if vcol not in df.columns or qcol not in df.columns:
                ax.set_visible(False)
                continue
            for i, (_, row) in enumerate(df.iterrows()):
                voltage = _parse_feature_array(row.get(vcol))
                ica = _parse_feature_array(row.get(qcol))
                n = min(len(voltage), len(ica))
                if n:
                    ax.plot(voltage[:n], ica[:n], color=colors[i],
                            label=labels[i], alpha=0.85)
                    plotted = True
            direction = "cha" if vcol == "Vfast_cha" else "dis"
            ax.set_title(f"{title}\n{trace_currents.get(direction, 'current unavailable')}")
            ax.set_xlabel("Voltage [V]")
            ax.set_ylabel("dQ/dV [Ah/V]")
            ax.grid(alpha=0.25)
        if plotted:
            axes[0].legend(fontsize=7, title="Checkup", loc="best")
            fig.suptitle(f"{title_prefix}: high-rate incremental-capacity curves")
            fig.tight_layout()
            save(fig, "high_rate_ica")
        else:
            plt.close(fig)

    # The feature CSV retains OCV DVA peaks but drops its full DVA arrays.
    # Recover the selected checkup's curves using the pipeline's own method.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ocv_dva_plotted = False
    for ax, direction, title in zip(axes, ("cha", "dis"),
                                    ("Charge OCV DVA", "Discharge OCV DVA")):
        plotted_dva = []
        if direction in ocv_dva:
            capacity, dva, _ = ocv_dva[direction]
            n = min(len(capacity), len(dva))
            if n:
                ax.plot(capacity[:n], dva[:n], color=colors[0], label=labels[0])
                plotted_dva.append(dva[:n])
                ocv_dva_plotted = True
        for i, (_, row) in enumerate(df.iterrows()):
            for kind, marker in (("max", "x"), ("min", "o")):
                capacity = _parse_feature_array(row.get(f"peakQ_{kind}_{direction}"))
                dva = _parse_feature_array(row.get(f"peakDVA_{kind}_{direction}"))
                n = min(len(capacity), len(dva))
                if n:
                    ax.scatter(capacity[:n], dva[:n], color=[colors[i]], marker=marker,
                               s=35, label=labels[i] if kind == "max" else None)
                    plotted_dva.append(dva[:n])
                    ocv_dva_plotted = True
        rate = ocv_dva[direction][2] if direction in ocv_dva else "current unavailable"
        ax.set_title(f"{title}\n{rate}")
        ax.set_xlabel("Capacity [Ah]")
        ax.set_ylabel("dV/dQ [V/Ah]")
        ax.grid(alpha=0.25)
        _limit_dva_axis(ax, plotted_dva)
    if ocv_dva_plotted:
        axes[0].legend(fontsize=7, title="Checkup", loc="best")
        fig.suptitle(f"{title_prefix}: OCV differential voltage (dV/dQ)"
                     if ocv_dva else f"{title_prefix}: OCV dV/dQ peak positions (raw curves unavailable)")
        fig.tight_layout()
        save(fig, "ocv_dva")
    else:
        plt.close(fig)

    # High-rate DVA arrays are retained in the feature CSV as full curves.
    fast_dva_specs = [("Qfast_cha", "dVdQfast_cha", "High-rate charge DVA"),
                      ("Qfast_dis", "dVdQfast_dis", "High-rate discharge DVA")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fast_dva_plotted = False
    for ax, (qcol, dvcol, title) in zip(axes, fast_dva_specs):
        plotted_dva = []
        for i, (_, row) in enumerate(df.iterrows()):
            capacity = _parse_feature_array(row.get(qcol))
            dva = _parse_feature_array(row.get(dvcol))
            n = min(len(capacity), len(dva))
            if n:
                ax.plot(capacity[:n], dva[:n], color=colors[i], label=labels[i], alpha=0.85)
                plotted_dva.append(dva[:n])
                fast_dva_plotted = True
        direction = "cha" if qcol == "Qfast_cha" else "dis"
        ax.set_title(f"{title}\n{trace_currents.get(direction, 'current unavailable')}")
        ax.set_xlabel("Capacity [Ah]")
        ax.set_ylabel("dV/dQ [V/Ah]")
        ax.grid(alpha=0.25)
        _limit_dva_axis(ax, plotted_dva)
    if fast_dva_plotted:
        axes[0].legend(fontsize=7, title="Checkup", loc="best")
        fig.suptitle(f"{title_prefix}: high-rate differential-voltage curves (dV/dQ)")
        fig.tight_layout()
        save(fig, "high_rate_dva")
    else:
        plt.close(fig)

    # Plot the original pulse and high-rate voltage/current samples from NPZ.
    trace_specs = [
        ("pulse_time_s", "pulse_voltage_V", "pulse_current_A", "Pulse"),
        ("time_fast_raw_cha_s", "Vfast_raw_cha", "current_fast_raw_cha_A", "Fast charge"),
        ("time_fast_raw_dis_s", "Vfast_raw_dis", "current_fast_raw_dis_A", "Fast discharge"),
    ]
    for i, (_, row) in enumerate(df.iterrows()):
        if not npz_path.exists():
            continue
        with np.load(npz_path) as traces:
            available = [(t, v, cur, title) for t, v, cur, title in trace_specs
                         if t in traces and v in traces and cur in traces
                         and traces[t].size and traces[v].size and traces[cur].size]
            if not available:
                continue
            fig, axes = plt.subplots(len(available), 1,
                                     figsize=(10, 3.5 * len(available)), squeeze=False)
            for ax, (tcol, vcol, ccol, title) in zip(axes[:, 0], available):
                t, voltage, current = traces[tcol], traces[vcol], traces[ccol]
                n = min(len(t), len(voltage), len(current))
                ax.plot(t[:n], voltage[:n], color="tab:blue", label="Voltage [V]")
                ax.set_ylabel("Voltage [V]", color="tab:blue")
                ax.set_title(f"{title}\n{_current_rate_label(current[:n])}")
                ax.grid(alpha=0.25)
                ax2 = ax.twinx()
                ax2.plot(t[:n], current[:n], color="tab:red", alpha=0.75,
                         label="Current [A]")
                ax2.set_ylabel("Current [A]", color="tab:red")
                ax.set_xlabel("Elapsed time [s]")
            fig.suptitle(f"{title_prefix}: raw checkup traces — {labels[i]}")
            fig.tight_layout()
            save(fig, f"raw_traces_{checkup_index:02d}")

    if not outputs:
        raise ValueError(f"No plottable feature columns or traces found for {cell} in {feature_path}")
    print(f"[OK] wrote {len(outputs)} plot(s) to {dest}")
    return outputs


# -----------------------------
# Helpers
# -----------------------------
def filter_exp_first_row(df: pd.DataFrame, exp_conds: list[str]) -> pd.Series:
    """Experimental conditions from row 0 of the raw CSV."""
    return df.loc[0, exp_conds]


def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric; invalid -> NaN."""
    return df.apply(pd.to_numeric, errors="coerce")


def coerce_series_to_float_1d(s: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series to a 1D float array.
    Handles values like "[8.366986e-3]" by stripping brackets.
    """
    s2 = s.copy()

    # Convert to str, strip whitespace
    s2 = s2.astype(str).str.strip()

    # Remove a single surrounding bracket pair: "[0.12]" -> "0.12"
    s2 = s2.str.replace(r"^\[|\]$", "", regex=True)

    # Convert to numeric
    s2 = pd.to_numeric(s2, errors="coerce")

    return s2.to_numpy(dtype=float).reshape(-1)


def load_and_interpolate(
    df: pd.DataFrame,
    target_soh: float,
    interpolation_typ: str = "weeks",
) -> pd.DataFrame | None:
    """
    Interpolate feature columns over a reference axis until SOH reaches target_soh.
    Returns interpolated dataframe including SOH for safe indexing.
    """
    df = df.copy()

    # Fill first row NaNs with 0 (as in your original)
    df.iloc[0] = df.iloc[0].fillna(0)

    # Compute SOH
    if "cap_ocv_dis" not in df.columns:
        raise ValueError("cap_ocv_dis is required to compute SOH.")
    df["SOH"] = df["cap_ocv_dis"] / df["cap_ocv_dis"].iloc[0]

    # Choose reference column
    if interpolation_typ == "weeks":
        ref_col = "weeks"
    elif interpolation_typ == "throughput":
        ref_col = "throughput"
    else:
        raise ValueError("interpolation_typ must be 'weeks' or 'throughput'")

    # Keep first row + all rows where ref != 0 (your original logic)
    mask = (df.index == 0) | (df[ref_col] != 0)
    df = df[mask].reset_index(drop=True)

    # Coerce relevant columns to numeric
    df = coerce_numeric_df(df)

    # If reference or SOH contains NaNs, interpolation is impossible
    if df[ref_col].isna().any() or df["SOH"].isna().any():
        return None

    reference = df[ref_col].to_numpy(dtype=float)
    soh = df["SOH"].to_numpy(dtype=float)

    # Feature columns to interpolate (exclude ref and SOH)
    feature_cols = [c for c in df.columns if c not in [ref_col, "SOH"]]
    feature_mat = df[feature_cols].to_numpy(dtype=float)

    # Interpolate reference point where SOH hits target_soh
    # SOH typically decreases => use reversed arrays so "x" is increasing for np.interp
    try:
        ref_at_target = np.interp(target_soh, soh[::-1], reference[::-1])
    except Exception:
        return None

    # Create a reference grid: 6 points from 0..ref_at_target, then continue with same spacing
    new_ref_cut = np.linspace(0.0, float(ref_at_target), 6)
    spacing = float(np.mean(np.diff(new_ref_cut))) if len(new_ref_cut) > 1 else 0.0
    if spacing <= 0:
        return None

    remaining = []
    current = float(ref_at_target)
    iters = 0
    max_iters = 5000

    while current + spacing <= float(reference[-1]):
        current += spacing
        remaining.append(current)
        iters += 1
        if iters >= max_iters:
            return None

    new_ref = np.concatenate([new_ref_cut, np.array(remaining, dtype=float)])

    # Interpolate each feature col
    interpolated_cols = []
    for j in range(feature_mat.shape[1]):
        interpolated_cols.append(np.interp(new_ref, reference, feature_mat[:, j]))

    interpolated = pd.DataFrame(
        np.stack(interpolated_cols, axis=1),
        columns=feature_cols
    )
    interpolated[ref_col] = new_ref

    # ALSO interpolate SOH so you can pick the nearest row
    interpolated["SOH"] = np.interp(new_ref, reference, soh)

    return interpolated


def build_experiment_table(dir_path: Path, target_soh: float) -> pd.DataFrame:
    exp_conds = ["soc_start", "soc_end", "pause_h", "has_pulse", "low_soc", "temp_C"]
    rows = []

    # The current pipeline stores JGNE temperature in the label-derived
    # condition overview (`temp_C`), not in each cell's feature CSV.
    condition_by_cell: dict[str, pd.DataFrame] = {}
    try:
        from document_conditions import DEFAULT_RAW_DIR, build_table as build_condition_table

        condition_path = dir_path.parent / "condition_overview_homocomp_jgne.csv"
        if condition_path.exists():
            conditions = pd.read_csv(condition_path)
        else:
            conditions = build_condition_table(DEFAULT_RAW_DIR, "homocomp_jgne", None)
        if not conditions.empty and "cell" in conditions.columns:
            condition_by_cell = {
                str(cell): group for cell, group in conditions.groupby("cell", dropna=False)
            }
    except Exception as exc:
        print(f"[WARN] Could not load label-derived condition table for SHAP temperature: {exc}")

    def conditions_for_cell(cell_name: str, df: pd.DataFrame) -> dict[str, float]:
        """Match a program-label condition row to this cell's extracted conditions."""
        candidates = condition_by_cell.get(cell_name)
        if candidates is None or candidates.empty:
            return {}
        feature = df.iloc[0]
        match_pairs = [
            ("soc_start", "soc_start"), ("soc_end", "soc_end"),
            ("pause", "pause_h"), ("c_rate_chg", "c_rate_chg"),
            ("c_rate_dchg", "c_rate_dchg"), ("dyn", "dyn"),
            ("low_soc", "low_soc"), ("has_pulse", "has_pulse"),
        ]
        scores = []
        for _, condition in candidates.iterrows():
            score = 0
            for feature_col, condition_col in match_pairs:
                if feature_col not in df.columns or condition_col not in candidates.columns:
                    continue
                left = pd.to_numeric(pd.Series([feature.get(feature_col)]), errors="coerce").iloc[0]
                right = pd.to_numeric(pd.Series([condition.get(condition_col)]), errors="coerce").iloc[0]
                if pd.notna(left) and pd.notna(right) and np.isclose(left, right):
                    score += 1
            scores.append(score)
        best = candidates.iloc[int(np.argmax(scores))]
        result = {}
        for col in exp_conds:
            # SOC endpoints are already parsed into each feature row. The other
            # conditions come from the separate label-derived table.
            if col in {"soc_start", "soc_end"} and col in df.columns:
                result[col] = pd.to_numeric(pd.Series([feature.get(col)]), errors="coerce").iloc[0]
            else:
                result[col] = pd.to_numeric(pd.Series([best.get(col)]), errors="coerce").iloc[0]
        return result

    for csv_file in dir_path.glob("*.csv"):
        # This model uses the JGNE condition table and response features. Keep
        # the distinct LWHK protocol out of the training pool.
        if not csv_file.stem.upper().startswith("SPEED_JGNE_"):
            continue
        df = pd.read_csv(csv_file)

        # weeks from CU_time
        df = df.copy()
        df["weeks"] = (
            pd.to_datetime(df["CU_time"]) - pd.to_datetime(df["CU_time"]).iloc[0]
        ).dt.total_seconds() / (7 * 24 * 3600)

        cell_name = csv_file.stem

        label_conditions = conditions_for_cell(cell_name, df)

        # Columns needed for interpolation + target
        keep = [
            "weeks",
            "cap_ocv_dis",
            "mean_d_dqdv_m_c",   # target you asked for
            "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d",
            "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c",
            "mean_d_dqdv_l_c",
        ]
        # if some columns are missing, skip this file
        missing = [c for c in keep if c not in df.columns]
        if missing:
            continue

        df_filter = df[keep].copy()

        interpolated_df = load_and_interpolate(df_filter, target_soh, interpolation_typ="weeks")
        if interpolated_df is None or interpolated_df.empty:
            continue

        # Find row closest to target_soh
        idx = (interpolated_df["SOH"] - target_soh).abs().idxmin()
        row_at_soh = interpolated_df.loc[idx]

        # Conditions are a mix of pipeline-parsed SOC columns and the
        # condition-table values matched back to the cell's cycling label.
        exp_row = pd.Series(label_conditions, dtype=float)
        combined = pd.concat(
            [pd.Series({"cell_name": cell_name}), exp_row, row_at_soh],
            axis=0
        )
        rows.append(combined)

    return pd.DataFrame(rows).reset_index(drop=True)


# -----------------------------
# Train + SHAP (KernelExplainer style)
# -----------------------------
def train_and_shap_kernel_multi(
    df_exp: pd.DataFrame,
    exp_conds: List[str] = None,
    target_cols: List[str] = None,
    test_size: float = 0.1,
    random_state: int = 42,
    n_show: int = 500,
    n_iter_search: int = 60,     # increase for better search
    cv_splits: int = 5,
    early_stopping_rounds: int = 200,
) -> Dict[str, Dict[str, Any]]:
    """
    Train one XGBoost model per target column using RandomizedSearchCV + early stopping,
    MinMaxScaler instead of StandardScaler, and SHAP KernelExplainer per target.
    """

    if exp_conds is None:
        exp_conds = ["soc_start", "soc_end", "pause_h", "has_pulse", "low_soc", "temp_C"]

    if target_cols is None:
        target_cols = ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_l_c"]

    # ---- Build X ----
    X = df_exp[exp_conds].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    # ---- Feature engineering: SOC + DOD ----
    if {"soc_start", "soc_end"}.issubset(X.columns):
        X["soc"] = 0.5 * (X["soc_start"] + X["soc_end"])
        X["dod"] = X["soc_end"] - X["soc_start"]
        X = X.drop(columns=["soc_start", "soc_end"])
        exp_conds = ["soc", "dod"] + [c for c in exp_conds if c not in {"soc_start", "soc_end"}]

    # ---- Build y's (coerce each target) ----
    y_dict = {}
    for tcol in target_cols:
        y_dict[tcol] = coerce_series_to_float_1d(df_exp[tcol])

    # ---- Keep only rows where ALL targets are valid ----
    valid_mask = np.ones(len(df_exp), dtype=bool)
    for tcol in target_cols:
        valid_mask &= ~np.isnan(y_dict[tcol])

    X = X.loc[valid_mask].reset_index(drop=True)
    for tcol in target_cols:
        y_dict[tcol] = y_dict[tcol][valid_mask]

    # Fill missing X with median, cast float
    X = X.fillna(X.median(numeric_only=True)).astype(float)

    # ---- Split ONCE so all targets use same split ----
    idx_all = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=test_size, random_state=random_state
    )

    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)

    # ---- MinMax scale ----
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Make DataFrames so SHAP plots show feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=exp_conds)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=exp_conds)

    # ---- Create a validation split for early stopping (from train only) ----
    X_tr, X_val, tr_idx, val_idx = train_test_split(
        X_train_scaled,
        np.arange(X_train_scaled.shape[0]),
        test_size=0.2,
        random_state=random_state,
    )

    # Base model (tree method "hist" tends to be faster on CPU)
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state, # big ceiling; early stopping will pick best iteration
    )

    # A solid search space for your feature count (5 features) and typical tabular regression
    param_dist = {
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": np.linspace(0.005, 0.15, 30).tolist(),
        "min_child_weight": [1, 2, 5, 10, 20, 40],
        "subsample": np.linspace(0.5, 1.0, 11).tolist(),
        "colsample_bytree": np.linspace(0.5, 1.0, 11).tolist(),
        "gamma": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        "reg_alpha": [0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0],
        "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results: Dict[str, Dict[str, Any]] = {}

    for tcol in target_cols:
        y = y_dict[tcol].astype(float)
        y_train = y[idx_train]
        y_test = y[idx_test]

        # Map y_train onto train/val split indices created above
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        # RandomizedSearchCV optimizes CV score; early stopping is handled in fit
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            verbose=1,
            random_state=random_state,
            n_jobs=-1,
        )

        # Fit search on FULL training data (X_train_scaled), but with early stopping using (X_val, y_val)
        # We pass eval_set via **fit_params**.
        search.fit(
            X_train_scaled,  # keep as np array for XGBoost
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_model = search.best_estimator_

        # ---- Evaluate on held-out test ----
        y_pred = best_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        print(f"\nTarget: {tcol}")
        print("Best params:", search.best_params_)
        print("R²:", r2)
        print("RMSE:", rmse)

        # ---- SHAP KernelExplainer (per target) ----
        # Background: use a subset for speed if training set is large
        X_background = X_train_scaled_df  # DataFrame with column names
        X_sample = X_train_scaled_df

        def predict_fn(x):
            # KernelExplainer may pass a numpy array even if background is DF
            x_arr = np.asarray(x)
            return best_model.predict(x_arr)

        explainer = shap.KernelExplainer(predict_fn, X_background)

        n_plot = min(n_show, X_sample.shape[0])
        shap_values = explainer.shap_values(X_sample.iloc[:n_plot])

        feature_names = exp_conds

        # --- beeswarm ---
        shap.summary_plot(
            shap_values,
            X_sample.iloc[:n_plot],
            show=False
        )
        plt.title(f"SHAP summary (beeswarm) — target: {tcol}")
        plt.tight_layout()
        plt.show()

        # --- bar ---
        shap.summary_plot(
            shap_values,
            X_sample.iloc[:n_plot],
            plot_type="bar",
            show=False
        )
        plt.title(f"SHAP feature importance (bar) — target: {tcol}")
        plt.tight_layout()
        plt.show()

        results[tcol] = {
            "model": best_model,
            "best_params": search.best_params_,
            "explainer": explainer,
            "shap_values": shap_values,
            "scaler": scaler,
            "X_train": X_train,
            "X_train_scaled": X_train_scaled,
            "X_test": X_test,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "metrics": {"r2": r2, "rmse": rmse},
        }

    return results


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize one cell's extracted features or run the multi-cell SHAP analysis."
    )
    parser.add_argument("--cell", help="cell ID for one-cell feature plots, e.g. SPEED_LWHK_cycle_45")
    parser.add_argument("--run-shap", action="store_true",
                        help="run the multi-cell SHAP model instead of one-cell visualization")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"pipeline output root (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--dest", type=Path, default=None,
                        help="plot output folder (default: <out-dir>/cell_visualizations)")
    parser.add_argument("--show", action="store_true", default=None,
                        help="display plots interactively as well as saving them (on by default for visualization)")
    parser.add_argument("--target-soh", type=float, default=0.998,
                        help="target SoH for the multi-cell SHAP run (default: 0.998)")
    parser.add_argument("--shap-dir", type=Path, default=None,
                        help="feature CSV folder for SHAP (default: <out-dir>/cell_feature)")
    args = parser.parse_args()

    if not args.run_shap:
        # A normal IDE Run opens the plots. Default to a representative JGNE
        # cell because this analysis file's model targets are JGNE-derived.
        cell = args.cell or "SPEED_JGNE_cycle_61"
        show = True if args.show is None else args.show
        visualize_one_cell(
            cell=cell,
            feature_dir=args.out_dir / "cell_feature",
            trace_dir=args.out_dir / "raw_checkup_traces",
            dest=args.dest or (args.out_dir / "cell_visualizations"),
            show=show,
        )
    else:
        dir_path = args.shap_dir or (args.out_dir / "cell_feature")
        df_exp = build_experiment_table(dir_path, args.target_soh)
        print("df_exp shape:", df_exp.shape)
        if df_exp.empty:
            raise SystemExit(f"No SHAP input rows were built from {dir_path} at SoH {args.target_soh}.")
        targets = ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_l_c"]
        missing = [col for col in targets if col not in df_exp.columns]
        if missing:
            raise SystemExit(f"SHAP targets missing from input table: {missing}")
        print(df_exp[["cell_name", *targets]].head())
        train_and_shap_kernel_multi(df_exp, target_cols=targets)
