from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from speed_MultiReaderReporter.core.capacity import extract_DVA, extract_ICA, window_delta_mean_var


DATA_ROOT = Path(r"C:\Users\Victus\PycharmProjects\Exp_Sam")
SUMMARY_SUFFIX = "_summary.csv"
QOCV_DIRNAME = "qOCV_CHA"
QOCV_FILE_GLOB = "qOCV_CHA_*.csv"
QOCV_FEATURE_SUFFIX = "_summary_with_qocv_features.csv"
PLOT_SUFFIX = "_dQdVcha_vs_Vcha.png"
V_2 = 3.35

# Columns that define whether a summary row is still a valid qOCV checkpoint.
CORE_SUMMARY_COLUMNS = [
    "equivalent_cycle_count",
    "total_throughput",
    "resistance",
    "charge_Ah",
    "discharge_Ah",
    "capacity",
]


@dataclass
class CellResult:
    cell_name: str
    summary_file: Path
    summary_feature_file: Path
    qocv_dir: Path
    valid_rows_until_nan: int
    matched_qocv_files: int
    unmatched_summary_rows: int
    extra_qocv_files: int
    qocv_point_count: int
    curve_table: pd.DataFrame
    dqdv_plot_path: Path


def _extract_numeric_suffix(file_path: Path) -> int:
    match = re.search(r"(\d+)$", file_path.stem)
    if not match:
        return -1
    return int(match.group(1))


def _find_summary_files(data_root: Path) -> Iterable[Path]:
    return sorted(data_root.rglob(f"*{SUMMARY_SUFFIX}"))


def _count_valid_rows_until_nan(summary_df: pd.DataFrame) -> int:
    usable_cols = [c for c in CORE_SUMMARY_COLUMNS if c in summary_df.columns]
    if not usable_cols:
        raise ValueError(
            "Could not find any core summary columns. "
            f"Expected at least one of: {CORE_SUMMARY_COLUMNS}"
        )

    valid_mask = summary_df[usable_cols].notna().all(axis=1)
    for idx, is_valid in enumerate(valid_mask.tolist()):
        if not is_valid:
            return idx
    return len(summary_df)


def _load_qocv_files(qocv_dir: Path) -> list[Path]:
    return sorted(
        qocv_dir.glob(QOCV_FILE_GLOB),
        key=_extract_numeric_suffix,
    )


def _find_col_case_insensitive(columns: Iterable[str], preferred: str) -> str | None:
    p = preferred.lower()
    for c in columns:
        if c.lower() == p:
            return c
    return None


def _to_capacity_input_df(qocv_df: pd.DataFrame) -> pd.DataFrame:
    q_col = _find_col_case_insensitive(qocv_df.columns, "AhStep_CHA")
    v_col = _find_col_case_insensitive(qocv_df.columns, "qOCV_CHA")
    if q_col is None or v_col is None:
        raise ValueError("qOCV file must contain AhStep_CHA and qOCV_CHA columns")

    out = pd.DataFrame(
        {
            "qstep": pd.to_numeric(qocv_df[q_col], errors="coerce"),
            "voltage_V": pd.to_numeric(qocv_df[v_col], errors="coerce"),
        }
    ).dropna()

    if out.empty:
        raise ValueError("No valid points in qOCV curve after numeric conversion")

    out = out.sort_values("voltage_V").drop_duplicates(subset="voltage_V", keep="first")
    out["current_A"] = 1.0
    out = out.reset_index(drop=True)
    return out


def _array_to_csv_string(values: np.ndarray | list[float]) -> str:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return "[]"
    return np.array2string(arr, separator=" ", max_line_width=10_000)


def _qocv_cfg_from_curve(df_cap: pd.DataFrame) -> dict:
    v = df_cap["voltage_V"].to_numpy(dtype=float)
    return {
        "voltage": {
            "downlim": float(np.nanmin(v)),
            "uplim": float(np.nanmax(v)),
        }
    }


def _plot_ica_curves(
    cell_name: str,
    vcha_list: list[np.ndarray],
    dqdv_cha_list: list[np.ndarray],
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted = 0
    for i, (v_arr, dqdv_arr) in enumerate(zip(vcha_list, dqdv_cha_list)):
        if v_arr.size == 0 or dqdv_arr.size == 0:
            continue
        n = min(v_arr.size, dqdv_arr.size)
        ax.plot(v_arr[:n], dqdv_arr[:n], linewidth=0.9, alpha=0.35, label=f"row_{i:03d}")
        plotted += 1

    ax.set_xlabel("V_cha")
    ax.set_ylabel("dQdV_cha")
    ax.set_title(f"{cell_name}: dQdV_cha vs V_cha")
    ax.grid(True, alpha=0.3)
    if 0 < plotted <= 12:
        ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()

    output_path = output_dir / f"{cell_name}{PLOT_SUFFIX}"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def analyze_cell(summary_file: Path) -> CellResult:
    summary_df = pd.read_csv(summary_file)
    valid_rows_until_nan = _count_valid_rows_until_nan(summary_df)

    cell_dir = summary_file.parent
    qocv_dir = cell_dir / QOCV_DIRNAME
    if not qocv_dir.exists():
        raise FileNotFoundError(f"Missing qOCV folder: {qocv_dir}")

    qocv_files = _load_qocv_files(qocv_dir)
    matched_count = min(valid_rows_until_nan, len(qocv_files))

    selected_summary = summary_df.iloc[:matched_count].reset_index(drop=True)
    curve_frames: list[pd.DataFrame] = []
    qocv_point_count = 0
    vcha_list: list[np.ndarray] = []
    dqdv_cha_list: list[np.ndarray] = []
    q_intcha_list: list[np.ndarray] = []
    qcha_list: list[np.ndarray] = []
    dvdq_cha_list: list[np.ndarray] = []
    v_intcha_list: list[np.ndarray] = []

    for idx in range(matched_count):
        qocv_file = qocv_files[idx]
        qocv_df = pd.read_csv(qocv_file)
        qocv_point_count += len(qocv_df)
        cell_name = summary_file.name.removesuffix(SUMMARY_SUFFIX)

        try:
            cap_df = _to_capacity_input_df(qocv_df)
            cfg = _qocv_cfg_from_curve(cap_df)
            v_cha, dqdv_cha, q_intcha = extract_ICA(cap_df, cell_name, cfg)
            q_cha, dvdq_cha, v_intcha = extract_DVA(cap_df, cell_name, cfg)
        except Exception as exc:  # keep processing if one qOCV file is malformed
            print(f"[WARN] {cell_name} {qocv_file.name}: feature extraction failed ({exc})")
            v_cha = np.array([])
            dqdv_cha = np.array([])
            q_intcha = np.array([])
            q_cha = np.array([])
            dvdq_cha = np.array([])
            v_intcha = np.array([])

        vcha_list.append(np.asarray(v_cha, dtype=float))
        dqdv_cha_list.append(np.asarray(dqdv_cha, dtype=float))
        q_intcha_list.append(np.asarray(q_intcha, dtype=float))
        qcha_list.append(np.asarray(q_cha, dtype=float))
        dvdq_cha_list.append(np.asarray(dvdq_cha, dtype=float))
        v_intcha_list.append(np.asarray(v_intcha, dtype=float))

        curve_df = qocv_df.copy()
        curve_df.insert(0, "summary_row_index", idx)
        curve_df.insert(1, "qocv_file", qocv_file.name)
        curve_df.insert(2, "equivalent_cycle_count", selected_summary.iloc[idx].get("equivalent_cycle_count"))
        curve_df.insert(3, "total_throughput", selected_summary.iloc[idx].get("total_throughput"))
        curve_df.insert(4, "capacity", selected_summary.iloc[idx].get("capacity"))
        curve_frames.append(curve_df)

    curve_table = (
        pd.concat(curve_frames, ignore_index=True)
        if curve_frames
        else pd.DataFrame(
            columns=[
                "summary_row_index",
                "qocv_file",
                "equivalent_cycle_count",
                "total_throughput",
                "capacity",
            ]
        )
    )

    summary_with_features = summary_df.copy()
    summary_with_features["Vcha"] = pd.Series([None] * len(summary_with_features), dtype="object")
    summary_with_features["dQdVcha"] = pd.Series([None] * len(summary_with_features), dtype="object")
    summary_with_features["Q_intVcha"] = pd.Series([None] * len(summary_with_features), dtype="object")
    summary_with_features["Qcha"] = pd.Series([None] * len(summary_with_features), dtype="object")
    summary_with_features["dVdQcha"] = pd.Series([None] * len(summary_with_features), dtype="object")
    summary_with_features["V_intQcha"] = pd.Series([None] * len(summary_with_features), dtype="object")
    summary_with_features["mean_dQdV_2_to_v2_cha"] = np.nan
    summary_with_features["var_dQdV_2_to_v2_cha"] = np.nan
    summary_with_features["qocv_capacity_cha"] = np.nan

    if matched_count > 0:
        window_df = pd.DataFrame({"Vcha": vcha_list, "dQdVcha": dqdv_cha_list})
        mean_2_to_v2, var_2_to_v2 = window_delta_mean_var(
            window_df,
            x_col="Vcha",
            y_col="dQdVcha",
            x_lo=2.0,
            x_hi=V_2,
        )
        for idx in range(matched_count):
            summary_with_features.at[idx, "Vcha"] = _array_to_csv_string(vcha_list[idx])
            summary_with_features.at[idx, "dQdVcha"] = _array_to_csv_string(dqdv_cha_list[idx])
            summary_with_features.at[idx, "Q_intVcha"] = _array_to_csv_string(q_intcha_list[idx])
            summary_with_features.at[idx, "Qcha"] = _array_to_csv_string(qcha_list[idx])
            summary_with_features.at[idx, "dVdQcha"] = _array_to_csv_string(dvdq_cha_list[idx])
            summary_with_features.at[idx, "V_intQcha"] = _array_to_csv_string(v_intcha_list[idx])
            summary_with_features.at[idx, "mean_dQdV_2_to_v2_cha"] = float(mean_2_to_v2[idx])
            summary_with_features.at[idx, "var_dQdV_2_to_v2_cha"] = float(var_2_to_v2[idx])
            if q_intcha_list[idx].size:
                summary_with_features.at[idx, "qocv_capacity_cha"] = float(q_intcha_list[idx][-1])

    summary_feature_file = summary_file.with_name(
        summary_file.name.removesuffix(SUMMARY_SUFFIX) + QOCV_FEATURE_SUFFIX
    )
    summary_with_features.to_csv(summary_feature_file, index=False)
    dqdv_plot_path = _plot_ica_curves(
        cell_name=summary_file.name.removesuffix(SUMMARY_SUFFIX),
        vcha_list=vcha_list,
        dqdv_cha_list=dqdv_cha_list,
        output_dir=summary_file.parent,
    )

    return CellResult(
        cell_name=summary_file.name.removesuffix(SUMMARY_SUFFIX),
        summary_file=summary_file,
        summary_feature_file=summary_feature_file,
        qocv_dir=qocv_dir,
        valid_rows_until_nan=valid_rows_until_nan,
        matched_qocv_files=matched_count,
        unmatched_summary_rows=max(0, valid_rows_until_nan - matched_count),
        extra_qocv_files=max(0, len(qocv_files) - matched_count),
        qocv_point_count=qocv_point_count,
        curve_table=curve_table,
        dqdv_plot_path=dqdv_plot_path,
    )


def main() -> None:
    summary_files = list(_find_summary_files(DATA_ROOT))
    if not summary_files:
        print(f"No summary files found in: {DATA_ROOT}")
        return

    all_cells_frames: list[pd.DataFrame] = []
    for summary_file in summary_files:
        result = analyze_cell(summary_file)
        print(f"\nCell: {result.cell_name}")
        print(f"Summary file: {result.summary_file}")
        print(f"Summary + qOCV features: {result.summary_feature_file}")
        print(f"dQdV_cha vs V_cha plot: {result.dqdv_plot_path}")
        print(f"qOCV folder: {result.qocv_dir}")
        print(f"Valid rows until NaN: {result.valid_rows_until_nan}")
        print(f"Matched qOCV files: {result.matched_qocv_files}")
        if result.unmatched_summary_rows:
            print(f"[WARN] Summary rows without qOCV files: {result.unmatched_summary_rows}")
        if result.extra_qocv_files:
            print(f"[WARN] Extra qOCV files (unused): {result.extra_qocv_files}")
        print(f"Total qOCV points loaded: {result.qocv_point_count}")

        if not result.curve_table.empty:
            with_cell = result.curve_table.copy()
            with_cell.insert(0, "cell_name", result.cell_name)
            all_cells_frames.append(with_cell)

    if not all_cells_frames:
        print("\nNo qOCV curve points were loaded.")
        return

    combined = pd.concat(all_cells_frames, ignore_index=True)
    output_path = Path(__file__).resolve().parent / "qocv_curve_combined.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved combined qOCV curve table: {output_path}")
    print(f"Rows saved: {len(combined)}")


if __name__ == "__main__":
    main()
