from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LI_DATARUN_DIR = _PROJECT_ROOT / "Li_datarun"
for _import_path in (_LI_DATARUN_DIR, _PROJECT_ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.lines import Line2D
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from core_functions.interpolation import (
    interpolate_value_at_soh as _shared_interpolate_value_at_soh,
    interpolate_value_at_x,
)
from core_functions.plot_display import show_rendered_figure
import config as li_config
import eol_mlp_20w_IOWA as iowa_aug70

from runtime import (
    DEFAULT_FEATURE_ROW_INDEX,
    RAW_CAPACITY_COLUMN,
    RAW_PROCESSED_DATA_DIR,
    RAW_WEEK_COLUMN,
    build_mlp_hyperparameters,
    clean_input_dataframe_base as clean_input_dataframe,
    clean_input_dataframe_from_path,
    get_feature_columns,
    load_and_interpolate,
    parse_cell_name,
)
from core_functions.temporal_models import (
    EolPredictionMLP,
    ExtrapolationResidualTemporalCNN,
    build_log_feature_fit_summary as build_fit_summary,
    build_regression_loss,
    build_shared_extrapolation_tcnn_hyperparameters,
    build_shared_tcnn_hyperparameters,
    apply_tcnn_input_clipping,
    apply_tcnn_output_clipping,
    clip_with_fractional_margin,
    default_r2 as calculate_r2,
    fit_linear_relationship,
    get_required_features,
    make_model_target_transformer,
    make_neural_input_scaler,
    make_neural_target_transformer,
    set_torch_seed,
    split_validation_arrays,
    train_temporal_cnn_regressor,
)
from core_functions.prediction_evaluation import (
    attach_input_last_step_week_metrics,
    attach_interpolated_throughput_metrics,
    attach_interpolated_time_metrics,
    attach_real_target_scale_metrics,
    calculate_ape_metrics,
    calculate_ape_values,
    calculate_group_mapes as _calculate_group_mapes,
    calculate_max_ape_cell,
    convert_step_predictions_to_curve_values,
    convert_step_predictions_to_scaled_values,
    convert_step_predictions_to_time,
    get_real_step_values,
    get_rmse_arrays,
    interpolate_curve_value_at_step_index,
    interpolate_time_at_step_index,
    get_preferred_rmse_value,
)

SCRIPT_DIR = _LI_DATARUN_DIR
PROJECT_ROOT = _PROJECT_ROOT
from core_functions import augmentation_run_config as aug_cfg
from core_functions import autoanchor as core_autoanchor
from core_functions import publication_plot_config as plot_cfg
from core_functions import standard_prediction_plots as standard_plots
from core_functions import temporal_models
FEATURE_SELECTION_LABEL = "none"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / f"fig_iowa_tcnn_s70_{FEATURE_SELECTION_LABEL}"
TARGET_SOH = 0.70
INTERPOLATED_TARGET_COLUMN = "EOL70"
RAW_TARGET_COLUMN = "week_at_eol70_raw"
RAW_THROUGHPUT_TARGET_COLUMN = "throughput_at_eol70_raw"
INTERPOLATED_STEP_TIME_COLUMN = "weeks_per_interpolated_step"
INTERPOLATED_STEP_THROUGHPUT_COLUMN = "throughput_per_interpolated_step"
THRESHOLDS = [float(value) for value in li_config.THRESHOLDS]
DEFAULT_INTERPOLATION_CAPA_LIMS = [0.96, 0.97, 0.98]
DEFAULT_MIN_ABS_PEARSON = 0.40
LONG_EOL_VALIDATION_THRESHOLD_WEEKS = 30.0
EXTRA_LONG_EOL_VALIDATION_THRESHOLD_WEEKS = 40.0
LINEAR_FEATURE_SELECTION_METHOD = "highest_train_r2"
ADDITIONAL_RAW_LINEAR_FEATURE = "mean_delta_dqdv_mid"
LINEAR_AVG_MODEL_VARIANT = "avg_all_features"
LINEAR_RAW_FEATURE_WEEK = 6.0
LINEAR_RAW_FEATURE_WEEK_POST_G20 = 6.0
LINEAR_RAW_FEATURE_WEEK_EARLY = 4.0
LINEAR_RAW_FEATURE_WEEK_POST_G20_EARLY = 4.0
RAW_FEATURE_WEEK = 5.0
RAW_FEATURE_WEEK_POST_G20 = 6.0
LINEAR_INPUT_WEEK_VARIANT_DEFAULT = "wk6"
LINEAR_INPUT_WEEK_VARIANT_EARLY = "wk4"
TCNN_INTERPOLATED_ARTIFACT_DIR = "tcnn_i"
TCNN_RAW_ARTIFACT_DIR = "tcnn_r"
SUBPLOT_COLUMNS = 3
DEFAULT_THRESHOLD_BASIS = "week"
THRESHOLD_BASIS_CHOICES = ("week", "throughput")


def get_interpolated_target_column(target_soh: float) -> str:
    return f"EOL{int(round(float(target_soh) * 100.0))}"
THROUGHPUT_INTERPOLATION_TYPE = "throughput"
THROUGHPUT_DISCRETIZATION_COUNT = 11
THROUGHPUT_INTERPOLATED_INPUT_STEPS = 12
RAW_THROUGHPUT_LINEAR_FEATURE_TARGET = 150.0
RAW_THROUGHPUT_TCNN_INPUT_TARGET = 100.0

def format_feature_math_label(feature_name: object) -> str:
    return standard_plots.format_feature_math_label(feature_name)

def select_input_feature_trajectory_columns(feature_columns: list[str]) -> list[str]:
    preferred = ["mean_delta_Q", "var_delta_Q", "mean_delta_dqdv", "var_delta_dqdv"]
    selected = [feature for feature in preferred if feature in feature_columns]
    return selected or [feature for feature in feature_columns if str(feature).strip().lower() != "soh"][:4]

def normalize_threshold_basis(threshold_basis: str | None) -> str:
    basis = str(threshold_basis or DEFAULT_THRESHOLD_BASIS).strip().lower()
    basis = {"throughtput": "throughput", "throughut": "throughput", "through-put": "throughput"}.get(basis,
        basis)
    if basis not in THRESHOLD_BASIS_CHOICES:
        raise ValueError(f"Unsupported threshold basis: {threshold_basis!r}. Expected one of {THRESHOLD_BASIS_CHOICES}.")
    return basis

def format_threshold_value(threshold: float) -> str:
    numeric = float(threshold)
    if np.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")

def load_cleaned_input_dataframe(csv_path: Path) -> pd.DataFrame:
    path_aware_cleaner = globals().get("clean_input_dataframe_from_path")
    if callable(path_aware_cleaner):
        return path_aware_cleaner(csv_path)
    return clean_input_dataframe(pd.read_csv(csv_path))

def format_threshold_slug_value(threshold: float) -> str:
    return format_threshold_value(threshold).replace("-", "neg").replace(".", "p")

def format_threshold_display(threshold: float, threshold_basis: str) -> str:
    basis = normalize_threshold_basis(threshold_basis)
    return f"{format_threshold_value(threshold)} {'weeks' if basis == 'week' else 'EFC'}"

def build_threshold_scenario_stem(threshold: float, threshold_basis: str) -> str:
    basis = normalize_threshold_basis(threshold_basis)
    slug = format_threshold_slug_value(threshold)
    return f"eligible{slug}" if basis == "week" else f"eligible_throughput{slug}efc"

def get_threshold_metric_map(*, raw_week_by_cell: dict[str, float], raw_throughput_by_cell: dict[str, float],
    threshold_basis: str) -> dict[str, float]:
    if normalize_threshold_basis(threshold_basis) == "week":
        return {str(cell): float(value) for cell, value in raw_week_by_cell.items() if np.isfinite(float(value))}
    return {str(cell): float(value) for cell, value in raw_throughput_by_cell.items()
        if cell in raw_week_by_cell and np.isfinite(float(value))}

def build_threshold_split_by_basis(*, raw_week_by_cell: dict[str, float], raw_throughput_by_cell: dict[str, float],
    threshold: float, threshold_basis: str) -> tuple[list[str], list[str], dict[str, float]]:
    basis = normalize_threshold_basis(threshold_basis)
    metric_map = get_threshold_metric_map(raw_week_by_cell=raw_week_by_cell,
        raw_throughput_by_cell=raw_throughput_by_cell, threshold_basis=basis)
    train_cells = sorted(cell for cell, value in metric_map.items() if float(value) <= float(threshold))
    test_cells = sorted(cell for cell, value in metric_map.items() if float(value) > float(threshold))
    metric_label = "week-at-70% SOH" if basis == "week" else "throughput-at-70% SOH"
    threshold_text = format_threshold_display(threshold, basis)
    if len(train_cells) < 2:
        raise RuntimeError(f"Need at least two training cells with {metric_label} <= {threshold_text}; found {len(train_cells)}.")
    if not test_cells:
        raise RuntimeError(f"No test cells found with {metric_label} > {threshold_text}.")
    return train_cells, test_cells, metric_map

def build_raw_cumulative_throughput(df: pd.DataFrame) -> np.ndarray:
    if "current_throughput" not in df.columns:
        return np.array([], dtype=float)
    throughput = pd.to_numeric(df["current_throughput"], errors="coerce").to_numpy(dtype=float)
    if throughput.size == 0:
        return np.array([], dtype=float)
    finite_mask = np.isfinite(throughput)
    cumulative = np.cumsum(np.where(finite_mask, throughput, 0.0))
    cumulative[~finite_mask] = np.nan
    return cumulative

def remove_legacy_interpolated_linear_plot_outputs(scenario_dir: Path, label: str) -> None:
    patterns = [f"{label}_interpolated_log_feature_vs_log_throughput70*.png",
        f"{label}_interpolated_feature*throughput70*.png", f"{label}_*interpolated*linear*.png",
        f"{label}_*interpolated*feature*.png", f"{label}_raw_log_feature_vs_log_throughput70_wk*.png",
        f"{label}_raw_week_based_log_feature_vs_log_throughput70_wk*.png"]
    for pattern in patterns:
        for path in scenario_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass

def load_raw_throughput_dataframe(*, raw_input_dir: Path, train_cells: set[str], test_cells: set[str],
    raw_throughput_by_cell: dict[str, float],
    linear_feature_target_throughput: float = RAW_THROUGHPUT_LINEAR_FEATURE_TARGET,
    feature_target_max_week: float | None = None) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    rows = []
    skipped = {"read_fail": 0, "missing_core_columns": 0, "missing_feature_columns": 0, "invalid_target": 0,
        "throughput_before_feature_target": 0, "missing_throughput_target": 0,
        "throughput_target_after_week_limit": 0}
    feature_columns: list[str] | None = None
    feature_target_throughput = float(linear_feature_target_throughput)
    max_feature_target_week = float(feature_target_max_week) if feature_target_max_week is not None else np.nan

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            df = load_cleaned_input_dataframe(csv_path)
        except Exception:
            skipped["read_fail"] += 1
            continue
        if RAW_WEEK_COLUMN not in df.columns or RAW_CAPACITY_COLUMN not in df.columns or "current_throughput" not in df.columns:
            skipped["missing_core_columns"] += 1
            continue
        current_feature_columns = get_feature_columns(df.columns.tolist())
        if not current_feature_columns:
            skipped["missing_feature_columns"] += 1
            continue

        weeks = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
        capacities = pd.to_numeric(df[RAW_CAPACITY_COLUMN], errors="coerce").to_numpy(dtype=float)
        cumulative_throughput = build_raw_cumulative_throughput(df)
        if np.isfinite(capacities).any() and np.isfinite(capacities[0]) and capacities[0] != 0:
            df["SOH"] = capacities / capacities[0]
        week_at_target = interpolate_value_at_soh(weeks, weeks, capacities, TARGET_SOH)
        throughput_target = float(raw_throughput_by_cell.get(cell, np.nan))
        if (not np.isfinite(week_at_target) or week_at_target <= 0 or not np.isfinite(throughput_target)
                or throughput_target <= 0):
            skipped["missing_throughput_target" if not np.isfinite(throughput_target)
                or throughput_target <= 0 else "invalid_target"] += 1
            continue
        if not np.isfinite(cumulative_throughput).any() or float(np.nanmax(cumulative_throughput)) < feature_target_throughput:
            skipped["throughput_before_feature_target"] += 1
            continue
        feature_target_week = interpolate_value_at_x(cumulative_throughput, weeks, feature_target_throughput)
        if np.isfinite(max_feature_target_week) and (not np.isfinite(feature_target_week)
                or float(feature_target_week) > max_feature_target_week):
            skipped["throughput_target_after_week_limit"] += 1
            continue

        if "SOH" in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES and "SOH" in df.columns and "SOH" not in current_feature_columns:
            current_feature_columns = ["SOH", *current_feature_columns]
        if (RAW_CAPACITY_COLUMN in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES
                and RAW_CAPACITY_COLUMN in df.columns and RAW_CAPACITY_COLUMN not in current_feature_columns):
            current_feature_columns = [RAW_CAPACITY_COLUMN, *current_feature_columns]
        feature_columns = (current_feature_columns if feature_columns is None
            else [column for column in feature_columns if column in current_feature_columns])

        row = {"Cell": cell, RAW_TARGET_COLUMN: float(week_at_target),
            RAW_THROUGHPUT_TARGET_COLUMN: throughput_target, "is_outer_field_selected": cell in train_cells,
            "is_test_cell": cell in test_cells, "source_file": str(csv_path),
            "linear_input_target_throughput": feature_target_throughput,
            "linear_input_target_week": float(feature_target_week) if np.isfinite(feature_target_week) else np.nan}
        for column in current_feature_columns:
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            row[column] = interpolate_value_at_x(cumulative_throughput, values, feature_target_throughput)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No usable raw throughput rows found in {raw_input_dir}")
    if not feature_columns:
        raise RuntimeError(f"No usable raw throughput features found in {raw_input_dir}")
    data = pd.DataFrame(rows)
    feature_columns = [column for column in feature_columns if column in data.columns]
    if data.empty:
        raise RuntimeError(f"No usable raw throughput rows found in {raw_input_dir}")
    return data, feature_columns, skipped

def load_interpolated_throughput_dataframe(*, raw_input_dir: Path, feature_row_index: int = 0, train_cells: set[str],
    test_cells: set[str], capa_lim: float, raw_throughput_by_cell: dict[str, float],
    discretization_count: int = THROUGHPUT_DISCRETIZATION_COUNT) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    rows = []
    skipped = {"read_fail": 0, "missing_core_columns": 0, "missing_features": 0, "too_short": 0,
        "invalid_target": 0, "missing_throughput_target": 0}
    feature_columns: list[str] | None = None

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            raw_df = load_cleaned_input_dataframe(csv_path)
            df = load_and_interpolate(raw_df, capa_lim=capa_lim,
                interpolation_type=THROUGHPUT_INTERPOLATION_TYPE, discretization_count=int(discretization_count))
        except Exception:
            skipped["read_fail"] += 1
            continue
        if INTERPOLATED_TARGET_COLUMN not in df.columns:
            skipped["missing_core_columns"] += 1
            continue
        current_feature_columns = get_feature_columns(df.columns.tolist())
        if not current_feature_columns:
            skipped["missing_features"] += 1
            continue
        if len(df) <= feature_row_index:
            skipped["too_short"] += 1
            continue

        target_value = pd.to_numeric(df[INTERPOLATED_TARGET_COLUMN], errors="coerce").dropna()
        throughput_target = float(raw_throughput_by_cell.get(cell, np.nan))
        raw_week_at_target = get_raw_week_at_target_from_dataframe(raw_df, TARGET_SOH)
        if (target_value.empty or float(target_value.iloc[0]) <= 0 or not np.isfinite(raw_week_at_target)
                or raw_week_at_target <= 0 or not np.isfinite(throughput_target) or throughput_target <= 0):
            skipped["missing_throughput_target" if not np.isfinite(throughput_target)
                or throughput_target <= 0 else "invalid_target"] += 1
            continue

        throughput_per_step = throughput_target / float(target_value.iloc[0])
        weeks_per_step = raw_week_at_target / float(target_value.iloc[0])
        if (RAW_CAPACITY_COLUMN in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES
                and RAW_CAPACITY_COLUMN in df.columns and RAW_CAPACITY_COLUMN not in current_feature_columns):
            current_feature_columns = [RAW_CAPACITY_COLUMN, *current_feature_columns]
        feature_columns = (current_feature_columns if feature_columns is None
            else [column for column in feature_columns if column in current_feature_columns])

        row = {"Cell": cell, INTERPOLATED_TARGET_COLUMN: float(target_value.iloc[0]),
            RAW_TARGET_COLUMN: float(raw_week_at_target), RAW_THROUGHPUT_TARGET_COLUMN: throughput_target,
            INTERPOLATED_STEP_TIME_COLUMN: float(weeks_per_step),
            INTERPOLATED_STEP_THROUGHPUT_COLUMN: float(throughput_per_step),
            "is_outer_field_selected": cell in train_cells, "is_test_cell": cell in test_cells,
            "source_file": str(csv_path)}
        for column in current_feature_columns:
            value = pd.to_numeric(df[column], errors="coerce").iloc[feature_row_index]
            row[column] = float(value) if np.isfinite(value) else np.nan
        for column in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES:
            if column in row or column not in df.columns:
                continue
            value = pd.to_numeric(df[column], errors="coerce").iloc[feature_row_index]
            row[column] = float(value) if np.isfinite(value) else np.nan
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No usable interpolated throughput rows found in {raw_input_dir}")
    if not feature_columns:
        raise RuntimeError(f"No usable interpolated throughput features found in {raw_input_dir}")
    data = pd.DataFrame(rows)
    return data, [column for column in feature_columns if column in data.columns], skipped

def load_interpolated_throughput_sequence_dataset(*, raw_input_dir: Path, input_steps: int,
    sequence_features: list[str], train_cells: set[str], test_cells: set[str], capa_lim: float,
    discretization_count: int = THROUGHPUT_DISCRETIZATION_COUNT, target_soh: float = TARGET_SOH,
    target_column: str | None = None) -> tuple[dict[str, object], dict[str, int]]:
    resolved_target_column = str(target_column or get_interpolated_target_column(target_soh))
    sequences, targets, target_time_values, time_curves = [], [], [], []
    weeks_per_step_values, target_throughput_values, throughput_curves = [], [], []
    throughput_per_step_values, input_last_step_week_values, cells = [], [], []
    skipped = {"read_fail": 0, "missing_core_columns": 0, "missing_features": 0, "too_short": 0,
        "invalid_target": 0, "nonfinite_sequence": 0}

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            raw_df = load_cleaned_input_dataframe(csv_path)
            df = load_and_interpolate(raw_df, capa_lim=capa_lim,
                interpolation_type=THROUGHPUT_INTERPOLATION_TYPE, discretization_count=int(discretization_count))
        except Exception:
            skipped["read_fail"] += 1
            continue
        if resolved_target_column not in df.columns:
            skipped["missing_core_columns"] += 1
            continue
        if any(feature not in df.columns for feature in sequence_features):
            skipped["missing_features"] += 1
            continue
        if len(df) < input_steps:
            skipped["too_short"] += 1
            continue

        target_value = pd.to_numeric(df[resolved_target_column], errors="coerce").dropna()
        raw_week_at_target = get_raw_week_at_target_from_dataframe(raw_df, target_soh)
        raw_throughput_at_target = get_raw_throughput_at_target_from_dataframe(raw_df, target_soh)
        if (target_value.empty or float(target_value.iloc[0]) <= 0 or not np.isfinite(raw_week_at_target)
                or raw_week_at_target <= 0 or not np.isfinite(raw_throughput_at_target)
                or raw_throughput_at_target <= 0):
            skipped["invalid_target"] += 1
            continue

        sequence = df.loc[:input_steps - 1, sequence_features].apply(pd.to_numeric,
            errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(sequence).all():
            skipped["nonfinite_sequence"] += 1
            continue
        interpolated_target = float(target_value.iloc[0])
        weeks_per_step = float(raw_week_at_target) / interpolated_target
        throughput_per_step = float(raw_throughput_at_target) / interpolated_target
        time_curve = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
        throughput_curve = pd.to_numeric(df.get("current_throughput"), errors="coerce").to_numpy(dtype=float)
        if len(time_curve) == 0 or not np.isfinite(time_curve).any() or len(throughput_curve) == 0 \
                or not np.isfinite(throughput_curve).any():
            skipped["nonfinite_sequence"] += 1
            continue

        sequences.append(sequence)
        targets.append(interpolated_target)
        target_time_values.append(float(raw_week_at_target))
        time_curves.append(time_curve)
        weeks_per_step_values.append(weeks_per_step)
        target_throughput_values.append(float(raw_throughput_at_target))
        throughput_curves.append(throughput_curve)
        throughput_per_step_values.append(throughput_per_step)
        input_last_step_week_values.append(float(time_curve[input_steps - 1])
            if np.isfinite(time_curve[input_steps - 1]) else np.nan)
        cells.append(cell)

    if not sequences:
        raise RuntimeError(f"No usable interpolated throughput sequences found in {raw_input_dir}")
    return ({"sequences": np.asarray(sequences, dtype=float), "targets": np.asarray(targets, dtype=float),
        "target_time_values": np.asarray(target_time_values, dtype=float), "time_curves": list(time_curves),
        "weeks_per_step_values": np.asarray(weeks_per_step_values, dtype=float),
        "target_throughput_values": np.asarray(target_throughput_values, dtype=float),
        "throughput_curves": list(throughput_curves),
        "throughput_per_step_values": np.asarray(throughput_per_step_values, dtype=float),
        "input_last_step_week_values": np.asarray(input_last_step_week_values, dtype=float), "cells": list(cells),
        "train_mask": np.array([cell in train_cells for cell in cells], dtype=bool),
        "test_mask": np.array([cell in test_cells for cell in cells], dtype=bool),
        "feature_columns": list(sequence_features)}, skipped)

def load_raw_throughput_sequence_dataset(*, raw_input_dir: Path, input_steps: int,
    sequence_features: list[str], train_cells: set[str], test_cells: set[str],
    raw_throughput_by_cell: dict[str, float],
    input_target_throughput: float = RAW_THROUGHPUT_TCNN_INPUT_TARGET,
    input_throughput_by_cell: dict[str, float] | None = None,
    max_input_week: float | None = None) -> tuple[dict[str, object], dict[str, int]]:
    sequences, throughput_grids, targets = [], [], []
    input_target_throughput_values, input_last_step_week_values, cells = [], [], []
    skipped = {"read_fail": 0, "missing_core_columns": 0, "missing_features": 0, "too_short": 0,
        "invalid_target": 0, "throughput_before_feature_target": 0, "input_target_after_week_limit": 0,
        "nonfinite_sequence": 0, "missing_throughput_target": 0}
    max_allowed_input_week = float(max_input_week) if max_input_week is not None else np.nan
    resolved_input_target_throughput = float(input_target_throughput)

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            df = load_cleaned_input_dataframe(csv_path)
        except Exception:
            skipped["read_fail"] += 1
            continue
        if RAW_WEEK_COLUMN not in df.columns or RAW_CAPACITY_COLUMN not in df.columns or "current_throughput" not in df.columns:
            skipped["missing_core_columns"] += 1
            continue

        weeks = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
        capacities = pd.to_numeric(df[RAW_CAPACITY_COLUMN], errors="coerce").to_numpy(dtype=float)
        cumulative_throughput = build_raw_cumulative_throughput(df)
        if np.isfinite(capacities).any() and np.isfinite(capacities[0]) and capacities[0] != 0:
            df["SOH"] = capacities / capacities[0]
        if any(feature not in df.columns for feature in sequence_features):
            skipped["missing_features"] += 1
            continue
        if len(df) < 2 or not np.isfinite(cumulative_throughput).any():
            skipped["too_short"] += 1
            continue

        throughput_target = float(raw_throughput_by_cell.get(cell, np.nan))
        if not np.isfinite(throughput_target) or throughput_target <= 0:
            skipped["missing_throughput_target"] += 1
            continue
        cell_input_target = float(input_throughput_by_cell.get(cell, np.nan)
            if input_throughput_by_cell is not None else resolved_input_target_throughput)
        if not np.isfinite(cell_input_target) or cell_input_target <= 0:
            skipped["missing_throughput_target"] += 1
            continue
        if float(np.nanmax(cumulative_throughput)) < cell_input_target:
            skipped["throughput_before_feature_target"] += 1
            continue

        week_at_target = interpolate_value_at_soh(weeks, weeks, capacities, TARGET_SOH)
        if not np.isfinite(week_at_target) or week_at_target <= 0:
            skipped["invalid_target"] += 1
            continue
        throughput_grid = np.linspace(0.0, cell_input_target, int(input_steps))
        sequence_columns: list[np.ndarray] = []
        for feature in sequence_features:
            feature_values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(cumulative_throughput) & np.isfinite(feature_values)
            if not finite_mask.any():
                sequence_columns.append(np.zeros_like(throughput_grid, dtype=float))
                continue
            feature_throughput = cumulative_throughput[finite_mask].astype(float, copy=False)
            feature_series = feature_values[finite_mask].astype(float, copy=False)
            if raw_sequence_feature_should_zero_fill(feature) and float(np.nanmin(feature_throughput)) > 0.0:
                feature_throughput = np.concatenate(([0.0], feature_throughput))
                feature_series = np.concatenate(([0.0], feature_series))
            interpolated_feature = np.array([interpolate_value_at_x_with_edge_hold(feature_throughput,
                feature_series, target_x) for target_x in throughput_grid], dtype=float)
            sequence_columns.append(np.nan_to_num(interpolated_feature, nan=0.0, posinf=0.0, neginf=0.0))
        sequence = np.nan_to_num(np.column_stack(sequence_columns), nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(sequence).all():
            skipped["nonfinite_sequence"] += 1
            continue

        input_last_step_week = interpolate_value_at_x(cumulative_throughput, weeks, cell_input_target)
        if np.isfinite(max_allowed_input_week) and (not np.isfinite(input_last_step_week)
                or float(input_last_step_week) > max_allowed_input_week):
            skipped["input_target_after_week_limit"] += 1
            continue
        sequences.append(sequence)
        throughput_grids.append(np.asarray(throughput_grid, dtype=float))
        targets.append(throughput_target)
        input_target_throughput_values.append(cell_input_target)
        input_last_step_week_values.append(float(input_last_step_week)
            if np.isfinite(input_last_step_week) else np.nan)
        cells.append(cell)

    if not sequences:
        raise RuntimeError(f"No usable raw throughput sequences found in {raw_input_dir}")
    return ({"sequences": np.asarray(sequences, dtype=float), "time_grids": list(throughput_grids),
        "targets": np.asarray(targets, dtype=float), "target_throughput_values": np.asarray(targets, dtype=float),
        "input_target_throughput_values": np.asarray(input_target_throughput_values, dtype=float),
        "input_last_step_week_values": np.asarray(input_last_step_week_values, dtype=float), "cells": list(cells),
        "train_mask": np.array([cell in train_cells for cell in cells], dtype=bool),
        "test_mask": np.array([cell in test_cells for cell in cells], dtype=bool),
        "feature_columns": list(sequence_features),
        "input_target_throughput": float(resolved_input_target_throughput)}, skipped)

def attach_direct_throughput_metrics(model_result: dict[str, object]) -> dict[str, object]:
    model_result["preferred_rmse_metric"] = "throughput"
    for split in ("train", "test"):
        actual = np.asarray(model_result.get(f"{split}_actual", []), dtype=float)
        predicted = np.asarray(model_result.get(f"{split}_pred", []), dtype=float)
        if len(actual) == 0 or len(actual) != len(predicted):
            continue
        model_result[f"{split}_actual_throughput"] = actual
        model_result[f"{split}_pred_throughput"] = predicted
        model_result[f"{split}_throughput_rmse"] = calculate_rmse(actual, predicted)
    return model_result

def attach_interpolated_step_metrics(model_result: dict[str, object],
    sequence_dataset: dict[str, object]) -> dict[str, object]:
    for split in ("train", "test"):
        step_values = get_real_step_values(model_result, sequence_dataset, split)
        if step_values is None:
            continue
        actual_step, pred_step = step_values
        model_result[f"{split}_actual_step"] = actual_step
        model_result[f"{split}_pred_step"] = pred_step
        model_result[f"{split}_step_rmse"] = calculate_rmse(actual_step, pred_step)
    return model_result

def promote_throughput_metrics(model_result: dict[str, object], *, keep_step_fields: bool = True) -> dict[str, object]:
    model_result["preferred_rmse_metric"] = "throughput"
    for split in ("train", "test"):
        actual = np.asarray(model_result.get(f"{split}_actual_throughput", []), dtype=float)
        predicted = np.asarray(model_result.get(f"{split}_pred_throughput", []), dtype=float)
        if len(actual) == 0 or len(actual) != len(predicted):
            continue
        mape, max_ape = calculate_ape_metrics(actual, predicted)
        model_result[f"{split}_actual"] = actual
        model_result[f"{split}_pred"] = predicted
        model_result[f"{split}_rmse"] = calculate_rmse(actual, predicted)
        model_result[f"{split}_mape"] = mape
        model_result[f"{split}_max_ape"] = max_ape
        model_result[f"{split}_r2"] = calculate_r2(actual, predicted)
        if not keep_step_fields:
            model_result.pop(f"{split}_actual_step", None)
            model_result.pop(f"{split}_pred_step", None)
            model_result.pop(f"{split}_step_rmse", None)
    return model_result

def get_linear_artifact_dir_name(model_variant: object) -> str:
    """Keep model artifact paths below Windows MAX_PATH in long comparison folders."""
    variant = str(model_variant or "linear").strip() or "linear"
    if variant == "best_r2":
        return "l_raw"
    replacements = {
        "best_abs_pearson_train": "bp",
        "best_rmse": "br",
        "best_r2": "r2",
        "fixed_mintest_mean_delta_dqdv_mid": "fm",
        "avg_all_features": "avg",
        "mean_delta_dqdv": "dq",
        "mean_delta_Q": "mdq",
        "var_delta_dqdv": "vdq",
        "var_delta_Q": "vdq",
        "throughput": "th",
        "mintest": "min",
    }
    short = variant
    for old, new in replacements.items():
        short = short.replace(old, new)
    short = "".join(char if char.isalnum() or char in "._-" else "_" for char in short)
    short = "_".join(part for part in short.split("_") if part)
    if len(short) > 36:
        checksum = f"{sum(ord(char) for char in variant) % 10000:04d}"
        short = f"{short[:31]}_{checksum}"
    return f"l_{short}"
LINEAR_BEST_RMSE_MODEL_VARIANT = "best_rmse"
LINEAR_BEST_OUT_OF_RANGE_RMSE_MODEL_VARIANT = "best_out_of_range_rmse"
LINEAR_BEST_TRAIN_RMSE_MODEL_VARIANT = "best_train_rmse"
LINEAR_EARLY_MODEL_VARIANT = "best_r2_wk4"
VALIDATION_SPLIT_METHOD = "random" #stratified random
STRATIFIED_VALIDATION_BINS = 3
HIGHLIGHT_GROUPS = ["G1", "G6", "G26"]
HIGHLIGHT_COLOR = "#ff7f0e"
BEST_INTERPOLATED_TCNN_HYPERPARAMETERS = build_shared_tcnn_hyperparameters()
# Raw and interpolated arms share one TCNN so results stay comparable and reproducible.
BEST_RAW_TCNN_HYPERPARAMETERS = build_shared_tcnn_hyperparameters(
    dict(BEST_INTERPOLATED_TCNN_HYPERPARAMETERS)
)
EXTRAPOLATION_MLP_HYPERPARAMETERS = {
    "hidden_dim": 32,
    "hidden_layers": 2,
    "dropout_rate": 0.05,
    "epochs": 400,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "seeds": list(BEST_INTERPOLATED_TCNN_HYPERPARAMETERS["seeds"]),
    "input_extrapolation_margin": 0.15,
    "target_log_extrapolation_margin": 0.20,
    "validation_split": 0.3,
    "early_stopping_patience": 120,
}
EXTRAPOLATION_TCNN_HYPERPARAMETERS = build_shared_extrapolation_tcnn_hyperparameters(
    {"seeds": list(BEST_INTERPOLATED_TCNN_HYPERPARAMETERS["seeds"])}
)
INTERPOLATED_SEQUENCE_AND_MLP_FEATURES = [
    "SOH",
    RAW_CAPACITY_COLUMN,
    "mean_delta_Q",
    "var_delta_Q",
    "mean_delta_dqdv",
    "var_delta_dqdv",
    "mean_delta_dqdv_low",
    "var_delta_dqdv_low",
    "mean_delta_dqdv_mid",
    "var_delta_dqdv_mid",
    "mean_delta_dqdv_high",
    "var_delta_dqdv_high",
]
feature_num = aug_cfg.FEATURE_NUM

def is_charge_profile_feature_name(feature_name: object) -> bool:
    normalized = str(feature_name or "").strip().lower()
    return normalized.startswith(("mean_", "var_")) and normalized.endswith("_c")

def select_soh_meanvar_tcnn_features(candidate_features: list[str], *, label: str) -> list[str]:
    filtered_features = [
        str(name)
        for name in candidate_features
        if not (aug_cfg.EXCLUDE_CHARGE_PROFILE_FEATURES and is_charge_profile_feature_name(name))
    ]
    return temporal_models.select_soh_meanvar_tcnn_features(
        filtered_features, label=label, feature_num=feature_num,
        four_feature_names=aug_cfg.TCNN_FOUR_FEATURE_NAMES)

def get_sequence_and_mlp_feature_names(candidate_features: list[str] | None = None) -> list[str]:
    if candidate_features is not None:
        return select_soh_meanvar_tcnn_features(candidate_features, label="sequence/MLP")
    selected_feature_num = str(feature_num).strip().lower()
    if selected_feature_num == "4":
        wanted = {"soh", *aug_cfg.TCNN_FOUR_FEATURE_NAMES}
        return [
            str(feature_name)
            for feature_name in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES
            if str(feature_name).strip().lower() in wanted
        ]
    if selected_feature_num != "all":
        raise RuntimeError(f"feature_num must be '4' or 'all', got {feature_num!r}.")
    return [
        str(feature_name)
        for feature_name in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES
        if (
            str(feature_name).strip().lower() == "soh"
            or (
                str(feature_name).strip().lower().startswith(("mean_", "var_"))
                and not str(feature_name).strip().lower().startswith("log_")
                and not (aug_cfg.EXCLUDE_CHARGE_PROFILE_FEATURES and is_charge_profile_feature_name(feature_name))
            )
        )
    ]

def get_group_number(cell: str) -> int | None:
    token = str(cell).strip().split("C")[0]
    if len(token) < 2 or not token.startswith("G"):
        return None
    try:
        return int(token[1:])
    except ValueError:
        return None

def get_raw_feature_week(cell: str) -> float:
    group_number = get_group_number(cell)
    if group_number is not None and group_number > 20:
        return float(RAW_FEATURE_WEEK_POST_G20)
    return float(RAW_FEATURE_WEEK)

def get_linear_raw_feature_week(cell: str) -> float:
    return float(LINEAR_RAW_FEATURE_WEEK)

def get_linear_raw_feature_week_early(cell: str) -> float:
    return float(LINEAR_RAW_FEATURE_WEEK_EARLY)

def get_mean_var_feature_names(feature_names: list[str]) -> list[str]:
    return [
        str(feature_name)
        for feature_name in feature_names
        if (str(feature_name).startswith("mean_") or str(feature_name).startswith("var_"))
        and not (aug_cfg.EXCLUDE_CHARGE_PROFILE_FEATURES and is_charge_profile_feature_name(feature_name))
    ]

def finalize_figure(fig: plt.Figure, show: bool) -> None:
    if show:
        show_rendered_figure(fig)
        return
    plt.close(fig)

def save_figure_multiformat(
    fig: plt.Figure,
    output_path: Path,
    *,
    dpi: int = 200,
    bbox_inches: str | None = "tight",
) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = str(bbox_inches)
    fig.savefig(output_path, dpi=int(dpi), **save_kwargs)
    fig.savefig(output_path.with_suffix(".svg"), **save_kwargs)
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare IOWA raw and interpolated prediction performance across lifetime thresholds. "
            "For each threshold, train on cells with raw week-at-70%%SOH <= threshold and test on cells above it."
        )
    )
    parser.add_argument("--raw-input-dir", type=Path, default=RAW_PROCESSED_DATA_DIR)
    parser.add_argument("--feature-row-index", type=int, default=DEFAULT_FEATURE_ROW_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    parser.add_argument(
        "--interpolation-type",
        choices=("throughput", "realtime"),
        default="realtime",
        help="Interpolation reference axis.",
    )
    parser.add_argument(
        "--ref-cell",
        type=int,
        default=1,
        help="Reference cell group used to choose the default interpolation SOH limit.",
    )
    parser.add_argument(
        "--capa-lims",
        nargs="+",
        type=float,
        default=DEFAULT_INTERPOLATION_CAPA_LIMS,
        help="Interpolation SOH limits. Defaults to 0.96 and 0.98.",
    )
    parser.add_argument(
        "--min-abs-pearson",
        type=float,
        default=DEFAULT_MIN_ABS_PEARSON,
        help="Legacy option; MLP/TCNN now use the fixed train_modeltorch_average1.py feature list.",
    )
    parser.add_argument("--no-show", action="store_false", dest="show")
    parser.set_defaults(show=True)
    return parser.parse_args()

def format_soh_folder_name(capa_lim: float) -> str:
    return f"soh{int(round(float(capa_lim) * 100.0))}"

def interpolate_value_at_soh(
    x_values: np.ndarray,
    y_values: np.ndarray,
    capacities: np.ndarray,
    target_soh: float = TARGET_SOH,
) -> float:
    return _shared_interpolate_value_at_soh(x_values, y_values, capacities, target_soh)

def interpolate_value_at_x_with_edge_hold(x_values: np.ndarray, y_values: np.ndarray, target_x: float) -> float:
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]
    if len(x_values) == 0:
        return np.nan

    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    x_values, unique_indices = np.unique(x_values, return_index=True)
    y_values = y_values[unique_indices]
    if len(x_values) == 0:
        return np.nan
    if target_x <= x_values.min():
        return float(y_values[0])
    if target_x >= x_values.max():
        return float(y_values[-1])
    return float(np.interp(target_x, x_values, y_values))

def raw_sequence_feature_should_zero_fill(feature_name: str) -> bool:
    feature_name = str(feature_name).strip()
    return feature_name.startswith("mean_") or feature_name.startswith("var_")

def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual[valid] - predicted[valid]) ** 2)))

def calculate_max_abs_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() == 0:
        return np.nan
    return float(np.max(np.abs(actual[valid] - predicted[valid])))

def calculate_max_abs_error_cell(actual: np.ndarray, predicted: np.ndarray, cells: list[str]) -> str | None:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) != len(predicted) or len(actual) != len(cells):
        return None
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() == 0:
        return None
    valid_cells = [str(cell) for cell, is_valid in zip(cells, valid) if is_valid]
    abs_err = np.abs(actual[valid] - predicted[valid])
    if len(valid_cells) != len(abs_err) or len(abs_err) == 0:
        return None
    return valid_cells[int(np.argmax(abs_err))]

def resolve_plot_color(color_value: object, fallback: object) -> object:
    if color_value is None:
        return fallback
    if isinstance(color_value, str):
        stripped = color_value.strip()
        if not stripped:
            return fallback
        return stripped
    return color_value

def get_max_rmse_error_value(model_result: dict[str, object], split: str) -> float:
    actual, predicted = get_rmse_arrays(model_result, split)
    return calculate_max_abs_error(actual, predicted)

def get_max_rmse_error_cell(model_result: dict[str, object], split: str) -> str | None:
    cells = [str(cell) for cell in model_result.get(f"{split}_cell_list", [])]
    actual, predicted = get_rmse_arrays(model_result, split)
    return calculate_max_abs_error_cell(actual, predicted, cells)

def get_cell_group(cell: str) -> str:
    return str(cell).split("C", maxsplit=1)[0]

def calculate_group_mapes(
    cells: list[str],
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: list[str] = HIGHLIGHT_GROUPS,
) -> dict[str, float]:
    return _calculate_group_mapes(
        cells, actual, predicted, groups=groups, group_getter=get_cell_group)

def format_group_mapes(mapes: dict[str, float]) -> str:
    lines = ["Test MAPEs"]
    for group in HIGHLIGHT_GROUPS:
        mape = mapes.get(group, np.nan)
        lines.append(f"{group}: {mape:.2f}%" if np.isfinite(mape) else f"{group}: n/a")
    return "\n".join(lines)

def train_extrapolation_mlp_regressor(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    dataset_label: str,
) -> dict[str, object]:
    config = build_mlp_hyperparameters(EXTRAPOLATION_MLP_HYPERPARAMETERS)
    required_columns = ["Cell", "is_outer_field_selected", "is_test_cell", target_column] + feature_columns
    dataset = data[required_columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    dataset = dataset[dataset[target_column] > 0].copy()
    train_df = dataset[dataset["is_outer_field_selected"]].copy()
    test_df = dataset[dataset["is_test_cell"]].copy()
    if len(train_df) < 2:
        raise RuntimeError(f"Not enough selected training rows for {dataset_label} extrapolation MLP.")
    if test_df.empty:
        raise RuntimeError(f"No test rows available for {dataset_label} extrapolation MLP.")

    x_train_raw = train_df[feature_columns].to_numpy(dtype=float)
    x_test_raw = test_df[feature_columns].to_numpy(dtype=float)
    x_train_min = np.min(x_train_raw, axis=0)
    x_train_max = np.max(x_train_raw, axis=0)
    x_test_bounded = clip_with_fractional_margin(x_test_raw, x_train_min, x_train_max,
        margin_fraction=float(config["input_extrapolation_margin"]))

    x_scaler = make_neural_input_scaler()
    x_train = x_scaler.fit_transform(x_train_raw)
    x_test = x_scaler.transform(x_test_bounded)
    y_train_raw = train_df[target_column].to_numpy(dtype=float)
    target_transformer = make_neural_target_transformer(y_train_raw)
    target_divisor = float(target_transformer.target_divisor_)
    y_train_scaled = target_transformer.transform(y_train_raw)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    train_pred_scaled_collection: list[np.ndarray] = []
    test_pred_scaled_collection: list[np.ndarray] = []
    loss_fn = nn.MSELoss()

    for seed in config["seeds"]:
        x_fit, y_fit, x_val, y_val = split_validation_arrays(
            x_data=x_train,
            y_data=y_train_scaled,
            seed=int(seed),
            val_fraction=float(config["validation_split"]),
            method=VALIDATION_SPLIT_METHOD,
            stratified_bins=STRATIFIED_VALIDATION_BINS,
        )

        set_torch_seed(int(seed))
        model = EolPredictionMLP(
                len(feature_columns),
                [int(int(config["hidden_dim"])), max(8, int(int(config["hidden_dim"])) // 2)],
                float(float(config["dropout_rate"])),
            )
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )

        x_fit_tensor = torch.tensor(x_fit, dtype=torch.float32)
        y_fit_tensor = torch.tensor(y_fit, dtype=torch.float32)
        if x_val is not None and y_val is not None:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        else:
            x_val_tensor = None
            y_val_tensor = None

        best_metric = float("inf")
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        epochs_without_improvement = 0

        for _ in range(int(config["epochs"])):
            model.train()
            optimizer.zero_grad()
            predictions = model(x_fit_tensor)
            loss = loss_fn(predictions, y_fit_tensor)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                if x_val_tensor is not None and y_val_tensor is not None:
                    metric = float(loss_fn(model(x_val_tensor), y_val_tensor).item())
                else:
                    metric = float(loss.item())

            if metric < best_metric:
                best_metric = metric
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(config["early_stopping_patience"]):
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            train_pred_scaled_collection.append(model(torch.tensor(x_train, dtype=torch.float32)).cpu().numpy().reshape(-1, 1))
            test_pred_scaled_collection.append(model(x_test_tensor).cpu().numpy().reshape(-1, 1))

    train_pred_scaled = np.mean(np.stack(train_pred_scaled_collection, axis=0), axis=0)
    test_pred_scaled = np.mean(np.stack(test_pred_scaled_collection, axis=0), axis=0)
    target_transform_min = float(np.min(y_train_scaled))
    target_transform_max = float(np.max(y_train_scaled))
    train_pred_transformed = clip_with_fractional_margin(
        train_pred_scaled.reshape(-1),
        np.array([target_transform_min]),
        np.array([target_transform_max]),
        margin_fraction=float(config["target_log_extrapolation_margin"]),
    ).reshape(-1)
    test_pred_transformed = clip_with_fractional_margin(
        test_pred_scaled.reshape(-1),
        np.array([target_transform_min]),
        np.array([target_transform_max]),
        margin_fraction=float(config["target_log_extrapolation_margin"]),
    ).reshape(-1)
    train_pred = target_transformer.inverse_transform(train_pred_transformed)
    test_pred = target_transformer.inverse_transform(test_pred_transformed)
    train_actual = train_df[target_column].to_numpy(dtype=float)
    test_actual = test_df[target_column].to_numpy(dtype=float)
    train_pred = np.maximum(train_pred, 1e-9)
    test_pred = np.maximum(test_pred, 1e-9)
    test_features_clipped = bool(np.any(np.abs(x_test_bounded - x_test_raw) > 0))

    return {
        "model_name": "mlp_extrap",
        "dataset_label": dataset_label,
        "feature_count": len(feature_columns),
        "train_count": len(train_df),
        "test_count": len(test_df),
        "train_mape": calculate_ape_metrics(train_actual, train_pred)[0],
        "test_mape": calculate_ape_metrics(test_actual, test_pred)[0],
        "train_max_ape": calculate_ape_metrics(train_actual, train_pred)[1],
        "test_max_ape": calculate_ape_metrics(test_actual, test_pred)[1],
        "train_max_ape_cell": calculate_max_ape_cell(train_actual, train_pred, train_df["Cell"].astype(str).tolist()),
        "test_max_ape_cell": calculate_max_ape_cell(test_actual, test_pred, test_df["Cell"].astype(str).tolist()),
        "train_r2": calculate_r2(train_actual, train_pred),
        "test_r2": calculate_r2(test_actual, test_pred),
        "epochs": int(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "hidden_dim": int(config["hidden_dim"]),
        "hidden_layers": int(config["hidden_layers"]),
        "dropout_rate": float(config["dropout_rate"]),
        "validation_split": float(config["validation_split"]),
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "seed_count": len(config["seeds"]),
        "target_column": target_column,
        "input_transform": "minmax_scaler",
        "target_transform": target_transformer.mode,
        "target_divisor": target_divisor,
        "target_scaler": "identity",
        "target_transform_min": target_transform_min,
        "target_transform_max": target_transform_max,
        "input_extrapolation_margin": float(config["input_extrapolation_margin"]),
        "target_log_extrapolation_margin": float(config["target_log_extrapolation_margin"]),
        "test_features_clipped": test_features_clipped,
        "train_actual": train_actual,
        "train_pred": train_pred,
        "test_actual": test_actual,
        "test_pred": test_pred,
        "train_cell_list": train_df["Cell"].astype(str).tolist(),
        "test_cell_list": test_df["Cell"].astype(str).tolist(),
    }

def train_extrapolation_temporal_cnn_regressor(
    sequence_dataset: dict[str, object],
    dataset_label: str,
) -> dict[str, object]:
    config = dict(EXTRAPOLATION_TCNN_HYPERPARAMETERS)
    sequences = np.asarray(sequence_dataset["sequences"], dtype=float)
    targets = np.asarray(sequence_dataset["targets"], dtype=float)
    cells = list(sequence_dataset["cells"])
    train_mask = np.asarray(sequence_dataset["train_mask"], dtype=bool)
    test_mask = np.asarray(sequence_dataset["test_mask"], dtype=bool)
    feature_columns = list(sequence_dataset["feature_columns"])

    x_train_raw = sequences[train_mask]
    x_test_raw = sequences[test_mask]
    y_train_raw = targets[train_mask]
    y_test_raw = targets[test_mask]
    train_cells = [cell for cell, is_train in zip(cells, train_mask) if is_train]
    test_cells = [cell for cell, is_test in zip(cells, test_mask) if is_test]
    if len(x_train_raw) < 2:
        raise RuntimeError(f"Not enough selected training sequences for {dataset_label} extrapolation temporal CNN.")
    if len(x_test_raw) == 0:
        raise RuntimeError(f"No validation sequences available for {dataset_label} extrapolation temporal CNN.")

    x_train_min = np.min(x_train_raw, axis=(0, 1))
    x_train_max = np.max(x_train_raw, axis=(0, 1))
    x_test_bounded = apply_tcnn_input_clipping(x_test_raw, x_train_min, x_train_max)

    scaler = make_neural_input_scaler()
    x_train_scaled = scaler.fit_transform(x_train_raw.reshape(-1, x_train_raw.shape[-1])).reshape(x_train_raw.shape)
    x_test_scaled = scaler.transform(x_test_bounded.reshape(-1, x_test_raw.shape[-1])).reshape(x_test_raw.shape)
    target_transformer = make_neural_target_transformer(y_train_raw)
    target_divisor = float(target_transformer.target_divisor_)
    y_train_scaled = target_transformer.transform(y_train_raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
    target_transform_min = float(np.min(y_train_scaled))
    target_transform_max = float(np.max(y_train_scaled))
    train_actual = y_train_raw
    test_actual = y_test_raw
    train_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []

    for seed in config["seeds"]:
        x_fit, y_fit, x_val, y_val = split_validation_arrays(
            x_data=x_train_scaled,
            y_data=y_train_scaled,
            seed=int(seed),
            val_fraction=float(config["validation_split"]),
            method=VALIDATION_SPLIT_METHOD,
            stratified_bins=STRATIFIED_VALIDATION_BINS,
        )
        set_torch_seed(int(seed))
        model = ExtrapolationResidualTemporalCNN(
            input_dims=len(feature_columns),
            timesteps=int(config["input_steps"]),
            conv_channels=int(config["conv_channels"]),
            fc_hidden_dim=int(config["fc_hidden_dim"]),
            dropout_rate=float(config["dropout_rate"]),
        ).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(config["scheduler_patience"]),
            factor=float(config.get("scheduler_factor", 0.8)),
        )
        loss_fn = build_regression_loss(str(config.get("loss_function", "smooth_l1")))
        x_fit_tensor = torch.tensor(x_fit, dtype=torch.float32, device=device)
        y_fit_tensor = torch.tensor(y_fit, dtype=torch.float32, device=device).unsqueeze(1)
        batch_size = int(config["batch_size"] or len(x_fit_tensor))
        train_loader = DataLoader(
            TensorDataset(x_fit_tensor, y_fit_tensor),
            batch_size=batch_size,
            shuffle=True,
        )
        if x_val is not None and y_val is not None:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)
        else:
            x_val_tensor = None
            y_val_tensor = None

        best_metric = float("inf")
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        epochs_without_improvement = 0

        validation_interval = max(1, int(config.get("validation_interval", 1)))
        for epoch in range(int(config["epochs"]) + 1):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

            if epoch % validation_interval == 0 or epoch == int(config["epochs"]):
                model.eval()
                with torch.no_grad():
                    if x_val_tensor is not None and y_val_tensor is not None:
                        metric = float(loss_fn(model(x_val_tensor), y_val_tensor).item())
                    else:
                        metric = float(loss_fn(model(x_fit_tensor), y_fit_tensor).item())
                scheduler.step(metric)
                if metric < best_metric:
                    best_metric = metric
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= int(config["early_stopping_patience"]):
                        break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            train_pred_scaled = model(x_train_tensor).cpu().numpy().reshape(-1)
            test_pred_scaled = model(x_test_tensor).cpu().numpy().reshape(-1)

        train_pred_transformed = apply_tcnn_output_clipping(train_pred_scaled,
            np.array([target_transform_min]), np.array([target_transform_max])).reshape(-1)
        test_pred_transformed = apply_tcnn_output_clipping(test_pred_scaled,
            np.array([target_transform_min]), np.array([target_transform_max])).reshape(-1)
        train_pred = target_transformer.inverse_transform(train_pred_transformed)
        test_pred = target_transformer.inverse_transform(test_pred_transformed)
        train_predictions.append(np.maximum(train_pred, 1e-9))
        test_predictions.append(np.maximum(test_pred, 1e-9))

    train_pred = np.mean(np.vstack(train_predictions), axis=0)
    test_pred = np.mean(np.vstack(test_predictions), axis=0)

    return {
        "model_name": "tcnn_extrap",
        "dataset_label": dataset_label,
        "feature_count": len(feature_columns),
        "train_count": len(train_cells),
        "test_count": len(test_cells),
        "train_mape": float(np.mean(np.abs((train_actual - train_pred) / train_actual)) * 100),
        "test_mape": float(np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100),
        "train_max_ape": float(np.max(np.abs((train_actual - train_pred) / train_actual)) * 100),
        "test_max_ape": float(np.max(np.abs((test_actual - test_pred) / test_actual)) * 100),
        "train_max_ape_cell": calculate_max_ape_cell(train_actual, train_pred, train_cells),
        "test_max_ape_cell": calculate_max_ape_cell(test_actual, test_pred, test_cells),
        "train_r2": calculate_r2(train_actual, train_pred),
        "test_r2": calculate_r2(test_actual, test_pred),
        "epochs": int(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "dropout_rate": float(config["dropout_rate"]),
        "input_steps": int(config["input_steps"]),
        "conv_channels": int(config["conv_channels"]),
        "fc_hidden_dim": int(config["fc_hidden_dim"]),
        "batch_size": int(config["batch_size"] or len(x_train_raw)),
        "scheduler_patience": int(config["scheduler_patience"]),
        "model_structure": "extrapolation_residual",
        "validation_split": float(config["validation_split"]),
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "seed_count": len(config["seeds"]),
        "input_transform": "minmax_scaler",
        "target_transform": target_transformer.mode,
        "target_divisor": target_divisor,
        "target_scaler": "identity",
        "target_transform_min": target_transform_min,
        "target_transform_max": target_transform_max,
        "input_extrapolation_margin": float(config["input_extrapolation_margin"]),
        "target_log_extrapolation_margin": float(config["target_log_extrapolation_margin"]),
        "train_actual": train_actual,
        "train_pred": train_pred,
        "test_actual": test_actual,
        "test_pred": test_pred,
        "train_cell_list": train_cells,
        "test_cell_list": test_cells,
    }


def plot_prediction_comparison_axis(axis: plt.Axes, model_result: dict[str, object] | None, panel_title: str) -> None:
    if model_result is None:
        axis.set_axis_off()
        return

    test_actual, test_pred, target_label = get_preferred_plot_arrays(model_result, "test")
    test_cells = [str(cell) for cell in model_result.get("test_cell_list", [])]
    test_input_last_step_week = np.asarray(model_result.get("test_input_last_step_week", []), dtype=float)
    if len(test_actual) == 0 or len(test_pred) == 0 or len(test_cells) == 0:
        axis.set_axis_off()
        return

    valid = np.isfinite(test_actual) & np.isfinite(test_pred)
    if valid.sum() == 0:
        axis.set_axis_off()
        return

    actual = test_actual[valid]
    predicted = test_pred[valid]
    cells = [cell for cell, is_valid in zip(test_cells, valid) if is_valid]
    week_values: np.ndarray | None = None
    if len(test_input_last_step_week) == len(test_cells):
        week_values = test_input_last_step_week[valid]
    week_by_cell: dict[str, float] = {}
    if week_values is not None:
        for cell, week_value in zip(cells, week_values):
            if np.isfinite(week_value) and cell not in week_by_cell:
                week_by_cell[cell] = float(week_value)
    top_input_week_cells = {
        cell
        for cell, _ in sorted(week_by_cell.items(), key=lambda item: item[1], reverse=True)[:4]
    }
    lower = float(min(np.min(actual), np.min(predicted)))
    upper = float(max(np.max(actual), np.max(predicted)))
    padding = max(0.05 * (upper - lower), 0.05)

    regular_actual: list[float] = []
    regular_pred: list[float] = []
    highlighted: dict[str, tuple[list[float], list[float], list[str]]] = {
        group: ([], [], []) for group in HIGHLIGHT_GROUPS
    }
    for cell, actual_value, predicted_value in zip(cells, actual, predicted):
        group = get_cell_group(cell)
        if group in highlighted:
            highlighted[group][0].append(float(actual_value))
            highlighted[group][1].append(float(predicted_value))
            highlighted[group][2].append(cell)
        else:
            regular_actual.append(float(actual_value))
            regular_pred.append(float(predicted_value))

    if regular_actual:
        axis.scatter(regular_actual, regular_pred, s=30, alpha=0.6, color="#8c8c8c", edgecolors="none")
        for cell, actual_value, predicted_value in zip(cells, actual, predicted):
            if cell in top_input_week_cells and cell not in {item for values in highlighted.values() for item in values[2]}:
                axis.text(actual_value, predicted_value, f"{cell} (tmax={week_by_cell[cell]:.1f}w)", fontsize=7, ha="left", va="bottom")
    for group, (group_actual, group_predicted, group_cells) in highlighted.items():
        if not group_actual:
            continue
        axis.scatter(group_actual, group_predicted, s=56, alpha=0.92, color=HIGHLIGHT_COLOR, edgecolors="black", linewidths=0.4, zorder=3)
        for cell, x_value, y_value in zip(group_cells, group_actual, group_predicted):
            label_text = cell
            if cell in top_input_week_cells and cell in week_by_cell:
                label_text = f"{cell} (tmax={week_by_cell[cell]:.1f}w)"
            axis.text(x_value, y_value, label_text, fontsize=7, ha="left", va="bottom")

    diagonal = np.linspace(lower - padding, upper + padding, 200)
    axis.plot(diagonal, diagonal, linestyle="--", color="black", linewidth=1.0)
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(lower - padding, upper + padding)
    selected_anchor_soh = pd.to_numeric(model_result.get("selected_anchor_soh"), errors="coerce")
    display_panel_title = str(panel_title)
    if np.isfinite(selected_anchor_soh) and "interpolated" in display_panel_title.lower():
        display_panel_title = f"{display_panel_title}\nanchor SOH {float(selected_anchor_soh) * 100.0:.1f}%"
    axis.set_title(display_panel_title)
    axis.set_xlabel(f"Actual {target_label}")
    axis.set_ylabel(f"Predicted {target_label}")

    group_mapes = calculate_group_mapes(cells, actual, predicted)
    display_mape, _ = calculate_ape_metrics(actual, predicted)
    display_r2 = calculate_r2(actual, predicted)
    long_eol_threshold_weeks = model_result.get("long_eol_threshold_weeks")
    long_eol_test_cells = {str(cell) for cell in model_result.get("long_eol_test_cells", [])}
    long_eol_mask = np.array([str(cell) in long_eol_test_cells for cell in cells], dtype=bool) if long_eol_test_cells else np.zeros(len(cells), dtype=bool)
    long_eol_test_count = int(np.count_nonzero(long_eol_mask))
    long_eol_test_mape = np.nan
    if long_eol_test_count > 0:
        long_eol_test_mape, _ = calculate_ape_metrics(actual[long_eol_mask], predicted[long_eol_mask])
    test_rmse = get_preferred_rmse_value(model_result, "test")
    long_eol_test_rmse = float(model_result.get("long_eol_test_rmse", np.nan))
    extra_long_eol_threshold_weeks = model_result.get("extra_long_eol_threshold_weeks")
    extra_long_eol_test_count = int(model_result.get("extra_long_eol_test_count", 0) or 0)
    extra_long_eol_test_mape = pd.to_numeric(model_result.get("extra_long_eol_test_mape"), errors="coerce")
    extra_long_eol_test_rmse = pd.to_numeric(model_result.get("extra_long_eol_test_rmse"), errors="coerce")
    summary_lines = [
        f"n={len(actual)}",
        (
            f"MAPE all={display_mape:.2f}%"
            if np.isfinite(display_mape)
            else "MAPE all=n/a"
        ),
        f"RMSE={test_rmse:.2f}" if np.isfinite(test_rmse) else "RMSE=n/a",
        (
            f"R^2={display_r2:.3f}"
            if np.isfinite(display_r2)
            else "R^2=n/a"
        ),
    ]
    if (
        long_eol_threshold_weeks is not None
        and long_eol_test_count > 0
        and np.isfinite(long_eol_test_mape)
    ):
        summary_lines.append(
            f"MAPE >{int(float(long_eol_threshold_weeks))}w={float(long_eol_test_mape):.2f}%"
        )
        if np.isfinite(long_eol_test_rmse):
            summary_lines.append(f"RMSE >{int(float(long_eol_threshold_weeks))}w={long_eol_test_rmse:.2f}")
        summary_lines.append(f"n >{int(float(long_eol_threshold_weeks))}w={int(long_eol_test_count)}")
    if (
        extra_long_eol_threshold_weeks is not None
        and extra_long_eol_test_count > 0
        and np.isfinite(extra_long_eol_test_mape)
    ):
        summary_lines.append(
            f"MAPE >{int(float(extra_long_eol_threshold_weeks))}w={float(extra_long_eol_test_mape):.2f}%"
        )
        if np.isfinite(extra_long_eol_test_rmse):
            summary_lines.append(f"RMSE >{int(float(extra_long_eol_threshold_weeks))}w={float(extra_long_eol_test_rmse):.2f}")
        summary_lines.append(f"n >{int(float(extra_long_eol_threshold_weeks))}w={int(extra_long_eol_test_count)}")
    summary_lines.extend(
        [
            f"{group}: {group_mapes.get(group, np.nan):.2f}%"
            if np.isfinite(group_mapes.get(group, np.nan))
            else f"{group}: n/a"
            for group in HIGHLIGHT_GROUPS
        ]
    )
    axis.text(0.03, 0.97, "\n".join(summary_lines), transform=axis.transAxes, ha="left", va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.86, "edgecolor": HIGHLIGHT_COLOR})

def extract_test_ape(model_result: dict[str, object] | None) -> np.ndarray:
    if model_result is None:
        return np.array([], dtype=float)
    actual = np.asarray(model_result.get("test_actual", []), dtype=float)
    predicted = np.asarray(model_result.get("test_pred", []), dtype=float)
    return calculate_ape_values(actual, predicted)

def annotate_metric_label(
    axis: plt.Axes,
    x_value: float,
    y_value: float,
    text: str,
    color: str = "black",
    dx: float = 0.0,
    dy: float = 6.0,
    ha: str = "center",
    va: str = "bottom",
    fontsize: float = 8,
) -> None:
    axis.annotate(text, xy=(x_value, y_value), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize, color=color)

def scatter_ape_values_on_bar(
    axis: plt.Axes,
    x_value: float,
    ape_values: np.ndarray,
    color: object,
    bar_width: float = 0.72,
) -> None:
    values = np.asarray(ape_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    if values.size == 1:
        x_offsets = np.array([0.0], dtype=float)
    else:
        spread = min(0.22, float(bar_width) * 0.32)
        x_offsets = np.linspace(-spread, spread, values.size, dtype=float)
    axis.scatter(x_value + x_offsets, values, s=18, color=color, alpha=0.55, edgecolors="white", linewidths=0.3, zorder=2)

def calculate_abs_error_values(model_result: dict[str, object], split: str) -> np.ndarray:
    actual, predicted = get_rmse_arrays(model_result, split)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.size == 0 or actual.size != predicted.size:
        return np.array([], dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(valid):
        return np.array([], dtype=float)
    return np.abs(predicted[valid] - actual[valid])

def is_augmented_focus_label(label: object) -> bool:
    normalized = str(label or "").strip().lower()
    return "aug" in normalized and ("tcnn" in normalized or "mlp" in normalized)

def draw_vertical_metric_violin(
    axis: plt.Axes,
    x_value: float,
    values: np.ndarray,
    color: object,
    *,
    focus: bool = False,
    width: float = 0.72,
) -> None:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return
    distribution_style = standard_plots.draw_configured_distribution(axis,
        finite_values, position=float(x_value), color=str(color), width=width,
        violin_alpha=0.78 if focus else 0.42,
        violin_edgecolor="#111111" if focus else str(color),
        violin_linewidth=1.35 if focus else 0.7, zorder=1.0)
    if finite_values.size == 1:
        x_offsets = np.asarray([0.0], dtype=float)
    else:
        spread = min(0.24, width * 0.34)
        x_offsets = np.linspace(-spread, spread, finite_values.size, dtype=float)
    axis.scatter(float(x_value) + x_offsets, finite_values, s=22 if focus else 17,
        facecolors=str(color) if distribution_style == "whisker" else "white",
        edgecolors="none" if distribution_style == "whisker" else ("#111111" if focus else "#333333"),
        linewidths=0.0 if distribution_style == "whisker" else 0.45,
        alpha=standard_plots.configured_distribution_point_alpha(0.72 if focus else 0.56), zorder=3)





def get_preferred_plot_arrays(
    model_result: dict[str, object],
    split: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    actual_time = np.asarray(model_result.get(f"{split}_actual_time", []), dtype=float)
    pred_time = np.asarray(model_result.get(f"{split}_pred_time", []), dtype=float)
    if len(actual_time) > 0 and len(actual_time) == len(pred_time):
        return actual_time, pred_time, "week at EOL70"
    actual = np.asarray(model_result.get(f"{split}_actual", []), dtype=float)
    predicted = np.asarray(model_result.get(f"{split}_pred", []), dtype=float)
    return actual, predicted, "target"

def resolve_threshold_method_display_label(row: pd.Series) -> str:
    model_name = str(row.get("model_name") or "").strip().lower()
    dataset_kind = str(row.get("dataset_kind") or "").strip()
    if model_name == "linear":
        return str(row.get("display_name") or "Linear raw")
    if model_name == "tcnn" and dataset_kind == "Interp":
        return "TCNN interp"
    if model_name == "tcnn" and dataset_kind == "Raw":
        return "TCNN raw"
    return str(row.get("display_name") or row.get("model_name") or "Model")

def resolve_threshold_method_color(row: pd.Series) -> object:
    model_name = str(row.get("model_name") or "").strip().lower()
    dataset_kind = str(row.get("dataset_kind") or "").strip()
    if model_name == "linear":
        return get_linear_variant_color(row.get("model_variant"))
    if model_name == "tcnn" and dataset_kind == "Interp":
        return "#EF8881"
    if model_name == "tcnn" and dataset_kind == "Raw":
        return "#6FA2CC"
    return "#999999"

def build_threshold_metric_plot_df(
    summary_df: pd.DataFrame,
    *,
    metric_column: str,
    linear_variants: list[str] | None = None,
) -> pd.DataFrame:
    if summary_df.empty or metric_column not in summary_df.columns:
        return pd.DataFrame()

    plot_df = summary_df.copy()
    plot_df[metric_column] = pd.to_numeric(plot_df[metric_column], errors="coerce")
    plot_df["threshold_weeks"] = pd.to_numeric(plot_df["threshold_weeks"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df[metric_column]) & np.isfinite(plot_df["threshold_weeks"])].copy()
    if plot_df.empty:
        return pd.DataFrame()

    plot_df["dataset_kind"] = np.where(
        plot_df["dataset_label"].astype(str).str.contains("interpolated", case=False, na=False),
        "Interp",
        "Raw",
    )
    if linear_variants is not None:
        allowed_variants = {str(value).strip() for value in linear_variants}
        linear_mask = plot_df["model_name"].astype(str).str.strip().str.lower() == "linear"
        variant_mask = plot_df["model_variant"].astype(str).str.strip().isin(allowed_variants)
        plot_df = plot_df[~linear_mask | variant_mask].copy()
    if plot_df.empty:
        return pd.DataFrame()

    plot_df["display_label"] = [resolve_threshold_method_display_label(row) for _, row in plot_df.iterrows()]
    plot_df["display_color"] = [resolve_threshold_method_color(row) for _, row in plot_df.iterrows()]
    return plot_df

def get_threshold_method_order(plot_df: pd.DataFrame) -> list[str]:
    if plot_df.empty:
        return []

    method_meta = plot_df[
        ["display_label", "model_name", "model_variant", "dataset_kind"]
    ].drop_duplicates().copy()
    method_meta["sort_group"] = np.where(
        method_meta["model_name"].astype(str).str.strip().str.lower() == "linear",
        0,
        np.where(
            method_meta["dataset_kind"].astype(str).str.strip() == "Interp",
            1,
            np.where(method_meta["dataset_kind"].astype(str).str.strip() == "Raw", 2, 3),
        ),
    )
    method_meta["variant_sort"] = method_meta["model_variant"].map(get_linear_variant_sort_key)
    method_meta = method_meta.sort_values(
        ["sort_group", "variant_sort", "display_label"],
        ascending=[True, True, True],
        na_position="last",
    )
    return method_meta["display_label"].astype(str).tolist()

def plot_threshold_metric_trend(summary_df: pd.DataFrame, *, metric_column: str, ylabel: str, title: str, output_path: Path, show: bool, linear_variants: list[str] | None = None) -> None:
    plot_df = build_threshold_metric_plot_df(
        summary_df,
        metric_column=metric_column,
        linear_variants=linear_variants,
    )
    if plot_df.empty:
        return

    threshold_order = sorted(float(value) for value in plot_df["threshold_weeks"].dropna().unique().tolist())
    if not threshold_order:
        return

    method_order = get_threshold_method_order(plot_df)
    if not method_order:
        return

    fig_width = max(9.5, 1.9 * len(threshold_order) + 2.8)
    fig, axis = plt.subplots(figsize=(fig_width, 5.2))
    top_extent = 0.0
    value_format = "{:.1f}" if "mape" in metric_column.lower() else "{:.2f}"

    for method_label in method_order:
        method_rows = plot_df[plot_df["display_label"].astype(str) == method_label].copy()
        if method_rows.empty:
            continue
        method_rows = method_rows.sort_values("threshold_weeks")
        color = resolve_plot_color(method_rows.iloc[0].get("display_color"), "#999999")
        x_values: list[float] = []
        y_values: list[float] = []
        for threshold in threshold_order:
            subset = method_rows[method_rows["threshold_weeks"] == float(threshold)]
            x_values.append(float(threshold))
            y_values.append(float(subset.iloc[0][metric_column]) if not subset.empty else np.nan)

        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        valid_mask = np.isfinite(y_arr)
        if not np.any(valid_mask):
            continue

        axis.plot(x_arr[valid_mask], y_arr[valid_mask], marker="o", markersize=6, linewidth=2.0, color=color, label=method_label)
        axis.scatter(x_arr[valid_mask], y_arr[valid_mask], s=40, color=color, edgecolors="white", linewidths=0.7, zorder=3)
        top_extent = max(top_extent, float(np.nanmax(y_arr[valid_mask])))
        for x_value, y_value in zip(x_arr[valid_mask], y_arr[valid_mask]):
            annotate_metric_label(
                axis,
                x_value,
                y_value,
                value_format.format(y_value),
                dx=0,
                dy=7,
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

    axis.set_xticks(threshold_order)
    axis.set_xticklabels([f"{threshold:g}w" for threshold in threshold_order])
    axis.set_xlabel(f"Threshold on raw week at {int(round(float(TARGET_SOH) * 100.0))}% SOH")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_axisbelow(True)
    if top_extent > 0:
        axis.set_ylim(0, top_extent * 1.16 + (2.0 if "mape" in metric_column.lower() else 0.25))
    axis.legend(frameon=False, ncol=min(3, max(1, len(method_order))), fontsize=8)
    fig.tight_layout()
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    save_figure_multiformat(fig, output_path, dpi=200, bbox_inches="tight")
    finalize_figure(fig, show)

def plot_model_family_comparison_grid(threshold: float, output_dir: Path, comparison_rows: list[dict[str, object]], long_eol_threshold_weeks: float, show: bool) -> None:
    if not comparison_rows:
        return

    panel_specs: list[tuple[str, dict[str, object] | None]] = []
    for row_payload in comparison_rows:
        model_label = str(row_payload["model_label"])
        interpolated_full = row_payload.get("interpolated")
        raw_full = row_payload.get("raw")
        interpolated_long = row_payload.get("interpolated_long")
        raw_long = row_payload.get("raw_long")

        if interpolated_full is not None:
            panel_specs.append((f"{model_label}\nInterpolated\nTest > {int(threshold)}w", interpolated_full))
        if interpolated_long is not None:
            panel_specs.append((f"{model_label}\nInterpolated\nTest > {int(long_eol_threshold_weeks)}w", interpolated_long))
        if raw_full is not None:
            panel_specs.append((f"{model_label}\nRaw\nTest > {int(threshold)}w", raw_full))
        if raw_long is not None:
            panel_specs.append((f"{model_label}\nRaw\nTest > {int(long_eol_threshold_weeks)}w", raw_long))

    if not panel_specs:
        return

    n_panels = len(panel_specs)
    ncols = 2
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for axis, (panel_title, result_payload) in zip(axes_flat, panel_specs):
        plot_prediction_comparison_axis(axis, result_payload, panel_title)
    for axis in axes_flat[n_panels:]:
        axis.set_axis_off()

    fig.suptitle(
        f"Threshold {int(threshold)}w: raw vs interpolated prediction comparison "
        f"(including >{int(long_eol_threshold_weeks)}w subset)",
        y=1.01,
    )
    fig.tight_layout()
    finalize_figure(fig, show)

def plot_random_relationship_subplots(data: pd.DataFrame, feature_columns: list[str], fit_summary: pd.DataFrame, output_dir: Path, target_column: str, target_label: str,
    output_stem: str, title_prefix: str, show: bool) -> None:
    n_features = len(feature_columns)
    n_cols = min(SUBPLOT_COLUMNS, max(1, n_features))
    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    summary_by_feature = fit_summary.set_index("feature")

    for axis, feature in zip(axes, feature_columns):
        subset = build_feature_subset(data, feature, target_column)
        train_subset = subset[subset["is_outer_field_selected"]].copy()
        test_subset = subset[subset["is_test_cell"]].copy()

        if subset.empty:
            axis.set_axis_off()
            continue

        highlighted_train = train_subset[train_subset["Cell"].astype(str).map(get_cell_group).isin(HIGHLIGHT_GROUPS)]
        highlighted_test = test_subset[test_subset["Cell"].astype(str).map(get_cell_group).isin(HIGHLIGHT_GROUPS)]
        regular_train = train_subset.drop(index=highlighted_train.index)
        regular_test = test_subset.drop(index=highlighted_test.index)

        if not regular_test.empty:
            axis.scatter(regular_test["x_log"], regular_test["y_log"], s=26, alpha=0.6, color="steelblue", edgecolors="none", label="Test cells")
        if not regular_train.empty:
            axis.scatter(regular_train["x_log"], regular_train["y_log"], s=34, alpha=0.9, color="darkorange", edgecolors="black", linewidths=0.25, label="Train cells")
        if not highlighted_test.empty:
            axis.scatter(highlighted_test["x_log"], highlighted_test["y_log"], s=58, alpha=0.95, color=HIGHLIGHT_COLOR, edgecolors="black", linewidths=0.45, label="G1/G6/G26", zorder=4)
        if not highlighted_train.empty:
            axis.scatter(highlighted_train["x_log"], highlighted_train["y_log"], s=58, alpha=0.95, color=HIGHLIGHT_COLOR, edgecolors="black", linewidths=0.45, zorder=4)

        fit_row = summary_by_feature.loc[feature]
        if np.isfinite(fit_row["r2_train"]) and len(train_subset) >= 2:
            slope_train, intercept_train, _, _ = fit_linear_relationship(
                train_subset["x_log"].to_numpy(dtype=float),
                train_subset["y_log"].to_numpy(dtype=float),
            )
            x_fit = np.linspace(train_subset["x_log"].min(), train_subset["x_log"].max(), 200)
            axis.plot(x_fit, slope_train * x_fit + intercept_train, color="darkred", linewidth=1.8, label="Fit: train")

            if not highlighted_test.empty:
                predicted = np.exp(slope_train * highlighted_test["x_log"].to_numpy(dtype=float) + intercept_train)
                group_mapes = calculate_group_mapes(
                    highlighted_test["Cell"].astype(str).tolist(),
                    highlighted_test[target_column].to_numpy(dtype=float),
                    predicted,
                )
                axis.text(0.03, 0.97, format_group_mapes(group_mapes), transform=axis.transAxes, ha="left", va="top", fontsize=7,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": HIGHLIGHT_COLOR})

        axis.set_title(
            f"{feature}\n"
            f"train r={fit_row['pearson_r_train']:.3f}, R^2={fit_row['r2_train']:.3f}\n"
            f"test MAPE={fit_row['test_log_rescaled_mape_train']:.2f}%"
        )
        axis.set_xlabel(f"log(abs({feature}))")
        axis.set_ylabel(f"log({target_label})")

    for axis in axes[n_features:]:
        axis.set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels() if len(axes) else ([], [])
    if handles:
        unique_handles, unique_labels = [], []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_handles.append(handle)
                unique_labels.append(label)
        fig.legend(unique_handles, unique_labels, loc="upper center", ncol=min(4, len(unique_handles)), frameon=False)
    fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()
    finalize_figure(fig, show)

def calculate_long_eol_validation_metrics(
    model_result: dict[str, object],
    raw_week_by_cell: dict[str, float],
    long_eol_threshold_weeks: float = LONG_EOL_VALIDATION_THRESHOLD_WEEKS,
) -> dict[str, object]:
    test_cells = [str(cell) for cell in model_result.get("test_cell_list", [])]
    test_actual = np.asarray(model_result.get("test_actual", []), dtype=float)
    test_pred = np.asarray(model_result.get("test_pred", []), dtype=float)

    if len(test_cells) == 0 or len(test_actual) != len(test_cells) or len(test_pred) != len(test_cells):
        return {
            "long_eol_threshold_weeks": float(long_eol_threshold_weeks),
            "long_eol_test_count": 0,
            "long_eol_test_mape": np.nan,
            "long_eol_test_rmse": np.nan,
            "long_eol_test_throughput_rmse": np.nan,
            "long_eol_test_max_rmse_error": np.nan,
            "long_eol_test_max_rmse_error_cell": None,
            "long_eol_test_max_ape": np.nan,
            "long_eol_test_max_ape_cell": None,
            "long_eol_test_ape_values": [],
            "long_eol_test_cells": [],
        }

    subset_indices = [
        index
        for index, cell in enumerate(test_cells)
        if float(raw_week_by_cell.get(cell, np.nan)) > float(long_eol_threshold_weeks)
    ]
    if not subset_indices:
        return {
            "long_eol_threshold_weeks": float(long_eol_threshold_weeks),
            "long_eol_test_count": 0,
            "long_eol_test_mape": np.nan,
            "long_eol_test_rmse": np.nan,
            "long_eol_test_throughput_rmse": np.nan,
            "long_eol_test_max_rmse_error": np.nan,
            "long_eol_test_max_rmse_error_cell": None,
            "long_eol_test_max_ape": np.nan,
            "long_eol_test_max_ape_cell": None,
            "long_eol_test_ape_values": [],
            "long_eol_test_cells": [],
        }

    subset_actual = test_actual[subset_indices]
    subset_pred = test_pred[subset_indices]
    rmse_actual, rmse_pred = get_rmse_arrays(model_result, "test")
    subset_rmse = calculate_rmse(rmse_actual[subset_indices], rmse_pred[subset_indices])
    throughput_actual = np.asarray(model_result.get("test_actual_throughput", []), dtype=float)
    throughput_pred = np.asarray(model_result.get("test_pred_throughput", []), dtype=float)
    subset_throughput_rmse = (
        calculate_rmse(throughput_actual[subset_indices], throughput_pred[subset_indices])
        if len(throughput_actual) == len(test_cells) and len(throughput_pred) == len(test_cells)
        else np.nan
    )
    subset_mape, subset_max_ape = calculate_ape_metrics(subset_actual, subset_pred)
    subset_ape_values = calculate_ape_values(subset_actual, subset_pred)
    return {
        "long_eol_threshold_weeks": float(long_eol_threshold_weeks),
        "long_eol_test_count": len(subset_indices),
        "long_eol_test_mape": subset_mape,
        "long_eol_test_rmse": subset_rmse,
        "long_eol_test_throughput_rmse": subset_throughput_rmse,
        "long_eol_test_max_rmse_error": calculate_max_abs_error(rmse_actual[subset_indices], rmse_pred[subset_indices]),
        "long_eol_test_max_rmse_error_cell": calculate_max_abs_error_cell(
            rmse_actual[subset_indices],
            rmse_pred[subset_indices],
            [test_cells[index] for index in subset_indices],
        ),
        "long_eol_test_max_ape": subset_max_ape,
        "long_eol_test_max_ape_cell": calculate_max_ape_cell(
            subset_actual,
            subset_pred,
            [test_cells[index] for index in subset_indices],
        ),
        "long_eol_test_ape_values": subset_ape_values.tolist(),
        "long_eol_test_cells": [test_cells[index] for index in subset_indices],
    }

def attach_named_long_eol_metrics(
    model_result: dict[str, object],
    *,
    raw_week_by_cell: dict[str, float],
    threshold_weeks: float,
    prefix: str,
) -> dict[str, object]:
    metrics = calculate_long_eol_validation_metrics(
        model_result=model_result,
        raw_week_by_cell=raw_week_by_cell,
        long_eol_threshold_weeks=float(threshold_weeks),
    )
    for key, value in metrics.items():
        suffix = str(key).removeprefix("long_eol_")
        model_result[f"{prefix}_{suffix}"] = value
    return model_result

def build_long_eol_plot_result(
    model_result: dict[str, object],
    raw_week_by_cell: dict[str, float],
    long_eol_threshold_weeks: float = LONG_EOL_VALIDATION_THRESHOLD_WEEKS,
) -> dict[str, object] | None:
    test_cells = [str(cell) for cell in model_result.get("test_cell_list", [])]
    test_actual = np.asarray(model_result.get("test_actual", []), dtype=float)
    test_pred = np.asarray(model_result.get("test_pred", []), dtype=float)
    if len(test_cells) == 0 or len(test_actual) != len(test_cells) or len(test_pred) != len(test_cells):
        return None

    subset_indices = [
        index
        for index, cell in enumerate(test_cells)
        if float(raw_week_by_cell.get(cell, np.nan)) > float(long_eol_threshold_weeks)
    ]
    if not subset_indices:
        return None

    subset_actual = test_actual[subset_indices]
    subset_pred = test_pred[subset_indices]
    rmse_actual, rmse_pred = get_rmse_arrays(model_result, "test")
    subset_rmse_actual = rmse_actual[subset_indices]
    subset_rmse_pred = rmse_pred[subset_indices]
    subset_mape, subset_max_ape = calculate_ape_metrics(subset_actual, subset_pred)
    plot_result = dict(model_result)
    plot_result["dataset_label"] = (
        f"{model_result.get('dataset_label', 'dataset')}_long_eol_gt{int(long_eol_threshold_weeks)}"
    )
    plot_result["test_actual"] = subset_actual
    plot_result["test_pred"] = subset_pred
    plot_result["test_cell_list"] = [test_cells[index] for index in subset_indices]
    if len(np.asarray(model_result.get("test_input_last_step_week", []), dtype=float)) == len(test_cells):
        plot_result["test_input_last_step_week"] = np.asarray(
            model_result.get("test_input_last_step_week", []),
            dtype=float,
        )[subset_indices]
    plot_result["test_count"] = len(subset_indices)
    plot_result["test_mape"] = subset_mape
    plot_result["test_rmse"] = calculate_rmse(subset_rmse_actual, subset_rmse_pred)
    plot_result["test_max_rmse_error"] = calculate_max_abs_error(subset_rmse_actual, subset_rmse_pred)
    plot_result["test_max_rmse_error_cell"] = calculate_max_abs_error_cell(
        subset_rmse_actual,
        subset_rmse_pred,
        plot_result["test_cell_list"],
    )
    plot_result["test_max_ape"] = subset_max_ape
    plot_result["test_max_ape_cell"] = calculate_max_ape_cell(
        subset_actual,
        subset_pred,
        plot_result["test_cell_list"],
    )
    if len(np.asarray(model_result.get("test_actual_time", []), dtype=float)) == len(test_cells):
        plot_result["test_actual_time"] = subset_rmse_actual
    if len(np.asarray(model_result.get("test_pred_time", []), dtype=float)) == len(test_cells):
        plot_result["test_pred_time"] = subset_rmse_pred
    if len(np.asarray(model_result.get("test_actual_throughput", []), dtype=float)) == len(test_cells):
        plot_result["test_actual_throughput"] = np.asarray(model_result["test_actual_throughput"], dtype=float)[subset_indices]
    if len(np.asarray(model_result.get("test_pred_throughput", []), dtype=float)) == len(test_cells):
        plot_result["test_pred_throughput"] = np.asarray(model_result["test_pred_throughput"], dtype=float)[subset_indices]
    if "test_actual_throughput" in plot_result and "test_pred_throughput" in plot_result:
        plot_result["test_throughput_rmse"] = calculate_rmse(
            np.asarray(plot_result["test_actual_throughput"], dtype=float),
            np.asarray(plot_result["test_pred_throughput"], dtype=float),
        )
    plot_result["test_r2"] = calculate_r2(subset_actual, subset_pred)
    return plot_result

def get_raw_week_at_target_map(raw_input_dir: Path, target_soh: float) -> dict[str, float]:
    results: dict[str, float] = {}
    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        try:
            df = pd.read_csv(csv_path, usecols=[RAW_WEEK_COLUMN, RAW_CAPACITY_COLUMN])
        except Exception:
            continue

        weeks = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
        capacities = pd.to_numeric(df[RAW_CAPACITY_COLUMN], errors="coerce").to_numpy(dtype=float)
        week_at_target = interpolate_value_at_soh(weeks, weeks, capacities, target_soh)
        if np.isfinite(week_at_target) and week_at_target > 0:
            results[cell] = float(week_at_target)
    return results

def get_raw_throughput_at_target_map(raw_input_dir: Path, target_soh: float) -> dict[str, float]:
    results: dict[str, float] = {}
    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        throughput_at_target = get_raw_throughput_at_target_from_dataframe(df, target_soh)
        if np.isfinite(throughput_at_target) and throughput_at_target > 0:
            results[cell] = float(throughput_at_target)
    return results

def get_raw_week_at_target_from_dataframe(df: pd.DataFrame, target_soh: float = TARGET_SOH) -> float:
    weeks = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
    capacities = pd.to_numeric(df[RAW_CAPACITY_COLUMN], errors="coerce").to_numpy(dtype=float)
    return float(interpolate_value_at_soh(weeks, weeks, capacities, target_soh))

def get_raw_throughput_at_target_from_dataframe(df: pd.DataFrame, target_soh: float = TARGET_SOH) -> float:
    if "current_throughput" not in df.columns:
        return np.nan
    throughput = pd.to_numeric(df["current_throughput"], errors="coerce").to_numpy(dtype=float)
    capacities = pd.to_numeric(df[RAW_CAPACITY_COLUMN], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(throughput)
    if finite_mask.sum() == 0:
        return np.nan
    throughput_cumulative = np.cumsum(np.where(finite_mask, throughput, 0.0))
    return float(interpolate_value_at_soh(throughput_cumulative, throughput_cumulative, capacities, target_soh))

def build_threshold_split(raw_week_by_cell: dict[str, float], threshold: float) -> tuple[list[str], list[str]]:
    train_cells = sorted(cell for cell, week in raw_week_by_cell.items() if week <= threshold)
    test_cells = sorted(cell for cell, week in raw_week_by_cell.items() if week > threshold)
    if len(train_cells) < 2:
        raise RuntimeError(f"Need at least two training cells with week-at-70%%SOH <= {threshold:.0f}; found {len(train_cells)}.")
    if not test_cells:
        raise RuntimeError(f"No test cells found with week-at-70%%SOH > {threshold:.0f}.")
    return train_cells, test_cells

def filter_cells_before_or_at_feature_week(
    raw_week_by_cell: dict[str, float],
    raw_throughput_by_cell: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float | str]]]:
    filtered_week_map: dict[str, float] = {}
    filtered_throughput_map: dict[str, float] = {}
    excluded: list[dict[str, float | str]] = []
    for cell, week_at_target in raw_week_by_cell.items():
        feature_week = get_linear_raw_feature_week(cell)
        if float(week_at_target) <= float(feature_week):
            excluded.append(
                {
                    "Cell": str(cell),
                    "week_at_eol70_raw": float(week_at_target),
                    "feature_week_used": float(feature_week),
                }
            )
            continue
        filtered_week_map[str(cell)] = float(week_at_target)
        if cell in raw_throughput_by_cell and np.isfinite(float(raw_throughput_by_cell[cell])):
            filtered_throughput_map[str(cell)] = float(raw_throughput_by_cell[cell])
    return filtered_week_map, filtered_throughput_map, excluded

def load_interpolated_dataframe(
    raw_input_dir: Path,
    feature_row_index: int,
    train_cells: set[str],
    test_cells: set[str],
    capa_lim: float,
    interpolation_type: str,
    discretization_count: int = 10,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    rows = []
    skipped = {
        "read_fail": 0,
        "missing_core_columns": 0,
        "missing_features": 0,
        "too_short": 0,
        "invalid_target": 0,
    }
    feature_columns: list[str] | None = None

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            raw_df = load_cleaned_input_dataframe(csv_path)
            df = load_and_interpolate(
                raw_df,
                capa_lim=capa_lim,
                interpolation_type=interpolation_type,
                discretization_count=int(discretization_count),
            )
        except Exception:
            skipped["read_fail"] += 1
            continue

        if INTERPOLATED_TARGET_COLUMN not in df.columns:
            skipped["missing_core_columns"] += 1
            continue

        current_feature_columns = get_feature_columns(df.columns.tolist())
        if not current_feature_columns:
            skipped["missing_features"] += 1
            continue
        if len(df) <= feature_row_index:
            skipped["too_short"] += 1
            continue

        target_value = pd.to_numeric(df[INTERPOLATED_TARGET_COLUMN], errors="coerce").dropna()
        if target_value.empty or float(target_value.iloc[0]) <= 0:
            skipped["invalid_target"] += 1
            continue
        raw_week_at_target = get_raw_week_at_target_from_dataframe(raw_df, TARGET_SOH)
        raw_throughput_at_target = get_raw_throughput_at_target_from_dataframe(raw_df, TARGET_SOH)
        if (
            not np.isfinite(raw_week_at_target)
            or raw_week_at_target <= 0
            or not np.isfinite(raw_throughput_at_target)
            or raw_throughput_at_target <= 0
        ):
            skipped["invalid_target"] += 1
            continue
        weeks_per_step = raw_week_at_target / float(target_value.iloc[0])
        throughput_per_step = raw_throughput_at_target / float(target_value.iloc[0])

        if RAW_CAPACITY_COLUMN in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES and RAW_CAPACITY_COLUMN in df.columns and RAW_CAPACITY_COLUMN not in current_feature_columns:
            current_feature_columns = [RAW_CAPACITY_COLUMN, *current_feature_columns]

        if feature_columns is None:
            feature_columns = current_feature_columns
        else:
            feature_columns = [column for column in feature_columns if column in current_feature_columns]

        row = {
            "Cell": cell,
            INTERPOLATED_TARGET_COLUMN: float(target_value.iloc[0]),
            RAW_TARGET_COLUMN: float(raw_week_at_target),
            RAW_THROUGHPUT_TARGET_COLUMN: float(raw_throughput_at_target),
            INTERPOLATED_STEP_TIME_COLUMN: float(weeks_per_step),
            INTERPOLATED_STEP_THROUGHPUT_COLUMN: float(throughput_per_step),
            "is_outer_field_selected": cell in train_cells,
            "is_test_cell": cell in test_cells,
            "source_file": str(csv_path),
        }
        for column in current_feature_columns:
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            value = numeric_series.iloc[feature_row_index]
            row[column] = float(value) if np.isfinite(value) else np.nan

        for column in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES:
            if column in row or column not in df.columns:
                continue
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            value = numeric_series.iloc[feature_row_index]
            row[column] = float(value) if np.isfinite(value) else np.nan
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No usable interpolated rows found in {raw_input_dir}")
    if not feature_columns:
        raise RuntimeError(f"No usable interpolated features found in {raw_input_dir}")

    data = pd.DataFrame(rows)
    feature_columns = [column for column in feature_columns if column in data.columns]
    return data, feature_columns, skipped

def load_raw_dataframe(
    raw_input_dir: Path,
    train_cells: set[str],
    test_cells: set[str],
    feature_week_resolver: Callable[[str], float] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    rows = []
    skipped = {
        "read_fail": 0,
        "missing_core_columns": 0,
        "missing_feature_columns": 0,
        "invalid_target": 0,
        "target_before_feature_week": 0,
        "feature_week_out_of_range": 0,
    }
    feature_columns: list[str] | None = None
    resolved_feature_week = feature_week_resolver or get_raw_feature_week

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            skipped["read_fail"] += 1
            continue

        if RAW_WEEK_COLUMN not in df.columns or RAW_CAPACITY_COLUMN not in df.columns:
            skipped["missing_core_columns"] += 1
            continue

        current_feature_columns = get_feature_columns(df.columns.tolist())
        if not current_feature_columns:
            skipped["missing_feature_columns"] += 1
            continue

        weeks = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
        capacities = pd.to_numeric(df[RAW_CAPACITY_COLUMN], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(capacities).any() and np.isfinite(capacities[0]) and capacities[0] != 0:
            df["SOH"] = capacities / capacities[0]
        week_at_target = interpolate_value_at_soh(weeks, weeks, capacities, TARGET_SOH)
        if not np.isfinite(week_at_target) or week_at_target <= 0:
            skipped["invalid_target"] += 1
            continue
        feature_week = resolved_feature_week(cell)
        if np.isfinite(feature_week) and float(week_at_target) <= float(feature_week):
            skipped["target_before_feature_week"] += 1
            continue
        finite_weeks = weeks[np.isfinite(weeks)]
        if (
            not np.isfinite(feature_week)
            or finite_weeks.size == 0
            or float(feature_week) < float(np.min(finite_weeks)) - 1e-9
            or float(feature_week) > float(np.max(finite_weeks)) + 1e-9
        ):
            skipped["feature_week_out_of_range"] += 1
            continue
        if "SOH" in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES and "SOH" in df.columns and "SOH" not in current_feature_columns:
            current_feature_columns = ["SOH", *current_feature_columns]
        if RAW_CAPACITY_COLUMN in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES and RAW_CAPACITY_COLUMN in df.columns and RAW_CAPACITY_COLUMN not in current_feature_columns:
            current_feature_columns = [RAW_CAPACITY_COLUMN, *current_feature_columns]

        if feature_columns is None:
            feature_columns = current_feature_columns
        else:
            feature_columns = [column for column in feature_columns if column in current_feature_columns]

        row = {
            "Cell": cell,
            RAW_TARGET_COLUMN: float(week_at_target),
            "is_outer_field_selected": cell in train_cells,
            "is_test_cell": cell in test_cells,
            "source_file": str(csv_path),
        }
        for column in current_feature_columns:
            feature_values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            row[column] = interpolate_value_at_x(weeks, feature_values, feature_week)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No usable raw rows found in {raw_input_dir}")
    if not feature_columns:
        raise RuntimeError(f"No usable raw features found in {raw_input_dir}")

    data = pd.DataFrame(rows)
    feature_columns = [column for column in feature_columns if column in data.columns]
    return data, feature_columns, skipped

def load_interpolated_sequence_dataset(
    raw_input_dir: Path,
    input_steps: int,
    sequence_features: list[str],
    train_cells: set[str],
    test_cells: set[str],
    capa_lim: float,
    interpolation_type: str,
    discretization_count: int = 10,
    target_soh: float = TARGET_SOH,
    target_column: str | None = None,
) -> tuple[dict[str, object], dict[str, int]]:
    resolved_target_column = str(target_column or get_interpolated_target_column(target_soh))
    sequences = []
    targets = []
    target_time_values = []
    time_curves = []
    weeks_per_step_values = []
    target_throughput_values = []
    throughput_curves = []
    throughput_per_step_values = []
    input_last_step_week_values = []
    cells = []
    skipped = {
        "read_fail": 0,
        "missing_core_columns": 0,
        "missing_features": 0,
        "too_short": 0,
        "invalid_target": 0,
        "nonfinite_sequence": 0,
    }

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            raw_df = load_cleaned_input_dataframe(csv_path)
            df = load_and_interpolate(
                raw_df,
                capa_lim=capa_lim,
                interpolation_type=interpolation_type,
                discretization_count=int(discretization_count),
            )
        except Exception:
            skipped["read_fail"] += 1
            continue

        if resolved_target_column not in df.columns:
            skipped["missing_core_columns"] += 1
            continue
        if any(feature not in df.columns for feature in sequence_features):
            skipped["missing_features"] += 1
            continue
        if len(df) < input_steps:
            skipped["too_short"] += 1
            continue

        target_value = pd.to_numeric(df[resolved_target_column], errors="coerce").dropna()
        if target_value.empty or float(target_value.iloc[0]) <= 0:
            skipped["invalid_target"] += 1
            continue
        raw_week_at_target = get_raw_week_at_target_from_dataframe(raw_df, target_soh)
        raw_throughput_at_target = get_raw_throughput_at_target_from_dataframe(raw_df, target_soh)
        if (
            not np.isfinite(raw_week_at_target)
            or raw_week_at_target <= 0
            or not np.isfinite(raw_throughput_at_target)
            or raw_throughput_at_target <= 0
        ):
            skipped["invalid_target"] += 1
            continue

        sequence = df.loc[: input_steps - 1, sequence_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(sequence).all():
            skipped["nonfinite_sequence"] += 1
            continue

        interpolated_target = float(target_value.iloc[0])
        weeks_per_step = float(raw_week_at_target) / interpolated_target
        throughput_per_step = float(raw_throughput_at_target) / interpolated_target
        time_curve = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
        throughput_curve = pd.to_numeric(df.get("current_throughput"), errors="coerce").to_numpy(dtype=float)
        if len(time_curve) == 0 or not np.isfinite(time_curve).any():
            skipped["nonfinite_sequence"] += 1
            continue
        if len(throughput_curve) == 0 or not np.isfinite(throughput_curve).any():
            skipped["nonfinite_sequence"] += 1
            continue
        sequences.append(sequence)
        targets.append(interpolated_target)
        target_time_values.append(float(raw_week_at_target))
        time_curves.append(time_curve)
        weeks_per_step_values.append(float(weeks_per_step))
        target_throughput_values.append(float(raw_throughput_at_target))
        throughput_curves.append(throughput_curve)
        throughput_per_step_values.append(float(throughput_per_step))
        input_last_step_week_values.append(float(time_curve[input_steps - 1]) if np.isfinite(time_curve[input_steps - 1]) else np.nan)
        cells.append(cell)

    if not sequences:
        raise RuntimeError(f"No usable interpolated sequences found in {raw_input_dir}")

    return (
        {
            "sequences": np.asarray(sequences, dtype=float),
            "targets": np.asarray(targets, dtype=float),
            "target_time_values": np.asarray(target_time_values, dtype=float),
            "time_curves": list(time_curves),
            "weeks_per_step_values": np.asarray(weeks_per_step_values, dtype=float),
            "target_throughput_values": np.asarray(target_throughput_values, dtype=float),
            "throughput_curves": list(throughput_curves),
            "throughput_per_step_values": np.asarray(throughput_per_step_values, dtype=float),
            "input_last_step_week_values": np.asarray(input_last_step_week_values, dtype=float),
            "cells": list(cells),
            "train_mask": np.array([cell in train_cells for cell in cells], dtype=bool),
            "test_mask": np.array([cell in test_cells for cell in cells], dtype=bool),
            "feature_columns": list(sequence_features),
        },
        skipped,
    )

def load_raw_sequence_dataset(
    raw_input_dir: Path,
    input_steps: int,
    sequence_features: list[str],
    train_cells: set[str],
    test_cells: set[str],
    input_week_resolver: Callable[[str], float] | None = None,
) -> tuple[dict[str, object], dict[str, int]]:
    sequences = []
    time_grids = []
    targets = []
    input_last_step_week_values = []
    cells = []
    skipped = {
        "read_fail": 0,
        "missing_core_columns": 0,
        "missing_features": 0,
        "too_short": 0,
        "invalid_target": 0,
        "target_before_feature_week": 0,
        "nonfinite_sequence": 0,
    }

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            skipped["read_fail"] += 1
            continue

        if RAW_WEEK_COLUMN not in df.columns or RAW_CAPACITY_COLUMN not in df.columns:
            skipped["missing_core_columns"] += 1
            continue

        weeks = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
        capacities = pd.to_numeric(df[RAW_CAPACITY_COLUMN], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(capacities).any() and np.isfinite(capacities[0]) and capacities[0] != 0:
            df["SOH"] = capacities / capacities[0]
        if any(feature not in df.columns for feature in sequence_features):
            skipped["missing_features"] += 1
            continue
        feature_week = (input_week_resolver or get_raw_feature_week)(cell)
        if len(df) < 2 or not np.isfinite(weeks).any() or float(np.nanmax(weeks)) < feature_week:
            skipped["too_short"] += 1
            continue

        week_at_target = interpolate_value_at_soh(weeks, weeks, capacities, TARGET_SOH)
        if not np.isfinite(week_at_target) or week_at_target <= 0:
            skipped["invalid_target"] += 1
            continue
        sequence_grid = np.linspace(0.0, float(feature_week), input_steps)
        sequence_columns: list[np.ndarray] = []
        for feature in sequence_features:
            feature_values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(weeks) & np.isfinite(feature_values)
            if not finite_mask.any():
                sequence_columns.append(np.zeros_like(sequence_grid, dtype=float))
                continue
            first_feature_week = float(np.min(weeks[finite_mask]))
            zero_fill_before_start = raw_sequence_feature_should_zero_fill(feature)
            interpolated_feature = np.array(
                [
                    0.0
                    if zero_fill_before_start and float(target_x) < first_feature_week
                    else interpolate_value_at_x_with_edge_hold(weeks, feature_values, target_x)
                    for target_x in sequence_grid
                ],
                dtype=float,
            )
            interpolated_feature = np.nan_to_num(interpolated_feature, nan=0.0, posinf=0.0, neginf=0.0)
            sequence_columns.append(interpolated_feature)
        sequence = np.column_stack(sequence_columns)
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)

        sequences.append(sequence)
        time_grids.append(np.asarray(sequence_grid, dtype=float))
        targets.append(float(week_at_target))
        input_last_step_week_values.append(float(feature_week))
        cells.append(cell)

    if not sequences:
        raise RuntimeError(f"No usable raw sequences found in {raw_input_dir}")

    return (
        {
            "sequences": np.asarray(sequences, dtype=float),
            "time_grids": list(time_grids),
            "targets": np.asarray(targets, dtype=float),
            "input_last_step_week_values": np.asarray(input_last_step_week_values, dtype=float),
            "cells": list(cells),
            "train_mask": np.array([cell in train_cells for cell in cells], dtype=bool),
            "test_mask": np.array([cell in test_cells for cell in cells], dtype=bool),
            "feature_columns": list(sequence_features),
        },
        skipped,
    )

def build_interpolated_time_curve_dataframe(sequence_dataset: dict[str, object]) -> pd.DataFrame:
    cells = [str(cell) for cell in sequence_dataset.get("cells", [])]
    train_mask = np.asarray(sequence_dataset.get("train_mask", []), dtype=bool)
    test_mask = np.asarray(sequence_dataset.get("test_mask", []), dtype=bool)
    time_curves = [np.asarray(curve, dtype=float).reshape(-1) for curve in sequence_dataset.get("time_curves", [])]
    throughput_curves = [np.asarray(curve, dtype=float).reshape(-1) for curve in sequence_dataset.get("throughput_curves", [])]

    if (
        len(cells) == 0
        or len(train_mask) != len(cells)
        or len(test_mask) != len(cells)
        or len(time_curves) != len(cells)
        or len(throughput_curves) != len(cells)
    ):
        return pd.DataFrame(
            columns=["Cell", "split", "step_index", "time_weeks", "throughput"]
        )

    rows: list[dict[str, object]] = []
    for cell, is_train, is_test, time_curve, throughput_curve in zip(
        cells,
        train_mask,
        test_mask,
        time_curves,
        throughput_curves,
    ):
        split = "test" if bool(is_test) else "train" if bool(is_train) else "other"
        curve_length = min(len(time_curve), len(throughput_curve))
        for step_index in range(curve_length):
            time_value = time_curve[step_index]
            throughput_value = throughput_curve[step_index]
            rows.append(
                {
                    "Cell": cell,
                    "split": split,
                    "step_index": int(step_index),
                    "time_weeks": float(time_value) if np.isfinite(time_value) else np.nan,
                    "throughput": float(throughput_value) if np.isfinite(throughput_value) else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Cell", "split", "step_index", "time_weeks", "throughput"])
    return pd.DataFrame(rows)





def build_feature_subset(data: pd.DataFrame, feature: str, target_column: str) -> pd.DataFrame:
    subset = data[["Cell", "is_outer_field_selected", "is_test_cell", feature, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
    subset = subset[(subset[feature].abs() > 0) & (subset[target_column] > 0)].copy()
    if subset.empty:
        subset["x_log"] = pd.Series(dtype=float)
        subset["y_log"] = pd.Series(dtype=float)
        return subset
    subset["x_log"] = np.log(np.abs(subset[feature].to_numpy(dtype=float)))
    subset["y_log"] = np.log(subset[target_column].to_numpy(dtype=float))
    return subset

def choose_best_linear_feature(
    fit_summary: pd.DataFrame,
    *,
    data: pd.DataFrame | None = None,
    target_column: str | None = None,
) -> str:
    ranked = fit_summary.copy()
    ranked["pearson_r_train_abs"] = pd.to_numeric(ranked["pearson_r_train"], errors="coerce").abs()
    ranked = ranked.sort_values(
        ["pearson_r_train_abs", "feature"],
        ascending=[False, True],
        na_position="last",
    )
    for _, row in ranked.iterrows():
        feature = str(row["feature"]).strip()
        if feature and np.isfinite(pd.to_numeric(row.get("pearson_r_train_abs"), errors="coerce")):
            return feature
    raise RuntimeError("Could not determine a usable linear-model feature from threshold-filtered |pearson_r_train|.")

def choose_best_linear_model_by_rmse(
    candidate_models: list[dict[str, object]],
    *,
    model_variant: str = LINEAR_BEST_RMSE_MODEL_VARIANT,
    display_name: str = "Linear raw (best RMSE)",
    dataset_label: str | None = None,
    split: str = "test",
) -> dict[str, object]:
    metric_key = f"{str(split).strip().lower()}_rmse"
    ranked_candidates: list[tuple[float, float, str, dict[str, object]]] = []
    for model_result in candidate_models:
        rmse_value = pd.to_numeric(model_result.get(metric_key), errors="coerce")
        if not np.isfinite(rmse_value):
            continue
        secondary_mape = pd.to_numeric(model_result.get(f"{str(split).strip().lower()}_mape"), errors="coerce")
        selected_feature = str(model_result.get("selected_feature", "") or "").strip()
        ranked_candidates.append(
            (
                float(rmse_value),
                float(secondary_mape) if np.isfinite(secondary_mape) else math.inf,
                selected_feature,
                model_result,
            )
        )
    if not ranked_candidates:
        raise RuntimeError(f"Could not determine a usable best-{split}-RMSE linear-model feature.")

    ranked_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected_result = copy.deepcopy(ranked_candidates[0][3])
    selected_result["model_variant"] = str(model_variant)
    selected_result["display_name"] = str(display_name)
    if dataset_label is not None:
        selected_result["dataset_label"] = str(dataset_label)
    return selected_result

def get_linear_variant_name(feature_name: str | None) -> str:
    feature_text = str(feature_name).strip() if feature_name is not None else ""
    if not feature_text:
        return "best_r2"
    return f"fixed_{feature_text}"

def get_linear_display_name(feature_name: str | None) -> str:
    feature_text = str(feature_name).strip() if feature_name is not None else ""
    if not feature_text:
        return "Linear raw (best R^2)"
    return f"Linear raw ({feature_text})"

def get_raw_linear_fixed_feature_names(candidate_features: list[str] | None = None) -> list[str]:
    preferred_order = [
        feature_name
        for feature_name in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES
        if str(feature_name).startswith("mean_") or str(feature_name).startswith("var_")
    ]
    if candidate_features is None:
        return preferred_order

    available = {
        str(feature_name).strip()
        for feature_name in candidate_features
        if str(feature_name).strip().startswith("mean_") or str(feature_name).strip().startswith("var_")
    }
    ordered = [feature_name for feature_name in preferred_order if feature_name in available]
    extras = sorted(available.difference(ordered))
    return [*ordered, *extras]

def get_linear_variant_sort_key(model_variant: object) -> tuple[int, int, str]:
    variant_text = str(model_variant or "").strip()
    if variant_text == "best_r2":
        return (0, 0, "")
    if variant_text == "best_r2_tp100":
        return (0, 1, "")
    if variant_text == "best_r2_mintest":
        return (0, 2, "")
    if variant_text == "best_r2_halfmax":
        return (0, 3, "")
    if variant_text == "best_rmse_tp100":
        return (0, 4, "")
    if variant_text == "best_rmse_mintest":
        return (0, 5, "")
    if variant_text == "fixed_mintest_mean_delta_dqdv_mid":
        return (0, 6, "")
    if variant_text == "best_rmse_halfmax":
        return (0, 7, "")
    if variant_text == LINEAR_BEST_TRAIN_RMSE_MODEL_VARIANT:
        return (0, 8, "")
    if variant_text == LINEAR_BEST_RMSE_MODEL_VARIANT:
        return (0, 9, "")
    if variant_text == LINEAR_BEST_OUT_OF_RANGE_RMSE_MODEL_VARIANT:
        return (0, 10, "")
    if variant_text == LINEAR_EARLY_MODEL_VARIANT:
        return (0, 11, "")
    if variant_text.startswith("fixed_"):
        feature_name = variant_text.removeprefix("fixed_")
        fixed_feature_order = get_raw_linear_fixed_feature_names()
        feature_index = fixed_feature_order.index(feature_name) if feature_name in fixed_feature_order else len(fixed_feature_order)
        return (1, feature_index, feature_name)
    if variant_text == LINEAR_AVG_MODEL_VARIANT:
        return (2, 0, "")
    return (3, 0, variant_text)

def format_linear_row_label(row: pd.Series) -> str:
    threshold_value = int(float(row["threshold_weeks"]))
    variant_text = str(row.get("model_variant", "") or "").strip()
    if variant_text == "best_r2":
        model_text = "Linear best-R^2"
    elif variant_text == LINEAR_BEST_TRAIN_RMSE_MODEL_VARIANT:
        model_text = "Linear best-train-RMSE"
    elif variant_text == LINEAR_BEST_RMSE_MODEL_VARIANT:
        model_text = "Linear best-RMSE"
    elif variant_text == "fixed_mintest_mean_delta_dqdv_mid":
        model_text = "Linear @min-test mean_delta_dqdv_mid"
    elif variant_text == LINEAR_BEST_OUT_OF_RANGE_RMSE_MODEL_VARIANT:
        model_text = "Linear best-out-range-RMSE"
    elif variant_text == LINEAR_EARLY_MODEL_VARIANT:
        model_text = "Linear best-R^2 4w"
    elif variant_text == get_linear_variant_name(ADDITIONAL_RAW_LINEAR_FEATURE):
        model_text = f"Linear {ADDITIONAL_RAW_LINEAR_FEATURE}"
    elif variant_text == LINEAR_AVG_MODEL_VARIANT:
        model_text = "Linear mean/var avg"
    else:
        selected_feature = str(row.get("selected_feature", "") or "").strip()
        model_text = f"Linear {selected_feature}" if selected_feature else "Linear"
    return f"{model_text} {threshold_value}w\nRaw"


def get_linear_variant_color(model_variant: object) -> str:
    variant_text = str(model_variant or "").strip()
    if variant_text == "best_r2":
        return "#6FA2CC"
    if variant_text == "best_r2_tp100":
        return "#6FA2CC"
    if variant_text == "best_r2_mintest":
        return "#3ABABD"
    if variant_text == "best_r2_halfmax":
        return "#C19CC4"
    if variant_text == "best_rmse_tp100":
        return "#6FA2CC"
    if variant_text == "best_rmse_mintest":
        return "#3ABABD"
    if variant_text == "fixed_mintest_mean_delta_dqdv_mid":
        return "#3ABABD"
    if variant_text == "best_rmse_halfmax":
        return "#C19CC4"
    if variant_text == LINEAR_BEST_TRAIN_RMSE_MODEL_VARIANT:
        return "#F6C85F"
    if variant_text == LINEAR_BEST_RMSE_MODEL_VARIANT:
        return "#F2A65A"
    if variant_text == LINEAR_BEST_OUT_OF_RANGE_RMSE_MODEL_VARIANT:
        return "#C19CC4"
    if variant_text == LINEAR_EARLY_MODEL_VARIANT:
        return "#F3E9E0"
    if variant_text == LINEAR_AVG_MODEL_VARIANT:
        return "#3ABABD"
    if variant_text.startswith("elasticnet"):
        return "#EF8881"
    if variant_text.startswith("svr"):
        return "#DAD3D3"
    if variant_text.startswith("fixed_"):
        feature_name = variant_text.removeprefix("fixed_")
        fixed_feature_order = get_raw_linear_fixed_feature_names()
        feature_index = fixed_feature_order.index(feature_name) if feature_name in fixed_feature_order else len(fixed_feature_order)
        return plt.get_cmap("tab20")(feature_index % 20)
    return "#7A7A7A"

def select_features_by_abs_pearson(
    fit_summary: pd.DataFrame,
    candidate_features: list[str],
    min_abs_pearson: float,
    label: str,
) -> tuple[list[str], pd.DataFrame]:
    if not candidate_features:
        raise RuntimeError(f"No candidate features were provided for {label}.")

    summary_subset = fit_summary[fit_summary["feature"].isin(candidate_features)].copy()
    if summary_subset.empty:
        raise RuntimeError(f"No fit-summary rows were available for {label}.")

    summary_subset["pearson_r_train_abs"] = pd.to_numeric(
        summary_subset["pearson_r_train"],
        errors="coerce",
    ).abs()
    selected_summary = (
        summary_subset[summary_subset["pearson_r_train_abs"] > float(min_abs_pearson)]
        .copy()
        .sort_values(["pearson_r_train_abs", "feature"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    selected_features = [feature for feature in candidate_features if feature in set(selected_summary["feature"].astype(str))]

    if not selected_features:
        available_text = summary_subset[["feature", "pearson_r_train"]].sort_values(
            ["pearson_r_train"],
            ascending=[False],
            na_position="last",
        ).to_string(index=False)
        raise RuntimeError(
            f"No {label} features met |pearson_r_train| > {float(min_abs_pearson):.2f}.\n"
            f"Available Pearson values:\n{available_text}"
        )

    selected_summary = selected_summary[
        ["feature", "pearson_r_train", "pearson_r_train_abs", "r2_train", "n_train", "r2_all", "pearson_r_all", "n_all"]
    ].copy()
    return selected_features, selected_summary


def train_linear_feature_regressor(data: pd.DataFrame, fit_summary: pd.DataFrame, target_column: str,
    dataset_label: str, feature_name: str | None = None,
    input_week_resolver: Callable[[str], float] | None = None, model_variant_override: str | None = None,
    display_name_override: str | None = None, raw_feature_week_default: float | None = None,
    raw_feature_week_post_g20: float | None = None,
    linear_input_week_variant: str = LINEAR_INPUT_WEEK_VARIANT_DEFAULT) -> dict[str, object]:
    """Adapter onto the shared linear regressor in core_functions.temporal_models.

    Resolves this pipeline's input-week bookkeeping defaults before delegating, so the extra
    week columns keep exactly the values they had.
    """
    g = globals()
    requested = str(feature_name).strip() if feature_name is not None else None
    return temporal_models.train_linear_feature_regressor(data=data, fit_summary=fit_summary,
        target_column=target_column, dataset_label=dataset_label, feature_name=feature_name,
        model_name="linear", train_column="is_outer_field_selected",
        input_week_resolver=input_week_resolver or get_linear_raw_feature_week,
        model_variant=(str(model_variant_override).strip() if model_variant_override is not None
            else get_linear_variant_name(requested)),
        display_name=(str(display_name_override).strip() if display_name_override is not None
            else get_linear_display_name(requested)),
        raw_feature_week_default=(float(raw_feature_week_default) if raw_feature_week_default is not None
            else float(LINEAR_RAW_FEATURE_WEEK)),
        raw_feature_week_post_g20=(float(raw_feature_week_post_g20) if raw_feature_week_post_g20 is not None
            else float(LINEAR_RAW_FEATURE_WEEK_POST_G20)),
        linear_input_week_variant=str(linear_input_week_variant),
        choose_best_feature_fn=g["choose_best_linear_feature"], build_feature_subset_fn=g["build_feature_subset"],
        calculate_ape_metrics=g.get("calculate_ape_metrics"), calculate_r2=g.get("calculate_r2"))

def train_linear_feature_average_regressor(
    data: pd.DataFrame,
    fit_summary: pd.DataFrame,
    target_column: str,
    dataset_label: str,
    feature_names: list[str] | None = None,
    input_week_resolver: Callable[[str], float] | None = None,
    raw_feature_week_default: float | None = None,
    raw_feature_week_post_g20: float | None = None,
    linear_input_week_variant: str = LINEAR_INPUT_WEEK_VARIANT_DEFAULT,
) -> dict[str, object]:
    candidate_features = feature_names if feature_names is not None else fit_summary["feature"].astype(str).tolist()
    candidate_features = [feature for feature in candidate_features if feature in data.columns]
    resolved_input_week = input_week_resolver or get_linear_raw_feature_week
    resolved_week_default = float(raw_feature_week_default) if raw_feature_week_default is not None else float(LINEAR_RAW_FEATURE_WEEK)
    resolved_week_post_g20 = float(raw_feature_week_post_g20) if raw_feature_week_post_g20 is not None else float(LINEAR_RAW_FEATURE_WEEK_POST_G20)
    component_models: list[dict[str, object]] = []
    for feature_name in candidate_features:
        try:
            component_model = train_linear_feature_regressor(
                data=data,
                fit_summary=fit_summary,
                target_column=target_column,
                dataset_label=f"{dataset_label}_{feature_name}",
                feature_name=feature_name,
                input_week_resolver=resolved_input_week,
                raw_feature_week_default=resolved_week_default,
                raw_feature_week_post_g20=resolved_week_post_g20,
                linear_input_week_variant=linear_input_week_variant,
            )
        except RuntimeError:
            continue
        component_models.append(component_model)

    if not component_models:
        raise RuntimeError(f"No usable raw linear feature models were available for {dataset_label}.")

    train_order = data[data["is_outer_field_selected"]]["Cell"].astype(str).tolist()
    test_order = data[data["is_test_cell"]]["Cell"].astype(str).tolist()

    def aggregate_split(split_name: str, preferred_order: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
        by_cell: dict[str, dict[str, object]] = {}
        for component_model in component_models:
            cells = [str(cell) for cell in component_model.get(f"{split_name}_cell_list", [])]
            actual_values = np.asarray(component_model.get(f"{split_name}_actual", []), dtype=float)
            pred_values = np.asarray(component_model.get(f"{split_name}_pred", []), dtype=float)
            if len(cells) != len(actual_values) or len(cells) != len(pred_values):
                continue
            for cell, actual_value, pred_value in zip(cells, actual_values, pred_values):
                if not np.isfinite(actual_value) or not np.isfinite(pred_value):
                    continue
                entry = by_cell.setdefault(cell, {"actual": float(actual_value), "pred_values": []})
                entry["actual"] = float(actual_value)
                cast_list = entry["pred_values"]
                if isinstance(cast_list, list):
                    cast_list.append(float(pred_value))

        ordered_cells: list[str] = []
        actual_list: list[float] = []
        pred_list: list[float] = []
        for cell in preferred_order:
            entry = by_cell.get(cell)
            if entry is None:
                continue
            pred_values = entry["pred_values"]
            if not isinstance(pred_values, list) or not pred_values:
                continue
            ordered_cells.append(cell)
            actual_list.append(float(entry["actual"]))
            pred_list.append(float(np.mean(np.asarray(pred_values, dtype=float))))

        remaining_cells = sorted(cell for cell in by_cell.keys() if cell not in set(ordered_cells))
        for cell in remaining_cells:
            entry = by_cell[cell]
            pred_values = entry["pred_values"]
            if not isinstance(pred_values, list) or not pred_values:
                continue
            ordered_cells.append(cell)
            actual_list.append(float(entry["actual"]))
            pred_list.append(float(np.mean(np.asarray(pred_values, dtype=float))))

        return ordered_cells, np.asarray(actual_list, dtype=float), np.asarray(pred_list, dtype=float)

    train_cells, train_actual, train_pred = aggregate_split("train", train_order)
    test_cells, test_actual, test_pred = aggregate_split("test", test_order)
    if len(train_cells) < 2:
        raise RuntimeError(f"Not enough training rows after averaging for {dataset_label}.")
    if len(test_cells) == 0:
        raise RuntimeError(f"No test rows after averaging for {dataset_label}.")

    train_mape, train_max_ape = calculate_ape_metrics(train_actual, train_pred)
    test_mape, test_max_ape = calculate_ape_metrics(test_actual, test_pred)

    return {
        "model_name": "linear",
        "model_variant": LINEAR_AVG_MODEL_VARIANT,
        "display_name": "Linear raw (mean/var avg)",
        "dataset_label": dataset_label,
        "selected_feature": "mean_var_feature_average",
        "component_models": [
            {
                "selected_feature": str(model.get("selected_feature", "")),
                "slope": float(model.get("slope", np.nan)),
                "intercept": float(model.get("intercept", np.nan)),
            }
            for model in component_models
            if np.isfinite(float(model.get("slope", np.nan))) and np.isfinite(float(model.get("intercept", np.nan)))
        ],
        "component_features": [str(model.get("selected_feature", "")) for model in component_models],
        "component_model_count": len(component_models),
        "target_column": target_column,
        "raw_feature_week_default": resolved_week_default,
        "raw_feature_week_post_g20": resolved_week_post_g20,
        "linear_input_week_variant": str(linear_input_week_variant),
        "feature_count": len(component_models),
        "train_count": len(train_cells),
        "test_count": len(test_cells),
        "train_mape": train_mape,
        "test_mape": test_mape,
        "train_rmse": calculate_rmse(train_actual, train_pred),
        "test_rmse": calculate_rmse(test_actual, test_pred),
        "train_max_ape": train_max_ape,
        "test_max_ape": test_max_ape,
        "train_max_ape_cell": calculate_max_ape_cell(train_actual, train_pred, train_cells),
        "test_max_ape_cell": calculate_max_ape_cell(test_actual, test_pred, test_cells),
        "train_r2": calculate_r2(train_actual, train_pred),
        "test_r2": calculate_r2(test_actual, test_pred),
        "epochs": np.nan,
        "learning_rate": np.nan,
        "train_actual": train_actual,
        "train_pred": train_pred,
        "test_actual": test_actual,
        "test_pred": test_pred,
        "train_cell_list": train_cells,
        "test_cell_list": test_cells,
        "train_input_last_step_week": np.asarray(
            [resolved_input_week(cell) for cell in train_cells],
            dtype=float,
        ),
        "test_input_last_step_week": np.asarray(
            [resolved_input_week(cell) for cell in test_cells],
            dtype=float,
        ),
    }

def _build_all_feature_log_dataset(
    *,
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    dataset_label: str,
    model_label: str,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_features = [str(feature) for feature in feature_columns if str(feature) in data.columns]
    if not selected_features:
        raise RuntimeError(f"No {model_label} features are available for {dataset_label}.")

    required_columns = ["Cell", "is_outer_field_selected", "is_test_cell", target_column, *selected_features]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise RuntimeError(f"{model_label} dataset {dataset_label} is missing columns: {missing_columns}")

    dataset = data[required_columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if dataset.empty:
        raise RuntimeError(f"No finite {model_label} rows are available for {dataset_label}.")

    feature_values = dataset[selected_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    target_values = pd.to_numeric(dataset[target_column], errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(feature_values).all(axis=1) & np.isfinite(target_values) & (target_values > 0.0)
    dataset = dataset.loc[valid_mask].copy()
    feature_values = feature_values[valid_mask]
    target_values = target_values[valid_mask]
    if dataset.empty:
        raise RuntimeError(f"No positive-target {model_label} rows are available for {dataset_label}.")

    train_mask = dataset["is_outer_field_selected"].astype(bool).to_numpy()
    test_mask = dataset["is_test_cell"].astype(bool).to_numpy()
    if int(train_mask.sum()) < 2:
        raise RuntimeError(f"Not enough training rows for {dataset_label} {model_label} model.")
    if int(test_mask.sum()) == 0:
        raise RuntimeError(f"No test rows available for {dataset_label} {model_label} model.")

    x_log = np.log(np.maximum(np.abs(feature_values), 1e-30))
    y_log = np.log(target_values)
    return dataset, selected_features, x_log, y_log, target_values, train_mask, test_mask

def _build_all_feature_model_result(
    *,
    model_name: str,
    model_variant: str,
    display_name: str,
    dataset_label: str,
    selected_features: list[str],
    target_column: str,
    dataset: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    train_actual: np.ndarray,
    train_pred: np.ndarray,
    test_actual: np.ndarray,
    test_pred: np.ndarray,
    input_week_resolver: Callable[[str], float] | None,
    raw_feature_week_default: float | None,
    raw_feature_week_post_g20: float | None,
    linear_input_week_variant: str,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    resolved_input_week = input_week_resolver or get_linear_raw_feature_week
    resolved_week_default = float(raw_feature_week_default) if raw_feature_week_default is not None else float(LINEAR_RAW_FEATURE_WEEK)
    resolved_week_post_g20 = float(raw_feature_week_post_g20) if raw_feature_week_post_g20 is not None else float(LINEAR_RAW_FEATURE_WEEK_POST_G20)
    train_cells = dataset.loc[train_mask, "Cell"].astype(str).tolist()
    test_cells = dataset.loc[test_mask, "Cell"].astype(str).tolist()
    train_mape, train_max_ape = calculate_ape_metrics(train_actual, train_pred)
    test_mape, test_max_ape = calculate_ape_metrics(test_actual, test_pred)
    result: dict[str, object] = {
        "model_name": str(model_name),
        "model_variant": str(model_variant),
        "display_name": str(display_name),
        "dataset_label": str(dataset_label),
        "selected_feature": "all_features",
        "selected_features": json.dumps(selected_features),
        "target_column": str(target_column),
        "raw_feature_week_default": resolved_week_default,
        "raw_feature_week_post_g20": resolved_week_post_g20,
        "linear_input_week_variant": str(linear_input_week_variant),
        "feature_count": len(selected_features),
        "train_count": int(train_mask.sum()),
        "test_count": int(test_mask.sum()),
        "train_mape": train_mape,
        "test_mape": test_mape,
        "train_rmse": calculate_rmse(train_actual, train_pred),
        "test_rmse": calculate_rmse(test_actual, test_pred),
        "train_max_ape": train_max_ape,
        "test_max_ape": test_max_ape,
        "train_max_ape_cell": calculate_max_ape_cell(train_actual, train_pred, train_cells),
        "test_max_ape_cell": calculate_max_ape_cell(test_actual, test_pred, test_cells),
        "train_r2": calculate_r2(train_actual, train_pred),
        "test_r2": calculate_r2(test_actual, test_pred),
        "epochs": np.nan,
        "learning_rate": np.nan,
        "train_actual": train_actual,
        "train_pred": train_pred,
        "test_actual": test_actual,
        "test_pred": test_pred,
        "train_cell_list": train_cells,
        "test_cell_list": test_cells,
        "train_input_last_step_week": np.asarray([resolved_input_week(cell) for cell in train_cells], dtype=float),
        "test_input_last_step_week": np.asarray([resolved_input_week(cell) for cell in test_cells], dtype=float),
    }
    if extra_fields:
        result.update(extra_fields)
    return result

def train_elasticnet_all_feature_regressor(
    *,
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    dataset_label: str,
    input_week_resolver: Callable[[str], float] | None = None,
    model_name: str = "elasticnet",
    model_variant: str = "elasticnet_all_features",
    display_name: str = "Elastic Net raw (all features)",
    raw_feature_week_default: float | None = None,
    raw_feature_week_post_g20: float | None = None,
    linear_input_week_variant: str = LINEAR_INPUT_WEEK_VARIANT_DEFAULT,
) -> dict[str, object]:
    dataset, selected_features, x_log, _y_log, target_values, train_mask, test_mask = _build_all_feature_log_dataset(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        dataset_label=dataset_label,
        model_label="Elastic Net",
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_log[train_mask])
    x_test = scaler.transform(x_log[test_mask])
    target_transformer = make_model_target_transformer(target_values[train_mask])
    y_transformed = target_transformer.transform(target_values)
    model = ElasticNet(
        alpha=float(aug_cfg.ELASTICNET_ALPHA),
        l1_ratio=float(aug_cfg.ELASTICNET_L1_RATIO),
        fit_intercept=True,
        max_iter=int(aug_cfg.ELASTICNET_MAX_ITER),
        random_state=int(aug_cfg.ELASTICNET_RANDOM_STATE),
    )
    model.fit(x_train, y_transformed[train_mask])

    train_pred = target_transformer.inverse_transform(model.predict(x_train))
    test_pred = target_transformer.inverse_transform(model.predict(x_test))
    train_actual = target_values[train_mask]
    test_actual = target_values[test_mask]
    coefficients = np.asarray(model.coef_, dtype=float)
    return _build_all_feature_model_result(
        model_name=model_name,
        model_variant=model_variant,
        display_name=display_name,
        dataset_label=dataset_label,
        selected_features=selected_features,
        target_column=target_column,
        dataset=dataset,
        train_mask=train_mask,
        test_mask=test_mask,
        train_actual=train_actual,
        train_pred=train_pred,
        test_actual=test_actual,
        test_pred=test_pred,
        input_week_resolver=input_week_resolver,
        raw_feature_week_default=raw_feature_week_default,
        raw_feature_week_post_g20=raw_feature_week_post_g20,
        linear_input_week_variant=linear_input_week_variant,
        extra_fields={
            "elasticnet_alpha": float(aug_cfg.ELASTICNET_ALPHA),
            "elasticnet_l1_ratio": float(aug_cfg.ELASTICNET_L1_RATIO),
            "elasticnet_target_transform": target_transformer.mode,
            "elasticnet_target_divisor": float(target_transformer.target_divisor_),
            "elasticnet_nonzero_feature_count": int(np.count_nonzero(np.abs(coefficients) > 1e-12)),
            "elasticnet_feature_coefficients": json.dumps(
                {feature: float(coef) for feature, coef in zip(selected_features, coefficients)}
            ),
        },
    )

def train_svr_all_feature_regressor(
    *,
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    dataset_label: str,
    input_week_resolver: Callable[[str], float] | None = None,
    model_name: str = "svr",
    model_variant: str = "svr_all_features",
    display_name: str = "SVR raw (all features)",
    raw_feature_week_default: float | None = None,
    raw_feature_week_post_g20: float | None = None,
    linear_input_week_variant: str = LINEAR_INPUT_WEEK_VARIANT_DEFAULT,
) -> dict[str, object]:
    dataset, selected_features, x_log, _y_log, target_values, train_mask, test_mask = _build_all_feature_log_dataset(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        dataset_label=dataset_label,
        model_label="SVR",
    )
    scaler = MinMaxScaler()
    x_values = dataset[selected_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_train = scaler.fit_transform(x_values[train_mask])
    x_test = scaler.transform(x_values[test_mask])
    model = SVR(
        kernel=str(aug_cfg.SVR_KERNEL),
        C=float(aug_cfg.SVR_C),
        epsilon=float(aug_cfg.SVR_EPSILON),
        gamma=aug_cfg.SVR_GAMMA,
    )
    y_train = target_values[train_mask]
    target_transformer = make_model_target_transformer(y_train)
    model.fit(x_train, target_transformer.transform(y_train))

    train_pred_raw = target_transformer.inverse_transform(model.predict(x_train))
    test_pred_raw = target_transformer.inverse_transform(model.predict(x_test))
    train_pred = np.maximum(train_pred_raw, 1e-9)
    test_pred = np.maximum(test_pred_raw, 1e-9)
    train_actual = target_values[train_mask]
    test_actual = target_values[test_mask]
    return _build_all_feature_model_result(
        model_name=model_name,
        model_variant=model_variant,
        display_name=display_name,
        dataset_label=dataset_label,
        selected_features=selected_features,
        target_column=target_column,
        dataset=dataset,
        train_mask=train_mask,
        test_mask=test_mask,
        train_actual=train_actual,
        train_pred=train_pred,
        test_actual=test_actual,
        test_pred=test_pred,
        input_week_resolver=input_week_resolver,
        raw_feature_week_default=raw_feature_week_default,
        raw_feature_week_post_g20=raw_feature_week_post_g20,
        linear_input_week_variant=linear_input_week_variant,
        extra_fields={
            "svr_kernel": str(aug_cfg.SVR_KERNEL),
            "svr_c": float(aug_cfg.SVR_C),
            "svr_epsilon": float(aug_cfg.SVR_EPSILON),
            "svr_gamma": str(aug_cfg.SVR_GAMMA),
            "svr_input_transform": "raw_minmax_scaler",
            "svr_target_transform": target_transformer.mode,
            "svr_target_divisor": float(target_transformer.target_divisor_),
            "svr_target_scaler_min": float(target_transformer.data_min_[0]),
            "svr_target_scaler_max": float(target_transformer.data_max_[0]),
            "svr_support_vector_count": int(len(model.support_)),
        },
    )

def summarize_model_rows(threshold: float, model_results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in model_results:
        model_name = result.get("model_name")
        if not model_name:
            model_name = "mlp"
        test_ape_values = extract_test_ape(result)
        long_eol_ape_values = np.asarray(result.get("long_eol_test_ape_values", []), dtype=float)
        extra_long_eol_ape_values = np.asarray(result.get("extra_long_eol_test_ape_values", []), dtype=float)
        test_cells = [str(cell) for cell in result.get("test_cell_list", [])]
        test_max_rmse_error_cell = result.get("test_max_rmse_error_cell")
        if test_max_rmse_error_cell in (None, ""):
            test_max_rmse_error_cell = get_max_rmse_error_cell(result, "test")
        test_max_ape_cell = result.get("test_max_ape_cell")
        if test_max_ape_cell in (None, ""):
            test_max_ape_cell = calculate_max_ape_cell(
                np.asarray(result.get("test_actual", []), dtype=float),
                np.asarray(result.get("test_pred", []), dtype=float),
                test_cells,
            )
        long_eol_test_cells = [str(cell) for cell in result.get("long_eol_test_cells", [])]
        long_eol_test_max_rmse_error_cell = result.get("long_eol_test_max_rmse_error_cell")
        long_eol_test_max_ape_cell = result.get("long_eol_test_max_ape_cell")
        if long_eol_test_max_ape_cell in (None, ""):
            long_eol_test_max_ape_cell = calculate_max_ape_cell(
                np.asarray(result.get("test_actual", []), dtype=float),
                np.asarray(result.get("test_pred", []), dtype=float),
                long_eol_test_cells if long_eol_test_cells else test_cells,
            )
        rows.append(
            {
                "threshold_weeks": threshold,
                "model_name": model_name,
                "model_variant": result.get("model_variant"),
                "display_name": result.get("display_name"),
                "dataset_label": result.get("dataset_label"),
                "selected_feature": result.get("selected_feature"),
                "selected_features": result.get("selected_features"),
                "feature_count": result.get("feature_count"),
                "elasticnet_alpha": result.get("elasticnet_alpha"),
                "elasticnet_l1_ratio": result.get("elasticnet_l1_ratio"),
                "elasticnet_nonzero_feature_count": result.get("elasticnet_nonzero_feature_count"),
                "elasticnet_feature_coefficients": result.get("elasticnet_feature_coefficients"),
                "svr_kernel": result.get("svr_kernel"),
                "svr_c": result.get("svr_c"),
                "svr_epsilon": result.get("svr_epsilon"),
                "svr_gamma": result.get("svr_gamma"),
                "svr_support_vector_count": result.get("svr_support_vector_count"),
                "train_count": result.get("train_count"),
                "test_count": result.get("test_count"),
                "train_mape": result.get("train_mape"),
                "test_mape": result.get("test_mape"),
                "train_rmse": result.get("train_rmse"),
                "test_rmse": result.get("test_rmse"),
                "train_time_rmse": result.get("train_time_rmse"),
                "test_time_rmse": result.get("test_time_rmse"),
                "train_throughput_rmse": result.get("train_throughput_rmse"),
                "test_throughput_rmse": result.get("test_throughput_rmse"),
                "train_plot_rmse": get_preferred_rmse_value(result, "train"),
                "test_plot_rmse": get_preferred_rmse_value(result, "test"),
                "test_max_rmse_error": get_max_rmse_error_value(result, "test"),
                "test_max_rmse_error_cell": test_max_rmse_error_cell,
                "test_cells": test_cells,
                "test_actual": _json_ready(result.get("test_actual", [])),
                "test_pred": _json_ready(result.get("test_pred", [])),
                "test_actual_time": _json_ready(result.get("test_actual_time", [])),
                "test_pred_time": _json_ready(result.get("test_pred_time", [])),
                "test_actual_throughput": _json_ready(result.get("test_actual_throughput", [])),
                "test_pred_throughput": _json_ready(result.get("test_pred_throughput", [])),
                "test_ape_values": test_ape_values.tolist(),
                "test_ape_std": float(np.std(test_ape_values)) if test_ape_values.size else np.nan,
                "train_max_ape": result.get("train_max_ape"),
                "test_max_ape": result.get("test_max_ape"),
                "test_max_ape_cell": test_max_ape_cell,
                "long_eol_threshold_weeks": result.get("long_eol_threshold_weeks"),
                "long_eol_test_count": result.get("long_eol_test_count"),
                "long_eol_test_mape": result.get("long_eol_test_mape"),
                "long_eol_test_rmse": result.get("long_eol_test_rmse"),
                "long_eol_test_throughput_rmse": result.get("long_eol_test_throughput_rmse"),
                "long_eol_test_max_rmse_error": result.get("long_eol_test_max_rmse_error"),
                "long_eol_test_max_rmse_error_cell": long_eol_test_max_rmse_error_cell,
                "long_eol_test_ape_values": long_eol_ape_values.tolist(),
                "long_eol_test_cells": long_eol_test_cells,
                "long_eol_test_ape_std": float(np.std(long_eol_ape_values)) if long_eol_ape_values.size else np.nan,
                "long_eol_test_max_ape": result.get("long_eol_test_max_ape"),
                "long_eol_test_max_ape_cell": long_eol_test_max_ape_cell,
                "extra_long_eol_threshold_weeks": result.get("extra_long_eol_threshold_weeks"),
                "extra_long_eol_test_count": result.get("extra_long_eol_test_count"),
                "extra_long_eol_test_mape": result.get("extra_long_eol_test_mape"),
                "extra_long_eol_test_rmse": result.get("extra_long_eol_test_rmse"),
                "extra_long_eol_test_throughput_rmse": result.get("extra_long_eol_test_throughput_rmse"),
                "extra_long_eol_test_max_rmse_error": result.get("extra_long_eol_test_max_rmse_error"),
                "extra_long_eol_test_max_rmse_error_cell": result.get("extra_long_eol_test_max_rmse_error_cell"),
                "extra_long_eol_test_ape_std": float(np.std(extra_long_eol_ape_values)) if extra_long_eol_ape_values.size else np.nan,
                "extra_long_eol_test_max_ape": result.get("extra_long_eol_test_max_ape"),
                "extra_long_eol_test_max_ape_cell": result.get("extra_long_eol_test_max_ape_cell"),
                "train_r2": result.get("train_r2"),
                "test_r2": result.get("test_r2"),
                "epochs": result.get("epochs"),
                "learning_rate": result.get("learning_rate"),
                "target_soh": result.get("target_soh"),
            }
        )
    return pd.DataFrame(rows)

def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value

def _first_float(value, default: float = np.nan) -> float:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return float(default)
    if array.size == 0 or not np.isfinite(array[0]):
        return float(default)
    return float(array[0])

def save_linear_model_artifact(model_result: dict[str, object], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": str(model_result.get("model_name", "linear")),
        "model_variant": str(model_result.get("model_variant", "")),
        "display_name": str(model_result.get("display_name", "")),
        "dataset_label": str(model_result.get("dataset_label", "")),
        "selected_feature": str(model_result.get("selected_feature", "")),
        "target_column": str(model_result.get("target_column", RAW_TARGET_COLUMN)),
        "raw_feature_week_default": float(model_result.get("raw_feature_week_default", LINEAR_RAW_FEATURE_WEEK)),
        "raw_feature_week_post_g20": float(model_result.get("raw_feature_week_post_g20", LINEAR_RAW_FEATURE_WEEK_POST_G20)),
        "linear_input_week_variant": str(model_result.get("linear_input_week_variant", LINEAR_INPUT_WEEK_VARIANT_DEFAULT)),
        "train_count": int(model_result.get("train_count", 0)),
        "test_count": int(model_result.get("test_count", 0)),
    }
    if str(model_result.get("model_variant", "")) == LINEAR_AVG_MODEL_VARIANT:
        payload["component_models"] = [
            {
                "selected_feature": str(item.get("selected_feature", "")),
                "slope": float(item.get("slope", np.nan)),
                "intercept": float(item.get("intercept", np.nan)),
            }
            for item in model_result.get("component_models", [])
            if str(item.get("selected_feature", "")).strip()
        ]
        payload["component_features"] = [str(item.get("selected_feature", "")) for item in payload["component_models"]]
        payload["component_model_count"] = int(model_result.get("component_model_count", len(payload["component_models"])))
    elif "slope" in model_result and "intercept" in model_result:
        payload["slope"] = float(model_result["slope"])
        payload["intercept"] = float(model_result["intercept"])
    else:
        payload["selected_features"] = model_result.get("selected_features", "")
        for key in (
            "elasticnet_alpha",
            "elasticnet_l1_ratio",
            "elasticnet_nonzero_feature_count",
            "elasticnet_feature_coefficients",
            "svr_kernel",
            "svr_c",
            "svr_epsilon",
            "svr_gamma",
            "svr_support_vector_count",
        ):
            if key in model_result:
                payload[key] = _json_ready(model_result[key])
    with open(artifact_dir / "linear_model.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

def save_tcnn_ensemble_artifact(model_result: dict[str, object], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_artifacts = list(model_result.get("model_artifacts", []))
    first_state_dict = model_artifacts[0]["state_dict"] if model_artifacts and "state_dict" in model_artifacts[0] else {}
    conv1_weight = first_state_dict.get("conv1.weight")
    conv2_weight = first_state_dict.get("conv2.weight")
    metadata = {
        "model_name": str(model_result.get("model_name", "tcnn")),
        "dataset_label": str(model_result.get("dataset_label", "")),
        "feature_columns": list(model_result.get("feature_columns", [])),
        "input_steps": int(model_result.get("input_steps", 0)),
        "conv1_channels": int(model_result.get("conv1_channels", 0)),
        "conv_channels": int(model_result.get("conv_channels", 0)),
        "fc_hidden_dim": int(model_result.get("fc_hidden_dim", 0)),
        "fc_hidden_dim2": int(model_result.get("fc_hidden_dim2", 0)),
        "conv1_kernel_size": int(conv1_weight.shape[-1]) if conv1_weight is not None else 5,
        "conv2_kernel_size": int(conv2_weight.shape[-1]) if conv2_weight is not None else 5,
        "dropout_rate": float(model_result.get("dropout_rate", 0.0)),
        "weight_decay": float(model_result.get("weight_decay", 0.0)),
        "learning_rate": float(model_result.get("learning_rate", 0.0)),
        "epochs": int(model_result.get("epochs", 0)),
        "batch_size": int(model_result.get("batch_size", 0)),
        "scheduler_patience": int(model_result.get("scheduler_patience", 0)),
        "validation_split": float(model_result.get("validation_split", 0.0)),
        "early_stopping_patience": int(model_result.get("early_stopping_patience", 0)),
        "model_structure": str(model_result.get("model_structure", "")),
        "seed_count": int(model_result.get("seed_count", 0)),
        "seeds": [int(item.get("seed", 0)) for item in model_artifacts],
        "target_transform_mode": str(
            model_result.get("target_transform_mode", model_result.get("target_transform", "standard_scaler_only"))
        ),
        "target_scaler_mean": _first_float(model_result.get("target_scaler_mean", 0.0), 0.0),
        "target_scaler_scale": _first_float(model_result.get("target_scaler_scale", 1.0), 1.0),
        "target_scaler_var": _json_ready(np.asarray(model_result.get("target_scaler_var", []), dtype=float)),
        "target_scaler_min": _json_ready(np.asarray(model_result.get("target_scaler_min", []), dtype=float)),
        "target_scaler_data_min": _json_ready(np.asarray(model_result.get("target_scaler_data_min", []), dtype=float)),
        "target_scaler_data_max": _json_ready(np.asarray(model_result.get("target_scaler_data_max", []), dtype=float)),
        "input_extrapolation_margin": float(model_result.get("input_extrapolation_margin", 0.0)),
        "target_log_extrapolation_margin": float(model_result.get("target_log_extrapolation_margin", 0.0)),
        "x_train_min": _json_ready(np.asarray(model_result.get("x_train_min", []), dtype=float)),
        "x_train_max": _json_ready(np.asarray(model_result.get("x_train_max", []), dtype=float)),
        "target_transform_min": _first_float(model_result.get("target_transform_min", model_result.get("target_log_min", 0.0)), 0.0),
        "target_transform_max": _first_float(model_result.get("target_transform_max", model_result.get("target_log_max", 0.0)), 0.0),
        "scaler_mean": _json_ready(np.asarray(model_result.get("scaler_mean", []), dtype=float)),
        "scaler_scale": _json_ready(np.asarray(model_result.get("scaler_scale", []), dtype=float)),
        "scaler_var": _json_ready(np.asarray(model_result.get("scaler_var", []), dtype=float)),
        "scaler_n_features_in": int(model_result.get("scaler_n_features_in", 0)),
        "train_cells": [str(cell) for cell in model_result.get("train_cell_list", [])],
        "test_cells": [str(cell) for cell in model_result.get("test_cell_list", [])],
    }
    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    for artifact in model_artifacts:
        seed = int(artifact["seed"])
        state_dict = artifact["state_dict"]
        save_payload = {
            "seed": seed,
            "best_metric": float(artifact.get("best_metric", np.nan)),
            "state_dict": state_dict,
        }
        torch.save(save_payload, artifact_dir / f"seed_{seed:05d}.pt")

def save_throughput_threshold_model_artifacts(*, scenario_dir: Path, threshold: float, threshold_basis: str,
    capa_lim: float, raw_linear_models: list[dict[str, object]], feature_row_index: int | None = None,
    interpolated_tcnn: dict[str, object] | None, raw_tcnn: dict[str, object] | None) -> None:
    models_dir = scenario_dir / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    linear_dirs: dict[str, str] = {}
    for linear_model in raw_linear_models:
        model_variant = str(linear_model.get("model_variant", "") or "linear")
        model_dir_name = get_linear_artifact_dir_name(model_variant)
        save_linear_model_artifact(linear_model, models_dir / model_dir_name)
        linear_dirs[model_variant] = model_dir_name
    if interpolated_tcnn is not None:
        save_tcnn_ensemble_artifact(interpolated_tcnn, models_dir / TCNN_INTERPOLATED_ARTIFACT_DIR)
    if raw_tcnn is not None:
        save_tcnn_ensemble_artifact(raw_tcnn, models_dir / TCNN_RAW_ARTIFACT_DIR)
    basis = normalize_threshold_basis(threshold_basis)
    manifest = {"threshold_basis": basis, "threshold_value": float(threshold),
        "threshold_weeks": float(threshold) if basis == "week" else None,
        "threshold_throughput": float(threshold) if basis == "throughput" else None,
        "capa_lim": float(capa_lim), "interpolation_type": THROUGHPUT_INTERPOLATION_TYPE,
        **({"feature_row_index": int(feature_row_index)} if feature_row_index is not None else {}), "target_soh": float(TARGET_SOH),
        "raw_target_column": RAW_THROUGHPUT_TARGET_COLUMN,
        "interpolated_target_column": RAW_THROUGHPUT_TARGET_COLUMN,
        "linear_raw_dir": linear_dirs.get("best_r2", get_linear_artifact_dir_name("best_r2")),
        "linear_raw_dirs": linear_dirs,
        "tcnn_interpolated_dir": TCNN_INTERPOLATED_ARTIFACT_DIR if interpolated_tcnn is not None else None,
        "tcnn_raw_dir": TCNN_RAW_ARTIFACT_DIR if raw_tcnn is not None else None}
    with open(models_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

def plot_linear_training_input_week_vs_throughput(*, threshold: float,
    threshold_basis: str = DEFAULT_THRESHOLD_BASIS, capa_lim: float, output_dir: Path, raw_input_dir: Path,
    raw_linear_datasets: list[dict[str, object]], show: bool) -> None:
    if not raw_linear_datasets:
        return
    basis = normalize_threshold_basis(threshold_basis)
    if basis != "week" or not np.isclose(float(threshold), 13.0, rtol=0.0, atol=1e-9):
        return
    threshold_stem = build_threshold_scenario_stem(threshold, basis)
    target_colors = {"tp100": "#6FA2CC", "mintest": "#3ABABD", "halfmax": "#C19CC4"}
    panels: list[dict[str, object]] = []
    for raw_dataset in raw_linear_datasets:
        raw_data = raw_dataset.get("raw_data")
        linear_result = raw_dataset.get("linear_model", {})
        if not isinstance(raw_data, pd.DataFrame) or "linear_input_target_week" not in raw_data.columns:
            continue
        train_rows = raw_data[raw_data["is_outer_field_selected"]].copy()
        test_rows = raw_data[raw_data["is_test_cell"]].copy()
        if train_rows.empty and test_rows.empty:
            continue
        target_key = str(raw_dataset.get("target_key", "") or "").strip().lower()
        target_throughput = float(raw_dataset.get("target_throughput", np.nan))
        color = target_colors.get(target_key, "#7A7A7A")
        selected_feature = str(linear_result.get("selected_feature", "") or "")
        excluded = {"Cell", "is_outer_field_selected", "is_test_cell", RAW_THROUGHPUT_TARGET_COLUMN}
        plot_features = select_input_feature_trajectory_columns(
            [str(column) for column in raw_data.columns if str(column) not in excluded])

        def build_line_payloads(rows: pd.DataFrame) -> list[dict[str, object]]:
            payloads: list[dict[str, object]] = []
            for row in rows.itertuples(index=False):
                cell = str(getattr(row, "Cell"))
                target_week = pd.to_numeric(getattr(row, "linear_input_target_week", np.nan), errors="coerce")
                if not np.isfinite(target_week) or not np.isfinite(target_throughput):
                    continue
                csv_path = raw_input_dir / f"Capacity_{cell}.csv"
                if not csv_path.exists():
                    continue
                try:
                    raw_df = load_cleaned_input_dataframe(csv_path)
                except Exception:
                    continue
                if RAW_WEEK_COLUMN not in raw_df.columns or "current_throughput" not in raw_df.columns:
                    continue
                weeks = pd.to_numeric(raw_df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
                cumulative = build_raw_cumulative_throughput(raw_df)
                point_count = min(int(weeks.size), int(cumulative.size))
                weeks, cumulative = weeks[:point_count], cumulative[:point_count]
                finite_mask = np.isfinite(weeks) & np.isfinite(cumulative)
                if not np.any(finite_mask):
                    continue
                feature_payloads: dict[str, dict[str, np.ndarray]] = {}
                for feature_name in plot_features:
                    if feature_name not in raw_df:
                        continue
                    feature_values = pd.to_numeric(raw_df[feature_name], errors="coerce").to_numpy(dtype=float)
                    count = min(point_count, int(feature_values.size))
                    source_weeks, source_throughput = weeks[:count], cumulative[:count]
                    source_values = feature_values[:count]
                    mask = np.isfinite(source_weeks) & np.isfinite(source_throughput) & np.isfinite(source_values)
                    if np.count_nonzero(mask) < 2:
                        continue
                    feature_weeks = np.asarray(source_weeks[mask], dtype=float)
                    feature_throughput = np.asarray(source_throughput[mask], dtype=float)
                    feature_values = np.asarray(source_values[mask], dtype=float)
                    full_order = np.argsort(feature_throughput)
                    keep = feature_weeks <= float(target_week)
                    segment_order = np.argsort(feature_throughput[keep]) if np.any(keep) else np.asarray([], dtype=int)
                    feature_payloads[str(feature_name)] = {
                        "full_x": np.asarray(feature_throughput[full_order], dtype=float),
                        "full_y": np.asarray(feature_values[full_order], dtype=float),
                        "segment_x": (np.asarray(feature_throughput[keep][segment_order], dtype=float)
                            if segment_order.size else np.asarray([], dtype=float)),
                        "segment_y": (np.asarray(feature_values[keep][segment_order], dtype=float)
                            if segment_order.size else np.asarray([], dtype=float))}
                weeks, cumulative = weeks[finite_mask], cumulative[finite_mask]
                keep = weeks <= float(target_week)
                if not np.any(keep):
                    continue
                full_order = np.argsort(weeks)
                segment_weeks = np.append(weeks[keep], float(target_week))
                segment_throughput = np.append(cumulative[keep], float(target_throughput))
                segment_order = np.argsort(segment_weeks)
                payloads.append({"cell": cell, "full_weeks": np.asarray(weeks[full_order], dtype=float),
                    "full_throughput": np.asarray(cumulative[full_order], dtype=float),
                    "weeks": np.asarray(segment_weeks[segment_order], dtype=float),
                    "throughput": np.asarray(segment_throughput[segment_order], dtype=float),
                    "target_week": float(target_week), "features": feature_payloads})
            return payloads

        train_payloads, test_payloads = build_line_payloads(train_rows), build_line_payloads(test_rows)
        if train_payloads or test_payloads:
            panels.append({"target_key": target_key, "target_throughput": target_throughput,
                "display_name": str(linear_result.get("display_name", "") or target_key),
                "selected_feature": selected_feature, "plot_features": list(plot_features), "color": color,
                "train_line_payloads": train_payloads, "test_line_payloads": test_payloads})
    if not panels:
        return

    n_panels = len(panels)
    n_cols = min(3, max(1, n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))
    max_feature_count = max(len(panel.get("plot_features", [])) for panel in panels)
    feature_rows = max(1, int(np.ceil(max(1, max_feature_count) / 2.0)))
    rows_per_panel = 1 + feature_rows
    fig = plt.figure(figsize=(float(plot_cfg.AUGMENTATION_OVERVIEW_WIDTH_IN) * n_cols,
        (3.1 + 2.55 * feature_rows) * n_rows))
    gridspec = fig.add_gridspec(rows_per_panel * n_rows, 2 * n_cols,
        height_ratios=[1.08, *([1.0] * feature_rows)] * n_rows, hspace=0.52, wspace=0.28)
    panel_axes = []
    for index in range(n_rows * n_cols):
        panel_row, panel_col = divmod(index, n_cols)
        top_row, left_col = panel_row * rows_per_panel, 2 * panel_col
        panel_axes.append((fig.add_subplot(gridspec[top_row, left_col:left_col + 2]),
            [fig.add_subplot(gridspec[top_row + 1 + feature_index // 2,
                left_col + feature_index % 2]) for feature_index in range(max(1, max_feature_count))]))

    for (axis, feature_axes), panel in zip(panel_axes, panels):
        color = str(panel["color"])
        target_throughput = float(panel["target_throughput"])
        target_key = str(panel["target_key"])
        selected_feature = str(panel["selected_feature"])
        plot_features = [str(name) for name in panel.get("plot_features", [])]
        train_payloads = list(panel["train_line_payloads"])
        test_payloads = list(panel["test_line_payloads"])
        all_payloads = train_payloads + test_payloads
        max_target_week = max(float(item["target_week"]) for item in all_payloads)
        for idx, payload in enumerate(train_payloads):
            axis.plot(payload["full_weeks"], payload["full_throughput"], color="#B0B0B0", alpha=0.28,
                linewidth=1.0, zorder=1, label="Train full trajectory" if idx == 0 else None)
            axis.plot(payload["weeks"], payload["throughput"], color=color, alpha=0.75, linewidth=1.4,
                zorder=2, label="Train input segment" if idx == 0 else None)
            axis.scatter([float(payload["target_week"])], [target_throughput], s=24, color=color,
                edgecolors="black", linewidths=0.25, zorder=3)
            for feature_axis, feature_name in zip(feature_axes, plot_features):
                values = dict(dict(payload.get("features", {}) or {}).get(feature_name, {}) or {})
                full_x, full_y = np.asarray(values.get("full_x", [])), np.asarray(values.get("full_y", []))
                segment_x = np.asarray(values.get("segment_x", []))
                segment_y = np.asarray(values.get("segment_y", []))
                if full_x.size and full_y.size:
                    feature_axis.plot(full_x, full_y, color="#B0B0B0", alpha=0.22, linewidth=0.72, zorder=1)
                if segment_x.size and segment_y.size:
                    feature_axis.plot(segment_x, segment_y, color=color, alpha=0.64, linewidth=0.9, zorder=2)
        for idx, payload in enumerate(test_payloads):
            axis.plot(payload["full_weeks"], payload["full_throughput"], color="#EF8881", alpha=0.20,
                linewidth=1.0, zorder=1, label="Test full trajectory" if idx == 0 else None)
            axis.plot(payload["weeks"], payload["throughput"], color="#EF8881", alpha=0.62, linewidth=1.4,
                zorder=2, label="Test input segment" if idx == 0 else None)
            axis.scatter([float(payload["target_week"])], [target_throughput], s=30, color="#EF8881",
                edgecolors="black", linewidths=0.25, zorder=4)
            for feature_axis, feature_name in zip(feature_axes, plot_features):
                values = dict(dict(payload.get("features", {}) or {}).get(feature_name, {}) or {})
                full_x, full_y = np.asarray(values.get("full_x", [])), np.asarray(values.get("full_y", []))
                segment_x = np.asarray(values.get("segment_x", []))
                segment_y = np.asarray(values.get("segment_y", []))
                if full_x.size and full_y.size:
                    feature_axis.plot(full_x, full_y, color="#EF8881", alpha=0.18, linewidth=0.72, zorder=1)
                if segment_x.size and segment_y.size:
                    feature_axis.plot(segment_x, segment_y, color="#EF8881", alpha=0.46, linewidth=0.9, zorder=2)
        axis.axvline(float(threshold), color="black", linestyle="--", linewidth=1.1, label="Week threshold")
        axis.axhline(target_throughput, color=color, linestyle=":", linewidth=1.2,
            label="Linear input throughput")
        for feature_axis, feature_name in zip(feature_axes, plot_features):
            feature_axis.axvspan(0.0, target_throughput, color="#F2C14E", alpha=0.06, zorder=0)
            feature_axis.axvline(target_throughput, color=color, linestyle=":", linewidth=1.0, alpha=0.78,
                zorder=3)
            feature_axis.set_title(format_feature_math_label(feature_name), fontsize=8.5, fontweight="bold")
            feature_axis.set_xlabel("Cumulative throughput [Ah]")
            feature_axis.set_ylabel(format_feature_math_label(feature_name), fontsize=8)
            feature_axis.set_axisbelow(True)
            feature_axis.tick_params(axis="both", labelsize=7)
        for feature_axis in feature_axes[len(plot_features):]:
            feature_axis.set_axis_off()
        title_prefix = ("100 input" if target_key == "tp100" else
            (f"Min-test input {format_threshold_value(target_throughput)}" if target_key == "mintest"
             else f"Half-max input {format_threshold_value(target_throughput)}"))
        axis.set_title(f"{title_prefix} | {selected_feature}\n"
            f"n={len(train_payloads)}/{len(test_payloads)}, max week={max_target_week:.1f}", fontsize=10)
        axis.set_xlabel("Week")
        axis.set_ylabel("Cumulative throughput")
        axis.legend(frameon=False, fontsize=8)

    for top_axis, feature_axes in panel_axes[n_panels:]:
        top_axis.set_axis_off()
        for axis in feature_axes:
            axis.set_axis_off()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    output_path = output_dir / (f"{threshold_stem}_linear_training_week_vs_throughput_soh"
        f"{int(round(float(capa_lim) * 100.0))}_throughput_IOWA.png")
    if not show:
        plot_cfg.set_figure_exact_canvas(fig, True)
        save_figure_multiformat(fig, output_path, dpi=int(plot_cfg.SUMMARY_PAPER_EXPORT_DPI),
            bbox_inches=None)
        print(f"Saved linear training week-vs-throughput plot to {output_path}")
    finalize_figure(fig, show)

def save_threshold_model_artifacts(
    *,
    scenario_dir: Path,
    threshold: float,
    capa_lim: float,
    interpolation_type: str,
    raw_linear_models: list[dict[str, object]],
    interpolated_tcnn: dict[str, object] | None,
    raw_tcnn: dict[str, object],
    feature_row_index: int | None = None,
) -> None:
    models_dir = scenario_dir / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    linear_dirs: dict[str, str] = {}
    for linear_model in raw_linear_models:
        model_variant = str(linear_model.get("model_variant", "") or "linear")
        model_dir_name = get_linear_artifact_dir_name(model_variant)
        save_linear_model_artifact(linear_model, models_dir / model_dir_name)
        linear_dirs[model_variant] = model_dir_name
    if interpolated_tcnn is not None:
        save_tcnn_ensemble_artifact(interpolated_tcnn, models_dir / TCNN_INTERPOLATED_ARTIFACT_DIR)
    save_tcnn_ensemble_artifact(raw_tcnn, models_dir / TCNN_RAW_ARTIFACT_DIR)
    manifest = {
        "threshold_weeks": float(threshold),
        "capa_lim": float(capa_lim),
        "interpolation_type": str(interpolation_type),
        **({"feature_row_index": int(feature_row_index)} if feature_row_index is not None else {}),
        "target_soh": float(TARGET_SOH),
        "raw_target_column": RAW_TARGET_COLUMN,
        "interpolated_target_column": INTERPOLATED_TARGET_COLUMN,
        "linear_raw_dir": linear_dirs.get("best_r2", get_linear_artifact_dir_name("best_r2")),
        "linear_raw_dirs": linear_dirs,
        "tcnn_interpolated_dir": TCNN_INTERPOLATED_ARTIFACT_DIR if interpolated_tcnn is not None else None,
        "tcnn_raw_dir": TCNN_RAW_ARTIFACT_DIR,
    }
    with open(models_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

def run_threshold_scenario(
    threshold: float,
    raw_input_dir: Path,
    feature_row_index: int,
    output_dir: Path,
    raw_week_by_cell: dict[str, float],
    raw_throughput_by_cell: dict[str, float],
    capa_lim: float,
    interpolation_type: str,
    min_abs_pearson: float,
    show: bool,
) -> pd.DataFrame:
    interpolated_input_steps = core_autoanchor.resolve_interpolated_tcnn_input_steps(threshold)
    raw_input_steps = core_autoanchor.resolve_raw_tcnn_input_steps(threshold)
    interpolated_tcnn_hyperparameters = dict(BEST_INTERPOLATED_TCNN_HYPERPARAMETERS)
    interpolated_tcnn_hyperparameters["input_steps"] = interpolated_input_steps
    raw_tcnn_hyperparameters = dict(BEST_RAW_TCNN_HYPERPARAMETERS)
    raw_tcnn_hyperparameters["input_steps"] = raw_input_steps

    label = f"eligible{int(threshold)}_features_{FEATURE_SELECTION_LABEL}"
    scenario_dir = output_dir / label
    scenario_dir.mkdir(parents=True, exist_ok=True)

    train_cells, test_cells = build_threshold_split(raw_week_by_cell, threshold)
    train_cells_set = set(train_cells)
    test_cells_set = set(test_cells)

    interpolated_data, interpolated_features, interpolated_skipped = load_interpolated_dataframe(
        raw_input_dir=raw_input_dir,
        feature_row_index=feature_row_index,
        train_cells=train_cells_set,
        test_cells=test_cells_set,
        capa_lim=capa_lim,
        interpolation_type=interpolation_type,
    )
    raw_data, raw_features, raw_skipped = load_raw_dataframe(
        raw_input_dir=raw_input_dir,
        train_cells=train_cells_set,
        test_cells=test_cells_set,
    )
    raw_data_early, raw_features_early, raw_skipped_early = load_raw_dataframe(
        raw_input_dir=raw_input_dir,
        train_cells=train_cells_set,
        test_cells=test_cells_set,
        feature_week_resolver=get_linear_raw_feature_week_early,
    )

    interpolated_summary = build_fit_summary(interpolated_data, interpolated_features, target_column=INTERPOLATED_TARGET_COLUMN)
    raw_summary = build_fit_summary(raw_data, raw_features, target_column=RAW_TARGET_COLUMN)
    raw_summary_early = build_fit_summary(raw_data_early, raw_features_early, target_column=RAW_TARGET_COLUMN)

    raw_linear = train_linear_feature_regressor(
        data=raw_data,
        fit_summary=raw_summary,
        target_column=RAW_TARGET_COLUMN,
        dataset_label=f"{label}_raw_week70",
    )
    raw_linear_early = train_linear_feature_regressor(
        data=raw_data_early,
        fit_summary=raw_summary_early,
        target_column=RAW_TARGET_COLUMN,
        dataset_label=f"{label}_raw_week70_wk4",
        input_week_resolver=get_linear_raw_feature_week_early,
        model_variant_override=LINEAR_EARLY_MODEL_VARIANT,
        display_name_override="Linear raw 4w (best R^2)",
        raw_feature_week_default=LINEAR_RAW_FEATURE_WEEK_EARLY,
        raw_feature_week_post_g20=LINEAR_RAW_FEATURE_WEEK_POST_G20_EARLY,
        linear_input_week_variant=LINEAR_INPUT_WEEK_VARIANT_EARLY,
    )
    raw_linear_fixed_feature_names = get_raw_linear_fixed_feature_names(raw_features)
    raw_linear_feature_models: list[dict[str, object]] = []
    for feature_name in raw_linear_fixed_feature_names:
        raw_linear_feature_models.append(
            train_linear_feature_regressor(
                data=raw_data,
                fit_summary=raw_summary,
                target_column=RAW_TARGET_COLUMN,
                dataset_label=f"{label}_raw_week70_{feature_name}",
                feature_name=feature_name,
            )
        )
    raw_linear_best_train_rmse = choose_best_linear_model_by_rmse(
        raw_linear_feature_models,
        model_variant=LINEAR_BEST_TRAIN_RMSE_MODEL_VARIANT,
        display_name="Linear raw (best train RMSE)",
        dataset_label=f"{label}_raw_week70_best_train_rmse",
        split="train",
    )
    raw_linear_avg = train_linear_feature_average_regressor(
        data=raw_data,
        fit_summary=raw_summary,
        target_column=RAW_TARGET_COLUMN,
        dataset_label=f"{label}_raw_week70_all_feature_avg",
        feature_names=get_mean_var_feature_names(raw_features),
    )
    raw_linear_models = [raw_linear, raw_linear_best_train_rmse, raw_linear_early, *raw_linear_feature_models, raw_linear_avg]
    interpolated_mlp_features = get_required_features(
        interpolated_data.columns.tolist(),
        get_sequence_and_mlp_feature_names(interpolated_data.columns.tolist()),
        label=f"{label} interpolated MLP",
    )
    interpolated_mlp_selected_summary = build_fit_summary(
        interpolated_data,
        interpolated_mlp_features,
        target_column=INTERPOLATED_TARGET_COLUMN,
    )
    raw_mlp_features = get_required_features(
        raw_data.columns.tolist(),
        get_sequence_and_mlp_feature_names(raw_data.columns.tolist()),
        label=f"{label} raw MLP",
    )
    raw_mlp_selected_summary = build_fit_summary(
        raw_data,
        raw_mlp_features,
        target_column=RAW_TARGET_COLUMN,
    )
    interpolated_tcnn_features = select_soh_meanvar_tcnn_features(
        ["SOH", *interpolated_data.columns.tolist()],
        label=f"{label} interpolated",
    )
    raw_tcnn_features = select_soh_meanvar_tcnn_features(
        raw_data.columns.tolist(),
        label=f"{label} raw",
    )

    interpolated_sequence_data, interpolated_sequence_skipped = load_interpolated_sequence_dataset(
        raw_input_dir=raw_input_dir,
        input_steps=interpolated_input_steps,
        sequence_features=interpolated_tcnn_features,
        train_cells=train_cells_set,
        test_cells=test_cells_set,
        capa_lim=capa_lim,
        interpolation_type=interpolation_type,
    )
    raw_sequence_data, raw_sequence_skipped = load_raw_sequence_dataset(
        raw_input_dir=raw_input_dir,
        input_steps=raw_input_steps,
        sequence_features=raw_tcnn_features,
        train_cells=train_cells_set,
        test_cells=test_cells_set,
    )
    interpolated_time_curve_df = build_interpolated_time_curve_dataframe(interpolated_sequence_data)

    interpolated_tcnn = train_temporal_cnn_regressor(
        sequence_dataset=interpolated_sequence_data,
        dataset_label=f"{label}_interpolated_step70",
        hyperparameters=interpolated_tcnn_hyperparameters,
    )
    interpolated_tcnn = attach_input_last_step_week_metrics(interpolated_tcnn, interpolated_sequence_data)
    interpolated_tcnn = attach_interpolated_time_metrics(interpolated_tcnn, interpolated_sequence_data)
    interpolated_tcnn = attach_interpolated_throughput_metrics(interpolated_tcnn, interpolated_sequence_data)
    interpolated_tcnn = attach_real_target_scale_metrics(interpolated_tcnn)
    raw_tcnn = train_temporal_cnn_regressor(
        sequence_dataset=raw_sequence_data,
        dataset_label=f"{label}_raw_week70",
        hyperparameters=raw_tcnn_hyperparameters,
    )
    raw_tcnn = attach_input_last_step_week_metrics(raw_tcnn, raw_sequence_data)
    raw_tcnn = attach_real_target_scale_metrics(raw_tcnn)
    save_threshold_model_artifacts(
        scenario_dir=scenario_dir,
        threshold=float(threshold),
        capa_lim=float(capa_lim),
        interpolation_type=interpolation_type,
        feature_row_index=int(feature_row_index),
        raw_linear_models=[raw_linear, raw_linear_best_train_rmse, raw_linear_early, *raw_linear_feature_models, raw_linear_avg],
        interpolated_tcnn=interpolated_tcnn,
        raw_tcnn=raw_tcnn,
    )
    selection_df = pd.DataFrame(
        {
            "Cell": sorted(raw_week_by_cell),
            RAW_TARGET_COLUMN: [raw_week_by_cell[cell] for cell in sorted(raw_week_by_cell)],
            RAW_THROUGHPUT_TARGET_COLUMN: [raw_throughput_by_cell.get(cell, np.nan) for cell in sorted(raw_week_by_cell)],
            f"train_le_{int(threshold)}": [raw_week_by_cell[cell] <= threshold for cell in sorted(raw_week_by_cell)],
            f"test_gt_{int(threshold)}": [raw_week_by_cell[cell] > threshold for cell in sorted(raw_week_by_cell)],
            "selected_for_train": [cell in train_cells_set for cell in sorted(raw_week_by_cell)],
            "used_for_test": [cell in test_cells_set for cell in sorted(raw_week_by_cell)],
        }
    )
    selection_df.to_csv(scenario_dir / f"{label}_train_selection_IOWA.csv", index=False)
    interpolated_summary.to_csv(scenario_dir / f"{label}_interpolated_fit_summary_IOWA.csv", index=False)
    interpolated_data.to_csv(scenario_dir / f"{label}_interpolated_cell_data_IOWA.csv", index=False)
    interpolated_time_curve_df.to_csv(scenario_dir / f"{label}_interpolated_step_time_curve_IOWA.csv", index=False)
    raw_summary.to_csv(scenario_dir / f"{label}_raw_fit_summary_IOWA.csv", index=False)
    raw_data.to_csv(scenario_dir / f"{label}_raw_cell_data_IOWA.csv", index=False)
    interpolated_mlp_selected_summary.to_csv(
        scenario_dir / f"{label}_interpolated_fixed_mlp_tcnn_features_IOWA.csv",
        index=False,
    )
    raw_mlp_selected_summary.to_csv(
        scenario_dir / f"{label}_raw_fixed_mlp_tcnn_features_IOWA.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"dataset": "interpolated_tabular", **interpolated_skipped},
            {"dataset": "raw_tabular", **raw_skipped},
            {"dataset": "raw_tabular_wk4", **raw_skipped_early},
            {"dataset": "interpolated_tcnn", **interpolated_sequence_skipped},
            {"dataset": "raw_tcnn", **raw_sequence_skipped},
        ]
    ).to_csv(scenario_dir / f"{label}_skipped_summary_IOWA.csv", index=False)

    plot_random_relationship_subplots(data=raw_data, feature_columns=raw_features, fit_summary=raw_summary, output_dir=scenario_dir, target_column=RAW_TARGET_COLUMN,
        target_label="week at 70% SOH", output_stem=f"{label}_raw_log_feature_vs_log_week70", title_prefix=f"IOWA raw features, train <= {threshold:.0f} weeks, test > {threshold:.0f} weeks",
        show=show)
    week_prediction_df = pd.DataFrame(
        {
            "Cell": list(interpolated_tcnn.get("test_cell_list", [])),
            "actual_interpolated_step70": np.asarray(interpolated_tcnn.get("test_actual_step", interpolated_tcnn.get("test_actual", [])), dtype=float),
            "predicted_interpolated_step70": np.asarray(interpolated_tcnn.get("test_pred_step", interpolated_tcnn.get("test_pred", [])), dtype=float),
            "weeks_per_interpolated_step": np.asarray(interpolated_tcnn.get("test_weeks_per_step", []), dtype=float),
            "actual_week_at_eol70": np.asarray(interpolated_tcnn.get("test_actual_time", []), dtype=float),
            "predicted_week_at_eol70": np.asarray(interpolated_tcnn.get("test_pred_time", []), dtype=float),
        }
    )
    week_prediction_df.to_csv(
        scenario_dir / f"{label}_interpolated_test_week_predictions_IOWA.csv",
        index=False,
    )
    if not week_prediction_df.empty:
        print("Interpolated TCNN test week predictions")
        print(week_prediction_df.to_string(index=False))
    if str(interpolation_type).strip().lower() == "throughput":
        throughput_prediction_df = pd.DataFrame(
            {
                "Cell": list(interpolated_tcnn.get("test_cell_list", [])),
                "actual_interpolated_step70": np.asarray(interpolated_tcnn.get("test_actual_step", interpolated_tcnn.get("test_actual", [])), dtype=float),
                "predicted_interpolated_step70": np.asarray(interpolated_tcnn.get("test_pred_step", interpolated_tcnn.get("test_pred", [])), dtype=float),
                "throughput_per_interpolated_step": np.asarray(interpolated_tcnn.get("test_throughput_per_step", []), dtype=float),
                "actual_throughput_at_eol70": np.asarray(interpolated_tcnn.get("test_actual_throughput", []), dtype=float),
                "predicted_throughput_at_eol70": np.asarray(interpolated_tcnn.get("test_pred_throughput", []), dtype=float),
            }
        )
        throughput_prediction_df.to_csv(
            scenario_dir / f"{label}_interpolated_test_throughput_predictions_IOWA.csv",
            index=False,
        )
        if not throughput_prediction_df.empty:
            print("Interpolated TCNN test throughput predictions")
            print(throughput_prediction_df.to_string(index=False))

    long_eol_plot_results: dict[str, dict[str, object] | None] = {}
    for model_result, model_label in [
        *[(linear_model, "Linear") for linear_model in raw_linear_models],
        (interpolated_tcnn, "TCNN"),
        (raw_tcnn, "TCNN"),
    ]:
        model_result.update(
            calculate_long_eol_validation_metrics(
                model_result=model_result,
                raw_week_by_cell=raw_week_by_cell,
            )
        )
        attach_named_long_eol_metrics(
            model_result,
            raw_week_by_cell=raw_week_by_cell,
            threshold_weeks=EXTRA_LONG_EOL_VALIDATION_THRESHOLD_WEEKS,
            prefix="extra_long_eol",
        )
        long_eol_plot_result = build_long_eol_plot_result(
            model_result=model_result,
            raw_week_by_cell=raw_week_by_cell,
        )
        result_key = f"{model_result['dataset_label']}::{model_result['model_name']}"
        long_eol_plot_results[result_key] = long_eol_plot_result


    model_summary = summarize_model_rows(
        threshold=threshold,
        model_results=[
            *raw_linear_models,
            interpolated_tcnn,
            raw_tcnn,
        ],
    )
    model_summary.to_csv(scenario_dir / f"{label}_model_comparison_summary_IOWA.csv", index=False)

    print(f"\nThreshold <= {threshold:.0f} weeks")
    print(f"Train cells: {len(train_cells)}")
    print(f"Test cells: {len(test_cells)}")
    print(f"Interpolated skipped: {interpolated_skipped}")
    print(f"Raw skipped: {raw_skipped}")
    print(f"Interpolated sequence skipped: {interpolated_sequence_skipped}")
    print(f"Raw sequence skipped: {raw_sequence_skipped}")
    print(f"Raw linear best-R^2 feature: {raw_linear.get('selected_feature')}")
    print(f"Raw linear best-train-RMSE feature: {raw_linear_best_train_rmse.get('selected_feature')}")
    print(f"Raw linear 4w best-R^2 feature: {raw_linear_early.get('selected_feature')}")
    print(f"Raw linear single-feature models: {raw_linear_fixed_feature_names}")
    print(f"Raw linear mean/var avg model count: {raw_linear_avg.get('component_model_count')}")
    print(f"Interpolated TCNN fixed features: {interpolated_tcnn_features}")
    print(f"Raw TCNN fixed features: {raw_tcnn_features}")
    print(model_summary.to_string(index=False))

    return model_summary

def main() -> None:
    args = parse_args()
    raw_input_dir = args.raw_input_dir.resolve()
    output_dir = args.output_dir.resolve()
    capa_lims = [float(capa_lim) for capa_lim in args.capa_lims]

    if not raw_input_dir.exists():
        raise FileNotFoundError(f"Raw input directory does not exist: {raw_input_dir}")

    raw_week_by_cell = get_raw_week_at_target_map(raw_input_dir, TARGET_SOH)
    raw_throughput_by_cell = get_raw_throughput_at_target_map(raw_input_dir, TARGET_SOH)
    if not raw_week_by_cell:
        raise RuntimeError(f"No IOWA cells with usable week-at-70%%SOH targets were found in {raw_input_dir}")
    if not raw_throughput_by_cell:
        raise RuntimeError(f"No IOWA cells with usable throughput-at-70%%SOH targets were found in {raw_input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_early_cells: list[dict[str, object]] = []

    overall_failures: list[dict[str, object]] = []
    for capa_lim in capa_lims:
        capa_output_dir = output_dir / format_soh_folder_name(capa_lim)
        capa_output_dir.mkdir(parents=True, exist_ok=True)

        all_summaries: list[pd.DataFrame] = []
        failures: list[dict[str, object]] = []
        for threshold in args.thresholds:
            try:
                summary = run_threshold_scenario(
                    threshold=float(threshold),
                    raw_input_dir=raw_input_dir,
                    feature_row_index=args.feature_row_index,
                    output_dir=capa_output_dir,
                    raw_week_by_cell=raw_week_by_cell,
                    raw_throughput_by_cell=raw_throughput_by_cell,
                    capa_lim=float(capa_lim),
                    interpolation_type=args.interpolation_type,
                    min_abs_pearson=float(args.min_abs_pearson),
                    show=args.show,
                )
            except Exception as exc:
                failures.append(
                    {
                        "capa_lim": float(capa_lim),
                        "threshold_weeks": float(threshold),
                        "error": str(exc),
                    }
                )
                print(f"Skipping threshold {threshold:.0f} for capa_lim {capa_lim:.2f}: {exc}")
                continue
            all_summaries.append(summary)

        if not all_summaries:
            overall_failures.append(
                {
                    "capa_lim": float(capa_lim),
                    "error": "All requested thresholds failed for this SOH limit.",
                }
            )
            continue

        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary.to_csv(capa_output_dir / "threshold_model_comparison_summary_IOWA.csv", index=False)
        plot_threshold_metric_trend(combined_summary, metric_column="test_mape", ylabel="MAPE (%)", title=f"SOH {int(round(float(capa_lim) * 100))}: model MAPE vs week threshold",
            output_path=capa_output_dir / "all_threshold_method_mape_trend_IOWA.png", show=args.show)
        plot_threshold_metric_trend(combined_summary, metric_column="test_mape", ylabel="MAPE (%)", title=f"SOH {int(round(float(capa_lim) * 100))}: core model MAPE vs week threshold",
            output_path=capa_output_dir / "all_threshold_method_mape_trend_core_IOWA.png", show=args.show, linear_variants=["best_r2"])
        plot_threshold_metric_trend(combined_summary, metric_column="test_plot_rmse", ylabel="RMSE", title=f"SOH {int(round(float(capa_lim) * 100))}: model RMSE vs week threshold",
            output_path=capa_output_dir / "all_threshold_method_rmse_trend_IOWA.png", show=args.show)
        plot_threshold_metric_trend(combined_summary, metric_column="test_plot_rmse", ylabel="RMSE", title=f"SOH {int(round(float(capa_lim) * 100))}: core model RMSE vs week threshold",
            output_path=capa_output_dir / "all_threshold_method_rmse_trend_core_IOWA.png", show=args.show, linear_variants=["best_r2"])
        plot_linear_threshold_rmse_comparison(combined_summary=combined_summary, output_dir=capa_output_dir, show=args.show, long_eol_threshold_weeks=LONG_EOL_VALIDATION_THRESHOLD_WEEKS)
        pd.DataFrame(
            [
                {
                    "raw_input_dir": str(raw_input_dir),
                    "target_soh": TARGET_SOH,
                    "interpolation_type": args.interpolation_type,
                    "capa_lim": float(capa_lim),
                    "soh_folder": format_soh_folder_name(capa_lim),
                    "feature_selection": FEATURE_SELECTION_LABEL,
                    "linear_feature_selection_method": LINEAR_FEATURE_SELECTION_METHOD,
                    "min_abs_pearson": float(args.min_abs_pearson),
                    "feature_row_index": args.feature_row_index,
                    "raw_feature_week_default": float(RAW_FEATURE_WEEK),
                    "raw_feature_week_post_g20": float(RAW_FEATURE_WEEK_POST_G20),
                    "excluded_early_eol70_cell_count": int(len(excluded_early_cells)),
                }
            ]
        ).to_csv(capa_output_dir / "threshold_run_config_IOWA.csv", index=False)
        if failures:
            pd.DataFrame(failures).to_csv(capa_output_dir / "threshold_failures_IOWA.csv", index=False)

    if overall_failures:
        pd.DataFrame(overall_failures).to_csv(output_dir / "threshold_overall_failures_IOWA.csv", index=False)

if __name__ == "__main__":
    main()
