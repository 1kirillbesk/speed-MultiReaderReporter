from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from core_functions import TinyTemporalCNN, augmentation_methods as core_aug
from core_functions import augmentation_run_config as exported_aug_cfg
from core_functions.autoanchor_policy import build_anchor_candidates, select_lowest_valid_anchor
from core_functions.temporal_models import build_regression_loss
from core_functions.training_seeds import build_model_training_seed_list
from exported_core_functions import (
    get_exported_augmentation_methods_module,
    get_exported_temporal_models_module,
)


HERE = Path(__file__).resolve().parent
INTERP_DIR = HERE / "interp_feature"
RAW_FEATURE_DIR = Path(r"C:\Users\Public\Documents\takedata\Speed\out_lw\cell_feature")
DEFAULT_FEATURE_COLS = [
    "mean_d_dqdv_m_c",
    "mean_d_dqdv_h_c",
    "mean_d_dqdv_l_c",
    "var_d_dqdv_m_c",
    "var_d_dqdv_h_c",
    "var_d_dqdv_l_c",
    "SOH",
]
DEFAULT_TARGET_COL = "step_to_soh_target"
DEFAULT_TARGET_SOH = 0.95
DEFAULT_ANCHOR_SOH = 0.975
DEFAULT_INPUT_END_WEEK = 10.0
DEFAULT_INPUT_END_THROUGHPUT = 0.4e7
DEFAULT_INPUT_STEPS = 6
DEFAULT_TEST_NAMES = (
    "SPEED_LW_reference_1",
    "SPEED_LW_reference_2",
    "SPEED_LW_reference_3",
)
DEFAULT_TEST_SUBSTRING = "reference"
DEFAULT_TRAIN_SUBSTRING = "cycle"
DEFAULT_SEED_COUNT = 10
DEFAULT_SEED_SOURCE = 42
DEFAULT_AUTOANCHOR_MIN_SOH = 0.80
DEFAULT_AUTOANCHOR_MAX_SOH = 0.99
DEFAULT_AUTOANCHOR_STEP_SOH = 0.01
DEFAULT_EXEMPT_FROM_SCALING = ("SOH",)
DEFAULT_AUGMENTATION_METHOD = str(core_aug.DEFAULT_AUTOANCHOR_AUGMENTATION_METHOD).strip().lower()
DEFAULT_CAL_AUGMENTATION_SAMPLE_COUNT = int(core_aug.DEFAULT_CAL_AUGMENTATION_SAMPLE_COUNT)
DEFAULT_HYBRID_AUGMENTATION_SAMPLE_COUNT = int(core_aug.DEFAULT_HYBRID_AUGMENTATION_SAMPLE_COUNT)
DEFAULT_CODEX_AUGMENTATION_SAMPLE_COUNT = int(core_aug.DEFAULT_CODEX_AUGMENTATION_SAMPLE_COUNT)
LOCAL_AUGMENTATION_METHODS = tuple(str(name).strip().lower() for name in core_aug.VALID_AUGMENTATION_METHODS)
PREDICTION_MODEL_LEGACY_TCNN = "legacy_tcnn"
PREDICTION_MODEL_EXPORTED_EOL_TCNN = "exported_eol_tcnn"
PREDICTION_MODEL_EXPORTED_EOL_MLP = "exported_eol_mlp"
SUPPORTED_PREDICTION_MODELS = (
    PREDICTION_MODEL_LEGACY_TCNN,
    PREDICTION_MODEL_EXPORTED_EOL_TCNN,
    PREDICTION_MODEL_EXPORTED_EOL_MLP,
)
LFP_AUGMENTATION_HELPER_RAW_COLUMN_ALIASES = {
    "mean_delta_Q": "mean_dQ_c",
    "var_delta_Q": "var_dQ_c",
    "mean_delta_dqdv": "mean_d_dqdv_m_c",
    "var_delta_dqdv": "var_d_dqdv_m_c",
}
LFP_AUGMENTATION_HELPER_PROCESS_COLUMNS = tuple(LFP_AUGMENTATION_HELPER_RAW_COLUMN_ALIASES.keys())
LFP_AUGMENTATION_HELPER_PREFERRED_FEATURE = "var_delta_Q"
RAW_TIME_SOURCE_COLUMN = "CU_time"
TIME_COLUMN = "weeks"
REFERENCE_COLUMN = "reference_step"
CAPACITY_COLUMN = "cap_ocv_dis"
CAPACITY_FALLBACK_COLUMNS = ("cap_ocv_dis", "cap_ocv_cha", "cap_dis", "cap_cha")
THROUGHPUT_COLUMN = "throughput_cum"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class TrainingConfig:
    input_steps: int = DEFAULT_INPUT_STEPS
    epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    validation_split: float = 0.2
    seed_count: int = DEFAULT_SEED_COUNT
    seed_source: int = DEFAULT_SEED_SOURCE
    early_stopping_patience: int = 25
    scheduler_patience: int = 10
    conv1_channels: int = 16
    conv_channels: int = 16
    fc_hidden_dim: int = 16
    fc_hidden_dim2: int | None = 8
    conv1_kernel_size: int = 3
    conv2_kernel_size: int = 3
    loss_function: str = "mse"
    prediction_model: str = PREDICTION_MODEL_LEGACY_TCNN


def set_seed(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_balanced_val_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_fraction: float = 0.2,
    bins: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_array = np.asarray(y, dtype=float).reshape(-1)
    total = len(y_array)
    desired_val = max(1, int(round(float(val_fraction) * float(total))))
    effective_bins = max(2, min(int(bins), max(2, total // 3)))
    y_bins = pd.qcut(y_array, q=effective_bins, labels=False, duplicates="drop")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=max(1, min(total - 1, desired_val)),
        random_state=42,
    )
    train_idx, val_idx = next(splitter.split(X, y_bins))
    return X[train_idx], y_array[train_idx], X[val_idx], y_array[val_idx]


def supported_augmentation_methods() -> tuple[str, ...]:
    methods = list(LOCAL_AUGMENTATION_METHODS)
    try:
        exported_aug = get_exported_augmentation_methods_module()
    except Exception:
        return tuple(methods)
    for name in getattr(exported_aug, "VALID_AUGMENTATION_METHODS", ()):
        normalized = str(name).strip().lower()
        if normalized and normalized not in methods:
            methods.append(normalized)
    return tuple(methods)


def normalize_prediction_model(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_PREDICTION_MODELS:
        raise ValueError(
            f"Unsupported prediction model {value!r}. "
            f"Expected one of {list(SUPPORTED_PREDICTION_MODELS)!r}."
        )
    return normalized


def compute_symmetric_log_target_divisor(target_values: np.ndarray) -> float:
    target_values = np.asarray(target_values, dtype=float)
    valid = np.isfinite(target_values) & (target_values > 0.0)
    if np.count_nonzero(valid) == 0:
        raise RuntimeError("Cannot compute target divisor without positive target values.")
    positive_targets = target_values[valid]
    return float(np.sqrt(np.min(positive_targets) * np.max(positive_targets)))


def prepare_lfp_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if TIME_COLUMN not in result.columns:
        if RAW_TIME_SOURCE_COLUMN not in result.columns:
            raise ValueError(f"Missing required time column {TIME_COLUMN!r}.")
        timestamps = pd.to_datetime(result[RAW_TIME_SOURCE_COLUMN], errors="coerce", cache=True)
        if timestamps.isna().all():
            raise ValueError(f"Could not derive {TIME_COLUMN!r} from {RAW_TIME_SOURCE_COLUMN!r}.")
        t0 = timestamps.dropna().iloc[0]
        result[TIME_COLUMN] = (timestamps - t0).dt.total_seconds() / (7.0 * 24.0 * 3600.0)
    result[TIME_COLUMN] = pd.to_numeric(result[TIME_COLUMN], errors="coerce")
    if "SOH" not in result.columns:
        capacity = None
        source_column = None
        for candidate in CAPACITY_FALLBACK_COLUMNS:
            if candidate not in result.columns:
                continue
            candidate_values = pd.to_numeric(result[candidate], errors="coerce").to_numpy(dtype=float)
            finite_idx = np.flatnonzero(np.isfinite(candidate_values) & (candidate_values > 0.0))
            if finite_idx.size == 0:
                continue
            capacity = candidate_values
            source_column = candidate
            base_capacity = float(candidate_values[finite_idx[0]])
            result["SOH"] = candidate_values / base_capacity
            break
        if capacity is None or source_column is None:
            raise ValueError(
                "Missing both 'SOH' and any usable raw capacity column "
                f"from {CAPACITY_FALLBACK_COLUMNS!r}."
            )
    result["SOH"] = pd.to_numeric(result["SOH"], errors="coerce").clip(lower=0.0, upper=1.05)
    if THROUGHPUT_COLUMN in result.columns:
        result[THROUGHPUT_COLUMN] = pd.to_numeric(result[THROUGHPUT_COLUMN], errors="coerce")
    result[REFERENCE_COLUMN] = np.arange(len(result), dtype=float)
    return result.reset_index(drop=True)


def infer_train_test_names(
    cell_names: list[str],
    *,
    explicit_test_names: list[str] | tuple[str, ...] | None = None,
    test_substring: str = DEFAULT_TEST_SUBSTRING,
    train_substring: str = DEFAULT_TRAIN_SUBSTRING,
) -> tuple[list[str], list[str]]:
    normalized_by_name = {str(name).strip().lower(): str(name) for name in cell_names}
    matched_explicit: list[str] = []
    for test_name in explicit_test_names or ():
        key = str(test_name).strip().lower()
        if key in normalized_by_name:
            matched_explicit.append(normalized_by_name[key])
    if matched_explicit:
        test_names = sorted(set(matched_explicit))
    else:
        test_token = str(test_substring).strip().lower()
        test_names = sorted(name for name in cell_names if test_token and test_token in name.lower())

    train_token = str(train_substring).strip().lower()
    train_names = sorted(
        name
        for name in cell_names
        if name not in set(test_names) and train_token and train_token in name.lower()
    )
    if not train_names:
        train_names = sorted(name for name in cell_names if name not in set(test_names))
    return train_names, test_names


def load_lfp_cell_frames(
    *,
    interp_dir: Path,
    feature_columns: list[str],
    target_column: str | None,
    target_soh: float | None = DEFAULT_TARGET_SOH,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    loaded_frames: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for csv_path in sorted(interp_dir.glob("*.csv")):
        if csv_path.name.startswith("_"):
            continue
        cell_name = csv_path.stem
        try:
            df = prepare_lfp_dataframe(pd.read_csv(csv_path))
        except Exception as exc:
            skipped.append({"cell": cell_name, "reason": f"read_error:{exc}"})
            continue
        required_columns = [*feature_columns, "SOH", TIME_COLUMN, THROUGHPUT_COLUMN]
        if target_soh is None and target_column is not None:
            required_columns.append(str(target_column))
        missing = [name for name in required_columns if name not in df.columns]
        if missing:
            skipped.append({"cell": cell_name, "reason": f"missing_columns:{','.join(missing)}"})
            continue
        if target_soh is not None:
            target_value = compute_anchor_reference(
                df,
                float(target_soh),
                reference_column=TIME_COLUMN,
            )
            target_throughput_value = compute_anchor_reference(
                df,
                float(target_soh),
                reference_column=THROUGHPUT_COLUMN,
            )
        elif target_column is not None:
            target_value = float(pd.to_numeric(df[target_column], errors="coerce").iloc[0])
            target_throughput_value = np.nan
        else:
            target_value = np.nan
            target_throughput_value = np.nan
        if not np.isfinite(target_value) or target_value <= 0.0:
            skipped.append({"cell": cell_name, "reason": "invalid_target"})
            continue
        if target_soh is not None and (not np.isfinite(target_throughput_value) or target_throughput_value <= 0.0):
            skipped.append({"cell": cell_name, "reason": "invalid_target_throughput"})
            continue
        loaded_frames.append(
            {
                "cell": str(cell_name),
                "csv_path": csv_path,
                "raw_df": df,
                "target_value": float(target_value),
                "target_week_value": float(target_value),
                "target_throughput_value": float(target_throughput_value),
                "target_soh": float(target_soh) if target_soh is not None else np.nan,
            }
        )
    return loaded_frames, skipped


def _clean_curve(reference_values: object, values: object) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_values, dtype=float).reshape(-1)
    series = np.asarray(values, dtype=float).reshape(-1)
    point_count = min(int(reference.size), int(series.size))
    reference = reference[:point_count]
    series = series[:point_count]
    valid = np.isfinite(reference) & np.isfinite(series)
    reference = reference[valid]
    series = series[valid]
    if reference.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    order = np.argsort(reference)
    return reference[order], series[order]


def compute_anchor_reference(
    raw_df: pd.DataFrame,
    anchor_soh: float,
    *,
    reference_column: str = TIME_COLUMN,
) -> float:
    if not np.isfinite(float(anchor_soh)):
        return np.nan
    reference, soh = _clean_curve(
        raw_df[reference_column],
        pd.to_numeric(raw_df["SOH"], errors="coerce").to_numpy(dtype=float),
    )
    if reference.size < 2:
        return np.nan
    soh = np.clip(soh, 0.0, 1.05)
    soh = np.minimum.accumulate(soh)
    lower = float(np.min(soh))
    upper = float(np.max(soh))
    if float(anchor_soh) < lower or float(anchor_soh) > upper:
        return np.nan
    reverse_soh = soh[::-1]
    reverse_reference = reference[::-1]
    unique_soh, unique_indices = np.unique(reverse_soh, return_index=True)
    unique_reference = reverse_reference[unique_indices]
    if unique_soh.size < 2:
        return np.nan
    return float(np.interp(float(anchor_soh), unique_soh, unique_reference))


def interpolate_feature_vector_from_df(
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    reference_column: str,
    target_reference: float,
) -> np.ndarray | None:
    if not np.isfinite(float(target_reference)):
        return None
    reference = pd.to_numeric(raw_df[reference_column], errors="coerce").to_numpy(dtype=float)
    vector: list[float] = []
    for feature_name in feature_columns:
        values = pd.to_numeric(raw_df[feature_name], errors="coerce").to_numpy(dtype=float)
        ref_valid, value_valid = _clean_curve(reference, values)
        if ref_valid.size < 2:
            return None
        if float(target_reference) < float(ref_valid[0]) or float(target_reference) > float(ref_valid[-1]):
            return None
        vector.append(float(np.interp(float(target_reference), ref_valid, value_valid)))
    output = np.asarray(vector, dtype=float)
    return output if np.isfinite(output).all() else None


def interpolate_feature_sequence_from_df(
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    reference_column: str,
    target_reference: float,
    input_steps: int,
) -> np.ndarray | None:
    if not np.isfinite(float(target_reference)) or float(target_reference) <= 0.0:
        return None
    reference = pd.to_numeric(raw_df[reference_column], errors="coerce").to_numpy(dtype=float)
    grid = np.linspace(0.0, float(target_reference), int(input_steps), dtype=float)
    columns: list[np.ndarray] = []
    for feature_name in feature_columns:
        values = pd.to_numeric(raw_df[feature_name], errors="coerce").to_numpy(dtype=float)
        ref_valid, value_valid = _clean_curve(reference, values)
        if ref_valid.size < 2:
            return None
        if float(grid[0]) < float(ref_valid[0]) or float(grid[-1]) > float(ref_valid[-1]):
            return None
        columns.append(np.interp(grid, ref_valid, value_valid).reshape(-1, 1))
    sequence = np.hstack(columns)
    return sequence if np.isfinite(sequence).all() else None


def interpolate_feature_sequence_on_grid(
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    reference_column: str,
    grid: np.ndarray,
) -> np.ndarray | None:
    reference = pd.to_numeric(raw_df[reference_column], errors="coerce").to_numpy(dtype=float)
    if grid.size == 0 or not np.isfinite(grid).all():
        return None
    columns: list[np.ndarray] = []
    for feature_name in feature_columns:
        values = pd.to_numeric(raw_df[feature_name], errors="coerce").to_numpy(dtype=float)
        ref_valid, value_valid = _clean_curve(reference, values)
        if ref_valid.size < 2:
            return None
        if float(grid[0]) < float(ref_valid[0]) or float(grid[-1]) > float(ref_valid[-1]):
            return None
        columns.append(np.interp(grid, ref_valid, value_valid).reshape(-1, 1))
    sequence = np.hstack(columns)
    return sequence if np.isfinite(sequence).all() else None


def build_week_step_grid(anchor_week: float) -> np.ndarray:
    if not np.isfinite(float(anchor_week)) or float(anchor_week) <= 0.0:
        return np.asarray([], dtype=float)
    upper_int = int(np.floor(float(anchor_week)))
    grid = np.arange(0, upper_int + 1, dtype=float)
    if grid.size == 0:
        grid = np.asarray([0.0], dtype=float)
    if not np.isclose(float(grid[-1]), float(anchor_week), rtol=0.0, atol=1e-9):
        grid = np.concatenate([grid, np.asarray([float(anchor_week)], dtype=float)])
    return grid


def find_first_index_at_or_below_soh(raw_df: pd.DataFrame, threshold_soh: float) -> int | None:
    soh = pd.to_numeric(raw_df["SOH"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(soh) & (soh <= float(threshold_soh))
    if not np.any(mask):
        return None
    return int(np.flatnonzero(mask)[0])


def find_first_index_at_or_above_week(raw_df: pd.DataFrame, target_week: float) -> int | None:
    weeks = pd.to_numeric(raw_df[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(weeks) & (weeks >= float(target_week))
    if not np.any(mask):
        return None
    return int(np.flatnonzero(mask)[0])


def find_first_index_at_or_above_throughput(raw_df: pd.DataFrame, target_throughput: float) -> int | None:
    throughput = pd.to_numeric(raw_df[THROUGHPUT_COLUMN], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(throughput) & (throughput >= float(target_throughput))
    if not np.any(mask):
        return None
    return int(np.flatnonzero(mask)[0])


def pad_sequence_to_length(sequence: np.ndarray, target_length: int) -> np.ndarray:
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 2:
        raise ValueError("Expected a 2D sequence to pad.")
    desired = max(1, int(target_length))
    if seq.shape[0] >= desired:
        return seq[:desired]
    if seq.shape[0] == 0:
        raise ValueError("Cannot pad an empty sequence.")
    pad_count = desired - seq.shape[0]
    pad_block = np.repeat(seq[[-1], :], pad_count, axis=0)
    return np.vstack([seq, pad_block])


def pad_vector_to_length(values: np.ndarray, target_length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    desired = max(1, int(target_length))
    if vector.size >= desired:
        return vector[:desired]
    if vector.size == 0:
        raise ValueError("Cannot pad an empty vector.")
    pad_count = desired - vector.size
    pad_block = np.repeat(vector[[-1]], pad_count)
    return np.concatenate([vector, pad_block])


def resolve_week_step_count(
    cell_frames: list[dict[str, object]],
    *,
    anchor_week: float | None = None,
    anchor_throughput: float | None = None,
    anchor_soh: float | None = None,
    fallback_steps: int,
) -> int:
    if anchor_week is not None:
        lengths: list[int] = []
        for frame in cell_frames:
            anchor_index = find_first_index_at_or_above_week(frame["raw_df"], float(anchor_week))
            if anchor_index is not None and int(anchor_index) + 1 >= 2:
                lengths.append(int(anchor_index) + 1)
        if lengths:
            return max(lengths)
        return max(2, int(fallback_steps))
    if anchor_throughput is not None:
        lengths: list[int] = []
        for frame in cell_frames:
            anchor_index = find_first_index_at_or_above_throughput(frame["raw_df"], float(anchor_throughput))
            if anchor_index is not None and int(anchor_index) + 1 >= 2:
                lengths.append(int(anchor_index) + 1)
        if lengths:
            return max(lengths)
        return max(2, int(fallback_steps))
    if anchor_soh is not None:
        lengths: list[int] = []
        for frame in cell_frames:
            anchor_index = find_first_index_at_or_below_soh(frame["raw_df"], float(anchor_soh))
            if anchor_index is not None and int(anchor_index) + 1 >= 2:
                lengths.append(int(anchor_index) + 1)
        if lengths:
            return max(lengths)
    return max(2, int(fallback_steps))


def build_sequence_for_frame(
    frame: dict[str, object],
    *,
    feature_columns: list[str],
    input_steps: int,
    anchor_soh: float | None,
    anchor_week: float | None = None,
    anchor_throughput: float | None = None,
    sequence_steps: int | None = None,
) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    raw_df = frame["raw_df"]
    if anchor_week is not None:
        anchor_index = find_first_index_at_or_above_week(raw_df, float(anchor_week))
        if anchor_index is None or int(anchor_index) < 1:
            return None, np.nan, None
        feature_frame = raw_df[feature_columns].apply(pd.to_numeric, errors="coerce")
        sequence = feature_frame.iloc[: int(anchor_index) + 1].to_numpy(dtype=float)
        week_grid = pd.to_numeric(raw_df[TIME_COLUMN], errors="coerce").iloc[: int(anchor_index) + 1].to_numpy(dtype=float)
        if not np.isfinite(sequence).all() or not np.isfinite(week_grid).all():
            return None, np.nan, None
        if sequence_steps is not None:
            sequence = pad_sequence_to_length(sequence, int(sequence_steps))
            week_grid = pad_vector_to_length(week_grid, int(sequence_steps))
        return sequence, float(week_grid[min(int(anchor_index), len(week_grid) - 1)]), week_grid
    if anchor_throughput is not None:
        anchor_index = find_first_index_at_or_above_throughput(raw_df, float(anchor_throughput))
        if anchor_index is None or int(anchor_index) < 1:
            return None, np.nan, None
        feature_frame = raw_df[feature_columns].apply(pd.to_numeric, errors="coerce")
        sequence = feature_frame.iloc[: int(anchor_index) + 1].to_numpy(dtype=float)
        week_grid = pd.to_numeric(raw_df[TIME_COLUMN], errors="coerce").iloc[: int(anchor_index) + 1].to_numpy(dtype=float)
        if not np.isfinite(sequence).all() or not np.isfinite(week_grid).all():
            return None, np.nan, None
        throughput_curve = pd.to_numeric(raw_df[THROUGHPUT_COLUMN], errors="coerce").to_numpy(dtype=float)
        input_reference_value = float(throughput_curve[min(int(anchor_index), len(throughput_curve) - 1)])
        if sequence_steps is not None:
            sequence = pad_sequence_to_length(sequence, int(sequence_steps))
            week_grid = pad_vector_to_length(week_grid, int(sequence_steps))
        return sequence, input_reference_value, week_grid
    if anchor_soh is None:
        feature_frame = raw_df[feature_columns].apply(pd.to_numeric, errors="coerce")
        if len(feature_frame) < int(input_steps):
            return None, np.nan, None
        sequence = feature_frame.iloc[: int(input_steps)].to_numpy(dtype=float)
        if not np.isfinite(sequence).all():
            return None, np.nan, None
        week_grid = pd.to_numeric(raw_df[TIME_COLUMN], errors="coerce").iloc[: int(input_steps)].to_numpy(dtype=float)
        return sequence, float(int(input_steps) - 1), week_grid
    anchor_index = find_first_index_at_or_below_soh(raw_df, float(anchor_soh))
    if anchor_index is None or int(anchor_index) < 1:
        return None, np.nan, None
    feature_frame = raw_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    sequence = feature_frame.iloc[: int(anchor_index) + 1].to_numpy(dtype=float)
    week_grid = pd.to_numeric(raw_df[TIME_COLUMN], errors="coerce").iloc[: int(anchor_index) + 1].to_numpy(dtype=float)
    if not np.isfinite(sequence).all() or not np.isfinite(week_grid).all():
        return None, np.nan, None
    if sequence_steps is not None:
        sequence = pad_sequence_to_length(sequence, int(sequence_steps))
        week_grid = pad_vector_to_length(week_grid, int(sequence_steps))
    return sequence, float(week_grid[min(int(anchor_index), len(week_grid) - 1)]), week_grid


def resolve_default_anchor_soh(
    cell_frames: list[dict[str, object]],
    *,
    input_steps: int,
) -> float:
    step_index = max(0, int(input_steps) - 1)
    candidate_sohs: list[float] = []
    for frame in cell_frames:
        raw_df = frame["raw_df"]
        if len(raw_df) <= step_index:
            continue
        value = float(pd.to_numeric(raw_df["SOH"], errors="coerce").iloc[step_index])
        if np.isfinite(value):
            candidate_sohs.append(float(value))
    if not candidate_sohs:
        raise RuntimeError("Could not resolve a default common anchor SOH from the available cells.")
    return float(np.min(candidate_sohs))


def build_model_cells(
    cell_frames: list[dict[str, object]],
    *,
    feature_columns: list[str],
    input_steps: int,
    anchor_soh: float | None,
    anchor_week: float | None = None,
    anchor_throughput: float | None = None,
    sequence_steps: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    usable_cells: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for frame in cell_frames:
        sequence, input_reference_value, input_week_grid = build_sequence_for_frame(
            frame,
            feature_columns=feature_columns,
            input_steps=int(input_steps),
            anchor_soh=anchor_soh,
            anchor_week=anchor_week,
            anchor_throughput=anchor_throughput,
            sequence_steps=sequence_steps,
        )
        if sequence is None:
            skipped.append(
                {
                    "cell": str(frame["cell"]),
                    "reason": "unusable_sequence",
                    "anchor_soh": float(anchor_soh) if anchor_soh is not None else np.nan,
                }
            )
            continue
        raw_df = frame["raw_df"]
        reference_curve = pd.to_numeric(raw_df[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
        throughput_curve = pd.to_numeric(raw_df[THROUGHPUT_COLUMN], errors="coerce").to_numpy(dtype=float)
        soh_curve = pd.to_numeric(raw_df["SOH"], errors="coerce").to_numpy(dtype=float)
        if anchor_week is not None:
            input_end_index = find_first_index_at_or_above_week(raw_df, float(anchor_week))
        elif anchor_throughput is not None:
            input_end_index = find_first_index_at_or_above_throughput(raw_df, float(anchor_throughput))
        elif anchor_soh is not None:
            input_end_index = find_first_index_at_or_below_soh(raw_df, float(anchor_soh))
        else:
            input_end_index = len(reference_curve) - 1
        if input_end_index is None:
            input_throughput_value = np.nan
        else:
            safe_index = max(0, min(int(input_end_index), len(throughput_curve) - 1))
            input_throughput_value = float(throughput_curve[safe_index]) if len(throughput_curve) else np.nan
        usable_cells.append(
            {
                "cell": str(frame["cell"]),
                "raw_df": raw_df,
                "x": np.asarray(sequence[-1], dtype=float),
                "sequence": np.asarray(sequence, dtype=float),
                "target_value": float(frame["target_value"]),
                "target_week_value": float(frame.get("target_week_value", frame["target_value"])),
                "target_throughput_value": float(frame.get("target_throughput_value", np.nan)),
                "input_reference_value": float(input_reference_value),
                "input_throughput_value": float(input_throughput_value) if np.isfinite(input_throughput_value) else np.nan,
                "input_week_grid": np.asarray(input_week_grid, dtype=float) if input_week_grid is not None else np.asarray([], dtype=float),
                "time_curve": np.asarray(reference_curve, dtype=float),
                "throughput_curve": np.asarray(throughput_curve, dtype=float),
                "soh_curve": np.asarray(soh_curve, dtype=float),
                "source_kind": "real",
            }
        )
    return usable_cells, skipped


def evaluate_anchor_candidate(
    cell_frames: list[dict[str, object]],
    *,
    anchor_soh: float,
    input_steps: int,
) -> tuple[bool, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    valid = True
    for frame in cell_frames:
        anchor_reference = compute_anchor_reference(frame["raw_df"], float(anchor_soh))
        row = {
            "cell": str(frame["cell"]),
            "anchor_soh": float(anchor_soh),
            "input_steps": int(input_steps),
            "anchor_reference": float(anchor_reference) if np.isfinite(anchor_reference) else np.nan,
            "status": "ok",
            "reason": "",
        }
        if not np.isfinite(anchor_reference):
            row["status"] = "invalid"
            row["reason"] = "no_soh_crossing"
            valid = False
        elif float(anchor_reference) < float(int(input_steps) - 1):
            row["status"] = "invalid"
            row["reason"] = "insufficient_window"
            valid = False
        rows.append(row)
    return valid, rows


def evaluate_week_anchor_candidate(
    cell_frames: list[dict[str, object]],
    *,
    anchor_week: float,
    input_steps: int,
) -> tuple[bool, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    valid = True
    for frame in cell_frames:
        weeks = pd.to_numeric(frame["raw_df"][TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
        finite_weeks = weeks[np.isfinite(weeks)]
        max_week = float(np.max(finite_weeks)) if finite_weeks.size else np.nan
        row = {
            "cell": str(frame["cell"]),
            "anchor_week": float(anchor_week),
            "input_steps": int(input_steps),
            "max_week": float(max_week) if np.isfinite(max_week) else np.nan,
            "status": "ok",
            "reason": "",
        }
        if not np.isfinite(max_week) or float(max_week) < float(anchor_week):
            row["status"] = "invalid"
            row["reason"] = "insufficient_max_week"
            valid = False
        rows.append(row)
    return valid, rows


def evaluate_balanced_anchor_candidate(
    train_frames: list[dict[str, object]],
    test_frames: list[dict[str, object]],
    *,
    anchor_soh: float,
    input_steps: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    train_valid, train_rows = evaluate_anchor_candidate(
        train_frames,
        anchor_soh=float(anchor_soh),
        input_steps=int(input_steps),
    )
    test_valid, test_rows = evaluate_anchor_candidate(
        test_frames,
        anchor_soh=float(anchor_soh),
        input_steps=int(input_steps),
    )
    usable_train_count = sum(1 for row in train_rows if row["status"] == "ok")
    usable_test_count = sum(1 for row in test_rows if row["status"] == "ok")
    summary = {
        "anchor_soh": float(anchor_soh),
        "strict_valid": bool(train_valid and test_valid),
        "all_test_usable": bool(test_valid),
        "usable_train_count": int(usable_train_count),
        "total_train_count": int(len(train_rows)),
        "usable_test_count": int(usable_test_count),
        "total_test_count": int(len(test_rows)),
    }
    detailed_rows = [
        {**row, "split": "train"}
        for row in train_rows
    ] + [
        {**row, "split": "test"}
        for row in test_rows
    ]
    return summary, detailed_rows


def select_autoanchor_soh(
    cell_frames: list[dict[str, object]],
    *,
    input_steps: int,
    min_soh: float = DEFAULT_AUTOANCHOR_MIN_SOH,
    max_soh: float = DEFAULT_AUTOANCHOR_MAX_SOH,
    step_soh: float = DEFAULT_AUTOANCHOR_STEP_SOH,
) -> tuple[float, pd.DataFrame]:
    candidates = build_anchor_candidates(
        min_soh=float(min_soh),
        max_soh=float(max_soh),
        step_soh=float(step_soh),
        descending=False,
    )

    def evaluate(candidate_soh: float) -> tuple[bool, list[dict[str, object]]]:
        return evaluate_anchor_candidate(
            cell_frames,
            anchor_soh=float(candidate_soh),
            input_steps=int(input_steps),
        )

    selected_anchor, search_df = select_lowest_valid_anchor(
        candidates=candidates,
        evaluate_candidate=evaluate,
        require_scan_all=False,
    )
    return float(selected_anchor), search_df


def select_balanced_anchor_soh(
    train_frames: list[dict[str, object]],
    test_frames: list[dict[str, object]],
    *,
    input_steps: int,
    min_soh: float = DEFAULT_AUTOANCHOR_MIN_SOH,
    max_soh: float = DEFAULT_AUTOANCHOR_MAX_SOH,
    step_soh: float = DEFAULT_AUTOANCHOR_STEP_SOH,
) -> tuple[float, pd.DataFrame]:
    candidates = build_anchor_candidates(
        min_soh=float(min_soh),
        max_soh=float(max_soh),
        step_soh=float(step_soh),
        descending=False,
    )
    best_summary: dict[str, object] | None = None
    summary_rows: list[dict[str, object]] = []
    for candidate in candidates:
        summary, _rows = evaluate_balanced_anchor_candidate(
            train_frames,
            test_frames,
            anchor_soh=float(candidate),
            input_steps=int(input_steps),
        )
        summary_rows.append(summary)
        if best_summary is None:
            best_summary = dict(summary)
            continue
        current_key = (
            int(bool(summary["all_test_usable"])),
            int(summary["usable_test_count"]),
            int(summary["usable_train_count"]),
        )
        best_key = (
            int(bool(best_summary["all_test_usable"])),
            int(best_summary["usable_test_count"]),
            int(best_summary["usable_train_count"]),
        )
        if current_key > best_key:
            best_summary = dict(summary)
    if best_summary is None:
        raise RuntimeError("Could not evaluate any autoanchor candidates for the LFP dataset.")
    summary_df = pd.DataFrame(summary_rows)
    summary_df["selected"] = summary_df["anchor_soh"].eq(float(best_summary["anchor_soh"]))
    return float(best_summary["anchor_soh"]), summary_df


class LFPInterpolationModule:
    TIME_COLUMN = TIME_COLUMN
    CAPACITY_COLUMN = "SOH"
    THROUGHPUT_COLUMN = THROUGHPUT_COLUMN

    def __init__(
        self,
        *,
        input_steps: int,
        augmentation_reference_soh: float,
        max_sequence_steps: int | None = None,
        scale_spacing: str = core_aug.NORMAL_AUGMENTATION_SCALE_SPACING,
        fixed_low_scale: float = core_aug.NORMAL_FIXED_LOW_AUGMENTATION_SCALE,
        factor_count: int = core_aug.NORMAL_AUGMENTATION_FACTOR_COUNT,
        scale_step: float = core_aug.NORMAL_AUGMENTATION_SCALE_STEP,
    ) -> None:
        self.DISCRETIZATION_STEPS = int(input_steps)
        self.AUGMENTATION_REFERENCE_SOH = float(augmentation_reference_soh)
        self.MAX_SEQUENCE_STEPS = None if max_sequence_steps is None else int(max_sequence_steps)
        self.NORMAL_AUGMENTATION_SCALE_SPACING = str(scale_spacing)
        self.NORMAL_FIXED_LOW_AUGMENTATION_SCALE = float(fixed_low_scale)
        self.NORMAL_AUGMENTATION_FACTOR_COUNT = int(factor_count)
        self.NORMAL_AUGMENTATION_SCALE_STEP = float(scale_step)

    def extract_time_or_throughput_at_soh(
        self,
        cell: dict[str, object],
        *,
        task_kind: str = "time",
    ) -> float:
        return compute_anchor_reference(
            cell["raw_df"],
            float(self.AUGMENTATION_REFERENCE_SOH),
            reference_column=TIME_COLUMN if str(task_kind).strip().lower() == "time" else THROUGHPUT_COLUMN,
        )

    def collect_time_or_throughput_values_at_soh(
        self,
        cells: list[dict[str, object]],
        *,
        task_kind: str = "time",
    ) -> np.ndarray:
        return np.asarray(
            [self.extract_time_or_throughput_at_soh(cell, task_kind=str(task_kind)) for cell in cells],
            dtype=float,
        )

    def interpolate_feature_vector(
        self,
        raw_df: pd.DataFrame,
        feature_columns: list[str],
        *,
        reference_column: str,
        target_reference: float,
    ) -> np.ndarray | None:
        return interpolate_feature_vector_from_df(
            raw_df,
            [str(name) for name in feature_columns],
            reference_column=str(reference_column),
            target_reference=float(target_reference),
        )

    def interpolate_feature_sequence(
        self,
        raw_df: pd.DataFrame,
        feature_columns: list[str],
        *,
        reference_column: str,
        target_reference: float,
        discretization_steps: int,
    ) -> np.ndarray | None:
        if str(reference_column) == TIME_COLUMN:
            grid = build_week_step_grid(float(target_reference))
            sequence = interpolate_feature_sequence_on_grid(
                raw_df=raw_df,
                feature_columns=[str(name) for name in feature_columns],
                reference_column=str(reference_column),
                grid=grid,
            )
            if sequence is not None and self.MAX_SEQUENCE_STEPS is not None:
                sequence = pad_sequence_to_length(sequence, int(self.MAX_SEQUENCE_STEPS))
            return sequence
        return interpolate_feature_sequence_from_df(
            reference_column=str(reference_column),
            raw_df=raw_df,
            feature_columns=[str(name) for name in feature_columns],
            target_reference=float(target_reference),
            input_steps=int(discretization_steps),
        )


def build_augmented_cells(
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    feature_columns: list[str],
    input_steps: int,
    sequence_steps: int | None,
    anchor_soh: float,
    augmentation_method: str,
    task_kind: str = "time",
    sample_count: int | None = None,
    seed_source: int = DEFAULT_SEED_SOURCE,
    scale_factor_count: int = core_aug.NORMAL_AUGMENTATION_FACTOR_COUNT,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    method = str(augmentation_method).strip().lower()
    if method in {"", "none", "off"}:
        return [], {"augmentation_method": "none", "augmentation_reference_soh": float(anchor_soh)}
    if method not in LOCAL_AUGMENTATION_METHODS:
        return _build_exported_augmented_cells(
            train_cells,
            test_cells,
            feature_columns=feature_columns,
            input_steps=int(input_steps),
            sequence_steps=sequence_steps,
            anchor_soh=float(anchor_soh),
            augmentation_method=str(method),
            task_kind=str(task_kind),
        )
    module = LFPInterpolationModule(
        input_steps=int(input_steps),
        augmentation_reference_soh=float(anchor_soh),
        max_sequence_steps=sequence_steps,
        factor_count=int(scale_factor_count),
    )
    scales, scale_stats = core_aug.build_normal_augmentation_scales(
        module,
        train_cells,
        test_cells,
        task_kind=str(task_kind),
    )
    metadata: dict[str, object] = {
        **scale_stats,
        "augmentation_method": str(method),
        "augmentation_reference_soh": float(anchor_soh),
        "augmentation_scales": [float(scale) for scale in scales],
        "augmentation_scale_count": int(len(scales)),
        "augmentation_interpolation_mode": "raw_sequence_scaling" if method == "normal" else "interpolated_generation",
    }
    if method == "normal":
        augmented: list[dict[str, object]] = []
        for cell in train_cells:
            base_sequence = np.asarray(cell["sequence"], dtype=float)
            base_x = np.asarray(cell["x"], dtype=float)
            base_time_curve = np.asarray(cell["time_curve"], dtype=float)
            base_throughput_curve = np.asarray(cell["throughput_curve"], dtype=float)
            base_soh_curve = np.asarray(cell["soh_curve"], dtype=float)
            base_input_week_grid = np.asarray(cell.get("input_week_grid", []), dtype=float)
            base_target_value = float(cell["target_value"])
            base_target_throughput = float(cell.get("target_throughput_value", np.nan))
            base_input_reference = float(cell["input_reference_value"])
            for scale in scales:
                effective_scale = float(scale)
                if not np.isfinite(effective_scale) or effective_scale <= 1.0 + 1e-9:
                    continue
                scaled_week_target = (
                    float(base_target_value) * float(effective_scale)
                )
                scaled_throughput_target = (
                    float(base_target_throughput) * float(effective_scale)
                )
                active_target_value = (
                    float(scaled_throughput_target)
                    if str(task_kind).strip().lower() == "throughput"
                    else float(scaled_week_target)
                )
                if not np.isfinite(active_target_value) or active_target_value <= float(base_input_reference):
                    continue
                augmented.append(
                    {
                        "cell": f"{cell['cell']}_aug_{effective_scale:.3f}",
                        "source_kind": "cyclic_augmented",
                        "x": base_x.copy(),
                        "sequence": base_sequence.copy(),
                        "target_value": float(scaled_week_target),
                        "target_week_value": float(scaled_week_target),
                        "target_throughput_value": float(scaled_throughput_target),
                        "input_reference_value": float(base_input_reference),
                        "input_throughput_value": float(cell.get("input_throughput_value", np.nan)),
                        "input_week_grid": base_input_week_grid.copy(),
                        "time_curve": base_time_curve * float(effective_scale),
                        "throughput_curve": base_throughput_curve * float(effective_scale),
                        "soh_curve": base_soh_curve.copy(),
                        "normal_scale": float(effective_scale),
                    }
                )
        metadata["train_aug_count"] = int(len(augmented))
        return augmented, metadata
    if method == "cal":
        effective_count = int(
            DEFAULT_CAL_AUGMENTATION_SAMPLE_COUNT if sample_count is None else max(0, int(sample_count))
        )
        augmented = core_aug.build_cal_augmented_cells(
            module,
            train_cells,
            test_cells,
            feature_columns,
            task_kind=str(task_kind),
            sample_count=effective_count,
            random_seed=int(seed_source),
        )
        metadata.update(
            core_aug.build_cal_augmentation_stats(
                module,
                train_cells,
                test_cells,
                task_kind=str(task_kind),
                sample_count=effective_count,
            )
        )
        metadata["train_aug_count"] = int(len(augmented))
        return augmented, metadata
    if method == "hybrid":
        effective_count = int(
            DEFAULT_HYBRID_AUGMENTATION_SAMPLE_COUNT if sample_count is None else max(0, int(sample_count))
        )
        augmented = core_aug.build_hybrid_augmented_cells(
            module,
            train_cells,
            test_cells,
            feature_columns,
            task_kind=str(task_kind),
            augmentation_scales=scales,
            sample_count=effective_count,
            random_seed=int(seed_source),
        )
        metadata["train_aug_count"] = int(len(augmented))
        return augmented, metadata
    if method == "codex":
        effective_count = int(
            DEFAULT_CODEX_AUGMENTATION_SAMPLE_COUNT if sample_count is None else max(0, int(sample_count))
        )
        augmented = core_aug.build_codex_augmented_cells(
            module,
            train_cells,
            test_cells,
            feature_columns,
            task_kind=str(task_kind),
            sample_count=effective_count,
            random_seed=int(seed_source),
        )
        metadata["train_aug_count"] = int(len(augmented))
        return augmented, metadata
    raise ValueError(f"Unsupported augmentation method {augmentation_method!r}.")


def scale_cell_sequences(
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    feature_columns: list[str],
    exempt_feature_names: tuple[str, ...] = DEFAULT_EXEMPT_FROM_SCALING,
) -> tuple[list[dict[str, object]], list[dict[str, object]], StandardScaler]:
    feature_names = [str(name) for name in feature_columns]
    exempt = {str(name) for name in exempt_feature_names}
    scale_indices = [index for index, name in enumerate(feature_names) if name not in exempt]
    if not scale_indices:
        return train_cells, test_cells, StandardScaler()

    train_sequences = np.asarray([cell["sequence"] for cell in train_cells], dtype=float)
    flat_train = train_sequences.reshape(-1, train_sequences.shape[-1])
    scaler = StandardScaler()
    scaler.fit(flat_train[:, scale_indices])

    def transform_cells(cells: list[dict[str, object]]) -> list[dict[str, object]]:
        transformed: list[dict[str, object]] = []
        for cell in cells:
            sequence = np.asarray(cell["sequence"], dtype=float).copy()
            reshaped = sequence.reshape(-1, sequence.shape[-1])
            reshaped[:, scale_indices] = scaler.transform(reshaped[:, scale_indices])
            scaled_sequence = reshaped.reshape(sequence.shape)
            transformed.append(
                {
                    **cell,
                    "sequence": scaled_sequence,
                    "x": np.asarray(scaled_sequence[-1], dtype=float),
                }
            )
        return transformed

    return transform_cells(train_cells), transform_cells(test_cells), scaler


def build_prediction_dataframe(
    test_names: list[str],
    y_true_steps: np.ndarray,
    y_pred_steps: np.ndarray,
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_true_throughput: np.ndarray | None = None,
    y_pred_throughput: np.ndarray | None = None,
    y_pred_log_throughput: np.ndarray | None = None,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "cell": [str(name) for name in test_names],
            "true_weeks": np.asarray(y_true_steps, dtype=float),
            "pred_weeks": np.asarray(y_pred_steps, dtype=float),
            "abs_error_weeks": np.abs(np.asarray(y_true_steps, dtype=float) - np.asarray(y_pred_steps, dtype=float)),
            "true_log_weeks": np.asarray(y_true_log, dtype=float),
            "pred_log_weeks": np.asarray(y_pred_log, dtype=float),
        }
    )
    if y_true_throughput is not None:
        output["true_throughput_at_target"] = np.asarray(y_true_throughput, dtype=float)
    if y_pred_throughput is not None:
        output["pred_throughput_at_target"] = np.asarray(y_pred_throughput, dtype=float)
        if y_true_throughput is not None:
            output["abs_error_throughput_at_target"] = np.abs(
                np.asarray(y_true_throughput, dtype=float) - np.asarray(y_pred_throughput, dtype=float)
            )
    if y_pred_log_throughput is not None:
        output["pred_log_throughput"] = np.asarray(y_pred_log_throughput, dtype=float)
    return output


def _feature_columns_without_soh(feature_columns: list[str]) -> list[str]:
    return [str(name) for name in feature_columns if str(name).strip().lower() != "soh"]


def _build_lfp_helper_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    result = raw_df.copy()
    normalized_columns = {str(column).strip().lower(): str(column) for column in result.columns}
    for helper_name, source_name in LFP_AUGMENTATION_HELPER_RAW_COLUMN_ALIASES.items():
        helper_column = str(helper_name)
        source_column = normalized_columns.get(str(source_name).strip().lower())
        if source_column is None or helper_column in result.columns:
            continue
        result[helper_column] = pd.to_numeric(result[source_column], errors="coerce")
    return result


def _interpolate_helper_feature_vector(
    module: "LFPInterpolationModule",
    raw_df: pd.DataFrame,
    *,
    task_kind: str,
    input_reference_value: float,
    helper_feature_columns: list[str],
) -> np.ndarray | None:
    helper_raw_df = _build_lfp_helper_raw_df(raw_df)
    reference_column = TIME_COLUMN if str(task_kind).strip().lower() == "time" else THROUGHPUT_COLUMN
    return module.interpolate_feature_vector(
        helper_raw_df,
        list(helper_feature_columns),
        reference_column=reference_column,
        target_reference=float(input_reference_value),
    )


def _prepare_helper_feature_cells(
    exported_aug,
    module: "LFPInterpolationModule",
    cells: list[dict[str, object]],
    *,
    feature_columns: list[str],
    task_kind: str,
    helper_feature_columns: list[str],
) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for cell in cells:
        copied = exported_aug.preserve_process_features(cell, feature_columns)
        helper_raw_df = _build_lfp_helper_raw_df(copied["raw_df"])
        input_reference_value = float(copied.get("input_reference_value", np.nan))
        helper_vector = _interpolate_helper_feature_vector(
            module,
            helper_raw_df,
            task_kind=str(task_kind),
            input_reference_value=float(input_reference_value),
            helper_feature_columns=helper_feature_columns,
        )
        if helper_vector is None:
            helper_vector = np.asarray([], dtype=float)
        copied["raw_df"] = helper_raw_df
        copied[exported_aug.PROCESS_FEATURE_VECTOR_KEY] = np.asarray(helper_vector, dtype=float).reshape(-1)
        copied[exported_aug.PROCESS_FEATURE_COLUMNS_KEY] = tuple(helper_feature_columns)
        prepared.append(copied)
    return prepared


def _resolve_shared_input_reference(cells: list[dict[str, object]]) -> float:
    values = np.asarray(
        [float(cell.get("input_reference_value", np.nan)) for cell in cells],
        dtype=float,
    )
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise RuntimeError("Could not resolve a shared input reference from the available cells.")
    return float(np.median(values))


def _compute_test_feature_floor_logs(
    exported_aug,
    test_cells: list[dict[str, object]],
    feature_columns: list[str],
    fallback_columns: list[str],
) -> np.ndarray:
    floors: list[float] = []
    for feature_name in feature_columns:
        values: list[float] = []
        for cell in test_cells:
            feature_values = exported_aug.process_feature_values(
                cell,
                [str(feature_name)],
                fallback_columns,
            )
            if feature_values.size != 1:
                continue
            value = float(feature_values[0])
            if np.isfinite(value):
                values.append(float(np.log(max(abs(value), 1e-30))))
        floors.append(float(np.min(values)) if values else np.nan)
    return np.asarray(floors, dtype=float)


def _fit_feature_log_regression_diagnostic(
    exported_aug,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    feature_columns: list[str],
    fallback_columns: list[str],
    input_reference_value: float,
) -> tuple[dict[str, object] | None, np.ndarray, float]:
    rows: list[np.ndarray] = []
    y_values: list[float] = []
    train_indices: list[int] = []
    for index, cell in enumerate(train_cells):
        feature_values = exported_aug.process_feature_values(cell, feature_columns, fallback_columns)
        target_value = float(cell.get("target_value", np.nan))
        if (
            feature_values.size != len(feature_columns)
            or not np.isfinite(feature_values).all()
            or not np.isfinite(target_value)
            or target_value <= 0.0
        ):
            continue
        rows.append(np.log(np.maximum(np.abs(np.asarray(feature_values, dtype=float)), 1e-30)))
        y_values.append(float(np.log(target_value)))
        train_indices.append(int(index))
    if len(rows) < 2:
        return None, np.asarray([], dtype=float), np.nan

    x_fit = np.asarray(rows, dtype=float)
    y_fit_log = np.asarray(y_values, dtype=float)
    x_mean = np.mean(x_fit, axis=0)
    x_std = np.std(x_fit, axis=0)
    x_std = np.where(x_std > 1e-12, x_std, 1.0)
    x_scaled = (x_fit - x_mean) / x_std
    design = np.column_stack([np.ones(len(x_scaled), dtype=float), x_scaled])
    ridge_penalty = 1e-3
    gram = design.T @ design
    gram[1:, 1:] += ridge_penalty * np.eye(design.shape[1] - 1, dtype=float)
    rhs = design.T @ y_fit_log
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    train_pred_log = design @ coefficients
    residual_sigma = float(np.std(y_fit_log - train_pred_log)) if len(y_fit_log) > 1 else 0.0

    test_rows: list[np.ndarray] = []
    finite_test_indices: list[int] = []
    for index, cell in enumerate(test_cells):
        feature_values = exported_aug.process_feature_values(cell, feature_columns, fallback_columns)
        if feature_values.size != len(feature_columns) or not np.isfinite(feature_values).all():
            continue
        test_rows.append(np.log(np.maximum(np.abs(np.asarray(feature_values, dtype=float)), 1e-30)))
        finite_test_indices.append(int(index))
    if test_rows:
        x_eval = np.asarray(test_rows, dtype=float)
        test_pred_log = coefficients[0] + ((x_eval - x_mean) / x_std) @ coefficients[1:]
        predicted_test_targets = np.exp(test_pred_log)
    else:
        x_eval = np.empty((0, len(feature_columns)), dtype=float)
        test_pred_log = np.asarray([], dtype=float)
        predicted_test_targets = np.asarray([], dtype=float)

    diagnostic = {
        "diagnostic_basis": "model_input",
        "feature_names": [str(name) for name in feature_columns],
        "train_cells": [str(train_cells[index].get("cell", "")) for index in train_indices],
        "test_cells": [str(test_cells[index].get("cell", "")) for index in finite_test_indices],
        "train_x_log_abs": x_fit.tolist(),
        "train_y_log": y_fit_log.tolist(),
        "train_pred_log": train_pred_log.tolist(),
        "test_x_log_abs": x_eval.tolist(),
        "test_pred_log": test_pred_log.tolist(),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "coefficients": coefficients.tolist(),
        "residual_sigma": float(residual_sigma),
        "ridge_penalty": float(ridge_penalty),
        "input_reference_value": float(input_reference_value),
        "generated_input_reference_value": float(input_reference_value),
    }
    return diagnostic, predicted_test_targets, float(residual_sigma)


def _build_feature_log2_helper_prediction_scales(
    exported_aug,
    module: "LFPInterpolationModule",
    train_targets: np.ndarray,
    predicted_test_targets: np.ndarray,
) -> tuple[list[float], dict[str, object]]:
    train_values = np.asarray(train_targets, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted_test_targets, dtype=float).reshape(-1)
    train_values = train_values[np.isfinite(train_values) & (train_values > 0.0)]
    predicted_values = predicted_values[np.isfinite(predicted_values) & (predicted_values > 0.0)]
    policy = str(getattr(exported_aug_cfg, "FEATURE_LOG2_MAX_SCALE_POLICY", "")).strip().lower()
    multiplier = float(getattr(exported_aug_cfg, "FEATURE_LOG2_LIFETIME_PREDICTION_MULTIPLIER", 1.0))
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("FEATURE_LOG2_LIFETIME_PREDICTION_MULTIPLIER must be positive.")

    base_scales, base_stats = exported_aug.build_feature_log2_augmentation_scales_from_feature_coverage(
        module,
        feature_coverage_scale=np.nan,
        scale_scope="global",
    )
    if train_values.size == 0 or predicted_values.size == 0:
        return base_scales, dict(base_stats)

    raw_predicted_test_max = float(np.max(predicted_values))
    upper_multiplier = (
        float(multiplier)
        if policy == "max_var_delta_q_and_lifetime_prediction_multiplier"
        else 1.0
    )
    scaled_predicted_values = predicted_values * float(upper_multiplier)
    upper_stats_scales, upper_stats = exported_aug.build_feature_log_augmentation_scales_from_predictions(
        module,
        train_values,
        scaled_predicted_values,
        prediction_sigma=None,
        prediction_sigma_space="raw",
    )
    helper_lifetime_scale = float(upper_stats.get("lifetime_coverage_scale", np.nan))
    scales, scale_stats = exported_aug.build_feature_log2_augmentation_scales_from_feature_coverage(
        module,
        feature_coverage_scale=helper_lifetime_scale,
        scale_scope="global",
    )
    scale_stats = dict(scale_stats)
    scale_stats.update(
        {
            "augmentation_method": "feature_log2",
            "feature_log2_max_scale_policy": str(policy),
            "feature_log2_helper_prediction_multiplier": float(upper_multiplier),
            "feature_log2_lifetime_prediction_multiplier": float(multiplier),
            "predicted_test_max_cycle_life": float(raw_predicted_test_max),
            "predicted_test_upper_cycle_life": float(
                upper_stats.get("target_covered_cycle_life", np.nan)
            ),
            "test_max_cycle_life": float(raw_predicted_test_max),
            "target_covered_cycle_life": float(
                upper_stats.get("target_covered_cycle_life", np.nan)
            ),
            "prediction_sigma": np.nan,
            "prediction_sigma_space": "raw",
            "prediction_sigma_multiplier": np.nan,
            "coverage_multiplier": float(upper_multiplier),
            "lifetime_coverage_scale": float(helper_lifetime_scale),
            "feature_log2_lifetime_scale": float(helper_lifetime_scale),
            "max_scale_source": (
                "linear_prediction_multiplier"
                if policy == "max_var_delta_q_and_lifetime_prediction_multiplier"
                else "linear_prediction_without_multiplier"
            ),
            "scale_reference": (
                "linear_prediction_multiplier_from_test_features"
                if policy == "max_var_delta_q_and_lifetime_prediction_multiplier"
                else "linear_prediction_from_test_features"
            ),
        }
    )
    return upper_stats_scales if not scales else scales, scale_stats


def _annotate_augmented_cells_with_helper_vector(
    exported_aug,
    module: "LFPInterpolationModule",
    augmented_cells: list[dict[str, object]],
    *,
    task_kind: str,
    helper_feature_columns: list[str],
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for cell in augmented_cells:
        copied = dict(cell)
        raw_df = copied.get("raw_df")
        input_reference_value = float(copied.get("input_reference_value", np.nan))
        if isinstance(raw_df, pd.DataFrame):
            helper_vector = _interpolate_helper_feature_vector(
                module,
                raw_df,
                task_kind=str(task_kind),
                input_reference_value=float(input_reference_value),
                helper_feature_columns=helper_feature_columns,
            )
            if helper_vector is not None:
                copied[exported_aug.FEATURE_LOG2_LIFETIME_DIAGNOSTIC_VECTOR_KEY] = np.asarray(
                    helper_vector,
                    dtype=float,
                ).reshape(-1)
        annotated.append(copied)
    return annotated


def _parse_augmented_source_name(cell_name: str) -> str:
    text = str(cell_name)
    if "_aug_" in text:
        return text.split("_aug_", 1)[0]
    return text


def _attach_augmented_target_fields(
    augmented_cells: list[dict[str, object]],
    train_cells: list[dict[str, object]],
    *,
    task_kind: str,
) -> list[dict[str, object]]:
    train_by_name = {str(cell.get("cell", "")): cell for cell in train_cells}
    normalized_task_kind = str(task_kind).strip().lower()
    enriched: list[dict[str, object]] = []
    for cell in augmented_cells:
        copied = dict(cell)
        source_name = str(copied.get("matching_source_cell", "")) or _parse_augmented_source_name(copied.get("cell", ""))
        source_cell = train_by_name.get(source_name)
        scale = float(
            copied.get(
                "augmentation_scale",
                copied.get("matching_scale", copied.get("normal_scale", np.nan)),
            )
        )
        if source_cell is not None and np.isfinite(scale):
            base_week = float(source_cell.get("target_week_value", source_cell.get("target_value", np.nan)))
            base_throughput = float(source_cell.get("target_throughput_value", np.nan))
            existing_week = float(copied.get("target_week_value", np.nan))
            existing_throughput = float(copied.get("target_throughput_value", np.nan))
            if not np.isfinite(existing_week) and np.isfinite(base_week):
                copied["target_week_value"] = float(base_week) * float(scale)
            if not np.isfinite(existing_throughput) and np.isfinite(base_throughput):
                copied["target_throughput_value"] = float(base_throughput) * float(scale)
        if "target_week_value" not in copied:
            copied["target_week_value"] = float(copied.get("target_value", np.nan))
        if not np.isfinite(float(copied.get("target_value", np.nan))):
            copied["target_value"] = float(copied.get("target_week_value", np.nan))
        enriched.append(copied)
    return enriched


def _build_exported_feature_scale_metadata(
    exported_aug,
    module: "LFPInterpolationModule",
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    feature_columns: list[str],
    task_kind: str,
    input_reference_value: float,
) -> tuple[list[str], dict[str, object], list[float]]:
    helper_feature_columns = list(LFP_AUGMENTATION_HELPER_PROCESS_COLUMNS)
    helper_train_cells = _prepare_helper_feature_cells(
        exported_aug,
        module,
        train_cells,
        feature_columns=feature_columns,
        task_kind=str(task_kind),
        helper_feature_columns=helper_feature_columns,
    )
    helper_test_cells = _prepare_helper_feature_cells(
        exported_aug,
        module,
        test_cells,
        feature_columns=feature_columns,
        task_kind=str(task_kind),
        helper_feature_columns=helper_feature_columns,
    )
    selection_columns = list(getattr(exported_aug_cfg, "FEATURE_LOG_AUGMENTATION_CANDIDATE_FEATURES", ()))
    selected_helper_features, selection_stats = exported_aug.select_fixed_single_log_r2_feature(
        helper_train_cells,
        helper_feature_columns,
        selection_columns,
        preferred_feature_name=LFP_AUGMENTATION_HELPER_PREFERRED_FEATURE,
    )
    if not selected_helper_features:
        raise RuntimeError(
            "Could not resolve the LFP augmentation helper feature for feature_log2 scaling. "
            f"Expected {LFP_AUGMENTATION_HELPER_PREFERRED_FEATURE!r}."
        )
    diagnostic, predicted_test_targets, residual_sigma = _fit_feature_log_regression_diagnostic(
        exported_aug,
        helper_train_cells,
        helper_test_cells,
        feature_columns=selected_helper_features,
        fallback_columns=helper_feature_columns,
        input_reference_value=float(input_reference_value),
    )
    train_targets = np.asarray([cell["target_value"] for cell in helper_train_cells], dtype=float)
    scales, scale_stats = _build_feature_log2_helper_prediction_scales(
        exported_aug,
        module,
        train_targets,
        predicted_test_targets,
    )
    scale_stats = dict(scale_stats)
    scale_stats.update(
        {
            "feature_log2_helper_prediction_sigma_log": float(residual_sigma),
        }
    )
    if np.isfinite(float(scale_stats.get("feature_log2_lifetime_scale", np.nan))):
        scale_stats["lifetime_coverage_scale"] = float(scale_stats["feature_log2_lifetime_scale"])
    if np.isfinite(float(scale_stats.get("target_covered_cycle_life", np.nan))):
        scale_stats["predicted_test_upper_cycle_life"] = float(scale_stats["target_covered_cycle_life"])
    if np.isfinite(float(scale_stats.get("predicted_test_max_cycle_life", np.nan))):
        scale_stats["test_max_cycle_life"] = float(scale_stats["predicted_test_max_cycle_life"])
    if diagnostic is not None:
        diagnostic = dict(diagnostic)
        diagnostic["diagnostic_basis"] = "per_cell_augmentation_reference_soh"
    metadata = {
        **selection_stats,
        **scale_stats,
        "feature_log2_scale_scope": "global_helper_prediction",
        "feature_log2_helper_feature": ",".join(selected_helper_features),
        "feature_log2_coverage_columns": ",".join(selected_helper_features),
        "feature_log2_helper_prediction_sigma_log": float(residual_sigma),
        "feature_log2_helper_reference_column": str(
            TIME_COLUMN if str(task_kind).strip().lower() == "time" else THROUGHPUT_COLUMN
        ),
    }
    if diagnostic is not None:
        metadata[exported_aug.FEATURE_LOG_MODEL_INPUT_REGRESSION_DIAGNOSTIC_KEY] = diagnostic
    return selected_helper_features, metadata, [float(scale) for scale in scales]


def _build_exported_augmented_cells(
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    feature_columns: list[str],
    input_steps: int,
    sequence_steps: int | None,
    anchor_soh: float,
    augmentation_method: str,
    task_kind: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    exported_aug = get_exported_augmentation_methods_module()
    module = LFPInterpolationModule(
        input_steps=int(input_steps),
        augmentation_reference_soh=float(anchor_soh),
        max_sequence_steps=sequence_steps,
        factor_count=int(exported_aug_cfg.FEATURE_LOG_AUGMENTATION_FACTOR_COUNT),
    )
    prepared_train = [
        exported_aug.preserve_process_features(cell, feature_columns)
        for cell in train_cells
    ]
    prepared_test = [
        exported_aug.preserve_process_features(cell, feature_columns)
        for cell in test_cells
    ]
    input_reference_value = _resolve_shared_input_reference([*prepared_train, *prepared_test])
    matching_feature_columns, feature_scale_metadata, feature_log2_scales = _build_exported_feature_scale_metadata(
        exported_aug,
        module,
        prepared_train,
        prepared_test,
        feature_columns=feature_columns,
        task_kind=task_kind,
        input_reference_value=float(input_reference_value),
    )
    method = str(augmentation_method).strip().lower()
    metadata: dict[str, object] = {
        **feature_scale_metadata,
        "augmentation_method": str(method),
        "augmentation_reference_soh": float(anchor_soh),
        "augmentation_input_reference_value": float(input_reference_value),
        "augmentation_scale_count": int(len(feature_log2_scales)),
        "augmentation_scales": [float(scale) for scale in feature_log2_scales],
        "augmentation_interpolation_mode": "exported_feature_warping",
    }
    max_scale = float(feature_scale_metadata.get("max_augmentation_scale", np.nan))
    max_scale_by_source = feature_scale_metadata.get("feature_log2_per_cell_max_scales")
    if method == "feature_log2":
        generated = exported_aug.build_normal_augmented_cells(
            module,
            prepared_train,
            feature_columns,
            task_kind=str(task_kind),
            augmentation_scales=feature_log2_scales,
            input_reference_value=float(input_reference_value),
        )
        generated = _annotate_augmented_cells_with_helper_vector(
            exported_aug,
            module,
            generated,
            task_kind=str(task_kind),
            helper_feature_columns=list(matching_feature_columns),
        )
        filter_stats: dict[str, object] = {}
        if (
            generated
            and exported_aug.FEATURE_LOG_MODEL_INPUT_REGRESSION_DIAGNOSTIC_KEY in metadata
        ):
            generated, filter_stats = exported_aug.filter_feature_log2_augmented_cells_by_lifetime_prediction_limit(
                generated,
                feature_columns,
                metadata,
            )
        metadata.update(filter_stats)
    elif method == "matching":
        generated = exported_aug.build_matching_augmented_cells(
            module,
            prepared_train,
            prepared_test,
            feature_columns,
            task_kind=str(task_kind),
            input_reference_value=float(input_reference_value),
            max_scale=max_scale if np.isfinite(max_scale) else None,
            max_scale_source=str(feature_scale_metadata.get("max_scale_source", "")),
            max_scale_by_source=(
                dict(max_scale_by_source)
                if isinstance(max_scale_by_source, dict)
                else None
            ),
            matching_feature_columns=matching_feature_columns,
            max_scale_metadata=feature_scale_metadata,
        )
        metadata.update(
            exported_aug.build_matching_augmentation_stats(
                module,
                prepared_train,
                prepared_test,
                task_kind=str(task_kind),
                input_reference_value=float(input_reference_value),
                matching_feature_columns=matching_feature_columns,
                max_scale=max_scale if np.isfinite(max_scale) else None,
                max_scale_source=str(feature_scale_metadata.get("max_scale_source", "")),
                max_scale_by_source=(
                    dict(max_scale_by_source)
                    if isinstance(max_scale_by_source, dict)
                    else None
                ),
                max_scale_metadata=feature_scale_metadata,
            )
        )
    elif method == "matching_structured":
        generated = exported_aug.build_matching_structured_augmented_cells(
            module,
            prepared_train,
            prepared_test,
            feature_columns,
            task_kind=str(task_kind),
            input_reference_value=float(input_reference_value),
            max_scale=max_scale if np.isfinite(max_scale) else None,
            max_scale_source=str(feature_scale_metadata.get("max_scale_source", "")),
            max_scale_by_source=(
                dict(max_scale_by_source)
                if isinstance(max_scale_by_source, dict)
                else None
            ),
            matching_feature_columns=matching_feature_columns,
            max_scale_metadata=feature_scale_metadata,
        )
        metadata.update(
            exported_aug.build_matching_augmentation_stats(
                module,
                prepared_train,
                prepared_test,
                task_kind=str(task_kind),
                input_reference_value=float(input_reference_value),
                matching_feature_columns=matching_feature_columns,
                max_scale=max_scale if np.isfinite(max_scale) else None,
                max_scale_source=str(feature_scale_metadata.get("max_scale_source", "")),
                max_scale_by_source=(
                    dict(max_scale_by_source)
                    if isinstance(max_scale_by_source, dict)
                    else None
                ),
                max_scale_metadata=feature_scale_metadata,
            )
        )
    else:
        raise ValueError(f"Unsupported exported augmentation method {augmentation_method!r}.")
    augmented = _attach_augmented_target_fields(
        generated,
        prepared_train,
        task_kind=str(task_kind),
    )
    metadata["train_aug_count"] = int(len(augmented))
    return augmented, metadata


def _train_single_target_ensemble(
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    training_config: TrainingConfig,
    target_key: str,
) -> dict[str, object]:
    if not train_cells:
        raise RuntimeError("No training cells are available.")
    if not test_cells:
        raise RuntimeError("No test cells are available.")

    input_steps = int(training_config.input_steps)
    feature_count = int(np.asarray(train_cells[0]["sequence"], dtype=float).shape[-1])
    X_train_all = np.asarray([cell["sequence"] for cell in train_cells], dtype=float)
    y_train_all = np.asarray([cell[target_key] for cell in train_cells], dtype=float)
    X_test = np.asarray([cell["sequence"] for cell in test_cells], dtype=float)
    y_test = np.asarray([cell[target_key] for cell in test_cells], dtype=float)
    test_names = [str(cell["cell"]) for cell in test_cells]

    target_divisor = compute_symmetric_log_target_divisor(y_train_all)
    y_train_log = np.log(y_train_all / float(target_divisor))
    y_test_log = np.log(y_test / float(target_divisor))

    use_validation = len(X_train_all) >= 8
    if use_validation:
        X_train, y_train, X_val, y_val = create_balanced_val_split(
            X_train_all,
            y_train_log,
            val_fraction=float(training_config.validation_split),
            bins=3,
        )
    else:
        X_train, y_train = X_train_all, y_train_log
        X_val = y_val = None

    X_train_t = torch.from_numpy(X_train).float().to(DEVICE)
    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=float)).float().view(-1, 1).to(DEVICE)
    X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=int(training_config.batch_size),
        shuffle=True,
    )
    if use_validation and X_val is not None and y_val is not None:
        X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
        y_val_t = torch.from_numpy(np.asarray(y_val, dtype=float)).float().view(-1, 1).to(DEVICE)
    else:
        X_val_t = y_val_t = None

    seeds = build_model_training_seed_list(
        seed_count=int(training_config.seed_count),
        seed_source=int(training_config.seed_source),
        seed_upper_bound=100000,
    )
    loss_fn = build_regression_loss(str(training_config.loss_function))
    y_pred_log_by_seed: list[np.ndarray] = []
    seed_rows: list[dict[str, object]] = []

    for seed in seeds:
        set_seed(int(seed))
        model = TinyTemporalCNN(
            input_dims=int(feature_count),
            timesteps=int(input_steps),
            dropout_rate=float(training_config.dropout_rate),
            conv1_channels=int(training_config.conv1_channels),
            conv_channels=int(training_config.conv_channels),
            fc_hidden_dim=int(training_config.fc_hidden_dim),
            fc_hidden_dim2=training_config.fc_hidden_dim2,
            conv1_kernel_size=int(training_config.conv1_kernel_size),
            conv2_kernel_size=int(training_config.conv2_kernel_size),
        ).to(DEVICE)
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(training_config.learning_rate),
            weight_decay=float(training_config.weight_decay),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(training_config.scheduler_patience),
            factor=0.5,
        )
        best_state = deepcopy(model.state_dict())
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for _epoch in range(int(training_config.epochs)):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = loss_fn(predictions, y_batch)
                loss.backward()
                optimizer.step()

            if use_validation and X_val_t is not None and y_val_t is not None:
                model.eval()
                with torch.no_grad():
                    val_predictions = model(X_val_t)
                    val_loss = float(loss_fn(val_predictions, y_val_t).item())
                scheduler.step(val_loss)
                if val_loss + 1e-12 < best_val_loss:
                    best_val_loss = float(val_loss)
                    best_state = deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= int(training_config.early_stopping_patience):
                        break
            else:
                best_state = deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            y_pred_log = model(X_test_t).cpu().numpy().reshape(-1)
        y_pred_log_by_seed.append(y_pred_log)
        seed_rows.append(
            {
                "seed": int(seed),
                "validation_loss": float(best_val_loss) if np.isfinite(best_val_loss) else np.nan,
            }
        )

    y_pred_log_mean = np.mean(np.asarray(y_pred_log_by_seed, dtype=float), axis=0)
    y_pred_steps = np.exp(y_pred_log_mean) * float(target_divisor)
    rmse_steps = float(np.sqrt(np.mean((y_test - y_pred_steps) ** 2)))
    mape_steps = float(np.mean(np.abs((y_test - y_pred_steps) / y_test)) * 100.0)
    return {
        "test_names": test_names,
        "y_test": y_test,
        "y_pred": y_pred_steps,
        "y_test_log": y_test_log,
        "y_pred_log": y_pred_log_mean,
        "y_pred_by_seed": np.exp(np.asarray(y_pred_log_by_seed, dtype=float)) * float(target_divisor),
        "rmse": float(rmse_steps),
        "mape": float(mape_steps),
        "y_test_log": y_test_log,
        "seed_summary_df": pd.DataFrame(seed_rows),
        "seed_count": int(len(seeds)),
        "train_count": int(len(train_cells)),
        "test_count": int(len(test_cells)),
        "target_divisor": float(target_divisor),
    }


def _build_exported_tcnn_hyperparameters(training_config: TrainingConfig) -> dict[str, object]:
    return {
        "input_steps": int(training_config.input_steps),
        "epochs": int(training_config.epochs),
        "batch_size": int(training_config.batch_size),
        "learning_rate": float(training_config.learning_rate),
        "weight_decay": float(training_config.weight_decay),
        "dropout_rate": float(training_config.dropout_rate),
        "validation_split": float(training_config.validation_split),
        "early_stopping_patience": int(training_config.early_stopping_patience),
        "scheduler_patience": int(training_config.scheduler_patience),
        "conv1_channels": int(training_config.conv1_channels),
        "conv_channels": int(training_config.conv_channels),
        "fc_hidden_dim": int(training_config.fc_hidden_dim),
        "fc_hidden_dim2": training_config.fc_hidden_dim2,
        "conv1_kernel_size": int(training_config.conv1_kernel_size),
        "conv2_kernel_size": int(training_config.conv2_kernel_size),
        "loss_function": str(training_config.loss_function),
    }


def _build_exported_mlp_hyperparameters(training_config: TrainingConfig) -> dict[str, object]:
    hidden_layers = [int(training_config.fc_hidden_dim)]
    if training_config.fc_hidden_dim2 is not None:
        hidden_layers.append(int(training_config.fc_hidden_dim2))
    return {
        "hidden_layers": hidden_layers,
        "dropout_rate": float(training_config.dropout_rate),
        "learning_rate": float(training_config.learning_rate),
        "weight_decay": float(training_config.weight_decay),
        "batch_size": int(training_config.batch_size),
        "epochs": int(training_config.epochs),
        "validation_split": float(training_config.validation_split),
        "early_stopping_patience": int(training_config.early_stopping_patience),
        "scheduler_patience": int(training_config.scheduler_patience),
        "loss_function": str(training_config.loss_function),
    }


def _train_single_target_exported_ensemble(
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    training_config: TrainingConfig,
    target_key: str,
    prediction_model: str,
) -> dict[str, object]:
    if not train_cells:
        raise RuntimeError("No training cells are available.")
    if not test_cells:
        raise RuntimeError("No test cells are available.")

    exported_models = get_exported_temporal_models_module()
    normalized_model = normalize_prediction_model(prediction_model)
    use_mlp = normalized_model == PREDICTION_MODEL_EXPORTED_EOL_MLP
    y_train_all = np.asarray([cell[target_key] for cell in train_cells], dtype=float)
    y_test = np.asarray([cell[target_key] for cell in test_cells], dtype=float)
    test_names = [str(cell["cell"]) for cell in test_cells]
    target_divisor = compute_symmetric_log_target_divisor(y_train_all)
    y_test_log = np.log(y_test / float(target_divisor))
    seeds = build_model_training_seed_list(
        seed_count=int(training_config.seed_count),
        seed_source=int(training_config.seed_source),
        seed_upper_bound=100000,
    )
    y_pred_by_seed: list[np.ndarray] = []
    y_pred_log_by_seed: list[np.ndarray] = []
    seed_rows: list[dict[str, object]] = []

    if use_mlp:
        X_train_all = np.asarray([cell["x"] for cell in train_cells], dtype=float)
        X_test = np.asarray([cell["x"] for cell in test_cells], dtype=float)
        hyperparameters = _build_exported_mlp_hyperparameters(training_config)
    else:
        X_train_all = np.asarray([cell["sequence"] for cell in train_cells], dtype=float)
        X_test = np.asarray([cell["sequence"] for cell in test_cells], dtype=float)
        hyperparameters = _build_exported_tcnn_hyperparameters(training_config)

    for seed in seeds:
        if use_mlp:
            model, scaler, target_transformer, _train_losses, validation_losses = exported_models.train_augmented_eol_mlp(
                X_train_all,
                y_train_all,
                seed=int(seed),
                device=DEVICE,
                hyperparameters=hyperparameters,
            )
            X_test_scaled = scaler.transform(X_test)
            X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=DEVICE)
        else:
            model, scaler, target_transformer, _train_losses, validation_losses = exported_models.train_augmented_eol_tcnn(
                X_train_all,
                y_train_all,
                seed=int(seed),
                device=DEVICE,
                hyperparameters=hyperparameters,
                real_train_count=int(len(train_cells)),
            )
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=DEVICE)
        model.eval()
        with torch.no_grad():
            y_pred_transformed = model(X_test_t).cpu().numpy().reshape(-1)
        y_pred = target_transformer.inverse_transform(y_pred_transformed)
        y_pred = np.maximum(np.asarray(y_pred, dtype=float).reshape(-1), 1e-30)
        y_pred_by_seed.append(y_pred)
        y_pred_log_by_seed.append(np.log(y_pred / float(target_divisor)))
        best_validation_loss = (
            float(np.min(np.asarray(validation_losses, dtype=float)))
            if validation_losses
            else np.nan
        )
        seed_rows.append(
            {
                "seed": int(seed),
                "validation_loss": best_validation_loss,
                "target_transform_divisor": float(getattr(target_transformer, "target_divisor_", np.nan)),
            }
        )

    y_pred_matrix = np.asarray(y_pred_by_seed, dtype=float)
    y_pred = np.mean(y_pred_matrix, axis=0)
    y_pred_log_matrix = np.asarray(y_pred_log_by_seed, dtype=float)
    y_pred_log_mean = np.mean(y_pred_log_matrix, axis=0)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0)
    return {
        "test_names": test_names,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_test_log": y_test_log,
        "y_pred_log": y_pred_log_mean,
        "y_pred_by_seed": y_pred_matrix,
        "rmse": rmse,
        "mape": mape,
        "seed_summary_df": pd.DataFrame(seed_rows),
        "seed_count": int(len(seeds)),
        "train_count": int(len(train_cells)),
        "test_count": int(len(test_cells)),
        "target_divisor": float(target_divisor),
    }


def train_tcnn_ensemble(
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    training_config: TrainingConfig,
) -> dict[str, object]:
    week_result = _train_single_target_ensemble(
        train_cells,
        test_cells,
        training_config=training_config,
        target_key="target_value",
    )
    throughput_result = _train_single_target_ensemble(
        train_cells,
        test_cells,
        training_config=training_config,
        target_key="target_throughput_value",
    )
    prediction_df = build_prediction_dataframe(
        week_result["test_names"],
        week_result["y_test"],
        week_result["y_pred"],
        y_true_log=week_result["y_test_log"],
        y_pred_log=week_result["y_pred_log"],
        y_true_throughput=throughput_result["y_test"],
        y_pred_throughput=throughput_result["y_pred"],
        y_pred_log_throughput=throughput_result["y_pred_log"],
    )
    return {
        "test_names": week_result["test_names"],
        "y_test_weeks": week_result["y_test"],
        "y_pred_weeks": week_result["y_pred"],
        "y_pred_weeks_by_seed": week_result["y_pred_by_seed"],
        "rmse_weeks": float(week_result["rmse"]),
        "mape_weeks": float(week_result["mape"]),
        "week_target_divisor": float(week_result["target_divisor"]),
        "y_test_throughput": throughput_result["y_test"],
        "y_pred_throughput": throughput_result["y_pred"],
        "y_pred_throughput_by_seed": throughput_result["y_pred_by_seed"],
        "rmse_throughput": float(throughput_result["rmse"]),
        "mape_throughput": float(throughput_result["mape"]),
        "throughput_target_divisor": float(throughput_result["target_divisor"]),
        "y_test_steps": week_result["y_test"],
        "y_pred_steps": week_result["y_pred"],
        "y_test_log": week_result["y_test_log"],
        "y_pred_log": week_result["y_pred_log"],
        "y_pred_steps_by_seed": week_result["y_pred_by_seed"],
        "rmse_steps": float(week_result["rmse"]),
        "mape_steps": float(week_result["mape"]),
        "prediction_df": prediction_df,
        "seed_summary_df": week_result["seed_summary_df"],
        "seed_summary_df_throughput": throughput_result["seed_summary_df"],
        "seed_count": int(week_result["seed_count"]),
        "train_count": int(week_result["train_count"]),
        "test_count": int(week_result["test_count"]),
    }


def train_prediction_ensemble(
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    training_config: TrainingConfig,
) -> dict[str, object]:
    prediction_model = normalize_prediction_model(training_config.prediction_model)
    if prediction_model == PREDICTION_MODEL_LEGACY_TCNN:
        return train_tcnn_ensemble(
            train_cells,
            test_cells,
            training_config=training_config,
        )
    week_result = _train_single_target_exported_ensemble(
        train_cells,
        test_cells,
        training_config=training_config,
        target_key="target_value",
        prediction_model=prediction_model,
    )
    throughput_result = _train_single_target_exported_ensemble(
        train_cells,
        test_cells,
        training_config=training_config,
        target_key="target_throughput_value",
        prediction_model=prediction_model,
    )
    prediction_df = build_prediction_dataframe(
        week_result["test_names"],
        week_result["y_test"],
        week_result["y_pred"],
        y_true_log=week_result["y_test_log"],
        y_pred_log=week_result["y_pred_log"],
        y_true_throughput=throughput_result["y_test"],
        y_pred_throughput=throughput_result["y_pred"],
        y_pred_log_throughput=throughput_result["y_pred_log"],
    )
    return {
        "test_names": week_result["test_names"],
        "y_test_weeks": week_result["y_test"],
        "y_pred_weeks": week_result["y_pred"],
        "y_pred_weeks_by_seed": week_result["y_pred_by_seed"],
        "rmse_weeks": float(week_result["rmse"]),
        "mape_weeks": float(week_result["mape"]),
        "week_target_divisor": float(week_result["target_divisor"]),
        "y_test_throughput": throughput_result["y_test"],
        "y_pred_throughput": throughput_result["y_pred"],
        "y_pred_throughput_by_seed": throughput_result["y_pred_by_seed"],
        "rmse_throughput": float(throughput_result["rmse"]),
        "mape_throughput": float(throughput_result["mape"]),
        "throughput_target_divisor": float(throughput_result["target_divisor"]),
        "y_test_steps": week_result["y_test"],
        "y_pred_steps": week_result["y_pred"],
        "y_test_log": week_result["y_test_log"],
        "y_pred_log": week_result["y_pred_log"],
        "y_pred_steps_by_seed": week_result["y_pred_by_seed"],
        "rmse_steps": float(week_result["rmse"]),
        "mape_steps": float(week_result["mape"]),
        "prediction_df": prediction_df,
        "seed_summary_df": week_result["seed_summary_df"],
        "seed_summary_df_throughput": throughput_result["seed_summary_df"],
        "seed_count": int(week_result["seed_count"]),
        "train_count": int(week_result["train_count"]),
        "test_count": int(week_result["test_count"]),
    }


def plot_prediction_results(
    result: dict[str, object],
    *,
    title_prefix: str,
    show: bool = True,
) -> None:
    test_names = list(result["test_names"])
    y_test_steps = np.asarray(result.get("y_test_weeks", result["y_test_steps"]), dtype=float)
    y_pred_steps = np.asarray(result.get("y_pred_weeks", result["y_pred_steps"]), dtype=float)
    preds_steps_by_seed = np.asarray(result.get("y_pred_weeks_by_seed", result["y_pred_steps_by_seed"]), dtype=float)
    rmse_steps = float(result.get("rmse_weeks", result["rmse_steps"]))
    mape_steps = float(result.get("mape_weeks", result["mape_steps"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_positions = np.arange(len(test_names))
    width = 0.35

    axes[0].bar(x_positions - width / 2, y_test_steps, width, label="True", color="steelblue", alpha=0.85)
    axes[0].bar(x_positions + width / 2, y_pred_steps, width, label="Predicted", color="coral", alpha=0.85)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(test_names, rotation=20, ha="right")
    axes[0].set_ylabel("Weeks at 95% SOH")
    axes[0].set_title(f"{title_prefix} (RMSE={rmse_steps:.1f}, MAPE={mape_steps:.1f}%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for column_index, cell_name in enumerate(test_names):
        axes[1].scatter(
            np.full(preds_steps_by_seed.shape[0], column_index),
            preds_steps_by_seed[:, column_index],
            color="coral",
            alpha=0.4,
            s=20,
        )
        axes[1].scatter(column_index, y_test_steps[column_index], marker="*", s=200, color="steelblue", zorder=5)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(test_names, rotation=20, ha="right")
    axes[1].set_ylabel("Weeks at 95% SOH")
    axes[1].set_title("Ensemble spread")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def _extract_scale_values(cells: list[dict[str, object]]) -> np.ndarray:
    scale_keys = (
        "augmentation_scale",
        "matching_scale",
        "normal_scale",
        "cal_warp_factor",
        "hybrid_scale",
        "hybrid_base_scale",
    )
    values: list[float] = []
    for cell in cells:
        for key in scale_keys:
            if key in cell and np.isfinite(float(cell[key])):
                values.append(float(cell[key]))
                break
    return np.asarray(values, dtype=float)


def _sample_cells_for_plot(cells: list[dict[str, object]], max_count: int) -> list[dict[str, object]]:
    if max_count <= 0 or len(cells) <= max_count:
        return list(cells)
    sample_indices = np.linspace(0, len(cells) - 1, int(max_count), dtype=int)
    unique_indices: list[int] = []
    seen: set[int] = set()
    for index in sample_indices:
        value = int(index)
        if value in seen:
            continue
        seen.add(value)
        unique_indices.append(value)
    return [cells[index] for index in unique_indices]


def plot_training_trajectory_overlay(
    *,
    train_real_cells: list[dict[str, object]],
    augmented_cells: list[dict[str, object]] | None = None,
    test_cells: list[dict[str, object]],
    metadata: dict[str, object],
    title_prefix: str,
    show: bool = True,
) -> None:
    anchor_soh = float(metadata.get("selected_anchor_soh", np.nan))
    target_soh = float(metadata.get("target_soh", np.nan))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_full_labeled = False
    train_input_labeled = False
    train_eol_labeled = False
    aug_labeled = False
    ref_labeled = False

    for cell in train_real_cells:
        weeks = np.asarray(cell["time_curve"], dtype=float)
        throughput = np.asarray(cell["throughput_curve"], dtype=float)
        soh = np.asarray(cell["soh_curve"], dtype=float)
        valid = np.isfinite(weeks) & np.isfinite(throughput) & np.isfinite(soh)
        if not np.any(valid):
            continue
        weeks_valid = weeks[valid]
        throughput_valid = throughput[valid]
        soh_valid = soh[valid]

        full_label = "Train trajectory" if not train_full_labeled else None
        axes[0].plot(weeks_valid, soh_valid, color="0.7", alpha=0.45, linewidth=0.9, label=full_label)
        axes[1].plot(throughput_valid, soh_valid, color="0.7", alpha=0.45, linewidth=0.9, label=full_label)
        train_full_labeled = True

        input_end_index = None
        anchor_week = float(metadata.get("selected_anchor_week", np.nan))
        anchor_throughput = float(metadata.get("selected_anchor_throughput", np.nan))
        if np.isfinite(anchor_week):
            input_end_index = find_first_index_at_or_above_week(cell["raw_df"], anchor_week)
        elif np.isfinite(anchor_throughput):
            input_end_index = find_first_index_at_or_above_throughput(cell["raw_df"], anchor_throughput)
        elif np.isfinite(anchor_soh):
            input_end_index = find_first_index_at_or_below_soh(cell["raw_df"], anchor_soh)
        if input_end_index is None:
            input_end_index = len(soh_valid) - 1
        input_end_index = max(0, min(int(input_end_index), len(soh_valid) - 1))

        input_label = "Model input window" if not train_input_labeled else None
        axes[0].plot(
            weeks_valid[: input_end_index + 1],
            soh_valid[: input_end_index + 1],
            color="red",
            alpha=0.9,
            linewidth=1.5,
            label=input_label,
        )
        axes[1].plot(
            throughput_valid[: input_end_index + 1],
            soh_valid[: input_end_index + 1],
            color="red",
            alpha=0.9,
            linewidth=1.5,
            label=input_label,
        )
        train_input_labeled = True

        target_week = float(cell.get("target_week_value", cell.get("target_value", np.nan)))
        target_throughput = float(cell.get("target_throughput_value", np.nan))
        if np.isfinite(target_week) and np.isfinite(target_soh):
            eol_label = "95% SOH target" if not train_eol_labeled else None
            axes[0].scatter(target_week, target_soh, color="red", s=24, alpha=0.95, label=eol_label, zorder=5)
            if np.isfinite(target_throughput):
                axes[1].scatter(target_throughput, target_soh, color="red", s=24, alpha=0.95, label=eol_label, zorder=5)
            train_eol_labeled = True

    for cell in _sample_cells_for_plot(list(augmented_cells or []), max_count=60):
        weeks = np.asarray(cell["time_curve"], dtype=float)
        throughput = np.asarray(cell["throughput_curve"], dtype=float)
        soh = np.asarray(cell["soh_curve"], dtype=float)
        valid = np.isfinite(weeks) & np.isfinite(throughput) & np.isfinite(soh)
        if not np.any(valid):
            continue
        aug_label = "Augmented trajectory" if not aug_labeled else None
        axes[0].plot(weeks[valid], soh[valid], color="teal", alpha=0.18, linewidth=0.9, label=aug_label)
        axes[1].plot(throughput[valid], soh[valid], color="teal", alpha=0.18, linewidth=0.9, label=aug_label)
        aug_labeled = True

    for cell in test_cells:
        weeks = np.asarray(cell["time_curve"], dtype=float)
        throughput = np.asarray(cell["throughput_curve"], dtype=float)
        soh = np.asarray(cell["soh_curve"], dtype=float)
        valid = np.isfinite(weeks) & np.isfinite(throughput) & np.isfinite(soh)
        if not np.any(valid):
            continue
        ref_label = "Reference cell" if not ref_labeled else None
        axes[0].plot(weeks[valid], soh[valid], color="orange", alpha=0.95, linewidth=2.0, label=ref_label)
        axes[1].plot(throughput[valid], soh[valid], color="orange", alpha=0.95, linewidth=2.0, label=ref_label)
        ref_labeled = True

    if np.isfinite(anchor_soh):
        axes[0].axhline(anchor_soh, color="black", linestyle="--", linewidth=1.0)
        axes[1].axhline(anchor_soh, color="black", linestyle="--", linewidth=1.0)
    if np.isfinite(target_soh):
        axes[0].axhline(target_soh, color="black", linestyle=":", linewidth=1.0)
        axes[1].axhline(target_soh, color="black", linestyle=":", linewidth=1.0)
    anchor_throughput = float(metadata.get("selected_anchor_throughput", np.nan))
    if np.isfinite(anchor_throughput):
        axes[1].axvline(anchor_throughput, color="black", linestyle="--", linewidth=1.0)

    axes[0].set_xlabel("Weeks")
    axes[0].set_ylabel("SOH")
    axes[0].set_ylim(bottom=0.7)
    axes[0].set_title("Raw trajectories vs weeks")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Throughput")
    axes[1].set_ylabel("SOH")
    axes[1].set_ylim(bottom=0.7)
    axes[1].set_title("Raw trajectories vs throughput")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"{title_prefix}: training trajectories", fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_feature_log2_helper_diagnostics(
    *,
    train_real_cells: list[dict[str, object]],
    augmented_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    metadata: dict[str, object],
    title_prefix: str,
    show: bool = True,
    output_path: Path | None = None,
) -> None:
    exported_aug = get_exported_augmentation_methods_module()
    diagnostic = metadata.get(exported_aug.FEATURE_LOG_MODEL_INPUT_REGRESSION_DIAGNOSTIC_KEY)
    helper_feature_text = str(metadata.get("feature_log2_helper_feature", "")).strip()
    if not isinstance(diagnostic, dict) or not helper_feature_text:
        return
    helper_features = [part.strip() for part in helper_feature_text.split(",") if part.strip()]
    if len(helper_features) != 1:
        return

    helper_feature = str(helper_features[0])
    helper_reference_column = str(metadata.get("feature_log2_helper_reference_column", "reference"))
    multiplier = float(metadata.get("feature_log2_lifetime_prediction_multiplier", np.nan))
    train_lookup = {str(cell["cell"]): cell for cell in train_real_cells}
    test_lookup = {str(cell["cell"]): cell for cell in test_cells}
    train_names = [str(name) for name in diagnostic.get("train_cells", [])]
    test_names = [str(name) for name in diagnostic.get("test_cells", [])]

    train_x_log = np.asarray(diagnostic.get("train_x_log_abs", []), dtype=float)
    train_y_log = np.asarray(diagnostic.get("train_y_log", []), dtype=float).reshape(-1)
    test_x_log = np.asarray(diagnostic.get("test_x_log_abs", []), dtype=float)
    test_pred_log = np.asarray(diagnostic.get("test_pred_log", []), dtype=float).reshape(-1)
    if train_x_log.ndim != 2 or train_x_log.shape[1] != 1 or test_x_log.ndim != 2 or test_x_log.shape[1] != 1:
        return

    valid_test_names = [
        name for name in test_names
        if name in test_lookup and np.isfinite(float(test_lookup[name].get("target_value", np.nan)))
        and float(test_lookup[name].get("target_value", np.nan)) > 0.0
    ]
    test_x_log = test_x_log[: len(valid_test_names), :]
    test_pred_log = test_pred_log[: len(valid_test_names)]
    test_actual_log = np.asarray(
        [np.log(float(test_lookup[name]["target_value"])) for name in valid_test_names],
        dtype=float,
    )
    upper_test_log = test_pred_log + (
        np.log(float(multiplier))
        if np.isfinite(multiplier) and float(multiplier) > 0.0
        else 0.0
    )

    augmented_names: list[str] = []
    augmented_x_log_values: list[float] = []
    augmented_y_log_values: list[float] = []
    for cell in augmented_cells:
        helper_vector = np.asarray(
            cell.get(exported_aug.FEATURE_LOG2_LIFETIME_DIAGNOSTIC_VECTOR_KEY, []),
            dtype=float,
        ).reshape(-1)
        target_value = float(cell.get("target_value", np.nan))
        if helper_vector.size < 1 or not np.isfinite(helper_vector[0]) or not np.isfinite(target_value) or target_value <= 0.0:
            continue
        augmented_names.append(str(cell.get("cell", "")))
        augmented_x_log_values.append(float(np.log(max(abs(float(helper_vector[0])), 1e-30))))
        augmented_y_log_values.append(float(np.log(target_value)))
    augmented_x_log = np.asarray(augmented_x_log_values, dtype=float)
    augmented_y_log = np.asarray(augmented_y_log_values, dtype=float)

    x_train = train_x_log[:, 0]
    x_test = test_x_log[:, 0]
    x_parts = [x_train]
    if x_test.size:
        x_parts.append(x_test)
    if augmented_x_log.size:
        x_parts.append(augmented_x_log)
    x_all = np.concatenate(x_parts)
    x_grid = np.linspace(float(np.min(x_all)) - 0.1, float(np.max(x_all)) + 0.1, 200, dtype=float)
    x_mean = float(np.asarray(diagnostic.get("x_mean", []), dtype=float).reshape(-1)[0])
    x_std = float(np.asarray(diagnostic.get("x_std", []), dtype=float).reshape(-1)[0])
    coefficients = np.asarray(diagnostic.get("coefficients", []), dtype=float).reshape(-1)
    if coefficients.size != 2:
        return
    y_grid = coefficients[0] + ((x_grid - x_mean) / max(x_std, 1e-12)) * coefficients[1]
    y_grid_upper = y_grid + (
        np.log(float(multiplier))
        if np.isfinite(multiplier) and float(multiplier) > 0.0
        else 0.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(x_train, train_y_log, color="steelblue", alpha=0.8, s=32, label="Train actual")
    if augmented_x_log.size:
        axes[0].scatter(
            augmented_x_log,
            augmented_y_log,
            color="seagreen",
            alpha=0.45,
            s=18,
            label="Augmented",
        )
    axes[0].scatter(x_test, test_actual_log, color="coral", alpha=0.95, s=58, marker="D", label="Test actual")
    axes[0].scatter(x_test, test_pred_log, color="teal", alpha=0.95, s=48, marker="s", label="Test predicted")
    axes[0].plot(x_grid, y_grid, color="black", linewidth=1.6, label="Helper fit")
    if np.isfinite(multiplier) and float(multiplier) > 0.0:
        axes[0].plot(
            x_grid,
            y_grid_upper,
            color="purple",
            linewidth=1.4,
            linestyle="--",
            label=f"Predicted x {multiplier:g}",
        )
    for x_value, y_value, name in zip(x_test, test_actual_log, valid_test_names):
        axes[0].annotate(name, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8, color="coral")
    axes[0].set_xlabel(f"log(|{helper_feature}|) at {helper_reference_column} cutoff")
    axes[0].set_ylabel("log(weeks to target SOH)")
    axes[0].set_title("Log-domain helper model")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    raw_x_train = np.exp(x_train)
    raw_y_train = np.exp(train_y_log)
    raw_x_test = np.exp(x_test)
    raw_y_test = np.exp(test_actual_log)
    raw_pred_test = np.exp(test_pred_log)
    raw_upper_test = np.exp(upper_test_log)
    raw_x_grid = np.exp(x_grid)
    raw_y_grid = np.exp(y_grid)
    raw_y_grid_upper = np.exp(y_grid_upper)
    axes[1].scatter(raw_x_train, raw_y_train, color="steelblue", alpha=0.8, s=32, label="Train actual")
    if augmented_x_log.size:
        axes[1].scatter(
            np.exp(augmented_x_log),
            np.exp(augmented_y_log),
            color="seagreen",
            alpha=0.45,
            s=18,
            label="Augmented",
        )
    axes[1].scatter(raw_x_test, raw_y_test, color="coral", alpha=0.95, s=58, marker="D", label="Test actual")
    axes[1].scatter(raw_x_test, raw_pred_test, color="teal", alpha=0.95, s=48, marker="s", label="Test predicted")
    axes[1].plot(raw_x_grid, raw_y_grid, color="black", linewidth=1.6, label="Helper fit")
    if np.isfinite(multiplier) and float(multiplier) > 0.0:
        axes[1].plot(
            raw_x_grid,
            raw_y_grid_upper,
            color="purple",
            linewidth=1.4,
            linestyle="--",
            label=f"Predicted x {multiplier:g}",
        )
        axes[1].scatter(raw_x_test, raw_upper_test, color="purple", alpha=0.7, s=40, marker="^", label="Per-test upper")
    for x_value, y_value, name in zip(raw_x_test, raw_y_test, valid_test_names):
        axes[1].annotate(name, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8, color="coral")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(f"|{helper_feature}| at {helper_reference_column} cutoff")
    axes[1].set_ylabel("Weeks to target SOH")
    axes[1].set_title("Raw-domain view")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    max_scale = float(metadata.get("max_augmentation_scale", np.nan))
    scale_count = len(list(metadata.get("augmentation_scales", []) or []))
    fig.suptitle(
        f"{title_prefix}: feature_log2 helper model\n"
        f"feature={helper_feature}, multiplier={multiplier:g}, max_scale={max_scale:.3f}, scale_count={scale_count}",
        fontsize=12,
    )
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_feature_log2_helper_feature_trajectories(
    *,
    train_real_cells: list[dict[str, object]],
    augmented_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    metadata: dict[str, object],
    title_prefix: str,
    show: bool = True,
    output_path: Path | None = None,
) -> None:
    helper_feature_text = str(metadata.get("feature_log2_helper_feature", "")).strip()
    if not helper_feature_text:
        return
    helper_features = [part.strip() for part in helper_feature_text.split(",") if part.strip()]
    if not helper_features:
        return

    train_sample = _sample_cells_for_plot(train_real_cells, max_count=12)
    augmented_sample = _sample_cells_for_plot(augmented_cells, max_count=48)
    test_sample = _sample_cells_for_plot(test_cells, max_count=12)
    row_count = len(helper_features)
    fig, axes = plt.subplots(row_count, 2, figsize=(16, max(5, 4 * row_count)), squeeze=False)

    anchor_week = float(metadata.get("selected_anchor_week", np.nan))
    anchor_throughput = float(metadata.get("selected_anchor_throughput", np.nan))

    for row_index, helper_feature in enumerate(helper_features):
        week_axis = axes[row_index, 0]
        throughput_axis = axes[row_index, 1]
        train_label_used = False
        aug_label_used = False
        test_label_used = False
        train_point_label_used = False
        aug_point_label_used = False
        test_point_label_used = False

        for cells, color, alpha, linewidth, label_name, point_alpha in (
            (train_sample, "steelblue", 0.30, 1.0, "Train real", 0.75),
            (augmented_sample, "seagreen", 0.12, 0.9, "Train augmented", 0.45),
            (test_sample, "coral", 0.95, 1.7, "Test", 0.95),
        ):
            for cell in cells:
                raw_df = cell.get("raw_df")
                if not isinstance(raw_df, pd.DataFrame):
                    continue
                helper_df = _build_lfp_helper_raw_df(raw_df)
                if helper_feature not in helper_df.columns:
                    continue
                weeks = pd.to_numeric(helper_df[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
                throughput = pd.to_numeric(helper_df[THROUGHPUT_COLUMN], errors="coerce").to_numpy(dtype=float)
                values = pd.to_numeric(helper_df[helper_feature], errors="coerce").to_numpy(dtype=float)

                week_valid = np.isfinite(weeks) & np.isfinite(values)
                throughput_valid = np.isfinite(throughput) & np.isfinite(values)
                if np.count_nonzero(week_valid) >= 2:
                    label = label_name if (
                        label_name == "Train real" and not train_label_used
                        or label_name == "Train augmented" and not aug_label_used
                        or label_name == "Test" and not test_label_used
                    ) else None
                    week_axis.plot(
                        weeks[week_valid],
                        values[week_valid],
                        color=color,
                        alpha=alpha,
                        linewidth=linewidth,
                        label=label,
                    )
                    if label_name == "Train real":
                        train_label_used = True
                    elif label_name == "Train augmented":
                        aug_label_used = True
                    else:
                        test_label_used = True
                if np.count_nonzero(throughput_valid) >= 2:
                    label = label_name if (
                        label_name == "Train real" and not train_point_label_used
                        or label_name == "Train augmented" and not aug_point_label_used
                        or label_name == "Test" and not test_point_label_used
                    ) else None
                    throughput_axis.plot(
                        throughput[throughput_valid],
                        values[throughput_valid],
                        color=color,
                        alpha=alpha,
                        linewidth=linewidth,
                        label=label,
                    )
                    if label_name == "Train real":
                        train_point_label_used = True
                    elif label_name == "Train augmented":
                        aug_point_label_used = True
                    else:
                        test_point_label_used = True

                input_week_grid = np.asarray(cell.get("input_week_grid", []), dtype=float)
                input_week = (
                    float(input_week_grid[np.flatnonzero(np.isfinite(input_week_grid))[-1]])
                    if input_week_grid.size and np.any(np.isfinite(input_week_grid))
                    else np.nan
                )
                input_throughput = float(cell.get("input_throughput_value", np.nan))
                if np.isfinite(input_week):
                    helper_at_input_week = interpolate_feature_vector_from_df(
                        helper_df,
                        [helper_feature],
                        reference_column=TIME_COLUMN,
                        target_reference=float(input_week),
                    )
                    if helper_at_input_week is not None and np.isfinite(float(helper_at_input_week[0])):
                        week_axis.scatter(
                            [float(input_week)],
                            [float(helper_at_input_week[0])],
                            color=color,
                            alpha=point_alpha,
                            s=16 if label_name != "Test" else 28,
                            zorder=5,
                        )
                if np.isfinite(input_throughput) and input_throughput > 0.0:
                    helper_at_input_throughput = interpolate_feature_vector_from_df(
                        helper_df,
                        [helper_feature],
                        reference_column=THROUGHPUT_COLUMN,
                        target_reference=float(input_throughput),
                    )
                    if helper_at_input_throughput is not None and np.isfinite(float(helper_at_input_throughput[0])):
                        throughput_axis.scatter(
                            [float(input_throughput)],
                            [float(helper_at_input_throughput[0])],
                            color=color,
                            alpha=point_alpha,
                            s=16 if label_name != "Test" else 28,
                            zorder=5,
                        )

        if np.isfinite(anchor_week):
            week_axis.axvline(anchor_week, color="black", linestyle="--", linewidth=1.0)
        if np.isfinite(anchor_throughput) and anchor_throughput > 0.0:
            throughput_axis.axvline(anchor_throughput, color="black", linestyle="--", linewidth=1.0)

        week_axis.set_xlabel("Weeks")
        week_axis.set_ylabel(helper_feature)
        week_axis.set_title(f"{helper_feature} trajectory vs weeks")
        week_axis.grid(True, alpha=0.3)
        week_axis.legend()

        throughput_axis.set_xlabel("Throughput")
        throughput_axis.set_ylabel(helper_feature)
        throughput_axis.set_title(f"{helper_feature} trajectory vs throughput")
        throughput_axis.grid(True, alpha=0.3)
        if all(
            np.isfinite(float(cell.get("input_throughput_value", np.nan))) and float(cell.get("input_throughput_value", np.nan)) > 0.0
            for cell in [*train_sample[:1], *augmented_sample[:1], *test_sample[:1]]
            if cell is not None
        ):
            throughput_axis.set_xscale("log")
        throughput_axis.legend()

    fig.suptitle(f"{title_prefix}: augmentation-driver feature trajectories", fontsize=12)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_workflow_diagnostics(
    *,
    train_real_cells: list[dict[str, object]],
    augmented_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    metadata: dict[str, object],
    anchor_search_df: pd.DataFrame,
    title_prefix: str,
    show: bool = True,
) -> None:
    train_targets = np.asarray([cell["target_value"] for cell in train_real_cells], dtype=float)
    aug_targets = np.asarray([cell["target_value"] for cell in augmented_cells], dtype=float)
    test_targets = np.asarray([cell["target_value"] for cell in test_cells], dtype=float)
    train_target_throughput = np.asarray([cell.get("target_throughput_value", np.nan) for cell in train_real_cells], dtype=float)
    aug_target_throughput = np.asarray([cell.get("target_throughput_value", np.nan) for cell in augmented_cells], dtype=float)
    test_target_throughput = np.asarray([cell.get("target_throughput_value", np.nan) for cell in test_cells], dtype=float)
    train_input_throughput = np.asarray([cell.get("input_throughput_value", np.nan) for cell in train_real_cells], dtype=float)
    aug_input_throughput = np.asarray([cell.get("input_throughput_value", np.nan) for cell in augmented_cells], dtype=float)
    test_input_throughput = np.asarray([cell.get("input_throughput_value", np.nan) for cell in test_cells], dtype=float)

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    bins = min(20, max(6, len(train_targets) // 4 if len(train_targets) else 6))
    axes1[0].hist(train_targets, bins=bins, alpha=0.55, color="steelblue", label="Train real")
    if aug_targets.size:
        axes1[0].hist(aug_targets, bins=bins, alpha=0.45, color="teal", label="Train augmented")
    axes1[0].hist(test_targets, bins=min(10, max(3, len(test_targets))), alpha=0.70, color="coral", label="Test")
    axes1[0].set_xlabel("Weeks at 95% SOH")
    axes1[0].set_ylabel("Cell count")
    axes1[0].set_title("Target distribution")
    axes1[0].grid(True, alpha=0.3)
    axes1[0].legend()

    real_input_refs = np.asarray([cell["input_reference_value"] for cell in train_real_cells], dtype=float)
    aug_input_refs = np.asarray([cell["input_reference_value"] for cell in augmented_cells], dtype=float)
    test_input_refs = np.asarray([cell["input_reference_value"] for cell in test_cells], dtype=float)
    axes1[1].scatter(real_input_refs, train_targets, color="steelblue", alpha=0.7, label="Train real")
    if aug_targets.size:
        axes1[1].scatter(aug_input_refs, aug_targets, color="teal", alpha=0.45, s=18, label="Train augmented")
    axes1[1].scatter(test_input_refs, test_targets, color="coral", alpha=0.9, s=50, marker="D", label="Test")
    if np.isfinite(float(metadata.get("selected_anchor_week", np.nan))):
        axes1[1].axvline(float(metadata["selected_anchor_week"]), color="black", linestyle="--", linewidth=1.2, label="Anchor week")
        axes1[1].set_xlabel("Anchor week")
    elif np.isfinite(float(metadata.get("selected_anchor_throughput", np.nan))):
        axes1[1].axvline(float(metadata["selected_anchor_throughput"]), color="black", linestyle="--", linewidth=1.2, label="Anchor throughput")
        axes1[1].set_xlabel("Anchor throughput")
    else:
        axes1[1].set_xlabel("Anchor reference")
    axes1[1].set_ylabel("Weeks at 95% SOH")
    axes1[1].set_title("Target vs anchored input reference")
    axes1[1].grid(True, alpha=0.3)
    axes1[1].legend()
    fig1.suptitle(f"{title_prefix}: data distribution", fontsize=12)
    fig1.tight_layout()

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for cell in train_real_cells[: min(20, len(train_real_cells))]:
        weeks = np.asarray(cell["time_curve"], dtype=float)
        soh = np.asarray(cell["soh_curve"], dtype=float)
        axes2[0].plot(weeks, soh, color="steelblue", alpha=0.18, linewidth=1.0)
    for cell in test_cells:
        weeks = np.asarray(cell["time_curve"], dtype=float)
        soh = np.asarray(cell["soh_curve"], dtype=float)
        axes2[0].plot(weeks, soh, color="coral", alpha=0.95, linewidth=2.0)
    if np.isfinite(float(metadata.get("selected_anchor_week", np.nan))):
        axes2[0].axvline(float(metadata["selected_anchor_week"]), color="black", linestyle="--", linewidth=1.2)
    axes2[0].set_xlabel("Weeks")
    axes2[0].set_ylabel("SOH")
    axes2[0].set_title("SOH trajectories with anchor")
    axes2[0].grid(True, alpha=0.3)

    for cell in train_real_cells[: min(15, len(train_real_cells))]:
        seq = np.asarray(cell["sequence"], dtype=float)
        soh_seq = seq[:, -1]
        input_week_grid = np.asarray(cell.get("input_week_grid", []), dtype=float)
        x_values = input_week_grid if input_week_grid.size == len(soh_seq) else np.arange(len(soh_seq))
        axes2[1].plot(x_values, soh_seq, color="steelblue", alpha=0.25, linewidth=1.0)
    for cell in augmented_cells[: min(30, len(augmented_cells))]:
        seq = np.asarray(cell["sequence"], dtype=float)
        soh_seq = seq[:, -1]
        input_week_grid = np.asarray(cell.get("input_week_grid", []), dtype=float)
        x_values = input_week_grid if input_week_grid.size == len(soh_seq) else np.arange(len(soh_seq))
        axes2[1].plot(x_values, soh_seq, color="teal", alpha=0.12, linewidth=0.9)
    for cell in test_cells:
        seq = np.asarray(cell["sequence"], dtype=float)
        soh_seq = seq[:, -1]
        input_week_grid = np.asarray(cell.get("input_week_grid", []), dtype=float)
        x_values = input_week_grid if input_week_grid.size == len(soh_seq) else np.arange(len(soh_seq))
        axes2[1].plot(x_values, soh_seq, color="coral", alpha=0.95, linewidth=2.0)
    axes2[1].set_xlabel("Weeks")
    axes2[1].set_ylabel("SOH feature in model input")
    axes2[1].set_title("Anchored input sequences")
    axes2[1].grid(True, alpha=0.3)
    fig2.suptitle(f"{title_prefix}: anchor geometry", fontsize=12)
    fig2.tight_layout()

    if augmented_cells:
        fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
        scale_values = _extract_scale_values(augmented_cells)
        if scale_values.size:
            axes3[0, 0].hist(scale_values, bins=min(20, max(6, len(scale_values) // 10)), color="teal", alpha=0.75)
            axes3[0, 0].set_xlabel("Augmentation scale / warp factor")
            axes3[0, 0].set_ylabel("Synthetic cell count")
            axes3[0, 0].set_title("Augmentation scale distribution")
            axes3[0, 0].grid(True, alpha=0.3)
            axes3[0, 1].scatter(scale_values, aug_targets[: len(scale_values)], color="teal", alpha=0.5, s=18)
            axes3[0, 1].set_xlabel("Augmentation scale / warp factor")
            axes3[0, 1].set_ylabel("Synthetic target weeks")
            axes3[0, 1].set_title("Synthetic scale vs target")
            axes3[0, 1].grid(True, alpha=0.3)
        else:
            axes3[0, 0].hist(aug_targets, bins=min(20, max(6, len(aug_targets) // 10)), color="teal", alpha=0.75)
            axes3[0, 0].set_xlabel("Synthetic target weeks")
            axes3[0, 0].set_ylabel("Synthetic cell count")
            axes3[0, 0].set_title("Augmented target distribution")
            axes3[0, 0].grid(True, alpha=0.3)
            aug_input_refs = np.asarray([cell["input_reference_value"] for cell in augmented_cells], dtype=float)
            axes3[0, 1].scatter(aug_input_refs, aug_targets, color="teal", alpha=0.5, s=18)
            axes3[0, 1].set_xlabel("Synthetic anchor reference")
            axes3[0, 1].set_ylabel("Synthetic target weeks")
            axes3[0, 1].set_title("Synthetic input reference vs target")
            axes3[0, 1].grid(True, alpha=0.3)

        real_x = np.asarray([cell["x"] for cell in train_real_cells], dtype=float)
        aug_x = np.asarray([cell["x"] for cell in augmented_cells], dtype=float)
        test_x = np.asarray([cell["x"] for cell in test_cells], dtype=float)
        dim0 = 0
        dim1 = 1 if real_x.shape[1] > 1 else 0
        axes3[1, 0].scatter(real_x[:, dim0], real_x[:, dim1], color="steelblue", alpha=0.55, s=22, label="Train real")
        axes3[1, 0].scatter(aug_x[:, dim0], aug_x[:, dim1], color="teal", alpha=0.28, s=18, label="Train augmented")
        axes3[1, 0].scatter(test_x[:, dim0], test_x[:, dim1], color="coral", alpha=0.95, s=45, marker="D", label="Test")
        axes3[1, 0].set_xlabel("Feature 1")
        axes3[1, 0].set_ylabel("Feature 2" if dim1 != dim0 else "Feature 1")
        axes3[1, 0].set_title("Real vs augmented feature cloud")
        axes3[1, 0].grid(True, alpha=0.3)
        axes3[1, 0].legend()

        for cell in train_real_cells[: min(12, len(train_real_cells))]:
            weeks = np.asarray(cell["time_curve"], dtype=float)
            soh = np.asarray(cell["soh_curve"], dtype=float)
            axes3[1, 1].plot(weeks, soh, color="steelblue", alpha=0.16, linewidth=1.0)
        for cell in augmented_cells:
            weeks = np.asarray(cell["time_curve"], dtype=float)
            soh = np.asarray(cell["soh_curve"], dtype=float)
            axes3[1, 1].plot(weeks, soh, color="teal", alpha=0.12, linewidth=0.9)
        for cell in test_cells:
            weeks = np.asarray(cell["time_curve"], dtype=float)
            soh = np.asarray(cell["soh_curve"], dtype=float)
            axes3[1, 1].plot(weeks, soh, color="coral", alpha=0.9, linewidth=1.8)
        if np.isfinite(float(metadata.get("selected_anchor_soh", np.nan))):
            axes3[1, 1].axhline(float(metadata["selected_anchor_soh"]), color="black", linestyle="--", linewidth=1.1, label="Anchor SOH")
        if np.isfinite(float(metadata.get("target_soh", np.nan))):
            axes3[1, 1].axhline(float(metadata["target_soh"]), color="gray", linestyle=":", linewidth=1.1, label="Target SOH")
        axes3[1, 1].set_xlabel("Weeks")
        axes3[1, 1].set_ylabel("SOH")
        axes3[1, 1].set_title("Augmented trajectories")
        axes3[1, 1].grid(True, alpha=0.3)
        axes3[1, 1].legend()
        fig3.suptitle(f"{title_prefix}: augmentation diagnostics", fontsize=12)
        fig3.tight_layout()
    else:
        fig3 = None

    finite_throughput = np.isfinite(train_target_throughput).any() or np.isfinite(aug_target_throughput).any() or np.isfinite(test_target_throughput).any()
    if finite_throughput:
        fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
        thr_bins = min(20, max(6, len(train_target_throughput) // 4 if len(train_target_throughput) else 6))
        axes5[0].hist(train_target_throughput[np.isfinite(train_target_throughput)], bins=thr_bins, alpha=0.55, color="steelblue", label="Train real")
        if aug_target_throughput.size:
            axes5[0].hist(aug_target_throughput[np.isfinite(aug_target_throughput)], bins=thr_bins, alpha=0.45, color="teal", label="Train augmented")
        axes5[0].hist(test_target_throughput[np.isfinite(test_target_throughput)], bins=min(10, max(3, len(test_target_throughput))), alpha=0.70, color="coral", label="Test")
        axes5[0].set_xlabel("Throughput at 95% SOH")
        axes5[0].set_ylabel("Cell count")
        axes5[0].set_title("Target throughput distribution")
        axes5[0].grid(True, alpha=0.3)
        axes5[0].legend()

        axes5[1].scatter(train_input_throughput, train_target_throughput, color="steelblue", alpha=0.7, label="Train real")
        if aug_targets.size:
            axes5[1].scatter(aug_input_throughput, aug_target_throughput, color="teal", alpha=0.45, s=18, label="Train augmented")
        axes5[1].scatter(test_input_throughput, test_target_throughput, color="coral", alpha=0.9, s=50, marker="D", label="Test")
        axes5[1].set_xlabel("Input throughput at cutoff")
        axes5[1].set_ylabel("Throughput at 95% SOH")
        axes5[1].set_title("Input throughput vs target throughput")
        axes5[1].grid(True, alpha=0.3)
        axes5[1].legend()
        fig5.suptitle(f"{title_prefix}: target throughput", fontsize=12)
        fig5.tight_layout()
    else:
        fig5 = None

    if anchor_search_df is not None and not anchor_search_df.empty:
        fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
        if "anchor_soh" in anchor_search_df.columns and "usable_train_count" in anchor_search_df.columns:
            axes4[0].plot(anchor_search_df["anchor_soh"], anchor_search_df["usable_train_count"], marker="o", color="steelblue", label="Usable train")
            if "usable_test_count" in anchor_search_df.columns:
                axes4[0].plot(anchor_search_df["anchor_soh"], anchor_search_df["usable_test_count"], marker="D", color="coral", label="Usable test")
            axes4[0].set_xlabel("Candidate anchor SOH")
            axes4[0].set_ylabel("Usable cell count")
            axes4[0].set_title("Anchor candidate coverage")
            axes4[0].grid(True, alpha=0.3)
            axes4[0].legend()
            if "selected" in anchor_search_df.columns:
                selected_rows = anchor_search_df[anchor_search_df["selected"]]
                if not selected_rows.empty:
                    selected_x = float(selected_rows["anchor_soh"].iloc[0])
                    axes4[0].axvline(selected_x, color="black", linestyle="--", linewidth=1.2)
        else:
            axes4[0].axis("off")
        axes4[1].axis("off")
        summary_lines = [
            f"anchor_mode: {metadata.get('anchor_mode', 'unknown')}",
            f"selected_anchor_week: {metadata.get('selected_anchor_week', np.nan):.3f}" if np.isfinite(float(metadata.get("selected_anchor_week", np.nan))) else "selected_anchor_week: n/a",
            f"selected_anchor_throughput: {metadata.get('selected_anchor_throughput', np.nan):.6g}" if np.isfinite(float(metadata.get("selected_anchor_throughput", np.nan))) else "selected_anchor_throughput: n/a",
            f"selected_anchor_soh: {metadata.get('selected_anchor_soh', np.nan):.4f}" if np.isfinite(float(metadata.get("selected_anchor_soh", np.nan))) else "selected_anchor_soh: n/a",
            f"augmentation_method: {metadata.get('augmentation_method', 'none')}",
            f"train_real_count: {int(metadata.get('train_real_count', 0))}",
            f"train_aug_count: {int(metadata.get('train_aug_count', 0))}",
            f"test_count: {int(metadata.get('test_count', 0))}",
        ]
        axes4[1].text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=10)
        fig4.suptitle(f"{title_prefix}: anchor selection", fontsize=12)
        fig4.tight_layout()
    else:
        fig4 = None

    if show:
        plt.show()
    else:
        for figure in (fig1, fig2, fig3, fig4, fig5):
            if figure is not None:
                plt.close(figure)
    plot_training_trajectory_overlay(
        train_real_cells=train_real_cells,
        augmented_cells=augmented_cells,
        test_cells=test_cells,
        metadata=metadata,
        title_prefix=title_prefix,
        show=show,
    )


def run_lfp_prediction_pipeline(
    *,
    interp_dir: Path = RAW_FEATURE_DIR,
    feature_columns: list[str] | None = None,
    target_column: str | None = DEFAULT_TARGET_COL,
    target_soh: float | None = DEFAULT_TARGET_SOH,
    training_config: TrainingConfig | None = None,
    explicit_test_names: list[str] | tuple[str, ...] | None = None,
    test_substring: str = DEFAULT_TEST_SUBSTRING,
    train_substring: str = DEFAULT_TRAIN_SUBSTRING,
    fixed_anchor_soh: float | None = DEFAULT_ANCHOR_SOH,
    fixed_anchor_week: float | None = None,
    fixed_anchor_throughput: float | None = None,
    autoanchor: bool = False,
    autoanchor_min_soh: float = DEFAULT_AUTOANCHOR_MIN_SOH,
    autoanchor_max_soh: float = DEFAULT_AUTOANCHOR_MAX_SOH,
    autoanchor_step_soh: float = DEFAULT_AUTOANCHOR_STEP_SOH,
    augmentation_method: str = "none",
    augmentation_sample_count: int | None = None,
    show_plot: bool = True,
    plot_title_prefix: str = "LFP prediction",
) -> dict[str, object]:
    feature_columns = list(DEFAULT_FEATURE_COLS if feature_columns is None else feature_columns)
    training_config = TrainingConfig() if training_config is None else training_config
    cell_frames, skipped_frames = load_lfp_cell_frames(
        interp_dir=Path(interp_dir),
        feature_columns=feature_columns,
        target_column=(str(target_column) if target_column is not None else None),
        target_soh=target_soh,
    )
    all_names = [str(frame["cell"]) for frame in cell_frames]
    train_names, test_names = infer_train_test_names(
        all_names,
        explicit_test_names=explicit_test_names,
        test_substring=str(test_substring),
        train_substring=str(train_substring),
    )
    if not test_names:
        raise RuntimeError("No test cells were inferred from the available files.")
    train_frames = [frame for frame in cell_frames if str(frame["cell"]) in set(train_names)]
    test_frames = [frame for frame in cell_frames if str(frame["cell"]) in set(test_names)]
    if not train_frames:
        raise RuntimeError("No training cells were inferred from the available files.")
    if not test_frames:
        raise RuntimeError("No test cells were inferred from the available files.")

    selected_frames = [*train_frames, *test_frames]
    effective_input_steps = resolve_week_step_count(
        selected_frames,
        anchor_week=fixed_anchor_week,
        anchor_throughput=fixed_anchor_throughput,
        anchor_soh=(
            float(fixed_anchor_soh)
            if fixed_anchor_week is None and fixed_anchor_throughput is None and fixed_anchor_soh is not None
            else None
        ),
        fallback_steps=int(training_config.input_steps),
    )
    effective_training_config = replace(training_config, input_steps=int(effective_input_steps))

    if fixed_anchor_week is not None or fixed_anchor_throughput is not None:
        selected_anchor_soh = np.nan
        anchor_search_df = pd.DataFrame()
    elif autoanchor:
        try:
            selected_anchor_soh, anchor_search_df = select_autoanchor_soh(
                [*train_frames, *test_frames],
                input_steps=int(training_config.input_steps),
                min_soh=float(autoanchor_min_soh),
                max_soh=float(autoanchor_max_soh),
                step_soh=float(autoanchor_step_soh),
            )
        except RuntimeError:
            selected_anchor_soh, anchor_search_df = select_balanced_anchor_soh(
                train_frames,
                test_frames,
                input_steps=int(training_config.input_steps),
                min_soh=float(autoanchor_min_soh),
                max_soh=float(autoanchor_max_soh),
                step_soh=float(autoanchor_step_soh),
            )
    elif fixed_anchor_soh is not None:
        selected_anchor_soh = float(fixed_anchor_soh)
        anchor_search_df = pd.DataFrame()
    else:
        selected_anchor_soh, anchor_search_df = select_balanced_anchor_soh(
            train_frames,
            test_frames,
            input_steps=int(training_config.input_steps),
            min_soh=float(autoanchor_min_soh),
            max_soh=float(autoanchor_max_soh),
            step_soh=float(autoanchor_step_soh),
        )

    train_cells, skipped_train = build_model_cells(
        train_frames,
        feature_columns=feature_columns,
        input_steps=int(effective_training_config.input_steps),
        anchor_soh=float(selected_anchor_soh),
        anchor_week=fixed_anchor_week,
        anchor_throughput=fixed_anchor_throughput,
        sequence_steps=int(effective_input_steps),
    )
    test_cells, skipped_test = build_model_cells(
        test_frames,
        feature_columns=feature_columns,
        input_steps=int(effective_training_config.input_steps),
        anchor_soh=float(selected_anchor_soh),
        anchor_week=fixed_anchor_week,
        anchor_throughput=fixed_anchor_throughput,
        sequence_steps=int(effective_input_steps),
    )
    augmented_cells, augmentation_metadata = build_augmented_cells(
        train_cells,
        test_cells,
        feature_columns=feature_columns,
        input_steps=int(effective_training_config.input_steps),
        sequence_steps=int(effective_input_steps),
        anchor_soh=(
            float(selected_anchor_soh)
            if np.isfinite(float(selected_anchor_soh))
            else float(DEFAULT_ANCHOR_SOH)
        ),
        augmentation_method=str(augmentation_method),
        task_kind="throughput" if fixed_anchor_throughput is not None else "time",
        sample_count=augmentation_sample_count,
        seed_source=int(training_config.seed_source),
    )
    prediction_model = normalize_prediction_model(effective_training_config.prediction_model)
    if prediction_model == PREDICTION_MODEL_LEGACY_TCNN:
        effective_train_cells, effective_test_cells, _scaler = scale_cell_sequences(
            [*train_cells, *augmented_cells],
            test_cells,
            feature_columns=feature_columns,
        )
    else:
        effective_train_cells = [*train_cells, *augmented_cells]
        effective_test_cells = test_cells
    result = train_prediction_ensemble(
        effective_train_cells,
        effective_test_cells,
        training_config=effective_training_config,
    )
    metadata = {
        "selected_anchor_soh": float(selected_anchor_soh),
        "selected_anchor_week": float(fixed_anchor_week) if fixed_anchor_week is not None else np.nan,
        "selected_anchor_throughput": float(fixed_anchor_throughput) if fixed_anchor_throughput is not None else np.nan,
        "anchor_mode": "throughput" if fixed_anchor_throughput is not None else ("week" if fixed_anchor_week is not None else "soh"),
        "target_soh": float(target_soh) if target_soh is not None else np.nan,
        "target_unit": "weeks_at_target_soh",
        "target_throughput_unit": "throughput_at_target_soh",
        "week_target_divisor": float(result.get("week_target_divisor", np.nan)),
        "throughput_target_divisor": float(result.get("throughput_target_divisor", np.nan)),
        "input_sequence_steps": int(effective_input_steps),
        "train_real_count": int(len(train_cells)),
        "train_aug_count": int(len(augmented_cells)),
        "test_count": int(len(test_cells)),
        "prediction_model": str(prediction_model),
        "skipped_frame_count": int(len(skipped_frames)),
        "skipped_sequence_count": int(len(skipped_train) + len(skipped_test)),
        "augmentation_method": str(augmentation_method).strip().lower(),
        **augmentation_metadata,
    }
    if show_plot:
        plot_prediction_results(
            result,
            title_prefix=str(plot_title_prefix),
            show=True,
        )
        plot_workflow_diagnostics(
            train_real_cells=train_cells,
            augmented_cells=augmented_cells,
            test_cells=test_cells,
            metadata=metadata,
            anchor_search_df=anchor_search_df,
            title_prefix=str(plot_title_prefix),
            show=True,
        )
        plot_feature_log2_helper_diagnostics(
            train_real_cells=train_cells,
            augmented_cells=augmented_cells,
            test_cells=test_cells,
            metadata=metadata,
            title_prefix=str(plot_title_prefix),
            show=True,
        )
        plot_feature_log2_helper_feature_trajectories(
            train_real_cells=train_cells,
            augmented_cells=augmented_cells,
            test_cells=test_cells,
            metadata=metadata,
            title_prefix=str(plot_title_prefix),
            show=True,
        )
    return {
        **result,
        "metadata": metadata,
        "anchor_search_df": anchor_search_df,
        "skipped_frames_df": pd.DataFrame(skipped_frames),
        "skipped_sequences_df": pd.DataFrame([*skipped_train, *skipped_test]),
        "inferred_train_names": train_names,
        "inferred_test_names": test_names,
        "train_real_cells": train_cells,
        "augmented_cells": augmented_cells,
        "test_cells_data": test_cells,
        "feature_columns": feature_columns,
    }
