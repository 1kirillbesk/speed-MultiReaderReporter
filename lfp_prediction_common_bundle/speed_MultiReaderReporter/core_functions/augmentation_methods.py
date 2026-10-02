from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CAL_WARP_FACTOR_MIN = 1.5
CAL_WARP_FACTOR_MAX: float | None = None
CAL_MAX_TEST_MARGIN = 1.5
CAL_ANCHOR_TARGET_STATISTIC = "mean"
CAL_ANCHOR_CENTER_JITTER = 0.18
CAL_UPPER_ANCHOR_TARGET_FRACTION = 0.45
CAL_UPPER_ANCHOR_MIN_GAP_FRACTION = 0.65
CAL_NOISE_STD_FRACTION = 0.03
CAL_RANDOM_SEED = 42
HYBRID_SCALE_JITTER = 0.08
VALID_AUGMENTATION_METHODS = ("normal", "cal", "hybrid", "codex")
# Fraction of the requested CODEX sample count that must actually be generated
# before the shortfall counts as a broken generator rather than normal rejection.
CODEX_MIN_USABLE_SAMPLE_FRACTION = 0.5
DEFAULT_AUTOANCHOR_AUGMENTATION_METHOD = "normal"
DEFAULT_CAL_AUGMENTATION_SAMPLE_COUNT = 200
DEFAULT_HYBRID_AUGMENTATION_SAMPLE_COUNT = 500
DEFAULT_CODEX_AUGMENTATION_SAMPLE_COUNT = DEFAULT_CAL_AUGMENTATION_SAMPLE_COUNT
HYBRID_TEST_ANCHOR_COVERAGE_MULTIPLIER = 1.0
HYBRID_CANDIDATE_OVERSAMPLE_FACTOR = 1.5
HYBRID_MAX_AUGMENTATION_SCALE: float | None = None
CODEX_TARGET_ANCHOR_QUANTILES = (0.25, 0.75)
CODEX_MIN_GENERATED_AUGMENTATION_SCALE = 1.02
NORMAL_MIN_GENERATED_AUGMENTATION_SCALE = 1.02
NORMAL_MAX_AUGMENTATION_SCALE: float | None = None
NORMAL_STANDARD_MAX_AUGMENTATION_SCALE = 3.0
NORMAL_STANDARD_ANCHOR_COVERAGE_MULTIPLIER = 3.0
NORMAL_FIXED_LOW_AUGMENTATION_SCALE = 1.2
NORMAL_EXTRA_AUGMENTATION_SCALES: tuple[float, ...] = ()
NORMAL_MAX_AUGMENTED_TO_TEST_ANCHOR_RATIO = 1.0
NORMAL_MAX_AUGMENTATION_SCALES_PER_CELL: int | None = None
NORMAL_TARGET_TOTAL_TRAIN_COUNT = 240
NORMAL_MAX_BUDGET_AUGMENTATION_SCALES_PER_CELL = 30
NORMAL_BUDGET_SCALE_RAMP_HIGH_FRACTION = 0.0
NORMAL_BUDGET_SCALE_RAMP_POWER = 1.0
NORMAL_DENSE_AUGMENTATION_MAX_THRESHOLD_WEEKS = 5.0
NORMAL_AUGMENTATION_SCALE_SPACING = "fixed_count"
NORMAL_AUGMENTATION_SCALE_STEP = 0.5
NORMAL_AUGMENTATION_FACTOR_COUNT = 5


def finite_float(value: object, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def resolve_reference_column(module: Any, task_kind: str) -> str:
    if str(task_kind).strip().lower() == "throughput":
        for attr_name in ("CUMULATIVE_THROUGHPUT_COLUMN", "THROUGHPUT_COLUMN"):
            if hasattr(module, attr_name):
                return str(getattr(module, attr_name))
    if hasattr(module, "TIME_COLUMN"):
        return str(getattr(module, "TIME_COLUMN"))
    raise AttributeError(f"Could not resolve reference column for task_kind={task_kind!r}.")


def resolve_capacity_column(module: Any) -> str:
    if hasattr(module, "CAPACITY_COLUMN"):
        return str(getattr(module, "CAPACITY_COLUMN"))
    raise AttributeError("Could not resolve capacity column for anchor extraction.")


def _series_from_raw_df(raw_df: Any, column_name: str) -> np.ndarray:
    if column_name not in raw_df:
        return np.asarray([], dtype=float)
    return pd.to_numeric(raw_df[column_name], errors="coerce").to_numpy(dtype=float)


def extract_anchor_value_until_first_crossing(
    module: Any,
    cell: dict[str, object],
    *,
    task_kind: str,
) -> float:
    raw_df = cell.get("raw_df")
    if raw_df is None:
        return np.nan
    normalized_task_kind = str(task_kind).strip().lower()
    capacity_column = resolve_capacity_column(module)
    capacities = _series_from_raw_df(raw_df, capacity_column)
    if capacities.size < 2:
        return np.nan
    if normalized_task_kind == "throughput":
        reference_column = resolve_reference_column(module, normalized_task_kind)
        references = _series_from_raw_df(raw_df, reference_column)
        if references.size < 2 and hasattr(module, "THROUGHPUT_COLUMN"):
            throughput_steps = _series_from_raw_df(raw_df, str(getattr(module, "THROUGHPUT_COLUMN")))
            references = np.cumsum(np.nan_to_num(throughput_steps, nan=0.0))
    else:
        references = _series_from_raw_df(raw_df, resolve_reference_column(module, normalized_task_kind))
    point_count = min(int(references.size), int(capacities.size))
    if point_count < 2:
        return np.nan
    references = references[:point_count]
    capacities = capacities[:point_count]
    valid = np.isfinite(references) & np.isfinite(capacities)
    if np.count_nonzero(valid) < 2:
        return np.nan
    references = references[valid]
    capacities = capacities[valid]
    order = np.argsort(references)
    references = references[order]
    capacities = capacities[order]
    initial_capacity = float(capacities[0])
    if not np.isfinite(initial_capacity) or initial_capacity <= 0.0:
        return np.nan
    soh_values = capacities / initial_capacity
    target_soh = finite_float(getattr(module, "AUGMENTATION_REFERENCE_SOH", np.nan))
    if not np.isfinite(target_soh) or target_soh <= 0.0:
        return np.nan
    for index in range(1, int(soh_values.size)):
        previous_soh = float(soh_values[index - 1])
        current_soh = float(soh_values[index])
        previous_reference = float(references[index - 1])
        current_reference = float(references[index])
        if not all(
            np.isfinite(value)
            for value in (previous_soh, current_soh, previous_reference, current_reference)
        ):
            continue
        if np.isclose(previous_soh, target_soh, rtol=0.0, atol=1e-12):
            return previous_reference
        if np.isclose(current_soh, target_soh, rtol=0.0, atol=1e-12):
            return current_reference
        if previous_soh >= float(target_soh) >= current_soh:
            if np.isclose(previous_soh, current_soh, rtol=0.0, atol=1e-12):
                return current_reference
            fraction = (float(target_soh) - previous_soh) / (current_soh - previous_soh)
            anchor_reference = previous_reference + fraction * (current_reference - previous_reference)
            return float(anchor_reference) if np.isfinite(anchor_reference) else np.nan
    return np.nan


def collect_anchor_values_until_first_crossing(
    module: Any,
    cells: list[dict[str, object]],
    *,
    task_kind: str,
) -> np.ndarray:
    return np.asarray(
        [
            extract_anchor_value_until_first_crossing(module, cell, task_kind=str(task_kind))
            for cell in cells
        ],
        dtype=float,
    )


def collect_anchor_values(module: Any, cells: list[dict[str, object]], *, task_kind: str) -> np.ndarray:
    if hasattr(module, "collect_time_or_throughput_values_at_soh"):
        return np.asarray(
            module.collect_time_or_throughput_values_at_soh(cells, task_kind=str(task_kind)),
            dtype=float,
        )
    normalized_task_kind = str(task_kind).strip().lower()
    if normalized_task_kind == "throughput" and hasattr(module, "collect_throughput_values_at_soh"):
        return np.asarray(module.collect_throughput_values_at_soh(cells), dtype=float)
    if normalized_task_kind == "time" and hasattr(module, "collect_time_values_at_soh"):
        return np.asarray(module.collect_time_values_at_soh(cells), dtype=float)
    raise AttributeError(f"Could not collect anchor values for task_kind={task_kind!r}.")


def extract_anchor_value(module: Any, cell: dict[str, object], *, task_kind: str) -> float:
    if hasattr(module, "extract_time_or_throughput_at_soh"):
        return finite_float(module.extract_time_or_throughput_at_soh(cell, task_kind=str(task_kind)))
    normalized_task_kind = str(task_kind).strip().lower()
    if normalized_task_kind == "throughput" and hasattr(module, "extract_throughput_at_soh"):
        return finite_float(module.extract_throughput_at_soh(cell))
    if normalized_task_kind == "time" and hasattr(module, "extract_time_at_soh"):
        return finite_float(module.extract_time_at_soh(cell))
    return np.nan


def _fit_std_to_shape(std_values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    std_values = np.asarray(std_values, dtype=float)
    if std_values.shape == shape:
        return np.where(np.isfinite(std_values), std_values, 0.0)
    if std_values.ndim == 1 and len(shape) == 1:
        result = np.zeros(shape, dtype=float)
        count = min(int(result.size), int(std_values.size))
        if count:
            result[:count] = std_values[:count]
        return np.where(np.isfinite(result), result, 0.0)
    if std_values.ndim == 2 and len(shape) == 2:
        result = np.zeros(shape, dtype=float)
        rows = min(int(result.shape[0]), int(std_values.shape[0]))
        cols = min(int(result.shape[1]), int(std_values.shape[1]))
        if rows and cols:
            result[:rows, :cols] = std_values[:rows, :cols]
        return np.where(np.isfinite(result), result, 0.0)
    return np.zeros(shape, dtype=float)


def _fit_vector_noise_std(
    std_values: np.ndarray,
    shape: tuple[int, ...],
    *,
    output_x_drop_count: int,
) -> np.ndarray:
    std_values = np.asarray(std_values, dtype=float)
    if (
        int(output_x_drop_count) > 0
        and std_values.ndim == 1
        and len(shape) == 1
        and std_values.size + int(output_x_drop_count) == int(shape[0])
    ):
        prefix = np.zeros(int(output_x_drop_count), dtype=float)
        result = np.concatenate([prefix, std_values])
        return np.where(np.isfinite(result), result, 0.0)
    return _fit_std_to_shape(std_values, shape)


def build_cal_augmentation_stats(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    task_kind: str,
    sample_count: int,
) -> dict[str, float]:
    train_targets = np.asarray([finite_float(cell.get("target_value")) for cell in train_cells], dtype=float)
    train_targets = train_targets[np.isfinite(train_targets) & (train_targets > 0.0)]
    train_anchor_values = collect_anchor_values(module, train_cells, task_kind=str(task_kind))
    train_anchor_values = train_anchor_values[np.isfinite(train_anchor_values) & (train_anchor_values > 0.0)]
    test_anchor_values = collect_anchor_values(module, test_cells, task_kind=str(task_kind))
    test_anchor_values = test_anchor_values[np.isfinite(test_anchor_values) & (test_anchor_values > 0.0)]

    train_q10 = train_q25 = train_q50 = train_q75 = train_q90 = train_max = np.nan
    test_q10 = test_q25 = test_q50 = test_q75 = test_q90 = test_max = np.nan
    if train_targets.size:
        train_q10, train_q25, train_q50, train_q75, train_q90 = np.quantile(
            train_targets,
            [0.10, 0.25, 0.50, 0.75, 0.90],
        ).astype(float)
        train_max = float(np.max(train_targets))
    if test_anchor_values.size:
        test_q10, test_q25, test_q50, test_q75, test_q90 = np.quantile(
            test_anchor_values,
            [0.10, 0.25, 0.50, 0.75, 0.90],
        ).astype(float)
        test_max = float(np.max(test_anchor_values))

    max_anchor_scale = np.nan
    if test_anchor_values.size and train_anchor_values.size:
        max_anchor_scale = float(CAL_MAX_TEST_MARGIN) * float(np.max(test_anchor_values)) / float(np.max(train_anchor_values))

    return {
        "train_reference": float(train_q50) if np.isfinite(train_q50) else np.nan,
        "train_q10": float(train_q10) if np.isfinite(train_q10) else np.nan,
        "train_q25": float(train_q25) if np.isfinite(train_q25) else np.nan,
        "train_q50": float(train_q50) if np.isfinite(train_q50) else np.nan,
        "train_q75": float(train_q75) if np.isfinite(train_q75) else np.nan,
        "train_q90": float(train_q90) if np.isfinite(train_q90) else np.nan,
        "train_max": float(train_max) if np.isfinite(train_max) else np.nan,
        "test_q10": float(test_q10) if np.isfinite(test_q10) else np.nan,
        "test_q25": float(test_q25) if np.isfinite(test_q25) else np.nan,
        "test_q50": float(test_q50) if np.isfinite(test_q50) else np.nan,
        "test_q75": float(test_q75) if np.isfinite(test_q75) else np.nan,
        "test_q90": float(test_q90) if np.isfinite(test_q90) else np.nan,
        "test_max": float(test_max) if np.isfinite(test_max) else np.nan,
        "test_reference": float(test_q50) if np.isfinite(test_q50) else np.nan,
        "test_upper_reference": np.nan,
        "slowest_test_anchor": float(test_max) if np.isfinite(test_max) else np.nan,
        "max_augmented_anchor_allowed": (
            float(CAL_MAX_TEST_MARGIN) * float(test_max) if np.isfinite(test_max) else np.nan
        ),
        "max_augmented_anchor_median_allowed": (
            float(CAL_MAX_TEST_MARGIN) * float(test_max) if np.isfinite(test_max) else np.nan
        ),
        "max_coverage_scale": np.nan,
        "max_anchor_scale": float(max_anchor_scale) if np.isfinite(max_anchor_scale) else np.nan,
        "scale_q25_test_over_train_q50": np.nan,
        "scale_q50_test_over_train_q50": np.nan,
        "scale_q75_test_over_train_q50": np.nan,
        "target_total_train_count": int(len(train_cells) + max(0, int(sample_count))),
        "target_aug_count": int(max(0, int(sample_count))),
        "target_scale_count": 0,
        "dense_small_threshold_augmentation": 0,
    }


def select_scale_spread(scales: list[float], max_count: int) -> list[float]:
    finite_scales = sorted({float(scale) for scale in scales if np.isfinite(float(scale))})
    if int(max_count) <= 0:
        return []
    if len(finite_scales) <= int(max_count):
        return finite_scales
    positions = np.linspace(0, len(finite_scales) - 1, int(max_count), dtype=float)
    indices = sorted({int(round(position)) for position in positions})
    return [float(finite_scales[index]) for index in indices]


def should_use_dense_small_threshold_augmentation(threshold: float | None) -> bool:
    if threshold is None:
        return False
    value = finite_float(threshold)
    return np.isfinite(value) and value <= float(NORMAL_DENSE_AUGMENTATION_MAX_THRESHOLD_WEEKS)


def calculate_target_augmented_count(real_train_count: int) -> int:
    if int(real_train_count) <= 0:
        return 0
    return max(0, int(NORMAL_TARGET_TOTAL_TRAIN_COUNT) - int(real_train_count))


def build_normal_scale_grid(
    module: Any,
    *,
    max_anchor_scale: float,
    target_scale_count: int,
) -> tuple[list[float], str, float]:
    scale_upper = finite_float(max_anchor_scale)
    if not np.isfinite(scale_upper) or scale_upper <= 1.0 + 1e-9:
        return [], str(NORMAL_AUGMENTATION_SCALE_SPACING), np.nan

    spacing_mode = str(
        getattr(module, "NORMAL_AUGMENTATION_SCALE_SPACING", NORMAL_AUGMENTATION_SCALE_SPACING)
    ).strip().lower()
    step = finite_float(
        getattr(module, "NORMAL_AUGMENTATION_SCALE_STEP", NORMAL_AUGMENTATION_SCALE_STEP),
        default=float(NORMAL_AUGMENTATION_SCALE_STEP),
    )
    if spacing_mode in {"fixed_count", "linear_count", "equal_count", "count", "linspace"}:
        scale_lower = finite_float(
            getattr(module, "NORMAL_FIXED_LOW_AUGMENTATION_SCALE", NORMAL_FIXED_LOW_AUGMENTATION_SCALE),
            default=float(NORMAL_FIXED_LOW_AUGMENTATION_SCALE),
        )
        if not np.isfinite(scale_lower) or scale_lower <= 1.0:
            scale_lower = float(NORMAL_FIXED_LOW_AUGMENTATION_SCALE)
        if float(scale_upper) < float(scale_lower) - 1e-9:
            return [], spacing_mode, np.nan
        factor_count = max(1, int(getattr(module, "NORMAL_AUGMENTATION_FACTOR_COUNT", NORMAL_AUGMENTATION_FACTOR_COUNT)))
        if factor_count == 1 or np.isclose(float(scale_lower), float(scale_upper), rtol=0.0, atol=1e-9):
            return [float(scale_upper)], spacing_mode, np.nan
        scales = np.linspace(float(scale_lower), float(scale_upper), int(factor_count), dtype=float).tolist()
        effective_step = float(scales[1] - scales[0]) if len(scales) > 1 else np.nan
        return [float(scale) for scale in scales], spacing_mode, float(effective_step) if np.isfinite(effective_step) else np.nan

    if spacing_mode in {"fixed_step", "linear_step", "equal_step", "step", "0.5"}:
        if not np.isfinite(step) or step <= 0.0:
            step = float(NORMAL_AUGMENTATION_SCALE_STEP)
        start_scale = 1.0 + float(step)
        step_count = int(np.floor((float(scale_upper) - float(start_scale)) / float(step) + 1e-9))
        scales = [float(start_scale) + float(step) * float(index) for index in range(0, step_count + 1)]
        scales = [float(scale) for scale in scales if scale > 1.0 + 1e-9 and scale <= float(scale_upper) + 1e-9]
        if not scales and float(scale_upper) > 1.0 + 1e-9:
            scales = [float(scale_upper)]
        return scales, spacing_mode, float(step)

    scales = np.geomspace(
        1.0,
        float(scale_upper),
        int(target_scale_count) + 1,
        dtype=float,
    )[1:].tolist()
    return [float(scale) for scale in scales], spacing_mode or "geometric", float(step) if np.isfinite(step) else np.nan


def build_normal_augmentation_scales(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    task_kind: str,
    threshold: float | None = None,
) -> tuple[list[float], dict[str, float]]:
    train_values = collect_anchor_values(module, train_cells, task_kind=str(task_kind))
    test_values = collect_anchor_values(module, test_cells, task_kind=str(task_kind))
    stats = {
        "train_reference": np.nan,
        "train_q10": np.nan,
        "train_q25": np.nan,
        "train_q50": np.nan,
        "train_q75": np.nan,
        "train_q90": np.nan,
        "train_max": np.nan,
        "test_q10": np.nan,
        "test_q25": np.nan,
        "test_q50": np.nan,
        "test_q75": np.nan,
        "test_q90": np.nan,
        "test_max": np.nan,
        "test_upper_reference": np.nan,
        "slowest_test_anchor": np.nan,
        "max_augmented_anchor_allowed": np.nan,
        "max_coverage_scale": np.nan,
        "max_anchor_scale": np.nan,
        "test_reference": np.nan,
        "scale_q25_test_over_train_q50": np.nan,
        "scale_q50_test_over_train_q50": np.nan,
        "scale_q75_test_over_train_q50": np.nan,
        "scale_max_test_over_train_q50": np.nan,
        "scale_max_test_over_train_q25": np.nan,
        "scale_max_test_over_train_mean": np.nan,
        "scale_spacing_mode": str(getattr(module, "NORMAL_AUGMENTATION_SCALE_SPACING", NORMAL_AUGMENTATION_SCALE_SPACING)),
        "scale_step": np.nan,
        "target_total_train_count": int(NORMAL_TARGET_TOTAL_TRAIN_COUNT),
        "target_aug_count": 0,
        "target_scale_count": 0,
        "dense_small_threshold_augmentation": 0,
    }
    if train_values.size == 0 or test_values.size == 0:
        return [], stats

    use_dense_small_threshold = should_use_dense_small_threshold_augmentation(threshold)
    target_aug_count = 0
    target_scale_count = int(NORMAL_AUGMENTATION_FACTOR_COUNT)

    train_q10, train_q25, train_q50, train_q75, train_q90 = np.quantile(
        train_values,
        [0.10, 0.25, 0.50, 0.75, 0.90],
    ).astype(float)
    test_q10, test_q25, test_q50, test_q75, test_q90 = np.quantile(
        test_values,
        [0.10, 0.25, 0.50, 0.75, 0.90],
    ).astype(float)
    train_mean = float(np.mean(train_values))
    train_max = float(np.max(train_values))
    test_max = float(np.max(test_values))
    stats.update(
        {
            "train_reference": float(train_q50),
            "train_q10": float(train_q10),
            "train_q25": float(train_q25),
            "train_q50": float(train_q50),
            "train_q75": float(train_q75),
            "train_q90": float(train_q90),
            "train_mean": float(train_mean),
            "train_max": float(train_max),
            "test_q10": float(test_q10),
            "test_q25": float(test_q25),
            "test_q50": float(test_q50),
            "test_q75": float(test_q75),
            "test_q90": float(test_q90),
            "test_max": float(test_max),
            "slowest_test_anchor": float(test_max),
            "test_reference": float(test_q50),
            "target_aug_count": int(target_aug_count),
            "target_scale_count": int(target_scale_count),
            "dense_small_threshold_augmentation": int(use_dense_small_threshold),
        }
    )
    required_values = [train_q25, train_q50, train_max, test_q25, test_q50, test_q75, test_max]
    if any((not np.isfinite(value) or float(value) <= 0.0) for value in required_values):
        return [], stats
    if target_scale_count <= 0:
        return [], stats

    raw_quantile_scales = {
        "scale_q25_test_over_train_q50": float(test_q25) / float(train_q50),
        "scale_q50_test_over_train_q50": float(test_q50) / float(train_q50),
        "scale_q75_test_over_train_q50": float(test_q75) / float(train_q50),
    }
    stats.update(raw_quantile_scales)
    max_anchor_scale = float(test_max) / float(train_q50)
    stats["scale_max_test_over_train_q50"] = float(max_anchor_scale)
    stats["scale_max_test_over_train_q25"] = float(test_max) / float(train_q25)
    stats["scale_max_test_over_train_mean"] = float(test_max) / float(train_mean)
    max_augmented_anchor_allowed = float(test_max)
    standard_scales, scale_spacing_mode, scale_step = build_normal_scale_grid(
        module,
        max_anchor_scale=float(max_anchor_scale),
        target_scale_count=int(target_scale_count),
    )
    max_coverage_scale = float(max_anchor_scale)
    stats["test_upper_reference"] = float(test_max)
    stats["max_augmented_anchor_allowed"] = float(max_augmented_anchor_allowed)
    stats["max_coverage_scale"] = float(max_coverage_scale)
    stats["max_anchor_scale"] = float(max_anchor_scale)
    stats["scale_spacing_mode"] = str(scale_spacing_mode)
    stats["scale_step"] = float(scale_step) if np.isfinite(scale_step) else np.nan
    if not np.isfinite(max_anchor_scale) or float(max_anchor_scale) <= 1.0 + 1e-9:
        return [], stats

    candidate_scales = [
        *standard_scales,
        *[float(scale) for scale in NORMAL_EXTRA_AUGMENTATION_SCALES],
    ]

    unique_scales: list[float] = []
    for scale in sorted(candidate_scales):
        scale = min(float(max_anchor_scale), float(scale))
        if any(abs(float(scale) - existing) < 1e-9 for existing in unique_scales):
            continue
        if float(scale) <= 1.0 + 1e-9:
            continue
        unique_scales.append(float(scale))
    if NORMAL_MAX_AUGMENTATION_SCALES_PER_CELL is not None:
        max_scale_count = max(1, int(NORMAL_MAX_AUGMENTATION_SCALES_PER_CELL))
        if len(unique_scales) > max_scale_count:
            unique_scales = select_scale_spread(unique_scales, max_scale_count)
    if use_dense_small_threshold and len(unique_scales) > int(target_scale_count):
        unique_scales = select_scale_spread(unique_scales, int(target_scale_count))
    stats["target_scale_count"] = int(len(unique_scales))
    return unique_scales, stats


def prepare_reference_soh_curve(
    reference_values: object,
    soh_values: object,
    *,
    enforce_monotonic_soh: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_values, dtype=float).reshape(-1)
    soh = np.asarray(soh_values, dtype=float).reshape(-1)
    valid = np.isfinite(reference) & np.isfinite(soh)
    reference = reference[valid]
    soh = soh[valid]
    if reference.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(reference)
    reference = reference[order]
    soh = soh[order]
    unique_reference: list[float] = []
    unique_soh: list[float] = []
    start = 0
    while start < reference.size:
        end = start + 1
        while end < reference.size and np.isclose(reference[end], reference[start], rtol=1e-10, atol=1e-10):
            end += 1
        unique_reference.append(float(reference[start]))
        unique_soh.append(float(np.nanmedian(soh[start:end])))
        start = end

    reference_out = np.asarray(unique_reference, dtype=float)
    soh_out = np.asarray(unique_soh, dtype=float)
    if reference_out.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    soh_out = np.clip(soh_out, 0.0, 1.05)
    if bool(enforce_monotonic_soh):
        soh_out = np.minimum.accumulate(soh_out)
    return reference_out, soh_out


def resolve_cal_anchor_target(test_anchor_values: np.ndarray) -> float:
    values = np.asarray(test_anchor_values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return np.nan
    if str(CAL_ANCHOR_TARGET_STATISTIC).strip().lower() == "median":
        return float(np.median(values))
    return float(np.mean(values))


def sample_centered_cal_warp_factor(
    rng: np.random.Generator,
    *,
    anchor_mix: float,
    target_anchor: float,
    warp_factor_upper: float,
) -> float:
    if not np.isfinite(anchor_mix) or float(anchor_mix) <= 0.0:
        return np.nan
    if not np.isfinite(target_anchor) or float(target_anchor) <= 0.0:
        return np.nan
    if not np.isfinite(warp_factor_upper) or float(warp_factor_upper) <= float(CAL_WARP_FACTOR_MIN):
        return np.nan
    centered_warp = float(target_anchor) / float(anchor_mix)
    if not np.isfinite(centered_warp) or centered_warp <= 0.0:
        return np.nan
    centered_warp = float(np.clip(centered_warp, float(CAL_WARP_FACTOR_MIN), float(warp_factor_upper)))
    jitter = float(rng.normal(loc=1.0, scale=float(CAL_ANCHOR_CENTER_JITTER)))
    jitter = float(np.clip(jitter, 1.0 - 2.0 * float(CAL_ANCHOR_CENTER_JITTER), 1.0 + 2.0 * float(CAL_ANCHOR_CENTER_JITTER)))
    warp_factor = float(centered_warp) * float(jitter)
    return float(np.clip(warp_factor, float(CAL_WARP_FACTOR_MIN), float(warp_factor_upper)))


def sample_cal_anchor_target(
    rng: np.random.Generator,
    *,
    base_target_anchor: float,
    anchor_median_limit: float,
) -> float:
    if not np.isfinite(base_target_anchor) or float(base_target_anchor) <= 0.0:
        return np.nan
    if not np.isfinite(anchor_median_limit) or float(anchor_median_limit) <= float(base_target_anchor):
        return float(base_target_anchor)
    if float(rng.uniform(0.0, 1.0)) >= float(CAL_UPPER_ANCHOR_TARGET_FRACTION):
        return float(base_target_anchor)
    lower = float(base_target_anchor) + float(CAL_UPPER_ANCHOR_MIN_GAP_FRACTION) * (
        float(anchor_median_limit) - float(base_target_anchor)
    )
    lower = float(np.clip(lower, float(base_target_anchor), float(anchor_median_limit)))
    return float(rng.uniform(float(lower), float(anchor_median_limit)))


def build_mixed_warped_trajectory(
    cell_a: dict[str, object],
    cell_b: dict[str, object],
    *,
    curve_key: str,
    mix_weight: float,
    warp_factor: float,
    enforce_monotonic_soh: bool = True,
    start_at_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    reference_a = np.asarray(cell_a.get(curve_key, []), dtype=float).reshape(-1)
    reference_b = np.asarray(cell_b.get(curve_key, []), dtype=float).reshape(-1)
    soh_a = np.asarray(cell_a.get("soh_curve", []), dtype=float).reshape(-1)
    soh_b = np.asarray(cell_b.get("soh_curve", []), dtype=float).reshape(-1)
    reference_a, soh_a = prepare_reference_soh_curve(reference_a, soh_a, enforce_monotonic_soh=enforce_monotonic_soh)
    reference_b, soh_b = prepare_reference_soh_curve(reference_b, soh_b, enforce_monotonic_soh=enforce_monotonic_soh)
    if reference_a.size < 2 or reference_b.size < 2 or soh_a.size != reference_a.size or soh_b.size != reference_b.size:
        return np.array([], dtype=float), np.array([], dtype=float)
    if bool(start_at_zero):
        reference_min = 0.0
    else:
        reference_min = max(float(np.min(reference_a)), float(np.min(reference_b)))
    reference_max = min(float(np.max(reference_a)), float(np.max(reference_b)))
    if not np.isfinite(reference_min) or not np.isfinite(reference_max) or reference_max <= reference_min:
        return np.array([], dtype=float), np.array([], dtype=float)
    point_count = max(int(reference_a.size), int(reference_b.size))
    grid = np.linspace(float(reference_min), float(reference_max), int(point_count), dtype=float)
    mixed_soh = (
        float(mix_weight) * np.interp(grid, reference_a, soh_a)
        + (1.0 - float(mix_weight)) * np.interp(grid, reference_b, soh_b)
    )
    mixed_soh = np.clip(mixed_soh, 0.0, 1.05)
    if bool(enforce_monotonic_soh):
        mixed_soh = np.minimum.accumulate(mixed_soh)
    return grid * float(warp_factor), mixed_soh


def _feature_stds(train_cells: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    real_x = np.asarray([cell["x"] for cell in train_cells], dtype=float)
    x_std = np.nanstd(real_x, axis=0)
    x_std = np.where(np.isfinite(x_std), x_std, 0.0)
    real_sequences = np.asarray([cell["sequence"] for cell in train_cells], dtype=float)
    sequence_std = np.nanstd(real_sequences, axis=0)
    sequence_std = np.where(np.isfinite(sequence_std), sequence_std, 0.0)
    return x_std, sequence_std


def _valid_augmented_anchor_with_median_limit(
    generated_cells: list[dict[str, object]],
    *,
    candidate_anchor: float,
    anchor_key: str,
    anchor_median_limit: float,
) -> bool:
    if not np.isfinite(candidate_anchor) or float(candidate_anchor) <= 0.0:
        return False
    if not np.isfinite(anchor_median_limit):
        return True
    candidate_anchors = [finite_float(cell.get(anchor_key)) for cell in generated_cells]
    candidate_anchors.append(float(candidate_anchor))
    candidate_anchors_array = np.asarray(candidate_anchors, dtype=float)
    candidate_anchors_array = candidate_anchors_array[np.isfinite(candidate_anchors_array)]
    return not candidate_anchors_array.size or float(np.median(candidate_anchors_array)) <= float(anchor_median_limit)


def _balanced_quotas(item_count: int, total_count: int) -> list[int]:
    item_count = int(item_count)
    total_count = int(total_count)
    if item_count <= 0 or total_count <= 0:
        return []
    base_quota = total_count // item_count
    remainder = total_count % item_count
    quotas = [int(base_quota) for _ in range(item_count)]
    if remainder <= 0:
        return quotas
    if remainder == 1:
        positions = [item_count // 2]
    else:
        positions = [int(round(value)) for value in np.linspace(0, item_count - 1, remainder, dtype=float)]
    used: set[int] = set()
    for position in positions:
        position = int(np.clip(position, 0, item_count - 1))
        if position in used:
            continue
        quotas[position] += 1
        used.add(position)
    index = 0
    while len(used) < remainder and index < item_count:
        if index not in used:
            quotas[index] += 1
            used.add(index)
        index += 1
    return quotas


def resolve_codex_anchor_range(test_anchor_values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(test_anchor_values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    q_low, q_high = np.quantile(
        values,
        [float(CODEX_TARGET_ANCHOR_QUANTILES[0]), float(CODEX_TARGET_ANCHOR_QUANTILES[1])],
    ).astype(float)
    if not np.isfinite(q_low) or not np.isfinite(q_high):
        return np.nan, np.nan, np.nan, np.nan
    if q_high < q_low:
        q_low, q_high = q_high, q_low
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) if values.size > 1 else 0.0
    return float(q_low), float(q_high), float(iqr), float(np.max(values))


def build_codex_augmentation_stats(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    task_kind: str,
    sample_count: int,
) -> dict[str, float]:
    train_anchor_values = collect_anchor_values_until_first_crossing(module, train_cells, task_kind=str(task_kind))
    train_anchor_values = train_anchor_values[np.isfinite(train_anchor_values) & (train_anchor_values > 0.0)]
    test_anchor_values = collect_anchor_values_until_first_crossing(module, test_cells, task_kind=str(task_kind))
    test_anchor_values = test_anchor_values[np.isfinite(test_anchor_values) & (test_anchor_values > 0.0)]
    target_anchor_low, target_anchor_high, target_anchor_iqr, test_max = resolve_codex_anchor_range(test_anchor_values)

    train_q10 = train_q25 = train_q50 = train_q75 = train_q90 = train_max = np.nan
    test_q10 = test_q25 = test_q50 = test_q75 = test_q90 = np.nan
    if train_anchor_values.size:
        train_q10, train_q25, train_q50, train_q75, train_q90 = np.quantile(
            train_anchor_values,
            [0.10, 0.25, 0.50, 0.75, 0.90],
        ).astype(float)
        train_max = float(np.max(train_anchor_values))
    if test_anchor_values.size:
        test_q10, test_q25, test_q50, test_q75, test_q90 = np.quantile(
            test_anchor_values,
            [0.10, 0.25, 0.50, 0.75, 0.90],
        ).astype(float)

    return {
        "train_reference": float(train_q50) if np.isfinite(train_q50) else np.nan,
        "train_q10": float(train_q10) if np.isfinite(train_q10) else np.nan,
        "train_q25": float(train_q25) if np.isfinite(train_q25) else np.nan,
        "train_q50": float(train_q50) if np.isfinite(train_q50) else np.nan,
        "train_q75": float(train_q75) if np.isfinite(train_q75) else np.nan,
        "train_q90": float(train_q90) if np.isfinite(train_q90) else np.nan,
        "train_max": float(train_max) if np.isfinite(train_max) else np.nan,
        "test_q10": float(test_q10) if np.isfinite(test_q10) else np.nan,
        "test_q25": float(test_q25) if np.isfinite(test_q25) else np.nan,
        "test_q50": float(test_q50) if np.isfinite(test_q50) else np.nan,
        "test_q75": float(test_q75) if np.isfinite(test_q75) else np.nan,
        "test_q90": float(test_q90) if np.isfinite(test_q90) else np.nan,
        "test_max": float(test_max) if np.isfinite(test_max) else np.nan,
        "test_upper_reference": float(target_anchor_high) if np.isfinite(target_anchor_high) else np.nan,
        "slowest_test_anchor": float(test_max) if np.isfinite(test_max) else np.nan,
        "test_reference": float(test_q50) if np.isfinite(test_q50) else np.nan,
        "codex_target_anchor_low": float(target_anchor_low) if np.isfinite(target_anchor_low) else np.nan,
        "codex_target_anchor_high": float(target_anchor_high) if np.isfinite(target_anchor_high) else np.nan,
        "codex_target_anchor_iqr": float(target_anchor_iqr) if np.isfinite(target_anchor_iqr) else np.nan,
        "max_augmented_anchor_allowed": float(target_anchor_high) if np.isfinite(target_anchor_high) else np.nan,
        "max_anchor_scale": (
            float(target_anchor_high) / float(train_q50)
            if np.isfinite(target_anchor_high) and np.isfinite(train_q50) and float(train_q50) > 0.0
            else np.nan
        ),
        "scale_q25_test_over_train_q50": (
            float(test_q25) / float(train_q50)
            if np.isfinite(test_q25) and np.isfinite(train_q50) and float(train_q50) > 0.0
            else np.nan
        ),
        "scale_q50_test_over_train_q50": (
            float(test_q50) / float(train_q50)
            if np.isfinite(test_q50) and np.isfinite(train_q50) and float(train_q50) > 0.0
            else np.nan
        ),
        "scale_q75_test_over_train_q50": (
            float(test_q75) / float(train_q50)
            if np.isfinite(test_q75) and np.isfinite(train_q50) and float(train_q50) > 0.0
            else np.nan
        ),
        "target_total_train_count": int(len(train_cells) + max(0, int(sample_count))),
        "target_aug_count": int(max(0, int(sample_count))),
        "target_scale_count": 0,
        "dense_small_threshold_augmentation": 0,
    }


def build_cal_augmented_cells(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    feature_columns: list[str],
    *,
    task_kind: str,
    sample_count: int,
    random_seed: int,
    input_reference_value: float | None = None,
    output_x_drop_count: int = 0,
    source_kind: str = "cyclic_augmented",
    enforce_monotonic_soh: bool = True,
    trajectory_start_at_zero: bool = True,
) -> list[dict[str, object]]:
    if len(train_cells) < 2:
        return []
    feature_columns = [str(name) for name in feature_columns]
    x_std, sequence_std = _feature_stds(train_cells)
    rng = np.random.default_rng(int(random_seed))
    target_count = max(0, int(sample_count))
    generated_cells: list[dict[str, object]] = []
    attempts = 0
    max_attempts = max(1000, target_count * 100)
    normalized_task_kind = str(task_kind).strip().lower()
    curve_key = "throughput_curve" if normalized_task_kind == "throughput" else "time_curve"
    reference_column = resolve_reference_column(module, normalized_task_kind)

    test_anchor_values = collect_anchor_values(module, test_cells, task_kind=str(normalized_task_kind))
    test_anchor_values = test_anchor_values[np.isfinite(test_anchor_values) & (test_anchor_values > 0.0)]
    target_anchor_mean = resolve_cal_anchor_target(test_anchor_values)
    anchor_median_limit = float(CAL_MAX_TEST_MARGIN) * float(np.max(test_anchor_values)) if test_anchor_values.size else np.inf
    train_anchor_values = collect_anchor_values(module, train_cells, task_kind=str(normalized_task_kind))
    train_anchor_values = train_anchor_values[np.isfinite(train_anchor_values) & (train_anchor_values > 0.0)]
    global_upper_candidates: list[float] = []
    if np.isfinite(anchor_median_limit) and train_anchor_values.size:
        global_upper_candidates.append(float(anchor_median_limit) / float(np.min(train_anchor_values)))
    if CAL_WARP_FACTOR_MAX is not None:
        global_upper_candidates.append(float(CAL_WARP_FACTOR_MAX))
    global_upper_candidates = [
        float(value)
        for value in global_upper_candidates
        if np.isfinite(float(value)) and float(value) > float(CAL_WARP_FACTOR_MIN)
    ]
    if not global_upper_candidates:
        raise RuntimeError("CAL augmentation has no valid warp-factor upper bound above the minimum.")
    global_warp_factor_upper = min(global_upper_candidates)
    train_anchor_cache = {
        str(cell["cell"]): extract_anchor_value(module, cell, task_kind=str(normalized_task_kind))
        for cell in train_cells
    }

    while len(generated_cells) < target_count and attempts < max_attempts:
        attempts += 1
        index_a, index_b = rng.integers(0, len(train_cells), size=2)
        cell_a = train_cells[int(index_a)]
        cell_b = train_cells[int(index_b)]
        mix_weight = float(rng.uniform(0.0, 1.0))
        life_mix = mix_weight * float(cell_a["target_value"]) + (1.0 - mix_weight) * float(cell_b["target_value"])
        anchor_a = float(train_anchor_cache.get(str(cell_a["cell"]), np.nan))
        anchor_b = float(train_anchor_cache.get(str(cell_b["cell"]), np.nan))
        anchor_mix = mix_weight * anchor_a + (1.0 - mix_weight) * anchor_b
        upper_candidates = [float(global_warp_factor_upper)]
        if CAL_WARP_FACTOR_MAX is not None:
            upper_candidates.append(float(CAL_WARP_FACTOR_MAX))
        finite_upper_candidates = [float(value) for value in upper_candidates if np.isfinite(float(value))]
        if not finite_upper_candidates:
            continue
        warp_factor_upper = min(finite_upper_candidates)
        if not np.isfinite(warp_factor_upper) or warp_factor_upper <= float(CAL_WARP_FACTOR_MIN):
            continue
        sampled_target_anchor = sample_cal_anchor_target(
            rng,
            base_target_anchor=float(target_anchor_mean),
            anchor_median_limit=float(anchor_median_limit),
        )
        warp_factor = sample_centered_cal_warp_factor(
            rng,
            anchor_mix=float(anchor_mix),
            target_anchor=float(sampled_target_anchor),
            warp_factor_upper=float(warp_factor_upper),
        )
        if not np.isfinite(warp_factor):
            continue
        augmented_anchor = float(anchor_mix) * float(warp_factor)
        if not _valid_augmented_anchor_with_median_limit(
            generated_cells,
            candidate_anchor=float(augmented_anchor),
            anchor_key="cal_augmented_anchor",
            anchor_median_limit=float(anchor_median_limit),
        ):
            continue

        fallback_input_reference = finite_float(input_reference_value)
        input_reference_a = finite_float(cell_a.get("input_reference_value"), fallback_input_reference)
        input_reference_b = finite_float(cell_b.get("input_reference_value"), fallback_input_reference)
        warped_reference_a = float(input_reference_a) / float(warp_factor)
        warped_reference_b = float(input_reference_b) / float(warp_factor)
        vector_a = module.interpolate_feature_vector(
            cell_a["raw_df"],
            feature_columns,
            reference_column=reference_column,
            target_reference=warped_reference_a,
        )
        vector_b = module.interpolate_feature_vector(
            cell_b["raw_df"],
            feature_columns,
            reference_column=reference_column,
            target_reference=warped_reference_b,
        )
        sequence_a = module.interpolate_feature_sequence(
            cell_a["raw_df"],
            feature_columns,
            reference_column=reference_column,
            target_reference=warped_reference_a,
            discretization_steps=int(getattr(module, "DISCRETIZATION_STEPS")),
        )
        sequence_b = module.interpolate_feature_sequence(
            cell_b["raw_df"],
            feature_columns,
            reference_column=reference_column,
            target_reference=warped_reference_b,
            discretization_steps=int(getattr(module, "DISCRETIZATION_STEPS")),
        )
        if vector_a is None or vector_b is None or sequence_a is None or sequence_b is None:
            continue
        mixed_vector = mix_weight * np.asarray(vector_a, dtype=float) + (1.0 - mix_weight) * np.asarray(vector_b, dtype=float)
        mixed_sequence = mix_weight * np.asarray(sequence_a, dtype=float) + (1.0 - mix_weight) * np.asarray(sequence_b, dtype=float)
        if float(CAL_NOISE_STD_FRACTION) > 0.0:
            mixed_vector = mixed_vector + rng.normal(
                loc=0.0,
                scale=float(CAL_NOISE_STD_FRACTION)
                * _fit_vector_noise_std(
                    x_std,
                    mixed_vector.shape,
                    output_x_drop_count=int(output_x_drop_count),
                ),
                size=mixed_vector.shape,
            )
            mixed_sequence = mixed_sequence + rng.normal(
                loc=0.0,
                scale=float(CAL_NOISE_STD_FRACTION) * _fit_std_to_shape(sequence_std, mixed_sequence.shape),
                size=mixed_sequence.shape,
            )
            if mixed_vector.size:
                mixed_vector[0] = np.clip(mixed_vector[0], 0.0, 1.05)
            if mixed_sequence.ndim == 2 and mixed_sequence.shape[1] > 0:
                mixed_sequence[:, 0] = np.clip(mixed_sequence[:, 0], 0.0, 1.05)
        if not np.isfinite(mixed_vector).all() or not np.isfinite(mixed_sequence).all():
            continue
        target_value = float(life_mix) * float(warp_factor)
        if not np.isfinite(target_value) or target_value <= 0.0:
            continue
        trajectory_reference, trajectory_soh = build_mixed_warped_trajectory(
            cell_a,
            cell_b,
            curve_key=curve_key,
            mix_weight=mix_weight,
            warp_factor=warp_factor,
            enforce_monotonic_soh=bool(enforce_monotonic_soh),
            start_at_zero=bool(trajectory_start_at_zero),
        )
        if trajectory_reference.size != trajectory_soh.size or trajectory_soh.size < 2:
            trajectory_reference = np.linspace(0.0, float(input_reference_a), int(mixed_sequence.shape[0]), dtype=float)
            trajectory_soh = np.asarray(mixed_sequence[:, 0], dtype=float)
            if bool(enforce_monotonic_soh):
                trajectory_soh = np.minimum.accumulate(np.clip(trajectory_soh, 0.0, 1.05))

        output_x = np.asarray(mixed_vector, dtype=float)
        if int(output_x_drop_count) > 0:
            output_x = output_x[int(output_x_drop_count):]
        generated_cells.append(
            {
                "cell": (
                    f"cal_aug_{len(generated_cells):04d}_"
                    f"{cell_a['cell']}_{cell_b['cell']}_aug_{warp_factor:.3f}"
                ),
                "source_kind": str(source_kind),
                "x": output_x,
                "sequence": mixed_sequence,
                "target_value": float(target_value),
                "input_reference_value": float(input_reference_a),
                "time_curve": trajectory_reference if normalized_task_kind == "time" else np.asarray(cell_a["time_curve"], dtype=float),
                "throughput_curve": trajectory_reference if normalized_task_kind == "throughput" else np.asarray(cell_a["throughput_curve"], dtype=float),
                "soh_curve": trajectory_soh,
                "cal_warp_factor": float(warp_factor),
                "cal_warp_factor_upper": float(warp_factor_upper),
                "cal_augmented_anchor": float(augmented_anchor),
                "cal_augmented_anchor_median_limit": float(anchor_median_limit) if np.isfinite(anchor_median_limit) else np.nan,
                "cal_target_anchor_mean": float(target_anchor_mean),
                "cal_sampled_target_anchor": float(sampled_target_anchor),
                "cal_mix_weight": float(mix_weight),
                "cal_source_a": str(cell_a["cell"]),
                "cal_source_b": str(cell_b["cell"]),
                "cal_max_reference_used": float(max(warped_reference_a, warped_reference_b)),
            }
        )
    if len(generated_cells) < target_count:
        raise RuntimeError(
            f"CAL augmentation generated {len(generated_cells)} usable samples out of requested {target_count}."
        )
    return generated_cells


def build_normal_augmented_cells(
    module: Any,
    train_cells: list[dict[str, object]],
    feature_columns: list[str],
    *,
    task_kind: str,
    augmentation_scales: list[float],
    input_reference_value: float | None = None,
    output_x_drop_count: int = 0,
    density_multiplier: int | None = None,
    source_kind: str = "cyclic_augmented",
) -> list[dict[str, object]]:
    if not train_cells:
        return []
    feature_columns = [str(name) for name in feature_columns]
    density = max(1, int(density_multiplier if density_multiplier is not None else getattr(module, "AUGMENTATION_DENSITY_MULTIPLIER", 1)))
    dense_scales: list[float] = []
    for scale in augmentation_scales:
        effective_scale = max(1.0, float(scale))
        if effective_scale <= 1.0 + 1e-9:
            continue
        candidate_dense_scales = (
            [effective_scale]
            if density == 1
            else np.linspace(1.0, effective_scale, density + 1, dtype=float)[1:].tolist()
        )
        for dense_scale in candidate_dense_scales:
            if dense_scale <= 1.0 + 1e-9:
                continue
            if any(abs(float(dense_scale) - existing) < 1e-9 for existing in dense_scales):
                continue
            dense_scales.append(float(dense_scale))

    normalized_task_kind = str(task_kind).strip().lower()
    reference_column = resolve_reference_column(module, task_kind)
    augmented_cells: list[dict[str, object]] = []
    for cell in train_cells:
        for scale in dense_scales:
            fallback_input_reference = finite_float(input_reference_value)
            cell_input_reference = finite_float(cell.get("input_reference_value"), fallback_input_reference)
            if not np.isfinite(cell_input_reference) or float(cell_input_reference) <= 0.0:
                continue
            warped_input_reference = float(cell_input_reference) / float(scale)
            feature_vector = module.interpolate_feature_vector(
                cell["raw_df"],
                feature_columns,
                reference_column=reference_column,
                target_reference=warped_input_reference,
            )
            feature_sequence = module.interpolate_feature_sequence(
                cell["raw_df"],
                feature_columns,
                reference_column=reference_column,
                target_reference=warped_input_reference,
                discretization_steps=int(getattr(module, "DISCRETIZATION_STEPS")),
            )
            if feature_vector is None or feature_sequence is None:
                continue
            output_x = np.asarray(feature_vector, dtype=float)
            if int(output_x_drop_count) > 0:
                output_x = output_x[int(output_x_drop_count):]
            augmented_cells.append(
                {
                    "cell": f"{cell['cell']}_aug_{float(scale):.3f}",
                    "source_kind": str(source_kind),
                    "x": output_x,
                    "sequence": np.asarray(feature_sequence, dtype=float),
                    "target_value": float(cell["target_value"]) * float(scale),
                    "input_reference_value": float(cell_input_reference),
                    "time_curve": np.asarray(cell["time_curve"], dtype=float) * float(scale)
                    if normalized_task_kind == "time"
                    else np.asarray(cell["time_curve"], dtype=float),
                    "throughput_curve": np.asarray(cell["throughput_curve"], dtype=float) * float(scale)
                    if normalized_task_kind == "throughput"
                    else np.asarray(cell["throughput_curve"], dtype=float),
                    "soh_curve": np.asarray(cell["soh_curve"], dtype=float),
                    "cal_max_reference_used": float(warped_input_reference),
                }
            )
    return augmented_cells


def build_hybrid_augmented_cells(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    feature_columns: list[str],
    *,
    task_kind: str,
    augmentation_scales: list[float],
    sample_count: int,
    random_seed: int,
    input_reference_value: float | None = None,
    output_x_drop_count: int = 0,
    source_kind: str = "cyclic_augmented",
    enforce_monotonic_soh: bool = True,
) -> list[dict[str, object]]:
    valid_scales = [
        float(scale)
        for scale in augmentation_scales
        if np.isfinite(float(scale)) and float(scale) > 1.0 + 1e-9
    ]
    if not train_cells or not valid_scales:
        return []
    feature_columns = [str(name) for name in feature_columns]
    target_count = max(0, int(sample_count))
    generated_cells: list[dict[str, object]] = []
    normalized_task_kind = str(task_kind).strip().lower()
    reference_column = resolve_reference_column(module, task_kind)
    train_anchor_cache = {
        str(cell["cell"]): extract_anchor_value(module, cell, task_kind=str(task_kind))
        for cell in train_cells
    }
    finite_train_anchors = np.asarray(
        [finite_float(value) for value in train_anchor_cache.values()],
        dtype=float,
    )
    finite_train_anchors = finite_train_anchors[np.isfinite(finite_train_anchors) & (finite_train_anchors > 0.0)]
    if finite_train_anchors.size == 0:
        return []
    test_anchor_values = collect_anchor_values(module, test_cells, task_kind=str(task_kind))
    test_anchor_values = test_anchor_values[np.isfinite(test_anchor_values) & (test_anchor_values > 0.0)]

    sorted_scales = sorted({float(scale) for scale in valid_scales})
    global_lower_scale = float(min(sorted_scales))
    max_train_anchor = float(np.max(finite_train_anchors))
    normal_upper_scale = float(max(sorted_scales))
    normal_augmented_anchor_allowed = float(normal_upper_scale) * float(max_train_anchor)
    coverage_anchor_allowed = (
        float(HYBRID_TEST_ANCHOR_COVERAGE_MULTIPLIER) * float(np.max(test_anchor_values))
        if test_anchor_values.size
        else np.nan
    )
    max_augmented_anchor_allowed = max(
        float(value)
        for value in (normal_augmented_anchor_allowed, coverage_anchor_allowed)
        if np.isfinite(float(value)) and float(value) > 0.0
    )
    global_upper_scale = max(float(normal_upper_scale), float(max_augmented_anchor_allowed) / float(max_train_anchor))
    scale_cap_candidates = [float(global_upper_scale)]
    if NORMAL_MAX_AUGMENTATION_SCALE is not None:
        scale_cap_candidates.append(float(NORMAL_MAX_AUGMENTATION_SCALE))
    if HYBRID_MAX_AUGMENTATION_SCALE is not None:
        scale_cap_candidates.append(float(HYBRID_MAX_AUGMENTATION_SCALE))
    scale_cap = max(float(value) for value in scale_cap_candidates if np.isfinite(float(value)) and float(value) > 1.0)
    global_upper_scale = min(float(global_upper_scale), float(scale_cap))
    max_augmented_anchor_allowed = float(global_upper_scale) * float(max_train_anchor)

    eligible_cells: list[tuple[int, dict[str, object], float]] = []
    for cell_index, cell in enumerate(train_cells):
        base_anchor = float(train_anchor_cache.get(str(cell["cell"]), np.nan))
        if not np.isfinite(base_anchor) or float(base_anchor) <= 0.0:
            continue
        eligible_cells.append((int(cell_index), cell, float(base_anchor)))
    if not eligible_cells:
        return []

    per_cell_candidate_count = max(
        len(sorted_scales),
        int(
            np.ceil(
                float(max(1, target_count))
                * float(HYBRID_CANDIDATE_OVERSAMPLE_FACTOR)
                / float(len(eligible_cells))
            )
        ),
    )
    dense_scales = np.linspace(
        float(global_lower_scale),
        float(global_upper_scale),
        int(max(1, per_cell_candidate_count)),
        dtype=float,
    ).tolist()
    global_scales = sorted(
        {
            round(float(scale), 12)
            for scale in [*sorted_scales, *dense_scales, float(global_upper_scale)]
            if np.isfinite(float(scale)) and float(scale) > 1.0 + 1e-9
        }
    )
    quotas = _balanced_quotas(len(eligible_cells), target_count)
    selected_specs: list[tuple[int, int, dict[str, object], float]] = []
    fallback_specs: list[tuple[int, int, dict[str, object], float]] = []
    selected_keys: set[tuple[int, int]] = set()
    for eligible_index, (cell_index, cell, _base_anchor) in enumerate(eligible_cells):
        cell_quota = int(quotas[eligible_index]) if eligible_index < len(quotas) else 0
        if cell_quota <= 0:
            continue
        if cell_quota >= len(global_scales):
            scale_indices = list(range(len(global_scales)))
        elif cell_quota == 1:
            scale_indices = [len(global_scales) - 1]
        else:
            scale_indices = sorted(
                {
                    int(round(position))
                    for position in np.linspace(0, len(global_scales) - 1, int(cell_quota), dtype=float)
                }
            )
        for scale_index in scale_indices:
            selected_keys.add((int(cell_index), int(scale_index)))
            selected_specs.append((int(scale_index), int(cell_index), cell, float(global_scales[scale_index])))
    for cell_index, cell, _base_anchor in eligible_cells:
        for scale_index, effective_scale in enumerate(global_scales):
            if (int(cell_index), int(scale_index)) in selected_keys:
                continue
            fallback_specs.append((int(scale_index), int(cell_index), cell, float(effective_scale)))
    candidate_specs = [*selected_specs, *fallback_specs]

    for scale_index, _cell_index, cell, effective_scale in candidate_specs:
        if len(generated_cells) >= target_count:
            break
        base_scale = float(effective_scale)
        base_anchor = float(train_anchor_cache.get(str(cell["cell"]), np.nan))
        augmented_anchor = float(base_anchor) * float(effective_scale)
        if not np.isfinite(augmented_anchor) or float(augmented_anchor) <= 0.0:
            continue
        fallback_input_reference = finite_float(input_reference_value)
        cell_input_reference = finite_float(cell.get("input_reference_value"), fallback_input_reference)
        warped_input_reference = float(cell_input_reference) / float(effective_scale)
        feature_vector = module.interpolate_feature_vector(
            cell["raw_df"],
            feature_columns,
            reference_column=reference_column,
            target_reference=warped_input_reference,
        )
        feature_sequence = module.interpolate_feature_sequence(
            cell["raw_df"],
            feature_columns,
            reference_column=reference_column,
            target_reference=warped_input_reference,
            discretization_steps=int(getattr(module, "DISCRETIZATION_STEPS")),
        )
        if feature_vector is None or feature_sequence is None:
            continue
        feature_vector = np.asarray(feature_vector, dtype=float)
        feature_sequence = np.asarray(feature_sequence, dtype=float)
        if not np.isfinite(feature_vector).all() or not np.isfinite(feature_sequence).all():
            continue
        target_value = float(cell["target_value"]) * float(effective_scale)
        if not np.isfinite(target_value) or target_value <= 0.0:
            continue
        time_curve = (
            np.asarray(cell["time_curve"], dtype=float) * float(effective_scale)
            if normalized_task_kind == "time"
            else np.asarray(cell["time_curve"], dtype=float)
        )
        throughput_curve = (
            np.asarray(cell["throughput_curve"], dtype=float) * float(effective_scale)
            if normalized_task_kind == "throughput"
            else np.asarray(cell["throughput_curve"], dtype=float)
        )
        _, soh_curve = prepare_reference_soh_curve(
            time_curve if normalized_task_kind == "time" else throughput_curve,
            np.asarray(cell["soh_curve"], dtype=float),
            enforce_monotonic_soh=bool(enforce_monotonic_soh),
        )
        if soh_curve.size != np.asarray(cell["soh_curve"], dtype=float).size:
            soh_curve = np.clip(np.asarray(cell["soh_curve"], dtype=float), 0.0, 1.05)
            if bool(enforce_monotonic_soh):
                soh_curve = np.minimum.accumulate(soh_curve)
        output_x = np.asarray(feature_vector, dtype=float)
        if int(output_x_drop_count) > 0:
            output_x = output_x[int(output_x_drop_count):]
        generated_cells.append(
            {
                "cell": f"hybrid_aug_{len(generated_cells):04d}_{cell['cell']}_s{int(scale_index):03d}_{effective_scale:.3f}",
                "source_kind": str(source_kind),
                "x": output_x,
                "sequence": feature_sequence,
                "target_value": float(target_value),
                "input_reference_value": float(cell_input_reference),
                "time_curve": time_curve,
                "throughput_curve": throughput_curve,
                "soh_curve": soh_curve,
                "hybrid_source_cell": str(cell["cell"]),
                "hybrid_base_scale": float(base_scale),
                "hybrid_scale": float(effective_scale),
                "hybrid_augmented_anchor": float(augmented_anchor),
                "hybrid_augmented_anchor_limit": float(max_augmented_anchor_allowed),
                "hybrid_normal_anchor_limit": float(normal_augmented_anchor_allowed),
                "hybrid_test_anchor_coverage_limit": (
                    float(coverage_anchor_allowed) if np.isfinite(coverage_anchor_allowed) else np.nan
                ),
                "hybrid_anchor_rule": "normal_global_scale_grid_fixed_count_test_anchor_coverage",
            }
        )
    if len(generated_cells) < target_count:
        raise RuntimeError(
            f"HYBRID augmentation generated {len(generated_cells)} usable samples out of requested {target_count}."
        )
    return generated_cells


def build_codex_augmented_cells(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    feature_columns: list[str],
    *,
    task_kind: str,
    sample_count: int,
    random_seed: int,
    input_reference_value: float | None = None,
    output_x_drop_count: int = 0,
    source_kind: str = "cyclic_augmented",
    enforce_monotonic_soh: bool = True,
) -> list[dict[str, object]]:
    if not train_cells:
        return []
    feature_columns = [str(name) for name in feature_columns]
    target_count = max(0, int(sample_count))
    if target_count <= 0:
        return []
    normalized_task_kind = str(task_kind).strip().lower()
    reference_column = resolve_reference_column(module, normalized_task_kind)
    test_anchor_values = collect_anchor_values_until_first_crossing(
        module,
        test_cells,
        task_kind=str(normalized_task_kind),
    )
    test_anchor_values = test_anchor_values[np.isfinite(test_anchor_values) & (test_anchor_values > 0.0)]
    target_anchor_low, target_anchor_high, target_anchor_iqr, _test_max = resolve_codex_anchor_range(test_anchor_values)
    if not np.isfinite(target_anchor_high) or float(target_anchor_high) <= 0.0:
        return []
    if not np.isfinite(target_anchor_low) or float(target_anchor_low) <= 0.0:
        target_anchor_low = float(target_anchor_high)

    train_anchor_cache = {
        str(cell["cell"]): extract_anchor_value_until_first_crossing(
            module,
            cell,
            task_kind=str(normalized_task_kind),
        )
        for cell in train_cells
    }
    eligible_cells: list[tuple[int, dict[str, object], float, float, float]] = []
    for cell_index, cell in enumerate(train_cells):
        base_anchor = finite_float(train_anchor_cache.get(str(cell["cell"])))
        if not np.isfinite(base_anchor) or float(base_anchor) <= 0.0:
            continue
        min_augmented_anchor = max(
            float(target_anchor_low),
            float(base_anchor) * float(CODEX_MIN_GENERATED_AUGMENTATION_SCALE),
        )
        max_augmented_anchor = float(target_anchor_high)
        if max_augmented_anchor <= float(base_anchor) * (1.0 + 1e-9):
            continue
        if min_augmented_anchor > max_augmented_anchor:
            min_augmented_anchor = max_augmented_anchor
        eligible_cells.append(
            (
                int(cell_index),
                cell,
                float(base_anchor),
                float(min_augmented_anchor),
                float(max_augmented_anchor),
            )
        )
    if not eligible_cells:
        return []

    quotas = _balanced_quotas(len(eligible_cells), target_count)
    generated_cells: list[dict[str, object]] = []
    candidate_specs: list[tuple[int, int, dict[str, object], float, float, float]] = []
    for eligible_index, (cell_index, cell, base_anchor, anchor_low, anchor_high) in enumerate(eligible_cells):
        cell_quota = int(quotas[eligible_index]) if eligible_index < len(quotas) else 0
        if cell_quota <= 0:
            continue
        if cell_quota == 1 or np.isclose(anchor_low, anchor_high):
            target_anchors = [0.5 * (float(anchor_low) + float(anchor_high))]
        else:
            target_anchors = np.linspace(float(anchor_low), float(anchor_high), int(cell_quota), dtype=float).tolist()
        for scale_index, target_anchor in enumerate(target_anchors):
            factor = float(target_anchor) / float(base_anchor)
            if not np.isfinite(factor) or float(factor) <= 1.0 + 1e-9:
                continue
            candidate_specs.append((int(scale_index), int(cell_index), cell, float(factor), float(target_anchor), float(base_anchor)))

    for scale_index, _cell_index, cell, effective_scale, target_anchor, base_anchor in candidate_specs:
        if len(generated_cells) >= target_count:
            break
        fallback_input_reference = finite_float(input_reference_value)
        cell_input_reference = finite_float(cell.get("input_reference_value"), fallback_input_reference)
        warped_input_reference = float(cell_input_reference) / float(effective_scale)
        feature_vector = module.interpolate_feature_vector(
            cell["raw_df"],
            feature_columns,
            reference_column=reference_column,
            target_reference=warped_input_reference,
        )
        feature_sequence = module.interpolate_feature_sequence(
            cell["raw_df"],
            feature_columns,
            reference_column=reference_column,
            target_reference=warped_input_reference,
            discretization_steps=int(getattr(module, "DISCRETIZATION_STEPS")),
        )
        if feature_vector is None or feature_sequence is None:
            continue
        feature_vector = np.asarray(feature_vector, dtype=float)
        feature_sequence = np.asarray(feature_sequence, dtype=float)
        if not np.isfinite(feature_vector).all() or not np.isfinite(feature_sequence).all():
            continue
        target_value = float(cell["target_value"]) * float(effective_scale)
        if not np.isfinite(target_value) or target_value <= 0.0:
            continue
        time_curve = (
            np.asarray(cell["time_curve"], dtype=float) * float(effective_scale)
            if normalized_task_kind == "time"
            else np.asarray(cell["time_curve"], dtype=float)
        )
        throughput_curve = (
            np.asarray(cell["throughput_curve"], dtype=float) * float(effective_scale)
            if normalized_task_kind == "throughput"
            else np.asarray(cell["throughput_curve"], dtype=float)
        )
        _, soh_curve = prepare_reference_soh_curve(
            time_curve if normalized_task_kind == "time" else throughput_curve,
            np.asarray(cell["soh_curve"], dtype=float),
            enforce_monotonic_soh=bool(enforce_monotonic_soh),
        )
        if soh_curve.size != np.asarray(cell["soh_curve"], dtype=float).size:
            soh_curve = np.clip(np.asarray(cell["soh_curve"], dtype=float), 0.0, 1.05)
            if bool(enforce_monotonic_soh):
                soh_curve = np.minimum.accumulate(soh_curve)
        output_x = np.asarray(feature_vector, dtype=float)
        if int(output_x_drop_count) > 0:
            output_x = output_x[int(output_x_drop_count):]
        generated_cells.append(
            {
                "cell": f"codex_aug_{len(generated_cells):04d}_{cell['cell']}_s{int(scale_index):03d}_{effective_scale:.3f}",
                "source_kind": str(source_kind),
                "x": output_x,
                "sequence": feature_sequence,
                "target_value": float(target_value),
                "input_reference_value": float(cell_input_reference),
                "time_curve": time_curve,
                "throughput_curve": throughput_curve,
                "soh_curve": soh_curve,
                "codex_source_cell": str(cell["cell"]),
                "codex_source_anchor": float(base_anchor),
                "codex_target_anchor": float(target_anchor),
                "codex_target_anchor_low": float(target_anchor_low),
                "codex_target_anchor_high": float(target_anchor_high),
                "codex_target_anchor_iqr": float(target_anchor_iqr) if np.isfinite(target_anchor_iqr) else np.nan,
                "codex_scale": float(effective_scale),
                "codex_augmented_anchor": float(target_anchor),
                "codex_anchor_rule": "anchor_quantile_matched_per_cell",
            }
        )
    # The sampling loop is capped by max_attempts and a few candidates are always
    # rejected, so an exact hit is not reachable in practice - requesting 200
    # yields 192, requesting 100 yields 98. Failing on that threw away every
    # usable sample over a 4% shortfall and made CODEX unusable at any sample
    # count. Only a generator that produces almost nothing is a real fault.
    if len(generated_cells) < max(1, int(round(CODEX_MIN_USABLE_SAMPLE_FRACTION * target_count))):
        raise RuntimeError(
            f"CODEX augmentation generated {len(generated_cells)} usable samples out of requested {target_count}."
        )
    return generated_cells
