from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core_functions import augmentation_run_config as aug_cfg

VALID_AUGMENTATION_METHODS = (
    "feature_log2",
    "matching",
    "matching_structured",
)
NORMAL_MAX_AUGMENTATION_SCALE: float | None = None
NORMAL_STANDARD_MAX_AUGMENTATION_SCALE = 3.0
NORMAL_FIXED_LOW_AUGMENTATION_SCALE = 1.2
NORMAL_LOW_SCALE_AUGMENTATION_SCALES: tuple[float, ...] = (
    1.02,
    1.05,
    1.08,
    1.10,
    1.15,
    1.20,
    1.30,
    1.40,
)
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
FEATURE_LOG_TEST_CYCLE_LIFE_COVERAGE_MULTIPLIER = 1.0
FEATURE_LOG_PREDICTION_SIGMA_MULTIPLIER = 3.0
FEATURE_LOG2_SCALE_GRID_POWER = 3.0
FEATURE_LOG2_CELL_FEATURE_SCALE_KEY = "_feature_log2_feature_coverage_scale"
FEATURE_LOG2_CELL_SCALES_KEY = "_feature_log2_augmentation_scales"
FEATURE_LOG_REGRESSION_DIAGNOSTIC_KEY = "_feature_log_regression_diagnostic"
FEATURE_LOG_MODEL_INPUT_REGRESSION_DIAGNOSTIC_KEY = "_feature_log_model_input_regression_diagnostic"
FEATURE_LOG2_LIFETIME_DIAGNOSTIC_VECTOR_KEY = "_feature_log2_lifetime_diagnostic_x"
PROCESS_FEATURE_VECTOR_KEY = "_process_feature_vector"
PROCESS_FEATURE_SEQUENCE_KEY = "_process_feature_sequence"
PROCESS_FEATURE_COLUMNS_KEY = "_process_feature_columns"


def preserve_process_features(
    cell: dict[str, object], feature_columns: list[str] | tuple[str, ...],
) -> dict[str, object]:
    """Keep the SOH-inclusive representation before model-only feature filtering."""
    copied = dict(cell)
    columns = [str(name) for name in feature_columns]
    vector = np.asarray(copied.get("x", []), dtype=float).reshape(-1)
    if PROCESS_FEATURE_VECTOR_KEY not in copied and vector.size == len(columns):
        copied[PROCESS_FEATURE_VECTOR_KEY] = vector.copy()
        copied[PROCESS_FEATURE_COLUMNS_KEY] = tuple(columns)
    sequence = np.asarray(copied.get("sequence", []), dtype=float)
    if (PROCESS_FEATURE_SEQUENCE_KEY not in copied and sequence.ndim == 2
            and sequence.shape[1] == len(columns)):
        copied[PROCESS_FEATURE_SEQUENCE_KEY] = sequence.copy()
    return copied


def process_feature_values(
    cell: dict[str, object], feature_names: list[str] | tuple[str, ...],
    fallback_columns: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Read named process features, falling back to the model vector for datarun cells."""
    names = [str(name).strip().lower() for name in feature_names]
    vector = np.asarray(
        cell.get(PROCESS_FEATURE_VECTOR_KEY, cell.get("x", [])), dtype=float,
    ).reshape(-1)
    columns = [str(name) for name in cell.get(
        PROCESS_FEATURE_COLUMNS_KEY, fallback_columns)]
    if vector.size != len(columns):
        vector = np.asarray(cell.get("x", []), dtype=float).reshape(-1)
        columns = [str(name) for name in fallback_columns]
    lookup = {name.strip().lower(): index for index, name in enumerate(columns)}
    if vector.size != len(columns) or any(name not in lookup for name in names):
        return np.asarray([], dtype=float)
    return vector[np.asarray([lookup[name] for name in names], dtype=int)]

def supported_augmentation_methods(*methods: str) -> frozenset[str]:
    """Declare the subset of VALID_AUGMENTATION_METHODS an engine implements.

    Engines legitimately support different subsets, so each keeps its own set
    rather than accepting every valid name. Validating here means a typo or a
    method that no longer exists fails at import instead of falling through to
    the wrong augmentation path at run time.
    """
    selected = frozenset(str(method).strip().lower() for method in methods)
    unknown = sorted(selected - set(VALID_AUGMENTATION_METHODS))
    if unknown:
        raise ValueError(
            f"Unknown augmentation method(s) {unknown}. "
            f"Valid methods are {list(VALID_AUGMENTATION_METHODS)}."
        )
    return selected

def finite_float(value: object, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def resolve_augmented_targets(
    cell: dict[str, object],
    *,
    task_kind: str,
    scale: float,
) -> tuple[float, float]:
    base_week = finite_float(cell.get("target_week_value", cell.get("target_value")))
    base_throughput = finite_float(cell.get("target_throughput_value"))
    effective_scale = float(scale)
    return float(base_week) * effective_scale, float(base_throughput) * effective_scale


def augmentation_scale_summary_text(stats: dict[str, object] | None) -> str:
    """Describe only the scale policy that actually generated the augmentation."""
    values = dict(stats or {})
    method = str(values.get("augmentation_method", "normal") or "normal").strip().lower()
    if method in {"feature_log", "feature_log2"}:
        return (
            f"{method} augmentation stats: "
            f"feature floor={finite_float(values.get('feature_log2_feature_floor_scale', values.get('feature_coverage_scale'))):.3f}, "
            f"lifetime scale={finite_float(values.get('feature_log2_lifetime_scale', values.get('lifetime_coverage_scale'))):.3f}, "
            f"max factor={finite_float(values.get('max_augmentation_scale')):.3f}, "
            f"spacing={values.get('scale_spacing_mode', 'n/a')}"
        )
    if method.startswith("matching"):
        return (
            f"{method} augmentation stats: "
            f"candidates={int(finite_float(values.get('matching_candidate_count'), default=0.0))}, "
            f"median factor={finite_float(values.get('matching_scale_median')):.3f}, "
            f"max factor={finite_float(values.get('matching_scale_max')):.3f}"
        )
    return (
        f"Normal augmentation stats: test median={finite_float(values.get('test_q50')):.3f}, "
        f"test max={finite_float(values.get('test_max')):.3f}, "
        f"train q25={finite_float(values.get('train_q25')):.3f}, "
        f"train q50={finite_float(values.get('train_q50')):.3f}, "
        f"train q75={finite_float(values.get('train_q75')):.3f}"
    )

def build_feature_log_regression_diagnostic(
    *,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    train_row_indices: object,
    test_row_indices: object,
    finite_train: object,
    finite_test: object,
    x_fit: object,
    y_fit_log: object,
    train_pred_log: object,
    x_eval: object,
    test_pred_log: object,
    x_mean: object,
    x_std: object,
    coefficients: object,
    residual_sigma: float,
) -> dict[str, object]:
    """Keep the exact fitted rows needed to audit the lifetime-scale regression."""
    train_rows = np.asarray(train_row_indices, dtype=int).reshape(-1)
    test_rows = np.asarray(test_row_indices, dtype=int).reshape(-1)
    used_train_rows = train_rows[np.asarray(finite_train, dtype=bool).reshape(-1)]
    used_test_rows = test_rows[np.asarray(finite_test, dtype=bool).reshape(-1)]

    def cell_label(cell: dict[str, object]) -> str:
        return str(cell.get("cell", cell.get("Cell", cell.get("cell_id", ""))))

    test_y_log = []
    for index in used_test_rows:
        target = finite_float(test_cells[index].get("target_value"))
        test_y_log.append(float(np.log(target)) if target > 0.0 else np.nan)

    return {
        "feature_names": [],
        "train_cells": [cell_label(train_cells[index]) for index in used_train_rows],
        "test_cells": [cell_label(test_cells[index]) for index in used_test_rows],
        "train_x_log_abs": np.asarray(x_fit, dtype=float).tolist(),
        "train_y_log": np.asarray(y_fit_log, dtype=float).reshape(-1).tolist(),
        "train_pred_log": np.asarray(train_pred_log, dtype=float).reshape(-1).tolist(),
        "test_x_log_abs": np.asarray(x_eval, dtype=float).tolist(),
        "test_y_log": test_y_log,
        "test_pred_log": np.asarray(test_pred_log, dtype=float).reshape(-1).tolist(),
        "x_mean": np.asarray(x_mean, dtype=float).reshape(-1).tolist(),
        "x_std": np.asarray(x_std, dtype=float).reshape(-1).tolist(),
        "coefficients": np.asarray(coefficients, dtype=float).reshape(-1).tolist(),
        "residual_sigma": float(residual_sigma),
        "ridge_penalty": 1e-3,
    }


def filter_feature_log2_augmented_cells_by_lifetime_prediction_limit(
    augmented_cells: list[dict[str, object]],
    feature_columns: list[str],
    augmentation_stats: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Apply the training-only prediction-multiplier limit after generation."""
    generated_count = len(augmented_cells)
    prediction_multiplier = finite_float(
        aug_cfg.FEATURE_LOG2_LIFETIME_PREDICTION_MULTIPLIER)
    filter_stats: dict[str, object] = {
        "feature_log2_lifetime_prediction_filter_applied": False,
        "feature_log2_lifetime_prediction_generated_count": int(generated_count),
        "feature_log2_lifetime_prediction_kept_count": int(generated_count),
        "feature_log2_lifetime_prediction_rejected_count": 0,
        "feature_log2_lifetime_limit_mode": "linear_prediction_multiplier",
        "feature_log2_lifetime_prediction_multiplier": float(prediction_multiplier),
    }
    policy = str(aug_cfg.FEATURE_LOG2_MAX_SCALE_POLICY).strip().lower()
    if policy != "max_var_delta_q_and_lifetime_prediction_multiplier" or not augmented_cells:
        return list(augmented_cells), filter_stats
    if prediction_multiplier <= 1.0:
        raise ValueError("FEATURE_LOG2_LIFETIME_PREDICTION_MULTIPLIER must be greater than 1.")

    resolved_feature_columns = [str(name) for name in feature_columns]
    vector_lengths = {
        int(np.asarray(cell.get("x", []), dtype=float).size)
        for cell in augmented_cells
        if np.asarray(cell.get("x", []), dtype=float).size
    }
    # interpolate_feature_vector prepends SOH even though augmentation builders
    # receive only the mean/variance feature names.
    if (
        resolved_feature_columns
        and resolved_feature_columns[0].strip().lower() != "soh"
        and vector_lengths == {len(resolved_feature_columns) + 1}
    ):
        resolved_feature_columns.insert(0, "soh")
    column_indices = {
        str(name).strip().lower(): index
        for index, name in enumerate(resolved_feature_columns)
    }
    stats = dict(augmentation_stats or {})
    candidates = (
        ("model_input", stats.get(FEATURE_LOG_MODEL_INPUT_REGRESSION_DIAGNOSTIC_KEY)),
        ("augmentation_reference", stats.get(FEATURE_LOG_REGRESSION_DIAGNOSTIC_KEY)),
    )
    selected = None
    invalid_sources: list[str] = []
    for source, candidate in candidates:
        if not isinstance(candidate, dict) or not candidate:
            invalid_sources.append(f"{source}:missing")
            continue
        diagnostic_basis = str(candidate.get("diagnostic_basis", "")).strip()
        input_reference = finite_float(candidate.get("input_reference_value"))
        generated_reference = finite_float(candidate.get("generated_input_reference_value"))
        if (
            source == "augmentation_reference"
            and np.isfinite(input_reference)
            and np.isfinite(generated_reference)
            and not np.isclose(input_reference, generated_reference)
        ):
            invalid_sources.append(f"{source}:reference_mismatch")
            continue
        names = [str(name) for name in candidate.get("feature_names", [])]
        if diagnostic_basis == "per_cell_augmentation_reference_soh":
            feature_indices = list(range(len(names)))
        else:
            try:
                feature_indices = [column_indices[name.strip().lower()] for name in names]
            except KeyError:
                invalid_sources.append(f"{source}:unresolved_features")
                continue
        x_mean = np.asarray(candidate.get("x_mean", []), dtype=float).reshape(-1)
        x_std = np.asarray(candidate.get("x_std", []), dtype=float).reshape(-1)
        coefficients = np.asarray(candidate.get("coefficients", []), dtype=float).reshape(-1)
        feature_count = len(feature_indices)
        if (
            feature_count == 0
            or x_mean.size != feature_count
            or x_std.size != feature_count
            or coefficients.size != feature_count + 1
            or not np.isfinite(x_mean).all()
            or not np.isfinite(x_std).all()
            or not np.isfinite(coefficients).all()
        ):
            invalid_sources.append(f"{source}:incomplete_fit")
            continue
        selected = (diagnostic_basis or source, feature_indices, x_mean, x_std, coefficients)
        break
    if selected is None:
        raise RuntimeError(
            "feature_log2 lifetime prediction-multiplier filtering has no valid training-only "
            "regression diagnostic "
            f"({', '.join(invalid_sources)})."
        )
    diagnostic_source, feature_indices, x_mean, x_std, coefficients = selected
    x_std = np.where(x_std > 1e-12, x_std, 1.0)

    kept_cells: list[dict[str, object]] = []
    rejected_cells: list[str] = []
    for cell in augmented_cells:
        vector_source = (
            cell.get(FEATURE_LOG2_LIFETIME_DIAGNOSTIC_VECTOR_KEY, cell.get("x", []))
            if diagnostic_source == "per_cell_augmentation_reference_soh"
            else cell.get("x", []))
        vector = np.asarray(vector_source, dtype=float).reshape(-1)
        target = finite_float(cell.get("target_value"))
        if vector.size <= max(feature_indices) or target <= 0.0:
            rejected_cells.append(str(cell.get("cell", "")))
            continue
        x_log = np.log(np.maximum(np.abs(vector[np.asarray(feature_indices, dtype=int)]), 1e-30))
        upper_log = float(coefficients[0] + np.dot(coefficients[1:], (x_log - x_mean) / x_std))
        upper_log += float(np.log(prediction_multiplier))
        if not np.isfinite(upper_log) or float(np.log(target)) > upper_log + 1e-12:
            rejected_cells.append(str(cell.get("cell", "")))
            continue
        copied = dict(cell)
        copied["feature_log2_lifetime_upper_log"] = float(upper_log)
        copied["feature_log2_lifetime_margin_log"] = float(upper_log - np.log(target))
        copied["feature_log2_lifetime_prediction_multiplier"] = float(prediction_multiplier)
        kept_cells.append(copied)

    filter_stats.update({
        "feature_log2_lifetime_prediction_filter_applied": True,
        "feature_log2_lifetime_prediction_diagnostic_source": str(diagnostic_source),
        "feature_log2_lifetime_prediction_kept_count": int(len(kept_cells)),
        "feature_log2_lifetime_prediction_rejected_count": int(len(rejected_cells)),
        "feature_log2_lifetime_prediction_rejected_cells": " ".join(rejected_cells),
    })
    return kept_cells, filter_stats


def model_input_usable(cell: dict[str, object]) -> bool:
    if cell.get("model_input_usable") is False:
        return False
    x = np.asarray(cell.get("x", []), dtype=float)
    sequence = np.asarray(cell.get("sequence", []), dtype=float)
    return bool(x.size and sequence.size and np.isfinite(x).all() and np.isfinite(sequence).all())


def model_ready_cells(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    return [cell for cell in cells if model_input_usable(cell)]


def model_ready_cells_reaching_input_before_eol(
    cells: list[dict[str, object]], *, input_reference_value: float,
) -> list[dict[str, object]]:
    ready_cells = model_ready_cells(cells)
    if not bool(aug_cfg.EXCLUDE_REAL_TRAINING_CELLS_REACHING_EOL_BEFORE_INPUT):
        return ready_cells
    reference = float(input_reference_value)
    return [cell for cell in ready_cells
        if np.isfinite(float(cell["target_value"])) and float(cell["target_value"]) > reference]


def add_soh_coverage_feature(input_feature_columns: list[str], feature_columns: list[str]) -> list[str]:
    """Feature list whose test-set floor the per-cell max-scale cap (Limit 1) has to reach.

    FEATURE_LOG2_FEATURE_FLOOR_POLICY picks the source list. Under "driver_features_only" this is
    the driver list and the loop below is the one that has always run. The widened policies read
    the model input list instead, which every caller already passes as the first argument.

    Two invariants the widened policies have to keep. Every name here is looked up as a raw_df
    column by interpolate_feature_vector (eol_mlp_common.py:141), which has no guard and returns
    None for the whole vector if any one feature is unreadable at the warped reference -- so an
    unknown name is a hard failure, not a skipped feature. And Limit 2's ridge is driven by
    var_delta_Q, so the driver features are unioned back in to keep the two limits on the same
    footing.
    """
    policy = aug_cfg.feature_log2_feature_floor_policy()
    driver_columns = list(feature_columns)
    source_columns = driver_columns if policy == "driver_features_only" else list(input_feature_columns)
    drop_bands = policy == "all_model_features_no_bands"
    result: list[str] = []
    seen: set[str] = set()
    for name in source_columns:
        text = str(name)
        key = text.strip().lower()
        if drop_bands and key.rsplit("_", 1)[-1] in {"low", "mid", "high"}:
            continue
        if key and key != "soh" and key not in seen:
            result.append(text)
            seen.add(key)
    if policy == "driver_features_only":
        return result
    for name in driver_columns:
        text = str(name)
        key = text.strip().lower()
        if key and key != "soh" and key not in seen:
            result.append(text)
            seen.add(key)
    if not result:
        raise ValueError(
            f"FEATURE_LOG2_FEATURE_FLOOR_POLICY={policy!r} left the coverage floor empty. "
            f"Model inputs were {list(input_feature_columns)!r} and drivers {driver_columns!r}; "
            "an empty floor silently disables augmentation instead of failing."
        )
    return result

def select_best_single_log_r2_feature(
    train_cells: list[dict[str, object]],
    input_feature_columns: list[str],
    candidate_feature_columns: list[str],
) -> tuple[list[str], dict[str, object]]:
    input_columns = [str(name) for name in input_feature_columns]
    candidate_columns = [str(name) for name in candidate_feature_columns]
    stats: dict[str, object] = {
        "feature_selection_rule": "best_train_r2_single_log_abs_feature",
        "feature_selection_candidate_features": ",".join(candidate_columns),
        "feature_selection_selected_feature": "",
        "feature_selection_selected_r2": np.nan,
        "feature_selection_selected_n": 0,
        "feature_selection_candidate_r2": "",
    }
    ranked: list[tuple[float, str, int]] = []
    r2_by_feature: list[str] = []
    for feature_name in candidate_columns:
        if feature_name not in input_columns:
            continue
        feature_index = int(input_columns.index(feature_name))
        x_values: list[float] = []
        y_values: list[float] = []
        for cell in train_cells:
            feature_vector = np.asarray(cell.get("x", []), dtype=float).reshape(-1)
            if feature_vector.size <= feature_index:
                continue
            feature_value = finite_float(feature_vector[feature_index])
            target_value = finite_float(cell.get("target_value"))
            if not (np.isfinite(feature_value) and np.isfinite(target_value) and target_value > 0.0):
                continue
            x_values.append(float(np.log(max(abs(feature_value), 1e-30))))
            y_values.append(float(np.log(max(target_value, 1e-30))))
        if len(x_values) < 2:
            continue
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]
        if x_arr.size < 2 or float(np.var(x_arr)) <= 1e-24 or float(np.var(y_arr)) <= 1e-24:
            continue
        x_centered = x_arr - float(np.mean(x_arr))
        y_centered = y_arr - float(np.mean(y_arr))
        slope = float(np.sum(x_centered * y_centered) / max(float(np.sum(x_centered * x_centered)), 1e-30))
        intercept = float(np.mean(y_arr) - slope * np.mean(x_arr))
        predicted = slope * x_arr + intercept
        ss_res = float(np.sum((y_arr - predicted) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        if ss_tot <= 1e-30:
            continue
        r2_value = float(1.0 - ss_res / ss_tot)
        ranked.append((r2_value, feature_name, int(x_arr.size)))
        r2_by_feature.append(f"{feature_name}:{r2_value:.6g}")
    if not ranked:
        return [], stats
    ranked.sort(key=lambda item: (-float(item[0]), str(item[1])))
    best_r2, best_feature, best_count = ranked[0]
    stats["feature_selection_selected_feature"] = str(best_feature)
    stats["feature_selection_selected_r2"] = float(best_r2)
    stats["feature_selection_selected_n"] = int(best_count)
    stats["feature_selection_candidate_r2"] = ",".join(r2_by_feature)
    return [str(best_feature)], stats

def select_training_pearson_augmentation_features(
    train_cells: list[dict[str, object]],
    input_feature_columns: list[str],
    candidate_feature_columns: list[str],
    *,
    always_feature_columns: tuple[str, ...] = ("var_delta_Q",),
    min_abs_pearson: float | None = 0.80,
) -> tuple[list[str], dict[str, object]]:
    input_columns = [str(name) for name in input_feature_columns]
    candidates = [str(name) for name in candidate_feature_columns if str(name) in input_columns]
    always = {str(name).strip().lower() for name in always_feature_columns}
    selected: list[str] = []
    correlations: list[str] = []
    selected_correlations: list[float] = []
    selected_counts: list[int] = []
    for feature_name in candidates:
        mandatory = feature_name.strip().lower() in always
        pairs = []
        for cell in train_cells:
            values = process_feature_values(cell, [feature_name], input_columns)
            if values.size == 1:
                pairs.append((finite_float(values[0]), finite_float(cell.get("target_value"))))
        valid = [(x, y) for x, y in pairs if np.isfinite(x) and np.isfinite(y) and y > 0.0]
        rho = np.nan
        if not mandatory and min_abs_pearson is not None and len(valid) >= 3:
            x_values, y_values = (np.asarray(values, dtype=float) for values in zip(*valid))
            x_values = np.log(np.maximum(np.abs(x_values), 1e-30))
            y_values = np.log(np.maximum(y_values, 1e-30))
            if np.var(x_values) > 1e-24 and np.var(y_values) > 1e-24:
                rho = float(np.corrcoef(x_values, y_values)[0, 1])
        correlations.append(f"{feature_name}:{rho:.6g}" if np.isfinite(rho) else f"{feature_name}:ignored")
        if mandatory or (min_abs_pearson is not None and np.isfinite(rho) and abs(rho) > float(min_abs_pearson)):
            selected.append(feature_name)
            if np.isfinite(rho):
                selected_correlations.append(float(rho))
            selected_counts.append(len(valid))
    missing_always = sorted(always - {name.strip().lower() for name in selected})
    if missing_always:
        raise RuntimeError(f"Missing mandatory augmentation features: {missing_always}")
    pearson_enabled = min_abs_pearson is not None and any(name.strip().lower() not in always for name in candidates)
    return selected, {
        "feature_selection_rule": "mandatory_features_plus_train_log_abs_pearson" if pearson_enabled else "fixed_mandatory_features_no_pearson",
        "feature_selection_candidate_features": ",".join(candidates),
        "feature_selection_selected_feature": ",".join(selected),
        "feature_selection_selected_r2": np.nan,
        "feature_selection_selected_n": min(selected_counts) if selected_counts else 0,
        "feature_selection_candidate_r2": "",
        "feature_selection_candidate_pearson": ",".join(correlations),
        "feature_selection_correlation_space": "log_abs_feature_vs_log_eol" if pearson_enabled else "disabled",
        "feature_selection_min_abs_pearson": float(min_abs_pearson) if pearson_enabled else np.nan,
        "feature_selection_selected_abs_pearson_min": min((abs(value) for value in selected_correlations), default=np.nan),
    }

def select_fixed_single_log_r2_feature(
    train_cells: list[dict[str, object]],
    input_feature_columns: list[str],
    candidate_feature_columns: list[str],
    preferred_feature_name: str,
) -> tuple[list[str], dict[str, object]]:
    input_columns = [str(name) for name in input_feature_columns]
    candidate_columns = [str(name) for name in candidate_feature_columns]
    preferred_normalized = str(preferred_feature_name).strip().lower()
    stats: dict[str, object] = {
        "feature_selection_rule": f"fixed_{preferred_normalized}_single_log_abs_feature",
        "feature_selection_candidate_features": ",".join(candidate_columns),
        "feature_selection_selected_feature": "",
        "feature_selection_selected_r2": np.nan,
        "feature_selection_selected_n": 0,
        "feature_selection_candidate_r2": "",
    }
    selected_feature = ""
    for feature_name in candidate_columns:
        if str(feature_name).strip().lower() == preferred_normalized:
            selected_feature = str(feature_name)
            break
    if not selected_feature:
        return [], stats
    input_lookup = {str(name).strip().lower(): str(name) for name in input_columns}
    input_feature_name = input_lookup.get(preferred_normalized, "")
    if not input_feature_name:
        return [], stats

    _best_features, candidate_stats = select_best_single_log_r2_feature(
        train_cells,
        input_columns,
        candidate_columns,
    )
    stats["feature_selection_candidate_r2"] = candidate_stats.get("feature_selection_candidate_r2", "")

    feature_index = int(input_columns.index(input_feature_name))
    x_values: list[float] = []
    y_values: list[float] = []
    for cell in train_cells:
        feature_vector = np.asarray(cell.get("x", []), dtype=float).reshape(-1)
        if feature_vector.size <= feature_index:
            continue
        feature_value = finite_float(feature_vector[feature_index])
        target_value = finite_float(cell.get("target_value"))
        if not (np.isfinite(feature_value) and np.isfinite(target_value) and target_value > 0.0):
            continue
        x_values.append(float(np.log(max(abs(feature_value), 1e-30))))
        y_values.append(float(np.log(max(target_value, 1e-30))))

    stats["feature_selection_selected_feature"] = str(selected_feature)
    stats["feature_selection_selected_n"] = int(len(x_values))
    if len(x_values) >= 2:
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]
        if x_arr.size >= 2 and float(np.var(x_arr)) > 1e-24 and float(np.var(y_arr)) > 1e-24:
            x_centered = x_arr - float(np.mean(x_arr))
            y_centered = y_arr - float(np.mean(y_arr))
            slope = float(np.sum(x_centered * y_centered) / max(float(np.sum(x_centered * x_centered)), 1e-30))
            intercept = float(np.mean(y_arr) - slope * np.mean(x_arr))
            predicted = slope * x_arr + intercept
            ss_res = float(np.sum((y_arr - predicted) ** 2))
            ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
            if ss_tot > 1e-30:
                stats["feature_selection_selected_r2"] = float(1.0 - ss_res / ss_tot)
                stats["feature_selection_selected_n"] = int(x_arr.size)
    return [str(selected_feature)], stats

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

def _iter_source_rows(source_rows: Any) -> Any:
    if hasattr(source_rows, "iterrows"):
        for _, source_row in source_rows.iterrows():
            yield source_row
        return
    for source_row in source_rows:
        yield source_row

def build_reference_axis_warped_raw_df(
    raw_df: Any,
    *,
    reference_column: str,
    scale: float,
) -> Any:
    warped_df = raw_df.copy()
    linked_columns: list[str] = []
    for candidate in (
        str(reference_column),
        "weeks",
        "throughput_cum",
        "throughput",
    ):
        if candidate in warped_df and candidate not in linked_columns:
            linked_columns.append(candidate)
    for column_name in linked_columns:
        warped_df[str(column_name)] = (
            pd.to_numeric(warped_df[str(column_name)], errors="coerce").astype(float) * float(scale)
        )
    return warped_df

def select_warped_reference_row(
    raw_df: Any,
    *,
    reference_column: str,
    target_reference: float,
    feature_columns: list[str],
    scale: float,
    row_selector: Any,
) -> Any:
    warped_df = build_reference_axis_warped_raw_df(
        raw_df,
        reference_column=str(reference_column),
        scale=float(scale),
    )
    return row_selector(
        warped_df,
        str(reference_column),
        float(target_reference),
        [str(feature_name) for feature_name in feature_columns],
    )

def _normalize_feature_vector(feature_values: Any, feature_count: int) -> np.ndarray:
    values = np.asarray(feature_values, dtype=float).reshape(-1)
    if values.size == int(feature_count):
        return values
    if values.size == int(feature_count) + 1:
        return values[1:]
    if values.size > int(feature_count):
        return values[-int(feature_count) :]
    return np.asarray([], dtype=float)

def estimate_warped_feature_floor_scale_from_feature_vectors(
    train_rows: Any,
    *,
    target_min_log_by_feature: Any,
    get_source_df: Any,
    get_input_reference: Any,
    reference_column: str,
    feature_columns: list[str],
    feature_vector_selector: Any,
    max_scale: float = 100.0,
    candidate_count: int = 240,
    scale_scope: str = "global",
) -> tuple[float, dict[str, float]]:
    normalized_scale_scope = str(scale_scope).strip().lower().replace("-", "_")
    if normalized_scale_scope not in {"global", "per_cell"}:
        raise ValueError(
            "FEATURE_LOG2_SCALE_SCOPE must be 'global' or 'per_cell', "
            f"got {scale_scope!r}."
        )
    if normalized_scale_scope == "per_cell":
        source_rows = list(_iter_source_rows(train_rows))
        per_cell_stats: dict[str, dict[str, object]] = {}
        per_cell_scales: dict[str, float] = {}
        for source_index, source_row in enumerate(source_rows):
            source_name = str(source_row.get("cell", f"source_{source_index + 1}"))
            source_scale, source_stats = estimate_warped_feature_floor_scale_from_feature_vectors(
                [source_row],
                target_min_log_by_feature=target_min_log_by_feature,
                get_source_df=get_source_df,
                get_input_reference=get_input_reference,
                reference_column=reference_column,
                feature_columns=feature_columns,
                feature_vector_selector=feature_vector_selector,
                max_scale=max_scale,
                candidate_count=candidate_count,
                scale_scope="global",
            )
            per_cell_stats[source_name] = dict(source_stats)
            if np.isfinite(source_scale):
                per_cell_scales[source_name] = float(source_scale)
                try:
                    source_row[FEATURE_LOG2_CELL_FEATURE_SCALE_KEY] = float(source_scale)
                except (TypeError, KeyError):
                    pass
        valid_scales = np.asarray(list(per_cell_scales.values()), dtype=float)
        if valid_scales.size == 0:
            return np.nan, {
                "feature_log2_scale_scope": "per_cell",
                "feature_coverage_scale": np.nan,
                "feature_coverage_scale_by_cell": {},
                "feature_coverage_source_count": int(len(source_rows)),
                "feature_coverage_resolved_source_count": 0,
                "feature_scale_rule": "per_source_iterative_warped_feature_curve_at_input_reference",
            }
        hardest_name = max(per_cell_scales, key=per_cell_scales.get)
        stats = dict(per_cell_stats[hardest_name])
        stats.update({
            "feature_log2_scale_scope": "per_cell",
            "feature_coverage_scale": float(np.max(valid_scales)),
            "feature_coverage_scale_by_cell": dict(per_cell_scales),
            "feature_coverage_source_count": int(len(source_rows)),
            "feature_coverage_resolved_source_count": int(valid_scales.size),
            "feature_coverage_scale_min": float(np.min(valid_scales)),
            "feature_coverage_scale_median": float(np.median(valid_scales)),
            "feature_coverage_scale_max": float(np.max(valid_scales)),
            "feature_coverage_hardest_source": str(hardest_name),
            "feature_scale_rule": "per_source_iterative_warped_feature_curve_at_input_reference",
        })
        return float(np.max(valid_scales)), stats

    feature_columns = [str(feature_name) for feature_name in feature_columns]
    target_min_by_feature = np.asarray(target_min_log_by_feature, dtype=float).reshape(-1)
    feature_count = int(len(feature_columns))
    stats = {
        "target_min_test_log_abs_feature": np.nan,
        "train_min_log_abs_feature_at_scale_1": np.nan,
        "achieved_min_augmented_log_abs_feature": np.nan,
        "feature_coverage_scale": np.nan,
        "feature_coverage_found": 0.0,
        "feature_coverage_feature_count": 0,
        "feature_coverage_worst_feature": "",
        "feature_coverage_uncovered_count": 0,
        "feature_coverage_columns": ",".join(feature_columns),
        "feature_coverage_required_scales": "",
        "feature_coverage_required_scale_max": np.nan,
        "feature_coverage_required_scale_log_std": np.nan,
        "max_log_feature_gap": np.nan,
        "feature_scale_rule": "central_iterative_warped_feature_curve_at_input_reference",
        "feature_log2_scale_scope": "global",
    }
    if feature_count == 0 or target_min_by_feature.size != feature_count:
        return np.nan, stats
    valid_feature_mask = np.isfinite(target_min_by_feature)
    if not np.any(valid_feature_mask):
        return np.nan, stats
    stats["target_min_test_log_abs_feature"] = float(np.min(target_min_by_feature[valid_feature_mask]))
    stats["feature_coverage_feature_count"] = int(np.count_nonzero(valid_feature_mask))

    def min_augmented_log_features_by_column(scale: float) -> np.ndarray:
        values_by_feature: list[list[float]] = [[] for _ in range(feature_count)]
        for source_row in _iter_source_rows(train_rows):
            source_df = get_source_df(source_row)
            if source_df is None or getattr(source_df, "empty", False):
                continue
            input_reference = finite_float(get_input_reference(source_row))
            if not np.isfinite(input_reference) or float(input_reference) <= 0.0:
                continue
            warped_df = build_reference_axis_warped_raw_df(
                source_df,
                reference_column=str(reference_column),
                scale=float(scale),
            )
            feature_vector = feature_vector_selector(
                warped_df,
                str(reference_column),
                float(input_reference),
                feature_columns,
            )
            if feature_vector is None:
                continue
            feature_values = _normalize_feature_vector(feature_vector, feature_count)
            if feature_values.size != feature_count:
                continue
            for feature_index, feature_value in enumerate(feature_values):
                if np.isfinite(feature_value):
                    values_by_feature[feature_index].append(float(np.log(max(abs(float(feature_value)), 1e-30))))
        result = np.full(feature_count, np.nan, dtype=float)
        for feature_index, feature_values in enumerate(values_by_feature):
            if feature_values:
                result[feature_index] = float(np.min(feature_values))
        return result

    def coverage_summary(min_by_feature: np.ndarray) -> tuple[bool, int, str]:
        covered = np.isfinite(min_by_feature) & (min_by_feature <= target_min_by_feature)
        uncovered_mask = valid_feature_mask & ~covered
        uncovered_count = int(np.count_nonzero(uncovered_mask))
        worst_feature = ""
        if uncovered_count:
            gaps = min_by_feature - target_min_by_feature
            gaps = np.where(uncovered_mask & np.isfinite(gaps), gaps, -np.inf)
            worst_index = int(np.argmax(gaps))
            worst_feature = str(feature_columns[worst_index])
        return uncovered_count == 0, uncovered_count, worst_feature

    required_scale_by_feature = np.full(feature_count, np.nan, dtype=float)

    def mark_required_scales(min_by_feature: np.ndarray, scale: float) -> None:
        covered = valid_feature_mask & np.isfinite(min_by_feature) & (min_by_feature <= target_min_by_feature)
        missing = ~np.isfinite(required_scale_by_feature)
        required_scale_by_feature[covered & missing] = float(scale)

    def update_required_scale_stats() -> None:
        valid_scales = required_scale_by_feature[valid_feature_mask & np.isfinite(required_scale_by_feature)]
        stats["feature_coverage_required_scales"] = ",".join(
            f"{feature_columns[index]}:{required_scale_by_feature[index]:.6g}"
            for index in range(feature_count)
            if valid_feature_mask[index] and np.isfinite(required_scale_by_feature[index])
        )
        if valid_scales.size:
            log_scales = np.log(valid_scales[valid_scales > 0.0])
            log_std = float(np.std(log_scales)) if log_scales.size > 1 else 0.0
            stats["feature_coverage_required_scale_max"] = float(np.max(valid_scales))
            stats["feature_coverage_required_scale_log_std"] = log_std

    scale_one_min_by_feature = min_augmented_log_features_by_column(1.0)
    mark_required_scales(scale_one_min_by_feature, 1.0)
    finite_scale_one = scale_one_min_by_feature[np.isfinite(scale_one_min_by_feature)]
    stats["train_min_log_abs_feature_at_scale_1"] = float(np.min(finite_scale_one)) if finite_scale_one.size else np.nan
    covered_at_one, uncovered_at_one, worst_at_one = coverage_summary(scale_one_min_by_feature)
    finite_gap_mask = valid_feature_mask & np.isfinite(scale_one_min_by_feature)
    log_feature_gaps = np.where(
        finite_gap_mask,
        scale_one_min_by_feature - target_min_by_feature,
        np.nan,
    )
    positive_gaps = log_feature_gaps[np.isfinite(log_feature_gaps) & (log_feature_gaps > 0.0)]
    stats["max_log_feature_gap"] = float(np.max(positive_gaps)) if positive_gaps.size else 0.0
    if covered_at_one:
        achieved = scale_one_min_by_feature[valid_feature_mask]
        stats["achieved_min_augmented_log_abs_feature"] = float(np.min(achieved[np.isfinite(achieved)]))
        stats["feature_coverage_scale"] = 1.0
        stats["feature_coverage_found"] = 1.0
        stats["feature_coverage_uncovered_count"] = 0
        update_required_scale_stats()
        return 1.0, stats
    stats["feature_coverage_uncovered_count"] = int(uncovered_at_one)
    stats["feature_coverage_worst_feature"] = str(worst_at_one)

    candidate_scales = np.geomspace(1.001, float(max_scale), int(candidate_count), dtype=float)
    best_scale = np.nan
    best_min_by_feature = np.full(feature_count, np.nan, dtype=float)
    best_uncovered_count = int(uncovered_at_one)
    best_worst_feature = str(worst_at_one)
    for candidate_scale in candidate_scales:
        candidate_min_by_feature = min_augmented_log_features_by_column(float(candidate_scale))
        if not np.isfinite(candidate_min_by_feature[valid_feature_mask]).any():
            continue
        mark_required_scales(candidate_min_by_feature, float(candidate_scale))
        best_scale = float(candidate_scale)
        best_min_by_feature = candidate_min_by_feature
        covered, uncovered_count, worst_feature = coverage_summary(candidate_min_by_feature)
        best_uncovered_count = int(uncovered_count)
        best_worst_feature = str(worst_feature)
        if covered:
            achieved = candidate_min_by_feature[valid_feature_mask]
            stats["achieved_min_augmented_log_abs_feature"] = float(np.min(achieved[np.isfinite(achieved)]))
            stats["feature_coverage_scale"] = float(candidate_scale)
            stats["feature_coverage_found"] = 1.0
            stats["feature_coverage_uncovered_count"] = 0
            stats["feature_coverage_worst_feature"] = ""
            update_required_scale_stats()
            return float(candidate_scale), stats
    finite_best = best_min_by_feature[np.isfinite(best_min_by_feature)]
    stats["achieved_min_augmented_log_abs_feature"] = float(np.min(finite_best)) if finite_best.size else np.nan
    stats["feature_coverage_scale"] = float(best_scale) if np.isfinite(best_scale) else np.nan
    stats["feature_coverage_uncovered_count"] = int(best_uncovered_count)
    stats["feature_coverage_worst_feature"] = str(best_worst_feature)
    update_required_scale_stats()
    return float(best_scale) if np.isfinite(best_scale) else np.nan, stats

def estimate_warped_feature_floor_scale_from_rows(
    train_rows: Any,
    *,
    target_min_log_feature: float,
    get_source_df: Any,
    get_input_reference: Any,
    reference_column: str,
    feature_columns: list[str],
    row_selector: Any,
    max_scale: float = 100.0,
    candidate_count: int = 240,
) -> tuple[float, dict[str, float]]:
    feature_columns = [str(feature_name) for feature_name in feature_columns]
    stats = {
        "target_min_test_log_abs_feature": (
            float(target_min_log_feature) if np.isfinite(float(target_min_log_feature)) else np.nan
        ),
        "train_min_log_abs_feature_at_scale_1": np.nan,
        "achieved_min_augmented_log_abs_feature": np.nan,
        "feature_coverage_scale": np.nan,
        "feature_coverage_found": 0.0,
        "feature_scale_rule": "central_iterative_warped_reference_row",
    }
    if not feature_columns or not np.isfinite(float(target_min_log_feature)):
        return np.nan, stats

    def min_augmented_log_feature(scale: float) -> float:
        values: list[float] = []
        for source_row in _iter_source_rows(train_rows):
            source_df = get_source_df(source_row)
            if source_df is None or getattr(source_df, "empty", False):
                continue
            input_reference = finite_float(get_input_reference(source_row))
            if not np.isfinite(input_reference) or float(input_reference) <= 0.0:
                continue
            augmented_row = select_warped_reference_row(
                source_df,
                reference_column=str(reference_column),
                target_reference=float(input_reference),
                feature_columns=feature_columns,
                scale=float(scale),
                row_selector=row_selector,
            )
            if augmented_row is None:
                continue
            for feature_name in feature_columns:
                if feature_name not in augmented_row:
                    continue
                feature_value = finite_float(augmented_row[feature_name])
                if np.isfinite(feature_value):
                    values.append(float(np.log(max(abs(float(feature_value)), 1e-30))))
        return float(np.min(values)) if values else np.nan

    scale_one_min = min_augmented_log_feature(1.0)
    stats["train_min_log_abs_feature_at_scale_1"] = float(scale_one_min) if np.isfinite(scale_one_min) else np.nan
    if np.isfinite(scale_one_min) and float(scale_one_min) <= float(target_min_log_feature):
        stats["achieved_min_augmented_log_abs_feature"] = float(scale_one_min)
        stats["feature_coverage_scale"] = 1.0
        stats["feature_coverage_found"] = 1.0
        return 1.0, stats

    candidate_scales = np.geomspace(1.001, float(max_scale), int(candidate_count), dtype=float)
    best_scale = np.nan
    best_min = np.nan
    for candidate_scale in candidate_scales:
        candidate_min = min_augmented_log_feature(float(candidate_scale))
        if not np.isfinite(candidate_min):
            continue
        best_scale = float(candidate_scale)
        best_min = float(candidate_min)
        if float(candidate_min) <= float(target_min_log_feature):
            stats["achieved_min_augmented_log_abs_feature"] = float(candidate_min)
            stats["feature_coverage_scale"] = float(candidate_scale)
            stats["feature_coverage_found"] = 1.0
            return float(candidate_scale), stats
    stats["achieved_min_augmented_log_abs_feature"] = float(best_min) if np.isfinite(best_min) else np.nan
    stats["feature_coverage_scale"] = float(best_scale) if np.isfinite(best_scale) else np.nan
    return float(best_scale) if np.isfinite(best_scale) else np.nan, stats

def build_warped_reference_augmented_rows(
    train_rows: Any,
    *,
    scales: list[float],
    get_source_df: Any,
    get_input_reference: Any,
    reference_column: str,
    augmented_feature_columns: list[str],
    row_selector: Any,
    row_builder: Any,
) -> list[Any]:
    generated_rows: list[Any] = []
    feature_columns = [str(feature_name) for feature_name in augmented_feature_columns]
    for source_row in _iter_source_rows(train_rows):
        source_df = get_source_df(source_row)
        if source_df is None or getattr(source_df, "empty", False):
            continue
        input_reference = finite_float(get_input_reference(source_row))
        if not np.isfinite(input_reference) or float(input_reference) <= 0.0:
            continue
        for scale in scales:
            effective_scale = finite_float(scale)
            if not np.isfinite(effective_scale) or float(effective_scale) <= 1.0:
                continue
            augmented_row = select_warped_reference_row(
                source_df,
                reference_column=str(reference_column),
                target_reference=float(input_reference),
                feature_columns=feature_columns,
                scale=float(effective_scale),
                row_selector=row_selector,
            )
            if augmented_row is None:
                continue
            built_row = row_builder(source_row, augmented_row, float(effective_scale), float(input_reference))
            if built_row is not None:
                generated_rows.append(built_row)
    return generated_rows

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
    raise RuntimeError(
        "Normal augmentation has been removed. Run an autoanchor workflow with "
        "DEFAULT_AUGMENTATION_METHOD set to 'feature_log2', 'matching', or "
        "'matching_structured' where supported."
    )

def build_feature_log_augmentation_scales_from_predictions(
    module: Any,
    train_targets: object,
    predicted_test_targets: object,
    feature_coverage_scale: float | None = None,
    prediction_sigma: object | None = None,
    prediction_sigma_space: str = "raw",
) -> tuple[list[float], dict[str, float]]:
    train_values = np.asarray(train_targets, dtype=float).reshape(-1)
    raw_test_values = np.asarray(predicted_test_targets, dtype=float).reshape(-1)
    test_values = raw_test_values[np.isfinite(raw_test_values) & (raw_test_values > 0.0)]
    train_values = train_values[np.isfinite(train_values) & (train_values > 0.0)]
    sigma_multiplier = finite_float(
        getattr(module, "FEATURE_LOG_PREDICTION_SIGMA_MULTIPLIER", FEATURE_LOG_PREDICTION_SIGMA_MULTIPLIER),
        default=float(FEATURE_LOG_PREDICTION_SIGMA_MULTIPLIER),
    )
    stats = {
        "augmentation_method": "feature_log",
        "scale_reference": "linear_prediction_upper_3sigma_from_test_features",
        "train_max_cycle_life": np.nan,
        "predicted_test_max_cycle_life": np.nan,
        "test_max_cycle_life": np.nan,
        "predicted_test_upper_cycle_life": np.nan,
        "prediction_sigma": np.nan,
        "prediction_sigma_space": str(prediction_sigma_space),
        "prediction_sigma_multiplier": float(sigma_multiplier) if np.isfinite(sigma_multiplier) else np.nan,
        "target_covered_cycle_life": np.nan,
        "coverage_multiplier": np.nan,
        "lifetime_coverage_scale": np.nan,
        "feature_coverage_scale": np.nan,
        "max_augmentation_scale": np.nan,
        "max_scale_source": "",
        "scale_spacing_mode": str(getattr(module, "NORMAL_AUGMENTATION_SCALE_SPACING", NORMAL_AUGMENTATION_SCALE_SPACING)),
        "scale_step": np.nan,
        "target_scale_count": 0,
        "low_scale_candidate_count": 0,
    }
    if train_values.size == 0 or test_values.size == 0:
        return [], stats
    train_max = float(np.max(train_values))
    test_max = float(np.max(test_values))
    sigma_values = np.asarray([], dtype=float)
    if prediction_sigma is not None:
        try:
            sigma_values = np.asarray(prediction_sigma, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            sigma_values = np.asarray([], dtype=float)
        sigma_values = sigma_values[np.isfinite(sigma_values) & (sigma_values >= 0.0)]
    prediction_sigma_value = float(np.max(sigma_values)) if sigma_values.size else np.nan
    sigma_space = str(prediction_sigma_space).strip().lower()
    if sigma_space not in {"raw", "log"}:
        sigma_space = "raw"
    if np.isfinite(prediction_sigma_value) and prediction_sigma_value >= 0.0:
        if not np.isfinite(sigma_multiplier) or float(sigma_multiplier) <= 0.0:
            sigma_multiplier = float(FEATURE_LOG_PREDICTION_SIGMA_MULTIPLIER)
        if sigma_space == "log":
            target_covered_cycle_life = float(
                np.exp(
                    np.clip(
                        np.log(max(float(test_max), 1e-30))
                        + float(sigma_multiplier) * float(prediction_sigma_value),
                        np.log(1e-30),
                        np.log(1e30),
                    )
                )
            )
            scale_reference = "linear_prediction_log_upper_3sigma_from_test_features"
        else:
            target_covered_cycle_life = float(test_max) + float(sigma_multiplier) * float(prediction_sigma_value)
            scale_reference = "linear_prediction_upper_3sigma_from_test_features"
        max_scale_source = "linear_prediction_upper_3sigma"
    else:
        target_covered_cycle_life = float(test_max)
        max_scale_source = "linear_prediction_without_sigma"
        scale_reference = "linear_prediction_max_without_sigma"
    lifetime_scale = float(target_covered_cycle_life) / max(float(train_max), 1e-12)
    feature_scale = finite_float(feature_coverage_scale)
    max_scale = float(lifetime_scale)
    stats.update(
        {
            "scale_reference": str(scale_reference),
            "train_max_cycle_life": float(train_max),
            "predicted_test_max_cycle_life": float(test_max),
            "test_max_cycle_life": float(test_max),
            "predicted_test_upper_cycle_life": float(target_covered_cycle_life),
            "prediction_sigma": float(prediction_sigma_value) if np.isfinite(prediction_sigma_value) else np.nan,
            "prediction_sigma_space": str(sigma_space),
            "prediction_sigma_multiplier": float(sigma_multiplier) if np.isfinite(sigma_multiplier) else np.nan,
            "target_covered_cycle_life": float(target_covered_cycle_life),
            "coverage_multiplier": np.nan,
            "lifetime_coverage_scale": float(lifetime_scale) if np.isfinite(lifetime_scale) else np.nan,
            "feature_coverage_scale": float(feature_scale) if np.isfinite(feature_scale) else np.nan,
            "max_augmentation_scale": float(max_scale) if np.isfinite(max_scale) else np.nan,
            "max_scale_source": str(max_scale_source),
        }
    )
    if not np.isfinite(max_scale) or float(max_scale) <= 1.0 + 1e-9:
        return [], stats

    scale_count = max(
        1,
        int(getattr(module, "FEATURE_LOG_AUGMENTATION_FACTOR_COUNT", aug_cfg.FEATURE_LOG_AUGMENTATION_FACTOR_COUNT)),
    )
    if int(scale_count) == 1:
        candidate_scales = [float(max_scale)]
    else:
        candidate_scales = np.linspace(1.0, float(max_scale), int(scale_count) + 1, dtype=float)[1:].tolist()
    unique_scales: list[float] = []
    for scale in sorted(candidate_scales):
        scale = min(float(max_scale), float(scale))
        if float(scale) <= 1.0 + 1e-9:
            continue
        if any(abs(float(scale) - existing) < 1e-9 for existing in unique_scales):
            continue
        unique_scales.append(float(scale))
    stats["scale_spacing_mode"] = "feature_log_sparse_linear_to_max"
    stats["scale_step"] = np.nan
    stats["target_scale_count"] = int(len(unique_scales))
    stats["low_scale_candidate_count"] = 0
    return unique_scales, stats

def build_feature_log_augmentation_scales_from_targets(
    module: Any,
    train_targets: object,
    test_targets: object,
) -> tuple[list[float], dict[str, float]]:
    return build_feature_log_augmentation_scales_from_predictions(module, train_targets, test_targets)


def build_feature_log2_augmentation_scales_from_feature_coverage(
    module: Any,
    feature_coverage_scale: float | None,
    train_targets: object | None = None,
    predicted_test_targets: object | None = None,
    prediction_sigma: object | None = None,
    prediction_sigma_space: str = "raw",
    train_cells: list[dict[str, object]] | None = None,
    feature_coverage_scales_by_cell: dict[str, float] | None = None,
    scale_scope: str | None = None,
) -> tuple[list[float], dict[str, float]]:
    feature_scale = finite_float(feature_coverage_scale)
    lifetime_stats: dict[str, float] = {}
    lifetime_scale = np.nan
    if train_targets is not None and predicted_test_targets is not None:
        _lifetime_scales, lifetime_stats = build_feature_log_augmentation_scales_from_predictions(
            module,
            train_targets,
            predicted_test_targets,
            feature_coverage_scale=feature_coverage_scale,
            prediction_sigma=prediction_sigma,
            prediction_sigma_space=prediction_sigma_space,
        )
        lifetime_scale = finite_float(lifetime_stats.get("lifetime_coverage_scale"))
    max_scale_policy = str(aug_cfg.FEATURE_LOG2_MAX_SCALE_POLICY).strip().lower()
    if max_scale_policy == "var_delta_q_only":
        max_scale_candidates = [("test_feature_log_floor", feature_scale)]
        scale_reference = "test_feature_log_floor_only"
    elif max_scale_policy == "max_var_delta_q_and_lifetime_prediction_multiplier":
        max_scale_candidates = [("test_feature_log_floor", feature_scale)]
        scale_reference = "test_feature_log_floor_with_post_generation_prediction_multiplier_limit"
    else:
        raise ValueError(
            "FEATURE_LOG2_MAX_SCALE_POLICY must be 'var_delta_q_only' or "
            "'max_var_delta_q_and_lifetime_prediction_multiplier', "
            f"got {max_scale_policy!r}."
        )
    finite_scale_candidates = [
        (source, float(scale))
        for source, scale in max_scale_candidates
        if np.isfinite(float(scale)) and float(scale) > 0.0
    ]
    max_scale_source = "test_feature_log_floor"
    max_scale = feature_scale
    if finite_scale_candidates:
        max_scale_source, max_scale = max(finite_scale_candidates, key=lambda item: item[1])
    normalized_scale_scope = str(
        scale_scope or aug_cfg.FEATURE_LOG2_SCALE_SCOPE
    ).strip().lower().replace("-", "_")
    if normalized_scale_scope not in {"global", "per_cell"}:
        raise ValueError(
            "FEATURE_LOG2_SCALE_SCOPE must be 'global' or 'per_cell', "
            f"got {scale_scope or aug_cfg.FEATURE_LOG2_SCALE_SCOPE!r}."
        )
    stats = {
        "augmentation_method": "feature_log2",
        "feature_log2_scale_scope": str(normalized_scale_scope),
        "feature_log2_max_scale_policy": str(max_scale_policy),
        "scale_reference": str(scale_reference),
        "train_max_cycle_life": np.nan,
        "predicted_test_max_cycle_life": np.nan,
        "test_max_cycle_life": np.nan,
        "predicted_test_upper_cycle_life": np.nan,
        "prediction_sigma": np.nan,
        "prediction_sigma_space": str(prediction_sigma_space),
        "prediction_sigma_multiplier": np.nan,
        "target_covered_cycle_life": np.nan,
        "coverage_multiplier": np.nan,
        "lifetime_coverage_scale": float(lifetime_scale) if np.isfinite(lifetime_scale) else np.nan,
        "feature_coverage_scale": float(feature_scale) if np.isfinite(feature_scale) else np.nan,
        "max_augmentation_scale": float(max_scale) if np.isfinite(max_scale) else np.nan,
        "max_scale_source": str(max_scale_source),
        "scale_spacing_mode": "feature_log2_power_to_configured_max",
        "scale_step": np.nan,
        "scale_grid_power": float(
            finite_float(
                getattr(module, "FEATURE_LOG2_SCALE_GRID_POWER", FEATURE_LOG2_SCALE_GRID_POWER),
                default=float(FEATURE_LOG2_SCALE_GRID_POWER),
            )
        ),
        "target_scale_count": 0,
        "low_scale_candidate_count": 0,
    }
    for key in (
        "train_max_cycle_life",
        "predicted_test_max_cycle_life",
        "test_max_cycle_life",
        "predicted_test_upper_cycle_life",
        "prediction_sigma",
        "prediction_sigma_space",
        "prediction_sigma_multiplier",
        "target_covered_cycle_life",
        "coverage_multiplier",
        "lifetime_coverage_scale",
    ):
        if key in lifetime_stats:
            stats[key] = lifetime_stats[key]
    stats["feature_log2_feature_floor_scale"] = float(feature_scale) if np.isfinite(feature_scale) else np.nan
    stats["feature_log2_lifetime_scale"] = float(lifetime_scale) if np.isfinite(lifetime_scale) else np.nan
    if normalized_scale_scope == "per_cell" and train_cells is not None:
        coverage_by_cell = {
            str(key): finite_float(value)
            for key, value in dict(feature_coverage_scales_by_cell or {}).items()
        }
        per_cell_max_scales: dict[str, float] = {}
        per_cell_scale_sources: dict[str, str] = {}
        representative_scales: list[float] = []
        representative_scale_stats: dict[str, object] = {}
        for cell in train_cells:
            cell_name = str(cell.get("cell", ""))
            cell_feature_scale = finite_float(
                coverage_by_cell.get(cell_name, cell.get(FEATURE_LOG2_CELL_FEATURE_SCALE_KEY))
            )
            candidates = [("test_feature_log_floor", cell_feature_scale)]
            valid_candidates = [
                (source, float(value))
                for source, value in candidates
                if np.isfinite(value) and float(value) > 0.0
            ]
            if not valid_candidates:
                cell.pop(FEATURE_LOG2_CELL_SCALES_KEY, None)
                continue
            cell_scale_source, cell_max_scale = max(valid_candidates, key=lambda item: item[1])
            cell_scales, cell_stats = build_feature_log2_augmentation_scales_from_feature_coverage(
                module,
                feature_coverage_scale=cell_max_scale,
                train_targets=None,
                predicted_test_targets=None,
                prediction_sigma=None,
                prediction_sigma_space=prediction_sigma_space,
                scale_scope="global",
            )
            cell[FEATURE_LOG2_CELL_SCALES_KEY] = list(cell_scales)
            cell[FEATURE_LOG2_CELL_FEATURE_SCALE_KEY] = float(cell_feature_scale)
            per_cell_max_scales[cell_name] = float(cell_max_scale)
            per_cell_scale_sources[cell_name] = str(cell_scale_source)
            if float(cell_max_scale) >= max(per_cell_max_scales.values()):
                representative_scales = list(cell_scales)
                representative_scale_stats = dict(cell_stats)
        valid_cell_maxima = np.asarray(list(per_cell_max_scales.values()), dtype=float)
        stats.update({
            "feature_log2_per_cell_max_scales": dict(per_cell_max_scales),
            "feature_log2_per_cell_scale_sources": dict(per_cell_scale_sources),
            "feature_log2_per_cell_source_count": int(len(per_cell_max_scales)),
            "feature_log2_per_cell_scale_min": float(np.min(valid_cell_maxima)) if valid_cell_maxima.size else np.nan,
            "feature_log2_per_cell_scale_median": float(np.median(valid_cell_maxima)) if valid_cell_maxima.size else np.nan,
            "feature_log2_per_cell_scale_max": float(np.max(valid_cell_maxima)) if valid_cell_maxima.size else np.nan,
            "max_augmentation_scale": float(np.max(valid_cell_maxima)) if valid_cell_maxima.size else np.nan,
            "max_scale_source": "per_cell",
            "target_scale_count": int(aug_cfg.FEATURE_LOG_AUGMENTATION_FACTOR_COUNT),
            "scale_spacing_mode": representative_scale_stats.get("scale_spacing_mode", ""),
            "scale_step": representative_scale_stats.get("scale_step", np.nan),
            "scale_grid_power": representative_scale_stats.get("scale_grid_power", np.nan),
        })
        return representative_scales, stats
    if not np.isfinite(max_scale) or float(max_scale) <= 1.0 + 1e-9:
        return [], stats

    scale_count = max(
        1,
        int(getattr(module, "FEATURE_LOG_AUGMENTATION_FACTOR_COUNT", aug_cfg.FEATURE_LOG_AUGMENTATION_FACTOR_COUNT)),
    )
    if hasattr(module, "FEATURE_LOG2_SCALE_SPACING_METHOD"):
        raw_spacing_method = getattr(module, "FEATURE_LOG2_SCALE_SPACING_METHOD")
    elif hasattr(module, "FEATURE_LOG2_USE_EQUAL_STEP_FACTORS"):
        raw_spacing_method = (
            "equal_step"
            if bool(getattr(module, "FEATURE_LOG2_USE_EQUAL_STEP_FACTORS"))
            else "power"
        )
    else:
        raw_spacing_method = aug_cfg.FEATURE_LOG2_SCALE_SPACING_METHOD
    spacing_method = str(raw_spacing_method).strip().lower().replace("-", "_")
    spacing_aliases = {
        "equal": "equal_step",
        "equal_steps": "equal_step",
        "linear": "equal_step",
        "linear_equal_steps": "equal_step",
        "geometric": "log",
        "geomspace": "log",
        "logarithmic": "log",
        "log_uniform": "log",
        "power_grid": "power",
        "low_scale_biased": "power",
    }
    spacing_method = spacing_aliases.get(spacing_method, spacing_method)
    if not spacing_method:
        spacing_method = (
            "equal_step"
            if bool(getattr(module, "FEATURE_LOG2_USE_EQUAL_STEP_FACTORS", aug_cfg.FEATURE_LOG2_USE_EQUAL_STEP_FACTORS))
            else "power"
        )
    if spacing_method not in {"equal_step", "log", "power"}:
        raise ValueError(
            "FEATURE_LOG2_SCALE_SPACING_METHOD must be 'equal_step', 'log', or 'power', "
            f"got {spacing_method!r}."
        )
    if int(scale_count) == 1:
        candidate_scales = [float(max_scale)]
    elif spacing_method == "equal_step":
        candidate_scales = np.linspace(1.0, float(max_scale), int(scale_count) + 1, dtype=float)[1:].tolist()
        stats["scale_spacing_mode"] = "feature_log2_linear_equal_steps_to_configured_max"
        stats["scale_step"] = float((float(max_scale) - 1.0) / float(scale_count))
        stats["scale_grid_power"] = 1.0
    elif spacing_method == "log":
        candidate_scales = np.geomspace(
            1.0, float(max_scale), int(scale_count) + 1, dtype=float
        )[1:].tolist()
        stats["scale_spacing_mode"] = "feature_log2_log_equal_steps_to_configured_max"
        stats["scale_step"] = float(np.log(float(max_scale)) / float(scale_count))
        stats["scale_grid_power"] = np.nan
    else:
        grid_power = finite_float(
            getattr(module, "FEATURE_LOG2_SCALE_GRID_POWER", FEATURE_LOG2_SCALE_GRID_POWER),
            default=float(FEATURE_LOG2_SCALE_GRID_POWER),
        )
        if not np.isfinite(grid_power) or float(grid_power) <= 1.0:
            grid_power = float(FEATURE_LOG2_SCALE_GRID_POWER)
        fractions = np.linspace(0.0, 1.0, int(scale_count) + 1, dtype=float)[1:]
        candidate_scales = (1.0 + (float(max_scale) - 1.0) * np.power(fractions, float(grid_power))).tolist()
        stats["scale_grid_power"] = float(grid_power)
    unique_scales: list[float] = []
    for scale in sorted(candidate_scales):
        scale = min(float(max_scale), float(scale))
        if float(scale) <= 1.0 + 1e-9:
            continue
        if any(abs(float(scale) - existing) < 1e-9 for existing in unique_scales):
            continue
        unique_scales.append(float(scale))
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
    use_warped_feature_reference: bool = True,
) -> list[dict[str, object]]:
    if not train_cells:
        return []
    feature_columns = [str(name) for name in feature_columns]
    density = max(1, int(density_multiplier if density_multiplier is not None else getattr(module, "AUGMENTATION_DENSITY_MULTIPLIER", 1)))
    def densify_scales(scales: object) -> list[float]:
        dense_scales: list[float] = []
        for scale in scales:
            effective_scale = max(1.0, float(scale))
            if effective_scale <= 1.0 + 1e-9:
                continue
            candidates = ([effective_scale] if density == 1 else
                np.linspace(1.0, effective_scale, density + 1, dtype=float)[1:].tolist())
            for dense_scale in candidates:
                if dense_scale <= 1.0 + 1e-9:
                    continue
                if any(abs(float(dense_scale) - existing) < 1e-9 for existing in dense_scales):
                    continue
                dense_scales.append(float(dense_scale))
        return dense_scales

    normalized_task_kind = str(task_kind).strip().lower()
    reference_column = resolve_reference_column(module, task_kind)
    use_per_cell_scales = (
        str(aug_cfg.FEATURE_LOG2_SCALE_SCOPE).strip().lower().replace("-", "_") == "per_cell"
    )
    augmented_cells: list[dict[str, object]] = []
    for cell in train_cells:
        cell_scales = (
            cell.get(FEATURE_LOG2_CELL_SCALES_KEY, augmentation_scales)
            if use_per_cell_scales
            else augmentation_scales
        )
        for scale in densify_scales(cell_scales):
            fallback_input_reference = finite_float(input_reference_value)
            cell_input_reference = finite_float(cell.get("input_reference_value"), fallback_input_reference)
            if not np.isfinite(cell_input_reference) or float(cell_input_reference) <= 0.0:
                continue
            augmented_week_value, augmented_throughput_value = resolve_augmented_targets(
                cell,
                task_kind=str(normalized_task_kind),
                scale=float(scale),
            )
            augmented_target_value = float(augmented_week_value)
            active_target_value = (
                float(augmented_throughput_value)
                if normalized_task_kind == "throughput"
                else float(augmented_week_value)
            )
            if not np.isfinite(active_target_value) or active_target_value <= 0.0:
                continue
            # Both tasks, not just time: a cell whose EOL falls at or before the reference its own
            # features are read at is self-contradictory -- it died before the measurement it is
            # labelled with. The real cells are already dropped for this by the
            # throughput_before_feature_target filter; the augmented ones were not, so a short-lived
            # parent scaled by a factor close to 1.0 could still enter training with an EOL behind
            # its own input.
            if active_target_value <= float(cell_input_reference):
                continue
            warped_raw_df = build_reference_axis_warped_raw_df(
                cell["raw_df"],
                reference_column=reference_column,
                scale=float(scale),
            )
            feature_input_reference = float(cell_input_reference)
            source_feature_reference = float(cell_input_reference) / float(scale)
            feature_vector = module.interpolate_feature_vector(
                warped_raw_df if bool(use_warped_feature_reference) else cell["raw_df"],
                feature_columns,
                reference_column=reference_column,
                target_reference=feature_input_reference,
            )
            feature_sequence = module.interpolate_feature_sequence(
                warped_raw_df if bool(use_warped_feature_reference) else cell["raw_df"],
                feature_columns,
                reference_column=reference_column,
                target_reference=feature_input_reference,
                discretization_steps=int(getattr(module, "DISCRETIZATION_STEPS")),
            )
            if feature_vector is None or feature_sequence is None:
                continue
            output_x = np.asarray(feature_vector, dtype=float)
            if int(output_x_drop_count) > 0:
                output_x = output_x[int(output_x_drop_count):]
            # EOL in weeks, scaled the same way time_curve is: on the time task the whole
            # trajectory is stretched by `scale`, so the week at which it reaches EOL stretches
            # with it. Without this the augmented cells had no time EOL at all and every
            # time-domain figure showed an empty augmented population.
            try:
                source_week = float(cell.get("target_week", np.nan))
            except (TypeError, ValueError):
                source_week = float("nan")
            augmented_week = (
                float(augmented_week_value)
                if np.isfinite(float(augmented_week_value))
                else (
                    source_week * float(scale)
                    if normalized_task_kind == "time"
                    else source_week
                )
            )
            augmented_cells.append(
                {
                    "cell": f"{cell['cell']}_aug_{float(scale):.3f}",
                    "source_kind": str(source_kind),
                    "augmentation_scale": float(scale),
                    "x": output_x,
                    "sequence": np.asarray(feature_sequence, dtype=float),
                    "raw_df": warped_raw_df,
                    "target_value": float(augmented_target_value),
                    "target_week": augmented_week,
                    "target_week_value": float(augmented_week_value),
                    "target_throughput_value": float(augmented_throughput_value),
                    "input_reference_value": float(cell_input_reference),
                    "time_curve": np.asarray(cell["time_curve"], dtype=float) * float(scale),
                    "throughput_curve": np.asarray(cell["throughput_curve"], dtype=float) * float(scale),
                    "soh_curve": np.asarray(cell["soh_curve"], dtype=float),
                    "input_throughput_value": float(cell.get("input_throughput_value", np.nan)),
                    "cal_max_reference_used": float(feature_input_reference),
                    "augmented_input_reference_value": float(feature_input_reference),
                    "source_feature_reference_value": float(source_feature_reference),
                }
            )
    return augmented_cells


def project_cells_to_reference(
    module: Any,
    cells: list[dict[str, object]],
    feature_columns: list[str],
    *,
    task_kind: str,
    input_reference_value: float,
) -> list[dict[str, object]]:
    """Return shallow cell copies whose scalar features are evaluated at one reference."""
    input_reference = finite_float(input_reference_value)
    if not np.isfinite(input_reference) or input_reference <= 0.0:
        return []
    columns = [str(name) for name in feature_columns]
    if columns and columns[0].strip().lower() == "soh":
        columns = columns[1:]
    reference_column = resolve_reference_column(module, str(task_kind))
    projected: list[dict[str, object]] = []
    for cell in cells:
        raw_df = cell.get("raw_df")
        if raw_df is None:
            continue
        feature_vector = module.interpolate_feature_vector(
            raw_df,
            columns,
            reference_column=reference_column,
            target_reference=float(input_reference),
        )
        if feature_vector is None:
            continue
        feature_vector = np.asarray(feature_vector, dtype=float)
        if not np.isfinite(feature_vector).all():
            continue
        projected_cell = dict(cell)
        projected_cell["x"] = feature_vector
        projected_cell[PROCESS_FEATURE_VECTOR_KEY] = feature_vector.copy()
        projected_cell[PROCESS_FEATURE_COLUMNS_KEY] = tuple(["soh", *columns])
        projected_cell["input_reference_value"] = float(input_reference)
        projected.append(projected_cell)
    return projected

def _reference_soh_curve_from_cell(
    module: Any,
    cell: dict[str, object],
    *,
    task_kind: str,
    enforce_monotonic_soh: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_task_kind = str(task_kind).strip().lower()
    raw_df = cell.get("raw_df")
    if raw_df is not None:
        reference_column = resolve_reference_column(module, normalized_task_kind)
        capacity_column = resolve_capacity_column(module)
        references = _series_from_raw_df(raw_df, reference_column)
        capacities = _series_from_raw_df(raw_df, capacity_column)
        point_count = min(int(references.size), int(capacities.size))
        if point_count >= 2:
            references = references[:point_count]
            capacities = capacities[:point_count]
            valid_capacity = capacities[np.isfinite(capacities)]
            if valid_capacity.size:
                initial_capacity = float(valid_capacity[0])
                if np.isfinite(initial_capacity) and initial_capacity > 0.0:
                    return prepare_reference_soh_curve(
                        references,
                        capacities / initial_capacity,
                        enforce_monotonic_soh=bool(enforce_monotonic_soh),
                    )
    curve_key = "throughput_curve" if normalized_task_kind == "throughput" else "time_curve"
    return prepare_reference_soh_curve(
        cell.get(curve_key, []),
        cell.get("soh_curve", []),
        enforce_monotonic_soh=bool(enforce_monotonic_soh),
    )

def _soh_at_reference(
    module: Any,
    cell: dict[str, object],
    *,
    task_kind: str,
    reference_value: float,
) -> float:
    reference, soh = _reference_soh_curve_from_cell(
        module,
        cell,
        task_kind=str(task_kind),
        enforce_monotonic_soh=False,
    )
    if reference.size < 2 or soh.size < 2:
        return np.nan
    target_reference = finite_float(reference_value)
    if not np.isfinite(target_reference):
        return np.nan
    if target_reference < float(np.min(reference)) or target_reference > float(np.max(reference)):
        return np.nan
    return float(np.interp(float(target_reference), reference, soh))

def _reference_at_soh(
    module: Any,
    cell: dict[str, object],
    *,
    task_kind: str,
    target_soh: float,
) -> float:
    reference, soh = _reference_soh_curve_from_cell(
        module,
        cell,
        task_kind=str(task_kind),
        enforce_monotonic_soh=True,
    )
    if reference.size < 2 or soh.size < 2:
        return np.nan
    target = finite_float(target_soh)
    if not np.isfinite(target) or target <= 0.0:
        return np.nan
    for index in range(1, int(soh.size)):
        previous_soh = float(soh[index - 1])
        current_soh = float(soh[index])
        previous_reference = float(reference[index - 1])
        current_reference = float(reference[index])
        if not all(
            np.isfinite(value)
            for value in (previous_soh, current_soh, previous_reference, current_reference)
        ):
            continue
        if np.isclose(previous_soh, target, rtol=0.0, atol=1e-12):
            return previous_reference
        if np.isclose(current_soh, target, rtol=0.0, atol=1e-12):
            return current_reference
        if previous_soh >= float(target) >= current_soh:
            if np.isclose(previous_soh, current_soh, rtol=0.0, atol=1e-12):
                return current_reference
            fraction = (float(target) - previous_soh) / (current_soh - previous_soh)
            matched_reference = previous_reference + fraction * (current_reference - previous_reference)
            return float(matched_reference) if np.isfinite(matched_reference) else np.nan
    return np.nan


def _matching_feature_columns(
    cell: dict[str, object],
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    raw_df = cell.get("raw_df")
    if not isinstance(raw_df, pd.DataFrame):
        return []
    columns = {str(column).strip().lower(): str(column) for column in raw_df.columns}
    if feature_columns:
        result: list[str] = []
        seen: set[str] = set()
        for feature_name in feature_columns:
            normalized = str(feature_name).strip().lower()
            if not normalized or normalized == "soh" or normalized in seen:
                continue
            resolved = columns.get(normalized)
            if resolved is None:
                continue
            result.append(resolved)
            seen.add(normalized)
        return result
    fallback = columns.get("var_delta_q")
    return [fallback] if fallback is not None else []


def _matching_feature_column(cell: dict[str, object]) -> str | None:
    columns = _matching_feature_columns(cell)
    return columns[0] if columns else None


def _prepare_reference_feature_curve(reference: np.ndarray, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(reference) & np.isfinite(feature)
    reference = np.asarray(reference[valid], dtype=float)
    feature = np.asarray(feature[valid], dtype=float)
    if reference.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(reference)
    reference = reference[order]
    feature = feature[order]
    unique_reference, inverse = np.unique(reference, return_inverse=True)
    unique_feature = np.asarray(
        [np.nanmedian(feature[inverse == index]) for index in range(unique_reference.size)],
        dtype=float,
    )
    return unique_reference, unique_feature


def _feature_at_reference(
    module: Any,
    cell: dict[str, object],
    *,
    task_kind: str,
    reference_value: float,
    feature_column: str | None = None,
) -> float:
    raw_df = cell.get("raw_df")
    feature_column = str(feature_column) if feature_column else _matching_feature_column(cell)
    if not isinstance(raw_df, pd.DataFrame) or feature_column is None:
        return np.nan
    reference_column = resolve_reference_column(module, str(task_kind))
    if reference_column not in raw_df.columns:
        return np.nan
    reference = pd.to_numeric(raw_df[reference_column], errors="coerce").to_numpy(dtype=float)
    feature = pd.to_numeric(raw_df[feature_column], errors="coerce").to_numpy(dtype=float)
    reference, feature = _prepare_reference_feature_curve(reference, feature)
    target_reference = finite_float(reference_value)
    if reference.size < 2 or not np.isfinite(target_reference):
        return np.nan
    if target_reference < float(np.min(reference)) or target_reference > float(np.max(reference)):
        return np.nan
    return float(np.interp(target_reference, reference, feature))


def _features_at_reference(
    module: Any,
    cell: dict[str, object],
    *,
    task_kind: str,
    reference_value: float,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for feature_column in _matching_feature_columns(cell, feature_columns):
        value = _feature_at_reference(
            module,
            cell,
            task_kind=str(task_kind),
            reference_value=float(reference_value),
            feature_column=str(feature_column),
        )
        if np.isfinite(value):
            values[str(feature_column)] = float(value)
    return values


def _reference_at_feature(
    module: Any,
    cell: dict[str, object],
    *,
    task_kind: str,
    target_feature: float,
    feature_column: str | None = None,
) -> float:
    raw_df = cell.get("raw_df")
    feature_column = str(feature_column) if feature_column else _matching_feature_column(cell)
    if not isinstance(raw_df, pd.DataFrame) or feature_column is None:
        return np.nan
    reference_column = resolve_reference_column(module, str(task_kind))
    if reference_column not in raw_df.columns:
        return np.nan
    reference = pd.to_numeric(raw_df[reference_column], errors="coerce").to_numpy(dtype=float)
    feature = pd.to_numeric(raw_df[feature_column], errors="coerce").to_numpy(dtype=float)
    reference, feature = _prepare_reference_feature_curve(reference, feature)
    target = finite_float(target_feature)
    if reference.size < 2 or not np.isfinite(target) or np.isclose(target, 0.0):
        return np.nan
    transformed = np.log(np.maximum(np.abs(feature), 1e-30))
    target_transformed = float(np.log(max(abs(target), 1e-30)))
    for index in range(1, int(transformed.size)):
        previous = float(transformed[index - 1])
        current = float(transformed[index])
        if not all(np.isfinite(value) for value in (previous, current)):
            continue
        if np.isclose(previous, target_transformed, rtol=0.0, atol=1e-12):
            return float(reference[index - 1])
        if np.isclose(current, target_transformed, rtol=0.0, atol=1e-12):
            return float(reference[index])
        if min(previous, current) <= target_transformed <= max(previous, current):
            fraction = (target_transformed - previous) / (current - previous)
            return float(reference[index - 1] + fraction * (reference[index] - reference[index - 1]))
    return np.nan


def _references_at_features(
    module: Any,
    cell: dict[str, object],
    *,
    task_kind: str,
    target_features: dict[str, float],
) -> dict[str, float]:
    references: dict[str, float] = {}
    for feature_column, target_feature in target_features.items():
        reference = _reference_at_feature(
            module,
            cell,
            task_kind=str(task_kind),
            target_feature=float(target_feature),
            feature_column=str(feature_column),
        )
        if np.isfinite(reference) and float(reference) > 0.0:
            references[str(feature_column)] = float(reference)
    return references

def _matching_specs(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    task_kind: str,
    input_reference_value: float,
    matching_feature_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict[str, object]], np.ndarray]:
    input_reference = finite_float(input_reference_value)
    if not np.isfinite(input_reference) or input_reference <= 0.0:
        return [], np.asarray([], dtype=float)
    target_soh_by_test: dict[str, float] = {}
    target_features_by_test: dict[str, dict[str, float]] = {}
    for test_cell in test_cells:
        test_cell_name = str(test_cell.get("cell", ""))
        target_soh = _soh_at_reference(
            module,
            test_cell,
            task_kind=str(task_kind),
            reference_value=float(input_reference),
        )
        if not np.isfinite(target_soh):
            test_x = np.asarray(test_cell.get("x", []), dtype=float).reshape(-1)
            if test_x.size:
                target_soh = finite_float(test_x[0])
        if np.isfinite(target_soh) and 0.0 < float(target_soh) <= 1.05:
            target_soh_by_test[test_cell_name] = float(target_soh)
        target_features = _features_at_reference(
            module,
            test_cell,
            task_kind=str(task_kind),
            reference_value=float(input_reference),
            feature_columns=matching_feature_columns,
        )
        if target_features:
            target_features_by_test[test_cell_name] = dict(target_features)

    specs: list[dict[str, object]] = []
    for test_cell in test_cells:
        test_cell_name = str(test_cell.get("cell", ""))
        target_soh = target_soh_by_test.get(test_cell_name, np.nan)
        if not np.isfinite(target_soh):
            continue
        target_features = target_features_by_test.get(test_cell_name, {})
        for train_cell in train_cells:
            source_reference = _reference_at_soh(
                module,
                train_cell,
                task_kind=str(task_kind),
                target_soh=float(target_soh),
            )
            matching_rule = "warp_train_to_test_soh_at_input_reference"
            source_feature_references: dict[str, float] = {}
            if (not np.isfinite(source_reference) or float(source_reference) <= 0.0) and target_features:
                source_feature_references = _references_at_features(
                    module,
                    train_cell,
                    task_kind=str(task_kind),
                    target_features=target_features,
                )
                finite_feature_references = np.asarray(
                    list(source_feature_references.values()),
                    dtype=float,
                )
                finite_feature_references = finite_feature_references[
                    np.isfinite(finite_feature_references) & (finite_feature_references > 0.0)
                ]
                if finite_feature_references.size:
                    source_reference = float(np.median(finite_feature_references))
                    matching_rule = "warp_train_to_test_feature_median_at_input_reference"
            if not np.isfinite(source_reference) or float(source_reference) <= 0.0:
                continue
            scale = float(input_reference) / float(source_reference)
            if not np.isfinite(scale) or scale <= 0.0:
                continue
            target_var_delta_q = np.nan
            for feature_name, feature_value in target_features.items():
                if str(feature_name).strip().lower() == "var_delta_q":
                    target_var_delta_q = finite_float(feature_value)
                    break
            specs.append(
                {
                    "train_cell": train_cell,
                    "test_cell": test_cell,
                    "target_soh": float(target_soh),
                    "target_feature": float(target_var_delta_q) if np.isfinite(target_var_delta_q) else np.nan,
                    "target_features": dict(target_features),
                    "source_feature_references": dict(source_feature_references),
                    "source_reference": float(source_reference),
                    "scale": float(scale),
                    "matching_rule": str(matching_rule),
                }
            )
    return specs, np.asarray(list(target_soh_by_test.values()), dtype=float)


def build_matching_augmentation_specs(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    task_kind: str,
    input_reference_value: float,
    matching_feature_columns: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    specs, _ = _matching_specs(
        module,
        train_cells,
        test_cells,
        task_kind=str(task_kind),
        input_reference_value=float(input_reference_value),
        matching_feature_columns=matching_feature_columns,
    )
    return specs


def rebind_matching_augmentation_specs(
    specs: list[dict[str, object]],
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep calibration-derived scales while generating model-input trajectories."""
    train_by_name = {str(cell.get("cell", "")): cell for cell in train_cells}
    test_by_name = {str(cell.get("cell", "")): cell for cell in test_cells}
    rebound: list[dict[str, object]] = []
    for spec in specs:
        train_name = str(dict(spec.get("train_cell", {})).get("cell", ""))
        test_name = str(dict(spec.get("test_cell", {})).get("cell", ""))
        if train_name not in train_by_name or test_name not in test_by_name:
            continue
        rebound_spec = dict(spec)
        rebound_spec["train_cell"] = train_by_name[train_name]
        rebound_spec["test_cell"] = test_by_name[test_name]
        rebound.append(rebound_spec)
    return rebound


def _matching_scale_cap(max_scale: float | None) -> float:
    scale_cap = finite_float(max_scale)
    return float(scale_cap) if np.isfinite(scale_cap) and scale_cap > 0.0 else 1.0


def matching_cap_kwargs_from_feature_log2_stats(
    feature_log2_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    """Translate the shared feature-log policy into matching caps."""
    # A 1x ceiling yields no matching candidates because matching only creates scales > 1.
    # This mirrors feature_log2, which creates no samples when its feature floor has no valid cap.
    result: dict[str, object] = {
        "max_scale": 1.0,
        "max_scale_source": "feature_log2_unavailable",
    }
    stats = dict(feature_log2_stats or {})
    if not stats:
        return result

    max_scale = finite_float(stats.get("max_augmentation_scale"))
    if np.isfinite(max_scale) and max_scale > 0.0:
        result["max_scale"] = float(max_scale)
        result["max_scale_source"] = "feature_log2"

    scale_scope = str(aug_cfg.FEATURE_LOG2_SCALE_SCOPE).strip().lower().replace("-", "_")
    per_cell_scales = {
        str(cell_name): float(scale)
        for cell_name, raw_scale in dict(stats.get("feature_log2_per_cell_max_scales", {}) or {}).items()
        if np.isfinite(scale := finite_float(raw_scale)) and scale > 0.0
    }
    if scale_scope == "per_cell" and per_cell_scales:
        result["max_scale"] = 1.0
        result["max_scale_source"] = "feature_log2_per_cell"
        result["max_scale_by_source"] = per_cell_scales

    metadata_prefixes = (
        "feature_log_",
        "feature_log2_",
        "feature_coverage_",
        "feature_selection_",
    )
    result["max_scale_metadata"] = {
        key: value
        for key, value in stats.items()
        if str(key).startswith(metadata_prefixes)
        or key in {
            FEATURE_LOG_REGRESSION_DIAGNOSTIC_KEY,
            FEATURE_LOG_MODEL_INPUT_REGRESSION_DIAGNOSTIC_KEY,
        }
    }
    return result

def _filter_matching_specs_by_caps(
    specs: list[dict[str, object]],
    *,
    max_scale: float,
    max_scale_by_source: dict[str, float] | None = None,
) -> tuple[list[dict[str, object]], int]:
    source_caps = {
        str(cell_name): float(scale)
        for cell_name, raw_scale in dict(max_scale_by_source or {}).items()
        if np.isfinite(scale := finite_float(raw_scale)) and scale > 0.0
    }
    kept: list[dict[str, object]] = []
    rejected_by_scale = 0
    for spec in specs:
        scale = finite_float(spec.get("scale"))
        cell = spec.get("train_cell", {})
        source_name = str(cell.get("cell", "")) if isinstance(cell, dict) else ""
        source_scale_cap = source_caps.get(source_name, max_scale)
        if np.isfinite(source_scale_cap) and np.isfinite(scale) and scale > source_scale_cap:
            rejected_by_scale += 1
            continue
        kept.append(spec)
    return kept, rejected_by_scale

def build_matching_augmentation_stats(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    *,
    task_kind: str,
    input_reference_value: float,
    matching_feature_columns: list[str] | tuple[str, ...] | None = None,
    max_scale: float | None = None,
    max_scale_source: str = "",
    max_scale_by_source: dict[str, float] | None = None,
    max_scale_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    train_targets = np.asarray([finite_float(cell.get("target_value")) for cell in train_cells], dtype=float)
    train_targets = train_targets[np.isfinite(train_targets) & (train_targets > 0.0)]
    test_anchor_values = collect_anchor_values(module, test_cells, task_kind=str(task_kind))
    test_anchor_values = test_anchor_values[np.isfinite(test_anchor_values) & (test_anchor_values > 0.0)]
    train_q = (np.quantile(train_targets, [0.25, 0.50, 0.75]).astype(float)
        if train_targets.size else np.full(3, np.nan, dtype=float))
    test_q = (np.quantile(test_anchor_values, [0.25, 0.50, 0.75]).astype(float)
        if test_anchor_values.size else np.full(3, np.nan, dtype=float))
    test_max = float(np.max(test_anchor_values)) if test_anchor_values.size else np.nan
    stats = {
        "train_q25": float(train_q[0]) if np.isfinite(train_q[0]) else np.nan,
        "train_q50": float(train_q[1]) if np.isfinite(train_q[1]) else np.nan,
        "train_q75": float(train_q[2]) if np.isfinite(train_q[2]) else np.nan,
        "test_q25": float(test_q[0]) if np.isfinite(test_q[0]) else np.nan,
        "test_q50": float(test_q[1]) if np.isfinite(test_q[1]) else np.nan,
        "test_q75": float(test_q[2]) if np.isfinite(test_q[2]) else np.nan,
        "test_max": float(test_max) if np.isfinite(test_max) else np.nan,
        "slowest_test_anchor": float(test_max) if np.isfinite(test_max) else np.nan,
        "max_augmented_anchor_allowed": float(test_max) if np.isfinite(test_max) else np.nan,
        "max_augmented_anchor_median_allowed": float(test_max) if np.isfinite(test_max) else np.nan,
        "max_coverage_scale": np.nan,
        "max_anchor_scale": np.nan,
        "target_total_train_count": int(len(train_cells)),
        "target_aug_count": 0,
        "target_scale_count": 0,
        "dense_small_threshold_augmentation": 0,
    }
    specs, test_sohs = _matching_specs(
        module,
        train_cells,
        test_cells,
        task_kind=str(task_kind),
        input_reference_value=float(input_reference_value),
        matching_feature_columns=matching_feature_columns,
    )
    raw_spec_count = int(len(specs))
    raw_soh_match_count = sum(
        str(spec.get("matching_rule", "")) == "warp_train_to_test_soh_at_input_reference" for spec in specs
    )
    raw_feature_fallback_count = int(raw_spec_count - raw_soh_match_count)
    matching_features_used = sorted(
        {
            str(feature_name)
            for spec in specs
            for feature_name in (
                spec.get("target_features", {}).keys()
                if isinstance(spec.get("target_features", {}), dict)
                else []
            )
        }
    )
    scale_cap = _matching_scale_cap(max_scale)
    specs, rejected_by_scale = _filter_matching_specs_by_caps(
        specs,
        max_scale=scale_cap,
        max_scale_by_source=max_scale_by_source,
    )
    scales = np.asarray([finite_float(spec.get("scale")) for spec in specs], dtype=float)
    scales = scales[np.isfinite(scales) & (scales > 0.0)]
    source_references = np.asarray([finite_float(spec.get("source_reference")) for spec in specs], dtype=float)
    source_references = source_references[np.isfinite(source_references) & (source_references > 0.0)]
    test_sohs = test_sohs[np.isfinite(test_sohs)]
    stats = dict(stats)
    stats.update(
        {
            "augmentation_method": "matching",
            "scale_spacing_mode": "matching_soh_then_feature_median_at_input_reference",
            "scale_step": np.nan,
            "target_aug_count": int(len(specs)),
            "target_total_train_count": int(len(train_cells) + len(specs)),
            "target_scale_count": int(len(scales)),
            "low_scale_candidate_count": 0,
            "dense_small_threshold_augmentation": 0,
            "matching_input_reference_value": float(input_reference_value),
            "matching_candidate_count_raw": int(raw_spec_count),
            "matching_candidate_count": int(len(specs)),
            "matching_soh_candidate_count_raw": int(raw_soh_match_count),
            "matching_feature_columns": ",".join(matching_features_used),
            "matching_var_delta_q_fallback_count_raw": int(raw_feature_fallback_count),
            "matching_var_delta_q_fallback_count": int(
                sum(
                    str(spec.get("matching_rule", ""))
                    != "warp_train_to_test_soh_at_input_reference"
                    for spec in specs
                )
            ),
            "matching_feature_fallback_count_raw": int(raw_feature_fallback_count),
            "matching_feature_fallback_count": int(
                sum(
                    str(spec.get("matching_rule", ""))
                    != "warp_train_to_test_soh_at_input_reference"
                    for spec in specs
                )
            ),
            "matching_rejected_by_scale_cap": int(rejected_by_scale),
            "matching_max_scale_allowed": float(scale_cap) if np.isfinite(scale_cap) else np.nan,
            "matching_max_scale_source": str(max_scale_source),
            "matching_per_cell_scale_cap_count": int(len(max_scale_by_source or {})),
            "matching_test_soh_count": int(test_sohs.size),
            "matching_test_soh_min": float(np.min(test_sohs)) if test_sohs.size else np.nan,
            "matching_test_soh_median": float(np.median(test_sohs)) if test_sohs.size else np.nan,
            "matching_test_soh_mean": float(np.mean(test_sohs)) if test_sohs.size else np.nan,
            "matching_test_soh_max": float(np.max(test_sohs)) if test_sohs.size else np.nan,
            "matching_source_reference_min": float(np.min(source_references)) if source_references.size else np.nan,
            "matching_source_reference_median": float(np.median(source_references)) if source_references.size else np.nan,
            "matching_source_reference_max": float(np.max(source_references)) if source_references.size else np.nan,
            "matching_scale_median": float(np.median(scales)) if scales.size else np.nan,
            "matching_scale_mean": float(np.mean(scales)) if scales.size else np.nan,
            "matching_scale_max": float(np.max(scales)) if scales.size else np.nan,
            "max_anchor_scale": float(np.max(scales)) if scales.size else np.nan,
        }
    )
    if max_scale_metadata:
        stats.update(dict(max_scale_metadata))
    return stats

def build_matching_augmented_cells(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    feature_columns: list[str],
    *,
    task_kind: str,
    input_reference_value: float,
    output_x_drop_count: int = 0,
    source_kind: str = "cyclic_augmented",
    enforce_monotonic_soh: bool = True,
    max_scale: float | None = None,
    max_scale_source: str = "",
    max_scale_by_source: dict[str, float] | None = None,
    matching_feature_columns: list[str] | tuple[str, ...] | None = None,
    max_scale_metadata: dict[str, object] | None = None,
    matching_specs: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if not train_cells or not test_cells:
        return []
    feature_columns = [str(name) for name in feature_columns]
    normalized_task_kind = str(task_kind).strip().lower()
    reference_column = resolve_reference_column(module, normalized_task_kind)
    input_reference = finite_float(input_reference_value)
    if not np.isfinite(input_reference) or input_reference <= 0.0:
        return []
    if matching_specs is None:
        specs, _test_sohs = _matching_specs(
            module,
            train_cells,
            test_cells,
            task_kind=str(normalized_task_kind),
            input_reference_value=float(input_reference),
            matching_feature_columns=matching_feature_columns if matching_feature_columns is not None else feature_columns,
        )
    else:
        specs = [dict(spec) for spec in matching_specs]
    scale_cap = _matching_scale_cap(max_scale)
    source_scale_caps = dict(max_scale_by_source or {})
    specs, _rejected_by_scale = _filter_matching_specs_by_caps(
        specs,
        max_scale=scale_cap,
        max_scale_by_source=max_scale_by_source,
    )
    generated_cells: list[dict[str, object]] = []
    for spec in specs:
        cell = spec["train_cell"]
        test_cell = spec["test_cell"]
        effective_scale = finite_float(spec.get("scale"))
        source_feature_reference = finite_float(spec.get("source_reference"))
        target_soh = finite_float(spec.get("target_soh"))
        target_feature = finite_float(spec.get("target_feature"))
        target_features = spec.get("target_features", {})
        if not isinstance(target_features, dict):
            target_features = {}
        source_feature_references = spec.get("source_feature_references", {})
        if not isinstance(source_feature_references, dict):
            source_feature_references = {}
        matching_rule = str(spec.get("matching_rule", "warp_train_to_test_soh_at_input_reference"))
        source_scale_cap = finite_float(
            source_scale_caps.get(str(cell.get("cell", ""))), scale_cap
        )
        if not np.isfinite(effective_scale) or effective_scale <= 0.0:
            continue
        target_week_value, target_throughput_value = resolve_augmented_targets(
            cell,
            task_kind=str(normalized_task_kind),
            scale=float(effective_scale),
        )
        target_value = float(target_week_value)
        active_target_value = (
            float(target_throughput_value)
            if normalized_task_kind == "throughput"
            else float(target_week_value)
        )
        if not np.isfinite(active_target_value) or active_target_value <= 0.0:
            continue
        if active_target_value <= float(input_reference):
            continue
        warped_raw_df = build_reference_axis_warped_raw_df(
            cell["raw_df"],
            reference_column=reference_column,
            scale=float(effective_scale),
        )
        feature_vector = module.interpolate_feature_vector(
            warped_raw_df,
            feature_columns,
            reference_column=reference_column,
            target_reference=float(input_reference),
        )
        feature_sequence = module.interpolate_feature_sequence(
            warped_raw_df,
            feature_columns,
            reference_column=reference_column,
            target_reference=float(input_reference),
            discretization_steps=int(getattr(module, "DISCRETIZATION_STEPS")),
        )
        if feature_vector is None or feature_sequence is None:
            continue
        feature_vector = np.asarray(feature_vector, dtype=float)
        feature_sequence = np.asarray(feature_sequence, dtype=float)
        if not np.isfinite(feature_vector).all() or not np.isfinite(feature_sequence).all():
            continue
        time_curve = np.asarray(cell["time_curve"], dtype=float) * float(effective_scale)
        throughput_curve = np.asarray(cell["throughput_curve"], dtype=float) * float(effective_scale)
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
        generated_index = len(generated_cells)
        generated_cells.append(
            {
                "cell": (
                    f"matching_aug_{generated_index:04d}_{cell['cell']}_"
                    f"to_{test_cell['cell']}_aug_{effective_scale:.3f}"
                ),
                "source_kind": str(source_kind),
                "x": output_x,
                "sequence": feature_sequence,
                "raw_df": warped_raw_df,
                "target_value": float(target_value),
                "target_week_value": float(target_week_value),
                "target_throughput_value": float(target_throughput_value),
                "input_reference_value": float(input_reference),
                "input_throughput_value": float(cell.get("input_throughput_value", np.nan)),
                "time_curve": time_curve,
                "throughput_curve": throughput_curve,
                "soh_curve": soh_curve,
                "augmentation_scale": float(effective_scale),
                "matching_scale": float(effective_scale),
                "matching_source_cell": str(cell["cell"]),
                "matching_test_cell": str(test_cell["cell"]),
                "matching_target_soh": float(target_soh) if np.isfinite(target_soh) else np.nan,
                "matching_target_var_delta_q": (
                    float(target_feature) if np.isfinite(target_feature) else np.nan
                ),
                "matching_target_features": ";".join(
                    f"{feature_name}:{float(value):.8g}"
                    for feature_name, value in sorted(target_features.items())
                    if np.isfinite(finite_float(value))
                ),
                "matching_source_feature_references": ";".join(
                    f"{feature_name}:{float(value):.8g}"
                    for feature_name, value in sorted(source_feature_references.items())
                    if np.isfinite(finite_float(value))
                ),
                "matching_feature_columns": ",".join(str(name) for name in sorted(target_features)),
                "matching_input_reference_value": float(input_reference),
                "matching_max_scale_allowed": (
                    float(source_scale_cap) if np.isfinite(source_scale_cap) else np.nan
                ),
                "matching_max_scale_source": (
                    "feature_log2_per_cell"
                    if str(cell.get("cell", "")) in source_scale_caps
                    else str(max_scale_source)
                ),
                "matching_source_reference_value": (
                    float(source_feature_reference) if np.isfinite(source_feature_reference) else np.nan
                ),
                "augmented_input_reference_value": float(input_reference),
                "source_feature_reference_value": (
                    float(source_feature_reference) if np.isfinite(source_feature_reference) else np.nan
                ),
                "matching_rule": str(matching_rule),
                "matching_anchor_scales": str(spec.get("anchor_scales", "")),
                "matching_scale_cv": finite_float(spec.get("scale_cv")),
                "matching_feature_distance": finite_float(spec.get("feature_distance")),
                "matching_donor_rank": int(spec.get("donor_rank", 0) or 0),
            }
        )
    return generated_cells


def _select_structured_matching_donor(
    module: Any,
    train_cells: list[dict[str, object]],
    source_cell: dict[str, object],
    *,
    task_kind: str,
    target_soh: float,
) -> tuple[dict[str, object] | None, float]:
    source_name = str(source_cell.get("cell", ""))
    source_target = finite_float(source_cell.get("target_value"))
    candidates: list[tuple[float, str, dict[str, object], float]] = []
    for donor_cell in train_cells:
        donor_name = str(donor_cell.get("cell", ""))
        if donor_name == source_name:
            continue
        donor_reference = _reference_at_soh(
            module,
            donor_cell,
            task_kind=str(task_kind),
            target_soh=float(target_soh),
        )
        if not np.isfinite(donor_reference) or donor_reference <= 0.0:
            continue
        donor_target = finite_float(donor_cell.get("target_value"))
        target_distance = (
            abs(float(np.log(donor_target / source_target)))
            if np.isfinite(source_target) and source_target > 0.0 and np.isfinite(donor_target) and donor_target > 0.0
            else np.inf
        )
        candidates.append((float(target_distance), donor_name, donor_cell, float(donor_reference)))
    if not candidates:
        return None, np.nan
    _distance, _name, donor_cell, donor_reference = min(candidates, key=lambda item: (item[0], item[1]))
    return donor_cell, float(donor_reference)


def _apply_structured_matching_residual(
    raw_df: pd.DataFrame,
    donor_cell: dict[str, object],
    feature_columns: list[str],
    *,
    reference_column: str,
    input_reference_value: float,
    donor_reference_value: float,
    blend: float,
) -> tuple[pd.DataFrame, list[str]]:
    donor_df = donor_cell.get("raw_df")
    if not isinstance(donor_df, pd.DataFrame) or reference_column not in raw_df or reference_column not in donor_df:
        return raw_df, []
    source_reference = pd.to_numeric(raw_df[reference_column], errors="coerce").to_numpy(dtype=float)
    donor_reference = pd.to_numeric(donor_df[reference_column], errors="coerce").to_numpy(dtype=float)
    source_progress = source_reference / float(input_reference_value)
    donor_progress = donor_reference / float(donor_reference_value)
    source_columns = {str(column).strip().lower(): str(column) for column in raw_df.columns}
    donor_columns = {str(column).strip().lower(): str(column) for column in donor_df.columns}
    result = raw_df.copy()
    perturbed_columns: list[str] = []

    for feature_name in feature_columns:
        normalized_name = str(feature_name).strip().lower()
        source_column = source_columns.get(normalized_name)
        donor_column = donor_columns.get(normalized_name)
        if source_column is None or donor_column is None:
            continue
        source_values = pd.to_numeric(raw_df[source_column], errors="coerce").to_numpy(dtype=float)
        donor_values = pd.to_numeric(donor_df[donor_column], errors="coerce").to_numpy(dtype=float)
        donor_x, donor_y = _prepare_reference_feature_curve(donor_progress, donor_values)
        valid = (
            np.isfinite(source_progress)
            & np.isfinite(source_values)
            & (source_progress >= 0.0)
            & (source_progress <= 1.0)
        )
        if np.count_nonzero(valid) < 3 or donor_x.size < 2:
            continue
        progress = source_progress[valid]
        source_curve = source_values[valid]
        donor_curve = np.interp(progress, donor_x, donor_y)
        residual = donor_curve - source_curve
        order = np.argsort(progress)
        smooth_window = max(1, min(9, int(np.count_nonzero(valid) // 6) * 2 + 1))
        if smooth_window > 1:
            smooth = pd.Series(residual[order]).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
            residual[order] = smooth
        low_index = int(np.argmin(progress))
        high_index = int(np.argmax(progress))
        span = float(progress[high_index] - progress[low_index])
        if span <= 0.0:
            continue
        endpoint_trend = residual[low_index] + (
            (progress - progress[low_index]) / span
        ) * (residual[high_index] - residual[low_index])
        perturbed = source_curve + float(blend) * (residual - endpoint_trend)
        if normalized_name.startswith("var_"):
            perturbed = np.maximum(perturbed, 0.0)
        if normalized_name == "mean_delta_q":
            perturbed[low_index] = 0.0
        result.loc[valid, source_column] = perturbed
        perturbed_columns.append(source_column)
    return result, perturbed_columns


def build_matching_structured_augmented_cells(
    module: Any,
    train_cells: list[dict[str, object]],
    test_cells: list[dict[str, object]],
    feature_columns: list[str],
    *,
    task_kind: str,
    input_reference_value: float,
    structured_blend: float = 0.5,
    **matching_kwargs: object,
) -> list[dict[str, object]]:
    generated_cells = build_matching_augmented_cells(
        module,
        train_cells,
        test_cells,
        feature_columns,
        task_kind=str(task_kind),
        input_reference_value=float(input_reference_value),
        **matching_kwargs,
    )
    reference_column = resolve_reference_column(module, str(task_kind))
    train_by_name = {str(cell.get("cell", "")): cell for cell in train_cells}
    donor_cache: dict[tuple[str, float], tuple[dict[str, object] | None, float]] = {}
    output_x_drop_count = int(matching_kwargs.get("output_x_drop_count", 0) or 0)

    for generated_cell in generated_cells:
        source_name = str(generated_cell.get("matching_source_cell", ""))
        source_cell = train_by_name.get(source_name)
        target_soh = finite_float(generated_cell.get("matching_target_soh"))
        if source_cell is None or not np.isfinite(target_soh):
            continue
        cache_key = (source_name, round(float(target_soh), 8))
        if cache_key not in donor_cache:
            donor_cache[cache_key] = _select_structured_matching_donor(
                module,
                train_cells,
                source_cell,
                task_kind=str(task_kind),
                target_soh=float(target_soh),
            )
        donor_cell, donor_reference = donor_cache[cache_key]
        if donor_cell is None or not np.isfinite(donor_reference):
            continue
        structured_df, perturbed_columns = _apply_structured_matching_residual(
            generated_cell["raw_df"],
            donor_cell,
            feature_columns,
            reference_column=reference_column,
            input_reference_value=float(input_reference_value),
            donor_reference_value=float(donor_reference),
            blend=float(structured_blend),
        )
        if not perturbed_columns:
            continue
        feature_vector = module.interpolate_feature_vector(
            structured_df,
            feature_columns,
            reference_column=reference_column,
            target_reference=float(input_reference_value),
        )
        feature_sequence = module.interpolate_feature_sequence(
            structured_df,
            feature_columns,
            reference_column=reference_column,
            target_reference=float(input_reference_value),
            discretization_steps=int(getattr(module, "DISCRETIZATION_STEPS")),
        )
        if feature_vector is None or feature_sequence is None:
            continue
        feature_vector = np.asarray(feature_vector, dtype=float)
        feature_sequence = np.asarray(feature_sequence, dtype=float)
        if not np.isfinite(feature_vector).all() or not np.isfinite(feature_sequence).all():
            continue
        generated_cell["raw_df"] = structured_df
        generated_cell["x"] = feature_vector[output_x_drop_count:]
        generated_cell["sequence"] = feature_sequence
        generated_cell["cell"] = str(generated_cell.get("cell", "")).replace(
            "matching_aug_", "matching_structured_aug_", 1
        )
        generated_cell["augmentation_method"] = "matching_structured"
        generated_cell["structured_matching_donor_cell"] = str(donor_cell.get("cell", ""))
        generated_cell["structured_matching_donor_reference_value"] = float(donor_reference)
        generated_cell["structured_matching_blend"] = float(structured_blend)
        generated_cell["structured_matching_perturbed_columns"] = ",".join(perturbed_columns)
        generated_cell["matching_rule"] = f"{generated_cell.get('matching_rule', 'matching')}+structured_training_residual"
    return generated_cells
