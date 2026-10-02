from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd


def build_target_column(target_soh: float) -> tuple[int, str]:
    target_percent = int(round(float(target_soh) * 100))
    return target_percent, f"EOL{target_percent}"


def build_target_columns(
    target_soh: float,
    extra_target_sohs: tuple[float, ...] = (),
) -> list[tuple[float, int, str]]:
    ordered: list[float] = []
    for soh in [*extra_target_sohs, target_soh]:
        normalized = float(soh)
        if any(abs(normalized - existing) < 1e-9 for existing in ordered):
            continue
        ordered.append(normalized)
    return [(soh, *build_target_column(soh)) for soh in ordered]


def clean_numeric_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: list[str] | tuple[str, ...],
    sort_column: str,
    trim_after_last_valid_capacity: bool = True,
    capacity_column: str | None = None,
) -> pd.DataFrame:
    required = set(required_columns)
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    result = df.copy()
    numeric_columns: list[str] = []
    for column in result.columns:
        converted = pd.to_numeric(result[column], errors="coerce")
        if column in required or converted.notna().any():
            result[column] = converted
            numeric_columns.append(column)

    result = result[numeric_columns].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=list(required)).sort_values(sort_column)
    if result.empty:
        raise ValueError(f"No finite rows available after cleaning on {list(required)}.")

    if trim_after_last_valid_capacity and capacity_column and capacity_column in result.columns:
        last_valid_index = result[capacity_column].last_valid_index()
        if last_valid_index is not None:
            result = result.loc[:last_valid_index].copy()
    return result.reset_index(drop=True)


def interpolate_value_at_soh(
    x_values: np.ndarray,
    values: np.ndarray,
    capacities: np.ndarray,
    target_soh: float,
) -> float:
    finite_mask = np.isfinite(x_values) & np.isfinite(values) & np.isfinite(capacities)
    x_values = np.asarray(x_values, dtype=float)[finite_mask]
    values = np.asarray(values, dtype=float)[finite_mask]
    capacities = np.asarray(capacities, dtype=float)[finite_mask]
    if len(x_values) < 2 or capacities[0] == 0:
        return np.nan

    order = np.argsort(x_values)
    x_values = x_values[order]
    values = values[order]
    capacities = capacities[order]
    soh = capacities / capacities[0]
    crossing_indices = np.where(soh <= float(target_soh))[0]
    if len(crossing_indices) == 0:
        return np.nan

    crossing_idx = int(crossing_indices[0])
    if crossing_idx == 0:
        return float(values[0])

    value_before = values[crossing_idx - 1]
    value_after = values[crossing_idx]
    soh_before = soh[crossing_idx - 1]
    soh_after = soh[crossing_idx]
    if soh_after == soh_before:
        return float(value_after)
    return float(value_before + (target_soh - soh_before) * (value_after - value_before) / (soh_after - soh_before))


def build_uniform_reference_grid(
    reference: np.ndarray,
    *,
    anchor_reference: float,
    pre_anchor_points: int,
    append_final_reference: bool = False,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    if reference.size < 2:
        raise ValueError("Need at least two reference points for interpolation.")

    pre_anchor_points = max(2, int(pre_anchor_points))
    reference_start = float(reference[0])
    anchor_reference = float(anchor_reference)
    if not np.isfinite(anchor_reference) or anchor_reference < reference_start:
        raise ValueError("Interpolation anchor resolves before the first measured reference point.")

    new_ref_points_cut = np.linspace(reference_start, anchor_reference, pre_anchor_points)
    spacing = float(np.mean(np.diff(new_ref_points_cut))) if len(new_ref_points_cut) > 1 else 0.0
    remaining_points: list[float] = []
    current_point = anchor_reference
    while spacing > 0 and current_point + spacing <= float(reference[-1]):
        current_point += spacing
        remaining_points.append(current_point)

    new_ref_points = np.concatenate([new_ref_points_cut, np.asarray(remaining_points, dtype=float)])
    if append_final_reference and (
        new_ref_points.size == 0 or not np.isclose(float(new_ref_points[-1]), float(reference[-1]))
    ):
        new_ref_points = np.concatenate([new_ref_points, np.asarray([float(reference[-1])], dtype=float)])
    return new_ref_points


def interpolate_dataframe_on_reference(
    df: pd.DataFrame,
    *,
    reference_column: str,
    new_reference: np.ndarray,
    per_column_finite_mask: bool = False,
) -> pd.DataFrame:
    data_columns = [column for column in df.columns if column != reference_column]
    reference = pd.to_numeric(df[reference_column], errors="coerce").to_numpy(dtype=float)
    new_reference = np.asarray(new_reference, dtype=float)
    interpolated_columns: dict[str, np.ndarray] = {}

    for column in data_columns:
        column_values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(reference) & np.isfinite(column_values)
        if finite_mask.sum() < 2:
            interpolated_columns[column] = np.full_like(new_reference, np.nan, dtype=float)
            continue
        reference_valid = reference[finite_mask] if per_column_finite_mask else reference
        values_valid = column_values[finite_mask] if per_column_finite_mask else column_values
        if not per_column_finite_mask:
            valid_reference_mask = np.isfinite(reference_valid) & np.isfinite(values_valid)
            reference_valid = reference_valid[valid_reference_mask]
            values_valid = values_valid[valid_reference_mask]
        interpolated_columns[column] = np.interp(new_reference, reference_valid, values_valid)

    interpolated_df = pd.DataFrame(interpolated_columns)
    interpolated_df[reference_column] = new_reference
    return interpolated_df


def add_step_target_columns(
    interpolated_df: pd.DataFrame,
    *,
    capacity_column: str,
    target_soh: float,
    target_column: str | None = None,
    extra_target_sohs: tuple[float, ...] = (),
    strict_primary_target: bool = True,
) -> pd.DataFrame:
    result = interpolated_df.copy()
    capacities = pd.to_numeric(result[capacity_column], errors="coerce").to_numpy(dtype=float)
    step_indices = np.arange(len(result), dtype=float)
    resolved_target_column = target_column or build_target_column(target_soh)[1]
    for candidate_target_soh, _, candidate_target_column in build_target_columns(target_soh, extra_target_sohs):
        target_value = interpolate_value_at_soh(step_indices, step_indices, capacities, candidate_target_soh)
        is_primary_target = abs(float(candidate_target_soh) - float(target_soh)) < 1e-9
        output_column = resolved_target_column if is_primary_target else candidate_target_column
        if not np.isfinite(target_value):
            if is_primary_target and strict_primary_target:
                raise ValueError(f"Cell does not reach the requested target SOH {float(target_soh):.2f}.")
            result[output_column] = np.nan
            continue
        result[output_column] = float(target_value)
    return result


def load_and_interpolate_generic(
    df: pd.DataFrame,
    *,
    capacity_column: str,
    reference_column: str,
    anchor_soh: float,
    target_soh: float,
    pre_anchor_points: int,
    target_column: str | None = None,
    extra_target_sohs: tuple[float, ...] = (),
    append_final_reference: bool = False,
    per_column_finite_mask: bool = False,
    strict_primary_target: bool = True,
) -> pd.DataFrame:
    result = deepcopy(df)
    initial_capacity = float(pd.to_numeric(result[capacity_column], errors="coerce").iloc[0])
    if not np.isfinite(initial_capacity) or initial_capacity <= 0:
        raise ValueError("Invalid initial capacity.")

    result["SOH"] = pd.to_numeric(result[capacity_column], errors="coerce") / initial_capacity
    result = result.sort_values(reference_column).reset_index(drop=True)
    reference = pd.to_numeric(result[reference_column], errors="coerce").to_numpy(dtype=float)
    reference, unique_indices = np.unique(reference, return_index=True)
    result = result.iloc[unique_indices].reset_index(drop=True)
    reference = pd.to_numeric(result[reference_column], errors="coerce").to_numpy(dtype=float)
    capacities = pd.to_numeric(result[capacity_column], errors="coerce").to_numpy(dtype=float)
    if len(reference) < 2:
        raise ValueError(f"Not enough unique {reference_column} points for interpolation.")

    anchor_reference = interpolate_value_at_soh(reference, reference, capacities, anchor_soh)
    if not np.isfinite(anchor_reference):
        raise ValueError(f"Could not compute {reference_column} at SOH {float(anchor_soh):.2f}.")

    new_reference = build_uniform_reference_grid(
        reference,
        anchor_reference=float(anchor_reference),
        pre_anchor_points=pre_anchor_points,
        append_final_reference=append_final_reference,
    )
    interpolated_df = interpolate_dataframe_on_reference(
        result,
        reference_column=reference_column,
        new_reference=new_reference,
        per_column_finite_mask=per_column_finite_mask,
    )
    return add_step_target_columns(
        interpolated_df,
        capacity_column=capacity_column,
        target_soh=target_soh,
        target_column=target_column,
        extra_target_sohs=extra_target_sohs,
        strict_primary_target=strict_primary_target,
    )
