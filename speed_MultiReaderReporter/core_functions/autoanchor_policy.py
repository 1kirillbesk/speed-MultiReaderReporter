from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


LINEAR_TARGET_KEY_FIXED = "fixed"
LINEAR_TARGET_KEY_MINTEST = "mintest"
AUTOANCHOR_SELECTION_RULE = "lowest_valid_anchor_with_input_window_within_week_threshold"
LINEAR_MIN_TEST_INPUT_RULE = "minimum_test_cell_cumulative_throughput_at_week_threshold"


@dataclass(frozen=True)
class ThroughputLinearInputSpec:
    target_key: str
    target_limit: float
    target_label: str
    display_name: str
    model_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target_key": str(self.target_key),
            "target_limit": float(self.target_limit),
            "target_label": str(self.target_label),
            "display_name": str(self.display_name),
            "model_name": str(self.model_name),
        }


def format_percent(value: float) -> str:
    percent = float(value) * 100.0
    if np.isfinite(percent) and abs(percent - round(percent)) < 1e-9:
        return str(int(round(percent)))
    return f"{percent:.3f}".rstrip("0").rstrip(".")


def format_numeric(value: float) -> str:
    numeric = float(value)
    if np.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def build_anchor_candidates(
    *,
    min_soh: float = 0.90,
    max_soh: float = 0.99,
    step_soh: float = 0.01,
    descending: bool = False,
) -> tuple[float, ...]:
    if float(step_soh) <= 0.0:
        raise ValueError("step_soh must be positive.")
    values = np.arange(float(min_soh), float(max_soh) + 0.5 * float(step_soh), float(step_soh))
    values = np.round(values, 6)
    if descending:
        values = values[::-1]
    return tuple(float(value) for value in values)


def resolve_week_threshold_input_steps(week_threshold: float) -> int:
    return max(2, int(round(float(week_threshold))) + 1)


def select_lowest_valid_anchor(
    *,
    candidates: Sequence[float],
    evaluate_candidate: Callable[[float], tuple[bool, list[dict[str, object]]]],
    require_scan_all: bool = True,
) -> tuple[float, pd.DataFrame]:
    search_rows: list[dict[str, object]] = []
    selected_anchor: float | None = None

    for anchor_soh in candidates:
        is_valid, rows = evaluate_candidate(float(anchor_soh))
        search_rows.extend(rows)
        if is_valid:
            selected_anchor = float(anchor_soh)
            if not require_scan_all:
                break

    if selected_anchor is None:
        raise RuntimeError(
            "Could not find an interpolation anchor SOH in "
            f"{', '.join(format_percent(value) for value in candidates)}."
        )
    return float(selected_anchor), pd.DataFrame(search_rows)


def interpolate_value_at_week(weeks: object, values: object, week_threshold: float) -> float:
    week_array = np.asarray(weeks, dtype=float).reshape(-1)
    value_array = np.asarray(values, dtype=float).reshape(-1)
    point_count = min(int(week_array.size), int(value_array.size))
    if point_count == 0:
        return np.nan
    week_array = week_array[:point_count]
    value_array = value_array[:point_count]
    valid = np.isfinite(week_array) & np.isfinite(value_array)
    if not np.any(valid):
        return np.nan
    week_array = week_array[valid]
    value_array = value_array[valid]
    order = np.argsort(week_array)
    week_array = week_array[order]
    value_array = value_array[order]
    if float(np.nanmin(week_array)) <= float(week_threshold) <= float(np.nanmax(week_array)):
        return float(np.interp(float(week_threshold), week_array, value_array))
    before_mask = week_array <= float(week_threshold)
    if np.any(before_mask):
        return float(value_array[before_mask][-1])
    return np.nan


def get_min_test_throughput_at_week(
    *,
    cell_names: Iterable[str],
    week_threshold: float,
    load_cell_week_throughput: Callable[[str], tuple[object, object] | None],
) -> float:
    throughput_values: list[float] = []
    for cell_name in cell_names:
        loaded = load_cell_week_throughput(str(cell_name))
        if loaded is None:
            continue
        weeks, throughput_values_raw = loaded
        week_array = np.asarray(weeks, dtype=float).reshape(-1)
        if week_array.size == 0 or not np.any(np.isfinite(week_array)):
            continue
        if float(np.nanmax(week_array[np.isfinite(week_array)])) < float(week_threshold):
            continue
        throughput_at_threshold = interpolate_value_at_week(
            week_array,
            throughput_values_raw,
            float(week_threshold),
        )
        if np.isfinite(throughput_at_threshold) and float(throughput_at_threshold) > 0.0:
            throughput_values.append(float(throughput_at_threshold))
    if not throughput_values:
        raise RuntimeError(
            f"Could not determine a usable minimum test cumulative throughput by week {float(week_threshold):.1f}."
        )
    return float(min(throughput_values))


def build_throughput_linear_input_specs(
    *,
    target_percent: int,
    fixed_input_throughput: float | None,
    min_test_throughput: float,
) -> list[dict[str, object]]:
    specs: list[ThroughputLinearInputSpec] = []
    if fixed_input_throughput is not None and np.isfinite(float(fixed_input_throughput)):
        specs.append(
            ThroughputLinearInputSpec(
                target_key=LINEAR_TARGET_KEY_FIXED,
                target_limit=float(fixed_input_throughput),
                target_label=f"{format_numeric(float(fixed_input_throughput))} Ah",
                display_name="Linear raw",
                model_name="linear",
            )
        )
    specs.append(
        ThroughputLinearInputSpec(
            target_key=LINEAR_TARGET_KEY_MINTEST,
            target_limit=float(min_test_throughput),
            target_label=f"{format_numeric(float(min_test_throughput))} Ah",
            display_name="Linear @min-test",
            model_name=f"linear_mintest_eol{int(target_percent)}",
        )
    )
    return [spec.to_dict() for spec in specs]
