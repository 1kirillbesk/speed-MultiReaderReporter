from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from exported_core_functions import get_exported_augmentation_methods_module
from lfp_prediction_common import (
    DEFAULT_ANCHOR_SOH,
    DEFAULT_FEATURE_COLS,
    DEFAULT_INPUT_END_THROUGHPUT,
    DEFAULT_INPUT_END_WEEK,
    DEFAULT_INPUT_STEPS,
    DEFAULT_TARGET_COL,
    LFPInterpolationModule,
    RAW_FEATURE_DIR,
    _build_exported_feature_scale_metadata,
    build_augmented_cells,
    build_model_cells,
    infer_train_test_names,
    load_lfp_cell_frames,
    plot_feature_log2_helper_diagnostics,
    resolve_week_step_count,
)


DEFAULT_AUG_TARGET_SOH = 0.92
DEFAULT_AUG_TEST_NAMES = tuple(f"SPEED_LW_reference_{index}" for index in range(1, 10))
DEFAULT_OUTPUT_PATH = Path("generated_plots") / "lfp_feature_log2_var_delta_q_helper_model.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the feature_log2 helper regression based on var_delta_Q for the LFP augmentation workflow."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_FEATURE_DIR)
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument("--target-soh", type=float, default=DEFAULT_AUG_TARGET_SOH)
    parser.add_argument("--input-steps", type=int, default=DEFAULT_INPUT_STEPS)
    parser.add_argument(
        "--test-names",
        nargs="*",
        default=list(DEFAULT_AUG_TEST_NAMES),
        help="Explicit test cells. If none of them exist, the script falls back to all *reference* cells.",
    )
    parser.add_argument("--anchor-week", type=float, default=DEFAULT_INPUT_END_WEEK)
    parser.add_argument("--anchor-throughput", type=float, default=DEFAULT_INPUT_END_THROUGHPUT)
    parser.add_argument("--anchor-mode", choices=("throughput", "week"), default="throughput")
    parser.add_argument("--anchor-soh", type=float, default=DEFAULT_ANCHOR_SOH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--title", type=str, default="LFP feature_log2 helper model")
    parser.add_argument("--no-annotate", action="store_false", dest="annotate")
    parser.set_defaults(annotate=True)
    return parser.parse_args()


def _build_helper_model_inputs(args: argparse.Namespace) -> dict[str, object]:
    feature_columns = list(DEFAULT_FEATURE_COLS)
    cell_frames, skipped_frames = load_lfp_cell_frames(
        interp_dir=Path(args.data_dir),
        feature_columns=feature_columns,
        target_column=str(args.target_col),
        target_soh=float(args.target_soh) if args.target_soh is not None else None,
    )
    all_names = [str(frame["cell"]) for frame in cell_frames]
    train_names, test_names = infer_train_test_names(
        all_names,
        explicit_test_names=list(args.test_names),
    )
    if not train_names or not test_names:
        raise RuntimeError("Could not infer both training and test cells for helper-model plotting.")
    train_frames = [frame for frame in cell_frames if str(frame["cell"]) in set(train_names)]
    test_frames = [frame for frame in cell_frames if str(frame["cell"]) in set(test_names)]
    task_kind = "throughput" if str(args.anchor_mode) == "throughput" else "time"
    anchor_week = float(args.anchor_week) if task_kind == "week" else None
    anchor_throughput = float(args.anchor_throughput) if task_kind == "throughput" else None
    effective_input_steps = resolve_week_step_count(
        [*train_frames, *test_frames],
        anchor_week=anchor_week,
        anchor_throughput=anchor_throughput,
        anchor_soh=float(args.anchor_soh) if anchor_week is None and anchor_throughput is None else None,
        fallback_steps=int(args.input_steps),
    )
    train_cells, skipped_train = build_model_cells(
        train_frames,
        feature_columns=feature_columns,
        input_steps=int(effective_input_steps),
        anchor_soh=None if anchor_throughput is not None or anchor_week is not None else float(args.anchor_soh),
        anchor_week=anchor_week,
        anchor_throughput=anchor_throughput,
        sequence_steps=int(effective_input_steps),
    )
    test_cells, skipped_test = build_model_cells(
        test_frames,
        feature_columns=feature_columns,
        input_steps=int(effective_input_steps),
        anchor_soh=None if anchor_throughput is not None or anchor_week is not None else float(args.anchor_soh),
        anchor_week=anchor_week,
        anchor_throughput=anchor_throughput,
        sequence_steps=int(effective_input_steps),
    )
    if not train_cells or not test_cells:
        raise RuntimeError("No usable train/test cells remained after building model inputs.")

    augmented_cells, metadata = build_augmented_cells(
        train_cells,
        test_cells,
        feature_columns=feature_columns,
        input_steps=int(effective_input_steps),
        sequence_steps=int(effective_input_steps),
        anchor_soh=float(args.anchor_soh),
        augmentation_method="feature_log2",
        task_kind=task_kind,
    )
    exported_aug = get_exported_augmentation_methods_module()
    diagnostic = metadata.get(exported_aug.FEATURE_LOG_MODEL_INPUT_REGRESSION_DIAGNOSTIC_KEY)
    if not isinstance(diagnostic, dict):
        raise RuntimeError("Helper-model diagnostic was not produced.")
    module = LFPInterpolationModule(
        input_steps=int(effective_input_steps),
        augmentation_reference_soh=float(args.anchor_soh),
        max_sequence_steps=int(effective_input_steps),
    )
    prepared_train = [
        exported_aug.preserve_process_features(cell, feature_columns)
        for cell in train_cells
    ]
    prepared_test = [
        exported_aug.preserve_process_features(cell, feature_columns)
        for cell in test_cells
    ]
    selected_helper_features, feature_scale_metadata, scales = _build_exported_feature_scale_metadata(
        exported_aug,
        module,
        prepared_train,
        prepared_test,
        feature_columns=feature_columns,
        task_kind=task_kind,
        input_reference_value=float(metadata.get("augmentation_input_reference_value", np.nan)),
    )
    metadata = {**feature_scale_metadata, **metadata}
    return {
        "feature_columns": feature_columns,
        "effective_input_steps": int(effective_input_steps),
        "train_cells": train_cells,
        "augmented_cells": augmented_cells,
        "test_cells": test_cells,
        "metadata": metadata,
        "diagnostic": diagnostic,
        "selected_helper_features": selected_helper_features,
        "scales": [float(scale) for scale in scales],
        "skipped_frames": skipped_frames,
        "skipped_train": skipped_train,
        "skipped_test": skipped_test,
        "task_kind": task_kind,
    }


def main() -> None:
    args = parse_args()
    helper_data = _build_helper_model_inputs(args)
    output_path = Path(args.output)
    plot_feature_log2_helper_diagnostics(
        train_real_cells=list(helper_data["train_cells"]),
        augmented_cells=list(helper_data["augmented_cells"]),
        test_cells=list(helper_data["test_cells"]),
        metadata=dict(helper_data["metadata"]),
        title_prefix=str(args.title),
        show=False,
        output_path=output_path,
    )
    print(f"Saved helper-model plot to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
