from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from lfp_prediction_common import (
    DEFAULT_ANCHOR_SOH,
    DEFAULT_FEATURE_COLS,
    DEFAULT_INPUT_END_THROUGHPUT,
    DEFAULT_INPUT_END_WEEK,
    DEFAULT_INPUT_STEPS,
    DEFAULT_SEED_COUNT,
    DEFAULT_SEED_SOURCE,
    DEFAULT_TARGET_COL,
    DEFAULT_TARGET_SOH,
    RAW_FEATURE_DIR,
    TrainingConfig,
    plot_feature_log2_helper_diagnostics,
    supported_augmentation_methods,
    run_lfp_prediction_pipeline,
)

DEFAULT_AUTOANCHOR_TEST_NAMES = tuple(
    f"SPEED_LW_reference_{index}" for index in range(4, 7)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the LFP augmented TCNN on the raw cell_feature CSV files used upstream "
            "by analyze_3dim_top_selecttime_linear, using a fixed raw input cutoff "
            "in week or throughput and a 95% SOH end-of-life target."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_FEATURE_DIR)
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument("--target-soh", type=float, default=DEFAULT_TARGET_SOH)
    parser.add_argument("--input-steps", type=int, default=DEFAULT_INPUT_STEPS)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--seed-source", type=int, default=DEFAULT_SEED_SOURCE)
    parser.add_argument(
        "--test-names",
        nargs="*",
        default=list(DEFAULT_AUTOANCHOR_TEST_NAMES),
        help="Explicit test cells. If none of them exist, the script falls back to all *reference* cells.",
    )
    parser.add_argument(
        "--augmentation-method",
        choices=supported_augmentation_methods(),
        default="feature_log2",
    )
    parser.add_argument(
        "--augmentation-sample-count",
        type=int,
        default=None,
        help=(
            "Optional augmentation count (method-dependent)."
        ),
    )
    parser.add_argument("--anchor-week", type=float, default=DEFAULT_INPUT_END_WEEK)
    parser.add_argument("--anchor-throughput", type=float, default=DEFAULT_INPUT_END_THROUGHPUT)
    parser.add_argument("--anchor-mode", choices=("throughput", "week"), default="throughput")
    parser.add_argument("--anchor-soh", type=float, default=DEFAULT_ANCHOR_SOH)
    parser.add_argument("--no-show", action="store_false", dest="show")
    parser.set_defaults(show=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_config = TrainingConfig(
        input_steps=int(args.input_steps),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        dropout_rate=float(args.dropout),
        seed_count=int(args.seed_count),
        seed_source=int(args.seed_source),
    )
    result = run_lfp_prediction_pipeline(
        interp_dir=Path(args.data_dir),
        feature_columns=list(DEFAULT_FEATURE_COLS),
        target_column=str(args.target_col),
        target_soh=float(args.target_soh) if args.target_soh is not None else None,
        training_config=training_config,
        explicit_test_names=list(args.test_names),
        fixed_anchor_soh=None,
        fixed_anchor_week=float(args.anchor_week) if str(args.anchor_mode) == "week" and args.anchor_week is not None else None,
        fixed_anchor_throughput=float(args.anchor_throughput) if str(args.anchor_mode) == "throughput" and args.anchor_throughput is not None else None,
        autoanchor=False,
        augmentation_method=str(args.augmentation_method),
        augmentation_sample_count=args.augmentation_sample_count,
        show_plot=bool(args.show),
        plot_title_prefix=(
            f"LFP {str(args.augmentation_method).upper()} augmentation + "
            f"{('throughput ' + format(float(args.anchor_throughput), '.6g')) if str(args.anchor_mode) == 'throughput' else ('week ' + format(float(args.anchor_week), '.0f'))} cutoff"
        ),
    )

    if str(args.augmentation_method) == "feature_log2":
        figure_path = Path(args.data_dir).parent / "lfp_prediction_aug_autoanchor" / "var_delta_Q_helper_fit.png"
        plot_feature_log2_helper_diagnostics(
            train_real_cells=result["train_real_cells"],
            augmented_cells=result["augmented_cells"],
            test_cells=result["test_cells_data"],
            metadata=result["metadata"],
            title_prefix="LFP feature_log2 augmentation",
            show=False,
            output_path=figure_path,
        )
        print(f"var_delta_Q helper fit: {figure_path}")

    metadata = dict(result["metadata"])
    prediction_df = result["prediction_df"]
    if np.isfinite(float(metadata.get("selected_anchor_throughput", float('nan')))):
        print(f"Selected anchor throughput: {float(metadata['selected_anchor_throughput']):.6g}")
    else:
        print(f"Selected anchor week: {float(metadata['selected_anchor_week']):.2f}")
    print(f"Target SOH: {float(metadata['target_soh']):.4f}")
    print(
        f"Train real={int(metadata['train_real_count'])}  "
        f"train aug={int(metadata['train_aug_count'])}  "
        f"test={int(metadata['test_count'])}"
    )
    if metadata.get("augmentation_scales"):
        print(f"Augmentation scales: {metadata['augmentation_scales']}")
    print(f"Week target divisor: {float(result['week_target_divisor']):.6g}")
    print(f"Throughput target divisor: {float(result['throughput_target_divisor']):.6g}")
    print(f"RMSE (weeks): {float(result['rmse_weeks']):.2f}")
    print(f"MAPE (weeks): {float(result['mape_weeks']):.2f}%")
    print(f"RMSE (throughput): {float(result['rmse_throughput']):.2f}")
    print(f"MAPE (throughput): {float(result['mape_throughput']):.2f}%")
    print(prediction_df.to_string(index=False))


if __name__ == "__main__":
    main()
