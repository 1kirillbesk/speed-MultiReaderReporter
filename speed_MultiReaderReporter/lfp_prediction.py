from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lfp_prediction_common import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_SEED_COUNT,
    DEFAULT_SEED_SOURCE,
    DEFAULT_TEST_NAMES,
    RAW_FEATURE_DIR,
    TrainingConfig,
    run_lfp_prediction_pipeline,
)


LEGACY_DEFAULT_TARGET_SOH = 0.96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the legacy LFP prediction entrypoint on the current raw "
            "cell_feature CSV files instead of the stale interp_feature folder."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_FEATURE_DIR)
    parser.add_argument("--target-soh", type=float, default=LEGACY_DEFAULT_TARGET_SOH)
    parser.add_argument("--input-steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--seed-source", type=int, default=DEFAULT_SEED_SOURCE)
    parser.add_argument(
        "--test-names",
        nargs="*",
        default=list(DEFAULT_TEST_NAMES),
        help="Explicit test cells. Defaults to SPEED_LW_reference_1..3.",
    )
    parser.add_argument("--no-show", action="store_false", dest="show")
    parser.set_defaults(show=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    training_config = TrainingConfig(
        input_steps=int(args.input_steps),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        dropout_rate=float(args.dropout),
        seed_count=int(args.seed_count),
        seed_source=int(args.seed_source),
    )

    print(f"Reading raw feature tables from: {data_dir}")
    print(f"Target SOH: {float(args.target_soh):.4f}")

    result = run_lfp_prediction_pipeline(
        interp_dir=data_dir,
        feature_columns=list(DEFAULT_FEATURE_COLS),
        target_column=None,
        target_soh=float(args.target_soh),
        training_config=training_config,
        explicit_test_names=list(args.test_names),
        fixed_anchor_soh=None,
        fixed_anchor_week=None,
        fixed_anchor_throughput=None,
        autoanchor=False,
        augmentation_method="none",
        augmentation_sample_count=None,
        show_plot=bool(args.show),
        plot_title_prefix=f"LFP prediction ({float(args.target_soh):.3f} SOH target)",
    )

    metadata = dict(result["metadata"])
    prediction_df = result["prediction_df"]

    print(f"Input sequence steps: {int(metadata['input_sequence_steps'])}")
    print(
        f"Train real={int(metadata['train_real_count'])}  "
        f"train aug={int(metadata['train_aug_count'])}  "
        f"test={int(metadata['test_count'])}"
    )
    print(f"RMSE (weeks): {float(result['rmse_weeks']):.2f}")
    print(f"MAPE (weeks): {float(result['mape_weeks']):.2f}%")
    print(f"RMSE (throughput): {float(result['rmse_throughput']):.2f}")
    print(f"MAPE (throughput): {float(result['mape_throughput']):.2f}%")

    if np.isfinite(float(metadata.get("selected_anchor_soh", np.nan))):
        print(f"Anchor SOH: {float(metadata['selected_anchor_soh']):.4f}")
    if np.isfinite(float(metadata.get("selected_anchor_week", np.nan))):
        print(f"Anchor week: {float(metadata['selected_anchor_week']):.2f}")
    if np.isfinite(float(metadata.get("selected_anchor_throughput", np.nan))):
        print(f"Anchor throughput: {float(metadata['selected_anchor_throughput']):.6g}")

    print(prediction_df.to_string(index=False))


if __name__ == "__main__":
    main()
