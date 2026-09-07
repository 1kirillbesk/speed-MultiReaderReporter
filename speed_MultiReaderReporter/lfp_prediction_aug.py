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
    PREDICTION_MODEL_EXPORTED_EOL_MLP,
    PREDICTION_MODEL_EXPORTED_EOL_TCNN,
    SUPPORTED_PREDICTION_MODELS,
    DEFAULT_SEED_COUNT,
    DEFAULT_SEED_SOURCE,
    DEFAULT_TARGET_COL,
    RAW_FEATURE_DIR,
    TrainingConfig,
    supported_augmentation_methods,
    run_lfp_prediction_pipeline,
)

DEFAULT_AUG_TARGET_SOH = 0.92
DEFAULT_AUG_TEST_NAMES = tuple(f"SPEED_LW_reference_{index}" for index in range(1, 10))
DEFAULT_AUG_PREDICTION_MODELS = (
    PREDICTION_MODEL_EXPORTED_EOL_TCNN,
    PREDICTION_MODEL_EXPORTED_EOL_MLP,
)


def extract_effective_augmentation_factors(cells: list[dict[str, object]]) -> list[float]:
    scale_keys = (
        "augmentation_scale",
        "matching_scale",
        "normal_scale",
        "cal_warp_factor",
        "hybrid_scale",
        "hybrid_base_scale",
    )
    factors: list[float] = []
    for cell in cells:
        for key in scale_keys:
            value = cell.get(key)
            if value is None:
                continue
            try:
                factor = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(factor):
                factors.append(factor)
                break
    return sorted(factors)


def format_factor_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the LFP augmented prediction model on the raw cell_feature CSV files used upstream "
            "by analyze_3dim_top_selecttime_linear, using a fixed raw input cutoff "
            "in week or throughput and a configurable SOH end-of-life target."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_FEATURE_DIR)
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument("--target-soh", type=float, default=DEFAULT_AUG_TARGET_SOH)
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
        default=list(DEFAULT_AUG_TEST_NAMES),
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
            "Used by CAL/HYBRID/CODEX augmentation. NORMAL uses anchor-derived scale expansion instead."
        ),
    )
    parser.add_argument("--anchor-week", type=float, default=DEFAULT_INPUT_END_WEEK)
    parser.add_argument("--anchor-throughput", type=float, default=DEFAULT_INPUT_END_THROUGHPUT)
    parser.add_argument("--anchor-mode", choices=("throughput", "week"), default="throughput")
    parser.add_argument("--anchor-soh", type=float, default=DEFAULT_ANCHOR_SOH)
    parser.add_argument(
        "--prediction-model",
        choices=SUPPORTED_PREDICTION_MODELS,
        default=None,
        help="Run one prediction model only. If omitted, the script runs both exported_eol_tcnn and exported_eol_mlp.",
    )
    parser.add_argument(
        "--prediction-models",
        nargs="+",
        choices=SUPPORTED_PREDICTION_MODELS,
        default=None,
        help="Run multiple prediction models in one invocation. Overrides --prediction-model.",
    )
    parser.add_argument("--no-show", action="store_false", dest="show")
    parser.set_defaults(show=True)
    return parser.parse_args()


def resolve_prediction_models(args: argparse.Namespace) -> list[str]:
    if args.prediction_models:
        return [str(model) for model in args.prediction_models]
    if args.prediction_model:
        return [str(args.prediction_model)]
    return [str(model) for model in DEFAULT_AUG_PREDICTION_MODELS]


def run_prediction(args: argparse.Namespace, prediction_model: str) -> None:
    training_config = TrainingConfig(
        input_steps=int(args.input_steps),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        dropout_rate=float(args.dropout),
        seed_count=int(args.seed_count),
        seed_source=int(args.seed_source),
        prediction_model=str(prediction_model),
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

    metadata = dict(result["metadata"])
    prediction_df = result["prediction_df"]
    candidate_factors = [
        float(scale)
        for scale in list(metadata.get("augmentation_scales", []) or [])
        if np.isfinite(float(scale))
    ]
    effective_factors = extract_effective_augmentation_factors(result["augmented_cells"])
    print()
    print(f"=== Prediction model: {prediction_model} ===")
    if np.isfinite(float(metadata.get("selected_anchor_throughput", float('nan')))):
        print(f"Selected anchor throughput: {float(metadata['selected_anchor_throughput']):.6g}")
    else:
        print(f"Selected anchor week: {float(metadata['selected_anchor_week']):.2f}")
    print(f"Target SOH: {float(metadata['target_soh']):.4f}")
    print(f"Prediction model: {metadata.get('prediction_model', 'unknown')}")
    print(
        f"Train real={int(metadata['train_real_count'])}  "
        f"train aug={int(metadata['train_aug_count'])}  "
        f"test={int(metadata['test_count'])}"
    )
    if candidate_factors:
        print(f"Candidate augmentation factors: {format_factor_list(candidate_factors)}")
    else:
        print("Candidate augmentation factors: []")
    if effective_factors:
        print(f"Effective augmentation factors: {format_factor_list(effective_factors)}")
    else:
        print("Effective augmentation factors: []")
    print(f"Week target divisor: {float(result['week_target_divisor']):.6g}")
    print(f"Throughput target divisor: {float(result['throughput_target_divisor']):.6g}")
    print(f"RMSE (weeks): {float(result['rmse_weeks']):.2f}")
    print(f"MAPE (weeks): {float(result['mape_weeks']):.2f}%")
    print(f"RMSE (throughput): {float(result['rmse_throughput']):.2f}")
    print(f"MAPE (throughput): {float(result['mape_throughput']):.2f}%")
    print(prediction_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    for prediction_model in resolve_prediction_models(args):
        run_prediction(args, prediction_model)


if __name__ == "__main__":
    main()
