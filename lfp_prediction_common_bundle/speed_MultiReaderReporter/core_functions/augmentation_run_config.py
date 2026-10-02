from __future__ import annotations

import numpy as np


MODEL_TRAINING_SEED_COUNT = 1
MODEL_TRAINING_SEED_SOURCE = 42

TCNN_VALIDATION_SPLIT = 0.2
EOL_MLP_VALIDATION_SPLIT = 0.2

EXCLUDE_REAL_TRAINING_CELLS_REACHING_EOL_BEFORE_INPUT = False


def get_model_training_seed_count(value: int | None = None) -> int:
    if value is None:
        return max(1, int(MODEL_TRAINING_SEED_COUNT))
    return max(1, int(value))


def build_model_training_seed_list(
    seed_count: int | None = None,
    seed_source: int | None = None,
    *,
    seed_upper_bound: int = 10000,
) -> list[int]:
    effective_count = get_model_training_seed_count(seed_count)
    effective_source = int(MODEL_TRAINING_SEED_SOURCE if seed_source is None else seed_source)
    upper_bound = max(int(seed_upper_bound), int(effective_count) + 1)
    rng = np.random.default_rng(effective_source)
    return [
        int(seed)
        for seed in rng.choice(range(0, int(upper_bound)), size=int(effective_count), replace=False)
    ]


DEFAULT_AUGMENTATION_METHOD = "feature_log2"
FEATURE_LOG_AUGMENTATION_FACTOR_COUNT = 30
FEATURE_LOG2_SCALE_SPACING_METHOD = "log"
FEATURE_LOG2_USE_EQUAL_STEP_FACTORS = FEATURE_LOG2_SCALE_SPACING_METHOD == "equal_step"
FEATURE_LOG2_SCALE_SCOPE = "per_cell"
FEATURE_LOG2_MAX_SCALE_POLICY = "max_var_delta_q_and_lifetime_prediction_multiplier"
FEATURE_LOG2_LIFETIME_PREDICTION_MULTIPLIER = 2.0
FEATURE_LOG2_MAX_SCALE_REFERENCE_POLICY = "prediction_input"
FEATURE_LOG2_FEATURE_FLOOR_POLICY = "driver_features_only"

MODEL_INPUT_TRANSFORM_MODE = "standard_scaler"
MODEL_TARGET_TRANSFORM_MODE = "target_divisor_then_log"
MODEL_OUTPUT_HEAD_ACTIVATION = "tanh"

TCNN_INPUT_CLIPPING_MODE = "enabled"
TCNN_INPUT_CLIPPING_PERCENT = 5.0
TCNN_OUTPUT_CLIPPING_MODE = "disabled"
TCNN_OUTPUT_CLIPPING_PERCENT = 10.0
ELASTICNET_ALPHA = 0.01
ELASTICNET_L1_RATIO = 0.1
ELASTICNET_MAX_ITER = 10000
ELASTICNET_RANDOM_STATE = 42
SVR_KERNEL = "rbf"
SVR_C = 10.0
SVR_EPSILON = 0.1
SVR_GAMMA = "scale"

SCRIPT_TCNN_SEED_COUNT = MODEL_TRAINING_SEED_COUNT
INCLUDE_RAW_TCNN = True
FEATURE_LOG_GLOBAL_FEATURE_NAMES = {
    "mean_delta_q",
    "var_delta_q",
    "mean_delta_dqdv",
    "var_delta_dqdv",
}
FEATURE_LOG_AUGMENTATION_CANDIDATE_FEATURES = ("var_delta_Q",)
FEATURE_LOG_AUGMENTATION_ALWAYS_FEATURES = FEATURE_LOG_AUGMENTATION_CANDIDATE_FEATURES
FEATURE_LOG_AUGMENTATION_MIN_ABS_PEARSON = None
FEATURE_LOG_AUGMENTATION_DRIVER_FEATURES = FEATURE_LOG_AUGMENTATION_CANDIDATE_FEATURES


def _tcnn_clipping_enabled(mode: str, *, setting_name: str) -> bool:
    normalized = str(mode).strip().lower()
    if normalized not in {"enabled", "disabled"}:
        raise ValueError(f"{setting_name} must be 'enabled' or 'disabled', got {mode!r}.")
    return normalized == "enabled"


def tcnn_input_clipping_enabled() -> bool:
    return _tcnn_clipping_enabled(TCNN_INPUT_CLIPPING_MODE, setting_name="TCNN_INPUT_CLIPPING_MODE")


def tcnn_output_clipping_enabled() -> bool:
    return _tcnn_clipping_enabled(TCNN_OUTPUT_CLIPPING_MODE, setting_name="TCNN_OUTPUT_CLIPPING_MODE")


def tcnn_input_clipping_fraction() -> float:
    return max(0.0, float(TCNN_INPUT_CLIPPING_PERCENT)) / 100.0


def tcnn_output_clipping_fraction() -> float:
    return max(0.0, float(TCNN_OUTPUT_CLIPPING_PERCENT)) / 100.0


FEATURE_LOG2_FEATURE_FLOOR_POLICIES = (
    "driver_features_only",
    "all_model_features_no_bands",
    "all_model_features",
)


def feature_log2_feature_floor_policy(value: str | None = None) -> str:
    policy = str(FEATURE_LOG2_FEATURE_FLOOR_POLICY if value is None else value).strip().lower()
    if policy not in FEATURE_LOG2_FEATURE_FLOOR_POLICIES:
        raise ValueError(
            "FEATURE_LOG2_FEATURE_FLOOR_POLICY must be one of "
            f"{', '.join(FEATURE_LOG2_FEATURE_FLOOR_POLICIES)}, got "
            f"{value if value is not None else FEATURE_LOG2_FEATURE_FLOOR_POLICY!r}."
        )
    return policy


def augmentation_limit_rule() -> str:
    rule = (
        f"{DEFAULT_AUGMENTATION_METHOD}: {FEATURE_LOG2_SCALE_SCOPE} feature-derived "
        "Var(Delta Q) coverage cap"
    )
    if FEATURE_LOG2_FEATURE_FLOOR_POLICY != "driver_features_only":
        rule += f" (floor: {feature_log2_feature_floor_policy()})"
    if FEATURE_LOG2_MAX_SCALE_POLICY == "max_var_delta_q_and_lifetime_prediction_multiplier":
        rule += (
            f", then reject generated samples above "
            f"{FEATURE_LOG2_LIFETIME_PREDICTION_MULTIPLIER:g}x the training-only lifetime prediction"
        )
    return rule


def build_comparison_seeds(seeds):
    return list(seeds)[:SCRIPT_TCNN_SEED_COUNT]


FEATURE_NUM = "4"
TCNN_FOUR_FEATURE_NAMES = set(FEATURE_LOG_GLOBAL_FEATURE_NAMES)
FEATURE_LOG_SUBPLOT_FEATURES = ("mean_delta_Q", "var_delta_Q", "mean_delta_dqdv", "var_delta_dqdv")
EXCLUDE_CHARGE_PROFILE_FEATURES = True
EXCLUDE_VOLTAGE_BAND_FEATURES = True
MAPE_PLOT_MAX_CELL_APE_PERCENT = 120.0
AUTOANCHOR_MAPE_PLOT_MAX_CELL_APE_PERCENT = 150.0

COMPARISON_DISTRIBUTION_PLOT_STYLE = "whisker"


def comparison_distribution_plot_style(value: str | None = None) -> str:
    style = str(COMPARISON_DISTRIBUTION_PLOT_STYLE if value is None else value).strip().lower()
    if style == "violine":
        style = "violin"
    if style not in {"violin", "whisker"}:
        raise ValueError(
            "COMPARISON_DISTRIBUTION_PLOT_STYLE must be 'violin' (or 'violine') or 'whisker', "
            f"got {value if value is not None else COMPARISON_DISTRIBUTION_PLOT_STYLE!r}."
        )
    return style
