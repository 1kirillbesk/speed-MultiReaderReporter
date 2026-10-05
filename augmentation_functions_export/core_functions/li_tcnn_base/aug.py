from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LI_DATARUN_DIR = _PROJECT_ROOT / "Li_datarun"
for _import_path in (_LI_DATARUN_DIR, _PROJECT_ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.lines import Line2D
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from core_functions.interpolation import (
    interpolate_value_at_soh as _shared_interpolate_value_at_soh,
    interpolate_value_at_x,
)
from core_functions.prediction_evaluation import (
    calculate_ape_metrics,
    calculate_ape_values,
    calculate_group_mapes as _calculate_group_mapes,
    calculate_max_ape_cell,
)
from runtime import (
    DEFAULT_FEATURE_ROW_INDEX,
    RAW_CAPACITY_COLUMN,
    RAW_PROCESSED_DATA_DIR,
    RAW_WEEK_COLUMN,
    build_mlp_hyperparameters,
    clean_input_dataframe_base as clean_input_dataframe,
    get_feature_columns,
    get_week_model_feature_week,
    load_and_interpolate,
    parse_cell_name,
)
from core_functions.plot_display import show_rendered_figure
from core_functions.temporal_models import (
    EolPredictionMLP,
    ExtrapolationResidualTemporalCNN,
    build_log_feature_fit_summary as build_fit_summary,
    build_regression_loss,
    build_shared_extrapolation_tcnn_hyperparameters,
    build_shared_tcnn_hyperparameters,
    apply_tcnn_input_clipping,
    apply_tcnn_output_clipping,
    clip_with_fractional_margin,
    default_r2 as calculate_r2,
    fit_linear_relationship,
    get_required_features,
    make_neural_input_scaler,
    make_neural_target_transformer,
    set_torch_seed,
    split_validation_arrays,
    train_temporal_cnn_regressor,
)
import config as li_config
import eol_mlp_20w_IOWA as iowa_aug70
from eol_mlp_20w_IOWA import iowa_aug80

SCRIPT_DIR = _LI_DATARUN_DIR
PROJECT_ROOT = _PROJECT_ROOT
from core_functions import augmentation_run_config as aug_cfg
from core_functions import autoanchor as core_autoanchor
from core_functions import temporal_models
FEATURE_SELECTION_LABEL = "none"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / f"fig_iowa_tcnn_aug_s70_{FEATURE_SELECTION_LABEL}"
TARGET_SOH = 0.70
INTERPOLATED_TARGET_COLUMN = "EOL70"
RAW_TARGET_COLUMN = "week_at_eol70_raw"
THRESHOLDS = [float(value) for value in li_config.THRESHOLDS]
DEFAULT_INTERPOLATION_CAPA_LIMS = [0.96, 0.97, 0.98]
DEFAULT_MIN_ABS_PEARSON = 0.40
LONG_EOL_VALIDATION_THRESHOLD_WEEKS = 30.0
SUBPLOT_COLUMNS = 3
LINEAR_FEATURE_SELECTION_METHOD = "highest_train_r2"
VALIDATION_SPLIT_METHOD = "random" #stratified random
STRATIFIED_VALIDATION_BINS = 3
HIGHLIGHT_GROUPS = ["G1", "G6", "G26"]
HIGHLIGHT_COLOR = "#ff7f0e"
def build_augmented_tcnn_hyperparameters(overrides: dict[str, object] | None = None) -> dict[str, object]:
    return build_shared_tcnn_hyperparameters(overrides)


BEST_INTERPOLATED_TCNN_HYPERPARAMETERS = build_augmented_tcnn_hyperparameters()
EXTRAPOLATION_MLP_HYPERPARAMETERS = {
    "hidden_dim": 32,
    "hidden_layers": 2,
    "dropout_rate": 0.05,
    "epochs": 400,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "seeds": list(BEST_INTERPOLATED_TCNN_HYPERPARAMETERS["seeds"]),
    "input_extrapolation_margin": 0.15,
    "target_log_extrapolation_margin": 0.20,
    "validation_split": 0.3,
    "early_stopping_patience": 120,
}
EXTRAPOLATION_TCNN_HYPERPARAMETERS = build_shared_extrapolation_tcnn_hyperparameters(
    {"seeds": list(BEST_INTERPOLATED_TCNN_HYPERPARAMETERS["seeds"])}
)
INTERPOLATED_SEQUENCE_AND_MLP_FEATURES = [
    "SOH",
    "mean_delta_Q",
    "var_delta_Q",
    "mean_delta_dqdv",
    "var_delta_dqdv",
    "mean_delta_dqdv_low",
    "var_delta_dqdv_low",
    "mean_delta_dqdv_mid",
    "var_delta_dqdv_mid",
    "mean_delta_dqdv_high",
    "var_delta_dqdv_high",
]
feature_num = aug_cfg.FEATURE_NUM
COMPARISON_SEEDS = list(BEST_INTERPOLATED_TCNN_HYPERPARAMETERS["seeds"])
COLOR_SOFT_CORAL = "#F8ABAB"
COLOR_CORAL = "#EF8881"
COLOR_TEAL = "#3ABABD"
COLOR_SOFT_BLUE = "#6FAFD0"
COLOR_TEAL_MID = "#67B9BC"

def is_charge_profile_feature_name(feature_name: object) -> bool:
    normalized = str(feature_name or "").strip().lower()
    return normalized.startswith(("mean_", "var_")) and normalized.endswith("_c")

def select_soh_meanvar_prediction_features(candidate_features: list[str], *, label: str) -> list[str]:
    filtered_features = [
        str(name)
        for name in candidate_features
        if not (aug_cfg.EXCLUDE_CHARGE_PROFILE_FEATURES and is_charge_profile_feature_name(name))
    ]
    return temporal_models.select_soh_meanvar_tcnn_features(
        filtered_features, label=label, feature_num=feature_num,
        four_feature_names=aug_cfg.TCNN_FOUR_FEATURE_NAMES)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare IOWA raw and interpolated prediction performance across lifetime thresholds. "
            "For each threshold, train on cells with raw week-at-70%%SOH <= threshold and test on cells above it."
        )
    )
    parser.add_argument("--raw-input-dir", type=Path, default=RAW_PROCESSED_DATA_DIR)
    parser.add_argument("--feature-row-index", type=int, default=DEFAULT_FEATURE_ROW_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    parser.add_argument(
        "--interpolation-type",
        choices=("throughput", "realtime"),
        default="throughput",
        help="Interpolation reference axis.",
    )
    parser.add_argument(
        "--ref-cell",
        type=int,
        default=1,
        help="Reference cell group used to choose the default interpolation SOH limit.",
    )
    parser.add_argument(
        "--capa-lims",
        nargs="+",
        type=float,
        default=DEFAULT_INTERPOLATION_CAPA_LIMS,
        help="Interpolation SOH limits. Defaults to 0.96 and 0.98.",
    )
    parser.add_argument(
        "--min-abs-pearson",
        type=float,
        default=DEFAULT_MIN_ABS_PEARSON,
        help="Legacy option; MLP/TCNN now use the fixed train_modeltorch_average1.py feature list.",
    )
    parser.add_argument("--no-show", action="store_false", dest="show")
    parser.set_defaults(show=True)
    return parser.parse_args()

def format_soh_folder_name(capa_lim: float) -> str:
    return f"soh{int(round(float(capa_lim) * 100.0))}"

def interpolate_value_at_soh(
    x_values: np.ndarray,
    y_values: np.ndarray,
    capacities: np.ndarray,
    target_soh: float = TARGET_SOH,
) -> float:
    return _shared_interpolate_value_at_soh(x_values, y_values, capacities, target_soh)

def raw_sequence_feature_should_zero_fill(feature_name: str) -> bool:
    return str(feature_name).strip().lower().startswith(("mean_", "var_"))

def get_cell_group(cell: str) -> str:
    return str(cell).split("C", maxsplit=1)[0]

def configure_comparison_seeds() -> None:
    BEST_INTERPOLATED_TCNN_HYPERPARAMETERS["seeds"] = list(COMPARISON_SEEDS)
    EXTRAPOLATION_MLP_HYPERPARAMETERS["seeds"] = list(COMPARISON_SEEDS)
    EXTRAPOLATION_TCNN_HYPERPARAMETERS["seeds"] = list(COMPARISON_SEEDS)
    for module in (iowa_aug70, iowa_aug80):
        module.SEEDS = list(COMPARISON_SEEDS)
        module.BASE_SEED = int(COMPARISON_SEEDS[0])
        module.AUGMENTATION_DENSITY_MULTIPLIER = 1

@contextlib.contextmanager
def suppress_augmented_plots(module):
    original_show = module.plt.show
    original_plot_all_feature_examples = module.plot_all_feature_examples
    module.plt.show = lambda *args, **kwargs: None
    module.plot_all_feature_examples = lambda *args, **kwargs: None
    try:
        yield
    finally:
        module.plt.show = original_show
        module.plot_all_feature_examples = original_plot_all_feature_examples

def build_augmented_summary_row(
    summary: dict[str, object],
    *,
    threshold: float,
    target_soh: float,
) -> dict[str, object]:
    def safe_int(value: object, default: int = 0) -> int:
        numeric_value = pd.to_numeric(value, errors="coerce")
        if np.isfinite(numeric_value):
            return int(round(float(numeric_value)))
        return int(default)

    target_percent = int(round(float(target_soh) * 100.0))
    model_name = "tcnn_aug" if np.isclose(float(target_soh), float(TARGET_SOH)) else f"tcnn_aug_eol{target_percent}"
    return {
        "threshold_weeks": float(threshold),
        "model_name": model_name,
        "dataset_label": f"eligible{int(float(threshold))}_aug_week{target_percent}",
        "selected_feature": np.nan,
        "feature_count": len(str(summary.get("feature_columns", "")).split()),
        "train_count": safe_int(summary.get("train_real_count", 0)) + safe_int(summary.get("train_aug_count", 0)),
        "test_count": safe_int(summary.get("tcnn_test_count", summary.get("test_count", 0))),
        "train_mape": np.nan,
        "test_mape": float(summary.get("tcnn_test_mape", np.nan)),
        "test_ape_std": float(summary.get("tcnn_test_ape_std", summary.get("tcnn_seed_test_mape_std", np.nan))),
        "train_max_ape": np.nan,
        "test_max_ape": float(summary.get("tcnn_test_max_ape", np.nan)),
        "test_max_ape_cell": summary.get("tcnn_test_max_ape_cell", np.nan),
        "test_ape_values": summary.get("tcnn_test_ape_values", []),
        "test_cells": summary.get("tcnn_test_cell_list", []),
        "test_actual": summary.get("tcnn_test_actual", []),
        "test_pred": summary.get("tcnn_test_pred", []),
        "long_eol_threshold_weeks": float(summary.get("long_eol_threshold_weeks", iowa_aug70.LONG_EOL_HIGHLIGHT_WEEKS)),
        "long_eol_test_count": safe_int(summary.get("test_gt30w_count", 0)),
        "long_eol_test_mape": float(summary.get("tcnn_test_mape_gt30w", np.nan)),
        "long_eol_test_ape_std": float(summary.get("tcnn_test_ape_std_gt30w", np.nan)),
        "long_eol_test_ape_values": summary.get("tcnn_test_ape_values_gt30w", []),
        "long_eol_test_max_ape": float(summary.get("tcnn_test_max_ape_gt30w", np.nan)),
        "long_eol_test_max_ape_cell": summary.get("tcnn_test_max_ape_cell_gt30w", np.nan),
        "long_eol_test_cells": summary.get("tcnn_long_eol_test_cells", []),
        "train_r2": np.nan,
        "test_r2": float(summary.get("tcnn_test_r2", np.nan)),
        "epochs": np.nan,
        "learning_rate": np.nan,
        "target_soh": float(target_soh),
        "augmentation_reference_soh": summary.get("augmentation_reference_soh"),
        "augmentation_train_reference": summary.get("augmentation_train_reference"),
        "augmentation_test_reference": summary.get("augmentation_test_reference"),
        "augmentation_scales": summary.get("augmentation_scales"),
    }

def run_augmented_tcnn_threshold(
    *,
    module,
    threshold: float,
    state: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    with suppress_augmented_plots(module):
        summary, new_state = module.run_experiment(
            task_kind="time",
            title_prefix=f"IOWA TCNN aug ({int(round(float(module.TARGET_SOH) * 100.0))}% SOH, {int(float(threshold))}w threshold)",
            raw_dir=module.CYCLIC_DATA_DIR,
            threshold_weeks=float(threshold),
            input_reference_value=float(module.TIME_INPUT_WEEKS),
            x_label="Time [weeks]",
            y_units_label="weeks",
            all_cells=None if state is None else state.get("all_cells"),
            input_feature_columns=None if state is None else state.get("input_feature_columns"),
            previous_train_cell_ids=set() if state is None else state.get("train_cell_ids", set()),
            previous_aug_cells=[] if state is None else state.get("aug_cells", []),
        )
    return build_augmented_summary_row(summary, threshold=float(threshold), target_soh=float(module.TARGET_SOH)), new_state

def calculate_group_mapes(
    cells: list[str],
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: list[str] = HIGHLIGHT_GROUPS,
) -> dict[str, float]:
    return _calculate_group_mapes(
        cells, actual, predicted, groups=groups, group_getter=get_cell_group)

def format_group_mapes(mapes: dict[str, float]) -> str:
    lines = ["Test MAPEs"]
    for group in HIGHLIGHT_GROUPS:
        mape = mapes.get(group, np.nan)
        lines.append(f"{group}: {mape:.2f}%" if np.isfinite(mape) else f"{group}: n/a")
    return "\n".join(lines)

def train_extrapolation_mlp_regressor(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    dataset_label: str,
) -> dict[str, object]:
    config = build_mlp_hyperparameters(EXTRAPOLATION_MLP_HYPERPARAMETERS)
    required_columns = ["Cell", "is_outer_field_selected", "is_test_cell", target_column] + feature_columns
    dataset = data[required_columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    dataset = dataset[dataset[target_column] > 0].copy()
    train_df = dataset[dataset["is_outer_field_selected"]].copy()
    test_df = dataset[dataset["is_test_cell"]].copy()
    if len(train_df) < 2:
        raise RuntimeError(f"Not enough selected training rows for {dataset_label} extrapolation MLP.")
    if test_df.empty:
        raise RuntimeError(f"No test rows available for {dataset_label} extrapolation MLP.")

    target_transformer = make_neural_target_transformer(train_df[target_column].to_numpy(dtype=float))
    target_divisor = float(target_transformer.target_divisor_)
    train_df["target_log"] = target_transformer.transform(train_df[target_column].to_numpy(dtype=float))
    test_df["target_log"] = target_transformer.transform(test_df[target_column].to_numpy(dtype=float))

    x_train_raw = train_df[feature_columns].to_numpy(dtype=float)
    x_test_raw = test_df[feature_columns].to_numpy(dtype=float)
    x_train_min = np.min(x_train_raw, axis=0)
    x_train_max = np.max(x_train_raw, axis=0)
    x_test_bounded = clip_with_fractional_margin(x_test_raw, x_train_min, x_train_max,
        margin_fraction=float(config["input_extrapolation_margin"]))

    x_scaler = make_neural_input_scaler()
    x_train = x_scaler.fit_transform(x_train_raw)
    x_test = x_scaler.transform(x_test_bounded)
    y_train_raw = train_df[["target_log"]].to_numpy(dtype=float)
    y_train_scaled = y_train_raw.reshape(-1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    train_pred_scaled_collection: list[np.ndarray] = []
    test_pred_scaled_collection: list[np.ndarray] = []
    loss_fn = nn.MSELoss()

    for seed in config["seeds"]:
        x_fit, y_fit, x_val, y_val = split_validation_arrays(
            x_data=x_train,
            y_data=y_train_scaled,
            seed=int(seed),
            val_fraction=float(config["validation_split"]),
            method=VALIDATION_SPLIT_METHOD,
            stratified_bins=STRATIFIED_VALIDATION_BINS,
        )

        set_torch_seed(int(seed))
        model = EolPredictionMLP(
                len(feature_columns),
                [int(int(config["hidden_dim"])), max(8, int(int(config["hidden_dim"])) // 2)],
                float(float(config["dropout_rate"])),
            )
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )

        x_fit_tensor = torch.tensor(x_fit, dtype=torch.float32)
        y_fit_tensor = torch.tensor(y_fit, dtype=torch.float32)
        if x_val is not None and y_val is not None:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        else:
            x_val_tensor = None
            y_val_tensor = None

        best_metric = float("inf")
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        epochs_without_improvement = 0

        for _ in range(int(config["epochs"])):
            model.train()
            optimizer.zero_grad()
            predictions = model(x_fit_tensor)
            loss = loss_fn(predictions, y_fit_tensor)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                if x_val_tensor is not None and y_val_tensor is not None:
                    metric = float(loss_fn(model(x_val_tensor), y_val_tensor).item())
                else:
                    metric = float(loss.item())

            if metric < best_metric:
                best_metric = metric
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(config["early_stopping_patience"]):
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            train_pred_scaled_collection.append(model(torch.tensor(x_train, dtype=torch.float32)).cpu().numpy().reshape(-1, 1))
            test_pred_scaled_collection.append(model(x_test_tensor).cpu().numpy().reshape(-1, 1))

    train_pred_scaled = np.mean(np.stack(train_pred_scaled_collection, axis=0), axis=0)
    test_pred_scaled = np.mean(np.stack(test_pred_scaled_collection, axis=0), axis=0)
    train_pred_log = train_pred_scaled.reshape(-1)
    test_pred_log = test_pred_scaled.reshape(-1)
    target_log_min = float(np.min(y_train_raw))
    target_log_max = float(np.max(y_train_raw))
    train_pred_log = clip_with_fractional_margin(
        train_pred_log,
        np.array([target_log_min]),
        np.array([target_log_max]),
        margin_fraction=float(config["target_log_extrapolation_margin"]),
    ).reshape(-1)
    test_pred_log = clip_with_fractional_margin(
        test_pred_log,
        np.array([target_log_min]),
        np.array([target_log_max]),
        margin_fraction=float(config["target_log_extrapolation_margin"]),
    ).reshape(-1)
    train_actual = train_df[target_column].to_numpy(dtype=float) / target_divisor
    test_actual = test_df[target_column].to_numpy(dtype=float) / target_divisor
    train_pred = np.exp(train_pred_log)
    test_pred = np.exp(test_pred_log)
    test_features_clipped = bool(np.any(np.abs(x_test_bounded - x_test_raw) > 0))

    return {
        "model_name": "mlp_extrap",
        "dataset_label": dataset_label,
        "feature_count": len(feature_columns),
        "train_count": len(train_df),
        "test_count": len(test_df),
        "train_mape": calculate_ape_metrics(train_actual, train_pred)[0],
        "test_mape": calculate_ape_metrics(test_actual, test_pred)[0],
        "train_max_ape": calculate_ape_metrics(train_actual, train_pred)[1],
        "test_max_ape": calculate_ape_metrics(test_actual, test_pred)[1],
        "train_max_ape_cell": calculate_max_ape_cell(train_actual, train_pred, train_df["Cell"].astype(str).tolist()),
        "test_max_ape_cell": calculate_max_ape_cell(test_actual, test_pred, test_df["Cell"].astype(str).tolist()),
        "train_r2": calculate_r2(train_actual, train_pred),
        "test_r2": calculate_r2(test_actual, test_pred),
        "epochs": int(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "hidden_dim": int(config["hidden_dim"]),
        "hidden_layers": int(config["hidden_layers"]),
        "dropout_rate": float(config["dropout_rate"]),
        "validation_split": float(config["validation_split"]),
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "seed_count": len(config["seeds"]),
        "target_column": target_column,
        "target_divisor": float(target_divisor),
        "target_transform": target_transformer.mode,
        "target_scaler": "identity",
        "input_extrapolation_margin": float(config["input_extrapolation_margin"]),
        "target_log_extrapolation_margin": float(config["target_log_extrapolation_margin"]),
        "test_features_clipped": test_features_clipped,
        "train_actual": train_actual,
        "train_pred": train_pred,
        "test_actual": test_actual,
        "test_pred": test_pred,
        "train_cell_list": train_df["Cell"].astype(str).tolist(),
        "test_cell_list": test_df["Cell"].astype(str).tolist(),
    }

def train_extrapolation_temporal_cnn_regressor(
    sequence_dataset: dict[str, object],
    dataset_label: str,
) -> dict[str, object]:
    config = dict(EXTRAPOLATION_TCNN_HYPERPARAMETERS)
    sequences = np.asarray(sequence_dataset["sequences"], dtype=float)
    targets = np.asarray(sequence_dataset["targets"], dtype=float)
    cells = list(sequence_dataset["cells"])
    train_mask = np.asarray(sequence_dataset["train_mask"], dtype=bool)
    test_mask = np.asarray(sequence_dataset["test_mask"], dtype=bool)
    feature_columns = list(sequence_dataset["feature_columns"])

    x_train_raw = sequences[train_mask]
    x_test_raw = sequences[test_mask]
    y_train_raw = targets[train_mask]
    y_test_raw = targets[test_mask]
    train_cells = [cell for cell, is_train in zip(cells, train_mask) if is_train]
    test_cells = [cell for cell, is_test in zip(cells, test_mask) if is_test]
    if len(x_train_raw) < 2:
        raise RuntimeError(f"Not enough selected training sequences for {dataset_label} extrapolation temporal CNN.")
    if len(x_test_raw) == 0:
        raise RuntimeError(f"No validation sequences available for {dataset_label} extrapolation temporal CNN.")

    x_train_min = np.min(x_train_raw, axis=(0, 1))
    x_train_max = np.max(x_train_raw, axis=(0, 1))
    x_test_bounded = apply_tcnn_input_clipping(x_test_raw, x_train_min, x_train_max)

    scaler = make_neural_input_scaler()
    x_train_scaled = scaler.fit_transform(x_train_raw.reshape(-1, x_train_raw.shape[-1])).reshape(x_train_raw.shape)
    x_test_scaled = scaler.transform(x_test_bounded.reshape(-1, x_test_raw.shape[-1])).reshape(x_test_raw.shape)
    target_transformer = make_neural_target_transformer(y_train_raw)
    target_divisor = float(target_transformer.target_divisor_)
    y_train_log = target_transformer.transform(y_train_raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
    target_log_min = float(np.min(y_train_log))
    target_log_max = float(np.max(y_train_log))
    train_actual = y_train_raw / target_divisor
    test_actual = y_test_raw / target_divisor
    train_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []

    for seed in config["seeds"]:
        x_fit, y_fit, x_val, y_val = split_validation_arrays(
            x_data=x_train_scaled,
            y_data=y_train_log,
            seed=int(seed),
            val_fraction=float(config["validation_split"]),
            method=VALIDATION_SPLIT_METHOD,
            stratified_bins=STRATIFIED_VALIDATION_BINS,
        )
        set_torch_seed(int(seed))
        model = ExtrapolationResidualTemporalCNN(
            input_dims=len(feature_columns),
            timesteps=int(config["input_steps"]),
            conv_channels=int(config["conv_channels"]),
            fc_hidden_dim=int(config["fc_hidden_dim"]),
            dropout_rate=float(config["dropout_rate"]),
        ).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(config["scheduler_patience"]),
            factor=float(config.get("scheduler_factor", 0.8)),
        )
        loss_fn = build_regression_loss(str(config.get("loss_function", "smooth_l1")))
        x_fit_tensor = torch.tensor(x_fit, dtype=torch.float32, device=device)
        y_fit_tensor = torch.tensor(y_fit, dtype=torch.float32, device=device).unsqueeze(1)
        batch_size = int(config["batch_size"] or len(x_fit_tensor))
        train_loader = DataLoader(
            TensorDataset(x_fit_tensor, y_fit_tensor),
            batch_size=batch_size,
            shuffle=True,
        )
        if x_val is not None and y_val is not None:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)
        else:
            x_val_tensor = None
            y_val_tensor = None

        best_metric = float("inf")
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        epochs_without_improvement = 0

        for epoch in range(int(config["epochs"]) + 1):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0 or epoch == int(config["epochs"]):
                model.eval()
                with torch.no_grad():
                    if x_val_tensor is not None and y_val_tensor is not None:
                        metric = float(loss_fn(model(x_val_tensor), y_val_tensor).item())
                    else:
                        metric = float(loss_fn(model(x_fit_tensor), y_fit_tensor).item())
                scheduler.step(metric)
                if metric < best_metric:
                    best_metric = metric
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= int(config["early_stopping_patience"]):
                        break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            train_pred_log = model(x_train_tensor).cpu().numpy().reshape(-1)
            test_pred_log = model(x_test_tensor).cpu().numpy().reshape(-1)

        train_pred_log = apply_tcnn_output_clipping(train_pred_log,
            np.array([target_log_min]), np.array([target_log_max])).reshape(-1)
        test_pred_log = apply_tcnn_output_clipping(test_pred_log,
            np.array([target_log_min]), np.array([target_log_max])).reshape(-1)
        train_predictions.append(np.exp(train_pred_log))
        test_predictions.append(np.exp(test_pred_log))

    train_pred = np.mean(np.vstack(train_predictions), axis=0)
    test_pred = np.mean(np.vstack(test_predictions), axis=0)

    return {
        "model_name": "tcnn_extrap",
        "dataset_label": dataset_label,
        "feature_count": len(feature_columns),
        "train_count": len(train_cells),
        "test_count": len(test_cells),
        "train_mape": float(np.mean(np.abs((train_actual - train_pred) / train_actual)) * 100),
        "test_mape": float(np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100),
        "train_max_ape": float(np.max(np.abs((train_actual - train_pred) / train_actual)) * 100),
        "test_max_ape": float(np.max(np.abs((test_actual - test_pred) / test_actual)) * 100),
        "train_max_ape_cell": calculate_max_ape_cell(train_actual, train_pred, train_cells),
        "test_max_ape_cell": calculate_max_ape_cell(test_actual, test_pred, test_cells),
        "train_r2": calculate_r2(train_actual, train_pred),
        "test_r2": calculate_r2(test_actual, test_pred),
        "epochs": int(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "dropout_rate": float(config["dropout_rate"]),
        "input_steps": int(config["input_steps"]),
        "conv_channels": int(config["conv_channels"]),
        "fc_hidden_dim": int(config["fc_hidden_dim"]),
        "batch_size": int(config["batch_size"] or len(x_train_raw)),
        "scheduler_patience": int(config["scheduler_patience"]),
        "model_structure": "extrapolation_residual",
        "validation_split": float(config["validation_split"]),
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "seed_count": len(config["seeds"]),
        "target_divisor": float(target_divisor),
        "target_transform": target_transformer.mode,
        "target_scaler": "identity",
        "input_extrapolation_margin": float(config["input_extrapolation_margin"]),
        "target_log_extrapolation_margin": float(config["target_log_extrapolation_margin"]),
        "train_actual": train_actual,
        "train_pred": train_pred,
        "test_actual": test_actual,
        "test_pred": test_pred,
        "train_cell_list": train_cells,
        "test_cell_list": test_cells,
    }


def plot_prediction_comparison_axis(axis: plt.Axes, model_result: dict[str, object] | None, panel_title: str) -> None:
    if model_result is None:
        axis.set_axis_off()
        return

    test_actual = np.asarray(model_result.get("test_actual", []), dtype=float)
    test_pred = np.asarray(model_result.get("test_pred", []), dtype=float)
    test_cells = [str(cell) for cell in model_result.get("test_cell_list", [])]
    if len(test_actual) == 0 or len(test_pred) == 0 or len(test_cells) == 0:
        axis.set_axis_off()
        return

    valid = np.isfinite(test_actual) & np.isfinite(test_pred)
    if valid.sum() == 0:
        axis.set_axis_off()
        return

    actual = test_actual[valid]
    predicted = test_pred[valid]
    cells = [cell for cell, is_valid in zip(test_cells, valid) if is_valid]
    lower = float(min(np.min(actual), np.min(predicted)))
    upper = float(max(np.max(actual), np.max(predicted)))
    padding = max(0.05 * (upper - lower), 0.05)

    regular_actual: list[float] = []
    regular_pred: list[float] = []
    highlighted: dict[str, tuple[list[float], list[float], list[str]]] = {
        group: ([], [], []) for group in HIGHLIGHT_GROUPS
    }
    for cell, actual_value, predicted_value in zip(cells, actual, predicted):
        group = get_cell_group(cell)
        if group in highlighted:
            highlighted[group][0].append(float(actual_value))
            highlighted[group][1].append(float(predicted_value))
            highlighted[group][2].append(cell)
        else:
            regular_actual.append(float(actual_value))
            regular_pred.append(float(predicted_value))

    if regular_actual:
        axis.scatter(regular_actual, regular_pred, s=30, alpha=0.6, color="#8c8c8c", edgecolors="none")
    for group, (group_actual, group_predicted, group_cells) in highlighted.items():
        if not group_actual:
            continue
        axis.scatter(group_actual, group_predicted, s=56, alpha=0.92, color=HIGHLIGHT_COLOR, edgecolors="black", linewidths=0.4, zorder=3)
        for cell, x_value, y_value in zip(group_cells, group_actual, group_predicted):
            axis.text(x_value, y_value, cell, fontsize=7, ha="left", va="bottom")

    diagonal = np.linspace(lower - padding, upper + padding, 200)
    axis.plot(diagonal, diagonal, linestyle="--", color="black", linewidth=1.0)
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_title(panel_title)
    axis.set_xlabel("Actual")
    axis.set_ylabel("Predicted")

    group_mapes = calculate_group_mapes(cells, actual, predicted)
    long_eol_threshold_weeks = model_result.get("long_eol_threshold_weeks")
    long_eol_test_count = model_result.get("long_eol_test_count")
    long_eol_test_mape = model_result.get("long_eol_test_mape")
    summary_lines = [
        f"n={len(actual)}",
        (
            f"MAPE all={float(model_result.get('test_mape', np.nan)):.2f}%"
            if np.isfinite(float(model_result.get("test_mape", np.nan)))
            else "MAPE all=n/a"
        ),
        (
            f"R^2={float(model_result.get('test_r2', np.nan)):.3f}"
            if np.isfinite(float(model_result.get("test_r2", np.nan)))
            else "R^2=n/a"
        ),
    ]
    if (
        long_eol_threshold_weeks is not None
        and long_eol_test_count is not None
        and long_eol_test_mape is not None
        and int(long_eol_test_count) > 0
        and np.isfinite(float(long_eol_test_mape))
    ):
        summary_lines.append(
            f"MAPE >{int(float(long_eol_threshold_weeks))}w={float(long_eol_test_mape):.2f}%"
        )
        summary_lines.append(f"n >{int(float(long_eol_threshold_weeks))}w={int(long_eol_test_count)}")
    summary_lines.extend(
        [
            f"{group}: {group_mapes.get(group, np.nan):.2f}%"
            if np.isfinite(group_mapes.get(group, np.nan))
            else f"{group}: n/a"
            for group in HIGHLIGHT_GROUPS
        ]
    )
    axis.text(0.03, 0.97, "\n".join(summary_lines), transform=axis.transAxes, ha="left", va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.86, "edgecolor": HIGHLIGHT_COLOR})

def extract_test_ape(model_result: dict[str, object] | None) -> np.ndarray:
    if model_result is None:
        return np.array([], dtype=float)
    actual = np.asarray(model_result.get("test_actual", []), dtype=float)
    predicted = np.asarray(model_result.get("test_pred", []), dtype=float)
    return calculate_ape_values(actual, predicted)

def annotate_metric_label(
    axis: plt.Axes,
    x_value: float,
    y_value: float,
    text: str,
    color: str = "black",
    dx: float = 0.0,
    dy: float = 6.0,
    ha: str = "center",
    va: str = "bottom",
    fontsize: float = 8,
) -> None:
    axis.annotate(text, xy=(x_value, y_value), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize, color=color)

def plot_model_family_comparison_grid(threshold: float, output_dir: Path, comparison_rows: list[dict[str, object]], long_eol_threshold_weeks: float, show: bool) -> None:
    if not comparison_rows:
        return

    nrows = len(comparison_rows)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.8 * nrows), squeeze=False)
    column_titles = [
        f"Interpolated\nTest > {int(threshold)}w",
        f"Raw\nTest > {int(threshold)}w",
        f"Interpolated\nTest > {int(long_eol_threshold_weeks)}w",
        f"Raw\nTest > {int(long_eol_threshold_weeks)}w",
    ]

    for row_index, row_payload in enumerate(comparison_rows):
        model_label = str(row_payload["model_label"])
        interpolated_full = row_payload.get("interpolated")
        raw_full = row_payload.get("raw")
        interpolated_long = row_payload.get("interpolated_long")
        raw_long = row_payload.get("raw_long")
        row_results = [interpolated_full, raw_full, interpolated_long, raw_long]

        for col_index, (axis, result_payload, column_title) in enumerate(
            zip(axes[row_index], row_results, column_titles)
        ):
            full_title = column_title if row_index == 0 else ""
            panel_title = f"{full_title}\n{model_label}" if full_title else model_label
            plot_prediction_comparison_axis(axis, result_payload, panel_title)

    fig.suptitle(
        f"Threshold {int(threshold)}w: raw vs interpolated prediction comparison "
        f"(including >{int(long_eol_threshold_weeks)}w subset)",
        y=1.01,
    )
    fig.tight_layout()
    if show:
        show_rendered_figure(fig)
    plt.close(fig)

def plot_random_relationship_subplots(data: pd.DataFrame, feature_columns: list[str], fit_summary: pd.DataFrame, output_dir: Path, target_column: str, target_label: str,
    output_stem: str, title_prefix: str, show: bool) -> None:
    n_features = len(feature_columns)
    n_cols = min(SUBPLOT_COLUMNS, max(1, n_features))
    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    summary_by_feature = fit_summary.set_index("feature")

    for axis, feature in zip(axes, feature_columns):
        subset = build_feature_subset(data, feature, target_column)
        train_subset = subset[subset["is_outer_field_selected"]].copy()
        test_subset = subset[subset["is_test_cell"]].copy()

        if subset.empty:
            axis.set_axis_off()
            continue

        highlighted_train = train_subset[train_subset["Cell"].astype(str).map(get_cell_group).isin(HIGHLIGHT_GROUPS)]
        highlighted_test = test_subset[test_subset["Cell"].astype(str).map(get_cell_group).isin(HIGHLIGHT_GROUPS)]
        regular_train = train_subset.drop(index=highlighted_train.index)
        regular_test = test_subset.drop(index=highlighted_test.index)

        if not regular_test.empty:
            axis.scatter(regular_test["x_log"], regular_test["y_log"], s=26, alpha=0.6, color="steelblue", edgecolors="none", label="Test cells")
        if not regular_train.empty:
            axis.scatter(regular_train["x_log"], regular_train["y_log"], s=34, alpha=0.9, color="darkorange", edgecolors="black", linewidths=0.25, label="Train cells")
        if not highlighted_test.empty:
            axis.scatter(highlighted_test["x_log"], highlighted_test["y_log"], s=58, alpha=0.95, color=HIGHLIGHT_COLOR, edgecolors="black", linewidths=0.45, label="G1/G6/G26", zorder=4)
        if not highlighted_train.empty:
            axis.scatter(highlighted_train["x_log"], highlighted_train["y_log"], s=58, alpha=0.95, color=HIGHLIGHT_COLOR, edgecolors="black", linewidths=0.45, zorder=4)

        fit_row = summary_by_feature.loc[feature]
        if np.isfinite(fit_row["r2_train"]) and len(train_subset) >= 2:
            slope_train, intercept_train, _, _ = fit_linear_relationship(
                train_subset["x_log"].to_numpy(dtype=float),
                train_subset["y_log"].to_numpy(dtype=float),
            )
            x_fit = np.linspace(train_subset["x_log"].min(), train_subset["x_log"].max(), 200)
            axis.plot(x_fit, slope_train * x_fit + intercept_train, color="darkred", linewidth=1.8, label="Fit: train")

            if not highlighted_test.empty:
                predicted = np.exp(slope_train * highlighted_test["x_log"].to_numpy(dtype=float) + intercept_train)
                group_mapes = calculate_group_mapes(
                    highlighted_test["Cell"].astype(str).tolist(),
                    highlighted_test[target_column].to_numpy(dtype=float),
                    predicted,
                )
                axis.text(0.03, 0.97, format_group_mapes(group_mapes), transform=axis.transAxes, ha="left", va="top", fontsize=7,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": HIGHLIGHT_COLOR})

        axis.set_title(
            f"{feature}\n"
            f"train r={fit_row['pearson_r_train']:.3f}, R^2={fit_row['r2_train']:.3f}\n"
            f"test MAPE={fit_row['test_log_rescaled_mape_train']:.2f}%"
        )
        axis.set_xlabel(f"log(abs({feature}))")
        axis.set_ylabel(f"log({target_label})")

    for axis in axes[n_features:]:
        axis.set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels() if len(axes) else ([], [])
    if handles:
        unique_handles, unique_labels = [], []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_handles.append(handle)
                unique_labels.append(label)
        fig.legend(unique_handles, unique_labels, loc="upper center", ncol=min(4, len(unique_handles)), frameon=False)
    fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()
    if show:
        show_rendered_figure(fig)
    plt.close(fig)

def calculate_long_eol_validation_metrics(
    model_result: dict[str, object],
    raw_week_by_cell: dict[str, float],
    long_eol_threshold_weeks: float = LONG_EOL_VALIDATION_THRESHOLD_WEEKS,
) -> dict[str, object]:
    test_cells = [str(cell) for cell in model_result.get("test_cell_list", [])]
    test_actual = np.asarray(model_result.get("test_actual", []), dtype=float)
    test_pred = np.asarray(model_result.get("test_pred", []), dtype=float)

    if len(test_cells) == 0 or len(test_actual) != len(test_cells) or len(test_pred) != len(test_cells):
        return {
            "long_eol_threshold_weeks": float(long_eol_threshold_weeks),
            "long_eol_test_count": 0,
            "long_eol_test_mape": np.nan,
            "long_eol_test_max_ape": np.nan,
            "long_eol_test_max_ape_cell": None,
            "long_eol_test_ape_values": [],
            "long_eol_test_cells": [],
        }

    subset_indices = [
        index
        for index, cell in enumerate(test_cells)
        if float(raw_week_by_cell.get(cell, np.nan)) > float(long_eol_threshold_weeks)
    ]
    if not subset_indices:
        return {
            "long_eol_threshold_weeks": float(long_eol_threshold_weeks),
            "long_eol_test_count": 0,
            "long_eol_test_mape": np.nan,
            "long_eol_test_max_ape": np.nan,
            "long_eol_test_max_ape_cell": None,
            "long_eol_test_ape_values": [],
            "long_eol_test_cells": [],
        }

    subset_actual = test_actual[subset_indices]
    subset_pred = test_pred[subset_indices]
    subset_mape, subset_max_ape = calculate_ape_metrics(subset_actual, subset_pred)
    subset_ape_values = calculate_ape_values(subset_actual, subset_pred)
    return {
        "long_eol_threshold_weeks": float(long_eol_threshold_weeks),
        "long_eol_test_count": len(subset_indices),
        "long_eol_test_mape": subset_mape,
        "long_eol_test_max_ape": subset_max_ape,
        "long_eol_test_max_ape_cell": calculate_max_ape_cell(
            subset_actual,
            subset_pred,
            [test_cells[index] for index in subset_indices],
        ),
        "long_eol_test_ape_values": subset_ape_values.tolist(),
        "long_eol_test_cells": [test_cells[index] for index in subset_indices],
    }

def build_long_eol_plot_result(
    model_result: dict[str, object],
    raw_week_by_cell: dict[str, float],
    long_eol_threshold_weeks: float = LONG_EOL_VALIDATION_THRESHOLD_WEEKS,
) -> dict[str, object] | None:
    test_cells = [str(cell) for cell in model_result.get("test_cell_list", [])]
    test_actual = np.asarray(model_result.get("test_actual", []), dtype=float)
    test_pred = np.asarray(model_result.get("test_pred", []), dtype=float)
    if len(test_cells) == 0 or len(test_actual) != len(test_cells) or len(test_pred) != len(test_cells):
        return None

    subset_indices = [
        index
        for index, cell in enumerate(test_cells)
        if float(raw_week_by_cell.get(cell, np.nan)) > float(long_eol_threshold_weeks)
    ]
    if not subset_indices:
        return None

    subset_actual = test_actual[subset_indices]
    subset_pred = test_pred[subset_indices]
    subset_mape, subset_max_ape = calculate_ape_metrics(subset_actual, subset_pred)
    plot_result = dict(model_result)
    plot_result["dataset_label"] = (
        f"{model_result.get('dataset_label', 'dataset')}_long_eol_gt{int(long_eol_threshold_weeks)}"
    )
    plot_result["test_actual"] = subset_actual
    plot_result["test_pred"] = subset_pred
    plot_result["test_cell_list"] = [test_cells[index] for index in subset_indices]
    plot_result["test_count"] = len(subset_indices)
    plot_result["test_mape"] = subset_mape
    plot_result["test_max_ape"] = subset_max_ape
    plot_result["test_max_ape_cell"] = calculate_max_ape_cell(
        subset_actual,
        subset_pred,
        plot_result["test_cell_list"],
    )
    plot_result["test_r2"] = calculate_r2(subset_actual, subset_pred)
    return plot_result

def get_raw_week_at_target_map(raw_input_dir: Path, target_soh: float) -> dict[str, float]:
    results: dict[str, float] = {}
    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        try:
            df = pd.read_csv(csv_path, usecols=[RAW_WEEK_COLUMN, RAW_CAPACITY_COLUMN])
        except Exception:
            continue

        weeks = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
        capacities = pd.to_numeric(df[RAW_CAPACITY_COLUMN], errors="coerce").to_numpy(dtype=float)
        week_at_target = interpolate_value_at_soh(weeks, weeks, capacities, target_soh)
        if np.isfinite(week_at_target) and week_at_target > 0:
            results[cell] = float(week_at_target)
    return results

def build_threshold_split(raw_week_by_cell: dict[str, float], threshold: float) -> tuple[list[str], list[str]]:
    train_cells = sorted(cell for cell, week in raw_week_by_cell.items() if week <= threshold)
    test_cells = sorted(cell for cell, week in raw_week_by_cell.items() if week > threshold)
    if len(train_cells) < 2:
        raise RuntimeError(f"Need at least two training cells with week-at-70%%SOH <= {threshold:.0f}; found {len(train_cells)}.")
    if not test_cells:
        raise RuntimeError(f"No test cells found with week-at-70%%SOH > {threshold:.0f}.")
    return train_cells, test_cells

def load_interpolated_dataframe(
    raw_input_dir: Path,
    feature_row_index: int,
    train_cells: set[str],
    test_cells: set[str],
    capa_lim: float,
    interpolation_type: str,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    rows = []
    skipped = {
        "read_fail": 0,
        "missing_core_columns": 0,
        "missing_features": 0,
        "too_short": 0,
        "invalid_target": 0,
    }
    feature_columns: list[str] | None = None

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            raw_df = clean_input_dataframe(pd.read_csv(csv_path))
            df = load_and_interpolate(raw_df, capa_lim=capa_lim, interpolation_type=interpolation_type)
        except Exception:
            skipped["read_fail"] += 1
            continue

        if INTERPOLATED_TARGET_COLUMN not in df.columns:
            skipped["missing_core_columns"] += 1
            continue

        current_feature_columns = get_feature_columns(df.columns.tolist())
        if not current_feature_columns:
            skipped["missing_features"] += 1
            continue
        if len(df) <= feature_row_index:
            skipped["too_short"] += 1
            continue

        target_value = pd.to_numeric(df[INTERPOLATED_TARGET_COLUMN], errors="coerce").dropna()
        if target_value.empty or float(target_value.iloc[0]) <= 0:
            skipped["invalid_target"] += 1
            continue

        if feature_columns is None:
            feature_columns = current_feature_columns
        else:
            feature_columns = [column for column in feature_columns if column in current_feature_columns]

        row = {
            "Cell": cell,
            INTERPOLATED_TARGET_COLUMN: float(target_value.iloc[0]),
            "is_outer_field_selected": cell in train_cells,
            "is_test_cell": cell in test_cells,
            "source_file": str(csv_path),
        }
        for column in current_feature_columns:
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            value = numeric_series.iloc[feature_row_index]
            row[column] = float(value) if np.isfinite(value) else np.nan

        for column in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES:
            if column in row or column not in df.columns:
                continue
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            value = numeric_series.iloc[feature_row_index]
            row[column] = float(value) if np.isfinite(value) else np.nan
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No usable interpolated rows found in {raw_input_dir}")
    if not feature_columns:
        raise RuntimeError(f"No usable interpolated features found in {raw_input_dir}")

    data = pd.DataFrame(rows)
    feature_columns = [column for column in feature_columns if column in data.columns]
    return data, feature_columns, skipped

def load_raw_dataframe(
    raw_input_dir: Path,
    train_cells: set[str],
    test_cells: set[str],
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    rows = []
    skipped = {
        "read_fail": 0,
        "missing_core_columns": 0,
        "missing_feature_columns": 0,
        "invalid_target": 0,
    }
    feature_columns: list[str] | None = None

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            skipped["read_fail"] += 1
            continue

        if RAW_WEEK_COLUMN not in df.columns or RAW_CAPACITY_COLUMN not in df.columns:
            skipped["missing_core_columns"] += 1
            continue

        current_feature_columns = get_feature_columns(df.columns.tolist())
        if not current_feature_columns:
            skipped["missing_feature_columns"] += 1
            continue

        weeks = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
        capacities = pd.to_numeric(df[RAW_CAPACITY_COLUMN], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(capacities).any() and np.isfinite(capacities[0]) and capacities[0] != 0:
            df["SOH"] = capacities / capacities[0]
        week_at_target = interpolate_value_at_soh(weeks, weeks, capacities, TARGET_SOH)
        if not np.isfinite(week_at_target) or week_at_target <= 0:
            skipped["invalid_target"] += 1
            continue

        if "SOH" in INTERPOLATED_SEQUENCE_AND_MLP_FEATURES and "SOH" in df.columns and "SOH" not in current_feature_columns:
            current_feature_columns = ["SOH", *current_feature_columns]

        if feature_columns is None:
            feature_columns = current_feature_columns
        else:
            feature_columns = [column for column in feature_columns if column in current_feature_columns]

        feature_week = get_week_model_feature_week(cell)
        row = {
            "Cell": cell,
            RAW_TARGET_COLUMN: float(week_at_target),
            "is_outer_field_selected": cell in train_cells,
            "is_test_cell": cell in test_cells,
            "source_file": str(csv_path),
        }
        for column in current_feature_columns:
            feature_values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            row[column] = interpolate_value_at_x(weeks, feature_values, feature_week)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No usable raw rows found in {raw_input_dir}")
    if not feature_columns:
        raise RuntimeError(f"No usable raw features found in {raw_input_dir}")

    data = pd.DataFrame(rows)
    feature_columns = [column for column in feature_columns if column in data.columns]
    return data, feature_columns, skipped

def load_interpolated_sequence_dataset(
    raw_input_dir: Path,
    input_steps: int,
    sequence_features: list[str],
    train_cells: set[str],
    test_cells: set[str],
    capa_lim: float,
    interpolation_type: str,
) -> tuple[dict[str, object], dict[str, int]]:
    sequences = []
    targets = []
    cells = []
    skipped = {
        "read_fail": 0,
        "missing_core_columns": 0,
        "missing_features": 0,
        "too_short": 0,
        "invalid_target": 0,
        "nonfinite_sequence": 0,
    }

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            raw_df = clean_input_dataframe(pd.read_csv(csv_path))
            df = load_and_interpolate(raw_df, capa_lim=capa_lim, interpolation_type=interpolation_type)
        except Exception:
            skipped["read_fail"] += 1
            continue

        if INTERPOLATED_TARGET_COLUMN not in df.columns:
            skipped["missing_core_columns"] += 1
            continue
        if any(feature not in df.columns for feature in sequence_features):
            skipped["missing_features"] += 1
            continue
        if len(df) < input_steps:
            skipped["too_short"] += 1
            continue

        target_value = pd.to_numeric(df[INTERPOLATED_TARGET_COLUMN], errors="coerce").dropna()
        if target_value.empty or float(target_value.iloc[0]) <= 0:
            skipped["invalid_target"] += 1
            continue

        sequence = df.loc[: input_steps - 1, sequence_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(sequence).all():
            skipped["nonfinite_sequence"] += 1
            continue

        sequences.append(sequence)
        targets.append(float(target_value.iloc[0]))
        cells.append(cell)

    if not sequences:
        raise RuntimeError(f"No usable interpolated sequences found in {raw_input_dir}")

    return (
        {
            "sequences": np.asarray(sequences, dtype=float),
            "targets": np.asarray(targets, dtype=float),
            "cells": list(cells),
            "train_mask": np.array([cell in train_cells for cell in cells], dtype=bool),
            "test_mask": np.array([cell in test_cells for cell in cells], dtype=bool),
            "feature_columns": list(sequence_features),
        },
        skipped,
    )

def load_raw_sequence_dataset(
    raw_input_dir: Path,
    input_steps: int,
    sequence_features: list[str],
    train_cells: set[str],
    test_cells: set[str],
) -> tuple[dict[str, object], dict[str, int]]:
    sequences = []
    targets = []
    cells = []
    skipped = {
        "read_fail": 0,
        "missing_core_columns": 0,
        "missing_features": 0,
        "too_short": 0,
        "invalid_target": 0,
        "nonfinite_sequence": 0,
    }

    for csv_path in sorted(raw_input_dir.glob("Capacity_*.csv")):
        cell = parse_cell_name(csv_path)
        if cell not in train_cells and cell not in test_cells:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            skipped["read_fail"] += 1
            continue

        if RAW_WEEK_COLUMN not in df.columns or RAW_CAPACITY_COLUMN not in df.columns:
            skipped["missing_core_columns"] += 1
            continue

        weeks = pd.to_numeric(df[RAW_WEEK_COLUMN], errors="coerce").to_numpy(dtype=float)
        capacities = pd.to_numeric(df[RAW_CAPACITY_COLUMN], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(capacities).any() and np.isfinite(capacities[0]) and capacities[0] != 0:
            df["SOH"] = capacities / capacities[0]
        if any(feature not in df.columns for feature in sequence_features):
            skipped["missing_features"] += 1
            continue
        feature_week = get_week_model_feature_week(cell)
        if len(df) < 2 or not np.isfinite(weeks).any() or float(np.nanmax(weeks)) < feature_week:
            skipped["too_short"] += 1
            continue

        week_at_target = interpolate_value_at_soh(weeks, weeks, capacities, TARGET_SOH)
        if not np.isfinite(week_at_target) or week_at_target <= 0:
            skipped["invalid_target"] += 1
            continue

        feature_start_candidates: list[float] = []
        for feature in sequence_features:
            feature_values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
            finite_feature_weeks = weeks[np.isfinite(weeks) & np.isfinite(feature_values)]
            if finite_feature_weeks.size == 0:
                feature_start_candidates = []
                break
            feature_start_candidates.append(float(np.nanmin(finite_feature_weeks)))

        if not feature_start_candidates:
            skipped["missing_features"] += 1
            continue

        if max(feature_start_candidates) > feature_week:
            skipped["too_short"] += 1
            continue

        sequence_grid = np.linspace(0.0, feature_week, input_steps)
        sequence_columns: list[np.ndarray] = []
        for feature in sequence_features:
            feature_values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(weeks) & np.isfinite(feature_values)
            first_feature_week = float(np.min(weeks[finite_mask]))
            interpolated_feature = np.array(
                [
                    0.0
                    if raw_sequence_feature_should_zero_fill(feature) and float(target_x) < first_feature_week
                    else interpolate_value_at_x(weeks, feature_values, target_x)
                    for target_x in sequence_grid
                ],
                dtype=float,
            )
            sequence_columns.append(interpolated_feature)
        sequence = np.column_stack(sequence_columns)
        if not np.isfinite(sequence).all():
            skipped["nonfinite_sequence"] += 1
            continue

        sequences.append(sequence)
        targets.append(float(week_at_target))
        cells.append(cell)

    if not sequences:
        raise RuntimeError(f"No usable raw sequences found in {raw_input_dir}")

    return (
        {
            "sequences": np.asarray(sequences, dtype=float),
            "targets": np.asarray(targets, dtype=float),
            "cells": list(cells),
            "train_mask": np.array([cell in train_cells for cell in cells], dtype=bool),
            "test_mask": np.array([cell in test_cells for cell in cells], dtype=bool),
            "feature_columns": list(sequence_features),
        },
        skipped,
    )

def build_feature_subset(data: pd.DataFrame, feature: str, target_column: str) -> pd.DataFrame:
    subset = data[["Cell", "is_outer_field_selected", "is_test_cell", feature, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
    subset = subset[(subset[feature].abs() > 0) & (subset[target_column] > 0)].copy()
    if subset.empty:
        subset["x_log"] = pd.Series(dtype=float)
        subset["y_log"] = pd.Series(dtype=float)
        return subset
    subset["x_log"] = np.log(np.abs(subset[feature].to_numpy(dtype=float)))
    subset["y_log"] = np.log(subset[target_column].to_numpy(dtype=float))
    return subset

def choose_best_linear_feature(
    fit_summary: pd.DataFrame,
    *,
    data: pd.DataFrame | None = None,
    target_column: str | None = None,
) -> str:
    ranked = fit_summary.copy()
    ranked["pearson_r_train_abs"] = pd.to_numeric(ranked["pearson_r_train"], errors="coerce").abs()
    ranked = ranked.sort_values(
        ["pearson_r_train_abs", "feature"],
        ascending=[False, True],
        na_position="last",
    )
    for _, row in ranked.iterrows():
        feature = str(row["feature"]).strip()
        if feature and np.isfinite(pd.to_numeric(row.get("pearson_r_train_abs"), errors="coerce")):
            return feature
    raise RuntimeError("Could not determine a usable linear-model feature from threshold-filtered |pearson_r_train|.")

def select_features_by_abs_pearson(
    fit_summary: pd.DataFrame,
    candidate_features: list[str],
    min_abs_pearson: float,
    label: str,
) -> tuple[list[str], pd.DataFrame]:
    if not candidate_features:
        raise RuntimeError(f"No candidate features were provided for {label}.")

    summary_subset = fit_summary[fit_summary["feature"].isin(candidate_features)].copy()
    if summary_subset.empty:
        raise RuntimeError(f"No fit-summary rows were available for {label}.")

    summary_subset["pearson_r_train_abs"] = pd.to_numeric(
        summary_subset["pearson_r_train"],
        errors="coerce",
    ).abs()
    selected_summary = (
        summary_subset[summary_subset["pearson_r_train_abs"] > float(min_abs_pearson)]
        .copy()
        .sort_values(["pearson_r_train_abs", "feature"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    selected_features = [feature for feature in candidate_features if feature in set(selected_summary["feature"].astype(str))]

    if not selected_features:
        available_text = summary_subset[["feature", "pearson_r_train"]].sort_values(
            ["pearson_r_train"],
            ascending=[False],
            na_position="last",
        ).to_string(index=False)
        raise RuntimeError(
            f"No {label} features met |pearson_r_train| > {float(min_abs_pearson):.2f}.\n"
            f"Available Pearson values:\n{available_text}"
        )

    selected_summary = selected_summary[
        ["feature", "pearson_r_train", "pearson_r_train_abs", "r2_train", "n_train", "r2_all", "pearson_r_all", "n_all"]
    ].copy()
    return selected_features, selected_summary


def train_linear_feature_regressor(data: pd.DataFrame, fit_summary: pd.DataFrame, target_column: str,
    dataset_label: str, feature_name: str | None = None) -> dict[str, object]:
    """Adapter onto the shared linear regressor in core_functions.temporal_models."""
    g = globals()
    return temporal_models.train_linear_feature_regressor(data=data, fit_summary=fit_summary,
        target_column=target_column, dataset_label=dataset_label, feature_name=feature_name,
        model_name="linear", train_column="is_outer_field_selected",
        choose_best_feature_fn=g["choose_best_linear_feature"], build_feature_subset_fn=g["build_feature_subset"],
        calculate_ape_metrics=g.get("calculate_ape_metrics"), calculate_r2=g.get("calculate_r2"))

def summarize_model_rows(threshold: float, model_results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in model_results:
        model_name = result.get("model_name")
        if not model_name:
            model_name = "mlp"
        test_ape_values = extract_test_ape(result)
        long_eol_ape_values = np.asarray(result.get("long_eol_test_ape_values", []), dtype=float)
        test_cells = [str(cell) for cell in result.get("test_cell_list", [])]
        test_max_ape_cell = result.get("test_max_ape_cell")
        if test_max_ape_cell in (None, ""):
            test_max_ape_cell = calculate_max_ape_cell(
                np.asarray(result.get("test_actual", []), dtype=float),
                np.asarray(result.get("test_pred", []), dtype=float),
                test_cells,
            )
        long_eol_test_cells = [str(cell) for cell in result.get("long_eol_test_cells", [])]
        long_eol_test_max_ape_cell = result.get("long_eol_test_max_ape_cell")
        if long_eol_test_max_ape_cell in (None, ""):
            long_eol_test_max_ape_cell = calculate_max_ape_cell(
                np.asarray(result.get("test_actual", []), dtype=float),
                np.asarray(result.get("test_pred", []), dtype=float),
                long_eol_test_cells if long_eol_test_cells else test_cells,
            )
        rows.append(
            {
                "threshold_weeks": threshold,
                "model_name": model_name,
                "dataset_label": result.get("dataset_label"),
                "selected_feature": result.get("selected_feature"),
                "feature_count": result.get("feature_count"),
                "train_count": result.get("train_count"),
                "test_count": result.get("test_count"),
                "train_mape": result.get("train_mape"),
                "test_mape": result.get("test_mape"),
                "train_rmse": result.get("train_rmse"),
                "test_rmse": result.get("test_rmse"),
                "test_ape_std": float(np.std(test_ape_values)) if test_ape_values.size else np.nan,
                "test_ape_values": json.dumps(test_ape_values.astype(float).tolist()),
                "test_cells": json.dumps(test_cells),
                "train_actual": json.dumps(np.asarray(result.get("train_actual", []), dtype=float).tolist()),
                "train_pred": json.dumps(np.asarray(result.get("train_pred", []), dtype=float).tolist()),
                "test_actual": json.dumps(np.asarray(result.get("test_actual", []), dtype=float).tolist()),
                "test_pred": json.dumps(np.asarray(result.get("test_pred", []), dtype=float).tolist()),
                "test_actual_time": json.dumps(np.asarray(result.get("test_actual_time", []), dtype=float).tolist()),
                "test_pred_time": json.dumps(np.asarray(result.get("test_pred_time", []), dtype=float).tolist()),
                "test_actual_throughput": json.dumps(
                    np.asarray(result.get("test_actual_throughput", []), dtype=float).tolist()
                ),
                "test_pred_throughput": json.dumps(
                    np.asarray(result.get("test_pred_throughput", []), dtype=float).tolist()
                ),
                "train_max_ape": result.get("train_max_ape"),
                "test_max_ape": result.get("test_max_ape"),
                "test_max_ape_cell": test_max_ape_cell,
                "long_eol_threshold_weeks": result.get("long_eol_threshold_weeks"),
                "long_eol_test_count": result.get("long_eol_test_count"),
                "long_eol_test_mape": result.get("long_eol_test_mape"),
                "long_eol_test_ape_std": float(np.std(long_eol_ape_values)) if long_eol_ape_values.size else np.nan,
                "long_eol_test_ape_values": json.dumps(long_eol_ape_values.astype(float).tolist()),
                "long_eol_test_cells": json.dumps(long_eol_test_cells),
                "long_eol_test_max_ape": result.get("long_eol_test_max_ape"),
                "long_eol_test_max_ape_cell": long_eol_test_max_ape_cell,
                "train_r2": result.get("train_r2"),
                "test_r2": result.get("test_r2"),
                "epochs": result.get("epochs"),
                "learning_rate": result.get("learning_rate"),
            }
        )
    return pd.DataFrame(rows)

def run_threshold_scenario(
    threshold: float,
    raw_input_dir: Path,
    feature_row_index: int,
    output_dir: Path,
    raw_week_by_cell: dict[str, float],
    capa_lim: float,
    interpolation_type: str,
    min_abs_pearson: float,
    show: bool,
) -> pd.DataFrame:
    interpolated_input_steps = core_autoanchor.resolve_interpolated_tcnn_input_steps(threshold)
    raw_input_steps = core_autoanchor.resolve_raw_tcnn_input_steps(threshold)
    interpolated_tcnn_hyperparameters = dict(BEST_INTERPOLATED_TCNN_HYPERPARAMETERS)
    interpolated_tcnn_hyperparameters["input_steps"] = interpolated_input_steps
    raw_tcnn_hyperparameters = dict(BEST_INTERPOLATED_TCNN_HYPERPARAMETERS)
    raw_tcnn_hyperparameters["input_steps"] = raw_input_steps

    label = f"eligible{int(threshold)}_features_{FEATURE_SELECTION_LABEL}"
    scenario_dir = output_dir / label
    scenario_dir.mkdir(parents=True, exist_ok=True)

    train_cells, test_cells = build_threshold_split(raw_week_by_cell, threshold)
    train_cells_set = set(train_cells)
    test_cells_set = set(test_cells)

    interpolated_data, interpolated_features, interpolated_skipped = load_interpolated_dataframe(
        raw_input_dir=raw_input_dir,
        feature_row_index=feature_row_index,
        train_cells=train_cells_set,
        test_cells=test_cells_set,
        capa_lim=capa_lim,
        interpolation_type=interpolation_type,
    )
    raw_data, raw_features, raw_skipped = load_raw_dataframe(
        raw_input_dir=raw_input_dir,
        train_cells=train_cells_set,
        test_cells=test_cells_set,
    )

    interpolated_summary = build_fit_summary(interpolated_data, interpolated_features, target_column=INTERPOLATED_TARGET_COLUMN)
    raw_summary = build_fit_summary(raw_data, raw_features, target_column=RAW_TARGET_COLUMN)

    raw_linear = train_linear_feature_regressor(
        data=raw_data,
        fit_summary=raw_summary,
        target_column=RAW_TARGET_COLUMN,
        dataset_label=f"{label}_raw_week70",
    )

    interpolated_mlp_features = get_required_features(
        interpolated_data.columns.tolist(),
        select_soh_meanvar_prediction_features(
            interpolated_data.columns.tolist(),
            label=f"{label} interpolated MLP/TCNN",
        ),
        label=f"{label} interpolated MLP/TCNN",
    )
    interpolated_mlp_selected_summary = build_fit_summary(
        interpolated_data,
        interpolated_mlp_features,
        target_column=INTERPOLATED_TARGET_COLUMN,
    )
    raw_mlp_features = get_required_features(
        raw_data.columns.tolist(),
        select_soh_meanvar_prediction_features(
            raw_data.columns.tolist(),
            label=f"{label} raw MLP/TCNN",
        ),
        label=f"{label} raw MLP/TCNN",
    )
    raw_mlp_selected_summary = build_fit_summary(
        raw_data,
        raw_mlp_features,
        target_column=RAW_TARGET_COLUMN,
    )

    interpolated_sequence_data, interpolated_sequence_skipped = load_interpolated_sequence_dataset(
        raw_input_dir=raw_input_dir,
        input_steps=interpolated_input_steps,
        sequence_features=interpolated_mlp_features,
        train_cells=train_cells_set,
        test_cells=test_cells_set,
        capa_lim=capa_lim,
        interpolation_type=interpolation_type,
    )
    raw_sequence_data, raw_sequence_skipped = load_raw_sequence_dataset(
        raw_input_dir=raw_input_dir,
        input_steps=raw_input_steps,
        sequence_features=raw_mlp_features,
        train_cells=train_cells_set,
        test_cells=test_cells_set,
    )

    interpolated_tcnn = train_temporal_cnn_regressor(
        sequence_dataset=interpolated_sequence_data,
        dataset_label=f"{label}_interpolated_step70",
        hyperparameters=interpolated_tcnn_hyperparameters,
    )
    raw_tcnn = train_temporal_cnn_regressor(
        sequence_dataset=raw_sequence_data,
        dataset_label=f"{label}_raw_week70",
        hyperparameters=raw_tcnn_hyperparameters,
    )
    selection_df = pd.DataFrame(
        {
            "Cell": sorted(raw_week_by_cell),
            RAW_TARGET_COLUMN: [raw_week_by_cell[cell] for cell in sorted(raw_week_by_cell)],
            f"train_le_{int(threshold)}": [raw_week_by_cell[cell] <= threshold for cell in sorted(raw_week_by_cell)],
            f"test_gt_{int(threshold)}": [raw_week_by_cell[cell] > threshold for cell in sorted(raw_week_by_cell)],
            "selected_for_train": [cell in train_cells_set for cell in sorted(raw_week_by_cell)],
            "used_for_test": [cell in test_cells_set for cell in sorted(raw_week_by_cell)],
        }
    )
    selection_df.to_csv(scenario_dir / f"{label}_train_selection_IOWA.csv", index=False)
    interpolated_summary.to_csv(scenario_dir / f"{label}_interpolated_fit_summary_IOWA.csv", index=False)
    interpolated_data.to_csv(scenario_dir / f"{label}_interpolated_cell_data_IOWA.csv", index=False)
    raw_summary.to_csv(scenario_dir / f"{label}_raw_fit_summary_IOWA.csv", index=False)
    raw_data.to_csv(scenario_dir / f"{label}_raw_cell_data_IOWA.csv", index=False)
    interpolated_mlp_selected_summary.to_csv(
        scenario_dir / f"{label}_interpolated_fixed_mlp_tcnn_features_IOWA.csv",
        index=False,
    )
    raw_mlp_selected_summary.to_csv(
        scenario_dir / f"{label}_raw_fixed_mlp_tcnn_features_IOWA.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"dataset": "interpolated_tabular", **interpolated_skipped},
            {"dataset": "raw_tabular", **raw_skipped},
            {"dataset": "interpolated_tcnn", **interpolated_sequence_skipped},
            {"dataset": "raw_tcnn", **raw_sequence_skipped},
        ]
    ).to_csv(scenario_dir / f"{label}_skipped_summary_IOWA.csv", index=False)

    plot_random_relationship_subplots(
        data=interpolated_data,
        feature_columns=interpolated_features,
        fit_summary=interpolated_summary,
        output_dir=scenario_dir,
        target_column=INTERPOLATED_TARGET_COLUMN,
        target_label="interpolated step at 70% SOH",
        output_stem=f"{label}_interpolated_log_feature_vs_log_step70",
        title_prefix=(
            f"IOWA interpolated features ({interpolation_type}, capa_lim={capa_lim:.2f}), "
            f"train <= {threshold:.0f} weeks, test > {threshold:.0f} weeks"
        ),
        show=show,
    )
    plot_random_relationship_subplots(data=raw_data, feature_columns=raw_features, fit_summary=raw_summary, output_dir=scenario_dir, target_column=RAW_TARGET_COLUMN,
        target_label="week at 70% SOH", output_stem=f"{label}_raw_log_feature_vs_log_week70", title_prefix=f"IOWA raw features, train <= {threshold:.0f} weeks, test > {threshold:.0f} weeks",
        show=False)

    long_eol_plot_results: dict[str, dict[str, object] | None] = {}
    for model_result, model_label in [
        (raw_linear, "Linear"),
        (interpolated_tcnn, "TCNN"),
        (raw_tcnn, "TCNN"),
    ]:
        model_result.update(
            calculate_long_eol_validation_metrics(
                model_result=model_result,
                raw_week_by_cell=raw_week_by_cell,
            )
        )
        long_eol_plot_result = build_long_eol_plot_result(
            model_result=model_result,
            raw_week_by_cell=raw_week_by_cell,
        )
        result_key = f"{model_result['dataset_label']}::{model_result['model_name']}"
        long_eol_plot_results[result_key] = long_eol_plot_result

    plot_model_family_comparison_grid(
        threshold=threshold,
        output_dir=scenario_dir,
        comparison_rows=[
            {
                "model_label": "Linear raw",
                "interpolated": None,
                "raw": raw_linear,
                "interpolated_long": None,
                "raw_long": long_eol_plot_results.get(f"{raw_linear['dataset_label']}::{raw_linear['model_name']}"),
            },
            {
                "model_label": "TCNN",
                "interpolated": interpolated_tcnn,
                "raw": raw_tcnn,
                "interpolated_long": long_eol_plot_results.get(f"{interpolated_tcnn['dataset_label']}::{interpolated_tcnn['model_name']}"),
                "raw_long": long_eol_plot_results.get(f"{raw_tcnn['dataset_label']}::{raw_tcnn['model_name']}"),
            },
        ],
        long_eol_threshold_weeks=LONG_EOL_VALIDATION_THRESHOLD_WEEKS,
        show=show,
    )

    model_summary = summarize_model_rows(
        threshold=threshold,
        model_results=[
            raw_linear,
            interpolated_tcnn,
            raw_tcnn,
        ],
    )
    model_summary.to_csv(scenario_dir / f"{label}_model_comparison_summary_IOWA.csv", index=False)

    print(f"\nThreshold <= {threshold:.0f} weeks")
    print(f"Train cells: {len(train_cells)}")
    print(f"Test cells: {len(test_cells)}")
    print(f"Interpolated skipped: {interpolated_skipped}")
    print(f"Raw skipped: {raw_skipped}")
    print(f"Interpolated sequence skipped: {interpolated_sequence_skipped}")
    print(f"Raw sequence skipped: {raw_sequence_skipped}")
    print(f"Interpolated TCNN fixed features: {interpolated_mlp_features}")
    print(f"Raw TCNN fixed features: {raw_mlp_features}")
    print(model_summary.to_string(index=False))

    return model_summary

def main() -> None:
    args = parse_args()
    configure_comparison_seeds()
    raw_input_dir = args.raw_input_dir.resolve()
    output_dir = args.output_dir.resolve()
    capa_lims = [float(capa_lim) for capa_lim in args.capa_lims]

    if not raw_input_dir.exists():
        raise FileNotFoundError(f"Raw input directory does not exist: {raw_input_dir}")

    raw_week_by_cell = get_raw_week_at_target_map(raw_input_dir, TARGET_SOH)
    if not raw_week_by_cell:
        raise RuntimeError(f"No IOWA cells with usable week-at-70%%SOH targets were found in {raw_input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    overall_failures: list[dict[str, object]] = []
    aug70_state: dict[str, object] | None = None
    aug80_state: dict[str, object] | None = None
    for capa_lim in capa_lims:
        capa_output_dir = output_dir / format_soh_folder_name(capa_lim)
        capa_output_dir.mkdir(parents=True, exist_ok=True)

        all_summaries: list[pd.DataFrame] = []
        failures: list[dict[str, object]] = []
        for threshold in args.thresholds:
            try:
                summary = run_threshold_scenario(
                    threshold=float(threshold),
                    raw_input_dir=raw_input_dir,
                    feature_row_index=args.feature_row_index,
                    output_dir=capa_output_dir,
                    raw_week_by_cell=raw_week_by_cell,
                    capa_lim=float(capa_lim),
                    interpolation_type=args.interpolation_type,
                    min_abs_pearson=float(args.min_abs_pearson),
                    show=args.show,
                )
            except Exception as exc:
                failures.append(
                    {
                        "capa_lim": float(capa_lim),
                        "threshold_weeks": float(threshold),
                        "error": str(exc),
                    }
                )
                print(f"Skipping threshold {threshold:.0f} for capa_lim {capa_lim:.2f}: {exc}")
                continue
            extra_rows: list[dict[str, object]] = []
            aug70_row, aug70_state = run_augmented_tcnn_threshold(
                module=iowa_aug70,
                threshold=float(threshold),
                state=aug70_state,
            )
            extra_rows.append(aug70_row)
            aug80_row, aug80_state = run_augmented_tcnn_threshold(
                module=iowa_aug80,
                threshold=float(threshold),
                state=aug80_state,
            )
            extra_rows.append(aug80_row)
            summary = pd.concat([summary, pd.DataFrame(extra_rows)], ignore_index=True)
            all_summaries.append(summary)

        if not all_summaries:
            overall_failures.append(
                {
                    "capa_lim": float(capa_lim),
                    "error": "All requested thresholds failed for this SOH limit.",
                }
            )
            continue

        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary.to_csv(capa_output_dir / "threshold_model_comparison_summary_IOWA.csv", index=False)
        pd.DataFrame(
            [
                {
                    "raw_input_dir": str(raw_input_dir),
                    "target_soh": TARGET_SOH,
                    "interpolation_type": args.interpolation_type,
                    "capa_lim": float(capa_lim),
                    "soh_folder": format_soh_folder_name(capa_lim),
                    "feature_selection": FEATURE_SELECTION_LABEL,
                    "linear_feature_selection_method": LINEAR_FEATURE_SELECTION_METHOD,
                    "min_abs_pearson": float(args.min_abs_pearson),
                    "feature_row_index": args.feature_row_index,
                    "comparison_seeds": " ".join(str(int(seed)) for seed in COMPARISON_SEEDS),
                }
            ]
        ).to_csv(capa_output_dir / "threshold_run_config_IOWA_aug.csv", index=False)
        if failures:
            pd.DataFrame(failures).to_csv(capa_output_dir / "threshold_failures_IOWA.csv", index=False)

    if overall_failures:
        pd.DataFrame(overall_failures).to_csv(output_dir / "threshold_overall_failures_IOWA.csv", index=False)

if __name__ == "__main__":
    main()
