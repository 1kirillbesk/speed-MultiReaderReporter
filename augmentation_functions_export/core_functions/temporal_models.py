from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from core_functions.augmentation_run_config import (
    MODEL_TRAINING_SEED_COUNT,
    MODEL_TRAINING_SEED_SOURCE,
    build_model_training_seed_list,
)
import json
import pandas as pd
from sklearn.linear_model import ElasticNet
from core_functions import augmentation_run_config as aug_cfg
from sklearn.svm import SVR

DEFAULT_TCNN_SEED_SOURCE = MODEL_TRAINING_SEED_SOURCE
DEFAULT_TCNN_SEED_COUNT = MODEL_TRAINING_SEED_COUNT
DEFAULT_TCNN_SEED_UPPER_BOUND = 101
DEFAULT_TCNN_LOSS_FUNCTION = "mse"
DEFAULT_TCNN_SOURCE_BALANCE_MODE = "ordinary_mse"
SEED_TRAINING_MAX_WORKERS = 2
DEFAULT_MLP_BASE_CONFIG: dict[str, object] = {
    "hidden_dim": 16,
    "hidden_layers": 2,
    "dropout_rate": 0.0,
    "epochs": 200,
    "learning_rate": 0.01,
    "weight_decay": 1e-4,
    "seeds": build_model_training_seed_list(seed_upper_bound=101),
    "input_extrapolation_margin": 0.05,
    "target_log_extrapolation_margin": 0.10,
    "validation_split": 0.3,
    "early_stopping_patience": 20,
}
# Compatibility names resolved from the single global model-transform policy.
DEFAULT_TCNN_INPUT_SCALER = str(aug_cfg.MODEL_INPUT_TRANSFORM_MODE).strip().lower()
DEFAULT_TCNN_INPUT_TRANSFORM_MODE = f"{DEFAULT_TCNN_INPUT_SCALER}_only"
DEFAULT_TCNN_TARGET_TRANSFORM_MODE = str(aug_cfg.MODEL_TARGET_TRANSFORM_MODE).strip().lower()
VALID_TCNN_INPUT_SCALERS = frozenset({"standard_scaler", "minmax_scaler"})
DEFAULT_TCNN_BASE_CONFIG: dict[str, object] = {
    "input_steps": 10,
    "epochs": 1000,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "dropout_rate": 0.1,
    "seed_count": DEFAULT_TCNN_SEED_COUNT,
    "seed_source": DEFAULT_TCNN_SEED_SOURCE,
    "seed_upper_bound": DEFAULT_TCNN_SEED_UPPER_BOUND,
    "early_stopping_patience": 100,
    "scheduler_patience": 10,
    "scheduler_factor": 0.8,
    "batch_size": 64,
    "validation_split": float(aug_cfg.TCNN_VALIDATION_SPLIT),
    "validation_interval": 1,
    "input_clipping_mode": str(aug_cfg.TCNN_INPUT_CLIPPING_MODE),
    "input_extrapolation_margin": aug_cfg.tcnn_input_clipping_fraction(),
    "output_clipping_mode": str(aug_cfg.TCNN_OUTPUT_CLIPPING_MODE),
    "target_log_extrapolation_margin": aug_cfg.tcnn_output_clipping_fraction(),
    "conv1_channels": 32,
    "conv_channels": 16,
    "fc_hidden_dim": 16,
    "fc_hidden_dim2": 8,
    "conv1_kernel_size": 7,
    "conv2_kernel_size": 7,
    "model_structure": "average1",
    "loss_function": DEFAULT_TCNN_LOSS_FUNCTION,
    "source_balance_mode": DEFAULT_TCNN_SOURCE_BALANCE_MODE,
    "input_transform_mode": DEFAULT_TCNN_INPUT_TRANSFORM_MODE,
    "target_transform_mode": DEFAULT_TCNN_TARGET_TRANSFORM_MODE,
    "input_scaler": DEFAULT_TCNN_INPUT_SCALER,
}
DEFAULT_EXTRAPOLATION_TCNN_STRUCTURE: dict[str, object] = {
    "conv_channels": 24,
    "fc_hidden_dim": 12,
    "fc_hidden_dim2": None,
    "model_structure": "extrapolation_residual",
}

# Global EOL-prediction MLP: the augmented-data counterpart of the TCNN above.
# Values follow the Li reference configuration; change them here to move every dataset.
# (Distinct from DEFAULT_MLP_BASE_CONFIG, which belongs to FeedForwardRegressor.)
DEFAULT_EOL_MLP_BASE_CONFIG: dict[str, object] = {
    "hidden_layers": [32, 32],
    "dropout_rate": 0.1,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "seed_count": MODEL_TRAINING_SEED_COUNT,
    "seed_source": MODEL_TRAINING_SEED_SOURCE,
    "seed_upper_bound": DEFAULT_TCNN_SEED_UPPER_BOUND,
    "batch_size": 64,
    "epochs": 1000,
    "validation_split": float(aug_cfg.EOL_MLP_VALIDATION_SPLIT),
    "early_stopping_patience": 100,
    "scheduler_patience": 10,
    "scheduler_factor": 0.5,
    "loss_function": DEFAULT_TCNN_LOSS_FUNCTION,
    "input_scaler": DEFAULT_TCNN_INPUT_SCALER,
    "target_transform": DEFAULT_TCNN_TARGET_TRANSFORM_MODE,
    "input_clipping_mode": str(aug_cfg.TCNN_INPUT_CLIPPING_MODE),
    "input_extrapolation_margin": aug_cfg.tcnn_input_clipping_fraction(),
    "output_clipping_mode": str(aug_cfg.TCNN_OUTPUT_CLIPPING_MODE),
    "target_log_extrapolation_margin": aug_cfg.tcnn_output_clipping_fraction(),
}
VALID_EOL_MLP_TARGET_TRANSFORMS = frozenset(
    {
        "target_divisor_then_log",
        "target_divisor_then_log_then_standard_scaler",
        "log_only",
        "standard_scaler",
        "minmax_scaler",
    }
)

def apply_stabilized_regression_head(values: torch.Tensor) -> torch.Tensor:
    """Apply the globally configured neural-regression output head."""
    mode = str(aug_cfg.MODEL_OUTPUT_HEAD_ACTIVATION).strip().lower()
    if mode == "linear":
        return values
    if mode == "asinh":
        return torch.asinh(values)
    if mode == "tanh":
        return torch.tanh(values)
    raise ValueError(
        f"Unsupported MODEL_OUTPUT_HEAD_ACTIVATION={aug_cfg.MODEL_OUTPUT_HEAD_ACTIVATION!r}. "
        "Expected 'linear', 'asinh', or 'tanh'."
    )


class FeedForwardRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        hidden_layers: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = int(input_dim)
        for _ in range(max(1, int(hidden_layers))):
            layers.append(nn.Linear(in_features, int(hidden_dim)))
            layers.append(nn.ReLU())
            if float(dropout_rate) > 0:
                layers.append(nn.Dropout(float(dropout_rate)))
            in_features = int(hidden_dim)
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MLPWithMCDropout(nn.Module):
    """Small MLP regressor with dropout kept active for Monte Carlo prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.fc3 = nn.Linear(int(hidden_dim), int(output_dim))
        self.dropout = nn.Dropout(p=float(dropout_rate))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(inputs))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        inputs: torch.Tensor,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.train()
        predictions = torch.stack([self(inputs) for _ in range(max(1, int(num_samples)))])
        mean = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        if not was_training:
            self.eval()
        return mean, uncertainty

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        *,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for epoch in range(max(1, int(num_epochs))):
            self.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * int(inputs.size(0))

            train_loss /= max(1, len(train_loader.dataset))
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += float(loss.item()) * int(inputs.size(0))
            val_loss /= max(1, len(val_loader.dataset))
            history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{int(num_epochs)}, "
                    f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
                )
        return history


def normalize_mlp_hyperparameters(
    hyperparameters: dict[str, object] | None = None,
    *,
    base_config: dict[str, object] | None = None,
) -> dict[str, object]:
    config = deepcopy(DEFAULT_MLP_BASE_CONFIG if base_config is None else base_config)
    if hyperparameters:
        config.update(hyperparameters)
    config["hidden_dim"] = max(1, int(config["hidden_dim"]))
    config["hidden_layers"] = max(1, int(config["hidden_layers"]))
    config["dropout_rate"] = float(config["dropout_rate"])
    config["epochs"] = max(1, int(config["epochs"]))
    config["learning_rate"] = float(config["learning_rate"])
    config["weight_decay"] = float(config["weight_decay"])
    config["input_extrapolation_margin"] = max(0.0, float(config["input_extrapolation_margin"]))
    config["target_log_extrapolation_margin"] = max(0.0, float(config["target_log_extrapolation_margin"]))
    config["validation_split"] = float(config["validation_split"])
    config["early_stopping_patience"] = max(1, int(config["early_stopping_patience"]))
    config["seeds"] = [int(seed) for seed in config.get("seeds", build_model_training_seed_list(seed_upper_bound=101))]
    if not config["seeds"]:
        raise ValueError("MLP hyperparameters require at least one seed.")
    return config


class EolPredictionMLP(nn.Module):
    """Canonical augmented-data EOL regression MLP (Li reference architecture)."""

    def __init__(self, n_in: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(n_in)
        for hidden_dim in hidden:
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.BatchNorm1d(int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_stabilized_regression_head(self.net(x)).squeeze(-1)

def normalize_eol_mlp_hyperparameters(
    hyperparameters: dict[str, object] | None = None,
    *,
    base_config: dict[str, object] | None = None,
) -> dict[str, object]:
    config = deepcopy(DEFAULT_EOL_MLP_BASE_CONFIG if base_config is None else base_config)
    if hyperparameters:
        config.update(hyperparameters)
    config["hidden_layers"] = [max(1, int(width)) for width in config["hidden_layers"]]
    if not config["hidden_layers"]:
        raise ValueError("EOL MLP requires at least one hidden layer.")
    config["dropout_rate"] = float(config["dropout_rate"])
    config["learning_rate"] = float(config["learning_rate"])
    config["weight_decay"] = float(config["weight_decay"])
    config["seed_count"] = max(1, int(config["seed_count"]))
    config["seed_source"] = int(config["seed_source"])
    config["seed_upper_bound"] = max(int(config["seed_upper_bound"]), int(config["seed_count"]) + 1)
    config["seeds"] = build_model_training_seed_list(
        seed_count=int(config["seed_count"]),
        seed_source=int(config["seed_source"]),
        seed_upper_bound=int(config["seed_upper_bound"]),
    )
    config["batch_size"] = max(1, int(config["batch_size"]))
    config["epochs"] = max(1, int(config["epochs"]))
    config["validation_split"] = float(config["validation_split"])
    config["early_stopping_patience"] = max(1, int(config["early_stopping_patience"]))
    config["scheduler_patience"] = max(1, int(config["scheduler_patience"]))
    config["scheduler_factor"] = float(config["scheduler_factor"])
    config["loss_function"] = str(config["loss_function"]).strip().lower()
    config["input_scaler"] = str(config["input_scaler"]).strip().lower()
    if config["input_scaler"] not in VALID_TCNN_INPUT_SCALERS:
        raise ValueError(
            f"Unsupported input_scaler {config['input_scaler']!r}. "
            f"Expected one of: {sorted(VALID_TCNN_INPUT_SCALERS)}."
        )
    config["target_transform"] = str(config["target_transform"]).strip().lower()
    if config["target_transform"] not in VALID_EOL_MLP_TARGET_TRANSFORMS:
        raise ValueError(
            f"Unsupported target_transform {config['target_transform']!r}. "
            f"Expected one of: {sorted(VALID_EOL_MLP_TARGET_TRANSFORMS)}."
        )
    config["input_clipping_mode"] = str(aug_cfg.TCNN_INPUT_CLIPPING_MODE)
    config["input_extrapolation_margin"] = aug_cfg.tcnn_input_clipping_fraction()
    config["output_clipping_mode"] = str(aug_cfg.TCNN_OUTPUT_CLIPPING_MODE)
    config["target_log_extrapolation_margin"] = aug_cfg.tcnn_output_clipping_fraction()
    return config

def split_validation_arrays(x_data: np.ndarray, y_data: np.ndarray, seed: int, val_fraction: float = 0.2,
    method: str = "random", stratified_bins: int = 3,
    min_samples: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create a reproducible fit/validation split without touching the test set."""
    if val_fraction <= 0 or len(x_data) < int(min_samples):
        return x_data, y_data, None, None
    x_array = np.asarray(x_data)
    y_array = np.asarray(y_data, dtype=float).reshape(-1)
    if len(x_array) != len(y_array):
        raise ValueError("Validation inputs and targets must contain the same number of samples.")
    val_count = max(1, int(round(len(x_array) * float(val_fraction))))
    if val_count >= len(x_array):
        return x_array, y_array, None, None

    split_method = str(method).strip().lower()
    if split_method == "stratified":
        try:
            bins = pd.qcut(y_array, q=max(2, min(int(stratified_bins), len(y_array) // 2)),
                labels=False, duplicates="drop")
            bins = np.asarray(bins, dtype=int)
            class_count = int(np.unique(bins).size)
            if class_count < 2 or val_count < class_count or len(x_array) - val_count < class_count:
                raise ValueError("Insufficient samples for stratified validation.")
            fit_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=val_count,
                random_state=int(seed)).split(x_array, bins))
            return x_array[fit_idx], y_array[fit_idx], x_array[val_idx], y_array[val_idx]
        except (TypeError, ValueError):
            split_method = "random"
    if split_method != "random":
        raise ValueError(f"Unsupported validation split method {method!r}. Use 'random' or 'stratified'.")

    indices = np.random.default_rng(int(seed)).permutation(len(x_array))
    val_idx, fit_idx = indices[:val_count], indices[val_count:]
    return x_array[fit_idx], y_array[fit_idx], x_array[val_idx], y_array[val_idx]

def split_validation_indices(y_data: np.ndarray, seed: int, val_fraction: float = 0.2,
    split_fn=None) -> tuple[np.ndarray, np.ndarray]:
    """Return fit/validation indices using an existing array-split policy."""
    y_array = np.asarray(y_data, dtype=float).reshape(-1)
    sample_indices = np.arange(len(y_array), dtype=int).reshape(-1, 1)
    splitter = split_validation_arrays if split_fn is None else split_fn
    fit_rows, _, validation_rows, _ = splitter(sample_indices, y_array, seed=int(seed),
        val_fraction=float(val_fraction))
    fit_indices = np.asarray(fit_rows, dtype=int).reshape(-1)
    validation_indices = (np.asarray(validation_rows, dtype=int).reshape(-1)
        if validation_rows is not None else np.array([], dtype=int))
    return fit_indices, validation_indices

def build_eol_mlp_scheduler(
    optimizer: torch.optim.Optimizer,
    hyperparameters: dict[str, object] | None = None,
):
    config = build_shared_eol_mlp_hyperparameters(hyperparameters)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(config["scheduler_patience"]),
        factor=float(config["scheduler_factor"]),
    )

def build_shared_eol_mlp_hyperparameters(
    overrides: dict[str, object] | None = None,
    *,
    base_config: dict[str, object] | None = None,
) -> dict[str, object]:
    return normalize_eol_mlp_hyperparameters(overrides, base_config=base_config)

class TinyTemporalCNN(nn.Module):
    def __init__(
        self,
        input_dims: int,
        timesteps: int,
        dropout_rate: float = 0.1,
        conv1_channels: int = 32,
        conv_channels: int = 32,
        fc_hidden_dim: int = 16,
        fc_hidden_dim2: int | None = None,
        conv1_kernel_size: int = 7,
        conv2_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        conv1_kernel_size = max(1, int(conv1_kernel_size))
        conv2_kernel_size = max(1, int(conv2_kernel_size))
        conv1_channels = max(2, int(conv1_channels))
        conv_channels = max(2, int(conv_channels))
        fc_hidden_dim = max(1, int(fc_hidden_dim))
        fc_hidden_dim2 = None if fc_hidden_dim2 is None else max(1, int(fc_hidden_dim2))
        self.conv1 = nn.Conv1d(
            in_channels=int(input_dims),
            out_channels=conv1_channels,
            kernel_size=conv1_kernel_size,
            padding=conv1_kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv1_channels,
            out_channels=conv_channels,
            kernel_size=conv2_kernel_size,
            padding=conv2_kernel_size // 2,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(conv_channels, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim2) if fc_hidden_dim2 is not None else None
        self.fc_out = nn.Linear(fc_hidden_dim2 or fc_hidden_dim, 1)
        self.dropout = nn.Dropout(float(dropout_rate))
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        self.bn2 = nn.BatchNorm1d(conv_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        if self.fc2 is not None:
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
        return apply_stabilized_regression_head(self.fc_out(x))


class EolPredictionTemporalCNN(TinyTemporalCNN):
    """Canonical augmented-data EOL TCNN with a one-dimensional output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).squeeze(-1)

class ExtrapolationResidualTemporalCNN(nn.Module):
    def __init__(
        self,
        input_dims: int,
        timesteps: int,
        conv1_channels: int | None = None,
        conv_channels: int = 24,
        fc_hidden_dim: int = 12,
        dropout_rate: float = 0.05,
    ) -> None:
        super().__init__()
        conv1_channels = int(conv_channels if conv1_channels is None else conv1_channels)
        self.input_norm = nn.LayerNorm(input_dims)
        self.linear_skip = nn.Linear(timesteps * input_dims, 1)
        self.conv1 = nn.Conv1d(input_dims, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv1_channels, int(conv_channels), kernel_size=5, padding=4, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(int(conv_channels), int(fc_hidden_dim))
        self.fc2 = nn.Linear(int(fc_hidden_dim), 1)
        self.dropout = nn.Dropout(float(dropout_rate))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(inputs)
        linear_component = self.linear_skip(normalized.reshape(normalized.shape[0], -1))
        conv_inputs = normalized.permute(0, 2, 1)
        conv_features = F.silu(self.conv1(conv_inputs))
        conv_features = self.dropout(conv_features)
        conv_features = F.silu(self.conv2(conv_features))
        conv_features = self.pool(conv_features).squeeze(-1)
        conv_features = self.dropout(conv_features)
        conv_features = F.silu(self.fc1(conv_features))
        conv_features = self.dropout(conv_features)
        return apply_stabilized_regression_head(linear_component + self.fc2(conv_features))

def build_regression_loss(loss_name: str, *, reduction: str = "mean") -> nn.Module:
    normalized = str(loss_name).strip().lower()
    if normalized == "mse":
        return nn.MSELoss(reduction=reduction)
    if normalized in {"smooth_l1", "smoothl1", "smooth-l1", "huber"}:
        return nn.SmoothL1Loss(beta=0.5, reduction=reduction)
    raise ValueError(f"Unsupported loss function {loss_name!r}. Use 'mse' or 'smooth_l1'.")

def build_input_scaler(scaler_name: str | None = None):
    """Build the TCNN input scaler named by the global config (or an explicit override)."""
    normalized = str(DEFAULT_TCNN_INPUT_SCALER if scaler_name is None else scaler_name).strip().lower()
    if normalized == "standard_scaler":
        return StandardScaler()
    if normalized == "minmax_scaler":
        return MinMaxScaler()
    raise ValueError(
        f"Unsupported input_scaler {scaler_name!r}. Expected one of: {sorted(VALID_TCNN_INPUT_SCALERS)}."
    )

def make_neural_input_scaler():
    """Canonical TCNN/MLP input scaler from the live global model policy."""
    return build_input_scaler(aug_cfg.MODEL_INPUT_TRANSFORM_MODE)

def build_tcnn_seed_list(
    seed_count: int | None = None,
    seed_source: int | None = None,
    seed_upper_bound: int | None = None,
) -> list[int]:
    return build_model_training_seed_list(
        DEFAULT_TCNN_SEED_COUNT if seed_count is None else int(seed_count),
        DEFAULT_TCNN_SEED_SOURCE if seed_source is None else int(seed_source),
        seed_upper_bound=(
            DEFAULT_TCNN_SEED_UPPER_BOUND
            if seed_upper_bound is None
            else int(seed_upper_bound)
        ),
    )

def normalize_tcnn_hyperparameters(
    hyperparameters: dict[str, object] | None = None,
    *,
    base_config: dict[str, object] | None = None,
) -> dict[str, object]:
    config = deepcopy(DEFAULT_TCNN_BASE_CONFIG if base_config is None else base_config)
    if hyperparameters:
        config.update(hyperparameters)

    aug_cfg.tcnn_input_clipping_enabled()
    aug_cfg.tcnn_output_clipping_enabled()
    config["input_clipping_mode"] = str(aug_cfg.TCNN_INPUT_CLIPPING_MODE)
    config["input_extrapolation_margin"] = aug_cfg.tcnn_input_clipping_fraction()
    config["output_clipping_mode"] = str(aug_cfg.TCNN_OUTPUT_CLIPPING_MODE)
    config["target_log_extrapolation_margin"] = aug_cfg.tcnn_output_clipping_fraction()

    config["input_steps"] = max(2, int(config["input_steps"]))
    config["epochs"] = max(1, int(config["epochs"]))
    config["learning_rate"] = float(config["learning_rate"])
    config["weight_decay"] = float(config["weight_decay"])
    config["dropout_rate"] = float(config["dropout_rate"])
    config["early_stopping_patience"] = max(1, int(config["early_stopping_patience"]))
    config["scheduler_patience"] = max(1, int(config["scheduler_patience"]))
    config["scheduler_factor"] = float(config.get("scheduler_factor", 0.8))
    config["batch_size"] = max(1, int(config["batch_size"]))
    config["validation_split"] = float(config["validation_split"])
    config["validation_interval"] = max(1, int(config["validation_interval"]))
    config["input_extrapolation_margin"] = max(0.0, float(config["input_extrapolation_margin"]))
    config["target_log_extrapolation_margin"] = max(0.0, float(config["target_log_extrapolation_margin"]))
    conv1_default = int(config.get("conv_channels", 32))
    config["conv1_channels"] = max(2, int(config.get("conv1_channels", conv1_default)))
    config["conv_channels"] = max(2, int(config["conv_channels"]))
    config["fc_hidden_dim"] = max(1, int(config["fc_hidden_dim"]))
    fc_hidden_dim2 = config.get("fc_hidden_dim2", config["fc_hidden_dim"])
    config["fc_hidden_dim2"] = None if fc_hidden_dim2 in {None, "", 0, "0"} else max(1, int(fc_hidden_dim2))
    config["conv1_kernel_size"] = max(1, int(config.get("conv1_kernel_size", 7)))
    config["conv2_kernel_size"] = max(1, int(config.get("conv2_kernel_size", 7)))
    config["model_structure"] = str(config["model_structure"]).strip().lower()
    config["loss_function"] = str(config["loss_function"]).strip().lower()
    config["source_balance_mode"] = str(
        config.get("source_balance_mode", DEFAULT_TCNN_SOURCE_BALANCE_MODE)
    ).strip().lower()
    config["input_transform_mode"] = str(
        config.get("input_transform_mode", DEFAULT_TCNN_INPUT_TRANSFORM_MODE)
    ).strip().lower()
    config["target_transform_mode"] = str(
        config.get("target_transform_mode", DEFAULT_TCNN_TARGET_TRANSFORM_MODE)
    ).strip().lower()
    config["input_scaler"] = str(config.get("input_scaler", DEFAULT_TCNN_INPUT_SCALER)).strip().lower()
    if config["input_scaler"] not in VALID_TCNN_INPUT_SCALERS:
        raise ValueError(
            f"Unsupported input_scaler {config['input_scaler']!r}. "
            f"Expected one of: {sorted(VALID_TCNN_INPUT_SCALERS)}."
        )
    config["seed_count"] = max(1, int(config.get("seed_count", DEFAULT_TCNN_SEED_COUNT)))
    config["seed_source"] = int(config.get("seed_source", DEFAULT_TCNN_SEED_SOURCE))
    config["seed_upper_bound"] = max(
        config["seed_count"] + 1,
        int(config.get("seed_upper_bound", DEFAULT_TCNN_SEED_UPPER_BOUND)),
    )
    config["seeds"] = [
        int(seed)
        for seed in config.get(
            "seeds",
            build_tcnn_seed_list(
                config["seed_count"],
                config["seed_source"],
                config["seed_upper_bound"],
            ),
        )
    ]
    if not config["seeds"]:
        raise ValueError("TCNN hyperparameters require at least one seed.")
    config["ensemble_size"] = len(config["seeds"])
    return config

def build_shared_tcnn_hyperparameters(
    overrides: dict[str, object] | None = None,
    *,
    base_config: dict[str, object] | None = None,
) -> dict[str, object]:
    return normalize_tcnn_hyperparameters(overrides, base_config=base_config)

def build_tcnn_model_kwargs(
    hyperparameters: dict[str, object] | None = None,
) -> dict[str, object]:
    config = build_shared_tcnn_hyperparameters(hyperparameters)
    return {
        "dropout_rate": float(config["dropout_rate"]),
        "conv1_channels": int(config["conv1_channels"]),
        "conv_channels": int(config["conv_channels"]),
        "fc_hidden_dim": int(config["fc_hidden_dim"]),
        "fc_hidden_dim2": config["fc_hidden_dim2"],
        "conv1_kernel_size": int(config["conv1_kernel_size"]),
        "conv2_kernel_size": int(config["conv2_kernel_size"]),
    }

def build_tcnn_optimizer(
    model: nn.Module,
    hyperparameters: dict[str, object] | None = None,
) -> torch.optim.Optimizer:
    config = build_shared_tcnn_hyperparameters(hyperparameters)
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

def build_tcnn_scheduler(
    optimizer: torch.optim.Optimizer,
    hyperparameters: dict[str, object] | None = None,
):
    config = build_shared_tcnn_hyperparameters(hyperparameters)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(config["scheduler_patience"]),
        factor=float(config["scheduler_factor"]),
    )

def build_shared_extrapolation_tcnn_hyperparameters(
    overrides: dict[str, object] | None = None,
    *,
    base_config: dict[str, object] | None = None,
    structure_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    merged = deepcopy(DEFAULT_EXTRAPOLATION_TCNN_STRUCTURE)
    if structure_overrides:
        merged.update(structure_overrides)
    if overrides:
        merged.update(overrides)
    return normalize_tcnn_hyperparameters(merged, base_config=base_config)


# --- Shared all-feature linear regressors (ElasticNet, SVR) ------------------------------------
# Li_datarun's implementations are canonical. Datasets keep same-named thin adapters that pass
# their own metrics helpers and metadata; nothing here reads a host namespace. The returned dict
# is the union of the keys the datasets previously produced, so callers that read test_ape_values
# (the MAPE violin panels) or selected_features (feature reporting) keep working.
TRAIN_MASK_COLUMNS = ("is_outer_field_selected", "is_train_cell")


def _resolve_train_mask_column(data: pd.DataFrame, preferred: str | None) -> str:
    candidates = ([preferred] if preferred else []) + list(TRAIN_MASK_COLUMNS)
    for name in candidates:
        if name and name in data.columns:
            return str(name)
    raise RuntimeError(f"No training-mask column found; tried {candidates}.")


def _absolute_percentage_errors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        errors = np.abs((actual - predicted) / actual) * 100.0
    return errors[np.isfinite(errors)]


def default_ape_metrics(actual, predicted) -> tuple[float, float]:
    errors = _absolute_percentage_errors(actual, predicted)
    return (float(np.mean(errors)) if errors.size else np.nan,
            float(np.max(errors)) if errors.size else np.nan)


def default_max_ape_cell(actual, predicted, cells) -> str:
    errors = np.abs((np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
                    / np.asarray(actual, dtype=float)) * 100.0
    finite = np.isfinite(errors)
    if not finite.any():
        return ""
    index = int(np.argmax(np.where(finite, errors, -np.inf)))
    return str(cells[index]) if index < len(cells) else ""


def default_r2(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    if not np.isfinite(ss_tot) or ss_tot <= 1e-30:
        return np.nan
    return float(1.0 - np.sum((actual - predicted) ** 2) / ss_tot)

def calculate_finite_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() < 2:
        return np.nan
    actual = actual[valid]
    predicted = predicted[valid]
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan

class IdentityTargetScaler:
    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)

def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clip_with_fractional_margin(values: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    margin_fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    span = upper - lower
    margin = np.where(span > 0, span, 1.0) * float(max(margin_fraction, 0.0))
    return np.clip(values, lower - margin, upper + margin)


def apply_tcnn_input_clipping(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not aug_cfg.tcnn_input_clipping_enabled():
        return values
    return clip_with_fractional_margin(values, lower, upper, aug_cfg.tcnn_input_clipping_fraction())


def apply_tcnn_output_clipping(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not aug_cfg.tcnn_output_clipping_enabled():
        return values
    return clip_with_fractional_margin(values, lower, upper, aug_cfg.tcnn_output_clipping_fraction())


# EOL MLPs intentionally follow the same global inference policy as TCNNs.
apply_mlp_input_clipping = apply_tcnn_input_clipping
apply_mlp_output_clipping = apply_tcnn_output_clipping

def calculate_rescaled_mape(actual: np.ndarray, predicted_log: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.exp(np.asarray(predicted_log, dtype=float))
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs((actual[valid] - predicted[valid]) / actual[valid])) * 100.0)

def calculate_rescaled_max_ape(actual: np.ndarray, predicted_log: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.exp(np.asarray(predicted_log, dtype=float))
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    if not valid.any():
        return np.nan
    return float(np.max(np.abs((actual[valid] - predicted[valid]) / actual[valid])) * 100.0)

def build_log_feature_subset(data: pd.DataFrame, feature: str, target_column: str,
    train_column: str = "is_outer_field_selected", test_column: str = "is_test_cell") -> pd.DataFrame:
    subset = data[["Cell", train_column, test_column, feature, target_column]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    subset = subset[(subset[feature].abs() > 0) & (subset[target_column] > 0)].copy()
    subset["x_log"] = np.log(np.abs(subset[feature].to_numpy(dtype=float))) if not subset.empty else pd.Series(
        dtype=float)
    subset["y_log"] = np.log(subset[target_column].to_numpy(dtype=float)) if not subset.empty else pd.Series(
        dtype=float)
    return subset

def build_log_feature_fit_summary(data: pd.DataFrame, feature_columns: list[str], target_column: str,
    train_column: str = "is_outer_field_selected", test_column: str = "is_test_cell") -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for feature in feature_columns:
        subset = build_log_feature_subset(data, feature, target_column, train_column, test_column)
        train_subset = subset[subset[train_column]]
        test_subset = subset[subset[test_column]]
        row = {"feature": feature, "n_all": len(subset), "r2_all": np.nan, "pearson_r_all": np.nan,
            "n_train": len(train_subset), "r2_train": np.nan, "pearson_r_train": np.nan,
            "n_test": len(test_subset),
            "test_log_rescaled_mape_all": np.nan, "test_log_rescaled_max_ape_all": np.nan,
            "test_log_rescaled_mape_train": np.nan, "test_log_rescaled_max_ape_train": np.nan}
        if len(subset) >= 2:
            slope, intercept, row["r2_all"], row["pearson_r_all"] = fit_linear_relationship(
                subset["x_log"].to_numpy(dtype=float), subset["y_log"].to_numpy(dtype=float))
            if len(test_subset):
                predicted = slope * test_subset["x_log"].to_numpy(dtype=float) + intercept
                row["test_log_rescaled_mape_all"] = calculate_rescaled_mape(
                    test_subset[target_column].to_numpy(dtype=float), predicted)
                row["test_log_rescaled_max_ape_all"] = calculate_rescaled_max_ape(
                    test_subset[target_column].to_numpy(dtype=float), predicted)
        if len(train_subset) >= 2:
            slope, intercept, row["r2_train"], row["pearson_r_train"] = fit_linear_relationship(
                train_subset["x_log"].to_numpy(dtype=float), train_subset["y_log"].to_numpy(dtype=float))
            if len(test_subset):
                predicted = slope * test_subset["x_log"].to_numpy(dtype=float) + intercept
                row["test_log_rescaled_mape_train"] = calculate_rescaled_mape(
                    test_subset[target_column].to_numpy(dtype=float), predicted)
                row["test_log_rescaled_max_ape_train"] = calculate_rescaled_max_ape(
                    test_subset[target_column].to_numpy(dtype=float), predicted)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(["r2_all", "feature"], ascending=[False, True],
        na_position="last").reset_index(drop=True)

def build_training_feature_subset(data: pd.DataFrame, feature: str, target_column: str) -> pd.DataFrame:
    return build_log_feature_subset(
        data, feature, target_column, train_column="is_train_cell", test_column="is_test_cell")

def build_training_feature_fit_summary(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        subset = build_training_feature_subset(data, feature, target_column)
        train_subset = subset[subset["is_train_cell"]]
        test_subset = subset[subset["is_test_cell"]]
        if len(train_subset) < 2:
            rows.append({"feature": feature, "n_all": len(subset), "n_train": len(train_subset),
                "n_test": len(test_subset), "r2_train": np.nan, "pearson_r_train": np.nan,
                "test_log_rescaled_mape_train": np.nan, "test_log_rescaled_max_ape_train": np.nan})
            continue
        slope, intercept, r2_train, correlation = fit_linear_relationship(
            train_subset["x_log"].to_numpy(dtype=float), train_subset["y_log"].to_numpy(dtype=float))
        if len(test_subset):
            predicted_log = slope * test_subset["x_log"].to_numpy(dtype=float) + intercept
            mape = calculate_rescaled_mape(test_subset[target_column].to_numpy(dtype=float), predicted_log)
            max_ape = calculate_rescaled_max_ape(test_subset[target_column].to_numpy(dtype=float), predicted_log)
        else:
            mape = max_ape = np.nan
        rows.append({"feature": feature, "n_all": len(subset), "n_train": len(train_subset),
            "n_test": len(test_subset), "r2_train": r2_train, "pearson_r_train": correlation,
            "test_log_rescaled_mape_train": mape, "test_log_rescaled_max_ape_train": max_ape})
    return pd.DataFrame(rows).sort_values(
        ["r2_train", "feature"], ascending=[False, True], na_position="last")

def get_required_features(available_features: list[str], required_features: list[str], label: str) -> list[str]:
    missing_features = [feature for feature in required_features if feature not in available_features]
    if missing_features:
        raise RuntimeError(f"Missing required {label} features: {missing_features}")
    return [feature for feature in required_features if feature in available_features]

def select_soh_meanvar_tcnn_features(candidate_features: list[str], *, label: str, feature_num: object,
    four_feature_names: set[str] | frozenset[str]) -> list[str]:
    selected_feature_num = str(feature_num).strip().lower()
    if selected_feature_num not in {"4", "all"}:
        raise RuntimeError(f"{label} TCNN feature_num must be '4' or 'all', got {feature_num!r}.")
    soh_columns = [str(name) for name in candidate_features if str(name).strip().lower() == "soh"]
    meanvar_columns = [str(name) for name in candidate_features
        if str(name).strip().lower().startswith(("mean_", "var_"))
        and not str(name).strip().lower().startswith("log_")]
    if not soh_columns:
        raise RuntimeError(f"{label} TCNN requires an SOH column.")
    if not meanvar_columns:
        raise RuntimeError(f"{label} TCNN requires at least one mean_ or var_ feature column.")
    if selected_feature_num == "4":
        required = {str(name).strip().lower() for name in four_feature_names}
        meanvar_columns = [name for name in meanvar_columns if name.strip().lower() in required]
        missing = sorted(required.difference(name.strip().lower() for name in meanvar_columns))
        if missing:
            raise RuntimeError(f"{label} TCNN feature_num='4' is missing required features: {missing}")
    selected: list[str] = []
    seen: set[str] = set()
    for feature_name in [soh_columns[0], *meanvar_columns]:
        normalized = feature_name.strip().lower()
        if normalized not in seen:
            selected.append(feature_name)
            seen.add(normalized)
    return selected

def choose_best_linear_feature(fit_summary: pd.DataFrame) -> str:
    ranked = fit_summary.sort_values(["r2_train", "pearson_r_train", "feature"], ascending=[False, False, True],
        na_position="last")
    for _, row in ranked.iterrows():
        feature = str(row["feature"]).strip()
        if feature:
            return feature
    raise RuntimeError("Could not determine a usable linear-model feature.")

def get_seed_training_worker_count(seed_count: int, max_workers: int | None = None) -> int:
    configured_workers = SEED_TRAINING_MAX_WORKERS if max_workers is None else int(max_workers)
    return max(0, min(int(configured_workers), int(seed_count)))

def run_seed_training_jobs(worker_fn, payloads: list[dict[str, object]], *,
    max_workers: int | None = None) -> list[dict[str, object]]:
    if not payloads:
        return []
    worker_count = get_seed_training_worker_count(len(payloads), max_workers=max_workers)
    if worker_count <= 1:
        return [worker_fn(payload) for payload in payloads]
    results: list[dict[str, object] | None] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {executor.submit(worker_fn, payload): index for index, payload in enumerate(payloads)}
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
    return [result for result in results if result is not None]

def build_tcnn_training_loader(*tensors: torch.Tensor, batch_size: int, seed: int) -> tuple[DataLoader, int]:
    """Build the canonical shuffled TCNN fit loader used by raw/interpolated trainers."""
    if not tensors or len(tensors[0]) == 0:
        raise ValueError("TCNN training requires at least one fitting sample.")
    sample_count = len(tensors[0])
    effective_batch = min(max(1, int(batch_size)), sample_count)
    if effective_batch > 1 and sample_count % effective_batch == 1:
        effective_batch -= 1
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(TensorDataset(*tensors), batch_size=max(1, effective_batch), shuffle=True,
        drop_last=(sample_count > 1), generator=generator)
    return loader, max(1, effective_batch)

def train_temporal_cnn_seed_worker(payload: dict[str, object]) -> dict[str, object]:
    seed = int(payload["seed"])
    config = dict(payload["config"])
    x_train_scaled = np.asarray(payload["x_train_scaled"], dtype=float)
    x_test_scaled = np.asarray(payload["x_test_scaled"], dtype=float)
    y_train_scaled = np.asarray(payload["y_train_scaled"], dtype=float)
    feature_count = int(payload["feature_count"])
    if "fit_indices" in payload:
        fit_indices = np.asarray(payload["fit_indices"], dtype=int)
        validation_indices = np.asarray(payload.get("validation_indices", []), dtype=int)
        x_fit, y_fit = x_train_scaled[fit_indices], y_train_scaled[fit_indices]
        x_val = x_train_scaled[validation_indices] if len(validation_indices) else None
        y_val = y_train_scaled[validation_indices] if len(validation_indices) else None
    else:
        x_fit, y_fit, x_val, y_val = split_validation_arrays(x_train_scaled, y_train_scaled, seed=seed,
            val_fraction=float(config["validation_split"]))
    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTemporalCNN(input_dims=feature_count, timesteps=int(config["input_steps"]),
        **build_tcnn_model_kwargs(config)).to(device)
    optimizer = build_tcnn_optimizer(model, config)
    scheduler = build_tcnn_scheduler(optimizer, config)
    loss_fn = build_regression_loss(str(config["loss_function"]))
    x_fit_tensor = torch.tensor(x_fit, dtype=torch.float32, device=device)
    y_fit_tensor = torch.tensor(y_fit, dtype=torch.float32, device=device).unsqueeze(1)
    train_loader, effective_batch_size = build_tcnn_training_loader(x_fit_tensor, y_fit_tensor,
        batch_size=int(config["batch_size"]), seed=seed)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device) if x_val is not None else None
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1) if y_val is not None else None
    best_metric = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0
    for epoch in range(int(config["epochs"])):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        if epoch % int(config["validation_interval"]) != 0:
            continue
        model.eval()
        with torch.no_grad():
            metric = float(loss_fn(model(x_val_tensor), y_val_tensor).item()) if x_val_tensor is not None else float(
                loss_fn(model(x_fit_tensor), y_fit_tensor).item())
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
        train_pred_scaled = model(torch.tensor(x_train_scaled, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        test_pred_scaled = model(torch.tensor(x_test_scaled, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    return {"seed": seed, "best_metric": best_metric, "state_dict": best_state,
        "effective_batch_size": effective_batch_size,
        "train_pred_scaled": train_pred_scaled, "test_pred_scaled": test_pred_scaled}

def train_temporal_cnn_regressor(sequence_dataset: dict[str, object], dataset_label: str,
    hyperparameters: dict[str, object] | None = None, *,
    seed_training_max_workers: int | None = None) -> dict[str, object]:
    config = build_shared_tcnn_hyperparameters(hyperparameters)
    sequences = np.asarray(sequence_dataset["sequences"], dtype=float)
    targets = np.asarray(sequence_dataset["targets"], dtype=float)
    cells = list(sequence_dataset["cells"])
    train_mask = np.asarray(sequence_dataset["train_mask"], dtype=bool)
    test_mask = np.asarray(sequence_dataset["test_mask"], dtype=bool)
    feature_columns = list(sequence_dataset["feature_columns"])
    x_train_raw, x_test_raw = sequences[train_mask], sequences[test_mask]
    y_train_raw, y_test_raw = targets[train_mask], targets[test_mask]
    train_cells = [cell for cell, selected in zip(cells, train_mask) if selected]
    test_cells = [cell for cell, selected in zip(cells, test_mask) if selected]
    if len(x_train_raw) < 2:
        raise RuntimeError(f"Not enough selected training sequences for {dataset_label} temporal CNN.")
    if len(x_test_raw) == 0:
        raise RuntimeError(f"No validation sequences available for {dataset_label} temporal CNN.")

    x_train_min = np.min(x_train_raw, axis=(0, 1))
    x_train_max = np.max(x_train_raw, axis=(0, 1))
    x_test_bounded = apply_tcnn_input_clipping(x_test_raw, x_train_min, x_train_max)
    input_transform_mode = str(config["input_transform_mode"]).strip().lower()
    if input_transform_mode in {"standard_scaler_only", "minmax_scaler_only"}:
        x_train_prepared, x_test_prepared = x_train_raw, x_test_bounded
    elif input_transform_mode in {"log_abs_then_standard_scaler", "log_abs_then_minmax_scaler"}:
        x_train_prepared = np.log(np.abs(x_train_raw) + 1e-12)
        x_test_prepared = np.log(np.abs(x_test_bounded) + 1e-12)
    else:
        raise RuntimeError(f"Unsupported input_transform_mode for temporal CNN: {config['input_transform_mode']}.")
    preprocessing_seed = int(config["seeds"][0])
    fit_indices, validation_indices = split_validation_indices(y_train_raw, seed=preprocessing_seed,
        val_fraction=float(config["validation_split"]))
    scaler = make_neural_input_scaler()
    scaler.fit(x_train_prepared[fit_indices].reshape(-1, x_train_prepared.shape[-1]))
    x_train_scaled = scaler.transform(x_train_prepared.reshape(-1, x_train_prepared.shape[-1])).reshape(
        x_train_prepared.shape)
    x_test_scaled = scaler.transform(x_test_prepared.reshape(-1, x_test_prepared.shape[-1])).reshape(
        x_test_prepared.shape)

    target_transformer = make_neural_target_transformer(y_train_raw[fit_indices])
    target_transform_mode = target_transformer.mode
    target_divisor = float(target_transformer.target_divisor_)
    y_train_transformed = target_transformer.transform(y_train_raw)
    train_actual, test_actual = y_train_raw / target_divisor, y_test_raw / target_divisor
    target_scaler = IdentityTargetScaler()
    target_transform_min = float(np.min(y_train_transformed))
    target_transform_max = float(np.max(y_train_transformed))
    y_train_scaled = target_scaler.fit_transform(y_train_transformed.reshape(-1, 1)).reshape(-1)
    payloads = [{"seed": int(seed), "config": config, "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled, "y_train_scaled": y_train_scaled,
        "feature_count": len(feature_columns), "fit_indices": fit_indices,
        "validation_indices": validation_indices} for seed in config["seeds"]]
    train_predictions, test_predictions, model_artifacts = [], [], []
    seed_results = (run_seed_training_jobs(train_temporal_cnn_seed_worker, payloads)
        if seed_training_max_workers is None else run_seed_training_jobs(
            train_temporal_cnn_seed_worker, payloads, max_workers=seed_training_max_workers))
    for seed_result in seed_results:
        train_pred_transformed = target_scaler.inverse_transform(
            np.asarray(seed_result["train_pred_scaled"]).reshape(-1, 1)).reshape(-1)
        test_pred_transformed = target_scaler.inverse_transform(
            np.asarray(seed_result["test_pred_scaled"]).reshape(-1, 1)).reshape(-1)
        train_pred_transformed = apply_tcnn_output_clipping(train_pred_transformed,
            np.array([target_transform_min]), np.array([target_transform_max])).reshape(-1)
        test_pred_transformed = apply_tcnn_output_clipping(test_pred_transformed,
            np.array([target_transform_min]), np.array([target_transform_max])).reshape(-1)
        train_pred_transformed = np.exp(train_pred_transformed)
        test_pred_transformed = np.exp(test_pred_transformed)
        train_predictions.append(train_pred_transformed)
        test_predictions.append(test_pred_transformed)
        model_artifacts.append({"seed": int(seed_result["seed"]), "best_metric": float(seed_result["best_metric"]),
            "effective_batch_size": int(seed_result["effective_batch_size"]),
            "state_dict": seed_result["state_dict"]})
    train_pred = np.mean(np.vstack(train_predictions), axis=0)
    test_pred = np.mean(np.vstack(test_predictions), axis=0)
    return {"dataset_label": dataset_label, "model_name": "tcnn", "feature_count": len(feature_columns),
        "train_count": len(train_cells), "test_count": len(test_cells),
        "train_mape": default_ape_metrics(train_actual, train_pred)[0],
        "test_mape": default_ape_metrics(test_actual, test_pred)[0],
        "train_max_ape": default_ape_metrics(train_actual, train_pred)[1],
        "test_max_ape": default_ape_metrics(test_actual, test_pred)[1],
        "train_r2": default_r2(train_actual, train_pred), "test_r2": default_r2(test_actual, test_pred),
        "epochs": int(config["epochs"]), "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]), "dropout_rate": float(config["dropout_rate"]),
        "input_steps": int(config["input_steps"]), "conv1_channels": int(config["conv1_channels"]),
        "conv_channels": int(config["conv_channels"]), "fc_hidden_dim": int(config["fc_hidden_dim"]),
        "fc_hidden_dim2": int(config["fc_hidden_dim2"]), "batch_size": int(config["batch_size"] or len(x_train_raw)),
        "effective_batch_size": int(model_artifacts[0]["effective_batch_size"]),
        "scheduler_patience": int(config["scheduler_patience"]), "model_structure": str(config["model_structure"]),
        "validation_split": float(config["validation_split"]),
        "validation_split_method": "random", "validation_interval": int(config["validation_interval"]),
        "early_stopping_patience": int(config["early_stopping_patience"]), "seed_count": len(config["seeds"]),
        "preprocessing_fit_scope": "inner_fit_subset", "preprocessing_split_seed": preprocessing_seed,
        "preprocessing_fit_count": len(fit_indices),
        "preprocessing_validation_count": len(validation_indices),
        "seed_training_max_workers": SEED_TRAINING_MAX_WORKERS,
        "seed_training_workers_used": get_seed_training_worker_count(len(config["seeds"]),
            max_workers=seed_training_max_workers),
        "target_divisor": float(target_divisor), "target_transform_mode": target_transform_mode,
        "target_scaler": "identity",
        "input_clipping_mode": str(config["input_clipping_mode"]),
        "input_clipping_percent": float(aug_cfg.TCNN_INPUT_CLIPPING_PERCENT),
        "output_clipping_mode": str(config["output_clipping_mode"]),
        "output_clipping_percent": float(aug_cfg.TCNN_OUTPUT_CLIPPING_PERCENT),
        "input_extrapolation_margin": float(config["input_extrapolation_margin"]),
        "target_log_extrapolation_margin": float(config["target_log_extrapolation_margin"]),
        "feature_columns": feature_columns, "input_transform_mode": input_transform_mode,
        "x_train_min": x_train_min, "x_train_max": x_train_max,
        "target_transform_min": target_transform_min, "target_transform_max": target_transform_max,
        "scaler_mean": np.asarray(getattr(scaler, "mean_", [np.nan])),
        "scaler_scale": np.asarray(getattr(scaler, "scale_", [np.nan])),
        "scaler_var": np.asarray(getattr(scaler, "var_", [np.nan])),
        "scaler_min": np.asarray(getattr(scaler, "min_", [np.nan])),
        "scaler_data_min": np.asarray(getattr(scaler, "data_min_", [np.nan])),
        "scaler_data_max": np.asarray(getattr(scaler, "data_max_", [np.nan])),
        "scaler_n_features_in": int(scaler.n_features_in_),
        "target_scaler_mean": np.asarray(getattr(target_scaler, "mean_", [np.nan])),
        "target_scaler_scale": np.asarray(getattr(target_scaler, "scale_", [np.nan])),
        "target_scaler_var": np.asarray(getattr(target_scaler, "var_", [np.nan])),
        "target_scaler_min": np.asarray(getattr(target_scaler, "min_", [np.nan])),
        "target_scaler_data_min": np.asarray(getattr(target_scaler, "data_min_", [np.nan])),
        "target_scaler_data_max": np.asarray(getattr(target_scaler, "data_max_", [np.nan])),
        "model_artifacts": model_artifacts, "train_actual": train_actual, "train_pred": train_pred,
        "test_actual": test_actual, "test_pred": test_pred, "train_cell_list": train_cells,
        "test_cell_list": test_cells}


REGULARIZED_MODEL_SOH_SOURCE_COLUMN = "_regularized_model_soh"


def _add_regularized_model_soh(data: pd.DataFrame,
    feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Expose the input-reference SOH only to ElasticNet and SVR."""
    model_data = data
    if "SOH" not in model_data.columns and REGULARIZED_MODEL_SOH_SOURCE_COLUMN in model_data.columns:
        model_data = model_data.copy()
        model_data["SOH"] = model_data[REGULARIZED_MODEL_SOH_SOURCE_COLUMN]
    model_features = [str(feature) for feature in feature_columns if str(feature) != "SOH"]
    if "SOH" in model_data.columns:
        model_features = ["SOH", *model_features]
    return model_data, model_features


def prepare_all_feature_dataset(*, data: pd.DataFrame, feature_columns: list[str], target_column: str,
    dataset_label: str, model_label: str, train_mask_column: str | None = None):
    """Shared row/column validation and train-test masking for both regressors."""
    selected_features = [str(feature) for feature in feature_columns if str(feature) in data.columns]
    if not selected_features:
        raise RuntimeError(f"No {model_label} features are available for {dataset_label}.")
    train_column = _resolve_train_mask_column(data, train_mask_column)

    required_columns = ["Cell", train_column, "is_test_cell", target_column, *selected_features]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise RuntimeError(f"{model_label} dataset {dataset_label} is missing columns: {missing_columns}")

    dataset = data[required_columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if dataset.empty:
        raise RuntimeError(f"No finite {model_label} rows are available for {dataset_label}.")

    feature_values = dataset[selected_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    target_values = pd.to_numeric(dataset[target_column], errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(feature_values).all(axis=1) & np.isfinite(target_values) & (target_values > 0.0)
    dataset = dataset.loc[valid_mask].copy()
    feature_values = feature_values[valid_mask]
    target_values = target_values[valid_mask]
    if dataset.empty:
        raise RuntimeError(f"No positive-target {model_label} rows are available for {dataset_label}.")

    train_mask = dataset[train_column].astype(bool).to_numpy()
    test_mask = dataset["is_test_cell"].astype(bool).to_numpy()
    if int(train_mask.sum()) < 2:
        raise RuntimeError(f"Not enough training rows for {dataset_label} {model_label} model.")
    if int(test_mask.sum()) == 0:
        raise RuntimeError(f"No test rows available for {dataset_label} {model_label} model.")
    return dataset, selected_features, feature_values, target_values, train_mask, test_mask


def build_all_feature_result(*, dataset: pd.DataFrame, selected_features: list[str], target_values: np.ndarray,
    train_mask: np.ndarray, test_mask: np.ndarray, train_pred: np.ndarray, test_pred: np.ndarray,
    extra_fields: dict | None = None,
    model_name: str, dataset_label: str, selected_feature: str, target_column: str,
    calculate_ape_metrics=None, calculate_max_ape_cell=None, calculate_r2=None,
    display_name: str | None = None, data_variant: str | None = None, model_variant: str | None = None,
    task_kind: str | None = None, seed_count: int | None = None,
    source_kind: str | None = None) -> dict[str, object]:
    """Metrics and bookkeeping common to every all-feature regressor result row."""
    calculate_ape_metrics = calculate_ape_metrics or default_ape_metrics
    calculate_max_ape_cell = calculate_max_ape_cell or default_max_ape_cell
    calculate_r2 = calculate_r2 or default_r2
    train_actual = target_values[train_mask]
    test_actual = target_values[test_mask]
    train_cells = dataset.loc[train_mask, "Cell"].astype(str).tolist()
    test_cells = dataset.loc[test_mask, "Cell"].astype(str).tolist()
    train_mape, train_max_ape = calculate_ape_metrics(train_actual, train_pred)
    test_mape, test_max_ape = calculate_ape_metrics(test_actual, test_pred)
    return {
        "model_name": str(model_name), "dataset_label": dataset_label,
        "display_name": str(display_name) if display_name is not None else str(model_name),
        "model_variant": model_variant, "target_column": str(target_column),
        "selected_feature": str(selected_feature), "selected_features": json.dumps(selected_features),
        "feature_count": len(selected_features),
        "train_count": int(train_mask.sum()), "test_count": int(test_mask.sum()),
        "train_mape": train_mape, "test_mape": test_mape,
        "train_rmse": float(np.sqrt(np.mean((train_actual - train_pred) ** 2))),
        "test_rmse": float(np.sqrt(np.mean((test_actual - test_pred) ** 2))),
        "train_max_ape": train_max_ape, "test_max_ape": test_max_ape,
        "train_max_ape_cell": calculate_max_ape_cell(train_actual, train_pred, train_cells),
        "test_max_ape_cell": calculate_max_ape_cell(test_actual, test_pred, test_cells),
        "train_r2": calculate_r2(train_actual, train_pred), "test_r2": calculate_r2(test_actual, test_pred),
        "test_ape_values": _absolute_percentage_errors(test_actual, test_pred),
        "epochs": np.nan, "learning_rate": np.nan,
        "train_actual": train_actual, "train_pred": train_pred,
        "test_actual": test_actual, "test_pred": test_pred,
        "train_cell_list": train_cells, "test_cell_list": test_cells,
        **({"test_source_list": [str(source_kind)] * len(test_cells)} if source_kind else {}),
        **{k: v for k, v in (extra_fields or {}).items() if v is not None},
    }


def train_elasticnet_all_feature_regressor(*, data: pd.DataFrame, feature_columns: list[str],
    target_column: str, dataset_label: str, calculate_ape_metrics=None, calculate_max_ape_cell=None, calculate_r2=None,
    model_name: str = "elasticnet", display_name: str | None = None, data_variant: str | None = None,
    model_variant: str | None = None, task_kind: str | None = None, seed_count: int | None = None,
    source_kind: str | None = None, train_mask_column: str | None = None) -> dict[str, object]:
    """Li's log-feature / log-target ElasticNet, shared by every dataset."""
    data, feature_columns = _add_regularized_model_soh(data, feature_columns)
    dataset, selected_features, feature_values, target_values, train_mask, test_mask = \
        prepare_all_feature_dataset(data=data, feature_columns=feature_columns, target_column=target_column,
            dataset_label=dataset_label, model_label="ElasticNet", train_mask_column=train_mask_column)

    x_log = np.log(np.maximum(np.abs(feature_values), 1e-30))
    target_transformer = make_model_target_transformer(target_values[train_mask])
    y_transformed = target_transformer.transform(target_values)
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_log[train_mask])
    x_test = x_scaler.transform(x_log[test_mask])

    model = ElasticNet(alpha=float(aug_cfg.ELASTICNET_ALPHA), l1_ratio=float(aug_cfg.ELASTICNET_L1_RATIO),
        fit_intercept=True, max_iter=int(aug_cfg.ELASTICNET_MAX_ITER),
        random_state=int(aug_cfg.ELASTICNET_RANDOM_STATE))
    model.fit(x_train, y_transformed[train_mask])

    train_pred = target_transformer.inverse_transform(model.predict(x_train))
    test_pred = target_transformer.inverse_transform(model.predict(x_test))
    coefficients = np.asarray(model.coef_, dtype=float)

    result = build_all_feature_result(dataset=dataset, selected_features=selected_features,
        target_values=target_values, train_mask=train_mask, test_mask=test_mask,
        train_pred=train_pred, test_pred=test_pred, model_name=model_name, dataset_label=dataset_label,
        selected_feature="all_features", target_column=target_column,
        calculate_ape_metrics=calculate_ape_metrics, calculate_max_ape_cell=calculate_max_ape_cell,
        calculate_r2=calculate_r2, display_name=display_name, data_variant=data_variant,
        model_variant=model_variant if model_variant is not None else "all_features_log_scaled",
        source_kind=source_kind,
        extra_fields={"data_variant": data_variant, "task_kind": task_kind, "seed_count": seed_count})
    result.update({
        "elasticnet_alpha": float(aug_cfg.ELASTICNET_ALPHA),
        "elasticnet_l1_ratio": float(aug_cfg.ELASTICNET_L1_RATIO),
        "elasticnet_max_iter": int(aug_cfg.ELASTICNET_MAX_ITER),
        "elasticnet_target_transform": target_transformer.mode,
        "elasticnet_target_divisor": float(target_transformer.target_divisor_),
        "elasticnet_nonzero_feature_count": int(np.count_nonzero(np.abs(coefficients) > 1e-12)),
        "elasticnet_feature_coefficients": json.dumps(
            {feature: float(coef) for feature, coef in zip(selected_features, coefficients)}),
    })
    return result


# --- Target transforms shared by every model family ---------------------------------------------
# Moved out of Li_datarun/eol_mlp_20w_IOWA.py so TUM/Vilijmen can use the same target handling.
def normalize_target_transform_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {
        "standard": "standard_scaler",
        "standard_scaler_only": "standard_scaler",
        "minmax": "minmax_scaler",
        "minmax_scaler_only": "minmax_scaler",
        "log": "log_only",
        "log_then_standard_scaler": "log_only",
        "log_only_then_standard_scaler": "log_only",
        "target_divisor_log": "target_divisor_then_log",
        "log_target_divisor": "target_divisor_then_log",
        "target_divisor_log_standard": "target_divisor_then_log_then_standard_scaler",
        "target_divisor_then_log_standard": "target_divisor_then_log_then_standard_scaler",
        "target_divisor_then_log_standard_scaler": "target_divisor_then_log_then_standard_scaler",
    }
    normalized = aliases.get(normalized, normalized)
    valid_modes = {"standard_scaler", "minmax_scaler", "log_only", "target_divisor_then_log",
        "target_divisor_then_log_then_standard_scaler"}
    if normalized not in valid_modes:
        raise ValueError(f"Unsupported TARGET_TRANSFORM_MODE={mode!r}. Expected one of: " "standard_scaler, minmax_scaler, log_only, target_divisor_then_log, " "target_divisor_then_log_then_standard_scaler.")
    return normalized

def compute_symmetric_log_target_divisor(target_values: np.ndarray) -> float:
    target_values = np.asarray(target_values, dtype=float)
    valid = np.isfinite(target_values) & (target_values > 0)
    if not np.any(valid):
        raise RuntimeError("Cannot compute target divisor without positive finite targets.")
    positive_targets = target_values[valid]
    return float(np.sqrt(float(np.min(positive_targets)) * float(np.max(positive_targets))))

class TargetTransformer:
    def __init__(self, mode: str) -> None:
        self.mode = normalize_target_transform_mode(mode)
        self.target_divisor_ = 1.0
        self.scaler = None
        self.data_min_ = np.asarray([np.nan], dtype=float)
        self.data_max_ = np.asarray([np.nan], dtype=float)
        self.mean_ = np.asarray([np.nan], dtype=float)
        self.scale_ = np.asarray([np.nan], dtype=float)

    def _base_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if self.mode in {"log_only", "target_divisor_then_log", "target_divisor_then_log_then_standard_scaler"}:
            if np.any(values <= 0):
                raise RuntimeError("Log target transform requires positive targets.")
            if self.mode == "log_only":
                return np.log(values)
            return np.log(values / float(self.target_divisor_))
        return values

    def fit(self, values: np.ndarray) -> "TargetTransformer":
        values = np.asarray(values, dtype=float).reshape(-1)
        if self.mode in {"target_divisor_then_log", "target_divisor_then_log_then_standard_scaler"}:
            self.target_divisor_ = compute_symmetric_log_target_divisor(values)
        transformed = self._base_transform(values)
        finite = np.isfinite(transformed)
        if not np.any(finite):
            raise RuntimeError("Cannot fit target transformer without finite transformed targets.")
        fit_values = transformed[finite].reshape(-1, 1)
        if self.mode in {"standard_scaler", "log_only", "target_divisor_then_log_then_standard_scaler"}:
            self.scaler = StandardScaler().fit(fit_values)
        elif self.mode == "minmax_scaler":
            self.scaler = MinMaxScaler().fit(fit_values)
        else:
            self.scaler = None

        if self.scaler is None:
            self.data_min_ = np.asarray([float(np.min(fit_values))], dtype=float)
            self.data_max_ = np.asarray([float(np.max(fit_values))], dtype=float)
            self.mean_ = np.asarray([float(np.mean(fit_values))], dtype=float)
            self.scale_ = np.asarray([float(np.std(fit_values))], dtype=float)
        else:
            self.data_min_ = np.asarray(getattr(self.scaler, "data_min_", [float(np.min(fit_values))]), dtype=float)
            self.data_max_ = np.asarray(getattr(self.scaler, "data_max_", [float(np.max(fit_values))]), dtype=float)
            self.mean_ = np.asarray(getattr(self.scaler, "mean_", [np.nan]), dtype=float)
            self.scale_ = np.asarray(getattr(self.scaler, "scale_", [np.nan]), dtype=float)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = self._base_transform(values).reshape(-1, 1)
        if self.scaler is not None:
            transformed = self.scaler.transform(transformed)
        return np.asarray(transformed, dtype=float).reshape(-1)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(values, dtype=float).reshape(-1, 1)
        if self.scaler is not None:
            transformed = self.scaler.inverse_transform(transformed)
        transformed = transformed.reshape(-1)
        if self.mode == "log_only":
            return np.exp(transformed)
        if self.mode in {"target_divisor_then_log", "target_divisor_then_log_then_standard_scaler"}:
            return np.exp(transformed) * float(self.target_divisor_)
        return transformed


def make_model_target_transformer(values):
    """Target transformer for the configured global MODEL_TARGET_TRANSFORM_MODE."""
    return TargetTransformer(aug_cfg.MODEL_TARGET_TRANSFORM_MODE).fit(values)


def make_neural_target_transformer(values) -> TargetTransformer:
    """Canonical TCNN/MLP target transform: divisor, log, and no second scaler."""
    transformer = make_model_target_transformer(values)
    if transformer.mode != "target_divisor_then_log" or transformer.scaler is not None:
        raise RuntimeError(
            "TCNN/MLP targets require MODEL_TARGET_TRANSFORM_MODE="
            "'target_divisor_then_log' with no secondary target scaler."
        )
    return transformer


def set_eol_training_seed(seed: int) -> None:
    """Seed the augmented EOL trainers exactly once, including deterministic cuDNN."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_eol_train_validation_indices(sample_count: int, *, seed: int,
    validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic augmented-EOL split, always leaving fit and validation samples."""
    if sample_count < 2:
        raise RuntimeError("Need at least two samples to build a validation split.")
    validation_count = int(np.ceil(float(sample_count) * float(validation_fraction)))
    validation_count = min(max(1, validation_count), sample_count - 1)
    permutation = np.random.default_rng(int(seed)).permutation(sample_count)
    validation_indices = np.sort(permutation[:validation_count])
    train_indices = np.sort(permutation[validation_count:])
    return train_indices.astype(int), validation_indices.astype(int)


def _fit_augmented_eol_model(*, model: nn.Module, x_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor, train_indices: np.ndarray, validation_indices: np.ndarray,
    optimizer: torch.optim.Optimizer, scheduler, loss_function: str, batch_size: int,
    epochs: int, early_stopping_patience: int, seed: int,
    source_weights: np.ndarray | None = None,
    duplicate_single_fit_sample: bool = False) -> tuple[list[float], list[float]]:
    device = x_train_tensor.device
    fit_indices_tensor = torch.tensor(train_indices, dtype=torch.long, device=device)
    validation_indices_tensor = torch.tensor(validation_indices, dtype=torch.long, device=device)
    x_fit_tensor = x_train_tensor.index_select(0, fit_indices_tensor)
    y_fit_tensor = y_train_tensor.index_select(0, fit_indices_tensor)
    x_validation_tensor = x_train_tensor.index_select(0, validation_indices_tensor)
    y_validation_tensor = y_train_tensor.index_select(0, validation_indices_tensor)

    weights = np.ones(len(y_train_tensor), dtype=np.float32) if source_weights is None else np.asarray(
        source_weights, dtype=np.float32)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    weight_fit_tensor = weight_tensor.index_select(0, fit_indices_tensor)
    weight_validation_tensor = weight_tensor.index_select(0, validation_indices_tensor)
    if duplicate_single_fit_sample and len(x_fit_tensor) == 1:
        x_fit_tensor = torch.cat((x_fit_tensor, x_fit_tensor), dim=0)
        y_fit_tensor = torch.cat((y_fit_tensor, y_fit_tensor), dim=0)
        weight_fit_tensor = torch.cat((weight_fit_tensor, weight_fit_tensor), dim=0)

    loader, _ = build_tcnn_training_loader(x_fit_tensor, y_fit_tensor, weight_fit_tensor,
        batch_size=int(batch_size), seed=int(seed))
    loss_fn = build_regression_loss(str(loss_function), reduction="none")
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for _ in range(int(epochs)):
        model.train()
        batch_losses: list[float] = []
        for x_batch, y_batch, weight_batch in loader:
            optimizer.zero_grad()
            loss = torch.mean(loss_fn(model(x_batch), y_batch) * weight_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            validation_errors = loss_fn(model(x_validation_tensor), y_validation_tensor)
            validation_loss = float((torch.sum(validation_errors * weight_validation_tensor)
                / torch.sum(weight_validation_tensor)).item())
        validation_losses.append(validation_loss)
        scheduler.step(validation_loss)
        if np.isfinite(validation_loss) and validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(early_stopping_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return train_losses, validation_losses


def train_augmented_eol_mlp(x_train: np.ndarray, y_train: np.ndarray, *, seed: int,
    device: torch.device, hyperparameters: dict[str, object] | None = None,
    duplicate_single_fit_sample: bool = False,
) -> tuple[EolPredictionMLP, object, TargetTransformer, list[float], list[float]]:
    """Fit the augmented EOL MLP used by Li, TUM, and Vilijmen."""
    config = build_shared_eol_mlp_hyperparameters(hyperparameters)
    x_values = np.asarray(x_train, dtype=float)
    y_values = np.asarray(y_train, dtype=float)
    set_eol_training_seed(seed)
    train_indices, validation_indices = split_eol_train_validation_indices(len(x_values), seed=seed,
        validation_fraction=float(config["validation_split"]))
    scaler = make_neural_input_scaler().fit(x_values[train_indices])
    scaler.mlp_input_min_ = np.min(x_values, axis=0)
    scaler.mlp_input_max_ = np.max(x_values, axis=0)
    x_train_tensor = torch.tensor(scaler.transform(x_values), dtype=torch.float32, device=device)
    target_transformer = make_neural_target_transformer(y_values[train_indices])
    y_transformed = target_transformer.transform(y_values)
    y_train_tensor = torch.tensor(y_transformed, dtype=torch.float32, device=device)

    model = EolPredictionMLP(x_values.shape[1], list(config["hidden_layers"]),
        float(config["dropout_rate"])).to(device)
    target_min, target_max = float(np.min(y_transformed)), float(np.max(y_transformed))
    model.mlp_target_transform_min_ = target_min
    model.mlp_target_transform_max_ = target_max
    model.mlp_target_log_min_ = target_min
    model.mlp_target_log_max_ = target_max
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]))
    scheduler = build_eol_mlp_scheduler(optimizer, config)
    train_losses, validation_losses = _fit_augmented_eol_model(model=model,
        x_train_tensor=x_train_tensor, y_train_tensor=y_train_tensor, train_indices=train_indices,
        validation_indices=validation_indices, optimizer=optimizer, scheduler=scheduler,
        loss_function="mse", batch_size=int(config["batch_size"]), epochs=int(config["epochs"]),
        early_stopping_patience=int(config["early_stopping_patience"]), seed=seed,
        duplicate_single_fit_sample=duplicate_single_fit_sample)
    return model, scaler, target_transformer, train_losses, validation_losses


def train_augmented_eol_tcnn(x_train_seq: np.ndarray, y_train: np.ndarray, *, seed: int,
    device: torch.device, hyperparameters: dict[str, object] | None = None,
    real_train_count: int | None = None,
) -> tuple[EolPredictionTemporalCNN, object, TargetTransformer, list[float], list[float]]:
    """Fit the augmented EOL TCNN used by Li, TUM, and Vilijmen."""
    config = build_shared_tcnn_hyperparameters(hyperparameters)
    x_values = np.asarray(x_train_seq, dtype=float)
    y_values = np.asarray(y_train, dtype=float)
    sequence_shape = x_values.shape
    real_count = min(len(y_values), max(1, int(real_train_count or len(y_values))))
    set_eol_training_seed(seed)
    train_indices, validation_indices = split_eol_train_validation_indices(len(x_values), seed=seed,
        validation_fraction=float(config["validation_split"]))
    scaler = make_neural_input_scaler().fit(x_values[train_indices].reshape(-1, sequence_shape[-1]))
    real_train_flat = x_values[:real_count].reshape(-1, sequence_shape[-1])
    scaler.tcnn_input_min_ = np.min(real_train_flat, axis=0)
    scaler.tcnn_input_max_ = np.max(real_train_flat, axis=0)
    x_scaled = scaler.transform(x_values.reshape(-1, sequence_shape[-1])).reshape(sequence_shape)
    x_train_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    target_transformer = make_neural_target_transformer(y_values[train_indices])
    y_transformed = target_transformer.transform(y_values)
    y_train_tensor = torch.tensor(y_transformed, dtype=torch.float32, device=device)

    model = EolPredictionTemporalCNN(input_dims=sequence_shape[-1], timesteps=sequence_shape[1],
        **build_tcnn_model_kwargs(config)).to(device)
    target_min = float(np.min(y_transformed[:real_count]))
    target_max = float(np.max(y_transformed[:real_count]))
    model.tcnn_target_transform_min_ = target_min
    model.tcnn_target_transform_max_ = target_max
    model.tcnn_target_log_min_ = target_min
    model.tcnn_target_log_max_ = target_max
    source_weights = np.ones(len(y_values), dtype=np.float32)
    augmented_count = len(y_values) - real_count
    if str(config["source_balance_mode"]) == "equal_real_aug_total_loss" and real_count and augmented_count:
        source_weights[:real_count] = len(y_values) / (2.0 * float(real_count))
        source_weights[real_count:] = len(y_values) / (2.0 * float(augmented_count))
    optimizer = build_tcnn_optimizer(model, config)
    scheduler = build_tcnn_scheduler(optimizer, config)
    train_losses, validation_losses = _fit_augmented_eol_model(model=model,
        x_train_tensor=x_train_tensor, y_train_tensor=y_train_tensor, train_indices=train_indices,
        validation_indices=validation_indices, optimizer=optimizer, scheduler=scheduler,
        loss_function=str(config["loss_function"]), batch_size=int(config["batch_size"]),
        epochs=int(config["epochs"]), early_stopping_patience=int(config["early_stopping_patience"]),
        seed=seed, source_weights=source_weights)
    return model, scaler, target_transformer, train_losses, validation_losses


def train_svr_all_feature_regressor(*, data: pd.DataFrame, feature_columns: list[str],
    target_column: str, dataset_label: str, calculate_ape_metrics=None, calculate_max_ape_cell=None,
    calculate_r2=None, model_name: str = "svr", display_name: str | None = None,
    data_variant: str | None = None, model_variant: str | None = None, task_kind: str | None = None,
    seed_count: int | None = None, source_kind: str | None = None,
    train_mask_column: str | None = None) -> dict[str, object]:
    """Li's SVR: raw features through a StandardScaler, target through the global transform,
    predictions clipped to the training range. Shared by every dataset."""
    data, feature_columns = _add_regularized_model_soh(data, feature_columns)
    dataset, selected_features, feature_values, target_values, train_mask, test_mask =         prepare_all_feature_dataset(data=data, feature_columns=feature_columns, target_column=target_column,
            dataset_label=dataset_label, model_label="SVR", train_mask_column=train_mask_column)

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(feature_values[train_mask])
    x_test = x_scaler.transform(feature_values[test_mask])
    y_train = target_values[train_mask]
    y_scaler = make_model_target_transformer(y_train)

    model = SVR(kernel=str(aug_cfg.SVR_KERNEL), C=float(aug_cfg.SVR_C),
        epsilon=float(aug_cfg.SVR_EPSILON), gamma=aug_cfg.SVR_GAMMA)
    model.fit(x_train, y_scaler.transform(y_train))

    train_pred = np.maximum(
        y_scaler.inverse_transform(model.predict(x_train).reshape(-1, 1)).reshape(-1), 1e-9)
    test_pred = np.maximum(
        y_scaler.inverse_transform(model.predict(x_test).reshape(-1, 1)).reshape(-1), 1e-9)

    result = build_all_feature_result(dataset=dataset, selected_features=selected_features,
        target_values=target_values, train_mask=train_mask, test_mask=test_mask,
        train_pred=train_pred, test_pred=test_pred, model_name=model_name, dataset_label=dataset_label,
        selected_feature="model_features", target_column=target_column,
        calculate_ape_metrics=calculate_ape_metrics, calculate_max_ape_cell=calculate_max_ape_cell,
        calculate_r2=calculate_r2, display_name=display_name, model_variant=model_variant,
        source_kind=source_kind,
        extra_fields={"data_variant": data_variant, "task_kind": task_kind, "seed_count": seed_count})
    result.update({
        "svr_kernel": str(aug_cfg.SVR_KERNEL), "svr_c": float(aug_cfg.SVR_C),
        "svr_epsilon": float(aug_cfg.SVR_EPSILON), "svr_gamma": str(aug_cfg.SVR_GAMMA),
        "svr_input_transform": "raw_standard_scaler",
        "svr_target_transform": normalize_target_transform_mode(aug_cfg.MODEL_TARGET_TRANSFORM_MODE),
        "svr_target_divisor": float(getattr(y_scaler, "target_divisor_", 1.0)),
        "svr_target_scaler_min": float(y_scaler.data_min_[0]),
        "svr_target_scaler_max": float(y_scaler.data_max_[0]),
        "svr_support_vector_count": int(len(model.support_)),
    })
    return result


# --- Single-feature log-log linear regressor ----------------------------------------------------
# Every dataset fit the same model: slope/intercept on (log feature, log target), inverted with
# exp(). Only the result bookkeeping differed, so this returns the union of their keys.

def fit_linear_relationship(x_log: np.ndarray, y_log: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares line through log-log points; returns slope, intercept, R2 and correlation."""
    slope, intercept = np.polyfit(x_log, y_log, deg=1)
    y_pred = slope * x_log + intercept
    ss_res = np.sum((y_log - y_pred) ** 2)
    ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    corr = float(np.corrcoef(x_log, y_log)[0, 1]) if len(x_log) > 1 else np.nan
    return float(slope), float(intercept), float(r2), corr


def train_linear_feature_regressor(*, data: pd.DataFrame, fit_summary: pd.DataFrame, target_column: str,
    dataset_label: str, choose_best_feature_fn, build_feature_subset_fn, feature_name: str | None = None,
    train_column: str | None = None, test_column: str = "is_test_cell", epochs: object = np.nan,
    input_week_resolver=None, raw_feature_week_default: float | None = None,
    raw_feature_week_post_g20: float | None = None, linear_input_week_variant: str | None = None,
    calculate_ape_metrics=None, calculate_max_ape_cell=None, calculate_r2=None, model_name: str = "linear",
    display_name: str | None = None, data_variant: str | None = None, model_variant: str | None = None,
    task_kind: str | None = None, seed_count: int | None = None, source_kind: str | None = None,
    extra_fields: dict | None = None) -> dict[str, object]:
    """Fit log(target) ~ log(feature) on the training rows and score the held-out rows."""
    calculate_ape_metrics = calculate_ape_metrics or default_ape_metrics
    calculate_max_ape_cell = calculate_max_ape_cell or default_max_ape_cell
    calculate_r2 = calculate_r2 or default_r2

    if feature_name:
        best_feature = str(feature_name)
    else:
        try:                     # Li passes data/target_column too; TUM and Vilijmen take fit_summary alone
            best_feature = choose_best_feature_fn(fit_summary, data=data, target_column=target_column)
        except TypeError:
            best_feature = choose_best_feature_fn(fit_summary)
    dataset = build_feature_subset_fn(data, best_feature, target_column)
    # Datasets disagree on the training flag: Li uses is_outer_field_selected, others is_train_cell.
    train_flag = train_column or next(
        (c for c in ("is_train_cell", "is_outer_field_selected") if c in dataset.columns), None)
    if train_flag is None or test_column not in dataset.columns:
        raise RuntimeError(f"{dataset_label} linear subset lacks a train/test flag column.")
    train_subset = dataset[dataset[train_flag]].copy()
    test_subset = dataset[dataset[test_column]].copy()
    if len(train_subset) < 2:
        raise RuntimeError(f"Not enough training rows for {dataset_label} linear model.")
    if len(test_subset) == 0:
        raise RuntimeError(f"No test rows available for {dataset_label} linear model.")

    train_actual = train_subset[target_column].to_numpy(dtype=float)
    test_actual = test_subset[target_column].to_numpy(dtype=float)
    target_transformer = make_model_target_transformer(train_actual)
    train_x = train_subset["x_log"].to_numpy(dtype=float)
    test_x = test_subset["x_log"].to_numpy(dtype=float)
    train_y = target_transformer.transform(train_actual)
    slope, intercept, fit_r2, fit_corr = fit_linear_relationship(train_x, train_y)
    train_pred = target_transformer.inverse_transform(slope * train_x + intercept)
    test_pred = target_transformer.inverse_transform(slope * test_x + intercept)
    train_cells = train_subset["Cell"].astype(str).tolist()
    test_cells = test_subset["Cell"].astype(str).tolist()
    train_mape, train_max_ape = calculate_ape_metrics(train_actual, train_pred)
    test_mape, test_max_ape = calculate_ape_metrics(test_actual, test_pred)

    result = {
        "model_name": str(model_name), "dataset_label": dataset_label,
        "display_name": str(display_name) if display_name is not None else str(model_name),
        "selected_feature": best_feature, "feature_count": 1,
        "slope": float(slope), "intercept": float(intercept),
        "linear_target_transform": target_transformer.mode,
        "linear_target_divisor": float(target_transformer.target_divisor_),
        "fit_r2": float(fit_r2), "fit_corr": float(fit_corr),
        "train_count": len(train_subset), "test_count": len(test_subset),
        "train_mape": train_mape, "test_mape": test_mape,
        "train_max_ape": train_max_ape, "test_max_ape": test_max_ape,
        "train_max_ape_cell": calculate_max_ape_cell(train_actual, train_pred, train_cells),
        "test_max_ape_cell": calculate_max_ape_cell(test_actual, test_pred, test_cells),
        "train_rmse": float(np.sqrt(np.mean((train_actual - train_pred) ** 2))),
        "test_rmse": float(np.sqrt(np.mean((test_actual - test_pred) ** 2))),
        "train_r2": calculate_r2(train_actual, train_pred),
        "test_r2": calculate_r2(test_actual, test_pred),
        "test_ape_values": _absolute_percentage_errors(test_actual, test_pred),
        "epochs": epochs, "learning_rate": np.nan,
        "train_actual": train_actual, "train_pred": train_pred,
        "test_actual": test_actual, "test_pred": test_pred,
        "train_cell_list": train_cells, "test_cell_list": test_cells,
    }
    if source_kind:
        result["test_source_list"] = [str(source_kind)] * len(test_cells)
    # Optional input-week bookkeeping: only the non-augmented linear pipeline supplies a resolver,
    # so these keys appear exactly where they did before and nowhere else.
    if input_week_resolver is not None:
        result["train_input_last_step_week"] = np.asarray(
            [input_week_resolver(cell) for cell in train_cells], dtype=float)
        result["test_input_last_step_week"] = np.asarray(
            [input_week_resolver(cell) for cell in test_cells], dtype=float)
    for key, value in {"data_variant": data_variant, "model_variant": model_variant,
                       "task_kind": task_kind, "seed_count": seed_count,
                       "target_column": target_column if model_variant is not None else None,
                       "raw_feature_week_default": raw_feature_week_default,
                       "raw_feature_week_post_g20": raw_feature_week_post_g20,
                       "linear_input_week_variant": linear_input_week_variant,
                       **(extra_fields or {})}.items():
        if value is not None:
            result[key] = value
    return result
