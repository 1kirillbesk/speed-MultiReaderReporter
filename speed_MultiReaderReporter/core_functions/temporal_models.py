from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core_functions.training_seeds import (
    MODEL_TRAINING_SEED_COUNT,
    MODEL_TRAINING_SEED_SOURCE,
    build_model_training_seed_list,
)

DEFAULT_TCNN_SEED_SOURCE = MODEL_TRAINING_SEED_SOURCE
DEFAULT_TCNN_SEED_COUNT = MODEL_TRAINING_SEED_COUNT
DEFAULT_TCNN_LOSS_FUNCTION = "mse"
DEFAULT_TCNN_BASE_CONFIG: dict[str, object] = {
    "input_steps": 12,
    "epochs": 500,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "dropout_rate": 0.1,
    "seed_count": DEFAULT_TCNN_SEED_COUNT,
    "seed_source": DEFAULT_TCNN_SEED_SOURCE,
    "early_stopping_patience": 100,
    "scheduler_patience": 50,
    "batch_size": 16,
    "validation_split": 0.2,
    "validation_interval": 1,
    "input_extrapolation_margin": 0.05,
    "target_log_extrapolation_margin": 0.10,
    "conv1_channels": 16,
    "conv_channels": 16,
    "fc_hidden_dim": 16,
    "fc_hidden_dim2": 8,
    "conv1_kernel_size": 7,
    "conv2_kernel_size": 7,
    "model_structure": "average1",
    "loss_function": DEFAULT_TCNN_LOSS_FUNCTION,
}
DEFAULT_EXTRAPOLATION_TCNN_STRUCTURE: dict[str, object] = {
    "conv_channels": 24,
    "fc_hidden_dim": 12,
    "fc_hidden_dim2": None,
    "model_structure": "extrapolation_residual",
}


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
        return self.fc_out(x)


class ExtrapolationResidualTemporalCNN(nn.Module):
    def __init__(
        self,
        input_dims: int,
        timesteps: int,
        conv_channels: int = 24,
        fc_hidden_dim: int = 12,
        dropout_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dims)
        self.linear_skip = nn.Linear(timesteps * input_dims, 1)
        self.conv1 = nn.Conv1d(input_dims, int(conv_channels), kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(int(conv_channels), int(conv_channels), kernel_size=5, padding=4, dilation=2)
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
        return linear_component + self.fc2(conv_features)


def build_regression_loss(loss_name: str) -> nn.Module:
    normalized = str(loss_name).strip().lower()
    if normalized == "mse":
        return nn.MSELoss()
    if normalized in {"smooth_l1", "smoothl1", "smooth-l1", "huber"}:
        return nn.SmoothL1Loss(beta=0.5)
    raise ValueError(f"Unsupported loss function {loss_name!r}. Use 'mse' or 'smooth_l1'.")


def build_tcnn_seed_list(seed_count: int | None = None, seed_source: int | None = None) -> list[int]:
    return build_model_training_seed_list(
        DEFAULT_TCNN_SEED_COUNT if seed_count is None else int(seed_count),
        DEFAULT_TCNN_SEED_SOURCE if seed_source is None else int(seed_source),
    )


def normalize_tcnn_hyperparameters(
    hyperparameters: dict[str, object] | None = None,
    *,
    base_config: dict[str, object] | None = None,
) -> dict[str, object]:
    config = deepcopy(DEFAULT_TCNN_BASE_CONFIG if base_config is None else base_config)
    if hyperparameters:
        config.update(hyperparameters)

    config["input_steps"] = max(2, int(config["input_steps"]))
    config["epochs"] = max(1, int(config["epochs"]))
    config["learning_rate"] = float(config["learning_rate"])
    config["weight_decay"] = float(config["weight_decay"])
    config["dropout_rate"] = float(config["dropout_rate"])
    config["early_stopping_patience"] = max(1, int(config["early_stopping_patience"]))
    config["scheduler_patience"] = max(1, int(config["scheduler_patience"]))
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
    config["seed_count"] = max(1, int(config.get("seed_count", DEFAULT_TCNN_SEED_COUNT)))
    config["seed_source"] = int(config.get("seed_source", DEFAULT_TCNN_SEED_SOURCE))
    config["seeds"] = [
        int(seed)
        for seed in config.get("seeds", build_tcnn_seed_list(config["seed_count"], config["seed_source"]))
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
