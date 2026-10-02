from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from core_functions.training_seeds import build_model_training_seed_list


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
    "validation_split": 0.2,
    "early_stopping_patience": 20,
}


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
