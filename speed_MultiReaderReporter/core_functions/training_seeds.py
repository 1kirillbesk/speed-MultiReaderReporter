from __future__ import annotations

import numpy as np


# Change this value to control the default number of training seeds used by
# datarun model-training scripts.
MODEL_TRAINING_SEED_COUNT = 5
MODEL_TRAINING_SEED_SOURCE = 42


def get_model_training_seed_count(value: int | None = None) -> int:
    if value is None:
        return max(1, int(MODEL_TRAINING_SEED_COUNT))
    return max(1, int(value))


def build_model_training_seed_list(
    seed_count: int | None = None,
    seed_source: int | None = None,
    *,
    seed_upper_bound: int = 100000,
) -> list[int]:
    effective_count = get_model_training_seed_count(seed_count)
    effective_source = int(MODEL_TRAINING_SEED_SOURCE if seed_source is None else seed_source)
    upper_bound = max(int(seed_upper_bound), int(effective_count) + 1)
    rng = np.random.default_rng(effective_source)
    return [
        int(seed)
        for seed in rng.choice(range(0, int(upper_bound)), size=int(effective_count), replace=False)
    ]
