from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


EXPORT_CORE_FUNCTIONS_DIR = (
    Path(__file__).resolve().parent.parent / "augmentation_functions_export" / "core_functions"
)


def _load_exported_module(module_name: str) -> ModuleType:
    module_path = EXPORT_CORE_FUNCTIONS_DIR / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing exported module: {module_path}")
    qualified_name = f"speed_multi_reader_reporter_exported.{module_name}"
    cached = sys.modules.get(qualified_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load exported module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def get_exported_augmentation_methods_module() -> ModuleType:
    return _load_exported_module("augmentation_methods")


@lru_cache(maxsize=1)
def get_exported_temporal_models_module() -> ModuleType:
    return _load_exported_module("temporal_models")
