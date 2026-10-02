from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetColumns:
    week: str
    capacity: str
    throughput: str
    realtime: str | None = None


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    dataset_family: str
    legacy_script: str
    interpolation_axis: str
    target_soh: float
    uses_autoanchor: bool = False
    uses_augmentation: bool = False
    uses_tcnn: bool = True
    uses_linear: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def legacy_path(self) -> Path:
        return Path(self.legacy_script)


@dataclass(frozen=True)
class ThroughputAutoanchorRunConfig:
    raw_input_dir: Path
    output_dir_name_template: str
    feature_row_index: int
    thresholds: tuple[float, ...]
    min_abs_pearson: float
    show_plots: bool = True
    autoanchor_candidates: tuple[float, ...] = ()
    interpolated_tcnn_meanvar_filter_mode: str = "off"
    interpolated_tcnn_cell_availability_mode: str = "all_threshold_cells"
    tcnn_input_transform_mode: str = "standard_scaler_only"
    tcnn_target_transform_mode: str = "log_only"
    tcnn_validation_split_method: str = "random"
    tcnn_stratified_validation_bins: int = 3
    linear_fixed_throughput_input_targets: tuple[float, ...] = ()
    linear_target_key_prefix: str = "tp"
    linear_target_key_mintest: str = "mintest"
    linear_target_key_mintest_midfeature: str = "mintest_midfeature"

    def build_output_dir(self, *, script_dir: Path, feature_selection_label: str) -> Path:
        return script_dir / self.output_dir_name_template.format(
            feature_selection_label=str(feature_selection_label),
        )
