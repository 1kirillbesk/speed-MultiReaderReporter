from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


DEFAULT_CELL_IDS = (1, 2, 3, 4, 5, 6, 10, 11, 12)
DEFAULT_EOH_SOH = 0.95
RESISTANCE_COLUMN = "R_10s_ohm"
CAPACITY_COLUMN = "cap_ocv_dis"
TIME_COLUMN = "CU_time"
THROUGHPUT_COLUMN = "throughput_cum"


def default_feature_dir() -> Path:
    here = Path(__file__).resolve()
    speed_root = here.parents[2]
    return speed_root / "out_lw" / "cell_feature"


def default_output_dir() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    return repo_root / "generated_plots"


def default_plotfield_dir(feature_dir: Path) -> Path:
    return feature_dir.parent / "plots" / "reference_checkup_overview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build three separate plots: "
            "all-cell initial capacity vs resistance distribution, "
            "SOH trajectories for SPEED_LW_reference cells, and "
            "all-cell lifetime distribution at the chosen SOH threshold."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=default_feature_dir(),
        help="Directory containing per-cell feature CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where the SVG files will be saved.",
    )
    parser.add_argument(
        "--cell-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_CELL_IDS),
        help="Reference cell ids for the trajectory plot, e.g. --cell-ids 1 2 3 4.",
    )
    parser.add_argument(
        "--eol-soh",
        type=float,
        default=DEFAULT_EOH_SOH,
        help="SOH threshold used to define end of life.",
    )
    parser.add_argument(
        "--skip-plotfield-copy",
        action="store_true",
        help="Do not also save into the standard out_lw/plots area.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure with matplotlib.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write any SVG files.",
    )
    return parser.parse_args()


def parse_listlike(value: object) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=float).ravel()
        return arr[np.isfinite(arr)]
    if pd.isna(value):
        return np.array([], dtype=float)

    text = str(value).strip()
    if not text:
        return np.array([], dtype=float)

    try:
        parsed = literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            return np.array([float(text)], dtype=float)
        except ValueError:
            return np.array([], dtype=float)

    if isinstance(parsed, (list, tuple, np.ndarray)):
        arr = np.asarray(parsed, dtype=float).ravel()
        return arr[np.isfinite(arr)]

    try:
        return np.array([float(parsed)], dtype=float)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def reduce_resistance(value: object) -> float:
    arr = parse_listlike(value)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def build_cell_path(feature_dir: Path, cell_id: int) -> Path:
    return feature_dir / f"SPEED_LW_reference_{cell_id}.csv"


def list_feature_paths(feature_dir: Path) -> list[Path]:
    return sorted(feature_dir.glob("*.csv"))


def infer_numeric_id(stem: str) -> float:
    tail = stem.rsplit("_", 1)[-1]
    try:
        return float(int(tail))
    except ValueError:
        return float("nan")


def iqr_inlier_mask(values: pd.Series, factor: float = 3.0) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return vals.notna()
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return vals.between(lower, upper, inclusive="both")


def exclude_big_initial_outliers(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = summary_df.copy()
    mask = (
        iqr_inlier_mask(base["initial_capacity_Ah"], factor=3.0)
        & iqr_inlier_mask(base["initial_resistance_10s_mohm"], factor=3.0)
    )
    filtered = base.loc[mask].reset_index(drop=True)
    excluded = base.loc[~mask].reset_index(drop=True)
    return filtered, excluded


def interpolate_threshold_crossing(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> float | None:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return None
    if y[0] <= threshold:
        return float(x[0])

    crossing_idx = np.flatnonzero(y <= threshold)
    if crossing_idx.size == 0:
        return None

    idx = int(crossing_idx[0])
    if idx == 0:
        return float(x[0])

    x0, x1 = float(x[idx - 1]), float(x[idx])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    if not np.isfinite([x0, x1, y0, y1]).all():
        return None
    if y1 == y0:
        return float(x1)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def load_cell_frame(path: Path, eol_soh: float) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(path).copy()
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    df[CAPACITY_COLUMN] = pd.to_numeric(df[CAPACITY_COLUMN], errors="coerce")
    df[THROUGHPUT_COLUMN] = pd.to_numeric(df[THROUGHPUT_COLUMN], errors="coerce")

    base_capacity = float(df[CAPACITY_COLUMN].dropna().iloc[0])
    df["soh"] = df[CAPACITY_COLUMN] / base_capacity
    df["throughput_Ah"] = df[THROUGHPUT_COLUMN] / 3600.0
    df["efc"] = df["throughput_Ah"] / (2.0 * base_capacity)
    df["days"] = (df[TIME_COLUMN] - df[TIME_COLUMN].iloc[0]).dt.total_seconds() / 86400.0
    df["resistance_10s_ohm_median"] = df[RESISTANCE_COLUMN].apply(reduce_resistance)
    df["resistance_10s_mohm_median"] = 1000.0 * df["resistance_10s_ohm_median"]

    efc_95 = interpolate_threshold_crossing(df["efc"].to_numpy(), df["soh"].to_numpy(), eol_soh)
    days_95 = interpolate_threshold_crossing(df["days"].to_numpy(), df["soh"].to_numpy(), eol_soh)
    reached = efc_95 is not None

    summary = {
        "cell_name": path.stem,
        "cell_id": infer_numeric_id(path.stem),
        "n_checkups": int(len(df)),
        "initial_capacity_Ah": float(df[CAPACITY_COLUMN].iloc[0]),
        "initial_resistance_10s_mohm": float(df["resistance_10s_mohm_median"].iloc[0]),
        "last_soh": float(df["soh"].dropna().iloc[-1]),
        "last_efc": float(df["efc"].dropna().iloc[-1]),
        "lifetime_efc_at_soh": float(efc_95) if reached else np.nan,
        "lifetime_days_at_soh": float(days_95) if days_95 is not None else np.nan,
        "eol_reached": bool(reached),
        "observed_efc_if_censored": np.nan if reached else float(df["efc"].dropna().iloc[-1]),
    }
    return df, summary


def kde_or_hist_1d(ax: plt.Axes, values: np.ndarray, *, vertical: bool = False, color: str = "#2a7ab9") -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    if values.size >= 3 and np.ptp(values) > 0:
        kde = gaussian_kde(values)
        grid = np.linspace(values.min(), values.max(), 200)
        density = kde(grid)
        if vertical:
            ax.fill_betweenx(grid, 0.0, density, color=color, alpha=0.18)
            ax.plot(density, grid, color=color, linewidth=1.2)
        else:
            ax.fill_between(grid, 0.0, density, color=color, alpha=0.18)
            ax.plot(grid, density, color=color, linewidth=1.2)
    else:
        ax.hist(
            values,
            bins=min(8, max(3, values.size)),
            orientation="horizontal" if vertical else "vertical",
            color=color,
            alpha=0.25,
            edgecolor=color,
        )


def maybe_add_joint_kde(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = "#2a7ab9") -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return

    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        return

    x_pad = 0.1 * (x.max() - x.min())
    y_pad = 0.1 * (y.max() - y.min())
    xx, yy = np.meshgrid(
        np.linspace(x.min() - x_pad, x.max() + x_pad, 120),
        np.linspace(y.min() - y_pad, y.max() + y_pad, 120),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    levels = np.quantile(zz, [0.55, 0.72, 0.86, 0.94])
    levels = np.unique(levels[np.isfinite(levels)])
    if levels.size:
        ax.contour(xx, yy, zz, levels=levels, colors=color, linewidths=1.0, alpha=0.7)


def plot_initial_distribution(
    ax_joint: plt.Axes,
    ax_top: plt.Axes,
    ax_right: plt.Axes,
    summary_df: pd.DataFrame,
) -> None:
    x = summary_df["initial_resistance_10s_mohm"].to_numpy(dtype=float)
    y = summary_df["initial_capacity_Ah"].to_numpy(dtype=float)

    maybe_add_joint_kde(ax_joint, x, y)
    ax_joint.scatter(x, y, s=20, color="#2a7ab9", edgecolor="white", linewidth=0.4, alpha=0.65, zorder=3)

    kde_or_hist_1d(ax_top, x)
    kde_or_hist_1d(ax_right, y, vertical=True)

    ax_joint.grid(True, alpha=0.2)

    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    for axis in (ax_top, ax_right):
        for spine in axis.spines.values():
            spine.set_visible(False)


def plot_trajectories(
    ax: plt.Axes,
    cell_frames: dict[int, pd.DataFrame],
    eol_soh: float,
) -> None:
    colors = plt.cm.tab20(np.linspace(0.05, 0.95, max(len(cell_frames), 1)))

    for color, (cell_id, frame) in zip(colors, sorted(cell_frames.items())):
        ax.plot(
            frame["efc"],
            frame["soh"],
            linewidth=1.8,
            color=color,
            alpha=0.95,
        )

    ax.axhline(eol_soh, color="#4d4d4d", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(0.94, 1.005)
    ax.grid(True, alpha=0.2)


def plot_lifetime_bars(ax: plt.Axes, summary_df: pd.DataFrame, eol_soh: float) -> None:
    reached = pd.to_numeric(summary_df["lifetime_efc_at_soh"], errors="coerce").to_numpy(dtype=float)
    reached = reached[np.isfinite(reached)]

    combined = reached if reached.size else np.array([0.0])
    bins = np.histogram_bin_edges(combined, bins="auto")
    if np.unique(bins).size < 2:
        bins = np.linspace(combined.min(), combined.max() + 1.0, 5)

    if reached.size:
        ax.hist(
            reached,
            bins=bins,
            color="#2a7ab9",
            alpha=0.75,
            edgecolor="#1f4668",
        )

    ax.grid(True, axis="y", alpha=0.2)


def strip_plot_text(ax: plt.Axes, *, keep_ticklabels: bool = False) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if ax.legend_ is not None:
        ax.legend_.remove()
    if not keep_ticklabels:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def hide_axes_frame(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def style_marginal_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_initial_distribution_figure(summary_df: pd.DataFrame) -> plt.Figure:
    plot_df, excluded_df = exclude_big_initial_outliers(summary_df)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(8.6, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.22], height_ratios=[0.22, 1.0], wspace=0.05, hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_joint = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    plot_initial_distribution(ax_joint, ax_top, ax_right, plot_df)
    strip_plot_text(ax_joint, keep_ticklabels=False)
    hide_axes_frame(ax_joint)
    style_marginal_axis(ax_top)
    style_marginal_axis(ax_right)
    return fig


def build_trajectory_figure(cell_frames: dict[int, pd.DataFrame], eol_soh: float) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    plot_trajectories(ax, cell_frames, eol_soh)
    strip_plot_text(ax, keep_ticklabels=False)
    hide_axes_frame(ax)
    return fig


def build_lifetime_distribution_figure(summary_df: pd.DataFrame, eol_soh: float) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    plot_lifetime_bars(ax, summary_df, eol_soh)
    strip_plot_text(ax, keep_ticklabels=False)
    hide_axes_frame(ax)
    return fig


def filter_valid_feature_paths(paths: Iterable[Path]) -> list[Path]:
    valid: list[Path] = []
    for path in paths:
        try:
            frame = pd.read_csv(path, nrows=3)
        except Exception:
            continue
        needed = {CAPACITY_COLUMN, THROUGHPUT_COLUMN, TIME_COLUMN, RESISTANCE_COLUMN}
        if needed.issubset(frame.columns):
            valid.append(path)
    return valid


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()

    feature_dir = args.feature_dir.resolve()
    output_dir = args.output_dir.resolve()
    plotfield_dir = default_plotfield_dir(feature_dir).resolve()
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save and not args.skip_plotfield_copy:
        plotfield_dir.mkdir(parents=True, exist_ok=True)

    ref_paths = [build_cell_path(feature_dir, cell_id) for cell_id in args.cell_ids]
    ensure_paths_exist(ref_paths)
    all_paths = filter_valid_feature_paths(list_feature_paths(feature_dir))

    all_summaries: list[dict[str, object]] = []
    for path in all_paths:
        try:
            _, summary = load_cell_frame(path, eol_soh=args.eol_soh)
        except Exception:
            continue
        all_summaries.append(summary)

    ref_frames: dict[int, pd.DataFrame] = {}
    ref_summaries: list[dict[str, object]] = []
    for path in ref_paths:
        frame, summary = load_cell_frame(path, eol_soh=args.eol_soh)
        ref_frames[int(summary["cell_id"])] = frame
        ref_summaries.append(summary)

    all_summary_df = pd.DataFrame(all_summaries).sort_values(["cell_name"]).reset_index(drop=True)
    ref_summary_df = pd.DataFrame(ref_summaries).sort_values(["cell_id"]).reset_index(drop=True)

    _, initial_excluded_df = exclude_big_initial_outliers(all_summary_df)
    if not initial_excluded_df.empty:
        print("Excluded from initial capacity-vs-resistance plot:")
        print(initial_excluded_df[["cell_name", "initial_capacity_Ah", "initial_resistance_10s_mohm"]].to_string(index=False))

    fig_initial = build_initial_distribution_figure(all_summary_df)
    fig_traj = build_trajectory_figure(ref_frames, args.eol_soh)
    fig_lifetime = build_lifetime_distribution_figure(all_summary_df, args.eol_soh)

    ref_stub = "_".join(str(v) for v in args.cell_ids)
    save_items = [
        ("all_cells_initial_capacity_vs_resistance.svg", fig_initial),
        (f"lw_reference_{ref_stub}_soh_trajectories.svg", fig_traj),
        ("all_cells_lifetime_distribution.svg", fig_lifetime),
    ]

    if not args.no_save:
        for filename, fig in save_items:
            out_path = output_dir / filename
            fig.savefig(out_path, format="svg")
            print(f"Saved svg: {out_path}")
            if not args.skip_plotfield_copy:
                plot_path = plotfield_dir / filename
                fig.savefig(plot_path, format="svg")
                print(f"Saved plotfield svg: {plot_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig_initial)
        plt.close(fig_traj)
        plt.close(fig_lifetime)


if __name__ == "__main__":
    main()
