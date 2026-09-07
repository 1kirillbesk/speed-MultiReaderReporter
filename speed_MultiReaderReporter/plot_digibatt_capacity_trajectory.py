from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("E:/digibatt/out_digi")
TIME_COLUMN = "CU_time"
THROUGHPUT_COLUMN = "throughput_cum"
DEFAULT_CAPACITY_COLUMNS = ("cap_dis", "cap_cha", "cap_ocv_dis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read DigiBatt cell_feature CSV outputs and plot capacity "
            "trajectories versus checkup time and cumulative throughput."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "cell_feature",
        help="Directory containing per-cell feature CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "plots" / "capacity_trajectory",
        help="Directory where plots and the combined CSV will be saved.",
    )
    parser.add_argument(
        "--capacity-columns",
        nargs="+",
        default=list(DEFAULT_CAPACITY_COLUMNS),
        help="Capacity columns to plot from the feature CSVs.",
    )
    parser.add_argument(
        "--per-cell",
        action="store_true",
        help="Also save one two-panel plot per cell.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving.",
    )
    return parser.parse_args()


def load_feature_outputs(feature_dir: Path, capacity_columns: list[str]) -> pd.DataFrame:
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(feature_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty or TIME_COLUMN not in df.columns:
            continue

        keep_cols = [
            col
            for col in [TIME_COLUMN, THROUGHPUT_COLUMN, *capacity_columns]
            if col in df.columns
        ]
        if len(keep_cols) <= 1:
            continue

        cell_df = df[keep_cols].copy()
        cell_df.insert(0, "cell", csv_path.stem)
        frames.append(cell_df)

    if not frames:
        raise ValueError(f"No usable feature CSVs found in {feature_dir}")

    data = pd.concat(frames, ignore_index=True)
    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], errors="coerce")
    data = data.dropna(subset=[TIME_COLUMN])

    for col in [THROUGHPUT_COLUMN, *capacity_columns]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.sort_values(["cell", TIME_COLUMN]).reset_index(drop=True)
    first_time_by_cell = data.groupby("cell")[TIME_COLUMN].transform("min")
    data["elapsed_days"] = (data[TIME_COLUMN] - first_time_by_cell).dt.total_seconds() / 86400.0
    return data


def available_capacity_columns(data: pd.DataFrame, requested: list[str]) -> list[str]:
    return [col for col in requested if col in data.columns and data[col].notna().any()]


def _plot_all_cells(data: pd.DataFrame,
                    x_col: str,
                    x_label: str,
                    capacity_columns: list[str],
                    output_path: Path,
                    title: str) -> None:
    fig, axes = plt.subplots(
        len(capacity_columns),
        1,
        figsize=(11, max(4, 3.2 * len(capacity_columns))),
        sharex=True,
    )
    if len(capacity_columns) == 1:
        axes = [axes]

    for ax, cap_col in zip(axes, capacity_columns):
        for cell, cell_df in data.groupby("cell", sort=True):
            plot_df = cell_df[[x_col, cap_col]].dropna()
            if plot_df.empty:
                continue
            ax.plot(plot_df[x_col], plot_df[cap_col], marker="o", markersize=2.5, linewidth=1.0, alpha=0.65)
        ax.set_ylabel(f"{cap_col} [Ah]")
        ax.grid(True, alpha=0.3)

    axes[0].set_title(title)
    axes[-1].set_xlabel(x_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {output_path}")


def _plot_per_cell(data: pd.DataFrame,
                   capacity_columns: list[str],
                   output_dir: Path,
                   show: bool) -> None:
    per_cell_dir = output_dir / "per_cell"
    per_cell_dir.mkdir(parents=True, exist_ok=True)

    for cell, cell_df in data.groupby("cell", sort=True):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for cap_col in capacity_columns:
            plot_df = cell_df[["elapsed_days", THROUGHPUT_COLUMN, cap_col]].dropna(subset=[cap_col])
            if plot_df.empty:
                continue
            axes[0].plot(plot_df["elapsed_days"], plot_df[cap_col], marker="o", linewidth=1.2, label=cap_col)
            if THROUGHPUT_COLUMN in plot_df.columns and plot_df[THROUGHPUT_COLUMN].notna().any():
                axes[1].plot(plot_df[THROUGHPUT_COLUMN], plot_df[cap_col], marker="o", linewidth=1.2, label=cap_col)

        axes[0].set_title(f"{cell} capacity vs time")
        axes[0].set_xlabel("Elapsed days from first checkup")
        axes[0].set_ylabel("Capacity [Ah]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)

        axes[1].set_title(f"{cell} capacity vs throughput")
        axes[1].set_xlabel("Cumulative throughput [Ah]")
        axes[1].set_ylabel("Capacity [Ah]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        out_path = per_cell_dir / f"{cell}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"[OK] wrote per-cell plots to {per_cell_dir}")


def main() -> None:
    args = parse_args()
    data = load_feature_outputs(args.feature_dir, args.capacity_columns)
    cap_cols = available_capacity_columns(data, args.capacity_columns)
    if not cap_cols:
        raise ValueError(f"No requested capacity columns found: {args.capacity_columns}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = args.output_dir / "capacity_trajectory_data.csv"
    data[["cell", TIME_COLUMN, "elapsed_days", THROUGHPUT_COLUMN, *cap_cols]].to_csv(combined_csv, index=False)
    print(f"[OK] wrote {combined_csv}")

    _plot_all_cells(
        data=data,
        x_col="elapsed_days",
        x_label="Elapsed days from first checkup",
        capacity_columns=cap_cols,
        output_path=args.output_dir / "capacity_vs_time.png",
        title="DigiBatt capacity trajectory vs time",
    )

    if THROUGHPUT_COLUMN in data.columns and data[THROUGHPUT_COLUMN].notna().any():
        _plot_all_cells(
            data=data,
            x_col=THROUGHPUT_COLUMN,
            x_label="Cumulative throughput [Ah]",
            capacity_columns=cap_cols,
            output_path=args.output_dir / "capacity_vs_throughput.png",
            title="DigiBatt capacity trajectory vs throughput",
        )
        if data[THROUGHPUT_COLUMN].fillna(0).abs().max() == 0:
            print("[WARN] throughput_cum is present but all values are zero; throughput plot will collapse at x=0.")
    else:
        print(f"[WARN] {THROUGHPUT_COLUMN} is missing or empty; skipped throughput plot.")

    if args.per_cell:
        _plot_per_cell(data, cap_cols, args.output_dir, args.show)
    elif args.show:
        plt.show()


if __name__ == "__main__":
    main()
