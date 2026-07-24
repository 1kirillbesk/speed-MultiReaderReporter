# speed_MultiReaderReporter/core/plotting.py
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .capacity import *

def save_group_plot(cell: str, series_list: list, group_dir: Path, title_suffix: str, legend_ncol: int = 4):
    if not series_list:
        return
    group_dir.mkdir(parents=True, exist_ok=True)

    _save_time_series_plot(
        cell=cell,
        title_suffix=title_suffix,
        series_list=series_list,
        group_dir=group_dir,
        legend_ncol=legend_ncol,
        y_column="current_A",
        y_label="Strom [A]",
        title_metric="Strom vs Zeit",
        file_name="strom_vs_zeit.png",
        log_label="current plot",
    )

    _save_time_series_plot(
        cell=cell,
        title_suffix=title_suffix,
        series_list=series_list,
        group_dir=group_dir,
        legend_ncol=legend_ncol,
        y_column="voltage_V",
        y_label="Spannung [V]",
        title_metric="Spannung vs Zeit",
        file_name="spannung_vs_zeit.png",
        log_label="voltage plot",
    )


def _save_time_series_plot(
    *,
    cell: str,
    title_suffix: str,
    series_list: list,
    group_dir: Path,
    legend_ncol: int,
    y_column: str,
    y_label: str,
    title_metric: str,
    file_name: str,
    log_label: str,
):
    prepared: list[tuple[pd.Series, pd.Series, str]] = []
    column_present = False
    for df, label in series_list:
        if df.empty or "abs_time" not in df.columns:
            continue
        if y_column not in df.columns:
            continue
        column_present = True
        values = pd.to_numeric(df[y_column], errors="coerce")
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=df.index)
        mask = values.notna()
        if not mask.any():
            continue
        prepared.append((df.loc[mask, "abs_time"], values.loc[mask], label))

    if not prepared:
        if not column_present:
            print(f"[INFO] {cell} [{title_suffix}]: column '{y_column}' missing; skipping {log_label}.")
        else:
            print(f"[INFO] {cell} [{title_suffix}]: column '{y_column}' contains no numeric data; skipping {log_label}.")
        return

    plt.figure(figsize=(11, 6))
    for x, y, label in prepared:
        plt.plot(x.values, y.values, label=label)
    plt.xlabel("Zeit (absolute)")
    plt.ylabel(y_label)
    plt.title(f"Cell: {cell} — {title_metric} ({title_suffix})")
    plt.grid(True, alpha=0.3)
    plt.legend(
        fontsize=8,
        ncol=legend_ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    out_path = group_dir / file_name
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] {cell} [{title_suffix}]: {len(prepared)} series → {out_path}")

def _sanitize(name: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return s[:120] if len(s) > 120 else s

def _thin_xy(x, y, max_points: int):
    """Light decimator: keep at most max_points evenly spaced points."""
    import numpy as np
    n = len(x)
    if n <= max_points or max_points <= 0:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]

def save_grouped_checkup_plot(cell: str,
                              run_label: str,
                              segments: list[tuple],   # [(df_seg, group_label), ...]
                              out_dir: Path,
                              max_points_per_segment: int = 5000,
                              occurrence_index: int | None = None,
                              occurrence_total: int | None = None):
    """
    Visual check: plot a single checkup broken into groups (segments).
    Plots current vs absolute time; one color per segment with labels.
    Uses light decimation for speed when segments are huge.
    """
    if not segments:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    # Voltage trace is required for this debug plot (time vs voltage)
    has_voltage = any("voltage_V" in df_seg.columns for df_seg, _ in segments)
    if not has_voltage:
        print(f"[SKIP] {cell} — {run_label}: no voltage column for grouped plot.")
        return

    non_empty = [(df_seg, glabel) for df_seg, glabel in segments if not df_seg.empty]
    if len(non_empty) < 2:
        print(f"[SKIP] {cell} — {run_label}: less than two segments with data.")
        return

    plt.figure(figsize=(11, 4))
    for df_seg, glabel in non_empty:
        x = df_seg["abs_time"].values
        y = df_seg["voltage_V"].values
        x, y = _thin_xy(x, y, max_points_per_segment)
        plt.plot(x, y, label=glabel)

    plt.xlabel("Zeit (absolute)")
    plt.ylabel("Spannung [V]")
    display_label = run_label
    if occurrence_index is not None:
        if occurrence_total is not None and occurrence_total > 1:
            display_label = f"{run_label} #{occurrence_index}/{occurrence_total}"
        else:
            display_label = f"{run_label} #{occurrence_index}"
    plt.title(f"Cell: {cell} — Checkup grouped (Spannung): {display_label}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.18), frameon=False)
    plt.tight_layout(rect=[0, 0.20, 1, 1])

    base_name = _sanitize(run_label) or "checkup"
    if occurrence_index is not None and (occurrence_total or 0) > 1:
        suffix = f"_{occurrence_index:02d}of{occurrence_total}"
    elif occurrence_index is not None:
        suffix = f"_{occurrence_index:02d}"
    else:
        suffix = ""
    fname = f"{base_name}{suffix}_grouped_voltage.png"
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] grouped voltage plot → {out_path}")


def plot_curves(
    df,
    cfg,
    cell,
    x_col="V",
    y_col="dQdV",
    show=False,
    kmax=5,              # how many maxima markers per row
    kmin=5,              # how many minima markers per row
    peak_distance=None,  # passed to find_peaks
):
    """
    Plot curves (one per row) with color blue->red.

    Peak markers are computed FROM THE CURVE using your get_peaks/get_minima:
      - maxima: top kmax peaks by y
      - minima: top kmin minima by y (lowest values)
    Works for both dQdV and dVdQ, because it just uses x_col/y_col arrays.

    Saves to:
      cfg["output"]["root"]/plots/<y_col>/<cell>.png
    """
    highm = cfg['voltage']['highm']
    lowm= cfg['voltage']['lowm']
    low= cfg['voltage']['low']
    lowl= cfg['voltage']['lowl']
    n = len(df)
    cmap = cm.get_cmap("coolwarm")  # blue -> red

    output_dir = cfg["output"]["root"]
    plot_dir = os.path.join(output_dir, "plots", y_col)
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (_, row) in enumerate(df.iterrows()):
        x = np.asarray(row.get(x_col, []))
        y = np.asarray(row.get(y_col, []))

        # ensure 1D
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        # skip invalid curve rows
        if x.size == 0 or y.size == 0 or x.size != y.size:
            continue

        color = cmap(i / max(n - 1, 1))
        ax.plot(x, y, color=color, alpha=0.9)

        # ---- compute maxima/minima from curve ----
        peaks = get_peaks(x, y, distance=peak_distance)
        mins  = get_minima(x, y, distance=peak_distance)

        # choose only top-k points
        px_max, py_max = topk_by_y(peaks["x_peaks"], peaks["y_peaks"], kmax, largest=True)
        px_min, py_min = topk_by_y(mins["x_peaks"],  mins["y_peaks"],  kmin, largest=False)

        # plot markers
        if px_max.size > 0:
            ax.scatter(px_max, py_max, color=color, s=30, marker="x", alpha=0.95, linewidths=1.2)
        if px_min.size > 0:
            ax.scatter(px_min, py_min, color=color, s=30, marker="o", alpha=0.75, linewidths=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} curves (blue = first, red = last)")
    ax.grid(alpha=0.3)

    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([0, n - 1])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Row index")

    ref_lines = {
        "highm": highm,
        "lowm": lowm,
        "low": low,
        "lowl": lowl,
    }

    for name, val in ref_lines.items():
        if val is None:
            continue
        ax.axvline(
            x=val,
            linestyle="--",
            linewidth=1,
            color="k",
            alpha=0.6,
            zorder=0,
        )

    ax.text(
        0.01, 0.99,
        f"Peaks per row: max={kmax}, min={kmin} (distance={peak_distance})",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    fig.tight_layout()

    file_path = os.path.join(plot_dir, f"{cell}.png")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {file_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
