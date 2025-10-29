# speed_MultiReaderReporter/core/plotting.py
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def save_group_plot(cell: str, series_list: list, group_dir: Path, title_suffix: str, legend_ncol: int = 4):
    if not series_list:
        return
    group_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    for df, label in series_list:
        if df.empty: continue
        plt.plot(df["abs_time"].values, df["current_A"].values, label=label)
    plt.xlabel("Zeit (absolute)")
    plt.ylabel("Strom [A]")
    plt.title(f"Cell: {cell} — Strom vs Zeit ({title_suffix})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=legend_ncol, loc="upper center",
               bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    out_path = group_dir / "strom_vs_zeit.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] {cell} [{title_suffix}]: {len(series_list)} file(s) → {out_path}")

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
                              max_points_per_segment: int = 5000):
    """
    Visual check: plot a single checkup broken into groups (segments).
    Plots current vs absolute time; one color per segment with labels.
    Uses light decimation for speed when segments are huge.
    """
    if not segments:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 4))
    for df_seg, glabel in segments:
        if df_seg.empty:
            continue
        x = df_seg["abs_time"].values
        y = df_seg["current_A"].values
        x, y = _thin_xy(x, y, max_points_per_segment)
        plt.plot(x, y, label=glabel)

    plt.xlabel("Zeit (absolute)")
    plt.ylabel("Strom [A]")
    plt.title(f"Cell: {cell} — Checkup grouped: {run_label}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.18), frameon=False)
    plt.tight_layout(rect=[0, 0.20, 1, 1])

    fname = f"{_sanitize(run_label)}_grouped.png"
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] grouped plot → {out_path}")

