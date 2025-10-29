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
