# speed_MultiReaderReporter/plot_temp_soc_interaction.py
"""Fade rate per unit throughput: temperature x SOC window, on the matched cells.

Only the SOC-matched JGNE cells are used, i.e. the ones whose conditions are
identical apart from temperature:

    30 -> 70 %    25 C: cycle_77, cycle_78     40 C: cycle_95, cycle_96
    40 -> 100 %   25 C: cycle_75, cycle_76     40 C: cycle_97, cycle_98

The 25 C cells are the _ref programmes: same SOC window, no pause, no pulse, no
lowSOC. The 40 C cells are the _40T programmes. Nothing else differs.

Fade is normalised by CHARGE THROUGHPUT rather than time, because the 40 C cells
ran 25.6 weeks against 11-14 weeks for the 25 C ones - comparing endpoint SoH
directly would read that duration difference as a temperature effect.

    fade [%/100Ah] = (1 - SoH_last) / throughput_Ah_last * 100 * 100

Usage:
    python speed_MultiReaderReporter/plot_temp_soc_interaction.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path("E:/download/jgne/out_jgne")
DEFAULT_SUBFOLDER = "capacity_plots"

MATCHED = {
    "30 -> 70 %":  {25: ["SPEED_JGNE_cycle_77", "SPEED_JGNE_cycle_78"],
                    40: ["SPEED_JGNE_cycle_95", "SPEED_JGNE_cycle_96"]},
    "40 -> 100 %": {25: ["SPEED_JGNE_cycle_75", "SPEED_JGNE_cycle_76"],
                    40: ["SPEED_JGNE_cycle_97", "SPEED_JGNE_cycle_98"]},
}
COLOUR = {25: "#3b6ea5", 40: "#c1443c"}

def cell_metrics(out_dir: Path, cell: str) -> dict | None:
    f = out_dir / "cell_feature" / f"{cell}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, usecols=lambda c: c in ("CU_time", "cap_ocv_dis", "throughput_cum"))
    t = pd.to_datetime(d["CU_time"], errors="coerce")
    d = d.assign(t=t).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    soh = d["cap_ocv_dis"] / d["cap_ocv_dis"].iloc[0]
    thr = d["throughput_cum"] / 3600.0          # column is in ampere-seconds
    if thr.iloc[-1] <= 0:
        return None
    return {"cell": cell, "weeks": (d["t"].iloc[-1] - d["t"].iloc[0]).total_seconds() / (7*24*3600),
            "throughput_Ah": float(thr.iloc[-1]), "soh_last": float(soh.iloc[-1]),
            "fade_per_100Ah": float((1 - soh.iloc[-1]) / thr.iloc[-1] * 1e4),
            "fade_per_week": float((1 - soh.iloc[-1]) /
                                   ((d["t"].iloc[-1]-d["t"].iloc[0]).total_seconds()/(7*24*3600)) * 100)}

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dest", type=Path, default=None,
                    help=f"output folder (default: <out-dir>/{DEFAULT_SUBFOLDER})")
    args = ap.parse_args()

    rows = []
    for window, arms in MATCHED.items():
        for T, cells in arms.items():
            for c in cells:
                m = cell_metrics(args.out_dir, c)
                if m is None:
                    print(f"[WARN] {c}: no usable data")
                    continue
                rows.append({"window": window, "temp_C": T, **m})
    if not rows:
        print("[ERROR] no matched cells found")
        return
    df = pd.DataFrame(rows)

    windows = list(MATCHED)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, metric, ylab in ((axes[0], "fade_per_100Ah", "capacity fade  [% per 100 Ah]"),
                             (axes[1], "fade_per_week", "capacity fade  [% per week]")):
        for k, T in enumerate((25, 40)):
            xs, ys = [], []
            for i, w in enumerate(windows):
                sub = df[(df.window == w) & (df.temp_C == T)]
                jitter = np.linspace(-0.06, 0.06, len(sub)) if len(sub) > 1 else [0]
                for j, (_, r) in enumerate(sub.iterrows()):
                    ax.scatter(i + (k - 0.5) * 0.26 + jitter[j], r[metric],
                               s=70, color=COLOUR[T], edgecolor="k", linewidth=0.5, zorder=3)
                    ax.annotate(r["cell"].split("_")[-1],
                                (i + (k - 0.5) * 0.26 + jitter[j], r[metric]),
                                fontsize=7, xytext=(6, -2), textcoords="offset points")
                if len(sub):
                    xs.append(i + (k - 0.5) * 0.26); ys.append(sub[metric].mean())
            ax.plot(xs, ys, "-o", ms=9, color=COLOUR[T], alpha=0.45, lw=2,
                    label=f"{T} C  (group mean)", zorder=2)
        ax.set_xticks(range(len(windows)), windows)
        ax.set_xlabel("SOC window")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, axis="y")
        ax.set_xlim(-0.5, len(windows) - 0.5)
    axes[0].legend(fontsize=9)
    axes[0].set_title("normalised by throughput  (fair comparison)", fontsize=10)
    axes[1].set_title("normalised by time  (confounded: 40 C ran 2x longer)", fontsize=10)
    fig.suptitle("JGNE: temperature effect depends on the SOC window "
                 "(SOC-matched cells, 2 per arm)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    dest = args.dest or (args.out_dir / DEFAULT_SUBFOLDER)
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / "jgne_temp_soc_interaction.png"
    csv = dest / "jgne_temp_soc_interaction.csv"
    fig.savefig(png, dpi=140); plt.close(fig)
    df.to_csv(csv, index=False)
    print(f"[OK] {png}")
    print(f"[OK] {csv}")
    print()
    print(df[["window","temp_C","cell","weeks","throughput_Ah","soh_last",
              "fade_per_100Ah","fade_per_week"]].to_string(index=False))
    print()
    g = df.groupby(["window","temp_C"])[["fade_per_100Ah","fade_per_week"]].mean().round(3)
    print("group means:"); print(g.to_string())

if __name__ == "__main__":
    main()
