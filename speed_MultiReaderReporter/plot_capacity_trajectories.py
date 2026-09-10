# speed_MultiReaderReporter/plot_capacity_trajectories.py
"""Capacity / SoH trajectories per cell, for both capacity measurements.

Two different measurements, one script:

  --family jgne      cell_feature/SPEED_JGNE_*.csv     cap_ocv_dis
                     the C/10 OCV discharge of rul_JGNE_RPT_0C1 (step 6)

  --family lwhk_c2   cell_feature_cu_c2/SPEED_LWHK_*   cap_dis
                     the C/2 discharge of rul_eka_CU (step 6), 0.900 A vs
                     CNom 1.8 Ah, extracted by extract_lwhk_cu_discharge.py

These are NOT the same quantity - a C/2 discharge delivers less than a C/10 one
and carries polarisation - so the two families are drawn as separate figures and
should not be read off a shared axis. SoH is always normalised per cell against
that cell's own first checkup, which makes the SHAPES comparable even though the
absolute capacities are not.

Three panels: absolute capacity vs time, SoH vs time, and SoH vs cumulative
CYCLING throughput in Ah. Cells are coloured by an experimental condition taken
from the condition_overview_*.csv tables.

Usage:
    python speed_MultiReaderReporter/plot_capacity_trajectories.py --family jgne
    python speed_MultiReaderReporter/plot_capacity_trajectories.py --family jgne --group-by temp_C
    python speed_MultiReaderReporter/plot_capacity_trajectories.py --family lwhk_c2 --group-by fec
    python speed_MultiReaderReporter/plot_capacity_trajectories.py --family both
    python speed_MultiReaderReporter/plot_capacity_trajectories.py --family jgne \
        --group-by temp_C --only 75,76,97,98 --label-cells
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
DEFAULT_SUBFOLDER = "capacity_plots"   # figures land here, not in the output root

PRESETS = {
    "jgne": {
        "subdir": "cell_feature",
        "pattern": "SPEED_JGNE_*",
        "cap_col": "cap_ocv_dis",
        "cap_label": "OCV C/10 discharge capacity [Ah]",
        "cond_csv": "condition_overview_homocomp_jgne.csv",
        "group_by": "pause_h",
        "thr_source": "pipeline",      # throughput_cum column, in A.s
        "thr_exclude": None,
    },
    "lwhk_c2": {
        "subdir": "cell_feature_cu_c2",
        "pattern": "SPEED_LWHK_*",
        "cap_col": "cap_dis",
        "cap_label": "C/2 discharge capacity [Ah]",
        "cond_csv": "condition_overview_SPEED_LWHK.csv",
        "group_by": "fec",
        "thr_source": "report",        # summed from <cell>/cycling/report.csv
        "thr_exclude": "rul_eka_CU",   # count cycling only, not the checkup itself
    },
}

def load_conditions(out_dir: Path, cond_csv: str) -> pd.DataFrame:
    path = out_dir / cond_csv
    if not path.exists():
        print(f"[WARN] {path} not found; cells will not be grouped")
        return pd.DataFrame(columns=["cell"])
    c = pd.read_csv(path)
    return (c.sort_values("n_files", ascending=False)
             .drop_duplicates(subset="cell", keep="first"))


def throughput_ah(out_dir: Path, cell: str, times: pd.Series, cfg: dict,
                  df: pd.DataFrame) -> np.ndarray | None:
    """Cumulative CYCLING throughput in Ah at each checkup time.

    jgne     : the pipeline's throughput_cum, which integrates |I|dt in AMPERE-
               SECONDS over cycling runs only - divide by 3600 for Ah.
    lwhk_c2  : summed from <cell>/cycling/report.csv, skipping the rul_eka_CU
               rows so it counts cycling only, matching the pipeline definition.
               Including them shifts the total by >100 Ah and makes the series
               alternate, because the checkup itself passes charge.
    """
    if cfg["thr_source"] == "pipeline":
        if "throughput_cum" in df.columns:
            v = pd.to_numeric(df["throughput_cum"], errors="coerce").to_numpy(dtype=float)
            if np.nanmax(np.abs(v)) > 0:
                return v / 3600.0
        return None

    rep_path = out_dir / cell / "cycling" / "report.csv"
    if not rep_path.exists():
        rep_path = out_dir / cell / "total" / "report.csv"
    if not rep_path.exists():
        return None
    rep = pd.read_csv(rep_path)
    rep = rep[rep["program"].astype(str) != "TOTAL"]
    excl = cfg.get("thr_exclude")
    if excl:
        rep = rep[~rep["program"].astype(str).str.contains(excl, case=False, na=False)]
    if rep.empty or "throughput_Ah" not in rep.columns:
        return None
    rep = rep.assign(end=pd.to_datetime(rep["end_time"], errors="coerce")).dropna(subset=["end"])
    rep = rep.sort_values("end")
    rep["cum"] = rep["throughput_Ah"].cumsum()
    out = []
    for t in times:
        prior = rep.loc[rep["end"] <= t, "cum"]
        out.append(float(prior.max()) if len(prior) else 0.0)
    return np.asarray(out, dtype=float)

def cell_trajectory(path: Path, cap_col: str) -> pd.DataFrame | None:
    df = pd.read_csv(path, usecols=lambda c: c in (cap_col, "CU_time", "throughput_cum"))
    if cap_col not in df.columns or "CU_time" not in df.columns or df.empty:
        return None
    t = pd.to_datetime(df["CU_time"], errors="coerce")
    df = df.assign(t=t).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    if df.empty:
        return None
    df["weeks"] = (df["t"] - df["t"].iloc[0]).dt.total_seconds() / (7 * 24 * 3600)
    df["SOH"] = df[cap_col] / df[cap_col].iloc[0]
    return df


def draw_facets(family: str, args, dest: Path, trajectories, cond, cfg):
    """One column per level of --facet-by, cells coloured by --group-by.

    Isolates the second condition inside groups that share the first, e.g. LWHK
    rest_min compared only among cells at the same fec, so the fec effect cannot
    masquerade as a rest effect.
    """
    facet_by, group_by = args.facet_by, (args.group_by or cfg["group_by"])
    if facet_by not in cond.columns or group_by not in cond.columns:
        print(f"[WARN] need both {facet_by!r} and {group_by!r} in the condition table")
        return
    fmap = dict(zip(cond["cell"], pd.to_numeric(cond[facet_by], errors="coerce")))
    gmap = dict(zip(cond["cell"], pd.to_numeric(cond[group_by], errors="coerce")))

    items = [(c, tr, fmap.get(c, np.nan), gmap.get(c, np.nan)) for c, tr, _ in trajectories]
    facets = sorted({f for _, _, f, _ in items if pd.notna(f)})
    if not facets:
        print(f"[WARN] no usable {facet_by} levels")
        return
    glevels = sorted({g for _, _, _, g in items if pd.notna(g)})
    cmap = plt.get_cmap("turbo")
    colour = {lv: cmap(0.1 + 0.8 * i / max(1, len(glevels) - 1))
              for i, lv in enumerate(glevels)}

    nrow, ncol = 2, len(facets)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 8.2),
                             squeeze=False, sharey="row")
    for j, fv in enumerate(facets):
        seen = set()
        for cell, tr, f, g in items:
            if pd.isna(f) or f != fv:
                continue
            col = colour.get(g, "0.6")
            lab = None
            if pd.notna(g) and g not in seen:
                lab = f"{group_by} = {g:g}"
                seen.add(g)
            axes[0][j].plot(tr["weeks"], tr["SOH"], marker="o", ms=4, lw=1.4,
                            color=col, label=lab)
            if tr["throughput_Ah"].notna().any():
                axes[1][j].plot(tr["throughput_Ah"], tr["SOH"], marker="o", ms=4,
                                lw=1.4, color=col)
            axes[0][j].annotate(cell.split("_")[-1],
                                (tr["weeks"].iloc[-1], tr["SOH"].iloc[-1]),
                                fontsize=7, xytext=(3, 0), textcoords="offset points")
        n = sum(1 for _, _, f, _ in items if pd.notna(f) and f == fv)
        axes[0][j].set_title(f"{facet_by} = {fv:g}   ({n} cells)", fontsize=10)
        axes[0][j].set_xlabel("weeks since first checkup")
        axes[1][j].set_xlabel("cumulative cycling throughput [Ah]")
        for ax in (axes[0][j], axes[1][j]):
            ax.axhline(1.0, color="0.75", lw=0.8, ls="--", zorder=0)
            ax.grid(alpha=0.25)
        if seen:
            axes[0][j].legend(fontsize=8)
    axes[0][0].set_ylabel("SoH   (vs time)")
    axes[1][0].set_ylabel("SoH   (vs throughput)")
    fig.suptitle(f"{family.upper()}: effect of {group_by} within each {facet_by}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = dest / f"{family}_capacity_by_{facet_by}_{group_by}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[OK] {png}")

def draw_family(family: str, args, dest: Path):
    cfg = PRESETS[family]
    folder = args.out_dir / cfg["subdir"]
    if not folder.is_dir():
        print(f"[ERROR] {folder} not found")
        return

    only = None
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}

    cond = load_conditions(args.out_dir, cfg["cond_csv"])
    group_by = args.group_by or cfg["group_by"]
    gmap = {}
    if group_by in cond.columns:
        gmap = dict(zip(cond["cell"], pd.to_numeric(cond[group_by], errors="coerce")))

    trajectories = []
    for path in sorted(folder.glob(f"{cfg['pattern']}.csv")):
        if only and not any(path.stem.endswith(f"_{o}") or o == path.stem for o in only):
            continue
        tr = cell_trajectory(path, cfg["cap_col"])
        if tr is None or len(tr) < 1:
            continue
        thr = throughput_ah(args.out_dir, path.stem, tr["t"], cfg, tr)
        tr["throughput_Ah"] = thr if thr is not None else np.nan
        trajectories.append((path.stem, tr, gmap.get(path.stem, np.nan)))

    if not trajectories:
        print(f"[INFO] no usable cells for {family}")
        return

    levels = sorted({g for _, _, g in trajectories if pd.notna(g)})
    cmap = plt.get_cmap("viridis")
    colour = {lv: cmap(i / max(1, len(levels) - 1)) for i, lv in enumerate(levels)}

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.2))
    seen = set()
    for cell, tr, g in trajectories:
        col = colour.get(g, "0.6")
        lab = None
        if pd.notna(g) and g not in seen:
            lab = f"{group_by} = {g:g}"
            seen.add(g)
        elif pd.isna(g) and "na" not in seen:
            lab = f"{group_by} n/a"
            seen.add("na")
        for ax, ycol in ((axes[0], cfg["cap_col"]), (axes[1], "SOH")):
            ax.plot(tr["weeks"], tr[ycol], marker="o", ms=3.5, lw=1.2,
                    color=col, alpha=0.85, label=lab if ax is axes[0] else None)
        if tr["throughput_Ah"].notna().any():
            axes[2].plot(tr["throughput_Ah"], tr["SOH"], marker="o", ms=3.5,
                         lw=1.2, color=col, alpha=0.85)
        if args.label_cells:
            axes[1].annotate(cell.split("_")[-1],
                             (tr["weeks"].iloc[-1], tr["SOH"].iloc[-1]),
                             fontsize=7, xytext=(3, 0), textcoords="offset points")

    axes[0].set_ylabel(cfg["cap_label"])
    axes[1].set_ylabel("SoH  (capacity / first checkup)")
    axes[2].set_ylabel("SoH  (capacity / first checkup)")
    for ax in (axes[1], axes[2]):
        ax.axhline(1.0, color="0.7", lw=0.8, ls="--", zorder=0)
    for ax in (axes[0], axes[1]):
        ax.set_xlabel("weeks since first checkup")
    axes[2].set_xlabel("cumulative cycling throughput [Ah]")
    n_thr = sum(1 for _, tr, _ in trajectories if tr["throughput_Ah"].notna().any())
    axes[2].set_title(f"{n_thr}/{len(trajectories)} cells with throughput", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
    if levels or seen:
        axes[0].legend(fontsize=8, ncol=2)

    n_pts = sum(len(tr) for _, tr, _ in trajectories)
    fig.suptitle(f"{family.upper()}: {cfg['cap_label'].split(' [')[0]} "
                 f"- {len(trajectories)} cells, {n_pts} checkups", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if args.facet_by:
        draw_facets(family, args, dest, trajectories, cond, cfg)

    png = dest / f"{family}_capacity_trajectory_{group_by}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[OK] {png}")

    summary = pd.DataFrame([
        {"cell": c, "n_checkups": len(tr), "weeks": tr["weeks"].iloc[-1],
         "cap_first": tr[cfg["cap_col"]].iloc[0], "cap_last": tr[cfg["cap_col"]].iloc[-1],
         "soh_last": tr["SOH"].iloc[-1],
         "throughput_Ah_last": tr["throughput_Ah"].iloc[-1], group_by: g}
        for c, tr, g in trajectories])
    csv = dest / f"{family}_capacity_trajectory_{group_by}.csv"
    summary.to_csv(csv, index=False)
    print(f"[OK] {csv}")
    print(summary.to_string(index=False))

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["jgne", "lwhk_c2", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dest", type=Path, default=None,
                    help=f"output folder (default: <out-dir>/{DEFAULT_SUBFOLDER})")
    ap.add_argument("--facet-by", default=None,
                    help="condition to split into separate panels, e.g. fec; "
                         "cells inside a panel are coloured by --group-by")
    ap.add_argument("--group-by", default=None,
                    help="condition used to colour the cells (default: pause_h for jgne, fec for lwhk_c2)")
    ap.add_argument("--only", default=None,
                    help="comma separated cell numbers to keep, e.g. 75,76,97,98")
    ap.add_argument("--label-cells", action="store_true",
                    help="annotate the end of each trajectory with the cell number")
    args = ap.parse_args()

    dest = args.dest or (args.out_dir / DEFAULT_SUBFOLDER)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"writing to {dest}")
    for fam in (["jgne", "lwhk_c2"] if args.family == "both" else [args.family]):
        print(f"===== {fam}")
        draw_family(fam, args, dest)
        print()

if __name__ == "__main__":
    main()
