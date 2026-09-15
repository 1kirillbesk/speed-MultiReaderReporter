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
import re
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
        "prefer_col": "pause_h",
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
        "prefer_col": "fec",
        "thr_source": "report",        # summed from <cell>/cycling/report.csv
        "thr_exclude": "rul_eka_CU",   # count cycling only, not the checkup itself
    },
    "bald": {
        "subdir": "cell_feature",
        "pattern": "BALD_35E_*",
        "cap_col": "cap_ocv_dis",
        "cap_label": "OCV C/10 discharge capacity [Ah]",
        "cond_csv": "condition_overview_BALD_35E.csv",
        "group_by": "c_rate",
        "prefer_col": "c_rate",
        "thr_source": "pipeline",
        "thr_exclude": None,
    },
    "bald_temp": {
        "subdir": "cell_feature",
        "pattern": "*",
        "cap_col": "cap_ocv_dis",
        "cap_label": "OCV charge capacity [Ah]",
        "cond_csv": "condition_overview_BALD.csv",
        "group_by": "temp_C",
        "prefer_col": "soc_start",
        "thr_source": "pipeline",
        "thr_exclude": None,
    },
    "bald_c2": {
        "subdir": "cell_feature",
        "pattern": "BALD_35E_*",
        "cap_col": "cap_dis",
        "cap_label": "C/2 discharge capacity [Ah]",
        "cond_csv": "condition_overview_BALD_35E.csv",
        "group_by": "c_rate",
        "prefer_col": "c_rate",
        "thr_source": "pipeline",
        "thr_exclude": None,
    },
}

def load_conditions(out_dir: Path, cond_csv: str,
                    prefer_col: str | None = None) -> pd.DataFrame:
    """One condition row per cell.

    A cell normally has several labels - the cycling programme AND the checkup
    (rul_SAM_CU_varCapa etc.). Ranking purely by n_files picks the CHECKUP,
    because there are far more checkup files, which silently discards the
    cycling condition: for BALD that reported dyn on 2 cells instead of 21.

    So: scalar conditions come from the preferred row (one carrying `prefer_col`,
    e.g. c_rate, which only cycling labels have), and 0/1 FLAG columns are OR-ed
    across every label the cell ever ran - "this cell ever ran a dyn programme".
    """
    path = out_dir / cond_csv
    if not path.exists():
        print(f"[WARN] {path} not found; cells will not be grouped")
        return pd.DataFrame(columns=["cell"])
    c = pd.read_csv(path)

    c = c.copy()
    c["_pref"] = (c[prefer_col].notna().astype(int)
                  if prefer_col and prefer_col in c.columns else 0)
    chosen = (c.sort_values(["_pref", "n_files"], ascending=[False, False])
                .drop_duplicates(subset="cell", keep="first")
                .drop(columns="_pref"))

    flag_cols = [col for col in c.columns
                 if col not in ("cell", "label", "n_files", "first_test",
                                "last_test", "_pref")
                 and pd.api.types.is_numeric_dtype(c[col])
                 and set(pd.unique(c[col].dropna())) <= {0, 1}]
    if flag_cols:
        any_flag = c.groupby("cell")[flag_cols].max()
        chosen = chosen.set_index("cell")
        chosen[flag_cols] = any_flag.reindex(chosen.index)[flag_cols]
        chosen = chosen.reset_index()
    return chosen


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


def category_of(cond: pd.DataFrame, cols: list[str]) -> dict:
    """cell -> category string built from several condition columns.

    A 0/1 flag contributes its NAME when set ("dyn", "sdod"); anything else
    contributes "col=value" ("c_rate=0.25"). Cells with nothing set are "plain".
    Lets one axis of the plot encode a combination rather than a single column.
    """
    out = {}
    use = [c for c in cols if c in cond.columns]
    for c in cols:
        if c not in cond.columns:
            print(f"[WARN] no condition column {c!r}; ignored")
    if not use:
        return out
    for _, row in cond.iterrows():
        parts = []
        for c in use:
            v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
            if pd.isna(v):
                continue
            binary = set(pd.unique(pd.to_numeric(cond[c], errors="coerce").dropna())) <= {0, 1}
            if binary:
                if v != 0:
                    parts.append(c)
            else:
                parts.append(f"{c}={v:g}")
        out[row["cell"]] = "+".join(parts) if parts else "plain"
    return out

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
    if "," in facet_by:
        fmap = category_of(cond, [c.strip() for c in facet_by.split(",")])
        facet_by = facet_by.replace(",", "+")
    elif facet_by in cond.columns:
        fmap = dict(zip(cond["cell"], pd.to_numeric(cond[facet_by], errors="coerce")))
    else:
        print(f"[WARN] no condition column {facet_by!r}"); return
    if "," in group_by:
        gmap = category_of(cond, [c.strip() for c in group_by.split(",")])
        group_by = group_by.replace(",", "+")
    elif group_by in cond.columns:
        gmap = dict(zip(cond["cell"], pd.to_numeric(cond[group_by], errors="coerce")))
    else:
        print(f"[WARN] no condition column {group_by!r}"); return

    items = [(c, tr, fmap.get(c, np.nan), gmap.get(c, np.nan)) for c, tr, _ in trajectories]
    facets = sorted({f for _, _, f, _ in items if pd.notna(f)}, key=str)
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
                lab = f"{group_by} = {g:g}" if isinstance(g, (int, float)) else str(g)
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
        lab_fv = f"{fv:g}" if isinstance(fv, (int, float)) else str(fv)
        axes[0][j].set_title(f"{lab_fv}   ({n} cells)", fontsize=10)
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
    import re as _re
    png = dest / _re.sub(r"[^A-Za-z0-9_.]+", "_",
                         f"{family}_capacity_by_{facet_by}_{group_by}.png")
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

    cond = load_conditions(args.out_dir, cfg["cond_csv"], cfg.get("prefer_col"))
    group_by = args.group_by or cfg["group_by"]
    categorical = "," in group_by
    gmap = {}
    if categorical:
        gmap = category_of(cond, [c.strip() for c in group_by.split(",")])
        group_by = group_by.replace(",", "/")
    elif group_by in cond.columns:
        gmap = dict(zip(cond["cell"], pd.to_numeric(cond[group_by], errors="coerce")))
    # '/' reads fine in a legend label but is a path separator in a filename
    gtag = re.sub(r"[^A-Za-z0-9]+", "_", group_by).strip("_")

    def _flags(cols):
        """cell -> True if any of `cols` is truthy for that cell."""
        out = {}
        for col in cols:
            if col not in cond.columns:
                print(f"[WARN] no condition column {col!r}; ignored")
                continue
            v = pd.to_numeric(cond[col], errors="coerce").fillna(0) != 0
            for cell, hit in zip(cond["cell"], v):
                out[cell] = out.get(cell, False) or bool(hit)
        return out

    excl = _flags([c.strip() for c in args.exclude.split(",")]) if args.exclude else {}
    keep = _flags([c.strip() for c in args.only_flag.split(",")]) if args.only_flag else None
    # Category per cell from which flags are set: "plain", "dyn", "dyn+sdod", ...
    style_cols, cat_of = [], {}
    if args.style_by:
        style_cols = [c.strip() for c in args.style_by.split(",")
                      if c.strip() in cond.columns]
        missing = [c.strip() for c in args.style_by.split(",") if c.strip() not in cond.columns]
        for c in missing:
            print(f"[WARN] no condition column {c!r}; ignored")
        if style_cols:
            cat_of = category_of(cond, style_cols)

    trajectories, dropped = [], []
    for path in sorted(folder.glob(f"{cfg['pattern']}.csv")):
        if only and not any(path.stem.endswith(f"_{o}") or o == path.stem for o in only):
            continue
        if excl.get(path.stem, False):
            dropped.append(path.stem); continue
        if keep is not None and not keep.get(path.stem, False):
            dropped.append(path.stem); continue
        tr = cell_trajectory(path, cfg["cap_col"])
        if tr is None or len(tr) < 1:
            continue
        thr = throughput_ah(args.out_dir, path.stem, tr["t"], cfg, tr)
        tr["throughput_Ah"] = thr if thr is not None else np.nan
        tr.attrs["cat"] = cat_of.get(path.stem, "plain")
        trajectories.append((path.stem, tr, gmap.get(path.stem, np.nan)))

    if dropped:
        print(f"[filter] {len(dropped)} cell(s) excluded: "
              f"{', '.join(c.split('_')[-1] for c in dropped)}")

    if not trajectories:
        print(f"[INFO] no usable cells for {family}")
        return

    levels = sorted({g for _, _, g in trajectories if pd.notna(g)},
                    key=lambda v: (v != "plain", v) if categorical else v)
    if categorical:
        base = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Set2").colors)
        colour = {lv: base[i % len(base)] for i, lv in enumerate(levels)}
    else:
        cmap = plt.get_cmap("viridis")
        colour = {lv: cmap(i / max(1, len(levels) - 1)) for i, lv in enumerate(levels)}

    # line style + marker per flag combination; "plain" always solid circles
    STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"),
              ((0, (3, 1, 1, 1)), "v"), ((0, (5, 1)), "P"), ((0, (1, 1)), "X"),
              ((0, (3, 2, 1, 2, 1, 2)), "*")]
    cats = sorted({tr.attrs.get("cat", "plain") for _, tr, _ in trajectories},
                  key=lambda c: (c != "plain", c))
    style_map = {c: STYLES[i % len(STYLES)] for i, c in enumerate(cats)}

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.2))
    seen = set()
    for cell, tr, g in trajectories:
        col = colour.get(g, "0.6")
        lab = None
        if pd.notna(g) and g not in seen:
            n_g = sum(1 for _, _, gg in trajectories if gg == g)
            lab = (f"{g}  (n={n_g})" if categorical else f"{group_by} = {g:g}")
            seen.add(g)
        elif pd.isna(g) and "na" not in seen:
            lab = f"{group_by} n/a"
            seen.add("na")
        cat = tr.attrs.get("cat", "plain")
        ls, mk = style_map.get(cat, ("-", "o"))
        kw = dict(marker=mk, ms=3.5, lw=1.2, color=col, alpha=0.85, ls=ls)
        if cat != "plain":
            kw.update(mfc="none", lw=1.4)
        for ax, ycol in ((axes[0], cfg["cap_col"]), (axes[1], "SOH")):
            ax.plot(tr["weeks"], tr[ycol], label=lab if ax is axes[0] else None, **kw)
        if tr["throughput_Ah"].notna().any():
            axes[2].plot(tr["throughput_Ah"], tr["SOH"], **kw)
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
        first = axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    if len(style_map) > 1:
        from matplotlib.lines import Line2D
        n_by_cat = {}
        for _, tr, _ in trajectories:
            c = tr.attrs.get("cat", "plain")
            n_by_cat[c] = n_by_cat.get(c, 0) + 1
        handles = [Line2D([], [], color="0.25", ls=style_map[c][0],
                          marker=style_map[c][1], ms=5,
                          mfc="none" if c != "plain" else "0.25",
                          label=f"{c}  (n={n_by_cat.get(c, 0)})")
                   for c in cats]
        axes[1].legend(handles=handles, fontsize=8, title="programme variant",
                       title_fontsize=8, loc="lower left")

    n_pts = sum(len(tr) for _, tr, _ in trajectories)
    extra = ""
    if style_cols:
        extra = f"   (line style = {'/'.join(style_cols)})"
    if args.exclude:
        extra += f"   [excluded: {args.exclude}]"
    fig.suptitle(f"{family.upper()}: {cfg['cap_label'].split(' [')[0]} "
                 f"- {len(trajectories)} cells, {n_pts} checkups{extra}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if args.facet_by:
        draw_facets(family, args, dest, trajectories, cond, cfg)

    png = dest / f"{family}_capacity_trajectory_{gtag}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[OK] {png}")

    summary = pd.DataFrame([
        {"cell": c, "n_checkups": len(tr), "weeks": tr["weeks"].iloc[-1],
         "cap_first": tr[cfg["cap_col"]].iloc[0], "cap_last": tr[cfg["cap_col"]].iloc[-1],
         "soh_last": tr["SOH"].iloc[-1],
         "throughput_Ah_last": tr["throughput_Ah"].iloc[-1], group_by: g}
        for c, tr, g in trajectories])
    csv = dest / f"{family}_capacity_trajectory_{gtag}.csv"
    summary.to_csv(csv, index=False)
    print(f"[OK] {csv}")
    print(summary.to_string(index=False))

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family",
                    choices=["jgne", "lwhk_c2", "bald", "bald_c2", "bald_temp", "both"],
                    default="both")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dest", type=Path, default=None,
                    help=f"output folder (default: <out-dir>/{DEFAULT_SUBFOLDER})")
    ap.add_argument("--exclude", default=None,
                    help="comma separated flag columns; cells where any is truthy are "
                         "DROPPED, e.g. --exclude dyn,scur,sdod,ssoc")
    ap.add_argument("--only-flag", default=None,
                    help="comma separated flag columns; keep ONLY cells where any is "
                         "truthy (inverse of --exclude)")
    ap.add_argument("--style-by", default=None,
                    help="comma separated flag columns; each COMBINATION gets its own "
                         "line style + marker so the variants are distinguishable "
                         "without removing them, e.g. --style-by dyn,sdod,scur")
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
