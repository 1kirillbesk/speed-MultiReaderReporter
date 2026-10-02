# make_report_figures.py
"""Publication figures + markdown report for the JGNE and LWHK condition study.

Writes every figure into <dest>/figures/ and a report.md alongside that links to
them with relative paths, so the folder is self-contained and portable.

Feature and condition names are rendered in mathematical notation (see MATH
below) rather than the raw column names, on axes, heatmaps and SHAP plots alike.

Usage:
    python make_report_figures.py
    python make_report_figures.py --dest ./report_jgne_lwhk
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------- style
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.titlesize": 16, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 200, "mathtext.default": "regular",
})
C25, C40 = "#3b6ea5", "#c1443c"
PALETTE = ["#3b6ea5", "#c1443c", "#3f9b4f", "#8a5fbf", "#d98c00", "#22a0a8", "#b5476b"]

# ------------------------------------------------------- mathematical labels
MATH = {
    # capacities / basics
    "cap_ocv_dis": r"$C_\mathrm{OCV,dis}$",
    "cap_dis":     r"$C_\mathrm{dis}$",
    "cap_cha":     r"$C_\mathrm{cha}$",
    "SOH":         r"$\mathrm{SOH}$",
    "weeks":       r"$t$  [weeks]",
    "throughput_cum": r"$Q_\mathrm{thr}$  [Ah]",
    "throughput_sum": r"$\Delta Q_\mathrm{thr}$  [Ah]",
    # incremental capacity, charge side
    "mean_d_dqdv_l_c": r"$\overline{\Delta\,\mathrm{d}Q/\mathrm{d}V}^{\,\mathrm{cha}}_\mathrm{low}$",
    "mean_d_dqdv_m_c": r"$\overline{\Delta\,\mathrm{d}Q/\mathrm{d}V}^{\,\mathrm{cha}}_\mathrm{mid}$",
    "mean_d_dqdv_h_c": r"$\overline{\Delta\,\mathrm{d}Q/\mathrm{d}V}^{\,\mathrm{cha}}_\mathrm{high}$",
    "var_d_dqdv_l_c":  r"$\mathrm{Var}(\Delta\,\mathrm{d}Q/\mathrm{d}V)^{\,\mathrm{cha}}_\mathrm{low}$",
    "var_d_dqdv_m_c":  r"$\mathrm{Var}(\Delta\,\mathrm{d}Q/\mathrm{d}V)^{\,\mathrm{cha}}_\mathrm{mid}$",
    "var_d_dqdv_h_c":  r"$\mathrm{Var}(\Delta\,\mathrm{d}Q/\mathrm{d}V)^{\,\mathrm{cha}}_\mathrm{high}$",
    # discharge side
    "mean_d_dqdv_l_d": r"$\overline{\Delta\,\mathrm{d}Q/\mathrm{d}V}^{\,\mathrm{dis}}_\mathrm{low}$",
    "mean_d_dqdv_m_d": r"$\overline{\Delta\,\mathrm{d}Q/\mathrm{d}V}^{\,\mathrm{dis}}_\mathrm{mid}$",
    "mean_d_dqdv_h_d": r"$\overline{\Delta\,\mathrm{d}Q/\mathrm{d}V}^{\,\mathrm{dis}}_\mathrm{high}$",
    "var_d_dqdv_l_d":  r"$\mathrm{Var}(\Delta\,\mathrm{d}Q/\mathrm{d}V)^{\,\mathrm{dis}}_\mathrm{low}$",
    "var_d_dqdv_m_d":  r"$\mathrm{Var}(\Delta\,\mathrm{d}Q/\mathrm{d}V)^{\,\mathrm{dis}}_\mathrm{mid}$",
    "var_d_dqdv_h_d":  r"$\mathrm{Var}(\Delta\,\mathrm{d}Q/\mathrm{d}V)^{\,\mathrm{dis}}_\mathrm{high}$",
    # integrated charge
    "mean_dQ_c": r"$\overline{\Delta Q(V)}^{\,\mathrm{cha}}$",
    "var_dQ_c":  r"$\mathrm{Var}(\Delta Q(V))^{\,\mathrm{cha}}$",
    "mean_dQ_d": r"$\overline{\Delta Q(V)}^{\,\mathrm{dis}}$",
    "var_dQ_d":  r"$\mathrm{Var}(\Delta Q(V))^{\,\mathrm{dis}}$",
    # thermal
    "mean_dqdv_lt":  r"$\overline{\Delta\,\mathrm{d}T/\mathrm{d}V}_\mathrm{low}$",
    "mean_dqdv_mt":  r"$\overline{\Delta\,\mathrm{d}T/\mathrm{d}V}_\mathrm{mid}$",
    "mean_dqdv_ht":  r"$\overline{\Delta\,\mathrm{d}T/\mathrm{d}V}_\mathrm{high}$",
    "var_d_dqdv_lt": r"$\mathrm{Var}(\Delta\,\mathrm{d}T/\mathrm{d}V)_\mathrm{low}$",
    "var_d_dqdv_mt": r"$\mathrm{Var}(\Delta\,\mathrm{d}T/\mathrm{d}V)_\mathrm{mid}$",
    "var_d_dqdv_ht": r"$\mathrm{Var}(\Delta\,\mathrm{d}T/\mathrm{d}V)_\mathrm{high}$",
    "mean_d_Qt": r"$\overline{\Delta T(V)}$",
    "var_d_Qt":  r"$\mathrm{Var}(\Delta T(V))$",
    # resistances
    "R_0.2s_ohm": r"$R_{0.2\,\mathrm{s}}$", "R_1s_ohm": r"$R_{1\,\mathrm{s}}$",
    "R_10s_ohm": r"$R_{10\,\mathrm{s}}$",
    # conditions
    "temp_C": r"$T$  [$^\circ$C]", "pause_h": r"$t_\mathrm{pause}$  [h]",
    "has_pulse": r"pulse", "low_soc": r"lowSOC",
    "soc_start": r"$\mathrm{SOC}_\mathrm{min}$", "soc_end": r"$\mathrm{SOC}_\mathrm{max}$",
    "soc_mean": r"$\overline{\mathrm{SOC}}$", "soc_dod": r"$\mathrm{DOD}$",
    "fec": r"$\mathrm{FEC}$", "rest_min": r"$t_\mathrm{rest}$  [min]",
    "ref": r"ref",
    # derived targets
    "fade_per_100Ah": r"$\dot{q}_{Q}$  [% / 100 Ah]",
    "fade_per_week":  r"$\dot{q}_{t}$  [% / week]",
    "k_sqrt_Ah": r"$k_{\sqrt{Q}}$", "k_lin_Ah": r"$k_{Q}$",
}
def m(name: str) -> str:
    return MATH.get(name, name.replace("_", r"\_"))

# --------------------------------------------------------------------- data
JG = Path("E:/download/jgne/out_jgne")

DATASETS = {
    "JGNE": dict(feat=JG / "cell_feature", pattern="SPEED_JGNE_*",
                 cond=JG / "condition_overview_homocomp_jgne.csv",
                 soh_col="cap_ocv_dis", thr_exclude=None,
                 conditions=["temp_C", "pause_h", "has_pulse", "low_soc", "soc_mean"],
                 cap_label=r"$C_\mathrm{OCV,dis}$  [Ah]  (C/10 OCV discharge)"),
    "LWHK": dict(feat=JG / "cell_feature_cu_c2", pattern="SPEED_LWHK_*",
                 cond=JG / "condition_overview_SPEED_LWHK.csv",
                 soh_col="cap_dis", thr_exclude="rul_eka_CU",
                 conditions=["fec", "rest_min"],
                 cap_label=r"$C_\mathrm{dis}$  [Ah]  (C/2 discharge)"),
}

def thr_from_report(cell, times, exclude):
    rep = JG / cell / "cycling" / "report.csv"
    if not rep.exists():
        rep = JG / cell / "total" / "report.csv"
    if not rep.exists():
        return None
    r = pd.read_csv(rep)
    r = r[r["program"].astype(str) != "TOTAL"]
    if exclude:
        r = r[~r["program"].astype(str).str.contains(exclude, case=False, na=False)]
    if r.empty:
        return None
    r = r.assign(end=pd.to_datetime(r["end_time"], errors="coerce")).dropna(subset=["end"])
    r = r.sort_values("end"); r["cum"] = r["throughput_Ah"].cumsum()
    return np.array([float(r.loc[r["end"] <= t, "cum"].max()) if (r["end"] <= t).any() else 0.0
                     for t in times])

def load_traj(spec):
    out = {}
    for p in sorted(Path(spec["feat"]).glob(f"{spec['pattern']}.csv")):
        d = pd.read_csv(p, usecols=lambda c: c in (spec["soh_col"], "CU_time", "throughput_cum"))
        if spec["soh_col"] not in d.columns or len(d) < 2:
            continue
        t = pd.to_datetime(d["CU_time"], errors="coerce")
        d = d.assign(t=t).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
        if len(d) < 2:
            continue
        cap = pd.to_numeric(d[spec["soh_col"]], errors="coerce")
        soh = (cap / cap.iloc[0]).to_numpy()
        wk = ((d["t"] - d["t"].iloc[0]).dt.total_seconds() / (7 * 24 * 3600)).to_numpy()
        thr = (pd.to_numeric(d.get("throughput_cum", pd.Series(np.nan, index=d.index)),
                             errors="coerce") / 3600).to_numpy()
        if not np.isfinite(thr).any() or np.nanmax(thr) <= 0:
            alt = thr_from_report(p.stem, d["t"], spec["thr_exclude"])
            if alt is not None:
                thr = alt
        out[p.stem] = dict(cap=cap.to_numpy(), soh=soh, weeks=wk, thr=thr)
    return out

def load_cond(spec):
    c = pd.read_csv(spec["cond"])
    prefer = next((k for k in ("fec", "pause_h", "soc_start") if k in c.columns), None)
    c = c.copy()
    c["_p"] = c[prefer].notna().astype(int) if prefer else 0
    chosen = (c.sort_values(["_p", "n_files"], ascending=[False, False])
                .drop_duplicates("cell").drop(columns="_p"))
    flags = [k for k in c.columns
             if k not in ("cell", "label", "n_files", "first_test", "last_test", "_p")
             and pd.api.types.is_numeric_dtype(c[k])
             and set(pd.unique(c[k].dropna())) <= {0, 1}]
    if flags:
        anyf = c.groupby("cell")[flags].max()
        chosen = chosen.set_index("cell")
        chosen[flags] = anyf.reindex(chosen.index)[flags]
        chosen = chosen.reset_index()
    return chosen.set_index("cell")

# ------------------------------------------------------------------ figures

def fig_trajectory(name, spec, traj, cond, group_by, figdir):
    levels, gof = [], {}
    for cell in traj:
        v = cond[group_by].get(cell, np.nan) if group_by in cond.columns else np.nan
        gof[cell] = v
        if pd.notna(v) and v not in levels:
            levels.append(v)
    levels = sorted(levels)
    colour = {lv: PALETTE[i % len(PALETTE)] for i, lv in enumerate(levels)}

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    seen = set()
    for cell, d in traj.items():
        g = gof[cell]
        col = colour.get(g, "0.7")
        lab = None
        if pd.notna(g) and g not in seen:
            lab = f"{m(group_by).split('  [')[0]} = {g:g}"; seen.add(g)
        kw = dict(color=col, lw=1.8, marker="o", ms=4.5, alpha=0.9)
        ax[0].plot(d["weeks"], d["cap"], label=lab, **kw)
        ax[1].plot(d["weeks"], d["soh"], **kw)
        if np.isfinite(d["thr"]).any():
            ax[2].plot(d["thr"], d["soh"], **kw)
    ax[0].set_ylabel(spec["cap_label"]); ax[0].set_xlabel(m("weeks"))
    ax[1].set_ylabel(m("SOH")); ax[1].set_xlabel(m("weeks"))
    ax[2].set_ylabel(m("SOH")); ax[2].set_xlabel(m("throughput_cum"))
    for a in ax[1:]:
        a.axhline(1.0, color="0.6", lw=1, ls="--", zorder=0)
    ax[0].legend(frameon=False, ncol=2)
    fig.suptitle(f"{name}: capacity trajectories, coloured by {m(group_by).split('  [')[0]}"
                 f"   ({len(traj)} cells)")
    fig.tight_layout()
    f = figdir / f"{name.lower()}_trajectory_{group_by}.png"
    fig.savefig(f); plt.close(fig)
    return f.name

def fig_facet(name, traj, cond, facet_by, group_by, figdir):
    rows = [(c, cond[facet_by].get(c, np.nan), cond[group_by].get(c, np.nan))
            for c in traj if facet_by in cond.columns and group_by in cond.columns]
    facets = sorted({f for _, f, _ in rows if pd.notna(f)})
    glev = sorted({g for _, _, g in rows if pd.notna(g)})
    colour = {lv: PALETTE[i % len(PALETTE)] for i, lv in enumerate(glev)}
    if not facets:
        return None
    fig, axes = plt.subplots(1, len(facets), figsize=(4.9 * len(facets), 5.0),
                             squeeze=False, sharey=True)
    for j, fv in enumerate(facets):
        a = axes[0][j]; seen = set()
        for cell, f, g in rows:
            if pd.isna(f) or f != fv:
                continue
            d = traj[cell]
            lab = None
            if pd.notna(g) and g not in seen:
                lab = f"{m(group_by).split('  [')[0]} = {g:g}"; seen.add(g)
            x = d["thr"] if np.isfinite(d["thr"]).any() else d["weeks"]
            a.plot(x, d["soh"], color=colour.get(g, "0.7"), lw=1.9,
                   marker="o", ms=5, label=lab)
        a.set_title(f"{m(facet_by).split('  [')[0]} = {fv:g}")
        a.set_xlabel(m("throughput_cum"))
        a.axhline(1.0, color="0.6", lw=1, ls="--", zorder=0)
        if seen:
            a.legend(frameon=False)
    axes[0][0].set_ylabel(m("SOH"))
    fig.suptitle(f"{name}: effect of {m(group_by).split('  [')[0]} within each "
                 f"{m(facet_by).split('  [')[0]}")
    fig.tight_layout()
    f = figdir / f"{name.lower()}_facet_{facet_by}_{group_by}.png"
    fig.savefig(f); plt.close(fig)
    return f.name

def fig_heatmap(name, tbl, conds, feats, figdir):
    rho = np.full((len(feats), len(conds)), np.nan)
    pv = np.full_like(rho, np.nan)
    for i, ft in enumerate(feats):
        for j, cd in enumerate(conds):
            x = pd.to_numeric(tbl[cd], errors="coerce")
            y = pd.to_numeric(tbl[ft], errors="coerce")
            msk = x.notna() & y.notna()
            if msk.sum() < 4 or x[msk].nunique() < 2 or y[msk].nunique() < 2:
                continue
            r, p = spearmanr(x[msk], y[msk])
            rho[i, j], pv[i, j] = r, p
    fig, ax = plt.subplots(figsize=(1.55 * len(conds) + 4.2, 0.46 * len(feats) + 2.4))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(conds)), [m(c) for c in conds], rotation=30, ha="right")
    ax.set_yticks(range(len(feats)), [m(f) for f in feats])
    for i in range(len(feats)):
        for j in range(len(conds)):
            if np.isnan(rho[i, j]):
                continue
            star = "*" if pv[i, j] < 0.05 else ""
            ax.text(j, i, f"{rho[i,j]:+.2f}{star}", ha="center", va="center",
                    fontsize=10, color="white" if abs(rho[i, j]) > 0.55 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=r"Spearman $\rho$", fraction=0.035, pad=0.02)
    ax.set_title(f"{name}: condition–feature rank correlation\n"
                 r"$*$ marks $p<0.05$ (uncorrected)")
    fig.tight_layout()
    f = figdir / f"{name.lower()}_heatmap.png"
    fig.savefig(f); plt.close(fig)
    return f.name

def fig_shap(name, tbl, conds, target, figdir, n_boot=200, seed=0):
    """XGBoost + exact TreeSHAP, with BOOTSTRAP replicates so the beeswarm shows
    the spread of the attribution rather than a single fit.

    Every plotted point is a REAL cell: the cells are resampled with replacement
    `n_boot` times, the model refitted on each resample, and SHAP recomputed for
    all original cells. So the cloud is n_cells x n_boot points and its width is
    model uncertainty at this sample size - it is NOT synthetic data, and the
    caption states the true number of cells.

    shap 0.49 cannot parse the xgboost 3.2 tree dump, so contributions come from
    xgboost's own pred_contribs (the same exact TreeSHAP algorithm).
    """
    import xgboost as xgb
    from sklearn.metrics import r2_score
    from sklearn.model_selection import LeaveOneOut

    sub = tbl[conds + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    keep = [c for c in conds if sub[c].nunique() > 1]
    if len(sub) < 6 or len(keep) < 1:
        return None, np.nan
    X, y = sub[keep].to_numpy(float), sub[target].to_numpy(float)
    par = dict(n_estimators=120, max_depth=2, learning_rate=0.08,
               subsample=0.9, colsample_bytree=0.9, random_state=0, verbosity=0)

    pred = np.empty(len(y))
    for tr, te in LeaveOneOut().split(X):
        pred[te] = xgb.XGBRegressor(**par).fit(X[tr], y[tr]).predict(X[te])
    cv = r2_score(y, pred)

    rng = np.random.default_rng(seed)
    dm_all = xgb.DMatrix(X)
    SV = []
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        mdl = xgb.XGBRegressor(**{**par, "random_state": b}).fit(X[idx], y[idx])
        SV.append(mdl.get_booster().predict(dm_all, pred_contribs=True)[:, :-1])
    SV = np.stack(SV)                      # (B, n, k)
    mean_abs = np.abs(SV).mean(axis=(0, 1))
    order = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(11, 0.95 * len(keep) + 3.2))
    for row, j in enumerate(order):
        v = X[:, j].astype(float)
        norm = (v - v.min()) / (np.ptp(v) or 1)
        vals = SV[:, :, j].ravel()                       # B*n points
        cols = np.tile(norm, SV.shape[0])
        jit = rng.normal(0, 0.11, vals.size)
        ax.scatter(vals, row + jit, c=cols, cmap="coolwarm", s=11,
                   alpha=0.35, linewidth=0, vmin=0, vmax=1, zorder=2, rasterized=True)
        # the single full-data fit, drawn on top as larger markers
        ax.scatter(SV.mean(0)[:, j], row + rng.normal(0, 0.05, len(v)),
                   c=norm, cmap="coolwarm", s=70, edgecolor="k", linewidth=0.6,
                   vmin=0, vmax=1, zorder=4)
    ax.axvline(0, color="0.3", lw=1.3)
    ax.set_yticks(range(len(order)), [m(keep[j]) for j in order])
    ax.set_xlabel("SHAP value   (impact on model output)")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.set_label("condition value")
    t1 = f"{name}: {m(target)}      LOO $R^2$ = {cv:.2f}"
    t2 = (f"{len(sub)} cells, {SV.shape[0]} bootstrap refits "
          f"(pale cloud = bootstrap spread, solid = mean)")
    ax.set_title(t1 + chr(10) + t2, fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    import re as _re
    tag = _re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    f = figdir / f"{tag}_shap_{target}.png"
    fig.savefig(f); plt.close(fig)
    return f.name, cv

def fig_temp_matched(figdir):
    """JGNE: the SOC-matched _ref (25 C) vs _40T (40 C) cells."""
    G = {"30 $\\to$ 70 %": {25: ["SPEED_JGNE_cycle_77", "SPEED_JGNE_cycle_78"],
                            40: ["SPEED_JGNE_cycle_95", "SPEED_JGNE_cycle_96"]},
         "40 $\\to$ 100 %": {25: ["SPEED_JGNE_cycle_75", "SPEED_JGNE_cycle_76"],
                             40: ["SPEED_JGNE_cycle_97", "SPEED_JGNE_cycle_98"]}}
    def cell(c):
        d = pd.read_csv(JG / "cell_feature" / f"{c}.csv",
                        usecols=lambda x: x in ("cap_ocv_dis", "CU_time", "throughput_cum"))
        t = pd.to_datetime(d["CU_time"], errors="coerce")
        d = d.assign(t=t).dropna(subset=["t"]).sort_values("t")
        soh = (d.cap_ocv_dis / d.cap_ocv_dis.iloc[0]).to_numpy()
        thr = (d.throughput_cum / 3600).to_numpy()
        return thr, soh
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    rates = {}
    for k, (win, arms) in enumerate(G.items()):
        a = axes[k]
        for T, cells in arms.items():
            for i, c in enumerate(cells):
                thr, soh = cell(c)
                a.plot(thr, soh, color=C25 if T == 25 else C40, lw=2,
                       marker="o", ms=5, label=f"$T$ = {T} $^\\circ$C" if i == 0 else None)
                rates.setdefault((win, T), []).append((1 - soh[-1]) / thr[-1] * 1e4)
        a.set_title(f"SOC window {win}"); a.set_xlabel(m("throughput_cum"))
        a.axhline(1.0, color="0.6", lw=1, ls="--", zorder=0)
        a.legend(frameon=False)
    axes[0].set_ylabel(m("SOH"))
    a = axes[2]
    wins = list(G)
    for T, col in ((25, C25), (40, C40)):
        ys = [np.mean(rates[(w, T)]) for w in wins]
        a.plot(range(len(wins)), ys, "-o", ms=12, lw=2.5, color=col,
               label=f"$T$ = {T} $^\\circ$C")
        for i, w in enumerate(wins):
            a.scatter([i] * len(rates[(w, T)]), rates[(w, T)], color=col,
                      s=55, edgecolor="k", zorder=4)
    a.set_xticks(range(len(wins)), wins)
    a.set_xlabel("SOC window"); a.set_ylabel(m("fade_per_100Ah"))
    a.set_title("fade rate, normalised by throughput"); a.legend(frameon=False)
    a.set_xlim(-0.35, len(wins) - 0.65)
    fig.suptitle("JGNE: temperature effect is confined to the high-SOC window "
                 "(SOC-matched cells, 2 per arm)")
    fig.tight_layout()
    f = figdir / "jgne_temperature_matched.png"
    fig.savefig(f); plt.close(fig)
    return f.name, rates

def build_cell_table(spec, traj, cond):
    rows = []
    for cell, d in traj.items():
        r = {"cell": cell, "n_checkups": len(d["soh"]),
             "weeks": d["weeks"][-1], "soh_end": d["soh"][-1]}
        if np.isfinite(d["thr"]).any() and d["thr"][-1] > 0:
            r["throughput_cum"] = d["thr"][-1]
            r["fade_per_100Ah"] = (1 - d["soh"][-1]) / d["thr"][-1] * 1e4
            x = np.sqrt(d["thr"]); y = 1 - d["soh"]
            msk = np.isfinite(x) & np.isfinite(y)
            if msk.sum() >= 3:
                r["k_sqrt_Ah"] = float(np.dot(x[msk], y[msk]) / np.dot(x[msk], x[msk]))
        if d["weeks"][-1] > 0:
            r["fade_per_week"] = (1 - d["soh"][-1]) / d["weeks"][-1] * 100
        rows.append(r)
    t = pd.DataFrame(rows).set_index("cell")
    return t.join(cond, how="left").reset_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("./report_jgne_lwhk"))
    args = ap.parse_args()
    figdir = args.dest / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made = {}

    for name, spec in DATASETS.items():
        traj = load_traj(spec)
        cond = load_cond(spec)
        print(f"{name}: {len(traj)} cells")
        tbl = build_cell_table(spec, traj, cond)
        tbl.to_csv(args.dest / f"{name.lower()}_cell_summary.csv", index=False)

        gb = "pause_h" if name == "JGNE" else "fec"
        made[f"{name}_traj"] = fig_trajectory(name, spec, traj, cond, gb, figdir)
        if name == "JGNE":
            made["JGNE_traj_T"] = fig_trajectory(name, spec, traj, cond, "temp_C", figdir)
        else:
            made["LWHK_facet"] = fig_facet(name, traj, cond, "fec", "rest_min", figdir)

        conds = [c for c in spec["conditions"]
                 if c in tbl.columns and pd.to_numeric(tbl[c], errors="coerce").nunique() > 1]
        feats = [f for f in ("fade_per_100Ah", "fade_per_week", "k_sqrt_Ah",
                             "soh_end", "weeks", "throughput_cum")
                 if f in tbl.columns and tbl[f].notna().sum() >= 5]
        made[f"{name}_heat"] = fig_heatmap(name, tbl, conds, feats, figdir)
        for tgt in ("fade_per_100Ah", "fade_per_week"):
            if tgt in tbl.columns:
                fn, cv = fig_shap(name, tbl, conds, tgt, figdir)
                if fn:
                    made[f"{name}_shap_{tgt}"] = (fn, cv)
        print(f"  conditions {conds}   targets {feats}")

    made["JGNE_temp"], rates = fig_temp_matched(figdir)
    print("\nfigures:")
    for k, v in made.items():
        print("  ", k, v)
    return args.dest, made, rates

if __name__ == "__main__":
    main()


# ------------------------------------------------- SHAP on the curve features
# Targets here are the dQ/dV and Q(V) descriptors themselves, interpolated onto a
# common SoH so that every cell contributes one value and cells are compared at
# equal degradation rather than at equal time.
SOH_TABLES = {
    "JGNE": dict(path=JG / "jgne_features_at_soh995.csv", soh=0.995,
                 conds=["temp_C", "pause_h", "has_pulse", "low_soc", "soc_mean"]),
    "LWHK": dict(path=JG / "lwhk_c2_features_at_soh98.csv", soh=0.98,
                 conds=["fec", "rest_min"]),
}
CURVE_TARGETS = [
    "mean_dQ_c", "var_dQ_c", "mean_dQ_d", "var_dQ_d",
    "mean_d_dqdv_m_c", "var_d_dqdv_m_c", "mean_d_dqdv_l_c", "mean_d_dqdv_h_c",
    "mean_d_dqdv_m_d", "var_d_dqdv_m_d", "mean_d_dqdv_l_d", "mean_d_dqdv_h_d",
    "mean_d_Qt", "var_d_Qt",
]

def curve_shap(name, figdir, max_targets=8):
    spec = SOH_TABLES[name]
    if not Path(spec["path"]).exists():
        print(f"[--] {name}: {spec['path']} missing"); return []
    tbl = pd.read_csv(spec["path"])
    conds = [c for c in spec["conds"]
             if c in tbl.columns and pd.to_numeric(tbl[c], errors="coerce").nunique() > 1]
    made = []
    for tgt in CURVE_TARGETS:
        if tgt not in tbl.columns:
            continue
        if pd.to_numeric(tbl[tgt], errors="coerce").notna().sum() < 6:
            continue
        fn, cv = fig_shap(f"{name}  (SoH {spec['soh']})", tbl, conds, tgt, figdir,
                          n_boot=150)
        if fn:
            made.append((tgt, fn, cv))
        if len(made) >= max_targets:
            break
    return made

def build_curve_figures(dest):
    figdir = Path(dest) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in SOH_TABLES:
        res = curve_shap(name, figdir)
        out[name] = res
        print(f"\n{name}: SHAP on curve features")
        for tgt, fn, cv in res:
            flag = "" if cv > 0 else "   <- not predictive"
            print(f"   {tgt:18} LOO R2={cv:+.2f}  {fn}{flag}")
    rows = [{"dataset": k, "target": t, "loo_r2": cv, "figure": f}
            for k, v in out.items() for t, f, cv in v]
    pd.DataFrame(rows).to_csv(Path(dest) / "curve_shap_scores.csv", index=False)
    return out

if __name__ == "__main__" and False:
    pass
