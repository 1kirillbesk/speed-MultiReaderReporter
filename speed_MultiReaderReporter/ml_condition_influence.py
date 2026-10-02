# speed_MultiReaderReporter/ml_condition_influence.py
"""Machine-learning attribution of ageing to the experimental conditions.

One row per CELL (not per checkup), so the sample size is the number of cells.
Targets are whole-trajectory degradation rates rather than a single SoH slice:

    fade_per_100Ah   (1 - SoH_end) / throughput * 1e4      [% per 100 Ah]
    fade_per_week    (1 - SoH_end) / weeks * 100           [% per week]
    k_sqrt_Ah        b in  SoH = 1 - b*sqrt(Ah)            [1/sqrt(Ah)]
    k_lin_Ah         b in  SoH = 1 - b*Ah                  [1/Ah]
    thr_to_<target>  throughput needed to reach a SoH      [Ah]
    weeks_to_<target> time needed to reach a SoH           [weeks]

WHY THE STATISTICS MATTER HERE. These datasets have 8-29 cells against 2-6
conditions. Any flexible model fits that perfectly, so a reported R2 or a SHAP
ranking on its own is meaningless. This script therefore reports, for each
(target, model):

  cv_r2        cross-validated R2 on held-out CELLS (LOO when n <= 20, else
               repeated 5-fold). Negative means "worse than predicting the mean".
  p_perm       permutation test: the label vector is shuffled `--permutations`
               times and cv_r2 recomputed, giving an empirical null. p is the
               fraction of shuffles reaching the observed cv_r2. This is the
               number to quote - it is valid at small n and needs no normality
               assumption.
  importance   permutation importance on held-out folds, repeated, mean +/- sd.
               Model-agnostic and does not inherit tree-impurity bias.
  direction    sign of the Spearman correlation between condition and target,
               so an "important" condition also has a readable direction.

Models: Ridge (linear baseline, standardised), RandomForest, and optionally
GradientBoosting. With this few cells the linear model is often the honest one;
if the trees do not beat it, say so in the paper.

Usage:
    python speed_MultiReaderReporter/ml_condition_influence.py --dataset jgne_temp
    python speed_MultiReaderReporter/ml_condition_influence.py --dataset lwhk --permutations 500
    python speed_MultiReaderReporter/ml_condition_influence.py --dataset all
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- datasets
# feature_dir / cond_csv are relative to out_dir. `cells` optionally restricts to
# a condition-matched subset (the temperature comparisons).
DATASETS = {
    "jgne_temp": dict(
        out_dir="E:/download/jgne/out_jgne", feature_dir="cell_feature",
        cond_csv="condition_overview_homocomp_jgne.csv", soh_col="cap_ocv_dis",
        pattern="SPEED_JGNE_*", conditions=["temp_C", "soc_mean"],
        cells=["SPEED_JGNE_cycle_75", "SPEED_JGNE_cycle_76", "SPEED_JGNE_cycle_77",
               "SPEED_JGNE_cycle_78", "SPEED_JGNE_cycle_95", "SPEED_JGNE_cycle_96",
               "SPEED_JGNE_cycle_97", "SPEED_JGNE_cycle_98"],
        note="SOC-matched _ref vs _40T cells; the only controlled JGNE temperature set",
    ),
    "jgne_pause": dict(
        out_dir="E:/download/jgne/out_jgne", feature_dir="cell_feature",
        cond_csv="condition_overview_homocomp_jgne.csv", soh_col="cap_ocv_dis",
        pattern="SPEED_JGNE_*", conditions=["pause_h", "has_pulse", "low_soc", "soc_mean"],
        cells=None,
        note="all JGNE cells carrying a pause token",
    ),
    "lwhk": dict(
        out_dir="E:/download/jgne/out_jgne", feature_dir="cell_feature_cu_c2",
        cond_csv="condition_overview_SPEED_LWHK.csv", soh_col="cap_dis",
        pattern="SPEED_LWHK_*", conditions=["fec", "rest_min"],
        cells=None, thr_exclude="rul_eka_CU",
        note="C/2 discharge capacity from rul_eka_CU",
    ),
    "bald_temp": dict(
        out_dir="E:/download/BALD/out_bald_homocomp", feature_dir="cell_feature",
        cond_csv="condition_overview_BALD.csv", soh_col="cap_ocv_dis",
        pattern="*", conditions=["temp_C", "soc_mean", "pause_h"],
        cells=None,
        note="condition-matched 25 vs 40 C; SoH from the OCV CHARGE sweep",
    ),
    "bald_cyc": dict(
        out_dir="E:/download/BALD/out_bald", feature_dir="cell_feature",
        cond_csv="condition_overview_BALD_35E.csv", soh_col="cap_ocv_dis",
        pattern="BALD_35E_*", conditions=["c_rate", "soc_mean", "dyn", "sdod", "scur"],
        cells=None,
        note="inhomoSAM cycling study, all 25 C",
    ),
}

# ---------------------------------------------------------------- per-cell table

def throughput_from_report(out_dir: Path, cell: str, times: pd.Series,
                           exclude: str | None) -> np.ndarray | None:
    """Cumulative cycling throughput [Ah] at each checkup, summed from
    <cell>/cycling/report.csv. Used when the pipeline's throughput_cum column is
    absent or zero, which is the case for the LWHK C/2 dataset. `exclude` drops
    the checkup's own rows (rul_eka_CU) so only cycling charge is counted."""
    rep = out_dir / cell / "cycling" / "report.csv"
    if not rep.exists():
        rep = out_dir / cell / "total" / "report.csv"
    if not rep.exists():
        return None
    r = pd.read_csv(rep)
    r = r[r["program"].astype(str) != "TOTAL"]
    if exclude:
        r = r[~r["program"].astype(str).str.contains(exclude, case=False, na=False)]
    if r.empty or "throughput_Ah" not in r.columns:
        return None
    r = r.assign(end=pd.to_datetime(r["end_time"], errors="coerce")).dropna(subset=["end"])
    r = r.sort_values("end")
    r["cum"] = r["throughput_Ah"].cumsum()
    return np.array([float(r.loc[r["end"] <= t, "cum"].max())
                     if (r["end"] <= t).any() else 0.0 for t in times])

def cell_targets(csv_path: Path, soh_col: str, soh_marks=(0.95, 0.90),
                 out_dir: Path | None = None, thr_exclude: str | None = None) -> dict | None:
    d = pd.read_csv(csv_path, usecols=lambda c: c in (soh_col, "CU_time", "throughput_cum"))
    if soh_col not in d.columns or "CU_time" not in d.columns or len(d) < 3:
        return None
    t = pd.to_datetime(d["CU_time"], errors="coerce")
    d = d.assign(t=t).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    if len(d) < 3:
        return None

    soh = pd.to_numeric(d[soh_col], errors="coerce")
    soh = soh / soh.iloc[0]
    weeks = (d["t"] - d["t"].iloc[0]).dt.total_seconds() / (7 * 24 * 3600)
    thr = (pd.to_numeric(d.get("throughput_cum", pd.Series(np.nan, index=d.index)),
                         errors="coerce") / 3600.0)
    if (thr.isna().all() or float(np.nanmax(thr.to_numpy(dtype=float), initial=0)) <= 0)             and out_dir is not None:
        alt = throughput_from_report(out_dir, csv_path.stem, d["t"], thr_exclude)
        if alt is not None and np.nanmax(alt) > 0:
            thr = pd.Series(alt, index=d.index)

    out = {"n_checkups": len(d), "weeks": float(weeks.iloc[-1]),
           "soh_end": float(soh.iloc[-1]), "soh_min": float(soh.min())}
    if weeks.iloc[-1] > 0:
        out["fade_per_week"] = float((1 - soh.iloc[-1]) / weeks.iloc[-1] * 100)
    if thr.notna().any() and thr.iloc[-1] > 0:
        out["throughput_Ah"] = float(thr.iloc[-1])
        out["fade_per_100Ah"] = float((1 - soh.iloc[-1]) / thr.iloc[-1] * 1e4)
        # rate constants: SoH = 1 - b*sqrt(Ah) and SoH = 1 - b*Ah, through the origin
        x_s, x_l, y = np.sqrt(thr.to_numpy()), thr.to_numpy(), (1 - soh).to_numpy()
        m = np.isfinite(x_s) & np.isfinite(y)
        if m.sum() >= 3:
            out["k_sqrt_Ah"] = float(np.dot(x_s[m], y[m]) / np.dot(x_s[m], x_s[m]))
            out["k_lin_Ah"] = float(np.dot(x_l[m], y[m]) / np.dot(x_l[m], x_l[m]))

    # how far the cell had to go to reach a given SoH (NaN if never reached)
    for mark in soh_marks:
        tag = f"{mark:.2f}".replace("0.", "")
        if soh.min() <= mark <= soh.max():
            o = np.argsort(soh.to_numpy())
            out[f"weeks_to_{tag}"] = float(np.interp(mark, soh.to_numpy()[o], weeks.to_numpy()[o]))
            if thr.notna().any():
                out[f"thr_to_{tag}"] = float(np.interp(mark, soh.to_numpy()[o], thr.to_numpy()[o]))
    return out

def build_table(spec: dict) -> pd.DataFrame:
    out_dir = Path(spec["out_dir"])
    fdir = out_dir / spec["feature_dir"]
    rows = []
    for p in sorted(fdir.glob(f"{spec['pattern']}.csv")):
        if spec.get("cells") and p.stem not in spec["cells"]:
            continue
        t = cell_targets(p, spec["soh_col"], out_dir=out_dir,
                         thr_exclude=spec.get("thr_exclude"))
        if t is None:
            continue
        rows.append({"cell": p.stem, **t})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    cond = pd.read_csv(out_dir / spec["cond_csv"])
    prefer = next((c for c in ("c_rate", "fec", "pause_h", "soc_start") if c in cond.columns), None)
    cond = cond.copy()
    cond["_pref"] = cond[prefer].notna().astype(int) if prefer else 0
    chosen = (cond.sort_values(["_pref", "n_files"], ascending=[False, False])
                  .drop_duplicates("cell", keep="first").drop(columns="_pref"))
    flags = [c for c in cond.columns
             if c not in ("cell", "label", "n_files", "first_test", "last_test", "_pref")
             and pd.api.types.is_numeric_dtype(cond[c])
             and set(pd.unique(cond[c].dropna())) <= {0, 1}]
    if flags:
        anyf = cond.groupby("cell")[flags].max()
        chosen = chosen.set_index("cell")
        chosen[flags] = anyf.reindex(chosen.index)[flags]
        chosen = chosen.reset_index()
    return df.merge(chosen, on="cell", how="left", suffixes=("", "_cond"))

# ---------------------------------------------------------------- modelling

def make_models(seed=0):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "ridge": make_pipeline(StandardScaler(),
                               RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "rf": RandomForestRegressor(n_estimators=200, min_samples_leaf=1,
                                    random_state=seed, n_jobs=1),
        "gbr": GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                         learning_rate=0.05, random_state=seed),
    }

def cv_splitter(n: int, seed=0):
    from sklearn.model_selection import LeaveOneOut, RepeatedKFold
    if n <= 20:
        return LeaveOneOut(), "LOO"
    return RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed), "5-fold x5"

def cv_score(model, X, y, cv) -> float:
    """Pooled out-of-fold R2 - computed on all held-out predictions at once, not
    averaged per fold, which is undefined for LOO (one point per fold)."""
    from sklearn.base import clone
    from sklearn.metrics import r2_score
    preds, truth = [], []
    for tr, te in cv.split(X):
        m = clone(model).fit(X[tr], y[tr])
        preds.append(m.predict(X[te])); truth.append(y[te])
    return float(r2_score(np.concatenate(truth), np.concatenate(preds)))

def permutation_pvalue(model, X, y, cv, observed, n_perm, rng) -> float:
    """Fraction of label shuffles reaching the observed cv_r2 (add-one corrected)."""
    hits = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if cv_score(model, X, yp, cv) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)

def perm_importance(model, X, y, cols, cv, n_repeats, rng):
    """Permutation importance measured on held-out folds, so it reflects
    generalisation rather than in-sample fit."""
    from sklearn.base import clone
    from sklearn.metrics import r2_score
    base_pred = np.zeros_like(y, dtype=float)
    fitted = []
    for tr, te in cv.split(X):
        m = clone(model).fit(X[tr], y[tr])
        base_pred[te] = m.predict(X[te]); fitted.append((m, te))
    base = r2_score(y, base_pred)
    out = {}
    for j, col in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            pred = np.zeros_like(y, dtype=float)
            for m, te in fitted:
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j]) if len(te) > 1 else Xp[:, j]
                pred[te] = m.predict(Xp)
            drops.append(base - r2_score(y, pred))
        out[col] = (float(np.mean(drops)), float(np.std(drops)))
    # LOO leaves one sample per fold, so within-fold shuffling is a no-op; fall
    # back to shuffling the column globally and refitting-free scoring.
    if all(abs(v[0]) < 1e-12 for v in out.values()):
        for j, col in enumerate(cols):
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                pred = np.zeros_like(y, dtype=float)
                for m, te in fitted:
                    pred[te] = m.predict(Xp[te])
                drops.append(base - r2_score(y, pred))
            out[col] = (float(np.mean(drops)), float(np.std(drops)))
    return out, base

# ---------------------------------------------------------------- driver

def analyse(name: str, spec: dict, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import spearmanr

    df = build_table(spec)
    if df.empty:
        print(f"[{name}] no usable cells"); return pd.DataFrame(), pd.DataFrame()

    conds = [c for c in spec["conditions"]
             if c in df.columns
             and pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) >= 2]
    dropped = [c for c in spec["conditions"] if c not in conds]
    targets = [t for t in ("fade_per_100Ah", "fade_per_week", "k_sqrt_Ah", "k_lin_Ah",
                           "thr_to_95", "weeks_to_95", "thr_to_90", "weeks_to_90")
               if t in df.columns and pd.to_numeric(df[t], errors="coerce").notna().sum() >= 6]

    print(f"\n{'='*74}\n{name}   {spec['note']}\n{'='*74}")
    print(f"cells      : {len(df)}")
    print(f"conditions : {conds}" + (f"   (dropped, constant: {dropped})" if dropped else ""))
    print(f"targets    : {targets}")
    if len(df) < 4 * len(conds):
        print(f"[WARN] {len(df)} cells vs {len(conds)} conditions - severely "
              f"underpowered; quote p_perm, not R2, and treat as exploratory.")

    rng = np.random.default_rng(args.seed)
    models = make_models(args.seed)
    score_rows, imp_rows = [], []

    for tgt in targets:
        sub = df[["cell"] + conds + [tgt]].apply(
            lambda c: pd.to_numeric(c, errors="coerce") if c.name != "cell" else c)
        sub = sub.dropna()
        if len(sub) < 6:
            continue
        X = sub[conds].to_numpy(float)
        y = sub[tgt].to_numpy(float)
        cv, cv_name = cv_splitter(len(sub), args.seed)

        for mname, model in models.items():
            if mname == "gbr" and not args.with_gbr:
                continue
            r2 = cv_score(model, X, y, cv)
            p = permutation_pvalue(model, X, y, cv, r2, args.permutations, rng)
            imps, _ = perm_importance(model, X, y, conds, cv, args.imp_repeats, rng)
            score_rows.append({"dataset": name, "target": tgt, "model": mname,
                               "n_cells": len(sub), "cv": cv_name,
                               "cv_r2": r2, "p_perm": p,
                               "significant": int(p < 0.05 and r2 > 0)})
            for c, (m_, s_) in imps.items():
                rho = spearmanr(sub[c], y).statistic if sub[c].nunique() > 1 else np.nan
                imp_rows.append({"dataset": name, "target": tgt, "model": mname,
                                 "condition": c, "importance": m_, "importance_sd": s_,
                                 "spearman_r": rho,
                                 "direction": ("+" if rho > 0 else "-") if pd.notna(rho) else "?",
                                 "cv_r2": r2, "p_perm": p})
    return pd.DataFrame(score_rows), pd.DataFrame(imp_rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", choices=list(DATASETS) + ["all"])
    ap.add_argument("--permutations", type=int, default=200,
                    help="label shuffles for the permutation test (default 200; "
                         "use >=1000 for a final paper number)")
    ap.add_argument("--imp-repeats", type=int, default=20)
    ap.add_argument("--with-gbr", action="store_true", help="also fit GradientBoosting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dest", type=Path, default=Path("./ml_condition_influence"))
    args = ap.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    all_s, all_i = [], []
    for n in names:
        s, i = analyse(n, DATASETS[n], args)
        if not s.empty:
            all_s.append(s); all_i.append(i)
            best = s.sort_values("cv_r2", ascending=False).head(6)
            print("\n  best models:")
            print(best[["target", "model", "n_cells", "cv_r2", "p_perm", "significant"]]
                  .to_string(index=False))
            sig = s[s.significant == 1]
            if sig.empty:
                print("\n  NOTHING significant: no model beat its own mean on held-out "
                      "cells more often than chance.")
            else:
                print(f"\n  {len(sig)} significant (p<0.05 and cv_r2>0):")
                key = i.merge(sig[["target", "model"]], on=["target", "model"])
                key = key.sort_values("importance", ascending=False)
                print(key[["target", "model", "condition", "importance",
                           "importance_sd", "direction", "p_perm"]].head(10).to_string(index=False))

    if not all_s:
        print("\n[ERROR] nothing analysed"); return
    args.dest.mkdir(parents=True, exist_ok=True)
    S = pd.concat(all_s, ignore_index=True); I = pd.concat(all_i, ignore_index=True)
    S.to_csv(args.dest / "ml_model_scores.csv", index=False)
    I.to_csv(args.dest / "ml_permutation_importance.csv", index=False)
    print(f"\n[OK] {len(S)} model fits -> {args.dest / 'ml_model_scores.csv'}")
    print(f"[OK] {len(I)} importances -> {args.dest / 'ml_permutation_importance.csv'}")

if __name__ == "__main__":
    main()
