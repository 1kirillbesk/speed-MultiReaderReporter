# speed_MultiReaderReporter/shap_condition_influence.py
"""XGBoost + SHAP: which experimental condition drives which feature.

Consumes <family>_features_at_soh<NN>.csv written by
analyze_condition_influence_{jgne,lwhk}.py - one row per cell, conditions and
features already interpolated onto a common SoH - then, for every feature:

    X = the condition columns        y = that feature
    fit an XGBRegressor, score it by leave-one-out CV,
    explain it with shap.TreeExplainer

SAMPLE SIZE WARNING. These tables have ~23 (JGNE) and ~13 (LWHK) rows against
up to 9 conditions. A gradient-boosted tree will fit that perfectly and the SHAP
values will faithfully explain a model that has memorised the data. The guard is
`cv_r2`: leave-one-out R2 on held-out cells. Where cv_r2 <= 0 the model predicts
worse than the mean and its SHAP ranking carries no evidence - the report marks
those rows `predictive = 0`. Read the SHAP importances only for targets that
clear the bar, and treat the whole thing as hypothesis-generating.

Usage:
    python speed_MultiReaderReporter/shap_condition_influence.py --family jgne
    python speed_MultiReaderReporter/shap_condition_influence.py --family lwhk --target-soh 0.995
    python speed_MultiReaderReporter/shap_condition_influence.py --family jgne --plots
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path("E:/download/jgne/out_jgne")

FAMILY_CONDITIONS = {
    "jgne": ["soc_start", "soc_end", "pause_h", "has_pulse", "low_soc",
             "is_ref", "temp_C", "c_rate_chg", "c_rate_dchg"],
    "lwhk": ["fec", "rest_min", "ref"],
}

# Bookkeeping columns in the features_at_soh table that are not model targets.
NOT_A_TARGET = {"cell", "label", "n_checkups", "soh_min", "soh_max",
                "n_files", "first_test", "last_test", "has_feature_csv",
                "pause_raw", "pause_h_2", "test_date"}

def load_table(out_dir: Path, family: str, target_soh: float,
               prefix: str | None = None) -> tuple[pd.DataFrame, Path]:
    tag = f"{target_soh:.3f}".replace("0.", "").rstrip("0") or "0"
    path = out_dir / f"{prefix or family}_features_at_soh{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run analyze_condition_influence_{family}.py "
            f"--target-soh {target_soh} first"
        )
    return pd.read_csv(path), path

def usable_conditions(df: pd.DataFrame, conditions: list[str]) -> list[str]:
    """Conditions that are present, numeric and actually vary."""
    out = []
    for c in conditions:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 3 and s.nunique(dropna=True) >= 2:
            out.append(c)
    return out

def target_columns(df: pd.DataFrame, conditions: list[str]) -> list[str]:
    return [c for c in df.columns
            if c not in NOT_A_TARGET and c not in conditions
            and pd.api.types.is_numeric_dtype(df[c])
            and pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) >= 3]

def fit_and_explain(X: pd.DataFrame, y: pd.Series, seed: int = 0):
    """Returns (cv_r2, train_r2, mean_abs_shap per column, model, shap values).

    SHAP comes from XGBoost's own pred_contribs (exact TreeSHAP) rather than
    shap.TreeExplainer: shap 0.49 cannot parse the tree dump emitted by
    xgboost 3.2 and raises "could not convert string to float". Same algorithm,
    and the contributions still sum to the prediction.
    """
    import xgboost as xgb
    from sklearn.metrics import r2_score
    from sklearn.model_selection import LeaveOneOut

    # Deliberately small: the tables have tens of rows, not thousands.
    params = dict(n_estimators=120, max_depth=2, learning_rate=0.08,
                  subsample=0.9, colsample_bytree=0.9,
                  reg_lambda=1.0, random_state=seed, verbosity=0)

    # leave-one-out CV - the only honest score at this sample size
    preds = np.full(len(y), np.nan)
    for tr, te in LeaveOneOut().split(X):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        preds[te] = m.predict(X.iloc[te])
    cv_r2 = float(r2_score(y, preds))

    full = xgb.XGBRegressor(**params).fit(X, y)
    train_r2 = float(r2_score(y, full.predict(X)))

    contribs = full.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    sv = contribs[:, :-1]          # last column is the bias / expected value
    mean_abs = dict(zip(X.columns, np.abs(sv).mean(axis=0)))
    return cv_r2, train_r2, mean_abs, full, sv

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILY_CONDITIONS), required=True)
    ap.add_argument("--target-soh", type=float, default=0.995)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dest", type=Path, default=None, help="output folder (default: --out-dir)")
    ap.add_argument("--min-cv-r2", type=float, default=0.0,
                    help="a target is flagged predictive when cv_r2 exceeds this (default: 0.0)")
    ap.add_argument("--conditions", default=None,
                    help="comma separated subset of conditions to model. Rows need "
                         "EVERY listed condition present, so dropping a sparse one "
                         "(e.g. pause_h, undefined for the _40T/_ref/_1C2C cells) "
                         "keeps far more cells in the fit.")
    ap.add_argument("--prefix", default=None,
                    help="input/output filename prefix (default: the family name); "
                         "use e.g. lwhk_c2 for the C/2 discharge table")
    ap.add_argument("--plots", action="store_true",
                    help="write a SHAP beeswarm per predictive target")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    df, src = load_table(args.out_dir, args.family, args.target_soh, args.prefix)
    wanted = ([c.strip() for c in args.conditions.split(",") if c.strip()]
              if args.conditions else FAMILY_CONDITIONS[args.family])
    conditions = usable_conditions(df, wanted)
    targets = target_columns(df, FAMILY_CONDITIONS[args.family])

    print(f"source     : {src}")
    print(f"cells      : {len(df)}")
    print(f"conditions : {conditions}")
    print(f"targets    : {len(targets)} feature(s)")
    if len(df) < 2 * len(conditions):
        print(f"[WARN] {len(df)} cells vs {len(conditions)} conditions - "
              f"far too few rows for a tree model to generalise; "
              f"treat every result below as a hypothesis, not a finding.")
    print()

    X_all = df[conditions].apply(pd.to_numeric, errors="coerce")

    rows, per_target = [], []
    for tgt in targets:
        y_all = pd.to_numeric(df[tgt], errors="coerce")
        m = X_all.notna().all(axis=1) & y_all.notna()
        X, y = X_all[m], y_all[m]
        if len(X) < 5 or y.nunique() < 3:
            continue
        try:
            cv_r2, train_r2, mean_abs, model, sv = fit_and_explain(X, y)
        except Exception as e:
            print(f"[WARN] {tgt}: {e}")
            continue

        predictive = int(cv_r2 > args.min_cv_r2)
        per_target.append({"feature": tgt, "n_cells": len(X),
                           "cv_r2": cv_r2, "train_r2": train_r2,
                           "predictive": predictive})
        for cond, val in mean_abs.items():
            rows.append({"feature": tgt, "condition": cond,
                         "mean_abs_shap": float(val),
                         "cv_r2": cv_r2, "train_r2": train_r2,
                         "n_cells": len(X), "predictive": predictive})

        if args.plots and predictive:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import shap
                dest_p = (args.dest or args.out_dir) / f"shap_{args.prefix or args.family}"
                dest_p.mkdir(parents=True, exist_ok=True)
                shap.summary_plot(sv, X, show=False)
                plt.title(f"{args.family.upper()} {tgt} (cv R2={cv_r2:.2f})")
                plt.tight_layout()
                plt.savefig(dest_p / f"{tgt}.png", dpi=120)
                plt.close("all")
            except Exception as e:
                print(f"[WARN] plot for {tgt} failed: {e}")

    if not rows:
        print("[ERROR] nothing could be modelled (too few usable rows).")
        return

    imp = pd.DataFrame(rows)
    scores = pd.DataFrame(per_target).sort_values("cv_r2", ascending=False)

    dest = args.dest or args.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    tag = f"{args.target_soh:.3f}".replace("0.", "").rstrip("0") or "0"
    pref = args.prefix or args.family
    imp_path = dest / f"{pref}_shap_importance_soh{tag}.csv"
    sc_path = dest / f"{pref}_shap_model_scores_soh{tag}.csv"
    imp.sort_values(["predictive", "mean_abs_shap"], ascending=False).to_csv(imp_path, index=False)
    scores.to_csv(sc_path, index=False)

    print(f"[OK] {len(imp):4} row(s) -> {imp_path}")
    print(f"[OK] {len(scores):4} row(s) -> {sc_path}")

    good = scores[scores.predictive == 1]
    print(f"\nmodels that beat the mean on held-out cells: {len(good)}/{len(scores)}")
    if not good.empty:
        print(good.head(12).to_string(index=False))
        print("\nSHAP importance for those targets:")
        keep = imp[imp.predictive == 1].sort_values("mean_abs_shap", ascending=False)
        print(keep.head(20)[["feature", "condition", "mean_abs_shap", "cv_r2", "n_cells"]]
              .to_string(index=False))
    else:
        print("None. No condition set predicts any feature better than its own mean "
              "on held-out cells, so the SHAP values describe memorisation only.")
        print("\nbest (still non-predictive) models:")
        print(scores.head(8).to_string(index=False))

if __name__ == "__main__":
    main()
