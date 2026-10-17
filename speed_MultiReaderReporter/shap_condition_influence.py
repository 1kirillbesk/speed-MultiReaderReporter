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
    "jgne": ["soc_start", "soc_end", "soc_mean", "soc_dod", "pause_h", "has_pulse", "low_soc", "temp_C", "c_rate_chg", "c_rate_dchg"],
    "lwhk": ["fec", "rest_min", "ref"],
    "bald": ["soc_start", "soc_end", "soc_mean", "soc_dod", "c_rate", "temp_C", "dyn", "sdod", "scur",
             "pau", "short", "pulsecyc", "pause_h", "dod", "n_cyc"],
}

# Bookkeeping columns in the features_at_soh table that are not model targets.
NOT_A_TARGET = {"cell", "label", "n_checkups", "soh_min", "soh_max",
                 "n_files", "first_test", "last_test", "has_feature_csv",
                 "pause_raw", "pause_h_2", "test_date"}

FEATURE_LABELS = {
    "fade_per_100Ah": "Capacity fade [% / 100 Ah]",
    "fade_per_week": "Capacity fade [% / week]",
    "soh_end": "Endpoint capacity retention [fraction]",
    "k_sqrt_Ah": "Square-root throughput fade coefficient [Ah^-1/2]",
    "cap_ocv_dis": "Slow OCV discharge capacity (0.1 C) [Ah]",
    "cap_fast_dis": "Fast discharge capacity (0.5 C) [Ah]",
    "cap_fast_cha": "Fast charge capacity (0.5 C) [Ah]",
    "soh_fast_dis": "Fast discharge capacity retention (own baseline) [fraction]",
    "cap_ocv_minus_fast_dis_Ah": "Slow OCV − fast discharge capacity [Ah]",
    "soh_ocv_minus_fast_dis": "OCV − fast discharge retention (each own baseline) [fraction]",
    "dva_fast_cha_max_width_Ah": "Fast charge DVA main peak width (half-prominence) [Ah]",
    "dva_fast_cha_min_width_Ah": "Fast charge DVA main valley width (half-prominence) [Ah]",
    "dva_fast_dis_max_width_Ah": "Fast discharge DVA main peak width (half-prominence) [Ah]",
    "dva_fast_dis_min_width_Ah": "Fast discharge DVA main valley width (half-prominence) [Ah]",
}
CONDITION_LABELS = {
    "soc_start": "Starting state of charge [%]",
    "soc_end": "Ending state of charge [%]",
    "soc_mean": "Mean state of charge [%]",
    "soc_dod": "SOC swing [% points]",
    "pause_h": "Pause duration [h]",
    "has_pulse": "Pulse protocol (0 = no, 1 = yes)",
    "low_soc": "Low-SOC protocol (0 = no, 1 = yes)",
    "temp_C": "Temperature [°C]",
    "c_rate_chg": "Charge rate [C]",
    "c_rate_dchg": "Discharge rate [C]",
}


def feature_label(name: str) -> str:
    # E.g. mean_d_dqdv_m_c -> <Delta(dQ/dV)>, mid voltage, charge.
    # The delta is relative to the cell's first checkup, not a derivative in time.
    parts = name.split("_")
    if (len(parts) == 5 and parts[:3] in (["mean", "d", "dqdv"], ["var", "d", "dqdv"])
            and parts[3] in {"l", "m", "h"} and parts[4] in {"c", "d"}):
        window = {"l": "low", "m": "mid", "h": "high"}[parts[3]]
        direction = {"c": "charge", "d": "discharge"}[parts[4]]
        expr = (r"$\langle\Delta(dQ/dV)\rangle$" if parts[0] == "mean"
                else r"$\mathrm{Var}[\Delta(dQ/dV)]$")
        units = "Ah/V" if parts[0] == "mean" else "(Ah/V)^2"
        return f"{expr}, {window}-voltage {direction} [{units}]"
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def feature_title(name: str) -> str:
    """Concise figure title; full definition and units remain on the axes."""
    short = {
        "fade_per_100Ah": "Fade per 100 Ah",
        "fade_per_week": "Fade per week",
        "soh_end": "Endpoint retention",
        "cap_ocv_dis": "Slow-discharge capacity",
        "cap_fast_dis": "Fast-discharge capacity",
        "cap_fast_cha": "Fast-charge capacity",
        "soh_fast_dis": "Fast-discharge retention",
        "cap_ocv_minus_fast_dis_Ah": "Slow − fast discharge capacity",
        "soh_ocv_minus_fast_dis": "Slow − fast discharge retention",
    }
    if name in short:
        return short[name]
    parts = name.split("_")
    if (len(parts) == 6 and parts[:2] == ["dva", "fast"]
            and parts[2] in {"cha", "dis"} and parts[3] in {"max", "min"}):
        return f"Fast {'charge' if parts[2] == 'cha' else 'discharge'} DVA {'peak' if parts[3] == 'max' else 'valley'} width"
    if len(parts) == 5 and parts[:3] == ["mean", "d", "dqdv"]:
        band = {"l": "low", "m": "mid", "h": "high"}.get(parts[3], parts[3])
        return (r"$\langle\Delta(dQ/dV)\rangle$" +
                f", {band}-V " +
                ("charge" if parts[4] == "c" else "discharge"))
    return feature_label(name)


def condition_label(name: str) -> str:
    return CONDITION_LABELS.get(name, name.replace("_", " "))

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


def plot_model_responses(model, X: pd.DataFrame, y: pd.Series, target: str,
                         cv_r2: float, output_dir: Path, family: str = "JGNE") -> None:
    """Plot observed cells and dense, explicitly model-generated condition sweeps.

    One *observed* condition vector supplies the fixed values for the other
    predictors. Sweeps stay within each condition's measured range; intermediate
    settings are predictions, never additional measurements or SHAP samples.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    ranges = X.max() - X.min()
    scaled_distance = ((X - X.median()) / ranges.replace(0, 1)).abs().sum(axis=1)
    anchor = X.loc[scaled_distance.idxmin()].copy()
    fig, axes = plt.subplots(len(X.columns), 1, figsize=(9, 3.1 * len(X.columns)),
                             squeeze=False)
    generated = []
    for ax, condition in zip(axes[:, 0], X.columns):
        observed = np.sort(X[condition].unique())
        if len(observed) < 2:
            ax.set_visible(False)
            continue
        # Binary design factors have only two meaningful settings. For other
        # factors the dense line visualizes the fitted tree's step function,
        # including predictions at untested intermediate settings.
        sweep = observed if len(observed) == 2 else np.linspace(observed[0], observed[-1], 151)
        scenarios = pd.DataFrame([anchor.to_dict()] * len(sweep), columns=X.columns)
        scenarios[condition] = sweep
        predictions = model.predict(scenarios)
        for setting, predicted in zip(sweep, predictions):
            generated.append({"target": target, "varied_condition": condition,
                              "condition_value": setting, "model_prediction": predicted,
                              **{f"fixed_{col}": anchor[col] for col in X.columns if col != condition}})
        ax.scatter(X[condition], y, s=32, color="tab:blue", alpha=0.85,
                   label=f"Observed cells (n={len(X)})", zorder=3)
        if len(observed) == 2:
            ax.plot(sweep, predictions, "s--", color="tab:orange",
                    label="XGBoost predictions at measured settings")
            ax.set_xticks(observed)
        else:
            ax.step(sweep, predictions, where="post", color="tab:orange",
                    label="Model sweep (intermediate settings unmeasured)")
            ax.scatter(observed, np.interp(observed, sweep, predictions),
                       marker="s", color="tab:orange", s=27, zorder=4)
        ax.set_xlabel(condition_label(condition))
        ax.set_ylabel(feature_label(target))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{feature_title(target)}\n"
                 f"JGNE response | n={len(X)} | LOO R²={cv_r2:.2f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / f"{target}_model_response.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(generated).to_csv(output_dir / f"{target}_model_response.csv", index=False)


def supported_scenarios(X: pd.DataFrame, points_per_range: int = 51) -> pd.DataFrame:
    """Sweep each continuous condition within observed categorical protocols.

    A protocol is the combination of factors with at most two observed values.
    Other conditions remain at an actual cell's settings. This avoids generating
    contradictory categorical combinations or extrapolating beyond measured ranges.
    """
    categorical = [col for col in X if X[col].nunique() <= 2]
    continuous = [col for col in X if col not in categorical]
    scenarios = []
    for _, group in X.groupby(categorical, dropna=False) if categorical else [(None, X)]:
        anchors = group.drop_duplicates(subset=continuous) if continuous else group.iloc[:1]
        for _, anchor in anchors.iterrows():
            for varied in continuous:
                values = np.sort(group[varied].unique())
                if len(values) < 2:
                    continue
                for value in np.linspace(values[0], values[-1], points_per_range):
                    row = anchor.to_dict()
                    row[varied] = value
                    scenarios.append(row)
    if not scenarios:
        return X.drop_duplicates().reset_index(drop=True)
    return pd.DataFrame(scenarios, columns=X.columns).drop_duplicates().reset_index(drop=True)


def plot_scenario_shap(model, X: pd.DataFrame, target: str, cv_r2: float,
                       output_dir: Path, family: str = "JGNE") -> int:
    """Plot TreeSHAP for model-evaluated scenarios, not extra observations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    import xgboost as xgb

    scenarios = supported_scenarios(X)
    if scenarios.empty:
        return 0
    contributions = model.get_booster().predict(xgb.DMatrix(scenarios), pred_contribs=True)
    predictions = model.predict(scenarios)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = scenarios.copy()
    result.insert(0, "scenario_type", "model_evaluated_not_measured")
    result["model_prediction"] = predictions
    for col, vals in zip(X.columns, contributions[:, :-1].T):
        result[f"shap_{col}"] = vals
    result.to_csv(output_dir / f"{target}_scenario_shap.csv", index=False)

    shap.summary_plot(contributions[:, :-1], scenarios.rename(columns=condition_label),
                      show=False, max_display=len(X.columns))
    plt.title(f"{feature_title(target)}\n"
              f"Model scenarios={len(scenarios)} | cells={len(X)} | LOO R²={cv_r2:.2f}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{target}_scenario_shap.png", dpi=150)
    plt.close("all")
    return len(scenarios)


def plot_bootstrap_shap(X: pd.DataFrame, y: pd.Series, target: str,
                        cv_r2: float, output_dir: Path, family: str = "JGNE",
                        n_refits: int = 200, seed: int = 0,
                        context: str = "") -> None:
    """Report-style SHAP cloud and outlined per-cell means on beeswarm rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb

    params = dict(n_estimators=120, max_depth=2, learning_rate=0.08,
                  subsample=0.9, colsample_bytree=0.9,
                  reg_lambda=1.0, random_state=seed, verbosity=0)
    rng = np.random.default_rng(seed)
    explained = xgb.DMatrix(X)
    cloud = np.empty((n_refits, len(X), len(X.columns)))
    for i in range(n_refits):
        sample = rng.integers(0, len(X), len(X))
        model = xgb.XGBRegressor(**params).fit(X.iloc[sample], y.iloc[sample])
        cloud[i] = model.get_booster().predict(explained, pred_contribs=True)[:, :-1]

    means = cloud.mean(axis=0)
    order = np.argsort(np.abs(means).mean(axis=0))[::-1]
    # Match the report figures' broad landscape canvas and visible pastel
    # bootstrap clouds (rather than tiny near-transparent raster dots).
    fig, ax = plt.subplots(figsize=(14.35, max(6.5, 1.45 * len(X.columns) + 3.4)))
    cmap = plt.get_cmap("coolwarm")
    for row, col_idx in enumerate(order):
        values = X.iloc[:, col_idx].to_numpy(dtype=float)
        lo, hi = values.min(), values.max()
        normalized = ((values - lo) / (hi - lo) if hi > lo
                      else np.full(len(values), 0.5))
        # Cloud density reflects refit variation for the *same* measured cells.
        jitter = rng.uniform(-0.30, 0.30, size=cloud.shape[:2])
        ax.scatter(cloud[:, :, col_idx].ravel(), (row + jitter).ravel(),
                   c=cmap(np.tile(normalized, n_refits)), s=14, alpha=0.32,
                   edgecolors="none")
        ax.scatter(means[:, col_idx], row + rng.uniform(-0.1, 0.1, len(X)),
                   c=normalized, cmap=cmap, vmin=0, vmax=1, s=68,
                   edgecolors="black", linewidths=0.9, zorder=3)
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.set_yticks(range(len(order)), [condition_label(X.columns[j]) for j in order])
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_xlabel(f"SHAP value for {feature_label(target)}")
    ax.set_title(f"{feature_title(target)}{f' | {context}' if context else ''}\n"
                 f"Bootstrap SHAP | {n_refits} refits | n={len(X)} | LOO R²={cv_r2:.2f}",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.2)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap),
                            ax=ax, pad=0.02)
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.set_label("Condition value")
    fig.tight_layout(pad=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{target}_bootstrap_shap.png", dpi=150)
    plt.close(fig)

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
    ap.add_argument("--targets", default=None,
                    help="comma-separated feature columns to model (default: all numeric features)")
    ap.add_argument("--prefix", default=None,
                     help="input/output filename prefix (default: the family name); "
                          "use e.g. lwhk_c2 for the C/2 discharge table")
    ap.add_argument("--table-tag", default=None,
                    help="input/output table tag instead of soh995 (e.g. endpoint for trajectory metrics)")
    ap.add_argument("--plot-folder", default=None,
                    help="plot subfolder name (default: shap_<prefix>)")
    ap.add_argument("--plots", action="store_true",
                    help="write a SHAP beeswarm per predictive target")
    ap.add_argument("--plots-all", action="store_true",
                    help="also write clearly labelled SHAP beeswarms for non-predictive models")
    ap.add_argument("--response-plots", action="store_true",
                     help="plot measured cells alongside dense XGBoost condition sweeps for each target")
    ap.add_argument("--scenario-shap", action="store_true",
                    help="write separate dense TreeSHAP plots for model-evaluated, supported condition scenarios")
    ap.add_argument("--bootstrap-shap", action="store_true",
                    help="write report-style SHAP clouds from 200 bootstrap refits on measured cells")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    if args.table_tag:
        src = args.out_dir / f"{args.prefix or args.family}_features_{args.table_tag}.csv"
        df = pd.read_csv(src)
    else:
        df, src = load_table(args.out_dir, args.family, args.target_soh, args.prefix)
    wanted = ([c.strip() for c in args.conditions.split(",") if c.strip()]
              if args.conditions else FAMILY_CONDITIONS[args.family])
    conditions = usable_conditions(df, wanted)
    targets = target_columns(df, FAMILY_CONDITIONS[args.family])
    if args.targets:
        requested = [c.strip() for c in args.targets.split(",") if c.strip()]
        missing = [c for c in requested if c not in df.columns]
        if missing:
            ap.error(f"target columns missing from input: {missing}")
        targets = [c for c in requested if c in targets]
        if len(targets) != len(requested):
            ap.error("some selected targets are nonnumeric, constant, or condition columns")

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
    plot_dir = (args.dest or args.out_dir) / (args.plot_folder or f"shap_{args.prefix or args.family}")

    rows, per_target = [], []
    for tgt in targets:
        y_all = pd.to_numeric(df[tgt], errors="coerce")
        m = X_all.notna().all(axis=1) & y_all.notna()
        X, y = X_all[m], y_all[m]
        if len(X) < 5 or y.nunique() < 3:
            continue
        # A condition can vary in the input table yet be constant after the
        # target-specific complete-case filter (e.g. JGNE temperature when
        # pause duration is required). It has no estimable attribution.
        X = X.loc[:, X.nunique(dropna=True) > 1]
        if X.empty:
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

        if args.response_plots:
            try:
                plot_model_responses(model, X, y, tgt, cv_r2, plot_dir, args.family)
            except Exception as e:
                print(f"[WARN] response plot for {tgt} failed: {e}")

        if (args.plots and predictive) or args.plots_all:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import shap
                dest_p = plot_dir
                dest_p.mkdir(parents=True, exist_ok=True)
                named_X = X.rename(columns=condition_label)
                shap.summary_plot(sv, named_X, show=False)
                plt.title(f"{feature_title(tgt)}"
                          f"{f' | {args.table_tag}' if args.table_tag else ''}\n"
                          f"SHAP | n={len(X)} | LOO R²={cv_r2:.2f}")
                plt.tight_layout()
                plt.savefig(dest_p / f"{tgt}.png", dpi=120)
                plt.close("all")
            except Exception as e:
                print(f"[WARN] plot for {tgt} failed: {e}")

        if args.scenario_shap:
            try:
                count = plot_scenario_shap(model, X, tgt, cv_r2, plot_dir, args.family)
                print(f"[OK] {tgt}: {count} model-evaluated SHAP scenarios")
            except Exception as e:
                print(f"[WARN] scenario SHAP for {tgt} failed: {e}")

        if args.bootstrap_shap:
            try:
                plot_bootstrap_shap(X, y, tgt, cv_r2, plot_dir, args.family,
                                    context="endpoint" if args.table_tag == "endpoint" else "")
                print(f"[OK] {tgt}: 200 bootstrap-refit SHAP cloud")
            except Exception as e:
                print(f"[WARN] bootstrap SHAP for {tgt} failed: {e}")

    if not rows:
        print("[ERROR] nothing could be modelled (too few usable rows).")
        return

    imp = pd.DataFrame(rows)
    scores = pd.DataFrame(per_target).sort_values("cv_r2", ascending=False)

    dest = args.dest or args.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    tag = (args.table_tag or
           (f"soh{f'{args.target_soh:.3f}'.replace('0.', '').rstrip('0') or '0'}"))
    pref = args.prefix or args.family
    imp_path = dest / f"{pref}_shap_importance_{tag}.csv"
    sc_path = dest / f"{pref}_shap_model_scores_{tag}.csv"
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
