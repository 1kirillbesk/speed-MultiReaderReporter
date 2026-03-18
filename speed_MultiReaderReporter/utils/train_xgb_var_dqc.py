"""
train_xgb_var_dqc.py
─────────────────────────────────────────────────────────────────────────────
Trains an XGBoost model to predict  var_dQ_c @ throughput=500k
from experimental conditions (soc_start, soc_end, c_rate_chg, c_rate_dchg, temp).

Drop-in addition to speed_MultiReaderReporter/main.py.

Usage (inside main, after build_regression_table_cap93_and_var_at_thr):
    result = train_xgb_var_dqc_from_conditions(
        df_exp=df_exp,
        df_reg_table=df_all,          # output of build_regression_table_cap93_and_var_at_thr
        throughput_target=500_000.0,
    )
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

try:
    import xgboost as xgb
    _HAVE_XGB = True
except ImportError:
    xgb = None
    _HAVE_XGB = False

try:
    import shap
    _HAVE_SHAP = True
except ImportError:
    shap = None
    _HAVE_SHAP = False


# ── reuse the same feature pipeline as the rest of the codebase ──────────────
RAW_CONDS = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]
FEAT_CONDS = ["soc", "dod", "c_rate_chg", "c_rate_dchg", "temp"]


def _make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Identical feature engineering to make_features_from_raw in main.py:
      soc  = 0.5*(soc_start + soc_end)
      dod  = soc_end - soc_start
      c_rate_chg==15 → 1.5   (encoding fix)
      drop soc_start / soc_end
    """
    X = df[RAW_CONDS].copy().apply(pd.to_numeric, errors="coerce")
    X["soc"]  = 0.5 * (X["soc_start"] + X["soc_end"])
    X["dod"]  = X["soc_end"] - X["soc_start"]
    X.loc[X["c_rate_chg"] == 15, "c_rate_chg"] = 1.5
    X = X.drop(columns=["soc_start", "soc_end"])
    return X[FEAT_CONDS].copy(), FEAT_CONDS


# ─────────────────────────────────────────────────────────────────────────────
def train_xgb_var_dqc_from_conditions(
    df_exp: pd.DataFrame,
    df_reg_table: pd.DataFrame,
    *,
    throughput_target: float = 500_000.0,
    target_col: str = "var_dQ_c_at_thr500k",
    test_size: float = 0.20,
    random_state: int = 42,
    shap_max_display: int = 20,
) -> dict:
    """
    Train XGBoost to predict  var_dQ_c @ throughput_target
    from engineered experimental conditions.

    Parameters
    ----------
    df_exp        : DataFrame with columns cell_name + RAW_CONDS  (from main loop)
    df_reg_table  : output of build_regression_table_cap93_and_var_at_thr
                    must contain  cell_name  and  target_col
    throughput_target : float, used only for axis labels / print
    target_col    : column in df_reg_table that is the regression target
    test_size     : fraction held out for evaluation
    random_state  : RNG seed for train/test split and XGB
    shap_max_display : max features shown in SHAP plots

    Returns
    -------
    dict with keys:
        model, X_train, X_test, y_train, y_test,
        y_pred_train, y_pred_test,
        r2_train, r2_test, mae_train, mae_test,
        feature_names, explainer (or None), shap_values (or None)
    """
    if not _HAVE_XGB:
        raise RuntimeError("xgboost not installed — run: pip install xgboost")

    # ── 1. merge experimental conditions with regression targets ─────────────
    needed_exp = ["cell_name"] + RAW_CONDS
    missing = [c for c in needed_exp if c not in df_exp.columns]
    if missing:
        raise ValueError(f"df_exp is missing columns: {missing}")

    if target_col not in df_reg_table.columns:
        raise ValueError(
            f"target_col='{target_col}' not found in df_reg_table. "
            f"Available: {df_reg_table.columns.tolist()}"
        )

    df = (
        df_exp[needed_exp]
        .merge(df_reg_table[["cell_name", target_col]], on="cell_name", how="inner")
    )

    # ── 2. feature engineering ────────────────────────────────────────────────
    X_feat, feat_names = _make_features(df)
    y = pd.to_numeric(df[target_col], errors="coerce")

    # keep only rows where both X and y are fully finite
    ok = (
        ~y.isna()
        & np.isfinite(X_feat.to_numpy()).all(axis=1)
    )
    X_feat = X_feat.loc[ok].reset_index(drop=True)
    y      = y.loc[ok].to_numpy(dtype=float)

    n_total = len(y)
    if n_total < 10:
        raise ValueError(
            f"Only {n_total} valid rows after merging — need at least 10 to train."
        )

    print(f"\n{'='*70}")
    print(f"  XGBoost: exp conditions → {target_col}")
    print(f"  throughput reference point : {throughput_target:,.0f}")
    print(f"  samples after merge+filter : {n_total}")
    print(f"  features                   : {feat_names}")
    print(f"{'='*70}")

    # ── 3. train / test split ─────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feat, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    print(f"  train: {len(y_tr)}   test: {len(y_te)}")

    # ── 4. XGBoost model ──────────────────────────────────────────────────────
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        tree_method="hist",
        random_state=random_state,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )

    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    r2_tr  = r2_score(y_tr, y_pred_tr)
    r2_te  = r2_score(y_te, y_pred_te)
    mae_tr = mean_absolute_error(y_tr, y_pred_tr)
    mae_te = mean_absolute_error(y_te, y_pred_te)

    print(f"\n  Train  →  R²={r2_tr:.4f}   MAE={mae_tr:.4g}")
    print(f"  Test   →  R²={r2_te:.4f}   MAE={mae_te:.4g}")

    # ── 5. parity plot ────────────────────────────────────────────────────────
    fig_parity, ax_p = plt.subplots(figsize=(5.5, 5))
    ax_p.scatter(y_tr, y_pred_tr, s=20, alpha=0.6, label=f"train  R²={r2_tr:.3f}")
    ax_p.scatter(y_te, y_pred_te, s=30, alpha=0.9, marker="^",
                 label=f"test   R²={r2_te:.3f}")
    lims = [
        min(y.min(), y_pred_tr.min(), y_pred_te.min()),
        max(y.max(), y_pred_tr.max(), y_pred_te.max()),
    ]
    ax_p.plot(lims, lims, "k--", linewidth=1.0, label="perfect")
    ax_p.set_xlabel(f"Actual  {target_col}")
    ax_p.set_ylabel(f"Predicted  {target_col}")
    ax_p.set_title(
        f"Parity plot: exp conditions → {target_col}\n"
        f"@ throughput={throughput_target:,.0f}"
    )
    ax_p.legend()
    ax_p.grid(True, alpha=0.3)
    fig_parity.tight_layout()

    # ── 6. XGBoost native feature importance ─────────────────────────────────
    fig_imp, ax_i = plt.subplots(figsize=(6, 4))
    xgb.plot_importance(
        model,
        ax=ax_i,
        importance_type="gain",
        max_num_features=len(feat_names),
        title=f"XGB feature importance (gain)\ntarget: {target_col}",
    )
    fig_imp.tight_layout()

    # ── 7. SHAP ───────────────────────────────────────────────────────────────
    explainer   = None
    shap_values = None

    if _HAVE_SHAP:
        print("\n  Computing SHAP values (TreeExplainer) …")

        # TreeExplainer is exact & fast for XGBoost trees
        explainer = shap.KernelExplainer(lambda a: model.predict(np.asarray(a)), X_feat)
        shap_values = explainer.shap_values(X_feat)
        # beeswarm summary
        plt.figure()
        shap.summary_plot(
            shap_values, X_feat,
            show=False,
            max_display=shap_max_display,
        )
        plt.title(f"SHAP summary (beeswarm) — {target_col}")
        plt.tight_layout()

        # bar importance
        plt.figure()
        shap.summary_plot(
            shap_values, X_feat,
            plot_type="bar",
            show=False,
            max_display=shap_max_display,
        )
        plt.title(f"SHAP importance (bar) — {target_col}")
        plt.tight_layout()

        print("  SHAP done.")
    else:
        print("\n  [WARN] shap not installed — skipping SHAP plots.")
        print("         Install with: pip install shap")

    plt.show()

    return {
        "model":        model,
        "feature_names": feat_names,
        "X_train": X_tr,  "X_test": X_te,
        "y_train": y_tr,  "y_test": y_te,
        "y_pred_train": y_pred_tr, "y_pred_test": y_pred_te,
        "r2_train": r2_tr,  "r2_test": r2_te,
        "mae_train": mae_tr, "mae_test": mae_te,
        "explainer":   explainer,
        "shap_values": shap_values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO CALL THIS FROM main() — paste these lines after the
# build_regression_table_cap93_and_var_at_thr calls:
# ─────────────────────────────────────────────────────────────────────────────
#
#   from train_xgb_var_dqc import train_xgb_var_dqc_from_conditions
#
#   result_var_dqc = train_xgb_var_dqc_from_conditions(
#       df_exp        = df_exp,       # already built in main loop
#       df_reg_table  = df_all,       # build_regression_table on ALL cells
#       throughput_target = 500_000.0,
#   )
#
# The returned dict gives you the trained model + metrics + SHAP explainer.
# To predict on new conditions:
#
#   new_raw = pd.DataFrame([{
#       "soc_start": 20, "soc_end": 80,
#       "c_rate_chg": 0.5, "c_rate_dchg": 1.0, "temp": 25,
#   }])
#   from train_xgb_var_dqc import _make_features
#   X_new, _ = _make_features(new_raw)
#   pred = result_var_dqc["model"].predict(X_new)
#   print(f"Predicted var_dQ_c @ 500k throughput: {pred[0]:.4g}")