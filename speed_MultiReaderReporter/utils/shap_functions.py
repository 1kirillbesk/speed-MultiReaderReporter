from __future__ import annotations

from pathlib import Path
import sys
import math
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple
import shap
from scipy.interpolate import CubicSpline

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ML (Monte Carlo surrogate model)
import xgboost as xgb


def train_xgb_no_val_and_shap(
    df_exp: pd.DataFrame,
    dist_df: pd.DataFrame,
    ref_names: list[str],
    raw_conds: list[str],
    feat_conds: list[str],
    random_state: int = 42,
    shap_max_display: int = 20,
):
    """
    Train XGBRegressor on all data and optionally show SHAP plots.
    Returns: model, explainer, X_feat, y, feat_names
    """

    # ---------- merge + filter ----------
    df = df_exp.merge(
        dist_df[["cell_name", "dist_feat"]],
        on="cell_name",
        how="inner"
    )
    df = df[~df["cell_name"].isin(ref_names)].copy()

    # ---------- target ----------
    y = pd.to_numeric(df["dist_feat"], errors="coerce")

    # ---------- features ----------
    X_feat, feat_names = make_features_from_raw(
        df,
        raw_conds=raw_conds,
        feat_conds=feat_conds,
        drop_raw_soc=True
    )

    # ---------- clean ----------
    ok = ~y.isna()
    ok &= np.isfinite(X_feat.to_numpy()).all(axis=1)

    X_feat = X_feat.loc[ok].reset_index(drop=True)
    y = y.loc[ok].to_numpy(dtype=float)

    if len(X_feat) < 5:
        raise ValueError("Not enough valid rows to train XGB model.")

    # ---------- model ----------
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=random_state,
    )
    model.fit(X_feat, y)

    # ---------- SHAP ----------
    explainer = None
    # 🔥 use TreeExplainer (faster for XGB)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_feat)

    plt.figure()
    shap.summary_plot(
        shap_values, X_feat,
        show=False,
        max_display=shap_max_display
    )
    plt.title("SHAP summary — dist_feat")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(
        shap_values, X_feat,
        plot_type="bar",
        show=False,
        max_display=shap_max_display
    )
    plt.title("SHAP importance — dist_feat")
    plt.tight_layout()
    plt.show()

    return model, explainer, X_feat, y, feat_names


def make_features_from_raw(
    df: pd.DataFrame,
    raw_conds: list[str],
    feat_conds: list[str],
    *,
    drop_raw_soc: bool = True,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Build model features from raw experiment conditions.

    - Coerces to numeric
    - Creates soc, dod
    - Replaces c_rate_chg==15 with 1.5
    - Optionally drops soc_start/soc_end
    Returns: (X_features, feature_names)
    """
    X = df[raw_conds].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    X["soc"] = 0.5 * (X["soc_start"] + X["soc_end"])
    X["dod"] = X["soc_end"] - X["soc_start"]

    # fix weird encoding
    X.loc[X["c_rate_chg"] == 15, "c_rate_chg"] = 1.5

    if drop_raw_soc:
        X = X.drop(columns=["soc_start", "soc_end"])

    feature_names = feat_conds if drop_raw_soc else (raw_conds + ["soc", "dod"])
    X = X[feature_names].copy()
    return X, feature_names