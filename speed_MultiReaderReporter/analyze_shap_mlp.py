from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

import shap
import matplotlib.pyplot as plt


# -----------------------------
# Helpers (UNCHANGED from your working code)
# -----------------------------
def filter_exp_first_row(df: pd.DataFrame, exp_conds: list[str]) -> pd.Series:
    """Experimental conditions from row 0 of the raw CSV."""
    return df.loc[0, exp_conds]


def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric; invalid -> NaN."""
    return df.apply(pd.to_numeric, errors="coerce")


def coerce_series_to_float_1d(s: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series to a 1D float array.
    Handles values like "[8.366986e-3]" by stripping brackets.
    """
    s2 = s.copy()
    s2 = s2.astype(str).str.strip()
    s2 = s2.str.replace(r"^\[|\]$", "", regex=True)
    s2 = pd.to_numeric(s2, errors="coerce")
    return s2.to_numpy(dtype=float).reshape(-1)


def load_and_interpolate(
    df: pd.DataFrame,
    target_soh: float,
    interpolation_typ: str = "weeks",
) -> pd.DataFrame | None:
    """
    Interpolate feature columns over a reference axis until SOH reaches target_soh.
    Returns interpolated dataframe including SOH for safe indexing.
    """
    df = df.copy()

    df.iloc[0] = df.iloc[0].fillna(0)

    if "cap_ocv_dis" not in df.columns:
        raise ValueError("cap_ocv_dis is required to compute SOH.")
    df["SOH"] = df["cap_ocv_dis"] / df["cap_ocv_dis"].iloc[0]

    if interpolation_typ == "weeks":
        ref_col = "weeks"
    elif interpolation_typ == "throughput":
        ref_col = "throughput"
    else:
        raise ValueError("interpolation_typ must be 'weeks' or 'throughput'")

    mask = (df.index == 0) | (df[ref_col] != 0)
    df = df[mask].reset_index(drop=True)

    df = coerce_numeric_df(df)

    if df[ref_col].isna().any() or df["SOH"].isna().any():
        return None

    reference = df[ref_col].to_numpy(dtype=float)
    soh = df["SOH"].to_numpy(dtype=float)

    feature_cols = [c for c in df.columns if c not in [ref_col, "SOH"]]
    feature_mat = df[feature_cols].to_numpy(dtype=float)

    try:
        ref_at_target = np.interp(target_soh, soh[::-1], reference[::-1])
    except Exception:
        return None

    new_ref_cut = np.linspace(0.0, float(ref_at_target), 6)
    spacing = float(np.mean(np.diff(new_ref_cut))) if len(new_ref_cut) > 1 else 0.0
    if spacing <= 0:
        return None

    remaining = []
    current = float(ref_at_target)
    iters = 0
    max_iters = 5000

    while current + spacing <= float(reference[-1]):
        current += spacing
        remaining.append(current)
        iters += 1
        if iters >= max_iters:
            return None

    new_ref = np.concatenate([new_ref_cut, np.array(remaining, dtype=float)])

    interpolated_cols = []
    for j in range(feature_mat.shape[1]):
        interpolated_cols.append(np.interp(new_ref, reference, feature_mat[:, j]))

    interpolated = pd.DataFrame(
        np.stack(interpolated_cols, axis=1),
        columns=feature_cols
    )
    interpolated[ref_col] = new_ref
    interpolated["SOH"] = np.interp(new_ref, reference, soh)

    return interpolated


def build_experiment_table(dir_path: Path, target_soh: float) -> pd.DataFrame:
    exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]
    rows = []

    for csv_file in dir_path.glob("*.csv"):
        df = pd.read_csv(csv_file)

        df = df.copy()
        df["weeks"] = (
            pd.to_datetime(df["CU_time"]) - pd.to_datetime(df["CU_time"]).iloc[0]
        ).dt.total_seconds() / (7 * 24 * 3600)

        cell_name = csv_file.stem

        keep = [
            "weeks",
            "cap_ocv_dis",
            "mean_d_dqdv_m_c",
            "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d",
            "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c",
            "mean_d_dqdv_l_c",
        ]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            continue

        df_filter = df[keep].copy()

        interpolated_df = load_and_interpolate(df_filter, target_soh, interpolation_typ="weeks")
        if interpolated_df is None or interpolated_df.empty:
            continue

        idx = (interpolated_df["SOH"] - target_soh).abs().idxmin()
        row_at_soh = interpolated_df.loc[idx]

        if any(c not in df.columns for c in exp_conds):
            continue
        exp_row = filter_exp_first_row(df, exp_conds)

        combined = pd.concat(
            [pd.Series({"cell_name": cell_name}), exp_row, row_at_soh],
            axis=0
        )
        rows.append(combined)

    return pd.DataFrame(rows).reset_index(drop=True)


# -----------------------------
# Neural Network training + SHAP (KernelExplainer)
# -----------------------------
def train_and_shap_nn_multi(
    df_exp: pd.DataFrame,
    exp_conds: List[str] = None,
    target_cols: List[str] = None,
    test_size: float = 0.1,
    random_state: int = 42,
    n_iter_search: int = 80,
    cv_splits: int = 5,
    n_show: int = 300,          # SHAP is expensive for NN; keep modest
    background_size: int = 80,  # SHAP background subset for speed
) -> Dict[str, Dict[str, Any]]:
    """
    Replace XGBoost with a Neural Network (sklearn MLPRegressor),
    tune hyperparameters with RandomizedSearchCV, and compute SHAP using KernelExplainer.

    NOTE: KernelExplainer can be slow; keep n_show/background_size reasonable.
    """

    if exp_conds is None:
        exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    if target_cols is None:
        target_cols = ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_l_c"]

    # ---- Build X (same as your working code) ----
    X = df_exp[exp_conds].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    # Feature engineering: SOC + DOD
    X["soc"] = 0.5 * (X["soc_start"] + X["soc_end"])
    X["dod"] = X["soc_end"] - X["soc_start"]
    X.loc[X["c_rate_chg"] == 15, "c_rate_chg"] = 1.5

    X = X.drop(columns=["soc_start", "soc_end"])
    exp_conds = ["soc", "dod", "c_rate_chg", "c_rate_dchg", "temp"]

    # ---- Build y's ----
    y_dict = {}
    for tcol in target_cols:
        y_dict[tcol] = coerce_series_to_float_1d(df_exp[tcol])

    # ---- Keep only rows where ALL targets are valid ----
    valid_mask = np.ones(len(df_exp), dtype=bool)
    for tcol in target_cols:
        valid_mask &= ~np.isnan(y_dict[tcol])

    X = X.loc[valid_mask].reset_index(drop=True)
    for tcol in target_cols:
        y_dict[tcol] = y_dict[tcol][valid_mask]

    # Fill missing X with median, cast float
    X = X.fillna(X.median(numeric_only=True)).astype(float)

    # ---- Split ONCE so all targets use same split ----
    idx_all = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=test_size, random_state=random_state
    )
    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)

    # ---- Pipeline: scaler + NN ----
    # For neural nets, StandardScaler is usually best.
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            max_iter=4000,
            random_state=random_state,
            early_stopping=False,  # explicitly off
            n_iter_no_change=50,
        ))
    ])

    # ---- Hyperparameter search space for MLP ----
    param_dist = {
        "mlp__hidden_layer_sizes": [
            (16,), (32,), (64,),
            (32, 16), (64, 32), (128, 64),
            (64, 64), (128, 128)
        ],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": np.logspace(-6, -1, 12).tolist(),   # L2 regularization
        "mlp__learning_rate_init": np.logspace(-4, -2, 12).tolist(),
        "mlp__solver": ["adam"],                          # most stable
        "mlp__batch_size": [16, 32, 64, 128, 256],
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results: Dict[str, Dict[str, Any]] = {}

    for tcol in target_cols:
        y = y_dict[tcol].astype(float)
        y_train = y[idx_train]
        y_test = y[idx_test]

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            verbose=1,
            random_state=random_state,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        # ---- Evaluate ----
        y_pred = best_pipe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        print(f"\nTarget: {tcol}")
        print("Best params:", search.best_params_)
        print("R²:", r2)
        print("RMSE:", rmse)

        # ---- SHAP KernelExplainer ----
        # Use scaled data for SHAP so it matches what the NN actually sees
        scaler = best_pipe.named_steps["scaler"]
        model = best_pipe.named_steps["mlp"]

        X_train_scaled = scaler.transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=exp_conds)

        # background subset
        bg_n = min(background_size, len(X_train_scaled_df))
        background = X_train_scaled_df.iloc[:bg_n]

        # sample subset for SHAP plots
        n_plot = min(n_show, len(X_train_scaled_df))
        X_sample = X_train_scaled_df.iloc[:n_plot]

        def predict_fn(x):
            x_arr = np.asarray(x, dtype=float)
            return model.predict(x_arr)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"SHAP summary (beeswarm) — NN target: {tcol}")
        plt.tight_layout()
        plt.show()

        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title(f"SHAP feature importance (bar) — NN target: {tcol}")
        plt.tight_layout()
        plt.show()

        results[tcol] = {
            "pipeline": best_pipe,
            "best_params": search.best_params_,
            "explainer": explainer,
            "shap_values": shap_values,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "metrics": {"r2": float(r2), "rmse": rmse},
        }

    return results


def train_and_shap_nn_single(
    df_exp: pd.DataFrame,
    exp_conds: List[str] = None,
    target_col: str = "mean_d_dqdv_m_c",
    test_size: float = 0.1,
    random_state: int = 42,
    n_show: int = 200,
    background_size: int = 50,
):
    """
    Single neural network training + single SHAP analysis.
    NO tuning. batch_size = 16. One run.
    """

    if exp_conds is None:
        exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    # ---- Build X (unchanged logic) ----
    X = df_exp[exp_conds].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    X["soc"] = 0.5 * (X["soc_start"] + X["soc_end"])
    X["dod"] = X["soc_end"] - X["soc_start"]
    X.loc[X["c_rate_chg"] == 15, "c_rate_chg"] = 1.5

    X = X.drop(columns=["soc_start", "soc_end"])
    feature_names = ["soc", "dod", "c_rate_chg", "c_rate_dchg", "temp"]

    # ---- Build y ----
    y = coerce_series_to_float_1d(df_exp[target_col])

    valid_mask = ~np.isnan(y)
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]

    X = X.fillna(X.median(numeric_only=True)).astype(float)

    # ---- Train / test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ---- Model (fixed settings) ----
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            batch_size=16,          # 👈 FIXED
            max_iter=3000,
            random_state=random_state,
            early_stopping=False,
        ))
    ])

    # ---- Train ONCE ----
    pipe.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print("\nNeural Network results")
    print("R²:", r2)
    print("RMSE:", rmse)

    # ---- SHAP (KernelExplainer) ----
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["mlp"]

    X_train_scaled = scaler.transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)

    bg_n = min(background_size, len(X_train_scaled_df))
    background = X_train_scaled_df.iloc[:bg_n]

    n_plot = min(n_show, len(X_train_scaled_df))
    X_sample = X_train_scaled_df.iloc[:n_plot]

    def predict_fn(x):
        return model.predict(np.asarray(x, dtype=float))

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP summary — NN target: {target_col}")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f"SHAP importance — NN target: {target_col}")
    plt.tight_layout()
    plt.show()

    return {
        "pipeline": pipe,
        "metrics": {"r2": float(r2), "rmse": rmse},
        "explainer": explainer,
        "shap_values": shap_values,
    }


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    target_soh = 0.998

    df_exp = build_experiment_table(dir_path, target_soh)
    print("df_exp shape:", df_exp.shape)
    print(df_exp[["cell_name", "mean_d_dqdv_m_c"]].head())

    # results = train_and_shap_nn_multi(
    #     df_exp,
    #     target_cols=["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_l_c"],
    #     n_iter_search=80,     # increase if you want (e.g., 150)
    #     cv_splits=5,
    #     n_show=300,           # SHAP cost control
    #     background_size=80,   # SHAP cost control
    # )

    results = train_and_shap_nn_single(
        df_exp,
        target_col="mean_d_dqdv_m_c",  # ONE target
        n_show=200,
        background_size=50,
    )
