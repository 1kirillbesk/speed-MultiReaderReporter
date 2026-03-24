from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import xgboost as xgb
import shap
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
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

    # Convert to str, strip whitespace
    s2 = s2.astype(str).str.strip()

    # Remove a single surrounding bracket pair: "[0.12]" -> "0.12"
    s2 = s2.str.replace(r"^\[|\]$", "", regex=True)

    # Convert to numeric
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

    # Fill first row NaNs with 0 (as in your original)
    df.iloc[0] = df.iloc[0].fillna(0)

    # Compute SOH
    if "cap_ocv_dis" not in df.columns:
        raise ValueError("cap_ocv_dis is required to compute SOH.")
    df["SOH"] = df["cap_ocv_dis"] / df["cap_ocv_dis"].iloc[0]

    # Choose reference column
    if interpolation_typ == "weeks":
        ref_col = "weeks"
    elif interpolation_typ == "throughput":
        ref_col = "throughput"
    else:
        raise ValueError("interpolation_typ must be 'weeks' or 'throughput'")

    # Keep first row + all rows where ref != 0 (your original logic)
    mask = (df.index == 0) | (df[ref_col] != 0)
    df = df[mask].reset_index(drop=True)

    # Coerce relevant columns to numeric
    df = coerce_numeric_df(df)

    # If reference or SOH contains NaNs, interpolation is impossible
    if df[ref_col].isna().any() or df["SOH"].isna().any():
        return None

    reference = df[ref_col].to_numpy(dtype=float)
    soh = df["SOH"].to_numpy(dtype=float)

    # Feature columns to interpolate (exclude ref and SOH)
    feature_cols = [c for c in df.columns if c not in [ref_col, "SOH"]]
    feature_mat = df[feature_cols].to_numpy(dtype=float)

    # Interpolate reference point where SOH hits target_soh
    # SOH typically decreases => use reversed arrays so "x" is increasing for np.interp
    try:
        ref_at_target = np.interp(target_soh, soh[::-1], reference[::-1])
    except Exception:
        return None

    # Create a reference grid: 6 points from 0..ref_at_target, then continue with same spacing
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

    # Interpolate each feature col
    interpolated_cols = []
    for j in range(feature_mat.shape[1]):
        interpolated_cols.append(np.interp(new_ref, reference, feature_mat[:, j]))

    interpolated = pd.DataFrame(
        np.stack(interpolated_cols, axis=1),
        columns=feature_cols
    )
    interpolated[ref_col] = new_ref

    # ALSO interpolate SOH so you can pick the nearest row
    interpolated["SOH"] = np.interp(new_ref, reference, soh)

    return interpolated


def build_experiment_table(dir_path: Path, target_soh: float) -> pd.DataFrame:
    exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]
    rows = []

    for csv_file in dir_path.glob("*.csv"):
        df = pd.read_csv(csv_file)

        # weeks from CU_time
        df = df.copy()
        df["weeks"] = (
            pd.to_datetime(df["CU_time"]) - pd.to_datetime(df["CU_time"]).iloc[0]
        ).dt.total_seconds() / (7 * 24 * 3600)

        cell_name = csv_file.stem

        # Columns needed for interpolation + target
        keep = [
            "weeks",
            "cap_ocv_dis",
            "mean_d_dqdv_m_c",   # target you asked for
            "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d",
            "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c",
            "mean_d_dqdv_l_c",
        ]
        # if some columns are missing, skip this file
        missing = [c for c in keep if c not in df.columns]
        if missing:
            continue

        df_filter = df[keep].copy()

        interpolated_df = load_and_interpolate(df_filter, target_soh, interpolation_typ="weeks")
        if interpolated_df is None or interpolated_df.empty:
            continue

        # Find row closest to target_soh
        idx = (interpolated_df["SOH"] - target_soh).abs().idxmin()
        row_at_soh = interpolated_df.loc[idx]

        # Experimental conditions from row 0
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
# Train + SHAP (KernelExplainer style)
# -----------------------------
def train_and_shap_kernel_multi(
    df_exp: pd.DataFrame,
    exp_conds: List[str] = None,
    target_cols: List[str] = None,
    test_size: float = 0.1,
    random_state: int = 42,
    n_show: int = 500,
    n_iter_search: int = 60,     # increase for better search
    cv_splits: int = 5,
    early_stopping_rounds: int = 200,
) -> Dict[str, Dict[str, Any]]:
    """
    Train one XGBoost model per target column using RandomizedSearchCV + early stopping,
    MinMaxScaler instead of StandardScaler, and SHAP KernelExplainer per target.
    """

    if exp_conds is None:
        exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    if target_cols is None:
        target_cols = ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_l_c"]

    # ---- Build X ----
    X = df_exp[exp_conds].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    # ---- Feature engineering: SOC + DOD ----
    X["soc"] = 0.5 * (X["soc_start"] + X["soc_end"])
    X["dod"] = X["soc_end"] - X["soc_start"]
    X.loc[X["c_rate_chg"] == 15, "c_rate_chg"] = 1.5

    # Drop original columns
    X = X.drop(columns=["soc_start", "soc_end"])

    # Update exp_conds to match new feature set
    exp_conds = ["soc", "dod", "c_rate_chg", "c_rate_dchg", "temp"]

    # ---- Build y's (coerce each target) ----
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

    # ---- MinMax scale ----
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Make DataFrames so SHAP plots show feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=exp_conds)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=exp_conds)

    # ---- Create a validation split for early stopping (from train only) ----
    X_tr, X_val, tr_idx, val_idx = train_test_split(
        X_train_scaled,
        np.arange(X_train_scaled.shape[0]),
        test_size=0.2,
        random_state=random_state,
    )

    # Base model (tree method "hist" tends to be faster on CPU)
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state, # big ceiling; early stopping will pick best iteration
    )

    # A solid search space for your feature count (5 features) and typical tabular regression
    param_dist = {
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": np.linspace(0.005, 0.15, 30).tolist(),
        "min_child_weight": [1, 2, 5, 10, 20, 40],
        "subsample": np.linspace(0.5, 1.0, 11).tolist(),
        "colsample_bytree": np.linspace(0.5, 1.0, 11).tolist(),
        "gamma": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        "reg_alpha": [0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0],
        "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results: Dict[str, Dict[str, Any]] = {}

    for tcol in target_cols:
        y = y_dict[tcol].astype(float)
        y_train = y[idx_train]
        y_test = y[idx_test]

        # Map y_train onto train/val split indices created above
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        # RandomizedSearchCV optimizes CV score; early stopping is handled in fit
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            verbose=1,
            random_state=random_state,
            n_jobs=-1,
        )

        # Fit search on FULL training data (X_train_scaled), but with early stopping using (X_val, y_val)
        # We pass eval_set via **fit_params**.
        search.fit(
            X_train_scaled,  # keep as np array for XGBoost
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_model = search.best_estimator_

        # ---- Evaluate on held-out test ----
        y_pred = best_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        print(f"\nTarget: {tcol}")
        print("Best params:", search.best_params_)
        print("R²:", r2)
        print("RMSE:", rmse)

        # ---- SHAP KernelExplainer (per target) ----
        # Background: use a subset for speed if training set is large
        X_background = X_train_scaled_df  # DataFrame with column names
        X_sample = X_train_scaled_df

        def predict_fn(x):
            # KernelExplainer may pass a numpy array even if background is DF
            x_arr = np.asarray(x)
            return best_model.predict(x_arr)

        explainer = shap.KernelExplainer(predict_fn, X_background)

        n_plot = min(n_show, X_sample.shape[0])
        shap_values = explainer.shap_values(X_sample.iloc[:n_plot])

        feature_names = exp_conds

        # --- beeswarm ---
        shap.summary_plot(
            shap_values,
            X_sample.iloc[:n_plot],
            show=False
        )
        plt.title(f"SHAP summary (beeswarm) — target: {tcol}")
        plt.tight_layout()
        plt.show()

        # --- bar ---
        shap.summary_plot(
            shap_values,
            X_sample.iloc[:n_plot],
            plot_type="bar",
            show=False
        )
        plt.title(f"SHAP feature importance (bar) — target: {tcol}")
        plt.tight_layout()
        plt.show()

        results[tcol] = {
            "model": best_model,
            "best_params": search.best_params_,
            "explainer": explainer,
            "shap_values": shap_values,
            "scaler": scaler,
            "X_train": X_train,
            "X_train_scaled": X_train_scaled,
            "X_test": X_test,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "metrics": {"r2": r2, "rmse": rmse},
        }

    return results


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    target_soh = 0.998

    df_exp = build_experiment_table(dir_path, target_soh)
    print("df_exp shape:", df_exp.shape)
    print(df_exp[["cell_name", "mean_d_dqdv_m_c"]].head())

    results = train_and_shap_kernel_multi(
        df_exp,
        target_cols=["mean_d_dqdv_m_c", "mean_d_dqdv_h_c","mean_d_dqdv_l_c"]  # <-- your two outputs
    )
