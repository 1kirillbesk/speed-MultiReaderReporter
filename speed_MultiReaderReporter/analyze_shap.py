from __future__ import annotations
from typing import Dict, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

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
    df_exp,
    exp_conds: List[str] = None,
    target_cols: List[str] = None,
    test_size: float = 0.1,
    random_state: int = 42,
    n_show: int = 500,
):
    """
    Train one XGBoost model per target column and run SHAP KernelExplainer per target.
    Returns dicts of {target: model/scaler/explainer/shap_values}.
    """

    if exp_conds is None:
        exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    if target_cols is None:
        # <-- put your TWO outputs here
        target_cols = ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c"]

    # ---- Build X ----
    X = df_exp[exp_conds].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    # ---- Build all y's (coerce each target) ----
    y_dict = {}
    for tcol in target_cols:
        y_arr = coerce_series_to_float_1d(df_exp[tcol])  # your robust converter
        y_dict[tcol] = y_arr

    # ---- Keep only rows where ALL targets are valid (no NaN) ----
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
    X_test  = X.iloc[idx_test].reset_index(drop=True)

    # Scale once, reused for all targets (like your older code)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for tcol in target_cols:
        y = y_dict[tcol].astype(float)
        y_train = y[idx_train]
        y_test = y[idx_test]

        # ---- Train model for this target ----
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            base_score=float(np.mean(y_train)),  # critical scalar float
            random_state=random_state,
        )
        model.fit(X_train_scaled, y_train)

        # ---- Evaluate ----
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        print(f"\nTarget: {tcol}")
        print("R²:", r2)
        print("RMSE:", rmse)

        # ---- SHAP KernelExplainer (per target) ----
        X_background = X_train_scaled
        X_sample = X_train_scaled

        def predict_fn(x_array: np.ndarray) -> np.ndarray:
            return model.predict(x_array)

        explainer = shap.KernelExplainer(predict_fn, X_background)

        n_plot = min(n_show, X_sample.shape[0])
        shap_values = explainer.shap_values(X_sample[:n_plot])

        # ---- Plots per target ----
        feature_names = exp_conds

        shap.summary_plot(
            shap_values,
            X_sample[:n_plot],
            feature_names=feature_names,
            show=True
        )

        shap.summary_plot(
            shap_values,
            X_sample[:n_plot],
            feature_names=feature_names,
            plot_type="bar",
            show=True
        )

        for i, col in enumerate(feature_names):
            shap.dependence_plot(
                i,
                shap_values,
                X_sample[:n_plot],
                feature_names=feature_names,
                show=True
            )

        results[tcol] = {
            "model": model,
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
        target_cols=["mean_d_dqdv_m_c", "mean_d_dqdv_h_c"]  # <-- your two outputs
    )
