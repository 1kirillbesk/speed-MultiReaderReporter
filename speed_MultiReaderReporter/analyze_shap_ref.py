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

def scatter_features_vs_reference(
    df_exp: pd.DataFrame,
    feature_names: list[str],
    reference_key: str = "lw_reference",
    figsize=(7, 4),
    alpha=0.7,
):
    """
    For each feature in feature_names, make a scatter plot comparing:
      - reference rows (cell_name contains reference_key)
      - non-reference rows

    The x-axis is a categorical index (with jitter), y-axis is the feature value.
    """
    ref_mask = df_exp["cell_name"].astype(str).str.lower().str.contains(reference_key.lower())
    df_ref = df_exp.loc[ref_mask].copy()
    df_non = df_exp.loc[~ref_mask].copy()

    for feat in feature_names:
        if feat not in df_exp.columns:
            print(f"[skip] '{feat}' not in df_exp columns")
            continue

        y_ref = pd.to_numeric(df_ref[feat], errors="coerce").to_numpy()
        y_non = pd.to_numeric(df_non[feat], errors="coerce").to_numpy()

        # Remove NaNs
        y_ref = y_ref[~np.isnan(y_ref)]
        y_non = y_non[~np.isnan(y_non)]

        # x positions: 0 for ref, 1 for non-ref (+ jitter)
        x_ref = 0 + 0.08 * np.random.randn(len(y_ref))
        x_non = 1 + 0.08 * np.random.randn(len(y_non))

        plt.figure(figsize=figsize)
        plt.scatter(x_ref, y_ref, alpha=alpha, label="reference", marker="o")
        plt.scatter(x_non, y_non, alpha=alpha, label="non-reference", marker="x")
        plt.xticks([0, 1], ["reference", "non-reference"])
        plt.ylabel(feat)
        plt.title(f"{feat}: reference vs non-reference")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

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


def build_experiment_table(
    dir_path: Path,
    target_soh: float,
    reference_key: str = "lw_reference",
) -> pd.DataFrame:
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
        is_ref = reference_key in cell_name.lower()

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

        # --- Experimental conditions ---
        if all(c in df.columns for c in exp_conds):
            exp_row = filter_exp_first_row(df, exp_conds)
        else:
            # Keep reference cells even if exp_conds are missing
            if not is_ref:
                continue
            exp_row = pd.Series({c: np.nan for c in exp_conds})

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
    reference_key: str = "lw_reference",
    test_size: float = 0.1,
    random_state: int = 42,
    n_show: int = 500,
    n_iter_search: int = 60,
    cv_splits: int = 5,
    early_stopping_rounds: int = 200,
) -> Dict[str, Dict[str, Any]]:

    if exp_conds is None:
        exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    if target_cols is None:
        target_cols = ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_l_c"]

    # --- Identify reference rows ---
    ref_mask = df_exp["cell_name"].astype(str).str.lower().str.contains(reference_key.lower())

    if ref_mask.sum() == 0:
        raise ValueError(f"No reference rows found: cell_name contains '{reference_key}'")

    # --- Build reference baseline (one value per target) ---
    ref_baseline = {}
    for tcol in target_cols:
        ref_vals = coerce_series_to_float_1d(df_exp.loc[ref_mask, tcol])
        ref_baseline[tcol] = float(np.nanmean(ref_vals))
        if np.isnan(ref_baseline[tcol]):
            raise ValueError(f"Reference baseline for {tcol} is NaN (check reference data).")

    # --- Use ONLY non-reference rows for training ---
    df_train = df_exp.loc[~ref_mask].reset_index(drop=True)

    # ---- Build X ----
    X = df_train[exp_conds].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    # ---- Feature engineering: SOC + DOD ----
    X["soc"] = 0.5 * (X["soc_start"] + X["soc_end"])
    X["dod"] = X["soc_end"] - X["soc_start"]
    X = X.drop(columns=["soc_start", "soc_end"])
    exp_conds = ["soc", "dod", "c_rate_chg", "c_rate_dchg", "temp"]
    X.loc[X["c_rate_chg"] == 15, "c_rate_chg"] = 1.5

    # ---- Build y deltas ----
    y_dict = {}
    for tcol in target_cols:
        y_raw = coerce_series_to_float_1d(df_train[tcol])
        y_dict[tcol] = y_raw - ref_baseline[tcol]   # <-- delta from reference

    # ---- Keep only rows where ALL targets are valid ----
    valid_mask = np.ones(len(df_train), dtype=bool)
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

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=exp_conds)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=exp_conds)

    # ---- Validation split for early stopping ----
    X_tr, X_val, tr_idx, val_idx = train_test_split(
        X_train_scaled,
        np.arange(X_train_scaled.shape[0]),
        test_size=0.2,
        random_state=random_state,
    )

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
    )

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

        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

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

        search.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        print(f"\nTarget (delta from reference): {tcol}")
        print("Reference baseline:", ref_baseline[tcol])
        print("Best params:", search.best_params_)
        print("R²:", r2)
        print("RMSE:", rmse)

        # ---- SHAP KernelExplainer ----
        X_background = X_train_scaled_df
        X_sample = X_train_scaled_df

        def predict_fn(x):
            return best_model.predict(np.asarray(x))

        explainer = shap.KernelExplainer(predict_fn, X_background)

        n_plot = min(n_show, X_sample.shape[0])
        shap_values = explainer.shap_values(X_sample.iloc[:n_plot])

        shap.summary_plot(shap_values, X_sample.iloc[:n_plot], show=False)
        plt.title(f"SHAP summary (beeswarm) — Δ target: {tcol}")
        plt.tight_layout()
        plt.show()

        shap.summary_plot(shap_values, X_sample.iloc[:n_plot], plot_type="bar", show=False)
        plt.title(f"SHAP feature importance (bar) — Δ target: {tcol}")
        plt.tight_layout()
        plt.show()

        results[tcol] = {
            "model": best_model,
            "best_params": search.best_params_,
            "ref_baseline": ref_baseline,   # store all baselines
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


def monte_carlo_best_conditions_normalized(
    df_exp: pd.DataFrame,
    results: dict,
    target_cols: list[str],
    n_samples: int = 1000,
    top_k: int = 10,
    reference_key: str = "lw_reference",
    random_state: int = 42,
    norm_method: str = "std",   # "std" or "range" or "ref"
    agg: str = "l1",            # "l1" or "l2" or "max" or None
):
    """
    Monte Carlo sample experimental conditions. Predict Δ targets.
    Normalize each target INDIVIDUALLY (per-target) using norm_method.
    Optionally aggregate normalized deltas into a single score for ranking.

    norm_method:
      - "std":   divide by std(y_train) for that target
      - "range": divide by (p95 - p5) of y_train for that target (robust range)
      - "ref":   divide by abs(reference baseline) for that target (physical-ish)

    agg:
      - "l1": sum(|normalized|)
      - "l2": sqrt(sum(normalized^2))
      - "max": max(|normalized|)  (minimax)
      - None: no aggregate score; you can rank manually per target
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(random_state)

    # ---- bounds from non-reference data ----
    ref_mask = df_exp["cell_name"].astype(str).str.lower().str.contains(reference_key.lower())
    df_train = df_exp.loc[~ref_mask].copy()

    exp_conds_raw = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]
    bounds = {}
    for c in exp_conds_raw:
        col = pd.to_numeric(df_train[c], errors="coerce")
        bounds[c] = (float(np.nanmin(col)), float(np.nanmax(col)))

    # ---- sample conditions ----
    soc_start = rng.uniform(*bounds["soc_start"], n_samples)
    soc_end   = rng.uniform(*bounds["soc_end"], n_samples)
    c_chg     = rng.uniform(*bounds["c_rate_chg"], n_samples)
    c_dchg    = rng.uniform(*bounds["c_rate_dchg"], n_samples)
    # temp      = rng.uniform(*bounds["temp"], n_samples)
    temp = rng.choice([15.0, 25.0, 40.0], size=n_samples, replace=True)

    # ---- constraints ----
    soc_start = np.clip(soc_start, 0, 100)
    soc_end   = np.clip(soc_end, 0, 100)
    swap = soc_end < soc_start
    soc_start[swap], soc_end[swap] = soc_end[swap], soc_start[swap]

    # ---- engineered features ----
    soc = 0.5 * (soc_start + soc_end)
    dod = soc_end - soc_start

    X_mc = pd.DataFrame({
        "soc": soc,
        "dod": dod,
        "c_rate_chg": c_chg,
        "c_rate_dchg": c_dchg,
        "temp": temp,
    })

    # ---- scale X with trained scaler ----
    expected_cols = ["soc", "dod", "c_rate_chg", "c_rate_dchg", "temp"]
    X_mc = X_mc[expected_cols]
    X_mc.loc[X_mc["c_rate_chg"] == 15, "c_rate_chg"] = 1.5

    scaler = results[target_cols[0]]["scaler"]
    if hasattr(scaler, "feature_names_in_"):
        X_mc = X_mc.loc[:, scaler.feature_names_in_]

    X_mc_scaled = scaler.transform(X_mc)

    # ---- predictions + per-target normalization ----
    out = X_mc.copy()

    norm_scales = {}

    for t in target_cols:
        if t not in results:
            raise ValueError(f"Target '{t}' not found in results.")

        pred = results[t]["model"].predict(X_mc_scaled)
        out[f"delta_pred_{t}"] = pred

        # choose an INDIVIDUAL scale for this target
        y_train = results[t]["y_train"]

        if norm_method == "std":
            scale = float(np.std(y_train))
        elif norm_method == "range":
            p5 = float(np.percentile(y_train, 5))
            p95 = float(np.percentile(y_train, 95))
            scale = p95 - p5
        elif norm_method == "ref":
            # baseline stored as dict in results[t]["ref_baseline"]
            ref_base = results[t].get("ref_baseline", {})
            scale = float(abs(ref_base.get(t, 1.0)))
        else:
            raise ValueError("norm_method must be 'std', 'range', or 'ref'")

        if (not np.isfinite(scale)) or scale == 0:
            scale = 1.0

        norm_scales[t] = scale

        out[f"delta_norm_{t}"] = out[f"delta_pred_{t}"] / scale
        out[f"abs_delta_norm_{t}"] = np.abs(out[f"delta_norm_{t}"])

    # ---- optional aggregate score (after individual normalization) ----
    if agg is not None:
        abs_cols = [f"abs_delta_norm_{t}" for t in target_cols]

        if agg == "l1":
            out["abs_delta_score"] = out[abs_cols].sum(axis=1)
        elif agg == "l2":
            out["abs_delta_score"] = np.sqrt((out[abs_cols] ** 2).sum(axis=1))
        elif agg == "max":
            out["abs_delta_score"] = out[abs_cols].max(axis=1)
        else:
            raise ValueError("agg must be 'l1', 'l2', 'max', or None")

        best = out.nsmallest(top_k, "abs_delta_score").reset_index(drop=True)
    else:
        best = out.head(top_k).reset_index(drop=True)

    return best, norm_scales


def monte_carlo_best_conditions(
    df_exp: pd.DataFrame,
    results: dict,
    target_cols: list[str],
    n_samples: int = 1000,
    top_k: int = 10,
    reference_key: str = "lw_reference",
    random_state: int = 42,
    bounds_from: str = "data",  # "data" or "custom"
    custom_bounds: dict | None = None,
    aggregate: str = "l1",  # "l1" (sum abs), "l2" (sqrt sum sq), or "single"
):
    """
    Monte Carlo sample experimental conditions within min/max boundaries and find top_k
    samples with the smallest |Δ| (predicted delta target).

    Parameters
    ----------
    df_exp : experiment table (includes exp_conds columns)
    results : output from train_and_shap_kernel_multi (dict keyed by target col)
    target_cols : targets to predict delta for (must exist in results)
    n_samples : number of random samples
    top_k : how many best samples to return
    reference_key : identifies reference rows (excluded from bounds)
    bounds_from : "data" uses min/max from df_exp (non-reference); "custom" uses custom_bounds
    custom_bounds : dict like {"temp": (15, 45), ...}
    aggregate :
        - "single": uses abs(delta) of the first target_cols[0]
        - "l1": sum of abs deltas across targets
        - "l2": sqrt(sum of squared deltas)
    """
    rng = np.random.default_rng(random_state)

    # ---- choose bounds from non-reference data ----
    ref_mask = df_exp["cell_name"].astype(str).str.lower().str.contains(reference_key.lower())
    df_train = df_exp.loc[~ref_mask].copy()

    exp_conds_raw = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]
    for c in exp_conds_raw:
        if c not in df_train.columns:
            raise ValueError(f"Missing required exp cond column: {c}")

    if bounds_from == "data":
        bounds = {}
        for c in exp_conds_raw:
            col = pd.to_numeric(df_train[c], errors="coerce")
            lo = float(np.nanmin(col))
            hi = float(np.nanmax(col))
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"Non-finite bounds for {c}: lo={lo}, hi={hi}")
            bounds[c] = (lo, hi)
    elif bounds_from == "custom":
        if not custom_bounds:
            raise ValueError("custom_bounds is required when bounds_from='custom'")
        bounds = custom_bounds
        for c in exp_conds_raw:
            if c not in bounds:
                raise ValueError(f"custom_bounds missing {c}")
    else:
        raise ValueError("bounds_from must be 'data' or 'custom'")

    # ---- sample uniformly within bounds ----
    soc_start = rng.uniform(bounds["soc_start"][0], bounds["soc_start"][1], n_samples)
    soc_end   = rng.uniform(bounds["soc_end"][0], bounds["soc_end"][1], n_samples)
    c_chg     = rng.uniform(bounds["c_rate_chg"][0], bounds["c_rate_chg"][1], n_samples)
    c_dchg    = rng.uniform(bounds["c_rate_dchg"][0], bounds["c_rate_dchg"][1], n_samples)
    # temp      = rng.uniform(bounds["temp"][0], bounds["temp"][1], n_samples)
    temp = rng.choice([15.0, 25.0, 40.0], size=n_samples, replace=True)

    # ---- enforce constraints ----
    soc_start = np.clip(soc_start, 0, 100)
    soc_end   = np.clip(soc_end, 0, 100)

    # ensure soc_end >= soc_start by swapping where needed
    swap_mask = soc_end < soc_start
    soc_start2 = soc_start.copy()
    soc_end2 = soc_end.copy()
    soc_start2[swap_mask], soc_end2[swap_mask] = soc_end2[swap_mask], soc_start2[swap_mask]

    soc_start, soc_end = soc_start2, soc_end2

    # ---- engineer features like training ----
    soc = 0.5 * (soc_start + soc_end)
    dod = soc_end - soc_start

    X_mc = pd.DataFrame({
        "soc": soc,
        "dod": dod,
        "c_rate_chg": c_chg,
        "c_rate_dchg": c_dchg,
        "temp": temp,
    })

    # ---- scale using the trained scaler (same for all targets) ----
    any_target = target_cols[0]
    if any_target not in results:
        raise ValueError(f"Target '{any_target}' not found in results keys: {list(results.keys())}")

    scaler = results[any_target]["scaler"]
    expected_cols = ["soc", "dod", "c_rate_chg", "c_rate_dchg", "temp"]
    X_mc = X_mc[expected_cols]

    scaler = results[any_target]["scaler"]

    # If scaler stores fit-time feature names, align to them (extra safety)
    if hasattr(scaler, "feature_names_in_"):
        X_mc = X_mc.loc[:, scaler.feature_names_in_]

    X_mc_scaled = scaler.transform(X_mc)

    # ---- predict deltas for each target ----
    preds = {}
    for t in target_cols:
        if t not in results:
            raise ValueError(f"Target '{t}' not found in results.")
        model = results[t]["model"]
        preds[t] = model.predict(X_mc_scaled)

    pred_df = X_mc.copy()
    for t in target_cols:
        pred_df[f"delta_pred_{t}"] = preds[t]

    # ---- score: how close to zero delta ----
    if aggregate == "single":
        score = np.abs(preds[target_cols[0]])
    elif aggregate == "l1":
        score = np.zeros(n_samples, dtype=float)
        for t in target_cols:
            score += np.abs(preds[t])
    elif aggregate == "l2":
        score = np.zeros(n_samples, dtype=float)
        for t in target_cols:
            score += preds[t] ** 2
        score = np.sqrt(score)
    else:
        raise ValueError("aggregate must be 'single', 'l1', or 'l2'")

    pred_df["abs_delta_score"] = score

    # ---- pick top_k ----
    best = pred_df.nsmallest(top_k, "abs_delta_score").reset_index(drop=True)
    return best

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
        target_cols=["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_l_c"],
        reference_key="lw_reference",
    )
    scatter_features_vs_reference(
        df_exp,
        feature_names=["temp", "c_rate_chg", "c_rate_dchg", "mean_d_dqdv_m_c"],
        reference_key="lw_reference",
    )

    best10 = monte_carlo_best_conditions(
        df_exp=df_exp,
        results=results,
        target_cols=["mean_d_dqdv_m_c","mean_d_dqdv_h_c"],
        n_samples=1000,
        top_k=10,
        reference_key="lw_reference",
        aggregate="l1",  # sum of abs deltas across the three
    )

    # best10, scales = monte_carlo_best_conditions_normalized(
    #     df_exp=df_exp,
    #     results=results,
    #     target_cols=["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_l_c"],
    #     n_samples=1000,
    #     top_k=10,
    #     norm_method="std",
    #     agg="l1",
    # )

    # print("Per-target scales:", scales)

    cond_cols = ["soc", "dod", "c_rate_chg", "c_rate_dchg", "temp"]
    # show_cols = cond_cols + ["abs_delta_score"] + \
    #             [f"delta_pred_{t}" for t in ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_l_c"]] + \
    #             [f"delta_norm_{t}" for t in ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_l_c"]]

    print(best10[cond_cols])