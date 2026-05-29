from __future__ import annotations

from pathlib import Path
import sys
import math
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple
from scipy.interpolate import CubicSpline
from analyze_linear_prediction import build_regression_table_cap93_and_var_at_thr
from utils.combined_cost_search import train_model_var_dqc


# 3D plotting
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ML (Monte Carlo surrogate model)
import xgboost as xgb

# SHAP is optional (you can comment it out if you don't want it)
try:
    import shap
    _HAVE_SHAP = True
except Exception:
    shap = None
    _HAVE_SHAP = False


# --- relative paths ---
here = Path(__file__).resolve().parent
sys.path.append(str(here))
sys.path.append(str(here / "core"))
sys.path.append(str(here / "loaders"))
sys.path.append(str(here / "utils"))

# -----------------------------
# Feature pipeline (single source of truth) for the surrogate XGB model
# -----------------------------
RAW_CONDS_DEFAULT = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]
FEAT_CONDS_DEFAULT = ["soc", "dod", "c_rate_chg", "c_rate_dchg", "temp"]


def make_features_from_raw(
    df: pd.DataFrame,
    raw_conds: list[str] = RAW_CONDS_DEFAULT,
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

    feature_names = FEAT_CONDS_DEFAULT if drop_raw_soc else (raw_conds + ["soc", "dod"])
    X = X[feature_names].copy()
    return X, feature_names


def train_xgb_no_val_and_shap(
    df_exp: pd.DataFrame,
    dist_df: pd.DataFrame,
    ref_names: list[str],
    raw_conds: list[str] = RAW_CONDS_DEFAULT,
    random_state: int = 42,
    shap_max_display: int = 20,
):
    """
    Fit XGBRegressor on ALL available exp data (no train/val split),
    target = dist_feat (distance-to-nearest-ref in feature space).

    Returns: model, explainer(or None), X_feat, y, feat_names
    """
    df = df_exp.merge(dist_df[["cell_name", "dist_feat"]], on="cell_name", how="inner")
    df = df[~df["cell_name"].isin(ref_names)].copy()

    y = pd.to_numeric(df["dist_feat"], errors="coerce")

    X_feat, feat_names = make_features_from_raw(df, raw_conds=raw_conds, drop_raw_soc=True)

    ok = ~y.isna()
    ok &= np.isfinite(X_feat.to_numpy()).all(axis=1)

    X_feat = X_feat.loc[ok].reset_index(drop=True)
    y = y.loc[ok].to_numpy(dtype=float)

    if len(X_feat) < 5:
        raise ValueError("Not enough rows with valid features + target to train XGB model.")

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
    )
    model.fit(X_feat, y)

    explainer = None
    if _HAVE_SHAP:
        # KernelExplainer works for any model, but is slower.
        explainer = shap.KernelExplainer(lambda a: model.predict(np.asarray(a)), X_feat)
        shap_values = explainer.shap_values(X_feat)

        plt.figure()
        shap.summary_plot(shap_values, X_feat, show=False, max_display=shap_max_display)
        plt.title("SHAP summary (beeswarm) — target: distance-to-nearest-ref (dist_feat)")
        plt.tight_layout()
        plt.show()

        shap.summary_plot(shap_values, X_feat, plot_type="bar", show=False, max_display=shap_max_display)
        plt.title("SHAP importance (bar) — target: distance-to-nearest-ref (dist_feat)")
        plt.tight_layout()
        plt.show()

    return model, explainer, X_feat, y, feat_names


def monte_carlo_best_conditions_for_distance(
    df_exp: pd.DataFrame,
    model: xgb.XGBRegressor,
    ref_names: list[str],
    raw_conds: list[str] = RAW_CONDS_DEFAULT,
    n_samples_per_temp: int = 200,
    top_k: int | None = 20,
    random_state: int = 42,
    allowed_temp: np.ndarray | None = None,
    allowed_soc_start: np.ndarray | None = None,
    allowed_soc_end: np.ndarray | None = None,
    allowed_cur_cha: np.ndarray | None = None,
    allowed_cur_dis: np.ndarray | None = None,
    min_soc_delta: float = 10.0,
) -> pd.DataFrame:
    """
    Monte Carlo search over RAW conditions (with your discrete constraints),
    n_samples_per_temp per temperature, then engineer SAME features as training,
    predict, and return smallest top_k.

    Discrete:
      - temp in {15,25,40}
      - soc_start in {0..60 step 10}
      - soc_end in {20..100 step 10}
    Constraints:
      - soc_end > soc_start
      - soc_end - soc_start > min_soc_delta
    Other raws sampled uniformly within observed min/max (non-reference).
    """
    rng = np.random.default_rng(random_state)

    df_train = df_exp[~df_exp["cell_name"].isin(ref_names)].copy()
    df_train["c_rate_chg"] = pd.to_numeric(df_train["c_rate_chg"], errors="coerce")
    df_train.loc[df_train["c_rate_chg"] == 15, "c_rate_chg"] = 1.5

    # bounds for continuous ones
    bounds = {}
    for c in raw_conds:
        if c in {"soc_start", "soc_end", "temp"}:
            continue
        col = pd.to_numeric(df_train[c], errors="coerce")
        lo = float(np.nanmin(col))
        hi = float(np.nanmax(col))
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Non-finite bounds for {c}: lo={lo}, hi={hi}")
        bounds[c] = (lo, hi)

    allowed_temp = (
        np.asarray(allowed_temp, dtype=float)
        if allowed_temp is not None
        else np.array([15.0, 25.0, 40.0], dtype=float)
    )
    allowed_soc_start = (
        np.asarray(allowed_soc_start, dtype=float)
        if allowed_soc_start is not None
        else np.arange(0, 80, 10, dtype=float)
    )
    allowed_soc_end = (
        np.asarray(allowed_soc_end, dtype=float)
        if allowed_soc_end is not None
        else np.arange(20, 110, 10, dtype=float)
    )
    allowed_cur_cha = (
        np.asarray(allowed_cur_cha, dtype=float)
        if allowed_cur_cha is not None
        else np.arange(0.5, 1.75, 0.25, dtype=float)
    )
    allowed_cur_dis = (
        np.asarray(allowed_cur_dis, dtype=float)
        if allowed_cur_dis is not None
        else np.arange(1, 3.25, 0.25, dtype=float)
    )

    valid_pairs = np.array(
        [(s0, s1) for s0 in allowed_soc_start for s1 in allowed_soc_end if (s1 - s0) > float(min_soc_delta)],
        dtype=float,
    )
    if len(valid_pairs) == 0:
        raise ValueError("No valid (soc_start, soc_end) pairs under the constraints.")

    blocks = []
    for temp in allowed_temp:
        idx = rng.integers(0, len(valid_pairs), size=n_samples_per_temp)
        soc_start = valid_pairs[idx, 0]
        soc_end = valid_pairs[idx, 1]
        c_rate_chg = rng.choice(allowed_cur_cha, size=n_samples_per_temp, replace=True)
        c_rate_dchg = rng.choice(allowed_cur_dis, size=n_samples_per_temp, replace=True)
        temp_col = np.full(n_samples_per_temp, temp, dtype=float)

        block = pd.DataFrame(
            {
                "soc_start": soc_start,
                "soc_end": soc_end,
                "c_rate_chg": c_rate_chg,
                "c_rate_dchg": c_rate_dchg,
                "temp": temp_col,
            }
        )
        blocks.append(block)

    X_mc_raw = pd.concat(blocks, ignore_index=True)[raw_conds].astype(float)

    X_mc_feat, _ = make_features_from_raw(X_mc_raw, raw_conds=raw_conds, drop_raw_soc=True)

    pred = model.predict(X_mc_feat)

    out = X_mc_raw.copy()
    out["soc"] = X_mc_feat["soc"].to_numpy()
    out["dod"] = X_mc_feat["dod"].to_numpy()
    out["pred_dist_feat"] = pred  # predicted distance-to-nearest-ref in feature space

    if top_k is None:
        return out.reset_index(drop=True)
    return out.nsmallest(top_k, "pred_dist_feat").reset_index(drop=True)


# -----------------------------
# Existing helper functions from your 3D script
# -----------------------------
def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = float(max(sigma, 1e-12))
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def load_and_interpolate(
    df: pd.DataFrame,
    target_soh: float,
    interpolation_typ: str,
    method: str = "cubic",
    throughput_max: float | None = None,
):
    """
    Interpolate all feature columns on a new reference grid.

    Parameters
    ----------
    df : DataFrame
        Input data containing either 'weeks' or 'throughput_cum' plus 'cap_ocv_dis'.
    target_soh : float
        SOH target used to define the dense early part of the new reference grid.
    interpolation_typ : str
        'weeks' or 'throughput'
    method : str
        'cubic' or 'linear'
    throughput_max : float | None
        If interpolation_typ == 'throughput', optionally keep only rows with
        throughput_cum <= throughput_max before fitting/interpolating.
    """
    df = df.copy()
    df.iloc[0] = df.iloc[0].fillna(0)

    df["SOH"] = df["cap_ocv_dis"] / df["cap_ocv_dis"].iloc[0]

    if interpolation_typ == "weeks":
        ref_name = "weeks"
    elif interpolation_typ == "throughput":
        ref_name = "throughput_cum"
    else:
        raise ValueError("interpolation_typ must be 'weeks' or 'throughput'")

    mask = (df.index == 0) | (df[ref_name] != 0)
    df = df.loc[mask].reset_index(drop=True)

    df = df.apply(pd.to_numeric, errors="coerce")

    if interpolation_typ == "throughput" and throughput_max is not None:
        df = df[df["throughput_cum"] <= float(throughput_max)].copy()
        df = df.reset_index(drop=True)

    if df.empty:
        return None

    if df[ref_name].isna().any() or df["SOH"].isna().any():
        return None

    reference = df[ref_name].to_numpy(dtype=float)
    target_data = df["SOH"].to_numpy(dtype=float)

    ref_s = pd.Series(reference)
    keep = ~ref_s.duplicated(keep="first")
    df = df.loc[keep.values].reset_index(drop=True)

    reference = df[ref_name].to_numpy(dtype=float)
    target_data = df["SOH"].to_numpy(dtype=float)

    if len(reference) < 2:
        return None

    if method == "cubic" and len(reference) < 3:
        # cubic spline really needs at least 3 points
        return None

    feature_cols = [c for c in df.columns if c != ref_name]
    feature = df[feature_cols].to_numpy(dtype=float)

    interpolated_ref = np.interp(target_soh, target_data[::-1], reference[::-1])

    new_ref_points_cut = np.linspace(0.0, float(interpolated_ref), 5)
    spacing = float(np.mean(np.diff(new_ref_points_cut))) if len(new_ref_points_cut) > 1 else 0.0
    if spacing <= 0:
        return None

    remaining_points: list[float] = []
    cur = float(interpolated_ref)
    max_iters = 5000
    iters = 0

    while cur + spacing <= float(reference[-1]):
        cur += spacing
        remaining_points.append(cur)
        iters += 1
        if iters >= max_iters:
            return None

    new_ref_points = np.concatenate([new_ref_points_cut, np.array(remaining_points, dtype=float)])

    interpolated_columns = []
    for j in range(feature.shape[1]):
        yj = feature[:, j]

        if np.isnan(yj).any():
            interpolated_columns.append(np.interp(new_ref_points, reference, yj))
            continue

        if method == "linear":
            y_new = np.interp(new_ref_points, reference, yj)
            interpolated_columns.append(y_new)

        elif method == "cubic":
            try:
                cs = CubicSpline(reference, yj, bc_type="natural", extrapolate=False)
                y_new = cs(new_ref_points)
                interpolated_columns.append(y_new)
            except Exception:
                interpolated_columns.append(np.interp(new_ref_points, reference, yj))
        else:
            raise ValueError("method must be 'linear' or 'cubic'")

    interpolated_data = np.stack(interpolated_columns, axis=1)
    out = pd.DataFrame(interpolated_data, columns=feature_cols)
    out[ref_name] = new_ref_points

    if "SOH" in out.columns:
        out["SOH"] = out["SOH"].clip(lower=0.0, upper=1.05)

    return out


def exp_row_from_first_line(df: pd.DataFrame, exp_conds: list[str]) -> pd.Series:
    return df.loc[0, exp_conds]


def weeks_to_reach_soh(traj_weeks: pd.DataFrame, soh_target: float) -> float | None:
    if traj_weeks is None or traj_weeks.empty:
        return None
    if "weeks" not in traj_weeks.columns or "SOH" not in traj_weeks.columns:
        return None

    w = traj_weeks["weeks"].to_numpy(dtype=float)
    s = traj_weeks["SOH"].to_numpy(dtype=float)

    ok = ~np.isnan(w) & ~np.isnan(s)
    w = w[ok]
    s = s[ok]
    if len(w) < 3:
        return None

    if np.nanmin(s) > soh_target:
        return None

    idx = np.argsort(s)
    s_sorted = s[idx]
    w_sorted = w[idx]

    keep = ~pd.Series(s_sorted).duplicated(keep="first")
    s_sorted = s_sorted[keep.values]
    w_sorted = w_sorted[keep.values]
    if len(s_sorted) < 2:
        return None

    return float(np.interp(soh_target, s_sorted, w_sorted))


def idx_first_reach_soh(traj: pd.DataFrame, soh_target: float) -> int | None:
    if traj is None or traj.empty or "SOH" not in traj.columns:
        return None

    s = traj["SOH"].to_numpy(dtype=float)
    ok = ~np.isnan(s)
    if not np.any(ok):
        return None
    s = s[ok]

    hit = np.where(s <= soh_target)[0]
    if len(hit) == 0:
        return None
    return int(hit[0])


def interp_step_at_soh(traj: pd.DataFrame, soh_target: float) -> float | None:
    if traj is None or traj.empty or "SOH" not in traj.columns:
        return None

    s = traj["SOH"].to_numpy(dtype=float)
    ok = ~np.isnan(s)
    if not np.any(ok):
        return None
    s = s[ok]

    if np.nanmin(s) > soh_target:
        return None

    idx = np.arange(len(s), dtype=float)
    order = np.argsort(s)
    s_sorted = s[order]
    idx_sorted = idx[order]

    keep = ~pd.Series(s_sorted).duplicated(keep="first")
    s_sorted = s_sorted[keep.values]
    idx_sorted = idx_sorted[keep.values]
    if len(s_sorted) < 2:
        return None

    return float(np.interp(soh_target, s_sorted, idx_sorted))


def exhaustive_best_conditions_for_distance(
    df_exp, model, ref_names, raw_conds=RAW_CONDS_DEFAULT,
    n_samples_per_temp=200, top_k=8, random_state=42,
):
    rng = np.random.default_rng(random_state)

    allowed_temp = np.array([15.0, 25.0, 40.0])
    allowed_soc_start = np.arange(0, 70, 10, dtype=float)
    allowed_soc_end = np.arange(20, 110, 10, dtype=float)
    allowed_cur_cha = np.arange(0.5, 1.75, 0.25, dtype=float)
    allowed_cur_dis = np.arange(1, 3.25, 0.25, dtype=float)

    valid_pairs = np.array(
        [(s0, s1) for s0 in allowed_soc_start for s1 in allowed_soc_end if (s1 - s0) > 10],
        dtype=float,
    )

    all_rows = []
    for temp in allowed_temp:
        idx = rng.integers(0, len(valid_pairs), size=n_samples_per_temp)
        soc_start = valid_pairs[idx, 0]
        soc_end = valid_pairs[idx, 1]
        c_rate_chg = rng.choice(allowed_cur_cha, size=n_samples_per_temp)
        c_rate_dchg = rng.choice(allowed_cur_dis, size=n_samples_per_temp)
        temp_col = np.full(n_samples_per_temp, temp)

        block = pd.DataFrame({
            "soc_start": soc_start,
            "soc_end": soc_end,
            "c_rate_chg": c_rate_chg,
            "c_rate_dchg": c_rate_dchg,
            "temp": temp_col,
        })
        all_rows.append(block)

    X_mc_raw = pd.concat(all_rows, ignore_index=True)[raw_conds].astype(float)

    X_mc_feat, _ = make_features_from_raw(X_mc_raw, raw_conds=raw_conds, drop_raw_soc=True)

    pred = model.predict(X_mc_feat)

    out = X_mc_raw.copy()
    out["soc"] = X_mc_feat["soc"].to_numpy()
    out["dod"] = X_mc_feat["dod"].to_numpy()
    out["pred_dist_feat"] = pred

    return out.nsmallest(top_k, "pred_dist_feat").reset_index(drop=True)


# -----------------------------
# Main (your 3D feature-space + NEW Monte Carlo surrogate)
# -----------------------------
def main(
    dir_path: Path,
    out_dir: Path,
    interp_method: str = "cubic",
    throughput_max: float | None = 8e7,
):
    exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    target_soh_features = 0.995   # feature row selection
    target_soh_plot = 0.98       # interpolation grid for plotting

    rows = []
    rows_ref = []

    # Reference cell names
    REF_NAMES = ["SPEED_LW_reference_1", "SPEED_LW_reference_2", "SPEED_LW_reference_3"]

    # CONFIG YOU WANT:
    K_CLOSEST_FEATURE = 25
    MC_SAMPLES_PER_TEMP = 200
    MC_TOP_K = 16
    MC_TOP_PER_TEMP = 4
    MC_TEMP_TARGETS = [25.0, 40.0]
    MC_TOP_FEATURE = 5
    FEATURE_RANK_COL = "var_dQ_c_at_thr500k"
    K_SLOWEST_BLACK = 1
    SOH_SLOW_TARGET = 0.96

    # Gaussian distribution target (index where SOH reaches this)
    SOH_DIST_TARGET = 0.96

    needed_cols = (
        ["CU_time"]
        + exp_conds
        + ["cap_ocv_dis"]
        + [
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c", "var_dQ_c",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d","var_d_dqdv_h_c","var_d_dqdv_l_c",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c", "mean_d_dqdv_l_c_l",
        ]
        + ["throughput_cum", "mean_d_dqdv_h_d", "mean_d_dqdv_l_d"]
    )

    # ---- Plot 1: index-based ----
    fig, ax = plt.subplots(figsize=(9, 5))
    added_blue_label = False
    added_red_label = False

    # ---- Plot 2: weeks on x-axis ----
    fig_w, ax_w = plt.subplots(figsize=(9, 5))
    added_blue_label_w = False
    added_red_label_w = False

    # ---- Plot 3: throughput_cum on x-axis ----
    fig_t, ax_t = plt.subplots(figsize=(9, 5))
    added_blue_label_t = False
    added_red_label_t = False

    traj_by_cell_weeks: dict[str, pd.DataFrame] = {}
    traj_by_cell_thr: dict[str, pd.DataFrame] = {}
    traj_interp_weeks_by_cell: dict[str, pd.DataFrame] = {}
    traj_by_cell_reg: dict[str, pd.DataFrame] = {}

    cells_below_08 = []

    for csv_file in dir_path.glob("*.csv"):
        cell_name = csv_file.stem

        try:
            df = pd.read_csv(csv_file, usecols=lambda c: c in needed_cols)
        except Exception as e:
            print(f"[WARN] {cell_name}: read failed ({e}), skipping.")
            continue

        if "CU_time" not in df.columns:
            print(f"[WARN] {cell_name}: missing CU_time, skipping.")
            continue

        t = pd.to_datetime(df["CU_time"], errors="coerce", cache=True)
        if t.isna().all():
            print(f"[WARN] {cell_name}: CU_time invalid, skipping.")
            continue

        t0 = t.iloc[0]
        df["weeks"] = (t - t0).dt.total_seconds() / (7 * 24 * 3600)

        reg_cols = ["weeks", "throughput_cum", "var_dQ_c", "cap_ocv_dis"]
        if all(c in df.columns for c in reg_cols):
            traj_by_cell_reg[cell_name] = df[reg_cols].copy()

        if any(c not in df.columns for c in exp_conds):
            print(f"[WARN] {cell_name}: missing exp_conds, skipping.")
            continue
        exp_row = exp_row_from_first_line(df, exp_conds)

        interp_cols_weeks = [
            "weeks", "cap_ocv_dis",
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c", "mean_d_dqdv_l_c_l",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c","var_d_dqdv_h_c","var_d_dqdv_l_c",
            "mean_d_dqdv_h_d", "mean_d_dqdv_l_d", "throughput_cum"
        ]
        missing = [c for c in interp_cols_weeks if c not in df.columns]
        if missing:
            print(f"[WARN] {cell_name}: missing {missing}, skipping.")
            continue

        df_filter_weeks = df[interp_cols_weeks].copy()

        # TRUE week-based interpolation here
        interpolated_plot_weeks = load_and_interpolate(
            df_filter_weeks,
            target_soh_plot,
            interpolation_typ="weeks",
            method=interp_method,
            throughput_max=throughput_max,
        )
        if interpolated_plot_weeks is None or interpolated_plot_weeks.empty:
            print(f"[WARN] {cell_name}: interpolation (weeks) failed, skipping.")
            continue

        if "SOH" in interpolated_plot_weeks.columns:
            min_soh = float(np.nanmin(interpolated_plot_weeks["SOH"].to_numpy(dtype=float)))
            if min_soh < 0.8:
                cells_below_08.append((cell_name, min_soh))

        traj_interp_weeks_by_cell[cell_name] = interpolated_plot_weeks.copy()
        traj_by_cell_weeks[cell_name] = interpolated_plot_weeks[["weeks", "SOH"]].copy()

        # throughput interpolation
        interp_cols_thr = [
            "throughput_cum", "cap_ocv_dis", "weeks",
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c", "mean_d_dqdv_l_c_l",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
            "mean_d_dqdv_h_d", "mean_d_dqdv_l_d","var_d_dqdv_h_c","var_d_dqdv_l_c"
        ]
        missing_thr = [c for c in interp_cols_thr if c not in df.columns]
        if missing_thr:
            interpolated_plot_thr = None
        else:
            df_filter_thr = df[interp_cols_thr].copy()
            interpolated_plot_thr = load_and_interpolate(
                df_filter_thr,
                target_soh_plot,
                interpolation_typ="throughput",
                method=interp_method,
                throughput_max=throughput_max,
            )

        if interpolated_plot_thr is not None and not interpolated_plot_thr.empty:
            if "SOH" in interpolated_plot_thr.columns and "throughput_cum" in interpolated_plot_thr.columns:
                traj_by_cell_thr[cell_name] = interpolated_plot_thr[["throughput_cum", "SOH"]].copy()

        # ---- Base plotting ----
        is_ref_case = exp_row.isna().any() and ("LW_reference" in cell_name)

        if is_ref_case:
            label = "LW_reference (exp NaN)" if not added_red_label else None
            added_red_label = True
            ax.plot(interpolated_plot_weeks["SOH"], color="red", alpha=0.25, linewidth=0.6, label=label, zorder=1)

            label_w = "LW_reference (exp NaN)" if not added_red_label_w else None
            added_red_label_w = True
            ax_w.plot(interpolated_plot_weeks["weeks"], interpolated_plot_weeks["SOH"],
                      color="red", alpha=0.25, linewidth=0.6, label=label_w, zorder=1)

            if cell_name in traj_by_cell_thr:
                label_t = "LW_reference (exp NaN)" if not added_red_label_t else None
                added_red_label_t = True
                ax_t.plot(traj_by_cell_thr[cell_name]["throughput_cum"], traj_by_cell_thr[cell_name]["SOH"],
                          color="red", alpha=0.25, linewidth=0.6, label=label_t, zorder=1)
        else:
            label = "other" if not added_blue_label else None
            added_blue_label = True
            ax.plot(interpolated_plot_weeks["SOH"], color="blue", alpha=0.20, linewidth=0.55, label=label, zorder=2)

            label_w = "other" if not added_blue_label_w else None
            added_blue_label_w = True
            ax_w.plot(interpolated_plot_weeks["weeks"], interpolated_plot_weeks["SOH"],
                      color="blue", alpha=0.20, linewidth=0.55, label=label_w, zorder=2)

            if cell_name in traj_by_cell_thr:
                label_t = "other" if not added_blue_label_t else None
                added_blue_label_t = True
                ax_t.plot(traj_by_cell_thr[cell_name]["throughput_cum"], traj_by_cell_thr[cell_name]["SOH"],
                          color="blue", alpha=0.20, linewidth=0.55, label=label_t, zorder=2)

        # Interpolate every feature column exactly at target_soh_features
        soh_arr = interpolated_plot_weeks["SOH"].to_numpy(dtype=float)
        feat_at_target = {}

        for col in interpolated_plot_weeks.columns:
            if col == "SOH":
                continue
            y_arr = interpolated_plot_weeks[col].to_numpy(dtype=float)
            # SOH is decreasing so flip arrays for np.interp (needs increasing xp)
            val = np.interp(target_soh_features, soh_arr[::-1], y_arr[::-1])
            feat_at_target[col] = val

        feat_at_target["SOH"] = target_soh_features
        row_feat = pd.Series(feat_at_target)

        combined_row = pd.concat([pd.Series({"cell_name": cell_name}), exp_row, row_feat], axis=0)

        if exp_row.isna().any():
            rows_ref.append(combined_row)
        else:
            rows.append(combined_row)

        print(f"{cell_name}: success")

    df_ref = pd.DataFrame(rows_ref).reset_index(drop=True)
    df_exp = pd.DataFrame(rows).reset_index(drop=True)

    # plot references on top
    added_ref_label = False
    for ref_name in REF_NAMES:
        traj_w = traj_by_cell_weeks.get(ref_name)
        if traj_w is None:
            continue

        ax.plot(traj_w["SOH"], color="red", alpha=0.95, linewidth=1.6,
                label="SPEED_LW_reference_3..5" if not added_ref_label else None, zorder=3)
        ax_w.plot(traj_w["weeks"], traj_w["SOH"], color="red", alpha=0.95, linewidth=1.6,
                  label="SPEED_LW_reference_3..5" if not added_ref_label else None, zorder=3)

        traj_t = traj_by_cell_thr.get(ref_name)
        if traj_t is not None:
            ax_t.plot(traj_t["throughput_cum"], traj_t["SOH"], color="red", alpha=0.95, linewidth=1.6,
                      label="SPEED_LW_reference_3..5" if not added_ref_label else None, zorder=3)

        added_ref_label = True

    # -----------------------------
    # Closest/farthest in 3D feature space + build dist_df for surrogate training
    # -----------------------------
    x_col = "mean_d_dqdv_m_c"
    y_col = "mean_d_dqdv_l_c_l"
    z_col = "mean_d_dqdv_h_c"
    K_FARTHEST = 20

    if not df_ref.empty and "cell_name" in df_ref.columns:
        ref_sub = (
            df_ref[df_ref["cell_name"].isin(REF_NAMES)][["cell_name", x_col, y_col, z_col]]
            .dropna()
            .reset_index(drop=True)
        )
    else:
        ref_sub = pd.DataFrame(columns=["cell_name", x_col, y_col, z_col])

    if not df_exp.empty and "cell_name" in df_exp.columns:
        exp_sub = df_exp[["cell_name", x_col, y_col, z_col]].dropna().reset_index(drop=True)
        exp_sub = exp_sub[~exp_sub["cell_name"].str.contains("reference", case=False, na=False)].reset_index(drop=True)
    else:
        exp_sub = pd.DataFrame(columns=["cell_name", x_col, y_col, z_col])

    closest_cellnames: list[str] = []
    farthest_cellnames: list[str] = []
    slowest_black_cellnames: list[str] = []

    dist_df = pd.DataFrame(columns=["cell_name", "dist_feat", "best_ref"])

    if ref_sub.empty:
        print(f"[WARN] None of REF_NAMES found in df_ref: {REF_NAMES}")
    elif exp_sub.empty:
        print("[WARN] df_exp has no valid rows for the selected feature columns.")
    else:
        ref_xyz = ref_sub[[x_col, y_col, z_col]].to_numpy(dtype=float)
        exp_xyz = exp_sub[[x_col, y_col, z_col]].to_numpy(dtype=float)

        # robust scaling by IQR on (refs + exps)
        all_xyz = np.vstack([ref_xyz, exp_xyz])
        q25 = np.quantile(all_xyz, 0.25, axis=0)
        q75 = np.quantile(all_xyz, 0.75, axis=0)
        scale = np.maximum(q75 - q25, 1e-12)

        ref_n = ref_xyz / scale
        exp_n = exp_xyz / scale

        # distance to each ref, then nearest
        dists = np.linalg.norm(exp_n[:, None, :] - ref_n[None, :, :], axis=2)  # (n_exp, n_ref)
        dist = np.min(dists, axis=1)
        best_ref_idx = np.argmin(dists, axis=1)
        best_ref_names = ref_sub.loc[best_ref_idx, "cell_name"].to_numpy()

        # dist_df used by surrogate model
        dist_df = pd.DataFrame(
            {
                "cell_name": exp_sub["cell_name"].to_numpy(),
                "dist_feat": dist.astype(float),
                "best_ref": best_ref_names,
            }
        ).sort_values("dist_feat").reset_index(drop=True)

        all_cells = sorted(traj_by_cell_reg.keys())
        df_all = build_regression_table_cap93_and_var_at_thr(
            all_cells, traj_by_cell_reg,
            cap_col="cap_ocv_dis", cap_frac=0.96, throughput_target=300_000.0, var_col="var_dQ_c"
        )

        k1 = min(K_CLOSEST_FEATURE, len(exp_sub))
        closest_idx = np.argsort(dist)[:k1]
        closest_cellnames = exp_sub.loc[closest_idx, "cell_name"].tolist()

        k2 = min(K_FARTHEST, len(exp_sub))
        farthest_idx = np.argsort(dist)[-k2:]
        farthest_cellnames = exp_sub.loc[farthest_idx, "cell_name"].tolist()

        print(f"\nClosest {len(closest_cellnames)} exp cells (orange) to NEAREST(ref) at SOH={target_soh_features}:")
        for cn in closest_cellnames:
            print(" -", cn)

        print("\nExperimental conditions for closest experiments:")
        cols_to_print = ["cell_name"] + exp_conds
        df_closest_conds = (
            df_exp[df_exp["cell_name"].isin(closest_cellnames)][cols_to_print]
            .sort_values("cell_name")
        )
        print(df_closest_conds.to_string(index=False))

        interp_cols_weeks = [
            "weeks", "cap_ocv_dis",
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c", "mean_d_dqdv_l_c_l",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c", "var_d_dqdv_h_c", "var_d_dqdv_l_c",
        ]

        x_feat = "weeks"
        feat_cols = [c for c in (interp_cols_weeks + ["SOH"]) if c != x_feat]
        feat_cols = list(dict.fromkeys(feat_cols))

        closest_set = set(closest_cellnames)
        farthest_set = set(farthest_cellnames)
        ref_set = set(REF_NAMES)

        n_feats = len(feat_cols)
        ncols = 3
        nrows = math.ceil(n_feats / ncols)

        fig_feat, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 3.5 * nrows),
            sharex=True
        )
        axes = np.array(axes).ravel()

        for i, feat in enumerate(feat_cols):
            axf = axes[i]

            for cell_name, df_interp in traj_interp_weeks_by_cell.items():
                if (x_feat not in df_interp.columns) or (feat not in df_interp.columns):
                    continue

                x = df_interp[x_feat].to_numpy(dtype=float)
                y = df_interp[feat].to_numpy(dtype=float)

                if cell_name in ref_set:
                    color, lw, a = "red", 1.8, 0.95
                elif cell_name in closest_set:
                    color, lw, a = "orange", 1.4, 0.90
                elif cell_name in farthest_set:
                    color, lw, a = "green", 1.4, 0.90
                else:
                    color, lw, a = "blue", 0.9, 0.20

                axf.plot(y, color=color, linewidth=lw, alpha=a)

            axf.set_title(feat)
            axf.grid(True, alpha=0.3)
            axf.set_xlabel("weeks")

        # hide unused axes
        for j in range(n_feats, len(axes)):
            axes[j].axis("off")

        legend_handles = [
            Line2D([0], [0], color="red", lw=2, label="refs"),
            Line2D([0], [0], color="orange", lw=2, label=f"closest {len(closest_set)}"),
            Line2D([0], [0], color="green", lw=2, label=f"farthest {len(farthest_set)}"),
            Line2D([0], [0], color="blue", lw=2, alpha=0.35, label="other"),
        ]
        fig_feat.legend(handles=legend_handles, loc="upper right")

        fig_feat.suptitle("Interpolated features vs weeks (refs=red, closest=orange, farthest=green)", y=1.02)
        fig_feat.tight_layout()

        IDX = 25  # index in the interpolated arrays

        fig_dist, axes_dist = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 3.5 * nrows),
            sharex=False
        )
        axes_dist = np.array(axes_dist).ravel()

        for i, feat in enumerate(feat_cols):
            axd = axes_dist[i]

            vals_ref, vals_close, vals_far, vals_other = [], [], [], []

            for cell_name, df_interp in traj_interp_weeks_by_cell.items():
                if feat not in df_interp.columns:
                    continue

                y = df_interp[feat].to_numpy(dtype=float)
                if len(y) <= IDX:
                    continue

                v = float(y[IDX])
                if not np.isfinite(v):
                    continue

                if cell_name in ref_set:
                    vals_ref.append(v)
                elif cell_name in closest_set:
                    vals_close.append(v)
                elif cell_name in farthest_set:
                    vals_far.append(v)
                else:
                    vals_other.append(v)

            all_vals = np.array(vals_ref + vals_close + vals_far + vals_other, dtype=float)
            if all_vals.size < 2:
                axd.set_title(f"{feat} (insufficient data @ idx={IDX})")
                axd.axis("off")
                continue

            x_min = float(np.min(all_vals))
            x_max = float(np.max(all_vals))
            x = np.linspace(x_min, x_max, 400)

            def plot_gauss(data, color, label):
                if len(data) < 2:
                    return
                mu = float(np.mean(data))
                sigma = float(np.std(data, ddof=1))
                sigma = max(sigma, 1e-6)

                y = gaussian_pdf(x, mu, sigma)
                axd.plot(x, y, color=color, linewidth=2.0, label=f"{label} (n={len(data)})")

            plot_gauss(vals_other, "blue", "other")
            plot_gauss(vals_close, "orange", "closest")
            plot_gauss(vals_far, "green", "farthest")
            plot_gauss(vals_ref, "red", "refs")

            axd.set_title(f"{feat} @ idx={IDX}")
            axd.set_xlabel("value")
            axd.set_ylabel("density")
            axd.grid(True, alpha=0.3)
            axd.legend(fontsize=8)

        # hide unused axes
        for j in range(len(feat_cols), len(axes_dist)):
            axes_dist[j].axis("off")

        fig_dist.suptitle(f"Feature distributions at interpolated index {IDX}", y=1.02)
        fig_dist.tight_layout()

        # slowest among closest
        scores: list[tuple[str, float]] = []
        for cn in closest_cellnames:
            traj_w = traj_by_cell_weeks.get(cn)
            w_at = weeks_to_reach_soh(traj_w, SOH_SLOW_TARGET)
            if w_at is None:
                continue
            scores.append((cn, float(w_at)))

        if len(scores) == 0:
            print(f"\n[WARN] None of the closest cells reached SOH={SOH_SLOW_TARGET}. No black highlight.")
        else:
            scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)
            n_black = min(K_SLOWEST_BLACK, len(scores_sorted))
            slowest_black_cellnames = [cn for cn, _ in scores_sorted[:n_black]]

            print(
                f"\nSlowest-aging among closest {len(closest_cellnames)} "
                f"(max weeks to reach SOH={SOH_SLOW_TARGET}):"
            )
            for cn, w_at in scores_sorted[:n_black]:
                print(f" - {cn}: ~{w_at:.2f} weeks")

        # 3D scatter
        ref_mean = ref_sub[[x_col, y_col, z_col]].mean().to_numpy(dtype=float)

        fig_sc = plt.subplots(figsize=(8, 7))[0]
        ax_sc = fig_sc.add_subplot(111, projection="3d")

        ax_sc.scatter(exp_sub[x_col], exp_sub[y_col], exp_sub[z_col],
                      s=18, alpha=0.7, color="blue", label="exp (all)")

        ax_sc.scatter(exp_sub.loc[farthest_idx, x_col], exp_sub.loc[farthest_idx, y_col], exp_sub.loc[farthest_idx, z_col],
                      s=70, alpha=0.95, color="green", edgecolors="k", linewidths=0.6, label=f"farthest {k2}")

        ax_sc.scatter(exp_sub.loc[closest_idx, x_col], exp_sub.loc[closest_idx, y_col], exp_sub.loc[closest_idx, z_col],
                      s=70, alpha=0.95, color="orange", edgecolors="k", linewidths=0.6, label=f"closest {k1}")

        if slowest_black_cellnames:
            row_black = exp_sub[exp_sub["cell_name"].isin(slowest_black_cellnames)]
            if not row_black.empty:
                ax_sc.scatter(row_black[x_col], row_black[y_col], row_black[z_col],
                              s=180, color="black", marker="*", edgecolors="k", linewidths=0.8,
                              label=f"slowest among closest (SOH={SOH_SLOW_TARGET})")

        ax_sc.scatter(ref_sub[x_col], ref_sub[y_col], ref_sub[z_col],
                      s=90, alpha=1.0, color="red", edgecolors="k", linewidths=0.8, label="refs")

        ax_sc.scatter([ref_mean[0]], [ref_mean[1]], [ref_mean[2]],
                      s=150, color="black", marker="X", label="mean(refs)")

        ax_sc.set_xlabel(x_col)
        ax_sc.set_ylabel(y_col)
        ax_sc.set_zlabel(z_col)
        ax_sc.set_title(f"3D feature space at SOH={target_soh_features}: distance to nearest(ref)")
        ax_sc.legend()
        ax_sc.view_init(elev=20, azim=45)
        fig_sc.tight_layout()

        added_orange_label = False
        added_orange_label_w = False
        added_orange_label_t = False

        for cn in closest_cellnames:
            traj_w = traj_by_cell_weeks.get(cn)
            if traj_w is None:
                continue

            ax.plot(
                traj_w["SOH"],
                color="orange",
                alpha=0.85,
                linewidth=1.0,
                label="closest exp (orange)" if not added_orange_label else None,
                zorder=4,
            )
            ax.set_ylim(0.8, 1.1)
            added_orange_label = True

            ax_w.plot(
                traj_w["weeks"],
                traj_w["SOH"],
                color="orange",
                alpha=0.85,
                linewidth=1.0,
                label="closest exp (orange)" if not added_orange_label_w else None,
                zorder=4,
            )
            ax_w.set_ylim(0.8, 1.1)
            added_orange_label_w = True

            traj_t = traj_by_cell_thr.get(cn)
            if traj_t is not None:
                ax_t.plot(
                    traj_t["throughput_cum"],
                    traj_t["SOH"],
                    color="orange",
                    alpha=0.85,
                    linewidth=1.0,
                    label="closest exp (orange)" if not added_orange_label_t else None,
                    zorder=4,
                )
                ax_t.set_ylim(0.8, 1.1)
                added_orange_label_t = True

        added_green_label = False
        added_green_label_w = False
        added_green_label_t = False

        for cn in farthest_cellnames:
            traj_w = traj_by_cell_weeks.get(cn)
            if traj_w is None:
                continue

            ax.plot(
                traj_w["SOH"],
                color="green",
                alpha=0.85,
                linewidth=1.0,
                label="farthest exp (green)" if not added_green_label else None,
                zorder=5,
            )
            added_green_label = True

            ax_w.plot(
                traj_w["weeks"],
                traj_w["SOH"],
                color="green",
                alpha=0.85,
                linewidth=1.0,
                label="farthest exp (green)" if not added_green_label_w else None,
                zorder=5,
            )
            added_green_label_w = True

            traj_t = traj_by_cell_thr.get(cn)
            if traj_t is not None:
                ax_t.plot(
                    traj_t["throughput_cum"],
                    traj_t["SOH"],
                    color="green",
                    alpha=0.85,
                    linewidth=1.0,
                    label="farthest exp (green)" if not added_green_label_t else None,
                    zorder=5,
                )
                added_green_label_t = True

    # -----------------------------
    # NEW: two-step Monte Carlo selection (no weighted cost)
    # 1) top MC_TOP_K by predicted dist_feat (feature similarity)
    # 2) from those, top MC_TOP_FEATURE by highest predicted FEATURE_RANK_COL
    # -----------------------------
    best = pd.DataFrame()
    if not dist_df.empty:
        if df_all is None or df_all.empty:
            print("[WARN] df_all empty; cannot train feature model for step 2.")
        elif FEATURE_RANK_COL not in df_all.columns:
            print(f"[WARN] FEATURE_RANK_COL not in df_all: {FEATURE_RANK_COL}")
        else:
            print("" + "=" * 90)
            print("Training surrogate XGB models for two-step Monte Carlo selection")
            print("=" * 90)

            model_dist, _, _, _, _ = train_xgb_no_val_and_shap(
                df_exp=df_exp,
                dist_df=dist_df,
                ref_names=REF_NAMES,
            )
            feat_series = pd.to_numeric(df_all[FEATURE_RANK_COL], errors="coerce")
            print(
                f"[DIAG] {FEATURE_RANK_COL}: n={int(feat_series.notna().sum())}, "
                f"unique={int(feat_series.nunique(dropna=True))}, "
                f"min={float(feat_series.min()) if feat_series.notna().any() else float('nan'):.6g}, "
                f"max={float(feat_series.max()) if feat_series.notna().any() else float('nan'):.6g}"
            )

            model_var = train_model_var_dqc(
                df_exp=df_exp,
                df_reg_table=df_all,
                target_col=FEATURE_RANK_COL,
            )

            mc_all = monte_carlo_best_conditions_for_distance(
                df_exp=df_exp,
                model=model_dist,
                ref_names=REF_NAMES,
                n_samples_per_temp=MC_SAMPLES_PER_TEMP,
                top_k=None,
            )
            if mc_all.empty:
                print("[WARN] Monte Carlo search returned no candidates.")
            else:
                mc_blocks = []
                for t in MC_TEMP_TARGETS:
                    block = mc_all[mc_all["temp"] == float(t)].nsmallest(
                        MC_TOP_PER_TEMP, "pred_dist_feat"
                    )
                    if block.empty:
                        print(f"[WARN] No candidates for temp={t}.")
                    else:
                        mc_blocks.append(block)

                if not mc_blocks:
                    print("[WARN] No per-temperature selections were created.")
                else:
                    mc_top = pd.concat(mc_blocks, ignore_index=True)

                    X_mc_feat, _ = make_features_from_raw(
                        mc_top, raw_conds=RAW_CONDS_DEFAULT, drop_raw_soc=True
                    )
                    pred_var = model_var.predict(X_mc_feat).astype(float)
                    mc_top["pred_var_dQc"] = pred_var

                    print("" + "=" * 90)
                    print(
                        f"Top {len(mc_top)} Monte Carlo by predicted dist_feat "
                        f"(per-temp: {MC_TOP_PER_TEMP} each for {MC_TEMP_TARGETS})"
                    )
                    print("=" * 90)
                    print(mc_top.to_string(index=False))
                    print("Conditions only (top MC by dist_feat):")
                    print(mc_top[exp_conds].to_string(index=False))

                    best = (
                        mc_top.sort_values("pred_var_dQc", ascending=False)
                        .head(MC_TOP_FEATURE)
                        .reset_index(drop=True)
                    )
                print("" + "=" * 90)
                print(
                    f"Top {len(best)} within closest {len(mc_top)} by predicted {FEATURE_RANK_COL}"
                )
                print("=" * 90)
                print(best.to_string(index=False))
                print("Conditions only (final top by feature):")
                print(best[exp_conds].to_string(index=False))
    else:
        print("[WARN] dist_df empty; skipping two-step Monte Carlo selection.")

    # -----------------------------
    # Overlay trajectories: closest = orange, farthest = green, slowest among closest = black
    # -----------------------------
    added_orange_label = added_orange_label_w = added_orange_label_t = False
    for cn in closest_cellnames:
        traj_w = traj_by_cell_weeks.get(cn)
        if traj_w is None:
            continue

        ax.plot(traj_w["SOH"], color="orange", alpha=0.85, linewidth=1.0,
                label="closest exp (orange)" if not added_orange_label else None, zorder=4)
        added_orange_label = True

        ax_w.plot(traj_w["weeks"], traj_w["SOH"], color="orange", alpha=0.85, linewidth=1.0,
                  label="closest exp (orange)" if not added_orange_label_w else None, zorder=4)
        added_orange_label_w = True

        traj_t = traj_by_cell_thr.get(cn)
        if traj_t is not None:
            ax_t.plot(traj_t["throughput_cum"], traj_t["SOH"], color="orange", alpha=0.85, linewidth=1.0,
                      label="closest exp (orange)" if not added_orange_label_t else None, zorder=4)
            added_orange_label_t = True

    added_green_label = added_green_label_w = added_green_label_t = False
    for cn in farthest_cellnames:
        traj_w = traj_by_cell_weeks.get(cn)
        if traj_w is None:
            continue

        ax.plot(traj_w["SOH"], color="green", alpha=0.85, linewidth=1.0,
                label="farthest exp (green)" if not added_green_label else None, zorder=5)
        added_green_label = True

        ax_w.plot(traj_w["weeks"], traj_w["SOH"], color="green", alpha=0.85, linewidth=1.0,
                  label="farthest exp (green)" if not added_green_label_w else None, zorder=5)
        added_green_label_w = True

        traj_t = traj_by_cell_thr.get(cn)
        if traj_t is not None:
            ax_t.plot(traj_t["throughput_cum"], traj_t["SOH"], color="green", alpha=0.85, linewidth=1.0,
                      label="farthest exp (green)" if not added_green_label_t else None, zorder=5)
            added_green_label_t = True

    added_black_label = False
    for cn in slowest_black_cellnames:
        traj_w_black = traj_by_cell_weeks.get(cn)
        if traj_w_black is None:
            continue

        ax.plot(traj_w_black["SOH"], color="black", alpha=1.0, linewidth=2.4,
                label="slowest among closest (black)" if not added_black_label else None, zorder=10)
        ax_w.plot(traj_w_black["weeks"], traj_w_black["SOH"], color="black", alpha=1.0, linewidth=2.4,
                  label="slowest among closest (black)" if not added_black_label else None, zorder=10)

        traj_t_black = traj_by_cell_thr.get(cn)
        if traj_t_black is not None:
            ax_t.plot(traj_t_black["throughput_cum"], traj_t_black["SOH"], color="black", alpha=1.0, linewidth=2.4,
                      label="slowest among closest (black)" if not added_black_label else None, zorder=10)

        added_black_label = True

    # -----------------------------
    # Gaussian distributions of index where SOH first reaches SOH_DIST_TARGET
    # Groups:
    #   ORANGE = closest_cellnames
    #   GREEN  = farthest_cellnames
    #   BLUE   = all other exp cells (excluding orange & green)
    # -----------------------------
    exp_all = df_exp["cell_name"].dropna().unique().tolist()
    orange_cells = list(dict.fromkeys(closest_cellnames))
    green_cells = list(dict.fromkeys(farthest_cellnames))
    orange_set = set(orange_cells)
    green_set = set(green_cells)
    blue_cells = [cn for cn in exp_all if (cn not in orange_set) and (cn not in green_set)]

    idx_blue, idx_orange, idx_green = [], [], []

    for cn in blue_cells:
        idx = idx_first_reach_soh(traj_by_cell_weeks.get(cn), SOH_DIST_TARGET)
        if idx is not None:
            idx_blue.append(idx)

    for cn in orange_cells:
        idx = idx_first_reach_soh(traj_by_cell_weeks.get(cn), SOH_DIST_TARGET)
        if idx is not None:
            idx_orange.append(idx)

    for cn in green_cells:
        idx = idx_first_reach_soh(traj_by_cell_weeks.get(cn), SOH_DIST_TARGET)
        if idx is not None:
            idx_green.append(idx)

    print(f"\nIndex where interpolated SOH first reaches <= {SOH_DIST_TARGET}:")
    print(f"  BLUE   (other exp): n={len(idx_blue)}   mean={np.mean(idx_blue) if idx_blue else np.nan:.2f}")
    print(f"  ORANGE (closest)  : n={len(idx_orange)} mean={np.mean(idx_orange) if idx_orange else np.nan:.2f}")
    print(f"  GREEN  (farthest) : n={len(idx_green)}  mean={np.mean(idx_green) if idx_green else np.nan:.2f}")

    fig_g, ax_g = plt.subplots(figsize=(9, 5))

    def plot_gauss(axx, data, label, color, bins=25):
        if len(data) < 2:
            return
        data = np.asarray(data, dtype=float)
        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=1))
        sigma = max(sigma, 1e-6)

        axx.hist(data, bins=bins, density=True, alpha=0.25, color=color)

        x_min = float(np.min(data) - 3.0 * sigma)
        x_max = float(np.max(data) + 3.0 * sigma)
        x = np.linspace(x_min, x_max, 400)
        y = gaussian_pdf(x, mu, sigma)
        axx.plot(x, y, color=color, linewidth=2.0, label=f"{label}: μ={mu:.1f}, σ={sigma:.1f}, n={len(data)}")

    plot_gauss(ax_g, idx_blue, "BLUE (other exp)", "blue")
    plot_gauss(ax_g, idx_orange, "ORANGE (closest)", "orange")
    plot_gauss(ax_g, idx_green, "GREEN (farthest)", "green")

    ax_g.set_ylim(0, 1.5)
    ax_g.set_xlabel(f"index where interpolated SOH first <= {SOH_DIST_TARGET}")
    ax_g.set_ylabel("density")
    ax_g.set_title(f"Gaussian distributions of reach-index at SOH={SOH_DIST_TARGET} (interpolated SOH)")
    ax_g.grid(True, alpha=0.3)
    ax_g.legend(loc="best")
    fig_g.tight_layout()

    # -----------------------------
    # Finalize plots
    # -----------------------------
    ax.set_xlabel("index")
    ax.set_ylabel("SOH")
    ax.set_title(
        f"SOH trajectories (index-x, method={interp_method}) "
        f"(plot interpolation target SOH={target_soh_plot}; "
        f"feature comparison at SOH={target_soh_features})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(bottom=0.8)
    ax.set_ylim(top=1.1)
    fig.tight_layout()

    ax_w.set_xlabel("weeks")
    ax_w.set_ylabel("SOH")
    ax_w.set_title(
        f"SOH trajectories vs weeks (method={interp_method}) "
        f"(plot interpolation target SOH={target_soh_plot}; "
        f"feature comparison at SOH={target_soh_features})"
    )
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="best")
    ax_w.set_ylim(bottom=0.8)
    ax_w.set_ylim(top=1.1)
    fig_w.tight_layout()

    ax_t.set_xlabel("throughput_cum")
    ax_t.set_ylabel("SOH")
    ax_t.set_title(
        f"SOH trajectories vs throughput_cum (method={interp_method}) "
        f"(plot interpolation target SOH={target_soh_plot}; "
        f"feature comparison at SOH={target_soh_features})"
    )
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="best")
    ax_t.set_ylim(bottom=0.8)
    ax_t.set_ylim(top=1.1)
    fig_t.tight_layout()

    df_exp["mean_soc"] = 0.5 * (
        pd.to_numeric(df_exp["soc_start"], errors="coerce")
        + pd.to_numeric(df_exp["soc_end"], errors="coerce")
    )

    soc_bins = [0, 33, 66, 100]
    soc_labels = ["0–33", "33–66", "66–100"]
    df_exp["soc_group"] = pd.cut(df_exp["mean_soc"], bins=soc_bins, labels=soc_labels, include_lowest=True)

    group_colors = {"0–33": "red", "33–66": "green", "66–100": "blue"}

    fig_soc, ax_soc = plt.subplots(figsize=(9, 5))
    fig_soc_w, ax_soc_w = plt.subplots(figsize=(9, 5))
    fig_soc_t, ax_soc_t = plt.subplots(figsize=(9, 5))

    added_labels = {g: False for g in soc_labels}

    for _, row in df_exp.iterrows():
        cn = row["cell_name"]
        grp = row["soc_group"]
        if pd.isna(grp):
            continue

        color = group_colors[grp]
        label = f"mean SOC {grp}" if not added_labels[grp] else None

        traj_w = traj_by_cell_weeks.get(cn)
        if traj_w is not None:
            ax_soc.plot(traj_w["SOH"], color=color, alpha=0.8, linewidth=1.0, label=label)
            ax_soc_w.plot(traj_w["weeks"], traj_w["SOH"], color=color, alpha=0.8, linewidth=1.0, label=label)

        traj_t = traj_by_cell_thr.get(cn)
        if traj_t is not None:
            ax_soc_t.plot(traj_t["throughput_cum"], traj_t["SOH"], color=color, alpha=0.8, linewidth=1.0, label=label)

        if label is not None:
            added_labels[grp] = True

    # overlay references in black dashed so they don't clash
    for ref_name in REF_NAMES:
        traj_w = traj_by_cell_weeks.get(ref_name)
        if traj_w is not None:
            ax_soc.plot(traj_w["SOH"], color="black", linestyle="--", alpha=0.9, linewidth=1.8)
            ax_soc_w.plot(traj_w["weeks"], traj_w["SOH"], color="black", linestyle="--", alpha=0.9, linewidth=1.8)
        traj_t = traj_by_cell_thr.get(ref_name)
        if traj_t is not None:
            ax_soc_t.plot(traj_t["throughput_cum"], traj_t["SOH"], color="black", linestyle="--", alpha=0.9,
                          linewidth=1.8)

    for a, xl, title_suffix in [
        (ax_soc, "index", "index-based"),
        (ax_soc_w, "weeks", "weeks-based"),
        (ax_soc_t, "throughput_cum", "throughput-based"),
    ]:
        a.set_xlabel(xl)
        a.set_ylabel("SOH")
        a.set_title(f"SOH trajectories by mean SOC (red=low → blue=high) [{title_suffix}]")
        a.set_ylim(0.8, 1.1)
        a.grid(True, alpha=0.3)
        a.legend(loc="best")

    fig_soc.tight_layout()
    fig_soc_w.tight_layout()
    fig_soc_t.tight_layout()

    SOH_STEP_TARGET = 0.96
    FEATURE_STEP = 16

    _closest_set = set(closest_cellnames) if closest_cellnames else set()
    _farthest_set = set(farthest_cellnames) if farthest_cellnames else set()
    _ref_set = set(REF_NAMES)

    # --- Save all interpolated features + step number to folder ---
    interp_feat_dir = here / "interp_feature"
    interp_feat_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for cname, df_interp in traj_interp_weeks_by_cell.items():
        if "SOH" not in df_interp.columns:
            continue

        soh_vals = df_interp["SOH"].to_numpy(dtype=float)

        # compute step to SOH_STEP_TARGET (None if never reached)
        last_soh = soh_vals[~np.isnan(soh_vals)]
        step_to_target = None
        if len(last_soh) > 0 and last_soh[-1] <= SOH_STEP_TARGET:
            hits = np.where(soh_vals <= SOH_STEP_TARGET)[0]
            if len(hits) > 0 and int(hits[0]) > 0:
                step_to_target = int(hits[0])

        # save full interpolated trajectory as CSV
        df_save = df_interp.copy()
        df_save.insert(0, "cell_name", cname)
        df_save["step_to_soh_target"] = step_to_target  # same value on every row
        df_save.to_csv(interp_feat_dir / f"{cname}.csv", index=False)

        # one summary row per cell
        row = {"cell_name": cname, "step_to_soh_target": step_to_target}
        # store every feature at FEATURE_STEP if available
        for col in df_interp.columns:
            arr = df_interp[col].to_numpy(dtype=float)
            if len(arr) > FEATURE_STEP:
                row[f"{col}_at_step{FEATURE_STEP}"] = arr[FEATURE_STEP]
            else:
                row[f"{col}_at_step{FEATURE_STEP}"] = np.nan
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(interp_feat_dir / "_summary_all_cells.csv", index=False)
    print(f"\nSaved {len(summary_rows)} cell files + summary to {interp_feat_dir}")

    # --- Log-log scatter plots for ALL mean*/var* features (farthest excluded) ---
    # x = log(|feature value at FEATURE_STEP|)
    # y = log(steps to SOH <= SOH_STEP_TARGET)

    # collect all candidate interpolated feature names
    candidate_loglog_features = set()
    for _, df_interp in traj_interp_weeks_by_cell.items():
        for col in df_interp.columns:
            if col.startswith("mean") or col.startswith("var"):
                candidate_loglog_features.add(col)

    candidate_loglog_features = sorted(candidate_loglog_features)

    if len(candidate_loglog_features) == 0:
        print("\n[WARN] No interpolated features starting with 'mean' or 'var' found for log-log plots.")
    else:
        n_feat = len(candidate_loglog_features)
        ncols = 3
        nrows = math.ceil(n_feat / ncols)

        fig_loggrid, axes_loggrid = plt.subplots(
            nrows, ncols,
            figsize=(5.8 * ncols, 4.5 * nrows),
            sharex=False,
            sharey=False
        )
        axes_loggrid = np.array(axes_loggrid).ravel()

        # optional summary table
        loglog_summary_rows = []

        for i, feat_name in enumerate(candidate_loglog_features):
            ax_ll = axes_loggrid[i]

            log_feat_arr = []
            log_steps_arr = []
            color_arr = []

            for cname, df_interp in traj_interp_weeks_by_cell.items():
                if cname in _farthest_set:
                    continue
                if "SOH" not in df_interp.columns:
                    continue
                if feat_name not in df_interp.columns:
                    continue

                soh_vals = df_interp["SOH"].to_numpy(dtype=float)

                # must actually reach target SOH
                last_soh = soh_vals[~np.isnan(soh_vals)]
                if len(last_soh) == 0 or last_soh[-1] > SOH_STEP_TARGET:
                    continue

                n_steps = interp_step_at_soh(df_interp, SOH_STEP_TARGET)
                if n_steps is None or n_steps <= 0:
                    continue

                feat_vals = df_interp[feat_name].to_numpy(dtype=float)
                if len(feat_vals) <= FEATURE_STEP:
                    continue

                fv = feat_vals[FEATURE_STEP]
                if not np.isfinite(fv):
                    continue
                if fv == 0.0:
                    continue

                # same transform style as your original code
                log_feat_arr.append(np.log(np.abs(fv)))
                log_steps_arr.append(np.log(n_steps))

                if cname in _ref_set:
                    color_arr.append("red")
                elif cname in _closest_set:
                    color_arr.append("orange")
                else:
                    color_arr.append("blue")

            log_feat_arr = np.array(log_feat_arr, dtype=float)
            log_steps_arr = np.array(log_steps_arr, dtype=float)

            if len(log_feat_arr) < 2:
                ax_ll.set_title(f"{feat_name}\ninsufficient data")
                ax_ll.grid(True, alpha=0.3)
                loglog_summary_rows.append({
                    "feature": feat_name,
                    "n": len(log_feat_arr),
                    "slope": np.nan,
                    "intercept": np.nan,
                    "corr_r": np.nan,
                })
                continue

            # scatter by group color
            label_map = {"blue": "other", "orange": "closest", "red": "refs"}
            for cv in ["blue", "orange", "red"]:
                m = np.array([c == cv for c in color_arr])
                if not m.any():
                    continue
                ax_ll.scatter(
                    log_feat_arr[m],
                    log_steps_arr[m],
                    c=cv,
                    s=35,
                    alpha=0.75,
                    edgecolors="k",
                    linewidths=0.35,
                    label=label_map[cv],
                )

            # linear fit in log-log space
            coeffs = np.polyfit(log_feat_arr, log_steps_arr, 1)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])

            xfit = np.linspace(log_feat_arr.min(), log_feat_arr.max(), 200)
            yfit = np.polyval(coeffs, xfit)
            ax_ll.plot(
                xfit, yfit,
                "k--", lw=1.3,
                label=f"slope={slope:.3f}, int={intercept:.3f}"
            )

            r = float(np.corrcoef(log_feat_arr, log_steps_arr)[0, 1])

            ax_ll.set_title(f"{feat_name}\nn={len(log_feat_arr)}, r={r:.3f}")
            ax_ll.set_xlabel(f"log(|{feat_name}|) @ step {FEATURE_STEP}")
            ax_ll.set_ylabel(f"log(steps to SOH \u2264 {SOH_STEP_TARGET})")
            ax_ll.grid(True, alpha=0.3)
            ax_ll.legend(loc="best", fontsize=8)

            loglog_summary_rows.append({
                "feature": feat_name,
                "n": len(log_feat_arr),
                "slope": slope,
                "intercept": intercept,
                "corr_r": r,
            })

        # hide unused axes
        for j in range(len(candidate_loglog_features), len(axes_loggrid)):
            axes_loggrid[j].axis("off")

        fig_loggrid.suptitle(
            f"Log-log relationships for all mean*/var* features\n"
            f"x = log(|feature at step {FEATURE_STEP}|), "
            f"y = log(steps to SOH \u2264 {SOH_STEP_TARGET})\n"
            f"(farthest excluded, cells not reaching target excluded)",
            y=0.995
        )
        fig_loggrid.tight_layout(rect=[0, 0, 1, 0.96])

        # save summary table
        df_loglog_summary = pd.DataFrame(loglog_summary_rows).sort_values(
            by="corr_r", ascending=False, na_position="last"
        )
        df_loglog_summary.to_csv(interp_feat_dir / "_loglog_summary_mean_var_features.csv", index=False)

        print("\nSaved log-log summary:")
        print(interp_feat_dir / "_loglog_summary_mean_var_features.csv")
        print(df_loglog_summary.to_string(index=False))

        plt.show()

    '''
    # --- Scatter plot (farthest excluded) ---
    log_feat_arr = []
    log_steps_arr = []
    color_arr = []

    for cname, df_interp in traj_interp_weeks_by_cell.items():
        if cname in _farthest_set:
            continue
        if "SOH" not in df_interp.columns or "mean_d_dqdv_m_c" not in df_interp.columns:
            continue

        soh_vals = df_interp["SOH"].to_numpy(dtype=float)

        last_soh = soh_vals[~np.isnan(soh_vals)]
        if len(last_soh) == 0 or last_soh[-1] > SOH_STEP_TARGET:
            continue

        n_steps = interp_step_at_soh(df_interp, SOH_STEP_TARGET)
        if n_steps is None or n_steps <= 0:
            continue

        feat_vals = df_interp["mean_d_dqdv_m_c"].to_numpy(dtype=float)
        if len(feat_vals) <= FEATURE_STEP:
            continue
        fv = feat_vals[FEATURE_STEP]
        if not np.isfinite(fv) or fv == 0.0:
            continue

        log_feat_arr.append(np.log(np.abs(fv)))
        log_steps_arr.append(np.log(n_steps))

        if cname in _ref_set:
            color_arr.append("red")
        elif cname in _closest_set:
            color_arr.append("orange")
        else:
            color_arr.append("blue")

    log_feat_arr = np.array(log_feat_arr, dtype=float)
    log_steps_arr = np.array(log_steps_arr, dtype=float)

    fig_loglog, ax_loglog = plt.subplots(figsize=(9, 6))

    if len(log_feat_arr) >= 2:
        _lmap = {"blue": "other", "orange": "closest", "red": "refs"}
        for cv in ["blue", "orange", "red"]:
            m = np.array([c == cv for c in color_arr])
            if not m.any():
                continue
            ax_loglog.scatter(
                log_feat_arr[m], log_steps_arr[m],
                c=cv, s=45, alpha=0.75, edgecolors="k", linewidths=0.4,
                label=_lmap[cv],
            )

        coeffs = np.polyfit(log_feat_arr, log_steps_arr, 1)
        xfit = np.linspace(log_feat_arr.min(), log_feat_arr.max(), 200)
        ax_loglog.plot(xfit, np.polyval(coeffs, xfit), "k--", lw=1.5,
                       label=f"fit: slope={coeffs[0]:.3f}, intercept={coeffs[1]:.3f}")

        r = np.corrcoef(log_feat_arr, log_steps_arr)[0, 1]
        ax_loglog.set_title(
            f"log|mean_d_dqdv_m_c| @ step {FEATURE_STEP}  vs  "
            f"log(steps to SOH\u2264{SOH_STEP_TARGET})\n"
            f"n={len(log_feat_arr)}, r={r:.3f}  "
            f"(farthest excluded, cells with last SOH>{SOH_STEP_TARGET} excluded)"
        )
    else:
        ax_loglog.set_title("Insufficient data for log\u2013log scatter")

    ax_loglog.set_xlabel(f"log( |mean_d_dqdv_m_c| )  at interpolated step {FEATURE_STEP}")
    ax_loglog.set_ylabel(f"log( steps to SOH \u2264 {SOH_STEP_TARGET} )")
    ax_loglog.grid(True, alpha=0.3)
    ax_loglog.legend(loc="best")
    fig_loglog.tight_layout()

    plt.show()
    '''

    if cells_below_08:
        cells_below_08_sorted = sorted(cells_below_08, key=lambda x: x[1])
        print("\nCells with SOH < 0.8 (min SOH shown):")
        for cn, m in cells_below_08_sorted:
            print(f" - {cn}: min SOH = {m:.3f}")
    else:
        print("\nNo cells reached SOH < 0.8.")

    return df_exp, df_ref, dist_df, best


if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Victus\PycharmProjects\ExpSpeed\out_lw\cell_feature")
    out_dir = Path(r"C:\Users\Victus\PycharmProjects\ExpSpeed\out_lw\feature_plots")

    # choose here:
    INTERP_METHOD = "linear"   # "cubic" or "linear"
    THROUGHPUT_MAX = 8e7      # set to None if you do NOT want the limit

    main(
        dir_path,
        out_dir,
        interp_method=INTERP_METHOD,
        throughput_max=THROUGHPUT_MAX,
    )
