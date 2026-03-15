# speed_MultiReaderReporter/main.py
from __future__ import annotations

from pathlib import Path
import sys
import math
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment

# --- ML (Monte Carlo surrogate model) ---
try:
    import xgboost as xgb
    _HAVE_XGB = True
except Exception:
    xgb = None
    _HAVE_XGB = False

# SHAP is optional
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


# ============================================================
# Monte Carlo surrogate feature pipeline (single source of truth)
# ============================================================
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
    - Replaces c_rate_chg==15 with 1.5 (your encoding fix)
    - Optionally drops soc_start/soc_end
    Returns: (X_features, feature_names)
    """
    X = df[raw_conds].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    X["soc"] = 0.5 * (X["soc_start"] + X["soc_end"])
    X["dod"] = X["soc_end"] - X["soc_start"]

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
    *,
    dist_col: str = "dist_w",
    raw_conds: list[str] = RAW_CONDS_DEFAULT,
    random_state: int = 42,
    shap_max_display: int = 20,
):
    """
    Fit XGBRegressor on ALL available exp data (no train/val split),
    target = dist_col (distance-to-nearest-ref).

    Returns: model, explainer(or None), X_feat, y, feat_names
    """
    if not _HAVE_XGB:
        raise RuntimeError("xgboost is not installed. Please `pip install xgboost` or disable surrogate training.")

    df = df_exp.merge(dist_df[["cell_name", dist_col]], on="cell_name", how="inner")
    df = df[~df["cell_name"].isin(ref_names)].copy()

    y = pd.to_numeric(df[dist_col], errors="coerce")
    X_feat, feat_names = make_features_from_raw(df, raw_conds=raw_conds, drop_raw_soc=True)

    ok = ~y.isna()
    ok &= np.isfinite(X_feat.to_numpy()).all(axis=1)

    X_feat = X_feat.loc[ok].reset_index(drop=True)
    y = y.loc[ok].to_numpy(dtype=float)

    if len(X_feat) < 10:
        raise ValueError(f"Not enough rows with valid features + target to train XGB model (got {len(X_feat)}).")

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
        # KernelExplainer works for any model, but can be slow.
        explainer = shap.KernelExplainer(lambda a: model.predict(np.asarray(a)), X_feat)
        shap_values = explainer.shap_values(X_feat)

        plt.figure()
        shap.summary_plot(shap_values, X_feat, show=False, max_display=shap_max_display)
        plt.title(f"SHAP summary — target: {dist_col}")
        plt.tight_layout()
        # plt.show()

        shap.summary_plot(shap_values, X_feat, plot_type="bar", show=False, max_display=shap_max_display)
        plt.title(f"SHAP importance — target: {dist_col}")
        plt.tight_layout()
        # plt.show()

    return model, explainer, X_feat, y, feat_names


def monte_carlo_best_conditions_for_distance(
    df_exp: pd.DataFrame,
    model,
    ref_names: list[str],
    *,
    raw_conds: list[str] = RAW_CONDS_DEFAULT,
    n_samples: int = 2000,
    top_k: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo search over RAW conditions (with your discrete constraints),
    engineer SAME features as training, predict, and return smallest top_k.

    Discrete:
      - temp in {15,25,40}
      - soc_start in {0..60 step 10}
      - soc_end in {20..100 step 10}
      - c_rate_chg in {0.5..1.5 step 0.1}
      - c_rate_dchg in {0.5..3.0 step 0.1}
    Constraints:
      - soc_end > soc_start
      - soc_end - soc_start > 10  (=> at least 20 with 10-step grids)

    Note: This only optimizes predicted *distance-to-reference* (your dist_w),
    not lifetime/aging speed.
    """
    rng = np.random.default_rng(random_state)

    df_train = df_exp[~df_exp["cell_name"].isin(ref_names)].copy()
    for c in raw_conds:
        if c in df_train.columns:
            df_train[c] = pd.to_numeric(df_train[c], errors="coerce")
    if "c_rate_chg" in df_train.columns:
        df_train.loc[df_train["c_rate_chg"] == 15, "c_rate_chg"] = 1.5

    allowed_temp = np.array([15.0, 25.0, 40.0], dtype=float)
    allowed_soc_start = np.arange(0, 61, 10, dtype=float)  # 0..60
    allowed_soc_end = np.arange(20, 101, 10, dtype=float)  # 20..100
    allowed_cur_cha = np.arange(0.5, 1.6, 0.1, dtype=float)  # 0.5..1.5
    allowed_cur_dis = np.arange(0.5, 3.1, 0.1, dtype=float)  # 0.5..3.0

    valid_pairs = np.array(
        [(s0, s1) for s0 in allowed_soc_start for s1 in allowed_soc_end if (s1 - s0) > 10],
        dtype=float,
    )
    if len(valid_pairs) == 0:
        raise ValueError("No valid (soc_start, soc_end) pairs under the constraints.")

    idx = rng.integers(0, len(valid_pairs), size=n_samples)
    soc_start = valid_pairs[idx, 0]
    soc_end = valid_pairs[idx, 1]

    temp = rng.choice(allowed_temp, size=n_samples, replace=True)
    c_rate_chg = rng.choice(allowed_cur_cha, size=n_samples, replace=True)
    c_rate_dchg = rng.choice(allowed_cur_dis, size=n_samples, replace=True)

    X_mc_raw = pd.DataFrame(
        {
            "soc_start": soc_start,
            "soc_end": soc_end,
            "c_rate_chg": c_rate_chg,
            "c_rate_dchg": c_rate_dchg,
            "temp": temp,
        }
    )[raw_conds].astype(float)

    X_mc_feat, _ = make_features_from_raw(X_mc_raw, raw_conds=raw_conds, drop_raw_soc=True)
    pred = model.predict(X_mc_feat)

    out = X_mc_raw.copy()
    out["soc"] = X_mc_feat["soc"].to_numpy()
    out["dod"] = X_mc_feat["dod"].to_numpy()
    out["pred_dist_w"] = pred  # predicted distance-to-nearest-ref (dist_w)

    best = out.nsmallest(top_k, "pred_dist_w").reset_index(drop=True)
    return best


# ============================================================
# Interpolation (returns SOH + reference axis)
# ============================================================
def load_and_interpolate(df: pd.DataFrame, target_soh: float, interpolation_typ: str):
    """
    Interpolate all columns (incl. SOH) onto a new reference grid using CubicSpline.
    Returns a dataframe that includes: interpolated feature columns + SOH + reference axis column.
    interpolation_typ: 'weeks' or 'throughput'
    """
    df = df.copy()
    df.iloc[0] = df.iloc[0].fillna(0)

    # compute SOH
    df["SOH"] = df["cap_ocv_dis"] / df["cap_ocv_dis"].iloc[0]

    if interpolation_typ == "weeks":
        ref_name = "weeks"
    elif interpolation_typ == "throughput":
        ref_name = "throughput_cum"
    else:
        raise ValueError("interpolation_typ must be 'weeks' or 'throughput'")

    # keep first row + rows where ref != 0
    mask = (df.index == 0) | (df[ref_name] != 0)
    df = df.loc[mask].reset_index(drop=True)

    # coerce to numeric (safe)
    df = df.apply(pd.to_numeric, errors="coerce")

    # abort if key columns are missing/invalid
    if df[ref_name].isna().any() or df["SOH"].isna().any():
        return None

    reference = df[ref_name].to_numpy(dtype=float)
    target_data = df["SOH"].to_numpy(dtype=float)

    # need strictly increasing x for CubicSpline
    keep = ~pd.Series(reference).duplicated(keep="first")
    df = df.loc[keep.values].reset_index(drop=True)

    reference = df[ref_name].to_numpy(dtype=float)
    target_data = df["SOH"].to_numpy(dtype=float)

    if len(reference) < 3:
        return None

    # interpolate EVERYTHING except ref (so SOH will be included)
    feature_cols = [c for c in df.columns if c != ref_name]
    feature = df[feature_cols].to_numpy(dtype=float)

    # find reference location where SOH reaches target_soh (SOH decreases => reverse)
    interpolated_ref = np.interp(target_soh, target_data[::-1], reference[::-1])

    # 15 points up to target, then continue with same spacing
    new_ref_points_cut = np.linspace(0.0, float(interpolated_ref), 15)
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

    # CubicSpline interpolation per column (fallback to linear if NaNs / errors)
    interpolated_columns = []
    for j in range(feature.shape[1]):
        yj = feature[:, j]

        if np.isnan(yj).any():
            interpolated_columns.append(np.interp(new_ref_points, reference, yj))
            continue

        try:
            cs = CubicSpline(reference, yj, bc_type="natural", extrapolate=False)
            y_new = cs(new_ref_points)
            interpolated_columns.append(y_new)
        except Exception:
            interpolated_columns.append(np.interp(new_ref_points, reference, yj))

    interpolated_data = np.stack(interpolated_columns, axis=1)
    out = pd.DataFrame(interpolated_data, columns=feature_cols)
    out[ref_name] = new_ref_points

    if "SOH" in out.columns:
        out["SOH"] = out["SOH"].clip(lower=0.0, upper=1.05)

    return out


def exp_row_from_first_line(df: pd.DataFrame, exp_conds: list[str]) -> pd.Series:
    """Return exp condition values from row 0 as a Series."""
    return df.loc[0, exp_conds]


def weeks_to_reach_soh(traj_weeks: pd.DataFrame, soh_target: float) -> float | None:
    """Return interpolated weeks when SOH reaches soh_target, else None."""
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

    idx = np.argsort(s)  # SOH ascending
    s_sorted = s[idx]
    w_sorted = w[idx]

    keep = ~pd.Series(s_sorted).duplicated(keep="first")
    s_sorted = s_sorted[keep.values]
    w_sorted = w_sorted[keep.values]
    if len(s_sorted) < 2:
        return None

    return float(np.interp(soh_target, s_sorted, w_sorted))


# ============================================================
# ND signature + ND Wasserstein (assignment)
# ============================================================
def feature_signature_nd(
    traj: pd.DataFrame,
    feature_cols: list[str],
    soh_hi: float,
    soh_lo: float,
    n_points: int,
    time_col: str = "weeks",
) -> np.ndarray | None:
    """
    Robust ND signature builder:
    - finds t1 at SOH=soh_lo
    - samples ts uniformly in [t0, t1]
    - interpolates each feature vs time using only non-NaN points
    """
    if traj is None or traj.empty:
        return None
    needed = {time_col, "SOH"} | set(feature_cols)
    if not needed.issubset(set(traj.columns)):
        return None

    t_all = pd.to_numeric(traj[time_col], errors="coerce").to_numpy(dtype=float)
    s_all = pd.to_numeric(traj["SOH"], errors="coerce").to_numpy(dtype=float)

    ok_ts = np.isfinite(t_all) & np.isfinite(s_all)
    if ok_ts.sum() < 3:
        return None

    t = t_all[ok_ts]
    s = s_all[ok_ts]

    if np.nanmin(s) > soh_lo:
        return None

    idx = np.argsort(s)
    s_sorted = s[idx]
    t_sorted = t[idx]
    keep_s = ~pd.Series(s_sorted).duplicated(keep="first")
    s_sorted = s_sorted[keep_s.values]
    t_sorted = t_sorted[keep_s.values]
    if len(s_sorted) < 2:
        return None

    t0 = float(t[0])
    t1 = float(np.interp(soh_lo, s_sorted, t_sorted))
    if not np.isfinite(t1) or t1 <= t0:
        return None

    ts = np.linspace(t0, t1, n_points)

    keep_t = ~pd.Series(t).duplicated(keep="first")
    t_u = t[keep_t.values]
    if len(t_u) < 2:
        return None

    sig_cols = []
    for col in feature_cols:
        f_all = pd.to_numeric(traj[col], errors="coerce").to_numpy(dtype=float)
        f = f_all[ok_ts]

        ok_f = np.isfinite(f) & np.isfinite(t)
        tf = t[ok_f]
        ff = f[ok_f]

        if len(tf) < 2:
            return None

        keep_tf = ~pd.Series(tf).duplicated(keep="first")
        tf = tf[keep_tf.values]
        ff = ff[keep_tf.values]
        if len(tf) < 2:
            return None

        if len(tf) >= 3:
            try:
                cs = CubicSpline(tf, ff, bc_type="natural", extrapolate=False)
                out = cs(ts)
            except Exception:
                out = np.interp(ts, tf, ff)
        else:
            out = np.interp(ts, tf, ff)

        if np.isnan(out).any():
            out = np.interp(ts, tf, ff)

        sig_cols.append(out)

    sig = np.stack(sig_cols, axis=1)
    if not np.isfinite(sig).all():
        return None
    return sig.astype(float)


def wasserstein_assignment(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """
    Mean matched Euclidean cost between two equal-weight point clouds via Hungarian assignment.
    Works for ANY dimension D (2D, 3D, ...).
    """
    if sig_a is None or sig_b is None:
        return float("nan")
    k = min(sig_a.shape[0], sig_b.shape[0])
    A = sig_a[:k]
    B = sig_b[:k]
    diff = A[:, None, :] - B[None, :, :]
    C = np.linalg.norm(diff, axis=2)
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())


# ============================================================
# Option C scaling: equalize feature weights across ALL cells
# ============================================================
def compute_feature_scales_from_signatures(
    all_sigs: list[np.ndarray],
    method: str = "robust_iqr",
    eps: float = 1e-12,
) -> np.ndarray:
    if len(all_sigs) == 0:
        raise ValueError("No signatures provided for scale computation.")
    X = np.concatenate(all_sigs, axis=0)
    if method == "robust_iqr":
        q25 = np.nanquantile(X, 0.25, axis=0)
        q75 = np.nanquantile(X, 0.75, axis=0)
        scale = q75 - q25
    elif method == "std":
        scale = np.nanstd(X, axis=0)
    elif method == "range":
        scale = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
    else:
        raise ValueError("method must be one of: 'robust_iqr', 'std', 'range'")

    scale = np.where(np.isfinite(scale), scale, 0.0)
    scale = np.maximum(scale, eps)
    return scale.astype(float)


def apply_feature_scaling(sig: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return sig / scale


# ============================================================
# Simple PCA-to-2D for signature visualization when D != 2
# ============================================================
def pca_project_to_2d(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[0] < 2:
        return points[:, :2].copy()
    X = points - np.mean(points, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    W = vt[:2].T
    return X @ W


# ============================================================
# NEW helper: Gaussian pdf for distributions
# ============================================================
def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = float(max(sigma, 1e-12))
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ============================================================
# Main
# ============================================================
def main(dir_path: Path, out_dir: Path):
    exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    target_soh_features = 0.995
    target_soh_plot = 0.975

    FEATURE_COLS = ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c"]
    # FEATURE_COLS = ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_m_d"]

    SCALE_METHOD = "robust_iqr"

    SOH_SIG_HI = 1.0
    SOH_SIG_LO = target_soh_features
    SIG_NPTS = 5

    REF_NAMES = ["SPEED_LW_reference_4", "SPEED_LW_reference_5", "SPEED_LW_reference_6"]

    K_CLOSEST_FEATURE = 20
    K_FARTHEST = 15
    K_SLOWEST_BLACK = 1
    SOH_SLOW_TARGET = 0.96

    RUN_MONTE_CARLO = True
    MC_SAMPLES = 200
    MC_TOPK = 20

    needed_cols = (
        ["CU_time"]
        + exp_conds
        + ["cap_ocv_dis"]
        + [
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c", "var_dQ_c",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
        ]
        + ["throughput_cum", "mean_d_dqdv_h_d", "mean_d_dqdv_l_d"]
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    added_blue_label = False
    added_red_label = False

    fig_w, ax_w = plt.subplots(figsize=(9, 5))
    added_blue_label_w = False
    added_red_label_w = False

    fig_t, ax_t = plt.subplots(figsize=(9, 5))
    added_blue_label_t = False
    added_red_label_t = False

    traj_by_cell_weeks: dict[str, pd.DataFrame] = {}
    traj_by_cell_thr: dict[str, pd.DataFrame] = {}

    rows = []
    rows_ref = []

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

        if any(c not in df.columns for c in exp_conds):
            print(f"[WARN] {cell_name}: missing exp_conds, skipping.")
            continue
        exp_row = exp_row_from_first_line(df, exp_conds)

        interp_cols_weeks = [
            "weeks",
            "cap_ocv_dis",
            "mean_d_dqdv_m_c",
            "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d",
            "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c",
            "mean_d_dqdv_l_c",
            "mean_d_dqdv_h_d",
            "mean_d_dqdv_l_d",
        ]
        missing = [c for c in interp_cols_weeks if c not in df.columns]
        if missing:
            print(f"[WARN] {cell_name}: missing {missing}, skipping.")
            continue

        df_filter_weeks = df[interp_cols_weeks].copy()

        interpolated_plot_weeks = load_and_interpolate(df_filter_weeks, target_soh_plot, interpolation_typ="weeks")
        if interpolated_plot_weeks is None or interpolated_plot_weeks.empty:
            print(f"[WARN] {cell_name}: interpolation (weeks) failed, skipping.")
            continue

        traj_by_cell_weeks[cell_name] = interpolated_plot_weeks.copy()

        interp_cols_thr = [
            "throughput_cum",
            "cap_ocv_dis",
            "mean_d_dqdv_m_c",
            "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d",
            "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c",
            "mean_d_dqdv_l_c",
            "mean_d_dqdv_h_d",
            "mean_d_dqdv_l_d",
        ]
        missing_thr = [c for c in interp_cols_thr if c not in df.columns]
        if not missing_thr:
            df_filter_thr = df[interp_cols_thr].copy()
            interpolated_plot_thr = load_and_interpolate(df_filter_thr, target_soh_plot, interpolation_typ="throughput")
            if interpolated_plot_thr is not None and not interpolated_plot_thr.empty:
                if "SOH" in interpolated_plot_thr.columns and "throughput_cum" in interpolated_plot_thr.columns:
                    traj_by_cell_thr[cell_name] = interpolated_plot_thr[["throughput_cum", "SOH"]].copy()

        is_ref_case = exp_row.isna().any() and ("LW_reference" in cell_name)

        if is_ref_case:
            label = "LW_reference (exp NaN)" if not added_red_label else None
            added_red_label = True
            ax.plot(interpolated_plot_weeks["SOH"], color="red", alpha=0.25, linewidth=0.6, label=label, zorder=1)

            label_w = "LW_reference (exp NaN)" if not added_red_label_w else None
            added_red_label_w = True
            ax_w.plot(
                interpolated_plot_weeks["weeks"], interpolated_plot_weeks["SOH"],
                color="red", alpha=0.25, linewidth=0.6, label=label_w, zorder=1
            )

            if cell_name in traj_by_cell_thr:
                label_t = "LW_reference (exp NaN)" if not added_red_label_t else None
                added_red_label_t = True
                ax_t.plot(
                    traj_by_cell_thr[cell_name]["throughput_cum"], traj_by_cell_thr[cell_name]["SOH"],
                    color="red", alpha=0.25, linewidth=0.6, label=label_t, zorder=1
                )
        else:
            label = "other" if not added_blue_label else None
            added_blue_label = True
            ax.plot(interpolated_plot_weeks["SOH"], color="blue", alpha=0.20, linewidth=0.55, label=label, zorder=2)

            label_w = "other" if not added_blue_label_w else None
            added_blue_label_w = True
            ax_w.plot(
                interpolated_plot_weeks["weeks"], interpolated_plot_weeks["SOH"],
                color="blue", alpha=0.20, linewidth=0.55, label=label_w, zorder=2
            )

            if cell_name in traj_by_cell_thr:
                label_t = "other" if not added_blue_label_t else None
                added_blue_label_t = True
                ax_t.plot(
                    traj_by_cell_thr[cell_name]["throughput_cum"], traj_by_cell_thr[cell_name]["SOH"],
                    color="blue", alpha=0.20, linewidth=0.55, label=label_t, zorder=2
                )

        idx_feat = (interpolated_plot_weeks["SOH"] - target_soh_features).abs().idxmin()
        row_feat = interpolated_plot_weeks.loc[idx_feat]
        combined_row = pd.concat([pd.Series({"cell_name": cell_name}), exp_row, row_feat], axis=0)

        if exp_row.isna().any():
            rows_ref.append(combined_row)
        else:
            rows.append(combined_row)

        print(f"{cell_name}: success")

    df_ref = pd.DataFrame(rows_ref).reset_index(drop=True)
    df_exp = pd.DataFrame(rows).reset_index(drop=True)

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

    ref_sigs_raw: dict[str, np.ndarray] = {}
    for rn in REF_NAMES:
        sig = feature_signature_nd(
            traj_by_cell_weeks.get(rn),
            FEATURE_COLS,
            soh_hi=SOH_SIG_HI,
            soh_lo=SOH_SIG_LO,
            n_points=SIG_NPTS,
            time_col="weeks",
        )
        if sig is not None:
            ref_sigs_raw[rn] = sig

    if len(ref_sigs_raw) == 0:
        print(f"[WARN] No valid reference signatures for features {FEATURE_COLS}.")
        closest_cellnames, farthest_cellnames, slowest_black_cellnames = [], [], []
        dist_df = pd.DataFrame(columns=["cell_name", "dist_w", "best_ref"])
    else:
        exp_names = df_exp["cell_name"].dropna().unique().tolist()

        exp_sigs_raw: dict[str, np.ndarray] = {}
        for cn in exp_names:
            sig = feature_signature_nd(
                traj_by_cell_weeks.get(cn),
                FEATURE_COLS,
                soh_hi=SOH_SIG_HI,
                soh_lo=SOH_SIG_LO,
                n_points=SIG_NPTS,
                time_col="weeks",
            )
            if sig is not None:
                exp_sigs_raw[cn] = sig

        all_sig_list = list(ref_sigs_raw.values()) + list(exp_sigs_raw.values())
        scale = compute_feature_scales_from_signatures(all_sig_list, method=SCALE_METHOD)

        ref_sigs = {k: apply_feature_scaling(v, scale) for k, v in ref_sigs_raw.items()}
        exp_sigs = {k: apply_feature_scaling(v, scale) for k, v in exp_sigs_raw.items()}

        dist_list: list[tuple[str, float, str]] = []
        for cn, sig_e in exp_sigs.items():
            best_d = np.inf
            best_r = ""
            for rn, sig_r in ref_sigs.items():
                d = wasserstein_assignment(sig_e, sig_r)
                if d < best_d:
                    best_d = d
                    best_r = rn
            if np.isfinite(best_d) and best_r:
                dist_list.append((cn, float(best_d), best_r))

        dist_df = (
            pd.DataFrame(dist_list, columns=["cell_name", "dist_w", "best_ref"])
            .sort_values("dist_w", ascending=True)
            .reset_index(drop=True)
        )

        if dist_df.empty:
            print("[WARN] No experimental signatures were valid for Wasserstein distance.")
            closest_cellnames, farthest_cellnames, slowest_black_cellnames = [], [], []
        else:
            closest_cellnames = dist_df["cell_name"].head(min(K_CLOSEST_FEATURE, len(dist_df))).tolist()
            farthest_cellnames = dist_df["cell_name"].tail(min(K_FARTHEST, len(dist_df))).tolist()

            cond_map = (
                df_exp[["cell_name"] + exp_conds]
                .dropna(subset=["cell_name"])
                .set_index("cell_name")
                .to_dict(orient="index")
            )

            print("\n" + "-" * 90)
            print(
                f"Closest {len(closest_cellnames)} exp cells (orange) by Wasserstein (assignment) on features={FEATURE_COLS}\n"
                f"Signature SOH {SOH_SIG_HI} -> {target_soh_features} (n={SIG_NPTS}) | "
                f"Option C scaling='{SCALE_METHOD}' using ALL cells | scales={scale}"
            )
            print("-" * 90)
            for i, cn in enumerate(closest_cellnames, start=1):
                row = dist_df.loc[dist_df["cell_name"] == cn].iloc[0]
                conds = cond_map.get(cn, {})
                cond_str = ", ".join(f"{k}={conds.get(k, np.nan)}" for k in exp_conds)
                print(f"{i:>2}. {cn:>25} | d={row['dist_w']:.6g} | best_ref={row['best_ref']} | {cond_str}")

            scores: list[tuple[str, float]] = []
            for cn in closest_cellnames:
                traj_w = traj_by_cell_weeks.get(cn)
                if traj_w is None:
                    continue
                w_at = weeks_to_reach_soh(traj_w[["weeks", "SOH"]], SOH_SLOW_TARGET)
                if w_at is None:
                    continue
                scores.append((cn, float(w_at)))

            if len(scores) == 0:
                print(f"\n[WARN] None of the closest cells reached SOH={SOH_SLOW_TARGET}. No black highlight.")
                slowest_black_cellnames = []
            else:
                scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)
                slowest_black_cellnames = [cn for cn, _ in scores_sorted[: min(K_SLOWEST_BLACK, len(scores_sorted))]]
                print(f"\nSlowest-aging among closest (max weeks to reach SOH={SOH_SLOW_TARGET}):")
                for cn, w_at in scores_sorted[: min(K_SLOWEST_BLACK, len(scores_sorted))]:
                    print(f" - {cn}: ~{w_at:.2f} weeks")

            # ============================================================
            # ADD: FEATURE SUBPLOTS + GAUSSIAN DISTRIBUTIONS @ IDX
            # (added only; does not replace your existing figures)
            # ============================================================
            x_feat = "weeks"
            IDX = 25  # you asked for index 25

            # feature columns available in the interpolated dfs
            interp_feature_cols = [
                "cap_ocv_dis",
                "mean_d_dqdv_m_c", "var_d_dqdv_m_c",
                "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
                "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
                "mean_d_dqdv_h_d", "mean_d_dqdv_l_d",
                "SOH",
            ]

            # only keep ones that exist somewhere
            feat_cols = []
            for f in interp_feature_cols:
                for _, df_interp in traj_by_cell_weeks.items():
                    if f in df_interp.columns:
                        feat_cols.append(f)
                        break
            feat_cols = list(dict.fromkeys(feat_cols))

            closest_set = set(closest_cellnames)
            farthest_set = set(farthest_cellnames)
            ref_set = set(REF_NAMES)

            n_feats = len(feat_cols)
            if n_feats > 0:
                ncols = 3
                nrows = math.ceil(n_feats / ncols)

                # ---- (A) feature trajectories vs weeks (subplots) ----
                fig_feat, axes = plt.subplots(
                    nrows, ncols,
                    figsize=(5.5 * ncols, 3.5 * nrows),
                    sharex=True
                )
                axes = np.array(axes).ravel()

                for i_f, feat in enumerate(feat_cols):
                    axf = axes[i_f]
                    for cell_name, df_interp in traj_by_cell_weeks.items():
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

                # ---- (B) Gaussian distributions at index IDX (NO trajectory lines) ----
                fig_dist, axes_dist = plt.subplots(
                    nrows, ncols,
                    figsize=(5.5 * ncols, 3.5 * nrows),
                    sharex=False
                )
                axes_dist = np.array(axes_dist).ravel()

                for i_f, feat in enumerate(feat_cols):
                    axd = axes_dist[i_f]

                    vals_ref, vals_close, vals_far, vals_other = [], [], [], []

                    for cell_name, df_interp in traj_by_cell_weeks.items():
                        if feat not in df_interp.columns:
                            continue
                        yy = df_interp[feat].to_numpy(dtype=float)
                        if len(yy) <= IDX:
                            continue
                        v = float(yy[IDX])
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
                        axd.set_title(f"{feat} (insufficient @ idx={IDX})")
                        axd.axis("off")
                        continue

                    x_min = float(np.min(all_vals))
                    x_max = float(np.max(all_vals))
                    xgrid = np.linspace(x_min, x_max, 400)

                    def plot_gauss_curve(data, color, label):
                        if len(data) < 2:
                            return
                        mu = float(np.mean(data))
                        sigma = float(np.std(data, ddof=1))
                        sigma = max(sigma, 1e-6)
                        ypdf = gaussian_pdf(xgrid, mu, sigma)
                        axd.plot(xgrid, ypdf, color=color, linewidth=2.0, label=f"{label} (n={len(data)})")

                    # Gaussian curves only
                    plot_gauss_curve(vals_other, "blue", "other")
                    plot_gauss_curve(vals_close, "orange", "closest")
                    plot_gauss_curve(vals_far, "green", "farthest")
                    plot_gauss_curve(vals_ref, "red", "refs")

                    axd.set_title(f"{feat} @ idx={IDX}")
                    axd.set_xlabel("value")
                    axd.set_ylabel("density")
                    axd.grid(True, alpha=0.3)
                    axd.legend(fontsize=8)

                for j in range(n_feats, len(axes_dist)):
                    axes_dist[j].axis("off")

                fig_dist.suptitle(f"Gaussian feature distributions at interpolated index {IDX}", y=1.02)
                fig_dist.tight_layout()

    # -----------------------------
    # NEW: Train surrogate model and run Monte Carlo selection
    # -----------------------------
    best_mc = pd.DataFrame()
    if RUN_MONTE_CARLO and (dist_df is not None) and (not dist_df.empty):
        if not _HAVE_XGB:
            print("\n[WARN] xgboost not installed => skipping surrogate + Monte Carlo. (`pip install xgboost`)")
        else:
            print("\n" + "=" * 90)
            print("Training surrogate XGB model: exp_conds -> Wasserstein distance-to-nearest-ref (dist_w)")
            print("=" * 90)
            try:
                model, explainer, X_feat, y, feat_names = train_xgb_no_val_and_shap(
                    df_exp=df_exp,
                    dist_df=dist_df,
                    ref_names=REF_NAMES,
                    dist_col="dist_w",
                )

                best_mc = monte_carlo_best_conditions_for_distance(
                    df_exp=df_exp,
                    model=model,
                    ref_names=REF_NAMES,
                    n_samples=MC_SAMPLES,
                    top_k=MC_TOPK,
                )

                print("\n" + "=" * 90)
                print(f"Top {MC_TOPK} Monte Carlo conditions with smallest predicted dist_w (nearest-ref Wasserstein)")
                print("=" * 90)
                print(best_mc.to_string(index=False))
            except Exception as e:
                print(f"\n[WARN] Surrogate/Monte Carlo failed: {e}")

    # -----------------------------
    # Visualize Wasserstein distances (sorted curve + histogram)
    # -----------------------------
    if "dist_df" in locals() and dist_df is not None and not dist_df.empty:
        y = dist_df["dist_w"].to_numpy(float)

        fig_d, ax_d = plt.subplots(figsize=(9, 4))
        ax_d.plot(np.arange(len(y)), y, linewidth=1.4)
        ax_d.set_title(
            f"Wasserstein(assignment) distance to nearest reference | features={FEATURE_COLS}\n"
            f"signature SOH {SOH_SIG_HI}->{target_soh_features}, n={SIG_NPTS} | scaling={SCALE_METHOD}"
        )
        ax_d.set_xlabel("cells (sorted by distance)")
        ax_d.set_ylabel("distance (mean matched Euclidean cost)")
        ax_d.grid(True, alpha=0.3)

        n_cl = min(K_CLOSEST_FEATURE, len(dist_df))
        n_fa = min(K_FARTHEST, len(dist_df))
        ax_d.scatter(np.arange(n_cl), y[:n_cl], s=35, color="orange", label="closest")
        ax_d.scatter(np.arange(len(y) - n_fa, len(y)), y[-n_fa:], s=35, color="green", label="farthest")
        ax_d.legend(loc="best")
        fig_d.tight_layout()

        fig_h, ax_h = plt.subplots(figsize=(7, 4))
        ax_h.hist(y, bins=30)
        ax_h.set_title("Histogram of Wasserstein distances")
        ax_h.set_xlabel("distance")
        ax_h.set_ylabel("count")
        ax_h.grid(True, alpha=0.3)
        fig_h.tight_layout()

        try:
            n_show = 3
            show_cells = []
            if "closest_cellnames" in locals():
                show_cells += closest_cellnames[:n_show]
            if "farthest_cellnames" in locals():
                show_cells += farthest_cellnames[-n_show:] if farthest_cellnames else []

            if "ref_sigs" in locals() and "exp_sigs" in locals():
                ref_vis = ref_sigs
                exp_vis = exp_sigs
            else:
                ref_vis = {}
                exp_vis = {}

            if len(ref_vis) > 0:
                D = next(iter(ref_vis.values())).shape[1]
                fig_sig, ax_sig = plt.subplots(figsize=(7, 6))

                if D == 2:
                    for rn, sig_r in ref_vis.items():
                        ax_sig.plot(sig_r[:, 0], sig_r[:, 1], marker="o", linewidth=1.5, label=f"ref {rn}")
                    for cn in show_cells:
                        if cn not in exp_vis:
                            continue
                        row = dist_df[dist_df["cell_name"] == cn].iloc[0]
                        sig_e = exp_vis[cn]
                        ax_sig.plot(sig_e[:, 0], sig_e[:, 1], marker="o", linewidth=1.5,
                                    label=f"{cn} (d={row['dist_w']:.3g})")

                    ax_sig.set_xlabel(FEATURE_COLS[0] + " (scaled)")
                    ax_sig.set_ylabel(FEATURE_COLS[1] + " (scaled)")
                    ax_sig.set_title("Signatures in scaled feature space (D=2)")
                else:
                    pts_list = []
                    seq_slices = []
                    cursor = 0

                    for rn, sig_r in ref_vis.items():
                        pts_list.append(sig_r)
                        seq_slices.append((cursor, cursor + sig_r.shape[0], f"ref {rn}"))
                        cursor += sig_r.shape[0]

                    for cn in show_cells:
                        if cn not in exp_vis:
                            continue
                        sig_e = exp_vis[cn]
                        row = dist_df[dist_df["cell_name"] == cn].iloc[0]
                        label = f"{cn} (d={row['dist_w']:.3g})"
                        pts_list.append(sig_e)
                        seq_slices.append((cursor, cursor + sig_e.shape[0], label))
                        cursor += sig_e.shape[0]

                    P = np.concatenate(pts_list, axis=0)
                    P2 = pca_project_to_2d(P)

                    for a, b, label in seq_slices:
                        seg = P2[a:b]
                        ax_sig.plot(seg[:, 0], seg[:, 1], marker="o", linewidth=1.5, label=label)

                    ax_sig.set_xlabel("PC1 (scaled space)")
                    ax_sig.set_ylabel("PC2 (scaled space)")
                    ax_sig.set_title(f"Signatures PCA-projected to 2D (original D={D})")

                ax_sig.grid(True, alpha=0.3)
                ax_sig.legend(loc="best", fontsize=8)
                fig_sig.tight_layout()
        except Exception as e:
            print(f"[WARN] Signature visualization skipped due to error: {e}")

    # -----------------------------
    # Overlay SOH trajectories: closest = orange, farthest = green, slowest = black
    # -----------------------------
    closest_cellnames = locals().get("closest_cellnames", [])
    farthest_cellnames = locals().get("farthest_cellnames", [])
    slowest_black_cellnames = locals().get("slowest_black_cellnames", [])

    added_orange_label = False
    added_orange_label_w = False
    added_orange_label_t = False
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

    added_green_label = False
    added_green_label_w = False
    added_green_label_t = False
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
    # Finalize plots
    # -----------------------------
    ax.set_xlabel("index")
    ax.set_ylabel("SOH")
    ax.set_title(
        f"SOH trajectories (index-x)\n"
        f"plot interpolation target SOH={target_soh_plot} | Wasserstein signature SOH 1.0->{target_soh_features} | "
        #f"features={FEATURE_COLS} | scaling={SCALE_METHOD}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(bottom=0.8)
    ax.set_ylim(top=1.1)
    fig.tight_layout()

    ax_w.set_xlabel("weeks")
    ax_w.set_ylabel("SOH")
    ax_w.set_title(
        f"SOH trajectories vs weeks\n"
        f"plot interpolation target SOH={target_soh_plot} | Wasserstein signature SOH 1.0->{target_soh_features} | "
        #f"features={FEATURE_COLS} | scaling={SCALE_METHOD}"
    )
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="best")
    ax_w.set_ylim(bottom=0.8)
    ax_w.set_ylim(top=1.1)
    fig_w.tight_layout()

    ax_t.set_xlabel("throughput_cum")
    ax_t.set_ylabel("SOH")
    ax_t.set_title(
        f"SOH trajectories vs throughput_cum\n"
        f"plot interpolation target SOH={target_soh_plot} | Wasserstein signature SOH 1.0->{target_soh_features} | "
        #f"features={FEATURE_COLS} | scaling={SCALE_METHOD}"
    )
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="best")
    ax_t.set_ylim(bottom=0.8)
    ax_t.set_ylim(top=1.1)
    fig_t.tight_layout()

    plt.show()

    return df_exp, df_ref, locals().get("dist_df", pd.DataFrame()), best_mc


if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    out_dir = Path(r"C:\Users\Public\Documents\RL_project\out_lw\feature_plots")  # not used for saving
    main(dir_path, out_dir)