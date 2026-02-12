# speed_MultiReaderReporter/main.py
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment

# NEW: model + shap
import xgboost as xgb
import shap


# --- relative paths ---
here = Path(__file__).resolve().parent
sys.path.append(str(here))
sys.path.append(str(here / "core"))
sys.path.append(str(here / "loaders"))
sys.path.append(str(here / "utils"))

# -----------------------------
# Feature pipeline (single source of truth)
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
    # start from raw condition columns
    X = df[raw_conds].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    # engineered features
    X["soc"] = 0.5 * (X["soc_start"] + X["soc_end"])
    X["dod"] = X["soc_end"] - X["soc_start"]

    # fix weird encoding
    X.loc[X["c_rate_chg"] == 15, "c_rate_chg"] = 1.5

    if drop_raw_soc:
        X = X.drop(columns=["soc_start", "soc_end"])

    feature_names = FEAT_CONDS_DEFAULT if drop_raw_soc else (raw_conds + ["soc", "dod"])
    X = X[feature_names].copy()

    return X, feature_names


# -----------------------------
# Train on ALL data + SHAP
# -----------------------------
def train_xgb_no_val_and_shap(
    df_exp: pd.DataFrame,
    dist_df: pd.DataFrame,
    ref_names: list[str],
    raw_conds: list[str] = RAW_CONDS_DEFAULT,
    random_state: int = 42,
    shap_max_display: int = 20,
):
    """
    Fit XGBRegressor on ALL available data (no train/val split), target = dist_w.
    Uses engineered features: ["soc","dod","c_rate_chg","c_rate_dchg","temp"].

    Returns: model, explainer, X_feat, y
    """
    # join distances to experiment table
    df = df_exp.merge(dist_df[["cell_name", "dist_w"]], on="cell_name", how="inner")

    # remove reference cells
    df = df[~df["cell_name"].isin(ref_names)].copy()

    # y
    y = pd.to_numeric(df["dist_w"], errors="coerce")

    # X (engineered feature set)
    X_feat, feat_names = make_features_from_raw(df, raw_conds=raw_conds, drop_raw_soc=True)

    # keep valid rows (y + all features finite)
    ok = ~y.isna()
    ok &= np.isfinite(X_feat.to_numpy()).all(axis=1)

    X_feat = X_feat.loc[ok].reset_index(drop=True)
    y = y.loc[ok].to_numpy(dtype=float)

    if len(X_feat) < 5:
        raise ValueError("Not enough rows with valid features + dist_w to train XGB model.")

    # simple, strong default params; no validation / early stopping
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

    # SHAP (TreeExplainer is best for XGB)
    # explainer = shap.TreeExplainer(model)
    # explainer = shap.KernelExplainer(model, X_feat)
    explainer = shap.KernelExplainer(lambda a: model.predict(np.asarray(a)), X_feat)

    shap_values = explainer.shap_values(X_feat)

    # plots
    plt.figure()
    shap.summary_plot(shap_values, X_feat, show=False, max_display=shap_max_display)
    plt.title("SHAP summary (beeswarm) — target: Wasserstein distance (dist_w)")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X_feat, plot_type="bar", show=False, max_display=shap_max_display)
    plt.title("SHAP importance (bar) — target: Wasserstein distance (dist_w)")
    plt.tight_layout()
    plt.show()

    return model, explainer, X_feat, y, feat_names


# -----------------------------
# Monte Carlo: sample RAW, then engineer SAME features, then predict
# -----------------------------
def monte_carlo_best_conditions_for_distance(
    df_exp: pd.DataFrame,
    model: xgb.XGBRegressor,
    ref_names: list[str],
    raw_conds: list[str] = RAW_CONDS_DEFAULT,
    n_samples: int = 200,
    top_k: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample RAW conditions uniformly within observed non-reference min/max,
    enforce soc_end >= soc_start, engineer SAME features as training,
    predict distance, return smallest top_k.

    Output includes both raw columns + engineered features + pred_dist_w.
    """
    rng = np.random.default_rng(random_state)

    df_train = df_exp[~df_exp["cell_name"].isin(ref_names)].copy()

    # bounds from data (raw columns)
    bounds = {}
    for c in raw_conds:
        col = pd.to_numeric(df_train[c], errors="coerce")
        lo = float(np.nanmin(col))
        hi = float(np.nanmax(col))
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Non-finite bounds for {c}: lo={lo}, hi={hi}")
        bounds[c] = (lo, hi)

    # sample raw
    soc_start = rng.uniform(*bounds["soc_start"], n_samples)
    soc_end = rng.uniform(*bounds["soc_end"], n_samples)
    c_rate_chg = rng.uniform(*bounds["c_rate_chg"], n_samples)
    c_rate_dchg = rng.uniform(*bounds["c_rate_dchg"], n_samples)
    temp = rng.uniform(*bounds["temp"], n_samples)

    # constraints: soc_end >= soc_start
    swap = soc_end < soc_start
    soc_start2 = soc_start.copy()
    soc_end2 = soc_end.copy()
    soc_start2[swap], soc_end2[swap] = soc_end2[swap], soc_start2[swap]
    soc_start, soc_end = soc_start2, soc_end2

    X_mc_raw = pd.DataFrame(
        {
            "soc_start": soc_start,
            "soc_end": soc_end,
            "c_rate_chg": c_rate_chg,
            "c_rate_dchg": c_rate_dchg,
            "temp": temp,
        }
    )[raw_conds].astype(float)

    # engineer features exactly like training
    X_mc_feat, feat_names = make_features_from_raw(X_mc_raw, raw_conds=raw_conds, drop_raw_soc=True)

    # predict
    pred = model.predict(X_mc_feat)
    out = X_mc_raw.copy()
    out["soc"] = X_mc_feat["soc"].to_numpy()
    out["dod"] = X_mc_feat["dod"].to_numpy()
    out["pred_dist_w"] = pred

    best = out.nsmallest(top_k, "pred_dist_w").reset_index(drop=True)
    return best


# -----------------------------
# Interpolation (returns SOH + reference axis)
# -----------------------------
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

    # need strictly increasing x for CubicSpline (remove duplicates)
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

    # Optional safety: clip SOH to [0, 1.05] (CubicSpline can overshoot)
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


# -----------------------------
# ND signature + Wasserstein (assignment)
# -----------------------------
def feature_signature_nd(
    traj: pd.DataFrame,
    feature_cols: list[str],
    soh_hi: float,
    soh_lo: float,
    n_points: int,
    time_col: str = "weeks",
) -> np.ndarray | None:
    """
    Build an ND signature with shape (n_points, D), D=len(feature_cols),
    sampled uniformly in TIME between t0 and t(SOH=soh_lo), and evaluated via CubicSpline vs time.
    """
    if traj is None or traj.empty:
        return None
    needed = {time_col, "SOH"} | set(feature_cols)
    if not needed.issubset(set(traj.columns)):
        return None

    t = traj[time_col].to_numpy(dtype=float)
    s = traj["SOH"].to_numpy(dtype=float)
    feats = [traj[c].to_numpy(dtype=float) for c in feature_cols]

    ok = ~np.isnan(t) & ~np.isnan(s)
    for f in feats:
        ok &= ~np.isnan(f)

    t = t[ok]
    s = s[ok]
    feats = [f[ok] for f in feats]

    if len(t) < 3:
        return None
    if np.nanmin(s) > soh_lo:
        return None

    # time at SOH=soh_lo (interp in SOH->time space)
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

    # CubicSpline requires strictly increasing time
    keep_t = ~pd.Series(t).duplicated(keep="first")
    t_u = t[keep_t.values]
    if len(t_u) < 3:
        return None

    sig_cols = []
    for f in feats:
        f_u = f[keep_t.values]
        try:
            cs = CubicSpline(t_u, f_u, bc_type="natural", extrapolate=False)
            out = cs(ts)
        except Exception:
            out = np.interp(ts, t_u, f_u)
        sig_cols.append(out)

    sig = np.stack(sig_cols, axis=1)  # (n_points, D)
    if np.isnan(sig).any():
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


# -----------------------------
# Option C scaling (IQR/std/range) on signatures
# -----------------------------
def compute_feature_scales_from_signatures(
    all_sigs: list[np.ndarray],
    method: str = "robust_iqr",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute per-feature scales so each feature contributes similarly to Euclidean cost.

    method:
      - "robust_iqr": scale = IQR (q75-q25) per feature (robust, recommended)
      - "std":        scale = standard deviation per feature
      - "range":      scale = max-min per feature
    """
    if len(all_sigs) == 0:
        raise ValueError("No signatures provided for scale computation.")
    X = np.concatenate(all_sigs, axis=0)  # (N_total_points, D)

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
    """Divide each feature dimension by its scale (no centering)."""
    return sig / scale


# -----------------------------
# Main
# -----------------------------
def main(dir_path: Path, out_dir: Path):
    exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    # semantics you requested:
    target_soh_features = 0.995   # used for WASSERSTEIN window (SOH_hi -> this)
    target_soh_plot = 0.98        # used for plotting interpolation/grid

    # --- Choose 2D or 3D by editing this list ---
    # 2D:
    # FEATURE_COLS = ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c"]
    # 3D:
    FEATURE_COLS = ["mean_d_dqdv_m_c", "mean_d_dqdv_h_c", "mean_d_dqdv_m_d"]

    # Option C scaling (IQR recommended)
    SCALE_METHOD = "robust_iqr"

    # signature config
    SOH_SIG_HI = 1.0
    SOH_SIG_LO = target_soh_features
    SIG_NPTS = 5

    # references
    REF_NAMES = ["SPEED_LW_reference_4", "SPEED_LW_reference_5", "SPEED_LW_reference_6"]

    # selection config
    K_CLOSEST_FEATURE = 20
    K_FARTHEST = 15
    K_SLOWEST_BLACK = 1
    SOH_SLOW_TARGET = 0.96

    needed_cols = (
        ["CU_time"]
        + exp_conds
        + ["cap_ocv_dis"]
        + [
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
        ]
        + ["throughput_cum", "mean_d_dqdv_h_d", "mean_d_dqdv_l_d"]
    )

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(9, 5))
    fig_w, ax_w = plt.subplots(figsize=(9, 5))
    fig_t, ax_t = plt.subplots(figsize=(9, 5))

    added_blue_label = added_red_label = False
    added_blue_label_w = added_red_label_w = False
    added_blue_label_t = added_red_label_t = False

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
            "weeks", "cap_ocv_dis",
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
            "mean_d_dqdv_h_d", "mean_d_dqdv_l_d",
        ]
        missing = [c for c in interp_cols_weeks if c not in df.columns]
        if missing:
            print(f"[WARN] {cell_name}: missing {missing}, skipping.")
            continue

        df_filter_weeks = df[interp_cols_weeks].copy()

        # trajectories / plotting use target_soh_plot
        interpolated_plot_weeks = load_and_interpolate(df_filter_weeks, target_soh_plot, interpolation_typ="weeks")
        if interpolated_plot_weeks is None or interpolated_plot_weeks.empty:
            print(f"[WARN] {cell_name}: interpolation (weeks) failed, skipping.")
            continue

        traj_by_cell_weeks[cell_name] = interpolated_plot_weeks.copy()

        # throughput interpolation (optional)
        interp_cols_thr = [
            "throughput_cum", "cap_ocv_dis",
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
            "mean_d_dqdv_h_d", "mean_d_dqdv_l_d",
        ]
        missing_thr = [c for c in interp_cols_thr if c not in df.columns]
        if not missing_thr:
            df_filter_thr = df[interp_cols_thr].copy()
            interpolated_plot_thr = load_and_interpolate(df_filter_thr, target_soh_plot, interpolation_typ="throughput")
            if interpolated_plot_thr is not None and not interpolated_plot_thr.empty:
                if "SOH" in interpolated_plot_thr.columns and "throughput_cum" in interpolated_plot_thr.columns:
                    traj_by_cell_thr[cell_name] = interpolated_plot_thr[["throughput_cum", "SOH"]].copy()

        # base plot all curves
        is_ref_case = exp_row.isna().any() and ("LW_reference" in cell_name)

        if is_ref_case:
            lab = "LW_reference (exp NaN)" if not added_red_label else None
            added_red_label = True
            ax.plot(interpolated_plot_weeks["SOH"], color="red", alpha=0.25, linewidth=0.6, label=lab, zorder=1)

            labw = "LW_reference (exp NaN)" if not added_red_label_w else None
            added_red_label_w = True
            ax_w.plot(interpolated_plot_weeks["weeks"], interpolated_plot_weeks["SOH"], color="red", alpha=0.25, linewidth=0.6, label=labw, zorder=1)

            if cell_name in traj_by_cell_thr:
                labt = "LW_reference (exp NaN)" if not added_red_label_t else None
                added_red_label_t = True
                ax_t.plot(traj_by_cell_thr[cell_name]["throughput_cum"], traj_by_cell_thr[cell_name]["SOH"], color="red", alpha=0.25, linewidth=0.6, label=labt, zorder=1)
        else:
            lab = "other" if not added_blue_label else None
            added_blue_label = True
            ax.plot(interpolated_plot_weeks["SOH"], color="blue", alpha=0.20, linewidth=0.55, label=lab, zorder=2)

            labw = "other" if not added_blue_label_w else None
            added_blue_label_w = True
            ax_w.plot(interpolated_plot_weeks["weeks"], interpolated_plot_weeks["SOH"], color="blue", alpha=0.20, linewidth=0.55, label=labw, zorder=2)

            if cell_name in traj_by_cell_thr:
                labt = "other" if not added_blue_label_t else None
                added_blue_label_t = True
                ax_t.plot(traj_by_cell_thr[cell_name]["throughput_cum"], traj_by_cell_thr[cell_name]["SOH"], color="blue", alpha=0.20, linewidth=0.55, label=labt, zorder=2)

        # metadata row at target_soh_features
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

    # plot references on top
    added_ref_label = False
    for ref_name in REF_NAMES:
        traj_w = traj_by_cell_weeks.get(ref_name)
        if traj_w is None:
            continue
        ax.plot(traj_w["SOH"], color="red", alpha=0.95, linewidth=1.6, label="SPEED_LW_reference_4..6" if not added_ref_label else None, zorder=3)
        ax_w.plot(traj_w["weeks"], traj_w["SOH"], color="red", alpha=0.95, linewidth=1.6, label="SPEED_LW_reference_4..6" if not added_ref_label else None, zorder=3)
        traj_t = traj_by_cell_thr.get(ref_name)
        if traj_t is not None:
            ax_t.plot(traj_t["throughput_cum"], traj_t["SOH"], color="red", alpha=0.95, linewidth=1.6, label="SPEED_LW_reference_4..6" if not added_ref_label else None, zorder=3)
        added_ref_label = True

    # -----------------------------
    # Wasserstein distance computation (3D if FEATURE_COLS has length 3)
    # -----------------------------
    ref_sigs_raw: dict[str, np.ndarray] = {}
    for rn in REF_NAMES:
        sig = feature_signature_nd(traj_by_cell_weeks.get(rn), FEATURE_COLS, SOH_SIG_HI, SOH_SIG_LO, SIG_NPTS, time_col="weeks")
        if sig is not None:
            ref_sigs_raw[rn] = sig

    if len(ref_sigs_raw) == 0:
        raise RuntimeError(f"No valid reference signatures for features={FEATURE_COLS}")

    exp_names = df_exp["cell_name"].dropna().unique().tolist()
    exp_sigs_raw: dict[str, np.ndarray] = {}
    for cn in exp_names:
        sig = feature_signature_nd(traj_by_cell_weeks.get(cn), FEATURE_COLS, SOH_SIG_HI, SOH_SIG_LO, SIG_NPTS, time_col="weeks")
        if sig is not None:
            exp_sigs_raw[cn] = sig

    # Option C scales from ALL cells (refs + exps)
    all_sig_list = list(ref_sigs_raw.values()) + list(exp_sigs_raw.values())
    scale = compute_feature_scales_from_signatures(all_sig_list, method=SCALE_METHOD)

    print("\nOption C scales used:")
    for name, sc in zip(FEATURE_COLS, scale):
        print(f"  {name}: {sc:g}")

    ref_sigs = {k: apply_feature_scaling(v, scale) for k, v in ref_sigs_raw.items()}
    exp_sigs = {k: apply_feature_scaling(v, scale) for k, v in exp_sigs_raw.items()}

    # compute distances (min over refs)
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

    dist_df = pd.DataFrame(dist_list, columns=["cell_name", "dist_w", "best_ref"]).sort_values("dist_w").reset_index(drop=True)

    # closest/farthest cellnames (for overlay)
    closest_cellnames = dist_df["cell_name"].head(min(K_CLOSEST_FEATURE, len(dist_df))).tolist()
    farthest_cellnames = dist_df["cell_name"].tail(min(K_FARTHEST, len(dist_df))).tolist()

    # slowest among closest
    scores: list[tuple[str, float]] = []
    for cn in closest_cellnames:
        traj_w = traj_by_cell_weeks.get(cn)
        if traj_w is None:
            continue
        w_at = weeks_to_reach_soh(traj_w[["weeks", "SOH"]], SOH_SLOW_TARGET)
        if w_at is None:
            continue
        scores.append((cn, float(w_at)))
    scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)
    slowest_black_cellnames = [cn for cn, _ in scores_sorted[: min(K_SLOWEST_BLACK, len(scores_sorted))]]

    # print closest with experimental conditions
    cond_map = (
        df_exp[["cell_name"] + exp_conds]
        .dropna(subset=["cell_name"])
        .set_index("cell_name")
        .to_dict(orient="index")
    )
    print("\n" + "-" * 100)
    print(
        f"Closest {len(closest_cellnames)} exp cells by Wasserstein distance\n"
        f"features={FEATURE_COLS} | signature SOH {SOH_SIG_HI}->{target_soh_features} (n={SIG_NPTS}) | scaling={SCALE_METHOD}"
    )
    print("-" * 100)
    for cn in closest_cellnames:
        row = dist_df.loc[dist_df["cell_name"] == cn].iloc[0]
        conds = cond_map.get(cn, {})
        cond_str = ", ".join(f"{k}={conds.get(k, np.nan)}" for k in exp_conds)
        print(f"{cn:>25} | d={row['dist_w']:.6g} | best_ref={row['best_ref']} | {cond_str}")

    # -----------------------------
    # Train XGB model to learn mapping exp_conds -> dist_w (NO validation split)
    # + SHAP
    # -----------------------------

    model, explainer, X_feat, y, feat_names = train_xgb_no_val_and_shap(
        df_exp=df_exp,
        dist_df=dist_df,
        ref_names=REF_NAMES,
    )

    best = monte_carlo_best_conditions_for_distance(
        df_exp=df_exp,
        model=model,
        ref_names=REF_NAMES,
        n_samples=200,
        top_k=20,
    )
    print(best)

    print("\n" + "=" * 90)
    print("Top 20 Monte Carlo conditions (out of 200) with smallest predicted Wasserstein distance")
    print("=" * 90)
    print(best.to_string(index=False))

    # -----------------------------
    # Overlay trajectories: closest orange / farthest green / slowest black
    # -----------------------------
    added_orange = added_green = added_black = False
    for cn in closest_cellnames:
        traj = traj_by_cell_weeks.get(cn)
        if traj is None:
            continue
        ax.plot(traj["SOH"], color="orange", alpha=0.85, linewidth=1.0, label="closest exp" if not added_orange else None, zorder=4)
        ax_w.plot(traj["weeks"], traj["SOH"], color="orange", alpha=0.85, linewidth=1.0, label="closest exp" if not added_orange else None, zorder=4)
        if cn in traj_by_cell_thr:
            ax_t.plot(traj_by_cell_thr[cn]["throughput_cum"], traj_by_cell_thr[cn]["SOH"], color="orange", alpha=0.85, linewidth=1.0, label="closest exp" if not added_orange else None, zorder=4)
        added_orange = True

    for cn in farthest_cellnames:
        traj = traj_by_cell_weeks.get(cn)
        if traj is None:
            continue
        ax.plot(traj["SOH"], color="green", alpha=0.85, linewidth=1.0, label="farthest exp" if not added_green else None, zorder=5)
        ax_w.plot(traj["weeks"], traj["SOH"], color="green", alpha=0.85, linewidth=1.0, label="farthest exp" if not added_green else None, zorder=5)
        if cn in traj_by_cell_thr:
            ax_t.plot(traj_by_cell_thr[cn]["throughput_cum"], traj_by_cell_thr[cn]["SOH"], color="green", alpha=0.85, linewidth=1.0, label="farthest exp" if not added_green else None, zorder=5)
        added_green = True

    for cn in slowest_black_cellnames:
        traj = traj_by_cell_weeks.get(cn)
        if traj is None:
            continue
        ax.plot(traj["SOH"], color="black", alpha=1.0, linewidth=2.4, label="slowest among closest" if not added_black else None, zorder=10)
        ax_w.plot(traj["weeks"], traj["SOH"], color="black", alpha=1.0, linewidth=2.4, label="slowest among closest" if not added_black else None, zorder=10)
        if cn in traj_by_cell_thr:
            ax_t.plot(traj_by_cell_thr[cn]["throughput_cum"], traj_by_cell_thr[cn]["SOH"], color="black", alpha=1.0, linewidth=2.4, label="slowest among closest" if not added_black else None, zorder=10)
        added_black = True

    # finalize plots
    ax.set_xlabel("index")
    ax.set_ylabel("SOH")
    ax.set_title(
        f"SOH trajectories (index-x)\n"
        f"plot interpolation target SOH={target_soh_plot} | Wasserstein signature SOH 1.0->{target_soh_features} | "
        f"features={FEATURE_COLS} | scaling={SCALE_METHOD}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(0.8, 1.0)
    fig.tight_layout()

    ax_w.set_xlabel("weeks")
    ax_w.set_ylabel("SOH")
    ax_w.set_title("SOH trajectories vs weeks")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="best")
    fig_w.tight_layout()

    ax_t.set_xlabel("throughput_cum")
    ax_t.set_ylabel("SOH")
    ax_t.set_title("SOH trajectories vs throughput_cum")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="best")
    fig_t.tight_layout()

    plt.show()

    return df_exp, df_ref, dist_df, best

def strip_brackets_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns to strings, strip whitespace, remove single surrounding brackets:
    '[3.2E0]' -> '3.2E0'
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        s = df[c].astype(str).str.strip()
        s = s.str.replace(r"^\[|\]$", "", regex=True)
        df[c] = s
    return df


def coerce_numeric_df_brackets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip brackets then coerce everything to numeric.
    """
    df = strip_brackets_in_df(df)
    return df.apply(pd.to_numeric, errors="coerce")


def coerce_series_to_float_1d(s: pd.Series) -> np.ndarray:
    """
    Series -> float array, handling '[...]' strings.
    """
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(r"^\[|\]$", "", regex=True)
    s2 = pd.to_numeric(s2, errors="coerce")
    return s2.to_numpy(dtype=float).reshape(-1)


if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    out_dir = Path(r"C:\Users\Public\Documents\RL_project\out_lw\feature_plots")
    main(dir_path, out_dir)
