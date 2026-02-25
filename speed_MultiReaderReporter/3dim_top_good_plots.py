# speed_MultiReaderReporter/main.py
from __future__ import annotations

from pathlib import Path
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple
from scipy.interpolate import CubicSpline

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

def style_paper_matplotlib():
    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })

style_paper_matplotlib()
COL_ALL      = "#8EC1E8"  # soft blue
COL_CLOSEST  = "#F2A65A"  # soft orange
COL_FARTHEST = "#66C2A5"  # soft green/teal
COL_REFS     = "#E76F51"  # soft red
COL_BLACK    = "#222222"


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
    n_samples: int = 200,
    top_k: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo search over RAW conditions (with your discrete constraints),
    then engineer SAME features as training, predict, and return smallest top_k.

    Discrete:
      - temp in {15,25,40}
      - soc_start in {0..60 step 10}
      - soc_end in {20..100 step 10}
    Constraints:
      - soc_end > soc_start
      - soc_end - soc_start > 10  (=> at least 20 with 10-step grids)
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

    allowed_temp = np.array([15.0, 25.0, 40.0])
    allowed_soc_start = np.arange(0, 61, 10, dtype=float)      # 0..60
    allowed_soc_end = np.arange(20, 101, 10, dtype=float)      # 20..100
    allowed_cur_cha = np.arange(0.5, 1.6, 0.1, dtype=float)
    allowed_cur_dis = np.arange(0.5, 3.1, 0.1, dtype=float)

    valid_pairs = np.array(
        [(s0, s1) for s0 in allowed_soc_start for s1 in allowed_soc_end if (s1 - s0) > 10],
        dtype=float,
    )
    if len(valid_pairs) == 0:
        raise ValueError("No valid (soc_start, soc_end) pairs under the constraints.")

    idx = rng.integers(0, len(valid_pairs), size=n_samples)
    soc_start = valid_pairs[idx, 0]
    soc_end = valid_pairs[idx, 1]

    # c_rate_chg = rng.uniform(*bounds["c_rate_chg"], n_samples)
    # c_rate_dchg = rng.uniform(*bounds["c_rate_dchg"], n_samples)
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
    out["pred_dist_feat"] = pred  # predicted distance-to-nearest-ref in feature space

    best = out.nsmallest(top_k, "pred_dist_feat").reset_index(drop=True)
    return best


# -----------------------------
# Existing helper functions from your 3D script
# -----------------------------
def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = float(max(sigma, 1e-12))
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def load_and_interpolate(df: pd.DataFrame, target_soh: float, interpolation_typ: str):
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

    if df[ref_name].isna().any() or df["SOH"].isna().any():
        return None

    reference = df[ref_name].to_numpy(dtype=float)
    target_data = df["SOH"].to_numpy(dtype=float)

    ref_s = pd.Series(reference)
    keep = ~ref_s.duplicated(keep="first")
    df = df.loc[keep.values].reset_index(drop=True)

    reference = df[ref_name].to_numpy(dtype=float)
    target_data = df["SOH"].to_numpy(dtype=float)

    if len(reference) < 3:
        return None

    feature_cols = [c for c in df.columns if c != ref_name]
    feature = df[feature_cols].to_numpy(dtype=float)

    interpolated_ref = np.interp(target_soh, target_data[::-1], reference[::-1])

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


# -----------------------------
# Main (your 3D feature-space + NEW Monte Carlo surrogate)
# -----------------------------
def main(dir_path: Path, out_dir: Path):
    exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    target_soh_features = 0.99   # feature row selection
    target_soh_plot = 0.98        # interpolation grid for plotting

    rows = []
    rows_ref = []

    # Reference cell names
    REF_NAMES = ["SPEED_LW_reference_1", "SPEED_LW_reference_2", "SPEED_LW_reference_3"]

    # CONFIG YOU WANT:
    K_CLOSEST_FEATURE = 25
    K_SLOWEST_BLACK = 1
    SOH_SLOW_TARGET = 0.96

    # Gaussian distribution target (index where SOH reaches this)
    SOH_DIST_TARGET = 0.97

    needed_cols = (
        ["CU_time"]
        + exp_conds
        + ["cap_ocv_dis"]
        + [
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c","mean_d_dqdv_l_c_l",
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

        if any(c not in df.columns for c in exp_conds):
            print(f"[WARN] {cell_name}: missing exp_conds, skipping.")
            continue
        exp_row = exp_row_from_first_line(df, exp_conds)

        interp_cols_weeks = [
            "weeks", "cap_ocv_dis",
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c","mean_d_dqdv_l_c_l",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
            "mean_d_dqdv_h_d", "mean_d_dqdv_l_d",
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

        if "SOH" in interpolated_plot_weeks.columns:
            min_soh = float(np.nanmin(interpolated_plot_weeks["SOH"].to_numpy(dtype=float)))
            if min_soh < 0.8:
                cells_below_08.append((cell_name, min_soh))

        traj_by_cell_weeks[cell_name] = interpolated_plot_weeks[["weeks", "SOH"]].copy()

        # throughput interpolation
        interp_cols_thr = [
            "throughput_cum", "cap_ocv_dis",
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c","mean_d_dqdv_l_c_l",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
            "mean_d_dqdv_h_d", "mean_d_dqdv_l_d",
        ]
        missing_thr = [c for c in interp_cols_thr if c not in df.columns]
        if missing_thr:
            interpolated_plot_thr = None
        else:
            df_filter_thr = df[interp_cols_thr].copy()
            interpolated_plot_thr = load_and_interpolate(df_filter_thr, target_soh_plot, interpolation_typ="throughput")

        if interpolated_plot_thr is not None and not interpolated_plot_thr.empty:
            if "SOH" in interpolated_plot_thr.columns and "throughput_cum" in interpolated_plot_thr.columns:
                traj_by_cell_thr[cell_name] = interpolated_plot_thr[["throughput_cum", "SOH"]].copy()

        # ---- Base plotting ----
        is_ref_case = exp_row.isna().any() and ("LW_reference" in cell_name)

        if is_ref_case:
            label = "LW_reference (exp NaN)" if not added_red_label else None
            added_red_label = True
            ax.plot(interpolated_plot_weeks["SOH"], color=COL_REFS, alpha=0.25, linewidth=0.6, label=label, zorder=1)

            label_w = "LW_reference (exp NaN)" if not added_red_label_w else None
            added_red_label_w = True
            ax_w.plot(interpolated_plot_weeks["weeks"], interpolated_plot_weeks["SOH"],
                      color=COL_REFS, alpha=0.25, linewidth=0.6, label=label_w, zorder=1)

            if cell_name in traj_by_cell_thr:
                label_t = "LW_reference (exp NaN)" if not added_red_label_t else None
                added_red_label_t = True
                ax_t.plot(traj_by_cell_thr[cell_name]["throughput_cum"], traj_by_cell_thr[cell_name]["SOH"],
                          color=COL_REFS, alpha=0.25, linewidth=0.6, label=label_t, zorder=1)
        else:
            label = "other" if not added_blue_label else None
            added_blue_label = True
            ax.plot(interpolated_plot_weeks["SOH"], color=COL_ALL, alpha=0.20, linewidth=0.55, label=label, zorder=2)

            label_w = "other" if not added_blue_label_w else None
            added_blue_label_w = True
            ax_w.plot(interpolated_plot_weeks["weeks"], interpolated_plot_weeks["SOH"],
                      color=COL_ALL, alpha=0.20, linewidth=0.55, label=label_w, zorder=2)

            if cell_name in traj_by_cell_thr:
                label_t = "other" if not added_blue_label_t else None
                added_blue_label_t = True
                ax_t.plot(traj_by_cell_thr[cell_name]["throughput_cum"], traj_by_cell_thr[cell_name]["SOH"],
                          color=COL_ALL, alpha=0.20, linewidth=0.55, label=label_t, zorder=2)

        # feature row at target_soh_features
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

    ref_sub = (
        df_ref[df_ref["cell_name"].isin(REF_NAMES)][["cell_name", x_col, y_col, z_col]]
        .dropna()
        .reset_index(drop=True)
    )
    exp_sub = df_exp[["cell_name", x_col, y_col, z_col]].dropna().reset_index(drop=True)

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
        '''
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
                added_orange_label_t = True
        '''
        ref_mean = ref_sub[[x_col, y_col, z_col]].mean().to_numpy(dtype=float)

        fig_sc = plt.figure(figsize=(8.2, 7.2))
        ax_sc = fig_sc.add_subplot(111, projection="3d")

        # soften 3D panes
        for axis in [ax_sc.xaxis, ax_sc.yaxis, ax_sc.zaxis]:
            axis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
            axis.pane.set_edgecolor((0.90, 0.90, 0.90, 1.0))

        # 1) all exp (background layer)
        ax_sc.scatter(
            exp_sub[x_col], exp_sub[y_col], exp_sub[z_col],
            s=16, alpha=0.35, color=COL_ALL,
            depthshade=False, label="exp (all)"
        )

        # 2) farthest (highlight)
        ax_sc.scatter(
            exp_sub.loc[farthest_idx, x_col],
            exp_sub.loc[farthest_idx, y_col],
            exp_sub.loc[farthest_idx, z_col],
            s=55, alpha=0.90, color=COL_FARTHEST,
            edgecolors="white", linewidths=0.6,
            depthshade=False, label=f"farthest {k2}"
        )

        # 3) closest (highlight)
        ax_sc.scatter(
            exp_sub.loc[closest_idx, x_col],
            exp_sub.loc[closest_idx, y_col],
            exp_sub.loc[closest_idx, z_col],
            s=55, alpha=0.90, color=COL_CLOSEST,
            edgecolors="white", linewidths=0.6,
            depthshade=False, label=f"closest {k1}"
        )

        # 4) slowest among closest (star)
        if slowest_black_cellnames:
            row_black = exp_sub[exp_sub["cell_name"].isin(slowest_black_cellnames)]
            if not row_black.empty:
                ax_sc.scatter(
                    row_black[x_col], row_black[y_col], row_black[z_col],
                    s=160, color=COL_BLACK, marker="*",
                    edgecolors="white", linewidths=0.8,
                    depthshade=False, label=f"slowest among closest (SOH={SOH_SLOW_TARGET})"
                )

        # 5) refs
        ax_sc.scatter(
            ref_sub[x_col], ref_sub[y_col], ref_sub[z_col],
            s=75, alpha=0.95, color=COL_REFS,
            edgecolors="white", linewidths=0.7,
            depthshade=False, label="refs"
        )

        # 6) mean(refs)
        ax_sc.scatter(
            [ref_mean[0]], [ref_mean[1]], [ref_mean[2]],
            s=140, color=COL_BLACK, marker="X",
            edgecolors="white", linewidths=0.8,
            depthshade=False, label="mean(refs)"
        )

        ax_sc.set_xlabel(x_col, labelpad=8)
        ax_sc.set_ylabel(y_col, labelpad=8)
        ax_sc.set_zlabel(z_col, labelpad=8)
        ax_sc.set_title(f"Feature space @ SOH={target_soh_features} (nearest-ref distance)", pad=14)

        # calmer view
        ax_sc.view_init(elev=18, azim=40)

        # move legend outside (paper-like)
        ax_sc.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),  # move below axes
            ncol=3,  # horizontal layout
            frameon=False,
            columnspacing=1.5,
            handletextpad=0.6
        )
        # fig_sc.tight_layout()
        fig_sc.tight_layout(rect=[0, 0.08, 1, 1])

        added_green_label = False
        added_green_label_w = False
        added_green_label_t = False

        for cn in farthest_cellnames:
            traj_w = traj_by_cell_weeks.get(cn)
            if traj_w is None:
                continue

            ax.plot(
                traj_w["SOH"],
                color=COL_FARTHEST,
                alpha=0.85,
                linewidth=1.0,
                label="farthest exp (green)" if not added_green_label else None,
                zorder=5,
            )
            added_green_label = True

            ax_w.plot(
                traj_w["weeks"],
                traj_w["SOH"],
                color=COL_FARTHEST,
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
                    color=COL_FARTHEST,
                    alpha=0.85,
                    linewidth=1.0,
                    label="farthest exp (green)" if not added_green_label_t else None,
                    zorder=5,
                )
                added_green_label_t = True

    # -----------------------------
    # NEW: Train surrogate model (exp_conds -> dist_feat) and run Monte Carlo search
    # -----------------------------
    if not dist_df.empty:
        print("\n" + "=" * 90)
        print("Training surrogate XGB model: exp_conds -> distance-to-nearest-ref (dist_feat)")
        print("=" * 90)

        model, explainer, X_feat, y, feat_names = train_xgb_no_val_and_shap(
            df_exp=df_exp,
            dist_df=dist_df,
            ref_names=REF_NAMES,
        )

        best = monte_carlo_best_conditions_for_distance(
            df_exp=df_exp,
            model=model,
            ref_names=REF_NAMES,
            n_samples=200,   # you can increase this
            top_k=10,
        )

        print("\n" + "=" * 90)
        print("Top 20 Monte Carlo conditions with smallest predicted distance-to-nearest-ref (dist_feat)")
        print("=" * 90)
        print(best.to_string(index=False))
    else:
        best = pd.DataFrame()

    # -----------------------------
    # Overlay trajectories: closest = orange, farthest = green, slowest among closest = black
    # -----------------------------
    added_orange_label = added_orange_label_w = added_orange_label_t = False
    for cn in closest_cellnames:
        traj_w = traj_by_cell_weeks.get(cn)
        if traj_w is None:
            continue

        ax.plot(traj_w["SOH"], color=COL_CLOSEST, alpha=0.85, linewidth=1.0,
                label="closest exp (orange)" if not added_orange_label else None, zorder=4)
        added_orange_label = True

        ax_w.plot(traj_w["weeks"], traj_w["SOH"], color=COL_CLOSEST, alpha=0.85, linewidth=1.0,
                  label="closest exp (orange)" if not added_orange_label_w else None, zorder=4)
        added_orange_label_w = True

        traj_t = traj_by_cell_thr.get(cn)
        if traj_t is not None:
            ax_t.plot(traj_t["throughput_cum"], traj_t["SOH"], color=COL_CLOSEST, alpha=0.85, linewidth=1.0,
                      label="closest exp (orange)" if not added_orange_label_t else None, zorder=4)
            added_orange_label_t = True

    added_green_label = added_green_label_w = added_green_label_t = False
    for cn in farthest_cellnames:
        traj_w = traj_by_cell_weeks.get(cn)
        if traj_w is None:
            continue

        ax.plot(traj_w["SOH"], color=COL_FARTHEST, alpha=0.85, linewidth=1.0,
                label="farthest exp (green)" if not added_green_label else None, zorder=5)
        ax.set_lim(0.8, 1.1)
        added_green_label = True

        ax_w.plot(traj_w["weeks"], traj_w["SOH"], color=COL_FARTHEST, alpha=0.85, linewidth=1.0,
                  label="farthest exp (green)" if not added_green_label_w else None, zorder=5)
        ax_w.set_lim(0.8, 1.1)
        added_green_label_w = True

        traj_t = traj_by_cell_thr.get(cn)
        if traj_t is not None:
            ax_t.plot(traj_t["throughput_cum"], traj_t["SOH"], color=COL_FARTHEST, alpha=0.85, linewidth=1.0,
                      label="farthest exp (green)" if not added_green_label_t else None, zorder=5)
            ax_t.set_lim(0.8, 1.1)
            added_green_label_t = True

    added_black_label = False
    for cn in slowest_black_cellnames:
        traj_w_black = traj_by_cell_weeks.get(cn)
        if traj_w_black is None:
            continue

        ax.plot(traj_w_black["SOH"], color=COL_BLACK, alpha=1.0, linewidth=2.4,
                label="slowest among closest (black)" if not added_black_label else None, zorder=10)
        ax.set_lim(0.8, 1.1)
        ax_w.plot(traj_w_black["weeks"], traj_w_black["SOH"], color=COL_BLACK, alpha=1.0, linewidth=2.4,
                  label="slowest among closest (black)" if not added_black_label else None, zorder=10)
        ax_w.set_lim(0.8,1.1)
        traj_t_black = traj_by_cell_thr.get(cn)
        if traj_t_black is not None:
            ax_t.plot(traj_t_black["throughput_cum"], traj_t_black["SOH"], color=COL_BLACK, alpha=1.0, linewidth=2.4,
                      label="slowest among closest (black)" if not added_black_label else None, zorder=10)
        ax_t.set_lim(0.8, 1.1)
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
        f"SOH trajectories (index-x) (plot interpolation target SOH={target_soh_plot}; "
        f"feature comparison at SOH={target_soh_features})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(0.8, 1.1)
    fig.tight_layout()

    ax_w.set_xlabel("weeks")
    ax_w.set_ylabel("SOH")
    ax_w.set_title(
        f"SOH trajectories vs weeks (plot interpolation target SOH={target_soh_plot}; "
        f"feature comparison at SOH={target_soh_features})"
    )
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="best")
    ax_w.set_ylim(0.8, 1.1)
    fig_w.tight_layout()

    ax_t.set_xlabel("throughput_cum")
    ax_t.set_ylabel("SOH")
    ax_t.set_title(
        f"SOH trajectories vs throughput_cum (plot interpolation target SOH={target_soh_plot}; "
        f"feature comparison at SOH={target_soh_features})"
    )
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="best")
    ax_t.set_ylim(0.8, 1.1)
    ax.set_ylim(0.8, 1.1)
    ax_w.set_ylim(0.8, 1.1)
    ax_t.set_ylim(0.8, 1.1)
    fig_t.tight_layout()
    plt.show()

    if cells_below_08:
        cells_below_08_sorted = sorted(cells_below_08, key=lambda x: x[1])
        print("\nCells with SOH < 0.8 (min SOH shown):")
        for cn, m in cells_below_08_sorted:
            print(f" - {cn}: min SOH = {m:.3f}")
    else:
        print("\nNo cells reached SOH < 0.8.")

    return df_exp, df_ref, dist_df, best


if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    out_dir = Path(r"C:\Users\Public\Documents\RL_project\out_lw\feature_plots")  # not used for saving
    main(dir_path, out_dir)
