# speed_MultiReaderReporter/main.py
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment


# --- relative paths ---
here = Path(__file__).resolve().parent
sys.path.append(str(here))
sys.path.append(str(here / "core"))
sys.path.append(str(here / "loaders"))
sys.path.append(str(here / "utils"))


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

    # need strictly increasing x for CubicSpline
    ref_s = pd.Series(reference)
    keep = ~ref_s.duplicated(keep="first")
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

    # Optional safety: clip SOH to [0, 1.05]
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
# 2D signature + Wasserstein (assignment)
# -----------------------------
def feature_signature_2d(
    traj: pd.DataFrame,
    x_col: str,
    y_col: str,
    soh_hi: float,
    soh_lo: float,
    n_points: int,
    time_col: str = "weeks",
) -> np.ndarray | None:
    """
    Build a 2D signature point cloud with shape (n_points, 2), sampled uniformly in TIME
    between t0 and t(SOH=soh_lo), and evaluated via CubicSpline vs time.
    """
    if traj is None or traj.empty:
        return None
    needed = {time_col, "SOH", x_col, y_col}
    if not needed.issubset(set(traj.columns)):
        return None

    t = traj[time_col].to_numpy(dtype=float)
    s = traj["SOH"].to_numpy(dtype=float)
    x = traj[x_col].to_numpy(dtype=float)
    y = traj[y_col].to_numpy(dtype=float)

    ok = ~np.isnan(t) & ~np.isnan(s) & ~np.isnan(x) & ~np.isnan(y)
    t, s, x, y = t[ok], s[ok], x[ok], y[ok]
    if len(t) < 3:
        return None

    # Must reach soh_lo
    if np.nanmin(s) > soh_lo:
        return None

    # time at SOH=soh_lo
    idx = np.argsort(s)  # SOH ascending
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
    x_u = x[keep_t.values]
    y_u = y[keep_t.values]
    if len(t_u) < 3:
        return None

    def eval_spline(tu: np.ndarray, fu: np.ndarray, tq: np.ndarray) -> np.ndarray:
        try:
            cs = CubicSpline(tu, fu, bc_type="natural", extrapolate=False)
            out = cs(tq)
        except Exception:
            out = np.interp(tq, tu, fu)
        return out

    x_sig = eval_spline(t_u, x_u, ts)
    y_sig = eval_spline(t_u, y_u, ts)

    if np.isnan(x_sig).any() or np.isnan(y_sig).any():
        return None

    return np.stack([x_sig.astype(float), y_sig.astype(float)], axis=1)


def wasserstein_2d_assignment(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Mean matched Euclidean cost (Hungarian assignment) between two equal-weight point clouds."""
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
# Main
# -----------------------------
def main(dir_path: Path, out_dir: Path):
    exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    # Two knobs:
    target_soh_features = 0.995   # <-- used for WASSERSTEIN signature (SOH_hi -> this)
    target_soh_plot = 0.98        # <-- used for plotting interpolation / grids

    # 2D Wasserstein feature config
    X_COL = "mean_d_dqdv_m_c"
    Y_COL = "mean_d_dqdv_h_c"

    # signature config (SOH_hi fixed at 1.0, SOH_lo = target_soh_features)
    SOH_SIG_HI = 1.0
    SOH_SIG_LO = target_soh_features
    SIG_NPTS = 5

    rows = []
    rows_ref = []

    REF_NAMES = ["SPEED_LW_reference_4", "SPEED_LW_reference_5", "SPEED_LW_reference_6"]

    K_CLOSEST_FEATURE = 15
    K_FARTHEST = 20
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

        # IMPORTANT CHANGE:
        # - for plotting & trajectory storage, interpolate to target_soh_plot (not target_soh_features)
        interpolated_plot_weeks = load_and_interpolate(df_filter_weeks, target_soh_plot, interpolation_typ="weeks")
        if interpolated_plot_weeks is None or interpolated_plot_weeks.empty:
            print(f"[WARN] {cell_name}: interpolation (weeks) failed, skipping.")
            continue

        traj_by_cell_weeks[cell_name] = interpolated_plot_weeks.copy()

        # throughput interpolation
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
        if missing_thr:
            interpolated_plot_thr = None
        else:
            df_filter_thr = df[interp_cols_thr].copy()
            interpolated_plot_thr = load_and_interpolate(df_filter_thr, target_soh_plot, interpolation_typ="throughput")

        if interpolated_plot_thr is not None and not interpolated_plot_thr.empty:
            if "SOH" in interpolated_plot_thr.columns and "throughput_cum" in interpolated_plot_thr.columns:
                traj_by_cell_thr[cell_name] = interpolated_plot_thr[["throughput_cum", "SOH"]].copy()

        # ---- Base plotting of "all curves" ----
        is_ref_case = exp_row.isna().any() and ("LW_reference" in cell_name)

        if is_ref_case:
            label = "LW_reference (exp NaN)" if not added_red_label else None
            added_red_label = True
            ax.plot(interpolated_plot_weeks["SOH"], color="red", alpha=0.25, linewidth=0.6, label=label, zorder=1)

            label_w = "LW_reference (exp NaN)" if not added_red_label_w else None
            added_red_label_w = True
            ax_w.plot(
                interpolated_plot_weeks["weeks"],
                interpolated_plot_weeks["SOH"],
                color="red",
                alpha=0.25,
                linewidth=0.6,
                label=label_w,
                zorder=1,
            )

            if cell_name in traj_by_cell_thr:
                label_t = "LW_reference (exp NaN)" if not added_red_label_t else None
                added_red_label_t = True
                ax_t.plot(
                    traj_by_cell_thr[cell_name]["throughput_cum"],
                    traj_by_cell_thr[cell_name]["SOH"],
                    color="red",
                    alpha=0.25,
                    linewidth=0.6,
                    label=label_t,
                    zorder=1,
                )
        else:
            label = "other" if not added_blue_label else None
            added_blue_label = True
            ax.plot(interpolated_plot_weeks["SOH"], color="blue", alpha=0.20, linewidth=0.55, label=label, zorder=2)

            label_w = "other" if not added_blue_label_w else None
            added_blue_label_w = True
            ax_w.plot(
                interpolated_plot_weeks["weeks"],
                interpolated_plot_weeks["SOH"],
                color="blue",
                alpha=0.20,
                linewidth=0.55,
                label=label_w,
                zorder=2,
            )

            if cell_name in traj_by_cell_thr:
                label_t = "other" if not added_blue_label_t else None
                added_blue_label_t = True
                ax_t.plot(
                    traj_by_cell_thr[cell_name]["throughput_cum"],
                    traj_by_cell_thr[cell_name]["SOH"],
                    color="blue",
                    alpha=0.20,
                    linewidth=0.55,
                    label=label_t,
                    zorder=2,
                )

        # --- metadata row for df_exp/df_ref should use target_soh_features (as you requested) ---
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

    # -----------------------------
    # Plot references on top
    # -----------------------------
    added_ref_label = False
    for ref_name in REF_NAMES:
        traj_w = traj_by_cell_weeks.get(ref_name)
        if traj_w is None:
            continue

        ax.plot(
            traj_w["SOH"],
            color="red",
            alpha=0.95,
            linewidth=1.6,
            label="SPEED_LW_reference_4..6" if not added_ref_label else None,
            zorder=3,
        )
        ax_w.plot(
            traj_w["weeks"],
            traj_w["SOH"],
            color="red",
            alpha=0.95,
            linewidth=1.6,
            label="SPEED_LW_reference_4..6" if not added_ref_label else None,
            zorder=3,
        )

        traj_t = traj_by_cell_thr.get(ref_name)
        if traj_t is not None:
            ax_t.plot(
                traj_t["throughput_cum"],
                traj_t["SOH"],
                color="red",
                alpha=0.95,
                linewidth=1.6,
                label="SPEED_LW_reference_4..6" if not added_ref_label else None,
                zorder=3,
            )

        added_ref_label = True

    # -----------------------------
    # 2D Wasserstein distance to nearest reference
    # Uses target_soh_features as the SOH_LO of the signature
    # -----------------------------
    ref_sigs: dict[str, np.ndarray] = {}
    for rn in REF_NAMES:
        sig = feature_signature_2d(
            traj_by_cell_weeks.get(rn),
            X_COL, Y_COL,
            soh_hi=SOH_SIG_HI,
            soh_lo=SOH_SIG_LO,   # == target_soh_features
            n_points=SIG_NPTS,
            time_col="weeks",
        )
        if sig is not None:
            ref_sigs[rn] = sig

    if len(ref_sigs) == 0:
        print(f"[WARN] No valid 2D reference signatures for ({X_COL}, {Y_COL}).")
        closest_cellnames, farthest_cellnames, slowest_black_cellnames = [], [], []
        dist_df = pd.DataFrame(columns=["cell_name", "dist_w2d", "best_ref"])
    else:
        exp_names = df_exp["cell_name"].dropna().unique().tolist()
        dist_list: list[tuple[str, float, str]] = []

        for cn in exp_names:
            sig_e = feature_signature_2d(
                traj_by_cell_weeks.get(cn),
                X_COL, Y_COL,
                soh_hi=SOH_SIG_HI,
                soh_lo=SOH_SIG_LO,   # == target_soh_features
                n_points=SIG_NPTS,
                time_col="weeks",
            )
            if sig_e is None:
                continue

            best_d = np.inf
            best_r = ""
            for rn, sig_r in ref_sigs.items():
                d = wasserstein_2d_assignment(sig_e, sig_r)
                if d < best_d:
                    best_d = d
                    best_r = rn

            if np.isfinite(best_d) and best_r:
                dist_list.append((cn, float(best_d), best_r))

        dist_df = pd.DataFrame(dist_list, columns=["cell_name", "dist_w2d", "best_ref"])
        dist_df = dist_df.sort_values("dist_w2d", ascending=True).reset_index(drop=True)

        k1 = min(K_CLOSEST_FEATURE, len(dist_df))
        closest_cellnames = dist_df.loc[: k1 - 1, "cell_name"].tolist()

        k2 = min(K_FARTHEST, len(dist_df))
        farthest_cellnames = dist_df.loc[len(dist_df) - k2 :, "cell_name"].tolist()

        # print closest with distance
        print(f"\nClosest {len(closest_cellnames)} exp cells (orange) by 2D Wasserstein (SOH 1.0 -> {target_soh_features}):")
        for cn in closest_cellnames:
            row = dist_df[dist_df["cell_name"] == cn].iloc[0]
            print(f" - {cn}: d={row['dist_w2d']:.6g} (best_ref={row['best_ref']})")

        # Slowest among closest
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
            n_black = min(K_SLOWEST_BLACK, len(scores_sorted))
            slowest_black_cellnames = [cn for cn, _ in scores_sorted[:n_black]]
            print(f"\nSlowest-aging among closest (max weeks to reach SOH={SOH_SLOW_TARGET}):")
            for cn, w_at in scores_sorted[:n_black]:
                print(f" - {cn}: ~{w_at:.2f} weeks")

    # -----------------------------
    # Visualize Wasserstein distances
    # -----------------------------
    if not dist_df.empty:
        fig_d, ax_d = plt.subplots(figsize=(9, 4))
        y = dist_df["dist_w2d"].to_numpy(float)
        x = np.arange(len(y))
        ax_d.plot(x, y, linewidth=1.4)
        ax_d.set_title(
            f"2D Wasserstein distance to nearest reference ({X_COL}, {Y_COL})\n"
            f"signature SOH 1.0 -> {target_soh_features}, n={SIG_NPTS}"
        )
        ax_d.set_xlabel("cells (sorted by distance)")
        ax_d.set_ylabel("distance (avg matched Euclidean cost)")
        ax_d.grid(True, alpha=0.3)

        if closest_cellnames:
            idx_cl = dist_df.index[: min(len(dist_df), K_CLOSEST_FEATURE)]
            ax_d.scatter(idx_cl, y[idx_cl], s=35, color="orange", label="closest")
        if farthest_cellnames:
            idx_fa = dist_df.index[len(dist_df) - min(len(dist_df), K_FARTHEST) :]
            ax_d.scatter(idx_fa, y[idx_fa], s=35, color="green", label="farthest")
        ax_d.legend(loc="best")
        fig_d.tight_layout()

        fig_h, ax_h = plt.subplots(figsize=(7, 4))
        ax_h.hist(y, bins=30)
        ax_h.set_title("Histogram of 2D Wasserstein distances")
        ax_h.set_xlabel("distance")
        ax_h.set_ylabel("count")
        ax_h.grid(True, alpha=0.3)
        fig_h.tight_layout()

    # -----------------------------
    # Overlay trajectories: closest = orange, farthest = green, slowest = black
    # -----------------------------
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
        f"SOH trajectories (index-x) (plot interpolation target SOH={target_soh_plot}; "
        f"W2D uses signature SOH 1.0 -> {target_soh_features}, n={SIG_NPTS})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    ax_w.set_xlabel("weeks")
    ax_w.set_ylabel("SOH")
    ax_w.set_title(
        f"SOH trajectories vs weeks (plot interpolation target SOH={target_soh_plot}; "
        f"W2D uses signature SOH 1.0 -> {target_soh_features}, n={SIG_NPTS})"
    )
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="best")
    fig_w.tight_layout()

    ax_t.set_xlabel("throughput_cum")
    ax_t.set_ylabel("SOH")
    ax_t.set_title(
        f"SOH trajectories vs throughput_cum (plot interpolation target SOH={target_soh_plot}; "
        f"W2D uses signature SOH 1.0 -> {target_soh_features}, n={SIG_NPTS})"
    )
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="best")
    fig_t.tight_layout()

    plt.show()
    return df_exp, df_ref, dist_df


if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    out_dir = Path(r"C:\Users\Public\Documents\RL_project\out_lw\feature_plots")  # not used for saving
    main(dir_path, out_dir)
