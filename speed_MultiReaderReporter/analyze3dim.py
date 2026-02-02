# speed_MultiReaderReporter/main.py
from __future__ import annotations

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

# NEW: 3D plotting
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# --- relative paths ---
here = Path(__file__).resolve().parent
sys.path.append(str(here))
sys.path.append(str(here / "core"))
sys.path.append(str(here / "loaders"))
sys.path.append(str(here / "utils"))


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian PDF; sigma is clamped to avoid division by zero."""
    sigma = float(max(sigma, 1e-12))
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


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
    # remove duplicates in reference if any
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

        # If there are NaNs, fallback to linear (CubicSpline cannot handle NaNs)
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
    """
    Return interpolated weeks when SOH reaches soh_target.
    Returns None if curve never reaches soh_target or data invalid.
    """
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

    # If it never reaches the target (SOH stays above target)
    if np.nanmin(s) > soh_target:
        return None

    # Interpolate weeks at target SOH:
    # sort by SOH to ensure monotonic xp for np.interp (robust to slight wiggles)
    idx = np.argsort(s)  # SOH ascending
    s_sorted = s[idx]
    w_sorted = w[idx]

    # remove duplicate SOH values
    keep = ~pd.Series(s_sorted).duplicated(keep="first")
    s_sorted = s_sorted[keep.values]
    w_sorted = w_sorted[keep.values]
    if len(s_sorted) < 2:
        return None

    return float(np.interp(soh_target, s_sorted, w_sorted))


def idx_first_reach_soh(traj: pd.DataFrame, soh_target: float) -> int | None:
    """
    Return the FIRST integer index where interpolated SOH reaches <= soh_target.
    Returns None if never reaches.
    """
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
# Main
# -----------------------------
def main(dir_path: Path, out_dir: Path):
    exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    # TWO target SOHs:
    target_soh_features = 0.995   # feature row selection
    target_soh_plot = 0.98       # interpolation grid for plotting

    rows = []
    rows_ref = []

    # Reference cell names
    REF_NAMES = ["SPEED_LW_reference_1", "SPEED_LW_reference_2", "SPEED_LW_reference_3"]

    # CONFIG YOU WANT:
    K_CLOSEST_FEATURE = 20   # keep 20 closest in feature space -> ORANGE
    K_SLOWEST_BLACK = 1      # among those closest, mark slowest-aging in BLACK
    SOH_SLOW_TARGET = 0.96   # "slow" defined by max weeks to reach this SOH

    # NEW: Gaussian distribution target (index where SOH reaches this)
    SOH_DIST_TARGET = 0.97

    # SPEED: read only needed columns
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

    # Keep interpolated trajectories
    traj_by_cell_weeks: dict[str, pd.DataFrame] = {}
    traj_by_cell_thr: dict[str, pd.DataFrame] = {}

    # For Gaussian distributions (optional bookkeeping you had)
    SOH_GAUSS_TARGET = 0.965
    idx_at_soh_gauss_exp: dict[str, int] = {}
    soh_series_exp: dict[str, np.ndarray] = {}

    for csv_file in dir_path.glob("*.csv"):
        cell_name = csv_file.stem

        # Read only columns we need
        try:
            df = pd.read_csv(csv_file, usecols=lambda c: c in needed_cols)
        except Exception as e:
            print(f"[WARN] {cell_name}: read failed ({e}), skipping.")
            continue

        if "CU_time" not in df.columns:
            print(f"[WARN] {cell_name}: missing CU_time, skipping.")
            continue

        # weeks computation
        t = pd.to_datetime(df["CU_time"], errors="coerce", cache=True)
        if t.isna().all():
            print(f"[WARN] {cell_name}: CU_time invalid, skipping.")
            continue

        t0 = t.iloc[0]
        df["weeks"] = (t - t0).dt.total_seconds() / (7 * 24 * 3600)

        # exp conditions
        if any(c not in df.columns for c in exp_conds):
            print(f"[WARN] {cell_name}: missing exp_conds, skipping.")
            continue
        exp_row = exp_row_from_first_line(df, exp_conds)

        # Interpolation input (weeks-based)
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

        traj_by_cell_weeks[cell_name] = interpolated_plot_weeks[["weeks", "SOH"]].copy()

        # gaussian bookkeeping for exp cells
        if not exp_row.isna().any():
            soh_arr = interpolated_plot_weeks["SOH"].to_numpy(dtype=float)
            idx_g = int(np.argmin(np.abs(soh_arr - SOH_GAUSS_TARGET)))
            idx_at_soh_gauss_exp[cell_name] = idx_g
            soh_series_exp[cell_name] = soh_arr

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

        # --- pick the feature row for distance comparisons (uses target_soh_features) ---
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
    # Explicitly plot the 3 references again (visible on top) on ALL plots
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
            label="SPEED_LW_reference_13..15" if not added_ref_label else None,
            zorder=3,
        )
        ax_w.plot(
            traj_w["weeks"],
            traj_w["SOH"],
            color="red",
            alpha=0.95,
            linewidth=1.6,
            label="SPEED_LW_reference_13..15" if not added_ref_label else None,
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
                label="SPEED_LW_reference_13..15" if not added_ref_label else None,
                zorder=3,
            )

        added_ref_label = True

    # -----------------------------
    # Closest/farthest in feature space (NOW 3D)
    # -----------------------------
    x_col = "mean_d_dqdv_m_c"
    y_col = "mean_d_dqdv_l_c"
    z_col = "mean_d_dqdv_h_c"  # NEW

    K_FARTHEST = 20

    ref_sub = df_ref[df_ref["cell_name"].isin(REF_NAMES)][["cell_name", x_col, y_col, z_col]].dropna().reset_index(drop=True)
    exp_sub = df_exp[["cell_name", x_col, y_col, z_col]].dropna().reset_index(drop=True)

    closest_cellnames: list[str] = []
    farthest_cellnames: list[str] = []
    slowest_black_cellnames: list[str] = []

    if ref_sub.empty:
        print(f"[WARN] None of REF_NAMES found in df_ref: {REF_NAMES}")
    elif exp_sub.empty:
        print("[WARN] df_exp has no valid rows for the selected feature columns.")
    else:
        ref_xyz = ref_sub[[x_col, y_col, z_col]].to_numpy(dtype=float)
        exp_xyz = exp_sub[[x_col, y_col, z_col]].to_numpy(dtype=float)

        # -------------------------------------------------
        # UNIFIED normalization (refs + exp together)
        # -------------------------------------------------
        all_xyz = np.vstack([ref_xyz, exp_xyz])

        mu = all_xyz.mean(axis=0)
        sd = all_xyz.std(axis=0, ddof=1)
        sd = np.maximum(sd, 1e-12)  # numerical safety

        ref_n = (ref_xyz - mu) / sd
        exp_n = (exp_xyz - mu) / sd

        # -------------------------------------------------
        # Equal weighting (explicit but trivial)
        # -------------------------------------------------
        # (kept explicit so future weighting is easy)
        weights = np.array([1.0, 1.0, 1.0], dtype=float)
        ref_n *= weights
        exp_n *= weights

        # -------------------------------------------------
        # Distance to NEAREST reference
        # -------------------------------------------------
        dists = np.linalg.norm(exp_n[:, None, :] - ref_n[None, :, :], axis=2)
        dist = np.min(dists, axis=1)

        k1 = min(K_CLOSEST_FEATURE, len(exp_sub))
        closest_idx = np.argsort(dist)[:k1]
        closest_cellnames = exp_sub.loc[closest_idx, "cell_name"].tolist()

        k2 = min(K_FARTHEST, len(exp_sub))
        farthest_idx = np.argsort(dist)[-k2:]
        farthest_cellnames = exp_sub.loc[farthest_idx, "cell_name"].tolist()

        print(f"\nClosest {len(closest_cellnames)} exp cells (orange) to NEAREST(ref) at SOH={target_soh_features}:")
        for cn in closest_cellnames:
            print(" -", cn)

        # -----------------------------
        # pick slowest-aging among the closest (weeks to reach SOH_SLOW_TARGET)
        # -----------------------------
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

        # -----------------------------
        # 3D scatter figure
        # -----------------------------
        ref_mean = ref_sub[[x_col, y_col, z_col]].mean().to_numpy(dtype=float)

        fig_sc = plt.subplots(figsize=(8, 7))[0]
        ax_sc = fig_sc.add_subplot(111, projection="3d")

        # all exp
        ax_sc.scatter(
            exp_sub[x_col], exp_sub[y_col], exp_sub[z_col],
            s=18, alpha=0.7, color="blue",
            label="exp (all)",
        )

        # farthest
        ax_sc.scatter(
            exp_sub.loc[farthest_idx, x_col],
            exp_sub.loc[farthest_idx, y_col],
            exp_sub.loc[farthest_idx, z_col],
            s=70, alpha=0.95, color="green",
            edgecolors="k", linewidths=0.6,
            label=f"farthest {k2}",
        )

        # closest
        ax_sc.scatter(
            exp_sub.loc[closest_idx, x_col],
            exp_sub.loc[closest_idx, y_col],
            exp_sub.loc[closest_idx, z_col],
            s=70, alpha=0.95, color="orange",
            edgecolors="k", linewidths=0.6,
            label=f"closest {k1}",
        )

        # highlight slowest among closest
        if slowest_black_cellnames:
            row_black = exp_sub[exp_sub["cell_name"].isin(slowest_black_cellnames)]
            if not row_black.empty:
                ax_sc.scatter(
                    row_black[x_col], row_black[y_col], row_black[z_col],
                    s=180, color="black", marker="*",
                    edgecolors="k", linewidths=0.8,
                    label=f"slowest among closest (SOH={SOH_SLOW_TARGET})",
                )

        # refs
        ax_sc.scatter(
            ref_sub[x_col], ref_sub[y_col], ref_sub[z_col],
            s=90, alpha=1.0, color="red",
            edgecolors="k", linewidths=0.8,
            label="refs",
        )

        # mean(refs)
        ax_sc.scatter(
            [ref_mean[0]], [ref_mean[1]], [ref_mean[2]],
            s=150, color="black", marker="X",
            label="mean(refs)",
        )

        ax_sc.set_xlabel(x_col)
        ax_sc.set_ylabel(y_col)
        ax_sc.set_zlabel(z_col)
        ax_sc.set_title(f"3D feature space at SOH={target_soh_features}: distance to nearest(ref)")
        ax_sc.legend()

        # optional nicer view angle
        ax_sc.view_init(elev=20, azim=45)

        fig_sc.tight_layout()

    # -----------------------------
    # Overlay trajectories: closest = orange, farthest = green
    # -----------------------------
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
    # Overlay slowest among closest in BLACK on all 3 plots
    # -----------------------------
    added_black_label = False
    for cn in slowest_black_cellnames:
        traj_w_black = traj_by_cell_weeks.get(cn)
        if traj_w_black is None:
            continue

        ax.plot(
            traj_w_black["SOH"],
            color="black",
            alpha=1.0,
            linewidth=2.4,
            label="slowest among closest (black)" if not added_black_label else None,
            zorder=10,
        )
        ax_w.plot(
            traj_w_black["weeks"],
            traj_w_black["SOH"],
            color="black",
            alpha=1.0,
            linewidth=2.4,
            label="slowest among closest (black)" if not added_black_label else None,
            zorder=10,
        )

        traj_t_black = traj_by_cell_thr.get(cn)
        if traj_t_black is not None:
            ax_t.plot(
                traj_t_black["throughput_cum"],
                traj_t_black["SOH"],
                color="black",
                alpha=1.0,
                linewidth=2.4,
                label="slowest among closest (black)" if not added_black_label else None,
                zorder=10,
            )

        added_black_label = True

    # -----------------------------
    # NEW: Gaussian distributions of INDEX where SOH first reaches 0.96
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

    def plot_gauss(ax, data, label, color, bins=25):
        if len(data) < 2:
            return  # not enough points to estimate sigma
        data = np.asarray(data, dtype=float)
        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=1))
        sigma = max(sigma, 1e-6)

        ax.hist(data, bins=bins, density=True, alpha=0.25, color=color)

        x_min = float(np.min(data) - 3.0 * sigma)
        x_max = float(np.max(data) + 3.0 * sigma)
        x = np.linspace(x_min, x_max, 400)
        y = gaussian_pdf(x, mu, sigma)
        ax.plot(x, y, color=color, linewidth=2.0, label=f"{label}: μ={mu:.1f}, σ={sigma:.1f}, n={len(data)}")

    plot_gauss(ax_g, idx_blue,   "BLUE (other exp)", "blue")
    plot_gauss(ax_g, idx_orange, "ORANGE (closest)", "orange")
    plot_gauss(ax_g, idx_green,  "GREEN (farthest)", "green")

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
    fig.tight_layout()

    ax_w.set_xlabel("weeks")
    ax_w.set_ylabel("SOH")
    ax_w.set_title(
        f"SOH trajectories vs weeks (plot interpolation target SOH={target_soh_plot}; "
        f"feature comparison at SOH={target_soh_features})"
    )
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="best")
    fig_w.tight_layout()

    ax_t.set_xlabel("throughput_cum")
    ax_t.set_ylabel("SOH")
    ax_t.set_title(
        f"SOH trajectories vs throughput_cum (plot interpolation target SOH={target_soh_plot}; "
        f"feature comparison at SOH={target_soh_features})"
    )
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="best")
    fig_t.tight_layout()

    plt.show()
    return df_exp, df_ref


if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    out_dir = Path(r"C:\Users\Public\Documents\RL_project\out_lw\feature_plots")  # not used for saving
    main(dir_path, out_dir)
