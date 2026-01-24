# speed_MultiReaderReporter/main.py
from __future__ import annotations

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

# --- relative paths ---
here = Path(__file__).resolve().parent
sys.path.append(str(here))
sys.path.append(str(here / "core"))
sys.path.append(str(here / "loaders"))
sys.path.append(str(here / "utils"))


# -----------------------------
# Interpolation (returns SOH + weeks)
# -----------------------------
def load_and_interpolate(df: pd.DataFrame, target_soh: float, interpolation_typ: str):
    """
    Interpolate all columns (incl. SOH) onto a new reference grid.
    Returns a dataframe that includes: interpolated feature columns + SOH + weeks.
    """
    df = df.copy()
    df.iloc[0] = df.iloc[0].fillna(0)

    # compute SOH
    df["SOH"] = df["cap_ocv_dis"] / df["cap_ocv_dis"].iloc[0]

    if interpolation_typ == "weeks":
        ref_name = "weeks"
    elif interpolation_typ == "throughput":
        ref_name = "throughput"
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

    # interpolate EVERYTHING except ref (so SOH will be included)
    feature_cols = [c for c in df.columns if c != ref_name]
    feature = df[feature_cols].to_numpy(dtype=float)

    # find reference location where SOH reaches target_soh (SOH decreases => reverse)
    interpolated_ref = np.interp(target_soh, target_data[::-1], reference[::-1])

    # 6 points up to target, then continue with same spacing
    new_ref_points_cut = np.linspace(0.0, float(interpolated_ref), 6)
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

    # PCHIP interpolation per column (fallback to linear if NaNs / errors)
    interpolated_columns = []
    for j in range(feature.shape[1]):
        yj = feature[:, j]

        if np.isnan(yj).any():
            interpolated_columns.append(np.interp(new_ref_points, reference, yj))
            continue

        try:
            f = PchipInterpolator(reference, yj, extrapolate=False)
            interpolated_columns.append(f(new_ref_points))
        except Exception:
            interpolated_columns.append(np.interp(new_ref_points, reference, yj))

    interpolated_data = np.stack(interpolated_columns, axis=1)
    out = pd.DataFrame(interpolated_data, columns=feature_cols)
    out[ref_name] = new_ref_points
    return out


def exp_row_from_first_line(df: pd.DataFrame, exp_conds: list[str]) -> pd.Series:
    """Return exp condition values from row 0 as a Series."""
    return df.loc[0, exp_conds]


# -----------------------------
# Main
# -----------------------------
def main(dir_path: Path, out_dir: Path):
    exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]
    target_soh = 0.98

    rows = []
    rows_ref = []

    # The 3 reference cell names you want to use for "closest/farthest"
    REF_NAMES = ["SPEED_LW_reference_4", "SPEED_LW_reference_5", "SPEED_LW_reference_6"]

    # SPEED: read only needed columns
    needed_cols = (
        ["CU_time"]
        + exp_conds
        + ["cap_ocv_dis"]
        + ["mean_d_dqdv_m_c", "var_d_dqdv_m_c", "mean_d_dqdv_m_d", "var_d_dqdv_m_d", "mean_d_dqdv_h_c"]
    )

    # ---- ONE plot for all cells (SOH trajectories) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    added_blue_label = False
    added_red_label = False

    # Keep interpolated trajectories so we can highlight "closest/farthest" later
    traj_by_cell: dict[str, pd.DataFrame] = {}

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

        # FAST weeks computation: parse datetime once
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

        # Interpolation input
        interp_cols = [
            "weeks",
            "cap_ocv_dis",
            "mean_d_dqdv_m_c",
            "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d",
            "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c",
        ]
        missing = [c for c in interp_cols if c not in df.columns]
        if missing:
            print(f"[WARN] {cell_name}: missing {missing}, skipping.")
            continue

        df_filter = df[interp_cols].copy()
        interpolated_df = load_and_interpolate(df_filter, target_soh, interpolation_typ="weeks")
        if interpolated_df is None or interpolated_df.empty:
            print(f"[WARN] {cell_name}: interpolation failed, skipping.")
            continue

        if "SOH" not in interpolated_df.columns or "weeks" not in interpolated_df.columns:
            print(f"[WARN] {cell_name}: interpolated_df missing SOH/weeks, skipping.")
            continue

        # Save trajectory for later highlighting
        traj_by_cell[cell_name] = interpolated_df[["weeks", "SOH"]].copy()

        # ---- Plot SOH for this cell (NO x-axis used: index only) ----
        # Red if (exp_row has NaNs) AND ("LW_reference" in cell name), else blue
        is_ref_case = exp_row.isna().any() and ("LW_reference" in cell_name)

        if is_ref_case:
            label = "LW_reference (exp NaN)" if not added_red_label else None
            added_red_label = True
            ax.plot(
                interpolated_df["SOH"],
                color="red",
                alpha=0.25,
                linewidth=0.6,
                label=label,
                zorder=1,
            )
        else:
            label = "other" if not added_blue_label else None
            added_blue_label = True
            ax.plot(
                interpolated_df["SOH"],
                color="blue",
                alpha=0.20,
                linewidth=0.55,
                label=label,
                zorder=2,
            )

        # ---- Build combined rows ----
        idx = (interpolated_df["SOH"] - target_soh).abs().idxmin()
        row_soh = interpolated_df.loc[idx]

        combined_row = pd.concat([pd.Series({"cell_name": cell_name}), exp_row, row_soh], axis=0)

        if exp_row.isna().any():
            rows_ref.append(combined_row)
        else:
            rows.append(combined_row)

        print(f"{cell_name}: success")

    # Finalize dataframes
    df_ref = pd.DataFrame(rows_ref).reset_index(drop=True)
    df_exp = pd.DataFrame(rows).reset_index(drop=True)

    # -----------------------------
    # Explicitly plot the 3 references again (guaranteed visible)
    # -----------------------------
    added_ref_label = False
    for ref_name in REF_NAMES:
        traj = traj_by_cell.get(ref_name)
        if traj is None:
            continue
        ax.plot(
            traj["SOH"],
            color="red",
            alpha=0.95,
            linewidth=1.6,
            label="SPEED_LW_reference_1..3" if not added_ref_label else None,
            zorder=3,
        )
        added_ref_label = True

    # -----------------------------
    # Closest/farthest selection based on TWO features (as requested)
    # distance = min over refs (|dx| + |dy|)
    # -----------------------------
    x_col = "mean_d_dqdv_m_d"
    y_col = "mean_d_dqdv_h_d"
    if y_col not in df_ref.columns or y_col not in df_exp.columns:
        y_col = "mean_d_dqdv_h_c"  # fallback

    K_CLOSEST = 30
    K_FARTHEST = 15

    ref_sub = df_ref[df_ref["cell_name"].isin(REF_NAMES)][["cell_name", x_col, y_col]].dropna().reset_index(drop=True)
    exp_sub = df_exp[["cell_name", x_col, y_col]].dropna().reset_index(drop=True)

    closest_cellnames: list[str] = []
    farthest_cellnames: list[str] = []

    if ref_sub.empty:
        print(f"[WARN] None of REF_NAMES found in df_ref: {REF_NAMES}")
    elif exp_sub.empty:
        print("[WARN] df_exp has no valid rows for the selected feature columns.")
    else:
        ref_xy = ref_sub[[x_col, y_col]].to_numpy(dtype=float)
        exp_xy = exp_sub[[x_col, y_col]].to_numpy(dtype=float)

        dist = np.abs(exp_xy[:, None, :] - ref_xy[None, :, :]).sum(axis=2)  # (n_exp, n_ref)
        min_dist = dist.min(axis=1)

        k1 = min(K_CLOSEST, len(exp_sub))
        closest_idx = np.argsort(min_dist)[:k1]
        closest_cellnames = exp_sub.loc[closest_idx, "cell_name"].tolist()

        k2 = min(K_FARTHEST, len(exp_sub))
        farthest_idx = np.argsort(min_dist)[-k2:]
        farthest_cellnames = exp_sub.loc[farthest_idx, "cell_name"].tolist()

        print(f"\nClosest {len(closest_cellnames)} exp cells to refs (by |dx|+|dy|):")
        for cn in closest_cellnames:
            print(" -", cn)

        print(f"\nFarthest {len(farthest_cellnames)} exp cells from refs (by |dx|+|dy|):")
        for cn in farthest_cellnames:
            print(" -", cn)

        # 2D scatter in its own figure (prevents blank plot issues)
        fig_sc, ax_sc = plt.subplots(figsize=(7, 6))
        ax_sc.scatter(ref_sub[x_col], ref_sub[y_col], s=70, alpha=0.95, color="red", label="refs (1..3)")
        ax_sc.scatter(exp_sub[x_col], exp_sub[y_col], s=30, alpha=0.45, color="blue", label="exp (all)")
        ax_sc.scatter(
            exp_sub.loc[closest_idx, x_col],
            exp_sub.loc[closest_idx, y_col],
            s=90,
            alpha=0.95,
            color="orange",
            edgecolors="k",
            linewidths=0.6,
            label=f"closest {k1}",
        )
        ax_sc.scatter(
            exp_sub.loc[farthest_idx, x_col],
            exp_sub.loc[farthest_idx, y_col],
            s=90,
            alpha=0.95,
            color="green",
            edgecolors="k",
            linewidths=0.6,
            label=f"farthest {k2}",
        )
        ax_sc.set_xlabel(x_col)
        ax_sc.set_ylabel(y_col)
        ax_sc.set_title("2D feature space: refs vs exp (closest orange, farthest green)")
        ax_sc.grid(True, alpha=0.25)
        ax_sc.legend()
        fig_sc.tight_layout()
        #plt.show()

    # -----------------------------
    # Overlay trajectories: closest = orange, farthest = green
    # -----------------------------
    added_orange_label = False
    for cn in closest_cellnames:
        traj = traj_by_cell.get(cn)
        if traj is None:
            continue
        ax.plot(
            traj["SOH"],
            color="orange",
            alpha=0.85,
            linewidth=1.0,
            label="closest exp (orange)" if not added_orange_label else None,
            zorder=4,
        )
        added_orange_label = True

    added_green_label = False
    for cn in farthest_cellnames:
        traj = traj_by_cell.get(cn)
        if traj is None:
            continue
        ax.plot(
            traj["SOH"],
            color="green",
            alpha=0.85,
            linewidth=1.0,
            label="farthest exp (green)" if not added_green_label else None,
            zorder=5,
        )
        added_green_label = True

    # Finalize trajectory plot (no real x-axis used)
    ax.set_xlabel("index")
    ax.set_ylabel("SOH")
    ax.set_title("SOH trajectories (blue exp, red refs, orange closest, green farthest)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # DO NOT SAVE — just show (activate correct figure to avoid blank)
    fig.tight_layout()
    plt.show()

    # Tag df_exp for later use (optional)
    df_exp = df_exp.copy()
    df_exp["is_closest_20_to_refs_by_2d"] = df_exp["cell_name"].isin(closest_cellnames)
    df_exp["is_farthest_10_from_refs_by_2d"] = df_exp["cell_name"].isin(farthest_cellnames)

    print("\nDone.")
    print("df_exp shape:", df_exp.shape)
    print("df_ref shape:", df_ref.shape)

    return df_exp, df_ref


if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    out_dir = Path(r"C:\Users\Public\Documents\RL_project\out_lw\feature_plots")  # kept for compatibility, not used for saving
    main(dir_path, out_dir)
