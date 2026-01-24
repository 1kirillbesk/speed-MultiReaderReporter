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

    remaining_points = []
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

    # interpolate each feature column
    # interpolated_columns = [
    #     np.interp(new_ref_points, reference, feature[:, j]) for j in range(feature.shape[1])
    # ]
    # interpolated_data = np.stack(interpolated_columns, axis=1)

    interpolated_columns = []
    for j in range(feature.shape[1]):
        yj = feature[:, j]

        # If there are NaNs, fall back to np.interp (PCHIP cannot handle NaNs)
        if np.isnan(yj).any():
            interpolated_columns.append(np.interp(new_ref_points, reference, yj))
            continue

        try:
            f = PchipInterpolator(reference, yj, extrapolate=False)
            interpolated_columns.append(f(new_ref_points))
        except Exception:
            # fallback if PCHIP fails for any reason
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
    target_soh = 0.995

    rows = []
    rows_ref = []

    # SPEED: read only needed columns
    needed_cols = (
        ["CU_time"] +
        exp_conds +
        ["cap_ocv_dis"] +
        ["mean_d_dqdv_m_c", "var_d_dqdv_m_c", "mean_d_dqdv_m_d", "var_d_dqdv_m_d", "mean_d_dqdv_h_c"]
    )

    # ---- ONE plot for all cells ----
    fig, ax = plt.subplots(figsize=(9, 5))
    added_blue_label = False
    added_red_label = False

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
        interp_cols = ["weeks", "cap_ocv_dis",
                       "mean_d_dqdv_m_c", "var_d_dqdv_m_c",
                       "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
                       "mean_d_dqdv_h_c"]
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

        # ---- Plot SOH for this cell onto the SAME axes ----
        # Red if (exp_row has NaNs) AND ("LW_reference" in cell name), else blue
        is_ref_case = exp_row.isna().any() and ("LW_reference" in cell_name)

        if is_ref_case:
            label = "LW_reference (exp NaN)" if not added_red_label else None
            added_red_label = True
            ax.plot(interpolated_df["SOH"], color="red", alpha=0.25, linewidth=0.6, label=label)
        else:
            label = "other" if not added_blue_label else None
            added_blue_label = True
            ax.plot(interpolated_df["SOH"], color="blue", alpha=0.45, linewidth=0.7, label=label)

        # ---- Build your combined rows (unchanged logic, but now SOH exists) ----
        # pick row closest to target_soh
        idx = (interpolated_df["SOH"] - target_soh).abs().idxmin()
        row_soh = interpolated_df.loc[idx]

        combined_row = pd.concat(
            [pd.Series({"cell_name": cell_name}), exp_row, row_soh],
            axis=0
        )

        if exp_row.isna().any():
            rows_ref.append(combined_row)
        else:
            rows.append(combined_row)

        print(f"{cell_name}: success")

    # Finalize dataframes
    df_ref = pd.DataFrame(rows_ref).reset_index(drop=True)
    df_exp = pd.DataFrame(rows).reset_index(drop=True)

    # Finalize plot
    ax.set_xlabel("weeks")
    ax.set_ylabel("SOH")
    ax.set_title("SOH vs weeks (all cells)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Save + show
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "SOH_all_cells.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.show()

    print("\nDone.")
    print("df_exp shape:", df_exp.shape)
    print("df_ref shape:", df_ref.shape)
    print(f"Saved plot: {plot_path}")

    x_col = "mean_d_dqdv_m_d"
    y_col = "mean_d_dqdv_h_d"
    if y_col not in df_ref.columns or y_col not in df_exp.columns:
        # fallback to the column that appears in your screenshot
        y_col = "mean_d_dqdv_h_c"

    print("Using columns:", x_col, y_col)

    # --- extract points & clean NaNs ---
    ref_pts = df_ref[["cell_name", x_col, y_col]].copy()
    exp_pts = df_exp[["cell_name", x_col, y_col]].copy()

    ref_pts = ref_pts.dropna(subset=[x_col, y_col]).reset_index(drop=True)
    exp_pts = exp_pts.dropna(subset=[x_col, y_col]).reset_index(drop=True)

    ref_xy = ref_pts[[x_col, y_col]].to_numpy(dtype=float)
    exp_xy = exp_pts[[x_col, y_col]].to_numpy(dtype=float)

    # --- find 20 closest exp points to the ref set ---
    # distance of each exp point to its nearest ref point (Euclidean)
    # (brute force but fine for your sizes)
    dists = np.sqrt(((exp_xy[:, None, :] - ref_xy[None, :, :]) ** 2).sum(axis=2))  # shape (n_exp, n_ref)
    min_dist = dists.min(axis=1)  # nearest ref distance for each exp point

    k = min(20, len(exp_pts))
    closest_idx = np.argsort(min_dist)[:k]
    closest_cellnames = exp_pts.loc[closest_idx, "cell_name"].tolist()

    print(f"\nSelected {k} closest exp cells (by nearest-ref distance):")
    for cn in closest_cellnames:
        print(" -", cn)

    # --- plot ---
    plt.figure(figsize=(7, 6))

    # red = ref
    plt.scatter(ref_pts[x_col], ref_pts[y_col], s=35, alpha=0.8, label="df_ref (red)")

    # blue = exp (all)
    plt.scatter(exp_pts[x_col], exp_pts[y_col], s=30, alpha=0.45, label="df_exp (blue)")

    # orange = closest exp
    plt.scatter(
        exp_pts.loc[closest_idx, x_col],
        exp_pts.loc[closest_idx, y_col],
        s=80,
        alpha=0.95,
        label=f"closest {k} exp (orange)",
        edgecolors="k",
        linewidths=0.6,
    )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"2D feature scatter + closest {k} exp to ref")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- OPTIONAL: tag df_exp with a flag so you can reuse it later ---
    df_exp = df_exp.copy()
    df_exp["is_closest_20_to_ref"] = df_exp["cell_name"].isin(closest_cellnames)


    return df_exp, df_ref


if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    out_dir = Path(r"C:\Users\Public\Documents\RL_project\out_lw\feature_plots")
    main(dir_path, out_dir)
