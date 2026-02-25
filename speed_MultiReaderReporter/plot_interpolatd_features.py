from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import math
import matplotlib.pyplot as plt

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


def main(dir_path: Path):
    exp_conds = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]

    needed_cols = (
        ["CU_time"]
        + exp_conds
        + ["cap_ocv_dis"]
        + [
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c", "mean_d_dqdv_l_c_l",
        ]
        + ["throughput_cum", "mean_d_dqdv_h_d", "mean_d_dqdv_l_d"]
    )
    traj_weeks_by_cell: dict[str, pd.DataFrame] = {}

    for csv_file in dir_path.glob("*.csv"):
        cell_name = csv_file.stem

        try:
            df = pd.read_csv(csv_file, usecols=lambda c: c in needed_cols)
        except Exception as e:
            print(f"[WARN] {cell_name}: read failed ({e}), skipping.")
            continue

        t = pd.to_datetime(df["CU_time"], errors="coerce", cache=True)
        if t.isna().all():
            print(f"[WARN] {cell_name}: CU_time invalid, skipping.")
            continue

        t0 = t.iloc[0]
        df["weeks"] = (t - t0).dt.total_seconds() / (7 * 24 * 3600)

        interp_cols_weeks = [
            "weeks", "cap_ocv_dis",
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c", "mean_d_dqdv_l_c_l",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
            "mean_d_dqdv_h_d", "mean_d_dqdv_l_d",
        ]

        interp_cols_weeks2 = [
            "weeks", "cap_ocv_dis","SOH",
            "mean_d_dqdv_m_c", "var_d_dqdv_m_c", "mean_d_dqdv_l_c_l",
            "mean_d_dqdv_m_d", "var_d_dqdv_m_d",
            "mean_d_dqdv_h_c", "mean_d_dqdv_l_c",
            "mean_d_dqdv_h_d", "mean_d_dqdv_l_d",
        ]

        # skip if any required column is missing
        missing = [c for c in interp_cols_weeks if c not in df.columns]
        if missing:
            print(f"[WARN] {cell_name}: missing {missing}, skipping.")
            continue

        df_filter_weeks = df[interp_cols_weeks].copy()

        interpolated_plot_weeks = load_and_interpolate(
            df_filter_weeks, target_soh=0.98, interpolation_typ="weeks"
        )
        if interpolated_plot_weeks is None or interpolated_plot_weeks.empty:
            print(f"[WARN] {cell_name}: interpolation failed, skipping.")
            continue

        # store for later plotting
        traj_weeks_by_cell[cell_name] = interpolated_plot_weeks

        print(f"{cell_name}: stored interpolated df")

    REF_NAMES = ["SPEED_LW_reference_1", "SPEED_LW_reference_2", "SPEED_LW_reference_3"]

    # Features to plot (everything except the x-axis column)
    x_col = "weeks"
    feat_cols = [c for c in interp_cols_weeks2 if c != x_col]

    n_feats = len(feat_cols)
    ncols = 3  # change to 2/4 if you like
    nrows = math.ceil(n_feats / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows), sharex=True)
    axes = axes.ravel()  # flatten to 1D for easy indexing

    # Plot: each feature gets its own subplot
    for i, feat in enumerate(feat_cols):
        ax = axes[i]

        for cell_name, df_interp in traj_weeks_by_cell.items():
            if x_col not in df_interp.columns or feat not in df_interp.columns:
                continue

            x = df_interp[x_col].to_numpy()
            y = df_interp[feat].to_numpy()

            if cell_name in REF_NAMES:
                ax.plot(y, color="red", linewidth=1.8, alpha=0.95)
            else:
                ax.plot(y, color="blue", linewidth=0.9, alpha=0.25)

        ax.set_title(feat)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots (if grid bigger than number of features)
    for j in range(n_feats, len(axes)):
        axes[j].axis("off")

    # Common labels
    for ax in axes[:n_feats]:
        ax.set_xlabel("weeks")

    # Simple legend (one handle per group)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="red", lw=2, label="REF_NAMES"),
        Line2D([0], [0], color="blue", lw=2, alpha=0.35, label="other cells"),
    ]
    fig.legend(handles=legend_handles, loc="upper right")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    out_dir = Path(r"C:\Users\Public\Documents\RL_project\out_lw\feature_plots")  # not used for saving
    main(dir_path)