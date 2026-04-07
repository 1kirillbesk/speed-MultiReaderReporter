from pathlib import Path
import math
import gc
import os
import random
import ast

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde

from core.capacity import *

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from analyze_linear_prediction import build_regression_table_cap93_and_var_at_thr
import xgboost as xgb
from sklearn.metrics import r2_score
from analyze_3dim_time_2step import (
    make_features_from_raw,
    train_xgb_no_val_and_shap,
    monte_carlo_best_conditions_for_distance,
)
from utils.combined_cost_search import train_model_var_dqc

# =============================================================================
# CONFIG
# =============================================================================
FEATURE_NAMES_FOR_DIST = [
    "mean_mid_cha",
    "mean_high_cha",
    "mean_pla_cha",
]

interp_dir = Path(r"C:\Users\Victus\PycharmProjects\ExpSpeed\out_lw\cell_feature")
out_fig_dir = Path(r"C:\Users\Victus\PycharmProjects\ExpSpeed\out_lw\out_figure")
out_fig_dir.mkdir(parents=True, exist_ok=True)

REF_NAMES = ["SPEED_LW_reference_1", "SPEED_LW_reference_2","SPEED_LW_reference_3"]

x_col = "Vcha"
y_col = "dQdVcha"

v_1 = 3.28
v_2 = 3.35
v_3 = 3.45

TARGET_SOH_FEATURES = 0.995   # SOH at which to compare features (closest cells)
TARGET_SOH_PLOT = 0.98        # SOH for interpolation grid
SOH_STEP_TARGET = 0.955
FEATURE_STEP_LOGLOG = 7
THROUGHPUT_FEATURE_LOGLOG = 2000000.0 / 3600.0 / 3.4
SOH_THROUGHPUT_TARGET = 0.955
EXTRAPOLATE_SOH_TARGET_IF_NOT_REACHED = False
REF_IGNORE_FIRST_WEEK = True
REF_IGNORE_WEEKS_LE = 1.0

K_CLOSEST = 20
K_FARTHEST = 20
PRINT_CLOSEST_EXISTING = False
TOP_TEMPS = [40, 25, 15]
TOP_PER_TEMP = 10

# Choose interpolation method here
INTERP_METHOD = "linear"      # options: "linear" or "cubic"

INTERP_FEATURES = [
    "mean_low_cha", "mean_mid_cha", "mean_high_cha", "mean_pla_cha",
    "var_low_cha", "var_mid_cha", "var_high_cha", "var_pla_cha"
]

EXP_CONDS = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]
MC_ALLOWED_TEMP = np.array([15.0, 25.0, 40.0], dtype=float)
MC_ALLOWED_SOC_START = np.arange(0, 90, 10, dtype=float)
MC_ALLOWED_SOC_END = np.arange(30, 110, 10, dtype=float)
MC_ALLOWED_C_RATE_CHG = np.arange(0.5, 1.75, 0.25, dtype=float)
MC_ALLOWED_C_RATE_DCHG = np.arange(1, 3.25, 0.25, dtype=float)
MC_MIN_SOC_DELTA = 10.0
MC_SAMPLES_PER_TEMP = 100
exp_conditions = {}
traj_by_cell_reg = {}

# Model config
FEATURE_COLS = ["SOH", "capacity", "var_low_cha", "var_mid_cha", "var_high_cha", "var_pla_cha", "mean_pla_cha"]
INPUT_STEPS = 8
INPUT_FEATURES = len(FEATURE_COLS)

TEST_NAMES = [
    "SPEED_LW_reference_1",
    "SPEED_LW_reference_2",
    "SPEED_LW_reference_3",
]

EPOCHS = 500
BATCH_SIZE = 8
LR = 0.001
DROPOUT = 0.1
VALIDATION_SPLIT = 0.2
USE_VALIDATION = True
N_SEEDS = 10
INITIAL_CAP_HIST_BINS = 18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse_loss = nn.MSELoss()


# =============================================================================
# HELPERS
# =============================================================================
def parse_array_string(s):
    s = str(s).replace("NBSP", " ").replace("\\n", " ").replace("\n", " ")
    s = s.strip("[] ")
    return np.fromstring(s, sep=" ")


def clean_xy_for_interp(x, y):
    """
    Clean x/y for interpolation:
    - convert to float arrays
    - remove non-finite values
    - sort by x ascending
    - remove duplicate x values (keep first)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return None, None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_ser = pd.Series(x)
    keep = ~x_ser.duplicated(keep="first")
    x = x[keep.values]
    y = y[keep.values]

    if len(x) < 2:
        return None, None

    return x, y


def interp_1d(x_old, y_old, x_new, method="linear"):
    """
    1D interpolation with selectable method:
    - linear: np.interp
    - cubic : CubicSpline with fallback to linear
    """
    x_old, y_old = clean_xy_for_interp(x_old, y_old)
    if x_old is None or y_old is None:
        return None

    x_new = np.asarray(x_new, dtype=float)

    if method == "linear":
        return np.interp(x_new, x_old, y_old)

    if method == "cubic":
        # Need at least 3 points for cubic spline
        if len(x_old) < 3:
            return np.interp(x_new, x_old, y_old)
        try:
            cs = CubicSpline(x_old, y_old, bc_type="natural", extrapolate=False)
            y_new = cs(x_new)

            # fallback where CubicSpline returns nan (outside range)
            nan_mask = ~np.isfinite(y_new)
            if np.any(nan_mask):
                y_lin = np.interp(x_new, x_old, y_old)
                y_new[nan_mask] = y_lin[nan_mask]
            return y_new
        except Exception:
            return np.interp(x_new, x_old, y_old)

    raise ValueError("method must be 'linear' or 'cubic'")


def interp_value_at_target(x, y, x_target, method="linear"):
    """
    Interpolate scalar y-value at x_target.
    """
    y_interp = interp_1d(x, y, np.array([x_target], dtype=float), method=method)
    if y_interp is None or len(y_interp) == 0 or not np.isfinite(y_interp[0]):
        return None
    return float(y_interp[0])


def interp_value_with_optional_extrapolation(x, y, x_target, method="linear", allow_extrapolation=False):
    """
    Interpolate scalar y-value at x_target.
    If x_target is outside data range and allow_extrapolation=True,
    use linear extrapolation from the nearest edge segment.
    """
    x_clean, y_clean = clean_xy_for_interp(x, y)
    if x_clean is None or y_clean is None:
        return None

    x_target = float(x_target)
    x_min = float(x_clean[0])
    x_max = float(x_clean[-1])

    if x_min <= x_target <= x_max:
        return interp_value_at_target(x_clean, y_clean, x_target, method=method)

    if not allow_extrapolation or len(x_clean) < 2:
        return None

    if x_target < x_min:
        x0, x1 = float(x_clean[0]), float(x_clean[1])
        y0, y1 = float(y_clean[0]), float(y_clean[1])
    else:
        x0, x1 = float(x_clean[-2]), float(x_clean[-1])
        y0, y1 = float(y_clean[-2]), float(y_clean[-1])

    dx = x1 - x0
    if abs(dx) < 1e-12:
        return None

    slope = (y1 - y0) / dx
    return float(y0 + slope * (x_target - x0))


def mape_on_references_from_log_fit(coeffs, ref_log_x, ref_log_y_true, ref_y_true=None):
    """
    Compute MAPE for reference points using a linear fit in log space.
    Returns:
      - mape_log   : MAPE (%) in log-space target
      - mape_orig  : MAPE (%) in original target units (if ref_y_true is provided), else np.nan
    """
    ref_log_x = np.asarray(ref_log_x, dtype=float)
    ref_log_y_true = np.asarray(ref_log_y_true, dtype=float)
    if len(ref_log_x) == 0 or len(ref_log_y_true) == 0:
        return np.nan, np.nan
    if len(ref_log_x) != len(ref_log_y_true):
        n = min(len(ref_log_x), len(ref_log_y_true))
        ref_log_x = ref_log_x[:n]
        ref_log_y_true = ref_log_y_true[:n]
    if len(ref_log_x) == 0:
        return np.nan, np.nan

    ref_log_pred = np.polyval(coeffs, ref_log_x)
    denom_log = np.maximum(np.abs(ref_log_y_true), 1e-12)
    mape_log = float(np.mean(np.abs((ref_log_pred - ref_log_y_true) / denom_log)) * 100.0)

    mape_orig = np.nan
    if ref_y_true is not None:
        ref_y_true = np.asarray(ref_y_true, dtype=float)
        if len(ref_y_true) != len(ref_log_pred):
            n = min(len(ref_y_true), len(ref_log_pred))
            ref_y_true = ref_y_true[:n]
            ref_log_pred = ref_log_pred[:n]
        if len(ref_y_true) > 0:
            ref_y_pred = np.exp(ref_log_pred)
            denom_orig = np.maximum(np.abs(ref_y_true), 1e-12)
            mape_orig = float(np.mean(np.abs((ref_y_pred - ref_y_true) / denom_orig)) * 100.0)

    return mape_log, mape_orig


def load_and_interpolate(df, target_soh, interpolation_typ="weeks", method="linear", throughput_max=None):
    """
    Interpolate all feature columns on a new reference grid.
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
        df = df[df["throughput_cum"] <= float(throughput_max)].copy().reset_index(drop=True)

    if df.empty or df[ref_name].isna().any() or df["SOH"].isna().any():
        return None

    reference = df[ref_name].to_numpy(dtype=float)
    target_data = df["SOH"].to_numpy(dtype=float)

    reference, target_data = clean_xy_for_interp(reference, target_data)
    if reference is None or target_data is None:
        return None

    # SOH typically decreases, so reverse for interpolation over SOH -> reference
    ref_for_soh = reference[::-1]
    soh_for_ref = target_data[::-1]

    if len(ref_for_soh) < 2:
        return None
    if method == "cubic" and len(ref_for_soh) < 3:
        method_local = "linear"
    else:
        method_local = method

    interpolated_ref = interp_value_at_target(
        x=soh_for_ref,
        y=ref_for_soh,
        x_target=target_soh,
        method=method_local
    )
    if interpolated_ref is None:
        return None

    new_ref_points_cut = np.linspace(0.0, float(interpolated_ref), 15)
    spacing = float(np.mean(np.diff(new_ref_points_cut))) if len(new_ref_points_cut) > 1 else 0.0
    if spacing <= 0:
        return None

    remaining_points = []
    cur = float(interpolated_ref)
    iters = 0
    while cur + spacing <= float(reference[-1]):
        cur += spacing
        remaining_points.append(cur)
        iters += 1
        if iters >= 5000:
            return None

    new_ref_points = np.concatenate([new_ref_points_cut, np.array(remaining_points, dtype=float)])

    feature_cols = [c for c in df.columns if c != ref_name]
    out = pd.DataFrame(index=np.arange(len(new_ref_points)))

    for col in feature_cols:
        yj = df[col].to_numpy(dtype=float)
        y_new = interp_1d(reference, yj, new_ref_points, method=method_local)
        if y_new is None:
            return None
        out[col] = y_new

    out[ref_name] = new_ref_points
    if "SOH" in out.columns:
        out["SOH"] = out["SOH"].clip(lower=0.0, upper=1.05)

    return out


def features_at_soh(feat_dict, target_soh=0.995, method="linear", feature_names=None):
    """
    Use capacity from feat_dict to get SOH per row,
    then interpolate each feature list to target_soh.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES_FOR_DIST
    cap = np.array(feat_dict["capacity"], dtype=float)
    cap = cap[np.isfinite(cap)]

    if len(cap) < 2 or cap[0] == 0:
        return None

    soh = cap / cap[0]
    soh_min = np.nanmin(soh)
    soh_max = np.nanmax(soh)
    if not (soh_min <= target_soh <= soh_max):
        return None
    out = {}

    for fname in feature_names:
        vals = feat_dict.get(fname, [])
        if len(vals) == 0:
            return None

        vals = np.array(vals, dtype=float)
        n = min(len(soh), len(vals))
        if n < 2:
            return None

        # SOH decreasing -> reverse so x is ascending
        x = soh[:n][::-1]
        y = vals[:n][::-1]

        val_at_target = interp_value_at_target(x, y, target_soh, method=method)
        if val_at_target is None:
            return None

        out[fname] = float(val_at_target)

    return out


def set_seed(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_lowest_distance_per_temp(df, temp_col="temp", dist_col="dist_feat", temps=None, n_per_temp=10):
    if df is None or df.empty:
        return df

    if temps is None:
        temps = TOP_TEMPS

    out_parts = []
    df_local = df.copy()
    df_local[temp_col] = pd.to_numeric(df_local[temp_col], errors="coerce")

    for temp in temps:
        df_temp = df_local[df_local[temp_col] == float(temp)].nsmallest(n_per_temp, dist_col)
        out_parts.append(df_temp)

    if not out_parts:
        return df_local.iloc[0:0].copy()

    return pd.concat(out_parts, ignore_index=True)


def create_balanced_val_split(X, y, val_fraction=0.2, bins=3):
    y = np.array(y)
    y_bins = pd.qcut(y, q=bins, labels=False, duplicates="drop")
    mid_val_frac = int(val_fraction * len(y)) / len(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=mid_val_frac, random_state=42)
    train_idx, val_idx = next(sss.split(X, y_bins))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# =============================================================================
# MODEL
# =============================================================================
class TinyTemporalCNN(nn.Module):
    def __init__(self, input_dims, timesteps, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dims, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(8)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =============================================================================
# STEP 1: LOAD CSVS, PLOT RAW CURVES, EXTRACT FEATURES
# =============================================================================
results = {}
results_ref = {}

for csv_file in sorted(interp_dir.glob("*.csv")):
    if csv_file.name.startswith("_"):
        continue

    cell_name = csv_file.stem

    # only process refs and cycle cells
    is_ref = cell_name in REF_NAMES
    is_cycle = "cycle" in cell_name.lower()
    if not is_ref and not is_cycle:
        continue

    df = pd.read_csv(csv_file)

    if all(c in df.columns for c in EXP_CONDS):
        exp_row = df.loc[0, EXP_CONDS]
        exp_conditions[cell_name] = exp_row.to_dict()
    else:
        missing_conds = [c for c in EXP_CONDS if c not in df.columns]
        if missing_conds:
            print(f"[WARN] {cell_name}: missing exp_conds {missing_conds}")

    if x_col not in df.columns or y_col not in df.columns or "Q_intVcha" not in df.columns:
        continue

    # keep original string columns for parse_array_string
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in range(len(df)):
        try:
            x_arr = parse_array_string(df[x_col].iloc[idx])
            y_arr = parse_array_string(df[y_col].iloc[idx])
        except Exception:
            continue

        if len(x_arr) != len(y_arr):
            continue

        ax.plot(x_arr, y_arr, alpha=0.3, linewidth=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{cell_name}: {y_col} vs {x_col}")
    ax.grid(True, alpha=0.3)
    ax.axvline(x=v_1, color="black", linestyle="--", linewidth=1.0, label=f"v_1={v_1}")
    ax.axvline(x=v_2, color="black", linestyle="-.", linewidth=1.0, label=f"v_2={v_2}")
    ax.axvline(x=v_3, color="black", linestyle=":", linewidth=1.0, label=f"v_3={v_3}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_fig_dir / f"{cell_name}.png", dpi=150)
    plt.close(fig)

    # Convert columns needed by feature functions
    df["dQdVcha"] = df["dQdVcha"].str.strip("[]").str.split().apply(lambda x: np.asarray(x, dtype=float))
    df["Vcha"] = df["Vcha"].str.strip("[]").str.split().apply(lambda x: np.asarray(x, dtype=float))
    df["Q_intVcha"] = df["Q_intVcha"].str.strip("[]").str.split().apply(lambda x: np.asarray(x, dtype=float))

    # calculate features
    mean_low_cha, var_low_cha = window_delta_mean_var(df, x_col="Vcha", y_col="dQdVcha", x_lo=2, x_hi=v_2)
    mean_mid_cha, var_mid_cha = window_delta_mean_var(df, x_col="Vcha", y_col="dQdVcha", x_lo=v_1, x_hi=v_2)
    mean_high_cha, var_high_cha = window_delta_mean_var(df, x_col="Vcha", y_col="dQdVcha", x_lo=v_2, x_hi=v_3)
    mean_pla_cha, var_pla_cha = window_delta_mean_var(df, x_col="Vcha", y_col="dQdVcha", x_lo=v_3, x_hi=3.6)

    capacity_arr = df["Q_intVcha"].apply(lambda x: x[-1]).to_numpy(dtype=float)
    throughput_arr = np.cumsum(np.array(df["throughput_sum"], dtype=float)) / 3600.0

    df["CU_time"] = pd.to_datetime(df["CU_time"])
    t0 = df["CU_time"].iloc[0]
    df["time_weeks"] = (df["CU_time"] - t0).dt.total_seconds() / (7 * 24 * 3600)
    time_array = df["time_weeks"].to_numpy(dtype=float)

    mean_low_cha = np.asarray(mean_low_cha, dtype=float)
    var_low_cha = np.asarray(var_low_cha, dtype=float)
    mean_mid_cha = np.asarray(mean_mid_cha, dtype=float)
    var_mid_cha = np.asarray(var_mid_cha, dtype=float)
    mean_high_cha = np.asarray(mean_high_cha, dtype=float)
    var_high_cha = np.asarray(var_high_cha, dtype=float)
    mean_pla_cha = np.asarray(mean_pla_cha, dtype=float)
    var_pla_cha = np.asarray(var_pla_cha, dtype=float)
    var_dQ_c_arr = pd.to_numeric(df["var_dQ_c"], errors="coerce").to_numpy(dtype=float) if "var_dQ_c" in df.columns else None

    if ("reference" in cell_name.lower()) and REF_IGNORE_FIRST_WEEK:
        ref_mask = np.isfinite(time_array) & (time_array > float(REF_IGNORE_WEEKS_LE))
        if int(np.sum(ref_mask)) < 2:
            print(f"[WARN] {cell_name}: <2 points after removing first-week data; skipping reference cell.")
            continue

        time_array = time_array[ref_mask]
        throughput_arr = throughput_arr[ref_mask]
        capacity_arr = capacity_arr[ref_mask]
        mean_low_cha = mean_low_cha[ref_mask]
        var_low_cha = var_low_cha[ref_mask]
        mean_mid_cha = mean_mid_cha[ref_mask]
        var_mid_cha = var_mid_cha[ref_mask]
        mean_high_cha = mean_high_cha[ref_mask]
        var_high_cha = var_high_cha[ref_mask]
        mean_pla_cha = mean_pla_cha[ref_mask]
        var_pla_cha = var_pla_cha[ref_mask]
        if var_dQ_c_arr is not None:
            var_dQ_c_arr = var_dQ_c_arr[ref_mask]

        # Start "real" time/throughput from the first point after week 1.
        time_array = time_array - time_array[0]
        throughput_arr = throughput_arr - throughput_arr[0]

    capacity = pd.Series(capacity_arr)
    SOH = capacity / capacity.iloc[0]
    throughput = np.asarray(throughput_arr, dtype=float)

    if ("throughput_sum" in df.columns) and ("var_dQ_c" in df.columns):
        var_mid_cha_norm = np.abs(np.array(var_mid_cha, dtype=float)) / float(capacity.iloc[0])
        traj_by_cell_reg[cell_name] = pd.DataFrame({
            "weeks": np.asarray(time_array, dtype=float),
            "throughput_cum": throughput,
            "var_dQ_c": np.asarray(var_dQ_c_arr, dtype=float),
            "var_mid_cha": var_mid_cha_norm,
            "capacity": capacity,
        })

    feat_dict = {
        "mean_low_cha": mean_low_cha,
        "var_low_cha": var_low_cha,
        "mean_mid_cha": mean_mid_cha,
        "var_mid_cha": np.abs(var_mid_cha),
        "mean_high_cha": mean_high_cha,
        "var_high_cha": var_high_cha,
        "mean_pla_cha": mean_pla_cha,
        "var_pla_cha": var_pla_cha,
        "capacity": np.array(capacity, dtype=float),
        "throughput": throughput,
        "Time": np.array(time_array, dtype=float),
        "SOH_raw": np.array(SOH, dtype=float),
    }

    for key in list(feat_dict.keys()):
        if key not in ("SOH", "SOH_raw", "capacity", "throughput", "Time"):
            feat_dict[key] = feat_dict[key] / capacity.iloc[0]

    if is_ref:
        results_ref[cell_name] = feat_dict
    else:
        results[cell_name] = feat_dict

    print(f"{'[REF] ' if is_ref else '[TRAIN]'} {cell_name}: done")


# =============================================================================
# STEP 2: BUILD FEATURE VECTORS AT SOH=0.995 AND FIND CLOSEST/FARTHEST
# =============================================================================
ref_feat_at_soh = {}
for cn, fd in results_ref.items():
    f = features_at_soh(fd, TARGET_SOH_FEATURES, method=INTERP_METHOD)
    f_var_mid = features_at_soh(fd, TARGET_SOH_FEATURES, method=INTERP_METHOD, feature_names=["var_mid_cha"])
    if f is not None and f_var_mid is not None:
        f.update(f_var_mid)
    if f is not None:
        ref_feat_at_soh[cn] = f
        print(f"[REF  @ SOH={TARGET_SOH_FEATURES}] {cn}: {f}")

exp_feat_at_soh = {}
for cn, fd in results.items():
    f = features_at_soh(fd, TARGET_SOH_FEATURES, method=INTERP_METHOD)
    f_var_mid = features_at_soh(fd, TARGET_SOH_FEATURES, method=INTERP_METHOD, feature_names=["var_mid_cha"])
    if f is not None and f_var_mid is not None:
        f.update(f_var_mid)
    if f is not None:
        exp_feat_at_soh[cn] = f

print(f"\nRef cells with features: {len(ref_feat_at_soh)}")
print(f"Exp cells with features: {len(exp_feat_at_soh)}")

closest_cellnames = []
farthest_cellnames = []

if len(ref_feat_at_soh) > 0 and len(exp_feat_at_soh) > 0:
    ref_names_list = list(ref_feat_at_soh.keys())
    exp_names_list = list(exp_feat_at_soh.keys())

    ref_mat = np.array([[ref_feat_at_soh[cn][f] for f in FEATURE_NAMES_FOR_DIST] for cn in ref_names_list])
    exp_mat = np.array([[exp_feat_at_soh[cn][f] for f in FEATURE_NAMES_FOR_DIST] for cn in exp_names_list])

    # IQR scaling
    all_mat = np.vstack([ref_mat, exp_mat])
    q25 = np.quantile(all_mat, 0.25, axis=0)
    q75 = np.quantile(all_mat, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-12)

    ref_n = ref_mat / scale
    exp_n = exp_mat / scale

    dists = np.linalg.norm(exp_n[:, None, :] - ref_n[None, :, :], axis=2)
    dist_min = np.min(dists, axis=1)

    k2 = min(K_FARTHEST, len(exp_names_list))
    farthest_idx = np.argsort(dist_min)[-k2:]
    farthest_cellnames = [exp_names_list[i] for i in farthest_idx]

    closest_df = pd.DataFrame(
        {
            "cell_name": exp_names_list,
            "temp": [exp_conditions.get(cn, {}).get("temp", np.nan) for cn in exp_names_list],
            "dist_feat": dist_min.astype(float),
        }
    )
    closest_df = select_lowest_distance_per_temp(
        closest_df,
        temps=TOP_TEMPS,
        n_per_temp=TOP_PER_TEMP,
    )
    closest_cellnames = closest_df["cell_name"].tolist()

    print(f"\nClosest {TOP_PER_TEMP} cells per temperature by feature distance:")
    for temp in TOP_TEMPS:
        temp_rows = closest_df[closest_df["temp"] == float(temp)]
        print(f"  Temp {temp}:")
        for _, row in temp_rows.iterrows():
            print(f"    {row['cell_name']:36s}  dist={row['dist_feat']:.4f}")

    print(f"\nFarthest {k2} cells:")
    for i in farthest_idx:
        print(f"  {exp_names_list[i]:40s}  dist={dist_min[i]:.4f}")


# =============================================================================
# STEP 3: INTERPOLATE TRAJECTORIES + FEATURES AT SOH=0.98
# =============================================================================
closest_set = set(closest_cellnames)
farthest_set = set(farthest_cellnames)
ref_set = set(REF_NAMES)

all_results = {**results_ref, **results}
interp_data = {}

for cell_name, fd in all_results.items():
    cap = np.array(fd["capacity"], dtype=float)
    time_arr = np.array(fd["throughput"], dtype=float)
    weeks_arr = np.array(fd.get("Time", []), dtype=float)

    if len(cap) < 3 or cap[0] == 0:
        continue

    soh = cap / cap[0]

    # find time where SOH = TARGET_SOH_PLOT
    if np.nanmin(soh) > TARGET_SOH_PLOT:
        t_target = time_arr[-1]
    else:
        t_target = interp_value_at_target(
            x=soh[::-1],
            y=time_arr[::-1],
            x_target=TARGET_SOH_PLOT,
            method=INTERP_METHOD
        )
        if t_target is None:
            continue

    # 5 points up to t_target, then continue with same spacing
    grid_dense = np.linspace(0, t_target, 5)
    spacing = float(np.mean(np.diff(grid_dense))) if len(grid_dense) > 1 else 0
    if spacing <= 0:
        continue

    grid_extra = []
    cur = t_target
    while cur + spacing <= time_arr[-1]:
        cur += spacing
        grid_extra.append(cur)

    t_grid = np.concatenate([grid_dense, np.array(grid_extra, dtype=float)])

    # interpolate SOH and capacity
    soh_interp = interp_1d(time_arr, soh, t_grid, method=INTERP_METHOD)
    cap_interp = interp_1d(time_arr, cap, t_grid, method=INTERP_METHOD)
    weeks_interp = None
    if len(weeks_arr) >= 2:
        n_tw = min(len(time_arr), len(weeks_arr))
        weeks_interp = interp_1d(time_arr[:n_tw], weeks_arr[:n_tw], t_grid, method=INTERP_METHOD)

    if soh_interp is None or cap_interp is None:
        continue

    cell_interp = {
        "time": t_grid,
        "SOH": soh_interp,
        "capacity": cap_interp,
    }
    if weeks_interp is not None:
        cell_interp["weeks"] = weeks_interp

    # interpolate all features onto the same grid
    for feat_name in INTERP_FEATURES:
        feat_vals = fd.get(feat_name, [])
        if len(feat_vals) == 0:
            continue

        feat_vals = np.array(feat_vals, dtype=float)
        n = min(len(time_arr), len(feat_vals))
        if n < 2:
            continue

        y_new = interp_1d(time_arr[:n], feat_vals[:n], t_grid, method=INTERP_METHOD)
        if y_new is None:
            continue

        cell_interp[feat_name] = y_new

    interp_data[cell_name] = cell_interp

def _group_for_cell(cn):
    if cn in ref_set:
        return "ref"
    if cn in closest_set:
        return "closest"
    if cn in farthest_set:
        return "farthest"
    return "other"


def _plot_soh_trajectories(interp_data_local, x_mode="index"):
    style_map_local = {
        "ref": {"color": "red", "lw": 1.8, "alpha": 0.95, "label": "refs", "z": 4},
        "closest": {"color": "orange", "lw": 1.2, "alpha": 0.85, "label": f"closest {K_CLOSEST}", "z": 3},
        "farthest": {"color": "green", "lw": 1.2, "alpha": 0.85, "label": f"farthest {K_FARTHEST}", "z": 2},
        "other": {"color": "blue", "lw": 0.6, "alpha": 0.20, "label": "other", "z": 1},
    }
    draw_order = ["other", "closest", "farthest", "ref"]  # ref last -> always on top
    shown_label = {k: False for k in style_map_local.keys()}

    fig_local, ax_local = plt.subplots(figsize=(10, 6))
    for grp in draw_order:
        for cn, cdict in interp_data_local.items():
            if _group_for_cell(cn) != grp:
                continue
            y = np.asarray(cdict["SOH"], dtype=float)
            if x_mode == "throughput":
                x = np.asarray(cdict["time"], dtype=float)
                xlabel = "Throughput"
                fname_suffix = "throughput"
            elif x_mode == "weeks":
                if "weeks" not in cdict:
                    continue
                x = np.asarray(cdict["weeks"], dtype=float)
                xlabel = "Weeks"
                fname_suffix = "weeks"
            else:
                x = np.arange(len(y), dtype=float)
                xlabel = "Step"
                fname_suffix = "index"
            st = style_map_local[grp]
            label = st["label"] if not shown_label[grp] else None
            shown_label[grp] = True
            ax_local.plot(
                x,
                y,
                color=st["color"],
                linewidth=st["lw"],
                alpha=st["alpha"],
                label=label,
                zorder=st["z"],
            )

    ax_local.set_xlabel(xlabel)
    ax_local.set_ylabel("SOH")
    ax_local.set_title(
        f"SOH trajectories vs {xlabel} ({INTERP_METHOD} interp @ {TARGET_SOH_PLOT}, "
        f"features @ {TARGET_SOH_FEATURES})"
    )
    ax_local.set_ylim(0.8, 1.05)
    ax_local.grid(True, alpha=0.3)
    ax_local.legend(loc="best")
    fig_local.tight_layout()
    fig_local.savefig(out_fig_dir / f"SOH_vs_{fname_suffix}_closest_farthest_{INTERP_METHOD}.png", dpi=150)
    plt.show()


_plot_soh_trajectories(interp_data, x_mode="index")
_plot_soh_trajectories(interp_data, x_mode="throughput")
_plot_soh_trajectories(interp_data, x_mode="weeks")

print(f"\nInterpolated {len(interp_data)} cells with features: {INTERP_FEATURES}")


# =============================================================================
# STEP 4: FIND INTERPOLATED STEP TO TARGET SOH
# =============================================================================
for cell_name, cell_dict in interp_data.items():
    soh = np.asarray(cell_dict["SOH"], dtype=float)

    if len(soh) < 2 or not np.all(np.isfinite(soh)):
        cell_dict["step_to_target"] = None
        continue

    # step axis on interpolated grid
    step_axis = np.arange(len(soh), dtype=float)

    # interpolate/extrapolate fractional step where SOH reaches target
    step_to_target = interp_value_with_optional_extrapolation(
        x=soh[::-1],          # SOH ascending after reverse
        y=step_axis[::-1],    # corresponding step positions
        x_target=SOH_STEP_TARGET,
        method=INTERP_METHOD,
        allow_extrapolation=EXTRAPOLATE_SOH_TARGET_IF_NOT_REACHED,
    )

    cell_dict["step_to_target"] = step_to_target


# =============================================================================
# STEP 5: PLOT ALL INTERPOLATED FEATURES
# =============================================================================
PLOT_FEATURES = [
    "SOH",
    "capacity",
    "mean_low_cha", "mean_mid_cha", "mean_high_cha", "mean_pla_cha",
    "var_low_cha", "var_mid_cha", "var_high_cha", "var_pla_cha",
]

plot_features_existing = [
    f for f in PLOT_FEATURES
    if any(f in cell_dict for cell_dict in interp_data.values())
]

n_feat = len(plot_features_existing)
n_cols = 3
n_rows = math.ceil(n_feat / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.0 * n_rows))
axes = np.array(axes).reshape(-1)

style_map = {
    "ref": {"color": "red", "lw": 1.8, "alpha": 0.95},
    "closest": {"color": "orange", "lw": 1.2, "alpha": 0.85},
    "farthest": {"color": "green", "lw": 1.2, "alpha": 0.85},
    "other": {"color": "blue", "lw": 0.6, "alpha": 0.20},
}

for ax, feat_name in zip(axes, plot_features_existing):
    for cell_name, cell_dict in interp_data.items():
        if feat_name not in cell_dict:
            continue

        y = np.asarray(cell_dict[feat_name], dtype=float)
        x = np.arange(len(y))

        if cell_name in ref_set:
            st = style_map["ref"]
        elif cell_name in closest_set:
            st = style_map["closest"]
        elif cell_name in farthest_set:
            st = style_map["farthest"]
        else:
            st = style_map["other"]

        ax.plot(x, y, color=st["color"], linewidth=st["lw"], alpha=st["alpha"])

    ax.set_title(feat_name)
    ax.set_xlabel("Interpolated step")
    ax.set_ylabel(feat_name)
    ax.grid(True, alpha=0.3)

for ax in axes[n_feat:]:
    ax.axis("off")

legend_handles = [
    Line2D([0], [0], color="red", lw=1.8, alpha=0.95, label="refs"),
    Line2D([0], [0], color="orange", lw=1.2, alpha=0.85, label=f"closest {K_CLOSEST}"),
    Line2D([0], [0], color="green", lw=1.2, alpha=0.85, label=f"farthest {K_FARTHEST}"),
    Line2D([0], [0], color="blue", lw=1.0, alpha=0.50, label="other"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=True)

fig.suptitle(
    f"Interpolated feature trajectories\n"
    f"({INTERP_METHOD} grid anchored at SOH={TARGET_SOH_PLOT}, "
    f"closest/farthest based on features at SOH={TARGET_SOH_FEATURES})",
    y=0.995
)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out_fig_dir / f"all_interpolated_features_subplots_{INTERP_METHOD}.png", dpi=150)
plt.show()

# Single plot: distribution of initial capacity (histogram + density curve)
initial_capacities = []
for cell_name, fd in all_results.items():
    cap = np.asarray(fd.get("capacity", []), dtype=float)
    if len(cap) == 0 or not np.isfinite(cap[0]):
        continue
    if float(cap[0]) < 1.0:
        continue
    initial_capacities.append(float(cap[0]))

if len(initial_capacities) > 0:
    init_caps = np.asarray(initial_capacities, dtype=float)

    fig_capdist, ax_capdist = plt.subplots(figsize=(8, 5))
    ax_capdist.hist(
        init_caps,
        bins=INITIAL_CAP_HIST_BINS,
        density=True,
        alpha=0.55,
        color="steelblue",
        edgecolor="white",
        label=f"Histogram (n={len(init_caps)})",
    )

    if len(init_caps) >= 2 and float(np.nanstd(init_caps)) > 0.0:
        x_kde = np.linspace(float(np.nanmin(init_caps)), float(np.nanmax(init_caps)), 300)
        kde = gaussian_kde(init_caps)
        y_kde = kde(x_kde)
        ax_capdist.plot(x_kde, y_kde, color="darkorange", linewidth=2.2, label="Density curve (KDE)")

    ax_capdist.set_xlabel("Initial capacity")
    ax_capdist.set_ylabel("Density")
    ax_capdist.set_title("Distribution of initial capacity")
    ax_capdist.grid(True, alpha=0.3)
    ax_capdist.legend(loc="best")
    fig_capdist.tight_layout()
    fig_capdist.savefig(out_fig_dir / "initial_capacity_distribution.png", dpi=150)
    plt.show()
else:
    print("[WARN] No valid initial capacity values found for distribution plot.")


# =============================================================================
# STEP 5B: LOG-LOG RELATIONSHIPS (FEATURES vs STEPS TO TARGET SOH)
# =============================================================================
exclude_keys = {"time", "SOH", "capacity", "step_to_target"}
candidate_loglog_features = set()
for _, cell_dict in interp_data.items():
    for key, val in cell_dict.items():
        if key in exclude_keys:
            continue
        if isinstance(val, (list, np.ndarray)):
            candidate_loglog_features.add(key)

candidate_loglog_features = sorted(candidate_loglog_features)

if len(candidate_loglog_features) == 0:
    print("\n[WARN] No interpolated features found for log-log plots.")
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

    fig_loggrid_closest, axes_loggrid_closest = plt.subplots(
        nrows, ncols,
        figsize=(5.8 * ncols, 4.5 * nrows),
        sharex=False,
        sharey=False
    )
    axes_loggrid_closest = np.array(axes_loggrid_closest).ravel()

    fig_loggrid_farthest, axes_loggrid_farthest = plt.subplots(
        nrows, ncols,
        figsize=(5.8 * ncols, 4.5 * nrows),
        sharex=False,
        sharey=False
    )
    axes_loggrid_farthest = np.array(axes_loggrid_farthest).ravel()

    loglog_summary_rows = []
    loglog_summary_rows_closest = []
    loglog_summary_rows_farthest = []

    for i, feat_name in enumerate(candidate_loglog_features):
        ax_ll = axes_loggrid[i]
        ax_ll_closest = axes_loggrid_closest[i]
        ax_ll_farthest = axes_loggrid_farthest[i]

        log_feat_arr = []
        log_steps_arr = []
        color_arr = []

        for cell_name, cell_dict in interp_data.items():
            if feat_name not in cell_dict:
                continue

            step_to_target = cell_dict.get("step_to_target")
            if step_to_target is None or not np.isfinite(step_to_target) or step_to_target <= 0:
                continue

            feat_vals = np.asarray(cell_dict[feat_name], dtype=float)
            if len(feat_vals) <= FEATURE_STEP_LOGLOG:
                continue

            fv = feat_vals[FEATURE_STEP_LOGLOG]
            if not np.isfinite(fv) or fv == 0.0:
                continue

            log_feat_arr.append(np.log(np.abs(fv)))
            log_steps_arr.append(np.log(step_to_target))

            if cell_name in ref_set:
                color_arr.append("red")
            elif cell_name in closest_set:
                color_arr.append("orange")
            elif cell_name in farthest_set:
                color_arr.append("green")
            else:
                color_arr.append("blue")

        log_feat_arr = np.array(log_feat_arr, dtype=float)
        log_steps_arr = np.array(log_steps_arr, dtype=float)
        color_arr = np.array(color_arr, dtype=str)

        ref_names_feat = []
        ref_log_x_feat = []
        ref_log_true_steps_feat = []
        ref_true_steps_feat = []
        for ref_name in REF_NAMES:
            ref_cell = interp_data.get(ref_name)
            if ref_cell is None or feat_name not in ref_cell:
                continue
            step_ref = ref_cell.get("step_to_target")
            if step_ref is None or not np.isfinite(step_ref) or step_ref <= 0:
                continue
            feat_vals_ref = np.asarray(ref_cell[feat_name], dtype=float)
            if len(feat_vals_ref) <= FEATURE_STEP_LOGLOG:
                continue
            fv_ref = feat_vals_ref[FEATURE_STEP_LOGLOG]
            if not np.isfinite(fv_ref) or fv_ref == 0.0:
                continue
            ref_names_feat.append(ref_name)
            ref_log_x_feat.append(np.log(np.abs(fv_ref)))
            ref_log_true_steps_feat.append(np.log(float(step_ref)))
            ref_true_steps_feat.append(float(step_ref))
        ref_log_x_feat = np.array(ref_log_x_feat, dtype=float)
        ref_log_true_steps_feat = np.array(ref_log_true_steps_feat, dtype=float)
        ref_true_steps_feat = np.array(ref_true_steps_feat, dtype=float)
        if len(ref_log_x_feat) > 0:
            ax_ll_closest.scatter(
                ref_log_x_feat,
                ref_log_true_steps_feat,
                c="red",
                marker="*",
                s=110,
                alpha=0.9,
                edgecolors="k",
                linewidths=0.35,
                label="refs true",
            )
            ax_ll_farthest.scatter(
                ref_log_x_feat,
                ref_log_true_steps_feat,
                c="red",
                marker="*",
                s=110,
                alpha=0.9,
                edgecolors="k",
                linewidths=0.35,
                label="refs true",
            )

        if len(log_feat_arr) < 2:
            ax_ll.set_title(f"{feat_name}\ninsufficient data")
            ax_ll.grid(True, alpha=0.3)
            loglog_summary_rows.append({
                "feature": feat_name,
                "n": len(log_feat_arr),
                "slope": np.nan,
                "intercept": np.nan,
                "corr_r": np.nan,
                "mape_ref_steps": np.nan,
            })

            ax_ll_closest.set_title(f"{feat_name}\nclosest: insufficient data")
            ax_ll_closest.grid(True, alpha=0.3)
            loglog_summary_rows_closest.append({
                "feature": feat_name,
                "n": 0,
                "slope": np.nan,
                "intercept": np.nan,
                "corr_r": np.nan,
                "mape_ref_steps": np.nan,
            })

            ax_ll_farthest.set_title(f"{feat_name}\nfarthest: insufficient data")
            ax_ll_farthest.grid(True, alpha=0.3)
            loglog_summary_rows_farthest.append({
                "feature": feat_name,
                "n": 0,
                "slope": np.nan,
                "intercept": np.nan,
                "corr_r": np.nan,
                "mape_ref_steps": np.nan,
            })
            continue

        label_map_ll = {"blue": "other", "orange": "closest", "green": "farthest", "red": "refs"}
        for cv in ["blue", "orange", "green", "red"]:
            m = color_arr == cv
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
                label=label_map_ll[cv],
            )

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
        _, mape_ref_steps_all = mape_on_references_from_log_fit(
            coeffs,
            ref_log_x_feat,
            ref_log_true_steps_feat,
            ref_true_steps_feat,
        )

        mape_ref_steps_all_txt = f"{mape_ref_steps_all:.1f}" if np.isfinite(mape_ref_steps_all) else "nan"
        ax_ll.set_title(
            f"{feat_name}\n"
            f"n={len(log_feat_arr)}, r={r:.3f}, MAPE_ref={mape_ref_steps_all_txt}%"
        )
        ax_ll.set_xlabel(f"log(|{feat_name}|) @ step {FEATURE_STEP_LOGLOG}")
        ax_ll.set_ylabel(f"log(steps to SOH <= {SOH_STEP_TARGET})")
        ax_ll.grid(True, alpha=0.3)
        ax_ll.legend(loc="best", fontsize=8)

        loglog_summary_rows.append({
            "feature": feat_name,
            "n": len(log_feat_arr),
            "slope": slope,
            "intercept": intercept,
            "corr_r": r,
            "mape_ref_steps": mape_ref_steps_all,
        })

        # additional fit: closest only
        m_closest = color_arr == "orange"
        n_closest = int(np.sum(m_closest))
        if n_closest >= 2:
            x_closest = log_feat_arr[m_closest]
            y_closest = log_steps_arr[m_closest]
            ax_ll_closest.scatter(
                x_closest,
                y_closest,
                c="orange",
                s=35,
                alpha=0.8,
                edgecolors="k",
                linewidths=0.35,
                label="closest",
            )
            coeffs_c = np.polyfit(x_closest, y_closest, 1)
            slope_c = float(coeffs_c[0])
            intercept_c = float(coeffs_c[1])
            xfit_c = np.linspace(x_closest.min(), x_closest.max(), 200)
            yfit_c = np.polyval(coeffs_c, xfit_c)
            ax_ll_closest.plot(
                xfit_c, yfit_c,
                "k--", lw=1.3,
                label=f"slope={slope_c:.3f}, int={intercept_c:.3f}"
            )
            r_c = float(np.corrcoef(x_closest, y_closest)[0, 1])
            _, mape_ref_steps_c = mape_on_references_from_log_fit(
                coeffs_c,
                ref_log_x_feat,
                ref_log_true_steps_feat,
                ref_true_steps_feat,
            )
            mape_ref_steps_c_txt = f"{mape_ref_steps_c:.1f}" if np.isfinite(mape_ref_steps_c) else "nan"
            ax_ll_closest.set_title(
                f"{feat_name}\n"
                f"closest n={n_closest}, r={r_c:.3f}, MAPE_ref={mape_ref_steps_c_txt}%"
            )
            loglog_summary_rows_closest.append({
                "feature": feat_name,
                "n": n_closest,
                "slope": slope_c,
                "intercept": intercept_c,
                "corr_r": r_c,
                "mape_ref_steps": mape_ref_steps_c,
            })
            ax_ll_closest.legend(loc="best", fontsize=8)
        else:
            ax_ll_closest.set_title(f"{feat_name}\nclosest: insufficient data")
            loglog_summary_rows_closest.append({
                "feature": feat_name,
                "n": n_closest,
                "slope": np.nan,
                "intercept": np.nan,
                "corr_r": np.nan,
                "mape_ref_steps": np.nan,
            })
        ax_ll_closest.set_xlabel(f"log(|{feat_name}|) @ step {FEATURE_STEP_LOGLOG}")
        ax_ll_closest.set_ylabel(f"log(steps to SOH <= {SOH_STEP_TARGET})")
        ax_ll_closest.grid(True, alpha=0.3)

        # additional fit: farthest only
        m_farthest = color_arr == "green"
        n_farthest = int(np.sum(m_farthest))
        if n_farthest >= 2:
            x_farthest = log_feat_arr[m_farthest]
            y_farthest = log_steps_arr[m_farthest]
            ax_ll_farthest.scatter(
                x_farthest,
                y_farthest,
                c="green",
                s=35,
                alpha=0.8,
                edgecolors="k",
                linewidths=0.35,
                label="farthest",
            )
            coeffs_f = np.polyfit(x_farthest, y_farthest, 1)
            slope_f = float(coeffs_f[0])
            intercept_f = float(coeffs_f[1])
            xfit_f = np.linspace(x_farthest.min(), x_farthest.max(), 200)
            yfit_f = np.polyval(coeffs_f, xfit_f)
            ax_ll_farthest.plot(
                xfit_f, yfit_f,
                "k--", lw=1.3,
                label=f"slope={slope_f:.3f}, int={intercept_f:.3f}"
            )
            r_f = float(np.corrcoef(x_farthest, y_farthest)[0, 1])
            _, mape_ref_steps_f = mape_on_references_from_log_fit(
                coeffs_f,
                ref_log_x_feat,
                ref_log_true_steps_feat,
                ref_true_steps_feat,
            )
            mape_ref_steps_f_txt = f"{mape_ref_steps_f:.1f}" if np.isfinite(mape_ref_steps_f) else "nan"
            ax_ll_farthest.set_title(
                f"{feat_name}\n"
                f"farthest n={n_farthest}, r={r_f:.3f}, MAPE_ref={mape_ref_steps_f_txt}%"
            )
            loglog_summary_rows_farthest.append({
                "feature": feat_name,
                "n": n_farthest,
                "slope": slope_f,
                "intercept": intercept_f,
                "corr_r": r_f,
                "mape_ref_steps": mape_ref_steps_f,
            })
            ax_ll_farthest.legend(loc="best", fontsize=8)
        else:
            ax_ll_farthest.set_title(f"{feat_name}\nfarthest: insufficient data")
            loglog_summary_rows_farthest.append({
                "feature": feat_name,
                "n": n_farthest,
                "slope": np.nan,
                "intercept": np.nan,
                "corr_r": np.nan,
                "mape_ref_steps": np.nan,
            })
        ax_ll_farthest.set_xlabel(f"log(|{feat_name}|) @ step {FEATURE_STEP_LOGLOG}")
        ax_ll_farthest.set_ylabel(f"log(steps to SOH <= {SOH_STEP_TARGET})")
        ax_ll_farthest.grid(True, alpha=0.3)

    for j in range(len(candidate_loglog_features), len(axes_loggrid)):
        axes_loggrid[j].axis("off")
        axes_loggrid_closest[j].axis("off")
        axes_loggrid_farthest[j].axis("off")

    fig_loggrid.suptitle(
        f"Log-log relationships for all interpolated features\n"
        f"x = log(|feature at step {FEATURE_STEP_LOGLOG}|), "
        f"y = log(steps to SOH <= {SOH_STEP_TARGET})\n"
        f"(cells not reaching target excluded)",
        y=0.995
    )
    fig_loggrid.tight_layout(rect=[0, 0, 1, 0.96])
    fig_loggrid.savefig(out_fig_dir / "loglog_all_features.png", dpi=150)

    fig_loggrid_closest.suptitle(
        f"Log-log relationships for closest cells only\n"
        f"x = log(|feature at step {FEATURE_STEP_LOGLOG}|), "
        f"y = log(steps to SOH <= {SOH_STEP_TARGET})",
        y=0.995
    )
    fig_loggrid_closest.tight_layout(rect=[0, 0, 1, 0.96])
    fig_loggrid_closest.savefig(out_fig_dir / "loglog_closest_features.png", dpi=150)

    fig_loggrid_farthest.suptitle(
        f"Log-log relationships for farthest cells only\n"
        f"x = log(|feature at step {FEATURE_STEP_LOGLOG}|), "
        f"y = log(steps to SOH <= {SOH_STEP_TARGET})",
        y=0.995
    )
    fig_loggrid_farthest.tight_layout(rect=[0, 0, 1, 0.96])
    fig_loggrid_farthest.savefig(out_fig_dir / "loglog_farthest_features.png", dpi=150)

    df_loglog_summary = pd.DataFrame(loglog_summary_rows).sort_values(
        by="corr_r", ascending=False, na_position="last"
    )
    df_loglog_summary.to_csv(out_fig_dir / "loglog_summary_all_features.csv", index=False)
    df_loglog_summary_closest = pd.DataFrame(loglog_summary_rows_closest).sort_values(
        by="corr_r", ascending=False, na_position="last"
    )
    df_loglog_summary_closest.to_csv(out_fig_dir / "loglog_summary_closest_features.csv", index=False)
    df_loglog_summary_farthest = pd.DataFrame(loglog_summary_rows_farthest).sort_values(
        by="corr_r", ascending=False, na_position="last"
    )
    df_loglog_summary_farthest.to_csv(out_fig_dir / "loglog_summary_farthest_features.csv", index=False)
    print("\nSaved log-log summary:")
    print(out_fig_dir / "loglog_summary_all_features.csv")
    print(df_loglog_summary.to_string(index=False))
    print(out_fig_dir / "loglog_summary_closest_features.csv")
    print(df_loglog_summary_closest.to_string(index=False))
    print(out_fig_dir / "loglog_summary_farthest_features.csv")
    print(df_loglog_summary_farthest.to_string(index=False))
    plt.show()


# =============================================================================
# STEP 5C: LOG-LOG @ THROUGHPUT=500k (FEATURE) VS THROUGHPUT @ SOH=0.955
# =============================================================================
if len(candidate_loglog_features) == 0:
    print("\n[WARN] No interpolated features found for throughput-based log-log plots.")
else:
    n_feat_thr = len(candidate_loglog_features)
    ncols_thr = 3
    nrows_thr = math.ceil(n_feat_thr / ncols_thr)

    fig_thr_all, axes_thr_all = plt.subplots(
        nrows_thr, ncols_thr,
        figsize=(5.8 * ncols_thr, 4.5 * nrows_thr),
        sharex=False,
        sharey=False
    )
    axes_thr_all = np.array(axes_thr_all).ravel()

    fig_thr_closest, axes_thr_closest = plt.subplots(
        nrows_thr, ncols_thr,
        figsize=(5.8 * ncols_thr, 4.5 * nrows_thr),
        sharex=False,
        sharey=False
    )
    axes_thr_closest = np.array(axes_thr_closest).ravel()

    fig_thr_farthest, axes_thr_farthest = plt.subplots(
        nrows_thr, ncols_thr,
        figsize=(5.8 * ncols_thr, 4.5 * nrows_thr),
        sharex=False,
        sharey=False
    )
    axes_thr_farthest = np.array(axes_thr_farthest).ravel()

    summary_thr_all = []
    summary_thr_closest = []
    summary_thr_farthest = []

    for i, feat_name in enumerate(candidate_loglog_features):
        ax_all = axes_thr_all[i]
        ax_close = axes_thr_closest[i]
        ax_far = axes_thr_farthest[i]

        log_x = []
        log_y = []
        color_arr = []

        ref_log_x = []
        ref_log_y_true = []
        ref_true_y = []

        for cell_name, cell_dict in interp_data.items():
            if feat_name not in cell_dict or "time" not in cell_dict or "SOH" not in cell_dict:
                continue

            t_arr = np.asarray(cell_dict["time"], dtype=float)
            soh_arr = np.asarray(cell_dict["SOH"], dtype=float)
            f_arr = np.asarray(cell_dict[feat_name], dtype=float)

            n = min(len(t_arr), len(soh_arr), len(f_arr))
            if n < 2:
                continue

            t_arr = t_arr[:n]
            soh_arr = soh_arr[:n]
            f_arr = f_arr[:n]

            if THROUGHPUT_FEATURE_LOGLOG < np.nanmin(t_arr) or THROUGHPUT_FEATURE_LOGLOG > np.nanmax(t_arr):
                continue

            f_at_thr = interp_value_at_target(
                x=t_arr,
                y=f_arr,
                x_target=THROUGHPUT_FEATURE_LOGLOG,
                method="linear",
            )
            if f_at_thr is None or (not np.isfinite(f_at_thr)) or f_at_thr == 0.0:
                continue

            thr_to_soh_target = interp_value_with_optional_extrapolation(
                x=soh_arr[::-1],
                y=t_arr[::-1],
                x_target=SOH_THROUGHPUT_TARGET,
                method=INTERP_METHOD,
                allow_extrapolation=EXTRAPOLATE_SOH_TARGET_IF_NOT_REACHED,
            )
            if thr_to_soh_target is None or (not np.isfinite(thr_to_soh_target)) or thr_to_soh_target <= 0.0:
                continue

            lx = np.log(np.abs(f_at_thr))
            ly = np.log(thr_to_soh_target)
            if not np.isfinite(lx) or not np.isfinite(ly):
                continue

            log_x.append(lx)
            log_y.append(ly)

            if cell_name in ref_set:
                color_arr.append("red")
                ref_log_x.append(lx)
                ref_log_y_true.append(ly)
                ref_true_y.append(thr_to_soh_target)
            elif cell_name in closest_set:
                color_arr.append("orange")
            elif cell_name in farthest_set:
                color_arr.append("green")
            else:
                color_arr.append("blue")

        log_x = np.array(log_x, dtype=float)
        log_y = np.array(log_y, dtype=float)
        color_arr = np.array(color_arr, dtype=str)
        ref_log_x = np.array(ref_log_x, dtype=float)
        ref_log_y_true = np.array(ref_log_y_true, dtype=float)
        ref_true_y = np.array(ref_true_y, dtype=float)

        if len(log_x) < 2:
            ax_all.set_title(f"{feat_name}\ninsufficient data")
            ax_all.grid(True, alpha=0.3)
            summary_thr_all.append({
                "feature": feat_name,
                "n": len(log_x),
                "slope": np.nan,
                "intercept": np.nan,
                "corr_r": np.nan,
                "mape_ref_thr": np.nan,
            })
        else:
            label_map = {"blue": "other", "orange": "closest", "green": "farthest", "red": "refs"}
            for cv in ["blue", "orange", "green", "red"]:
                m = color_arr == cv
                if not m.any():
                    continue
                ax_all.scatter(
                    log_x[m], log_y[m],
                    c=cv, s=35, alpha=0.75,
                    edgecolors="k", linewidths=0.35,
                    label=label_map[cv],
                )

            coeffs_all = np.polyfit(log_x, log_y, 1)
            slope_all = float(coeffs_all[0])
            intercept_all = float(coeffs_all[1])
            xfit_all = np.linspace(log_x.min(), log_x.max(), 200)
            yfit_all = np.polyval(coeffs_all, xfit_all)
            ax_all.plot(xfit_all, yfit_all, "k--", lw=1.3, label=f"slope={slope_all:.3f}, int={intercept_all:.3f}")
            r_all = float(np.corrcoef(log_x, log_y)[0, 1])
            _, mape_ref_thr_all = mape_on_references_from_log_fit(
                coeffs_all,
                ref_log_x,
                ref_log_y_true,
                ref_true_y,
            )
            mape_ref_thr_all_txt = f"{mape_ref_thr_all:.1f}" if np.isfinite(mape_ref_thr_all) else "nan"
            ax_all.set_title(
                f"{feat_name}\n"
                f"n={len(log_x)}, r={r_all:.3f}, MAPE_ref={mape_ref_thr_all_txt}%"
            )
            ax_all.legend(loc="best", fontsize=8)
            summary_thr_all.append({
                "feature": feat_name,
                "n": len(log_x),
                "slope": slope_all,
                "intercept": intercept_all,
                "corr_r": r_all,
                "mape_ref_thr": mape_ref_thr_all,
            })
        ax_all.set_xlabel(f"log(|{feat_name}|) @ throughput {int(THROUGHPUT_FEATURE_LOGLOG)}")
        ax_all.set_ylabel(f"log(throughput to SOH <= {SOH_THROUGHPUT_TARGET})")
        ax_all.grid(True, alpha=0.3)

        if len(ref_log_x) > 0:
            ax_close.scatter(
                ref_log_x, ref_log_y_true,
                c="red", marker="*",
                s=110, alpha=0.9,
                edgecolors="k", linewidths=0.35,
                label="refs true",
            )
            ax_far.scatter(
                ref_log_x, ref_log_y_true,
                c="red", marker="*",
                s=110, alpha=0.9,
                edgecolors="k", linewidths=0.35,
                label="refs true",
            )

        m_close = color_arr == "orange"
        n_close = int(np.sum(m_close))
        if n_close >= 2:
            x_close = log_x[m_close]
            y_close = log_y[m_close]
            ax_close.scatter(
                x_close, y_close,
                c="orange", s=35, alpha=0.8,
                edgecolors="k", linewidths=0.35,
                label="closest",
            )
            coeffs_close = np.polyfit(x_close, y_close, 1)
            slope_close = float(coeffs_close[0])
            intercept_close = float(coeffs_close[1])
            xfit_close = np.linspace(x_close.min(), x_close.max(), 200)
            yfit_close = np.polyval(coeffs_close, xfit_close)
            ax_close.plot(xfit_close, yfit_close, "k--", lw=1.3, label=f"slope={slope_close:.3f}, int={intercept_close:.3f}")
            r_close = float(np.corrcoef(x_close, y_close)[0, 1])
            _, mape_ref_thr_close = mape_on_references_from_log_fit(
                coeffs_close,
                ref_log_x,
                ref_log_y_true,
                ref_true_y,
            )
            mape_ref_thr_close_txt = f"{mape_ref_thr_close:.1f}" if np.isfinite(mape_ref_thr_close) else "nan"
            ax_close.set_title(
                f"{feat_name}\n"
                f"closest n={n_close}, r={r_close:.3f}, MAPE_ref={mape_ref_thr_close_txt}%"
            )
            summary_thr_closest.append({
                "feature": feat_name,
                "n": n_close,
                "slope": slope_close,
                "intercept": intercept_close,
                "corr_r": r_close,
                "mape_ref_thr": mape_ref_thr_close,
            })
            ax_close.legend(loc="best", fontsize=8)
        else:
            ax_close.set_title(f"{feat_name}\nclosest: insufficient data")
            summary_thr_closest.append({
                "feature": feat_name,
                "n": n_close,
                "slope": np.nan,
                "intercept": np.nan,
                "corr_r": np.nan,
                "mape_ref_thr": np.nan,
            })
        ax_close.set_xlabel(f"log(|{feat_name}|) @ throughput {int(THROUGHPUT_FEATURE_LOGLOG)}")
        ax_close.set_ylabel(f"log(throughput to SOH <= {SOH_THROUGHPUT_TARGET})")
        ax_close.grid(True, alpha=0.3)

        m_far = color_arr == "green"
        n_far = int(np.sum(m_far))
        if n_far >= 2:
            x_far = log_x[m_far]
            y_far = log_y[m_far]
            ax_far.scatter(
                x_far, y_far,
                c="green", s=35, alpha=0.8,
                edgecolors="k", linewidths=0.35,
                label="farthest",
            )
            coeffs_far = np.polyfit(x_far, y_far, 1)
            slope_far = float(coeffs_far[0])
            intercept_far = float(coeffs_far[1])
            xfit_far = np.linspace(x_far.min(), x_far.max(), 200)
            yfit_far = np.polyval(coeffs_far, xfit_far)
            ax_far.plot(xfit_far, yfit_far, "k--", lw=1.3, label=f"slope={slope_far:.3f}, int={intercept_far:.3f}")
            r_far = float(np.corrcoef(x_far, y_far)[0, 1])
            _, mape_ref_thr_far = mape_on_references_from_log_fit(
                coeffs_far,
                ref_log_x,
                ref_log_y_true,
                ref_true_y,
            )
            mape_ref_thr_far_txt = f"{mape_ref_thr_far:.1f}" if np.isfinite(mape_ref_thr_far) else "nan"
            ax_far.set_title(
                f"{feat_name}\n"
                f"farthest n={n_far}, r={r_far:.3f}, MAPE_ref={mape_ref_thr_far_txt}%"
            )
            summary_thr_farthest.append({
                "feature": feat_name,
                "n": n_far,
                "slope": slope_far,
                "intercept": intercept_far,
                "corr_r": r_far,
                "mape_ref_thr": mape_ref_thr_far,
            })
            ax_far.legend(loc="best", fontsize=8)
        else:
            ax_far.set_title(f"{feat_name}\nfarthest: insufficient data")
            summary_thr_farthest.append({
                "feature": feat_name,
                "n": n_far,
                "slope": np.nan,
                "intercept": np.nan,
                "corr_r": np.nan,
                "mape_ref_thr": np.nan,
            })
        ax_far.set_xlabel(f"log(|{feat_name}|) @ throughput {int(THROUGHPUT_FEATURE_LOGLOG)}")
        ax_far.set_ylabel(f"log(throughput to SOH <= {SOH_THROUGHPUT_TARGET})")
        ax_far.grid(True, alpha=0.3)

    for j in range(len(candidate_loglog_features), len(axes_thr_all)):
        axes_thr_all[j].axis("off")
        axes_thr_closest[j].axis("off")
        axes_thr_farthest[j].axis("off")

    fig_thr_all.suptitle(
        f"Log-log: feature @ throughput {int(THROUGHPUT_FEATURE_LOGLOG)} vs throughput to SOH={SOH_THROUGHPUT_TARGET}\n"
        f"(raw feature values; cells must reach target SOH)",
        y=0.995
    )
    fig_thr_all.tight_layout(rect=[0, 0, 1, 0.96])
    fig_thr_all.savefig(out_fig_dir / "loglog_thr500k_to_soh0955_all_features.png", dpi=150)

    fig_thr_closest.suptitle(
        f"Log-log: closest cells only, feature @ throughput {int(THROUGHPUT_FEATURE_LOGLOG)} vs throughput to SOH={SOH_THROUGHPUT_TARGET}",
        y=0.995
    )
    fig_thr_closest.tight_layout(rect=[0, 0, 1, 0.96])
    fig_thr_closest.savefig(out_fig_dir / "loglog_thr500k_to_soh0955_closest_features.png", dpi=150)

    fig_thr_farthest.suptitle(
        f"Log-log: farthest cells only, feature @ throughput {int(THROUGHPUT_FEATURE_LOGLOG)} vs throughput to SOH={SOH_THROUGHPUT_TARGET}",
        y=0.995
    )
    fig_thr_farthest.tight_layout(rect=[0, 0, 1, 0.96])
    fig_thr_farthest.savefig(out_fig_dir / "loglog_thr500k_to_soh0955_farthest_features.png", dpi=150)

    df_summary_thr_all = pd.DataFrame(summary_thr_all).sort_values(by="corr_r", ascending=False, na_position="last")
    df_summary_thr_all.to_csv(out_fig_dir / "loglog_thr500k_to_soh0955_summary_all_features.csv", index=False)
    df_summary_thr_closest = pd.DataFrame(summary_thr_closest).sort_values(by="corr_r", ascending=False, na_position="last")
    df_summary_thr_closest.to_csv(out_fig_dir / "loglog_thr500k_to_soh0955_summary_closest_features.csv", index=False)
    df_summary_thr_farthest = pd.DataFrame(summary_thr_farthest).sort_values(by="corr_r", ascending=False, na_position="last")
    df_summary_thr_farthest.to_csv(out_fig_dir / "loglog_thr500k_to_soh0955_summary_farthest_features.csv", index=False)

    print("\nSaved throughput-based log-log summary:")
    print(out_fig_dir / "loglog_thr500k_to_soh0955_summary_all_features.csv")
    print(df_summary_thr_all.to_string(index=False))
    print(out_fig_dir / "loglog_thr500k_to_soh0955_summary_closest_features.csv")
    print(df_summary_thr_closest.to_string(index=False))
    print(out_fig_dir / "loglog_thr500k_to_soh0955_summary_farthest_features.csv")
    print(df_summary_thr_farthest.to_string(index=False))
    plt.show()



# =============================================================================
# STEP 6: XGBOOST + MONTE CARLO (no CNN training)
# =============================================================================
rows_exp, rows_ref = [], []

# Build experimental rows (need exp_conds + features).
for cn, feats in exp_feat_at_soh.items():
    if cn not in exp_conditions:
        continue
    row = {"cell_name": cn, **exp_conditions[cn], **feats}
    rows_exp.append(row)

# Build reference rows using features only (exp_conds may be missing for refs).
for cn, feats in ref_feat_at_soh.items():
    rows_ref.append({"cell_name": cn, **feats})

if not rows_exp:
    print("[WARN] No experimental rows with exp_conds + features for XGB.")
else:
    df_exp = pd.DataFrame(rows_exp)
    df_ref = pd.DataFrame(rows_ref)

    # dist_feat based on windowed features
    dist_df = pd.DataFrame(columns=["cell_name", "temp", "dist_feat", "best_ref"])
    if not df_ref.empty:
        ref_xyz = df_ref[FEATURE_NAMES_FOR_DIST].to_numpy(dtype=float)
        exp_xyz = df_exp[FEATURE_NAMES_FOR_DIST].to_numpy(dtype=float)

        all_xyz = np.vstack([ref_xyz, exp_xyz])
        q25 = np.quantile(all_xyz, 0.25, axis=0)
        q75 = np.quantile(all_xyz, 0.75, axis=0)
        scale = np.maximum(q75 - q25, 1e-12)

        ref_n = ref_xyz / scale
        exp_n = exp_xyz / scale

        dists = np.linalg.norm(exp_n[:, None, :] - ref_n[None, :, :], axis=2)
        dist = np.min(dists, axis=1)
        best_ref_idx = np.argmin(dists, axis=1)
        best_ref_names = df_ref.iloc[best_ref_idx]["cell_name"].to_numpy()

        dist_df = pd.DataFrame(
            {
                "cell_name": df_exp["cell_name"].to_numpy(),
                "temp": pd.to_numeric(df_exp["temp"], errors="coerce").to_numpy(),
                "dist_feat": dist.astype(float),
                "best_ref": best_ref_names,
            }
        ).sort_values("dist_feat").reset_index(drop=True)
    else:
        print("[WARN] No reference rows for dist_feat model.")

    if PRINT_CLOSEST_EXISTING and not dist_df.empty:
        top_by_temp = select_lowest_distance_per_temp(
            dist_df,
            temps=TOP_TEMPS,
            n_per_temp=TOP_PER_TEMP,
        )
        print(f"\nTop {TOP_PER_TEMP} closest existing cells per temperature by feature distance to reference:")
        for temp in TOP_TEMPS:
            temp_rows = top_by_temp[top_by_temp["temp"] == float(temp)]
            print(f"  Temp {temp}:")
            for _, row in temp_rows.iterrows():
                print(
                    f"    {row['cell_name']:36s}  dist={row['dist_feat']:.4f}  best_ref={row['best_ref']}"
                )

    # Train XGB models to predict features at SOH=0.995
    def train_xgb_feature_model(df_in, target_col):
        df_m = df_in.dropna(subset=[target_col]).copy()
        if len(df_m) < 5:
            print(f"[WARN] Not enough rows for {target_col} model.")
            return None
        X_feat, _ = make_features_from_raw(df_m, raw_conds=EXP_CONDS, drop_raw_soc=True)
        y = pd.to_numeric(df_m[target_col], errors="coerce")
        ok = ~y.isna() & np.isfinite(X_feat.to_numpy()).all(axis=1)
        X_feat = X_feat.loc[ok].reset_index(drop=True)
        y = y.loc[ok].to_numpy(dtype=float)
        if len(y) < 5:
            print(f"[WARN] Not enough valid rows for {target_col} model.")
            return None
        X_train, X_val, y_train, y_val = train_test_split(
            X_feat, y, test_size=0.2, random_state=42
        )
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            min_child_weight=2.0,
            gamma=0.1,
            tree_method="hist",
            random_state=42,
        )
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "rmse",
            "early_stopping_rounds": 100,
            "verbose": False,
        }
        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            # Older xgboost versions don't accept eval_metric/early_stopping in fit
            model.fit(X_train, y_train)
        y_hat_train = model.predict(X_train)
        y_hat_val = model.predict(X_val)
        print(
            f"[XGB] {target_col}: train R2={r2_score(y_train, y_hat_train):.4f}, "
            f"val R2={r2_score(y_val, y_hat_val):.4f}, n={len(y)}"
        )
        y_hat_all = model.predict(X_feat)
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        ax.scatter(y, y_hat_all, s=22, alpha=0.7, color="steelblue")
        lo = float(min(y.min(), y_hat_all.min()))
        hi = float(max(y.max(), y_hat_all.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
        ax.set_xlabel(f"true {target_col}")
        ax.set_ylabel(f"pred {target_col}")
        ax.set_title(f"XGB parity: {target_col}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_fig_dir / f"xgb_parity_{target_col}.png", dpi=150)
        plt.close(fig)
        return model

    print("\nTraining XGB models: exp_conds -> features at SOH=0.995")
    feature_models = {}
    for feat in FEATURE_NAMES_FOR_DIST:
        feature_models[feat] = train_xgb_feature_model(df_exp, feat)

    # Build regression table for var_mid_cha at throughput and train model
    df_all = pd.DataFrame()
    if traj_by_cell_reg:
        all_cells = sorted(traj_by_cell_reg.keys())
        df_all = build_regression_table_cap93_and_var_at_thr(
            all_cells,
            traj_by_cell_reg,
            cap_col="capacity",
            time_col="weeks",
            thr_col="throughput_cum",
            var_col="var_mid_cha",
            throughput_target=500_000.0 / 3600.0,
        )

    if df_all is None or df_all.empty:
        print("[WARN] df_all empty; skipping var_mid_cha model + Monte Carlo.")
    elif dist_df.empty:
        print("[WARN] dist_df empty; skipping var_mid_cha model + Monte Carlo.")
    else:
        model_dist, _, _, _, _ = train_xgb_no_val_and_shap(
            df_exp=df_exp,
            dist_df=dist_df,
            ref_names=REF_NAMES,
        )
        def train_xgb_var_mid_model(df_exp_local, df_reg_table, target_col="var_mid_cha_at_thr500k"):
            needed = ["cell_name"] + EXP_CONDS
            df_m = (
                df_exp_local[needed]
                .merge(df_reg_table[["cell_name", target_col]], on="cell_name", how="inner")
            )
            y = pd.to_numeric(df_m[target_col], errors="coerce")
            X_feat, _ = make_features_from_raw(df_m, raw_conds=EXP_CONDS, drop_raw_soc=True)
            ok = ~y.isna() & np.isfinite(X_feat.to_numpy()).all(axis=1)
            X_feat = X_feat.loc[ok].reset_index(drop=True)
            y = y.loc[ok].to_numpy(dtype=float)
            if len(y) < 5:
                print(f"[WARN] Not enough valid rows for {target_col} model.")
                return None
            X_train, X_val, y_train, y_val = train_test_split(
                X_feat, y, test_size=0.2, random_state=42
            )
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=2000,
                learning_rate=0.02,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=2.0,
                min_child_weight=2.0,
                gamma=0.1,
                tree_method="hist",
                random_state=42,
            )
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "eval_metric": "rmse",
                "early_stopping_rounds": 100,
                "verbose": False,
            }
            try:
                model.fit(X_train, y_train, **fit_kwargs)
            except TypeError:
                model.fit(X_train, y_train)

            y_hat_train = model.predict(X_train)
            y_hat_val = model.predict(X_val)
            print(
                f"[XGB] {target_col}: train R2={r2_score(y_train, y_hat_train):.4f}, "
                f"val R2={r2_score(y_val, y_hat_val):.4f}, n={len(y)}"
            )
            y_hat_all = model.predict(X_feat)
            fig, ax = plt.subplots(figsize=(5.0, 4.5))
            ax.scatter(y, y_hat_all, s=22, alpha=0.7, color="steelblue")
            lo = float(min(y.min(), y_hat_all.min()))
            hi = float(max(y.max(), y_hat_all.max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
            ax.set_xlabel(f"true {target_col}")
            ax.set_ylabel(f"pred {target_col}")
            ax.set_title(f"XGB parity: {target_col}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_fig_dir / f"xgb_parity_{target_col}.png", dpi=150)
            plt.close(fig)
            return model

        model_var_mid = train_xgb_var_mid_model(
            df_exp,
            df_all,
            target_col="var_mid_cha_at_thr500k",
        )

        # Predict var_mid_cha@thr500k for all experimental rows.
        if model_var_mid is not None:
            try:
                X_var_all, _ = make_features_from_raw(df_exp, raw_conds=EXP_CONDS, drop_raw_soc=True)
                ok_all = np.isfinite(X_var_all.to_numpy()).all(axis=1)
                df_var_pred = df_exp.loc[ok_all, ["cell_name"]].copy()
                pred_all = model_var_mid.predict(X_var_all.loc[ok_all])
                df_var_pred["pred_var_mid_cha_at_thr500k"] = np.maximum(pred_all, 0.0)
                if not dist_df.empty:
                    top_by_temp_with_pred = select_lowest_distance_per_temp(
                        dist_df.merge(df_var_pred, on="cell_name", how="left")
                        ,
                        temps=TOP_TEMPS,
                        n_per_temp=TOP_PER_TEMP,
                    )
                    print(f"\nTop {TOP_PER_TEMP} closest cells per temperature with predicted var_mid_cha_at_thr500k:")
                    for temp in TOP_TEMPS:
                        temp_rows = top_by_temp_with_pred[top_by_temp_with_pred["temp"] == float(temp)]
                        print(f"  Temp {temp}:")
                        for _, row in temp_rows.iterrows():
                            pred_val = row["pred_var_mid_cha_at_thr500k"]
                            pred_str = f"{pred_val:.6g}" if np.isfinite(pred_val) else "nan"
                            print(
                                f"    {row['cell_name']:36s}  dist={row['dist_feat']:.4f}  "
                                f"best_ref={row['best_ref']}  pred_var_mid_cha={pred_str}"
                            )
                top10_var = df_var_pred.nlargest(10, "pred_var_mid_cha_at_thr500k")
                print("\nTop 10 cells by predicted var_mid_cha_at_thr500k:")
                for _, row in top10_var.iterrows():
                    print(f"  {row['cell_name']:40s}  pred_var_mid_cha={row['pred_var_mid_cha_at_thr500k']:.6g}")
            except Exception as e:
                print(f"[WARN] Could not rank predicted var_mid_cha: {e}")

        df_dist = df_exp.merge(dist_df[["cell_name", "temp", "dist_feat"]], on=["cell_name", "temp"], how="inner")
        df_dist = df_dist.dropna(subset=["dist_feat"]).copy()
        if not df_dist.empty:
            X_dist, _ = make_features_from_raw(df_dist, raw_conds=EXP_CONDS, drop_raw_soc=True)
            y_dist = pd.to_numeric(df_dist["dist_feat"], errors="coerce")
            ok = ~y_dist.isna() & np.isfinite(X_dist.to_numpy()).all(axis=1)
            X_dist = X_dist.loc[ok].reset_index(drop=True)
            y_dist = y_dist.loc[ok].to_numpy(dtype=float)
            if len(y_dist) >= 3:
                y_pred = model_dist.predict(X_dist)
                fig, ax = plt.subplots(figsize=(5.0, 4.5))
                ax.scatter(y_dist, y_pred, s=22, alpha=0.7, color="steelblue")
                lo = float(min(y_dist.min(), y_pred.min()))
                hi = float(max(y_dist.max(), y_pred.max()))
                ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
                ax.set_xlabel("true dist_feat")
                ax.set_ylabel("pred dist_feat")
                ax.set_title("XGB parity: dist_feat")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(out_fig_dir / "xgb_parity_dist_feat.png", dpi=150)
                plt.close(fig)

        mc_all = monte_carlo_best_conditions_for_distance(
            df_exp=df_exp,
            model=model_dist,
            ref_names=REF_NAMES,
            n_samples_per_temp=MC_SAMPLES_PER_TEMP,
            top_k=None,
            allowed_temp=MC_ALLOWED_TEMP,
            allowed_soc_start=MC_ALLOWED_SOC_START,
            allowed_soc_end=MC_ALLOWED_SOC_END,
            allowed_cur_cha=MC_ALLOWED_C_RATE_CHG,
            allowed_cur_dis=MC_ALLOWED_C_RATE_DCHG,
            min_soc_delta=MC_MIN_SOC_DELTA,
        )

        if mc_all.empty:
            print("[WARN] Monte Carlo search returned no candidates.")
        else:
            mc_all = mc_all.copy()
            if model_var_mid is not None:
                try:
                    X_mc_feat, _ = make_features_from_raw(mc_all, raw_conds=EXP_CONDS, drop_raw_soc=True)
                    ok_mc = np.isfinite(X_mc_feat.to_numpy()).all(axis=1)
                    pred_var_mid = np.full(len(mc_all), np.nan, dtype=float)
                    pred_var_mid[ok_mc] = np.maximum(
                        model_var_mid.predict(X_mc_feat.loc[ok_mc]),
                        0.0,
                    )
                    mc_all["pred_var_mid_cha_at_thr500k"] = pred_var_mid
                except Exception as e:
                    print(f"[WARN] Could not predict var_mid_cha for MC candidates: {e}")
            mc_top = select_lowest_distance_per_temp(
                mc_all,
                temp_col="temp",
                dist_col="pred_dist_feat",
                temps=TOP_TEMPS,
                n_per_temp=TOP_PER_TEMP,
            )

            print(f"\nTop {TOP_PER_TEMP} Monte Carlo conditions per temperature by predicted dist_feat")
            cols_show = EXP_CONDS + ["pred_dist_feat"]
            if "pred_var_mid_cha_at_thr500k" in mc_top.columns:
                cols_show.append("pred_var_mid_cha_at_thr500k")
            cols_show = [c for c in cols_show if c in mc_top.columns]
            for temp in TOP_TEMPS:
                temp_rows = mc_top[pd.to_numeric(mc_top["temp"], errors="coerce") == float(temp)]
                print(f"  Temp {temp}:")
                if temp_rows.empty:
                    print("    [none]")
                else:
                    print(temp_rows[cols_show].to_string(index=False))

            print(f"\nConditions only (top {TOP_PER_TEMP} per temperature by dist_feat):")
            for temp in TOP_TEMPS:
                temp_rows = mc_top[pd.to_numeric(mc_top["temp"], errors="coerce") == float(temp)]
                print(f"  Temp {temp}:")
                if temp_rows.empty:
                    print("    [none]")
                else:
                    print(temp_rows[EXP_CONDS].to_string(index=False))

# =============================================================================
# STEP 7: MAKE DATASET FOR MODEL (CNN)
# =============================================================================
print("Loading data from interp_data dictionary ...")

train_X_list, train_y_list, train_names = [], [], []
test_X_list, test_y_list, test_names_found = [], [], []

for cell_name, cell_dict in interp_data.items():
    missing = [f for f in FEATURE_COLS if f not in cell_dict]
    if missing:
        print(f"  [SKIP] {cell_name} — missing {missing}")
        continue

    step_to_target = cell_dict.get("step_to_target")
    if step_to_target is None or step_to_target <= 0:
        print(f"  [SKIP] {cell_name} — invalid step_to_target for SOH={SOH_STEP_TARGET}")
        continue
    step_to_target = float(step_to_target)

    feat_arrays = []
    for f in FEATURE_COLS:
        feat_arrays.append(np.asarray(cell_dict[f], dtype=float))
    feat_mat = np.stack(feat_arrays, axis=1)

    if len(feat_mat) < INPUT_STEPS:
        print(f"  [SKIP] {cell_name} — only {len(feat_mat)} steps, need {INPUT_STEPS}")
        continue

    X_seq = feat_mat[:INPUT_STEPS, :]
    if not np.all(np.isfinite(X_seq)):
        print(f"  [SKIP] {cell_name} — non-finite values")
        continue

    if cell_name in TEST_NAMES:
        test_X_list.append(X_seq)
        test_y_list.append(step_to_target)
        test_names_found.append(cell_name)
        print(f"  [TEST]  {cell_name}  step_target={step_to_target:.0f}")
    elif "cycle" in cell_name.lower():
        train_X_list.append(X_seq)
        train_y_list.append(step_to_target)
        train_names.append(cell_name)
        print(f"  [TRAIN] {cell_name}  step_target={step_to_target:.0f}")
    else:
        print(f"  [SKIP]  {cell_name} (not cycle, not ref)")

X_train_all = np.stack(train_X_list, axis=0)
y_train_all = np.array(train_y_list)
X_test = np.stack(test_X_list, axis=0)
y_test = np.array(test_y_list)

print(f"\nTrain: {X_train_all.shape[0]} cells,  Test: {X_test.shape[0]} cells")
print(f"Features: {FEATURE_COLS}")
print(f"Input shape per cell: ({INPUT_STEPS}, {INPUT_FEATURES})")


# log-transform target
y_train_all = np.log(y_train_all / 20.0)
y_test = np.log(y_test / 20.0)


# =============================================================================
# STEP 8: TRAIN / VAL SPLIT
# =============================================================================
use_validation_local = USE_VALIDATION
if use_validation_local and len(X_train_all) > 10:
    X_train, y_train, X_val, y_val = create_balanced_val_split(
        X_train_all,
        y_train_all,
        val_fraction=VALIDATION_SPLIT,
        bins=min(3, len(X_train_all) // 3)
    )
else:
    X_train, y_train = X_train_all, y_train_all
    X_val, y_val = None, None
    use_validation_local = False

X_train_t = torch.from_numpy(X_train).float().to(device)
y_train_t = torch.from_numpy(y_train).float().view(-1, 1).to(device)

if use_validation_local:
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().view(-1, 1).to(device)

X_test_t = torch.from_numpy(X_test).float().to(device)
y_test_t = torch.from_numpy(y_test).float().view(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# =============================================================================
# STEP 9: TRAIN ENSEMBLE
# =============================================================================
np.random.seed(42)
random_numbers = np.random.choice(range(0, 100), size=N_SEEDS, replace=False)
y_pred_list = []

for i in random_numbers:
    print(f"\n── Seed {i} ──")

    if "model" in locals():
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    set_seed(i)

    model = TinyTemporalCNN(INPUT_FEATURES, INPUT_STEPS, dropout_rate=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    best_val_loss = float("inf")
    early_patience = 10
    epochs_no_improve = 0
    early_stop = False

    save_dir = "savemodel"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_model_dqdv_{INTERP_METHOD}.pth")

    for epoch in range(EPOCHS + 1):
        if early_stop:
            print(f"  Early stop at epoch {epoch}")
            break

        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = mse_loss(y_batch, out)
            loss.backward()
            optimizer.step()

        if use_validation_local and (epoch % 10 == 0 or epoch == EPOCHS):
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = mse_loss(y_val_t, val_out)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_patience:
                        early_stop = True

        if epoch % 100 == 0:
            msg = f"  Epoch {epoch}/{EPOCHS}  loss={loss.item():.4f}"
            if use_validation_local:
                msg += f"  val_loss={val_loss.item():.4f}"
            print(msg)

    if use_validation_local and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy().reshape(-1)
    y_pred_list.append(y_pred)


# =============================================================================
# STEP 10: RESULTS
# =============================================================================
y_pred_mean = np.mean(y_pred_list, axis=0)

print("\n" + "=" * 70)
print(f"RESULTS (step scale) — interpolation={INTERP_METHOD}")
print("=" * 70)
for name, true, pred in zip(test_names_found, y_test, y_pred_mean):
    print(f"  {name:30s}  true={true:.4f}  pred={pred:.4f}")

print(f"\n  MAPE (log):  {np.mean(np.abs((y_test - y_pred_mean) / y_test)) * 100:.2f}%")

# Back to original scale
y_test_steps = np.exp(y_test) * 20.0
y_pred_steps = np.exp(y_pred_mean) * 20.0

print("\nRESULTS (original step scale)")
print("=" * 70)
for name, true, pred in zip(test_names_found, y_test_steps, y_pred_steps):
    print(f"  {name:30s}  true={true:.1f}  pred={pred:.1f}")

rmse = np.sqrt(np.mean((y_test_steps - y_pred_steps) ** 2))
mape = np.mean(np.abs((y_test_steps - y_pred_steps) / y_test_steps)) * 100
print(f"\n  RMSE (steps): {rmse:.2f}")
print(f"  MAPE (steps): {mape:.2f}%")


# =============================================================================
# STEP 11: FINAL PLOTS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
x_pos = np.arange(len(test_names_found))
width = 0.35
ax1.bar(x_pos - width / 2, y_test_steps, width, label="True", color="steelblue", alpha=0.8)
ax1.bar(x_pos + width / 2, y_pred_steps, width, label="Predicted", color="coral", alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([n.replace("SPEED_LW_", "") for n in test_names_found], rotation=15)
ax1.set_ylabel("Steps to SOH target")
ax1.set_title(f"True vs Predicted steps ({INTERP_METHOD}, RMSE={rmse:.1f}, MAPE={mape:.1f}%)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
preds_steps = np.exp(np.array(y_pred_list)) * 20.0
for j, name in enumerate(test_names_found):
    ax2.scatter(np.full(N_SEEDS, j), preds_steps[:, j], alpha=0.4, s=20, color="coral")
    ax2.scatter(j, y_test_steps[j], marker="*", s=200, color="steelblue", zorder=5)
ax2.set_xticks(range(len(test_names_found)))
ax2.set_xticklabels([n.replace("SPEED_LW_", "") for n in test_names_found], rotation=15)
ax2.set_ylabel("Steps to SOH target")
ax2.set_title(f"Ensemble predictions ({INTERP_METHOD}) vs true")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(out_fig_dir / f"prediction_results_{INTERP_METHOD}.png", dpi=150)
plt.show()
