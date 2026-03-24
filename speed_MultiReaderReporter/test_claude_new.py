from pathlib import Path
import numpy as np
import pandas as pd
import ast
from core.capacity import *
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_NAMES_FOR_DIST = ["mean_mid_cha", "mean_high_cha", "mean_pla_cha"]

def parse_array_string(s):
    s = str(s).replace("NBSP", " ").replace("\\n", " ").replace("\n", " ")
    s = s.strip("[] ")
    return np.fromstring(s, sep=" ")


def load_and_interpolate(df, target_soh, interpolation_typ="weeks", method="linear", throughput_max=None):
    """Interpolate all feature columns on a new reference grid."""
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
    ref_s = pd.Series(reference)
    keep = ~ref_s.duplicated(keep="first")
    df = df.loc[keep.values].reset_index(drop=True)
    reference = df[ref_name].to_numpy(dtype=float)
    target_data = df["SOH"].to_numpy(dtype=float)

    if len(reference) < 2:
        return None
    if method == "cubic" and len(reference) < 3:
        return None

    feature_cols = [c for c in df.columns if c != ref_name]
    feature = df[feature_cols].to_numpy(dtype=float)

    interpolated_ref = np.interp(target_soh, target_data[::-1], reference[::-1])
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

    interpolated_columns = []
    for j in range(feature.shape[1]):
        yj = feature[:, j]
        if np.isnan(yj).any() or method == "linear":
            interpolated_columns.append(np.interp(new_ref_points, reference, yj))
        elif method == "cubic":
            try:
                cs = CubicSpline(reference, yj, bc_type="natural", extrapolate=False)
                interpolated_columns.append(cs(new_ref_points))
            except:
                interpolated_columns.append(np.interp(new_ref_points, reference, yj))

    out = pd.DataFrame(np.stack(interpolated_columns, axis=1), columns=feature_cols)
    out[ref_name] = new_ref_points
    if "SOH" in out.columns:
        out["SOH"] = out["SOH"].clip(lower=0.0, upper=1.05)
    return out


def features_at_soh(feat_dict, target_soh=0.995):
    """
    Use capacity from feat_dict to get SOH per row,
    then interpolate each feature list to target_soh.
    """
    cap = np.array(feat_dict["capacity"], dtype=float)
    cap = cap[~np.isnan(cap)]
    if len(cap) < 2 or cap[0] == 0:
        return None
    soh = cap / cap[0]

    out = {}
    for fname in FEATURE_NAMES_FOR_DIST:
        vals = feat_dict.get(fname, [])
        if len(vals) == 0:
            return None
        vals = np.array(vals, dtype=float)
        n = min(len(soh), len(vals))
        if n < 2:
            return None
        # SOH is decreasing → flip for np.interp
        val_at_target = np.interp(target_soh, soh[:n][::-1], vals[:n][::-1])
        out[fname] = float(val_at_target)
    return out

interp_dir = Path(r"C:\Users\Victus\PycharmProjects\ExpSpeed\out_lw\cell_feature")
out_fig_dir = Path(r"C:\Users\Victus\PycharmProjects\ExpSpeed\out_lw\out_figure")
out_fig_dir.mkdir(parents=True, exist_ok=True)
REF_NAMES = ["SPEED_LW_reference_1","SPEED_LW_reference_2","SPEED_LW_reference_3"]

x_col = "Vcha"
y_col = "dQdVcha"
v_1 = 3.25
v_2 = 3.35
v_3 = 3.4

TARGET_SOH_FEATURES = 0.995   # SOH at which to compare features (closest cells)
TARGET_SOH_PLOT = 0.98        # SOH for interpolation grid
K_CLOSEST = 25
K_FARTHEST = 20
INTERP_METHOD = "linear"

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
    df["dQdVcha"] = (df["dQdVcha"].str.strip("[]").str.split().apply(lambda x: np.asarray(x, dtype=float)))
    df["Vcha"] = (df["Vcha"].str.strip("[]").str.split().apply(lambda x: np.asarray(x, dtype=float)))
    df["Q_intVcha"] = (df["Q_intVcha"].str.strip("[]").str.split().apply(lambda x: np.asarray(x, dtype=float)))

    if x_col not in df.columns or y_col not in df.columns:
        del df
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in range(len(df)):
        try:
            x_arr = parse_array_string(df[x_col].iloc[idx])
            y_arr = parse_array_string(df[y_col].iloc[idx])
        except:
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

    # calculate features
    mean_low_cha, var_low_cha = window_delta_mean_var(df, x_col="Vcha", y_col="dQdVcha", x_lo=2.7, x_hi=v_1)
    mean_mid_cha, var_mid_cha = window_delta_mean_var(df, x_col="Vcha", y_col="dQdVcha", x_lo=v_1, x_hi=v_2)
    mean_high_cha, var_high_cha = window_delta_mean_var(df, x_col="Vcha", y_col="dQdVcha", x_lo=v_2, x_hi=v_3)
    mean_pla_cha, var_pla_cha = window_delta_mean_var(df, x_col="Vcha", y_col="dQdVcha", x_lo=v_3, x_hi=3.6)
    capacity = df["Q_intVcha"].apply(lambda x: x[-1])
    SOH = capacity / capacity.iloc[0]
    throughput = np.array(df["throughput_sum"])

    df["CU_time"] = pd.to_datetime(df["CU_time"])
    t0 = df["CU_time"].iloc[0]
    df["time_weeks"] = (df["CU_time"] - t0).dt.total_seconds() / (7 * 24 * 3600)
    time_array = df["time_weeks"].to_numpy()

    feat_dict  = {
        "mean_low_cha": mean_low_cha,
        "var_low_cha": var_low_cha,
        "mean_mid_cha": mean_mid_cha,
        "var_mid_cha": var_mid_cha,
        "mean_high_cha": mean_high_cha,
        "var_high_cha": var_high_cha,
        "mean_pla_cha": mean_pla_cha,
        "var_pla_cha": var_pla_cha,
        "capacity": capacity,
        "throughput": throughput,
        "Time": time_array,
    }

    for key in list(feat_dict.keys()):
        if key not in ("SOH", "capacity"):
            feat_dict[f"{key}"] = feat_dict[key] / capacity.iloc[0]

    if is_ref:
        results_ref[cell_name] = feat_dict
    else:
        results[cell_name] = feat_dict
    # plt.show()
    # del df
    print(f"{'[REF] ' if is_ref else '[TRAIN]'} {cell_name}: done")

# build feature vectors at SOH=0.995
ref_feat_at_soh = {}
for cn, fd in results_ref.items():
    f = features_at_soh(fd, TARGET_SOH_FEATURES)
    if f is not None:
        ref_feat_at_soh[cn] = f
        print(f"[REF  @ SOH={TARGET_SOH_FEATURES}] {cn}: {f}")

exp_feat_at_soh = {}
for cn, fd in results.items():
    f = features_at_soh(fd, TARGET_SOH_FEATURES)
    if f is not None:
        exp_feat_at_soh[cn] = f

print(f"\nRef cells with features: {len(ref_feat_at_soh)}")
print(f"Exp cells with features: {len(exp_feat_at_soh)}")

# compute distances
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

    k1 = min(K_CLOSEST, len(exp_names_list))
    k2 = min(K_FARTHEST, len(exp_names_list))
    closest_idx = np.argsort(dist_min)[:k1]
    farthest_idx = np.argsort(dist_min)[-k2:]

    closest_cellnames = [exp_names_list[i] for i in closest_idx]
    farthest_cellnames = [exp_names_list[i] for i in farthest_idx]

    print(f"\nClosest {k1} cells:")
    for i in closest_idx:
        print(f"  {exp_names_list[i]:40s}  dist={dist_min[i]:.4f}")

# ── STEP 3: Interpolate trajectories + features at SOH=0.98 ────────────
closest_set = set(closest_cellnames)
farthest_set = set(farthest_cellnames)
ref_set = set(REF_NAMES)

INTERP_FEATURES = ["mean_low_cha", "mean_mid_cha", "mean_high_cha", "mean_pla_cha",
                    "var_low_cha", "var_mid_cha", "var_high_cha", "var_pla_cha"]

fig_w, ax_w = plt.subplots(figsize=(10, 6))
added_labels = {"blue": False, "orange": False, "green": False, "red": False}
label_map = {"red": "refs", "orange": f"closest {K_CLOSEST}", "green": f"farthest {K_FARTHEST}", "blue": "other"}

all_results = {**results_ref, **results}
interp_data = {}  # dict of dicts: interp_data[cell_name] = {"SOH": arr, "mean_mid_cha": arr, ...}

for cell_name, fd in all_results.items():
    cap = np.array(fd["capacity"], dtype=float)
    time_arr = np.array(fd["Time"], dtype=float)

    if len(cap) < 3 or cap[0] == 0:
        continue

    soh = cap / cap[0]

    # find time where SOH = TARGET_SOH_PLOT
    if np.nanmin(soh) > TARGET_SOH_PLOT:
        t_target = time_arr[-1]
    else:
        t_target = float(np.interp(TARGET_SOH_PLOT, soh[::-1], time_arr[::-1]))

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

    # interpolate SOH
    soh_interp = np.interp(t_grid, time_arr, soh)

    # interpolate all features onto the same grid
    cell_interp = {
        "time": t_grid,
        "SOH": soh_interp,
        "capacity": np.interp(t_grid, time_arr, cap),
    }

    for feat_name in INTERP_FEATURES:
        feat_vals = fd.get(feat_name, [])
        if len(feat_vals) == 0:
            continue
        feat_vals = np.array(feat_vals, dtype=float)
        n = min(len(time_arr), len(feat_vals))
        if n < 2:
            continue
        cell_interp[feat_name] = np.interp(t_grid, time_arr[:n], feat_vals[:n])

    interp_data[cell_name] = cell_interp

    # ── plot SOH ──
    if cell_name in ref_set:
        color, lw, alpha = "red", 1.8, 0.95
    elif cell_name in closest_set:
        color, lw, alpha = "orange", 1.2, 0.85
    elif cell_name in farthest_set:
        color, lw, alpha = "green", 1.2, 0.85
    else:
        color, lw, alpha = "blue", 0.6, 0.20

    label = None
    if not added_labels[color]:
        label = label_map[color]
        added_labels[color] = True

    ax_w.plot(soh_interp, color=color, linewidth=lw, alpha=alpha, label=label)

ax_w.set_xlabel("Step")
ax_w.set_ylabel("SOH")
ax_w.set_title(
    f"SOH trajectories (interp @ {TARGET_SOH_PLOT}, features @ {TARGET_SOH_FEATURES})\n"
    f"closest/farthest based on dQdV windowed features"
)
ax_w.set_ylim(0.8, 1.05)
ax_w.grid(True, alpha=0.3)
ax_w.legend(loc="best")
fig_w.tight_layout()
fig_w.savefig(out_fig_dir / "SOH_vs_time_closest_farthest.png", dpi=150)
plt.show()

# interp_data is now ready for model training
# Access like: interp_data["SPEED_LW_cycle_1"]["mean_mid_cha"]  → numpy array
# All arrays for a given cell have the same length (same t_grid)
print(f"\nInterpolated {len(interp_data)} cells with features: {INTERP_FEATURES}")

SOH_STEP_TARGET = 0.96

for cell_name, cell_dict in interp_data.items():
    soh = cell_dict["SOH"]
    hits = np.where(soh <= SOH_STEP_TARGET)[0]
    if len(hits) > 0 and hits[0] > 0:
        cell_dict["step_to_target"] = int(hits[0])
    else:
        cell_dict["step_to_target"] = None

# ── Plot all interpolated features as subplots ──────────────────────────
import math
from matplotlib.lines import Line2D

# features available in interp_data that you want to visualize
PLOT_FEATURES = [
    "SOH",
    "capacity",
    "mean_low_cha", "mean_mid_cha", "mean_high_cha", "mean_pla_cha",
    "var_low_cha", "var_mid_cha", "var_high_cha", "var_pla_cha",
]

# keep only features that actually exist in at least one cell
plot_features_existing = [
    f for f in PLOT_FEATURES
    if any(f in cell_dict for cell_dict in interp_data.values())
]

n_feat = len(plot_features_existing)
n_cols = 3
n_rows = math.ceil(n_feat / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.0 * n_rows))
axes = np.array(axes).reshape(-1)

# styling
style_map = {
    "ref":      {"color": "red",    "lw": 1.8, "alpha": 0.95},
    "closest":  {"color": "orange", "lw": 1.2, "alpha": 0.85},
    "farthest": {"color": "green",  "lw": 1.2, "alpha": 0.85},
    "other":    {"color": "blue",   "lw": 0.6, "alpha": 0.20},
}

for ax, feat_name in zip(axes, plot_features_existing):
    for cell_name, cell_dict in interp_data.items():
        if feat_name not in cell_dict:
            continue

        y = np.asarray(cell_dict[feat_name], dtype=float)
        x = np.arange(len(y))   # step index on interpolated grid

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

# hide unused axes
for ax in axes[n_feat:]:
    ax.axis("off")

# shared legend
legend_handles = [
    Line2D([0], [0], color="red",    lw=1.8, alpha=0.95, label="refs"),
    Line2D([0], [0], color="orange", lw=1.2, alpha=0.85, label=f"closest {K_CLOSEST}"),
    Line2D([0], [0], color="green",  lw=1.2, alpha=0.85, label=f"farthest {K_FARTHEST}"),
    Line2D([0], [0], color="blue",   lw=1.0, alpha=0.50, label="other"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=True)

fig.suptitle(
    f"Interpolated feature trajectories\n"
    f"(grid anchored at SOH={TARGET_SOH_PLOT}, closest/farthest based on features at SOH={TARGET_SOH_FEATURES})",
    y=0.995
)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out_fig_dir / "all_interpolated_features_subplots.png", dpi=150)
plt.show()

# make the model and train it
"""
Predict step number at SOH target from interpolated dQdV features.

Uses interp_data dictionary (already in memory from the previous script).
Training data : cells with 'cycle' in the name
Test data     : SPEED_LW_reference_1, SPEED_LW_reference_2, SPEED_LW_reference_3
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import os
import pandas as pd
import random
import gc

# ── Config ──────────────────────────────────────────────────────────────
FEATURE_COLS = ["SOH", "mean_mid_cha", "mean_high_cha", "mean_pla_cha",
                "var_mid_cha", "var_high_cha", "var_pla_cha"]

SOH_STEP_TARGET = 0.95   # target SOH to find the step number for
INPUT_STEPS = 6           # how many interpolated time steps to use as input
INPUT_FEATURES = len(FEATURE_COLS)

TEST_NAMES = [
    "SPEED_LW_reference_1",
    "SPEED_LW_reference_2",
    "SPEED_LW_reference_3",
]

# Training hyperparams
EPOCHS = 500
BATCH_SIZE = 16
LR = 0.001
DROPOUT = 0.1
VALIDATION_SPLIT = 0.2
USE_VALIDATION = True
N_SEEDS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse_loss = nn.MSELoss()


# ── Helpers ─────────────────────────────────────────────────────────────
def set_seed(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_balanced_val_split(X, y, val_fraction=0.2, bins=3):
    y = np.array(y)
    y_bins = pd.qcut(y, q=bins, labels=False, duplicates="drop")
    mid_val_frac = int(val_fraction * len(y)) / len(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=mid_val_frac, random_state=42)
    train_idx, val_idx = next(sss.split(X, y_bins))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ── Model ───────────────────────────────────────────────────────────────
class TinyTemporalCNN(nn.Module):
    def __init__(self, input_dims, timesteps, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dims, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)

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


# ── Load data from interp_data dictionary ──────────────────────────────
print("Loading data from interp_data dictionary ...")

train_X_list, train_y_list, train_names = [], [], []
test_X_list, test_y_list, test_names_found = [], [], []

for cell_name, cell_dict in interp_data.items():
    # check all features exist
    missing = [f for f in FEATURE_COLS if f not in cell_dict]
    if missing:
        print(f"  [SKIP] {cell_name} — missing {missing}")
        continue

    soh = cell_dict["SOH"]

    step_to_target = cell_dict.get("step_to_target")
    if step_to_target is None or step_to_target <= 0:
        print(f"  [SKIP] {cell_name} — never reached SOH={SOH_STEP_TARGET}")
        continue
    step_to_target = float(step_to_target)

    # build feature matrix (n_steps, n_features)
    feat_arrays = []
    for f in FEATURE_COLS:
        feat_arrays.append(cell_dict[f])
    feat_mat = np.stack(feat_arrays, axis=1)  # (n_steps, n_features)

    if len(feat_mat) < INPUT_STEPS:
        print(f"  [SKIP] {cell_name} — only {len(feat_mat)} steps, need {INPUT_STEPS}")
        continue

    X_seq = feat_mat[:INPUT_STEPS, :]
    if not np.all(np.isfinite(X_seq)):
        print(f"  [SKIP] {cell_name} — non-finite values")
        continue

    # assign to train or test
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

# ── Scale features ──────────────────────────────────────────────────────
combined = np.concatenate([X_train_all, X_test], axis=0)
flat = combined.reshape(-1, INPUT_FEATURES)
scaler = StandardScaler()
flat_scaled = scaler.fit_transform(flat)
combined_scaled = flat_scaled.reshape(combined.shape)
X_train_all = combined_scaled[:len(train_X_list)]
X_test = combined_scaled[len(train_X_list):]

# ── Log-transform target ───────────────────────────────────────────────
y_train_all = np.log(y_train_all)
y_test = np.log(y_test)

# ── Train / val split ──────────────────────────────────────────────────
if USE_VALIDATION and len(X_train_all) > 10:
    X_train, y_train, X_val, y_val = create_balanced_val_split(
        X_train_all, y_train_all, val_fraction=VALIDATION_SPLIT,
        bins=min(3, len(X_train_all) // 3)
    )
else:
    X_train, y_train = X_train_all, y_train_all
    X_val, y_val = None, None
    USE_VALIDATION = False

# ── Tensors ─────────────────────────────────────────────────────────────
X_train_t = torch.from_numpy(X_train).float().to(device)
y_train_t = torch.from_numpy(y_train).float().view(-1, 1).to(device)
if USE_VALIDATION:
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().view(-1, 1).to(device)
X_test_t = torch.from_numpy(X_test).float().to(device)
y_test_t = torch.from_numpy(y_test).float().view(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ── Training loop (ensemble over seeds) ─────────────────────────────────
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

        if USE_VALIDATION and (epoch % 10 == 0 or epoch == EPOCHS):
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = mse_loss(y_val_t, val_out)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    os.makedirs("savemodel", exist_ok=True)
                    torch.save(model.state_dict(), "savemodel/best_model_dqdv.pth")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_patience:
                        early_stop = True

        if epoch % 100 == 0:
            msg = f"  Epoch {epoch}/{EPOCHS}  loss={loss.item():.4f}"
            if USE_VALIDATION:
                msg += f"  val_loss={val_loss.item():.4f}"
            print(msg)

    # Load best model for prediction
    if USE_VALIDATION and os.path.exists("savemodel/best_model_dqdv.pth"):
        model.load_state_dict(torch.load("savemodel/best_model_dqdv.pth"))

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy().reshape(-1)
    y_pred_list.append(y_pred)

# ── Results ─────────────────────────────────────────────────────────────
y_pred_mean = np.mean(y_pred_list, axis=0)

print("\n" + "=" * 70)
print("RESULTS (log space)")
print("=" * 70)
for name, true, pred in zip(test_names_found, y_test, y_pred_mean):
    print(f"  {name:30s}  true={true:.4f}  pred={pred:.4f}")

print(f"\n  MAPE (log):  {np.mean(np.abs((y_test - y_pred_mean) / y_test)) * 100:.2f}%")

# Back to original scale (steps)
y_test_steps = np.exp(y_test)
y_pred_steps = np.exp(y_pred_mean)

print("\nRESULTS (original step scale)")
print("=" * 70)
for name, true, pred in zip(test_names_found, y_test_steps, y_pred_steps):
    print(f"  {name:30s}  true={true:.1f}  pred={pred:.1f}")

rmse = np.sqrt(np.mean((y_test_steps - y_pred_steps) ** 2))
mape = np.mean(np.abs((y_test_steps - y_pred_steps) / y_test_steps)) * 100
print(f"\n  RMSE (steps): {rmse:.2f}")
print(f"  MAPE (steps): {mape:.2f}%")

# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
x_pos = np.arange(len(test_names_found))
width = 0.35
ax1.bar(x_pos - width / 2, y_test_steps, width, label="True", color="steelblue", alpha=0.8)
ax1.bar(x_pos + width / 2, y_pred_steps, width, label="Predicted", color="coral", alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([n.replace("SPEED_LW_", "") for n in test_names_found], rotation=15)
ax1.set_ylabel("Steps to SOH target")
ax1.set_title(f"True vs Predicted steps  (RMSE={rmse:.1f}, MAPE={mape:.1f}%)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
preds_steps = np.exp(np.array(y_pred_list))
for j, name in enumerate(test_names_found):
    ax2.scatter(np.full(N_SEEDS, j), preds_steps[:, j], alpha=0.4, s=20, color="coral")
    ax2.scatter(j, y_test_steps[j], marker="*", s=200, color="steelblue", zorder=5)
ax2.set_xticks(range(len(test_names_found)))
ax2.set_xticklabels([n.replace("SPEED_LW_", "") for n in test_names_found], rotation=15)
ax2.set_ylabel("Steps to SOH target")
ax2.set_title("Ensemble predictions (dots) vs true (stars)")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()


