"""
Predict step_to_soh_target from interpolated features.

Training data : cells with 'cycle' in the name (from interp_feature folder)
Test data     : SPEED_LW_reference_1, SPEED_LW_reference_2, SPEED_LW_reference_3
Features      : cap_ocv_dis, mean_d_dqdv_m_c, var_d_dqdv_m_c,
                mean_d_dqdv_l_c_l, mean_d_dqdv_h_c, mean_d_dqdv_l_c
Target        : step_to_soh_target (saved by the previous code block)
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
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
here = Path(__file__).resolve().parent
INTERP_DIR = here / "interp_feature"

FEATURE_COLS = ["mean_d_dqdv_m_c","mean_d_dqdv_h_c","mean_d_dqdv_l_c",
    "var_d_dqdv_m_c","var_d_dqdv_h_c","var_d_dqdv_l_c","SOH"
]
TARGET_COL = "step_to_soh_target"
INPUT_STEPS = 6         # how many interpolated time steps to use as input
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
    seed = int(seed)  # ← add this line
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_balanced_val_split(X, y, val_fraction=0.2, bins=3):
    y = np.array(y)
    total = len(y)
    desired_val = int(val_fraction * total)
    y_bins = pd.qcut(y, q=bins, labels=False, duplicates="drop")
    mid_val_frac = desired_val / total
    sss = StratifiedShuffleSplit(n_splits=1, test_size=mid_val_frac, random_state=42)
    train_idx, val_idx = next(sss.split(X, y_bins))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ── Model (same TinyTemporalCNN from your code) ────────────────────────
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


# ── Load data from interp_feature folder ───────────────────────────────
def load_cell(csv_path: Path):
    """
    Read one cell CSV from interp_feature/.
    Returns (X_seq, y_target) or (None, None) if unusable.
    X_seq shape: (INPUT_STEPS, INPUT_FEATURES)
    y_target: scalar (step_to_soh_target)
    """
    df = pd.read_csv(csv_path)

    # target
    if TARGET_COL not in df.columns:
        return None, None
    target_val = df[TARGET_COL].iloc[0]
    if pd.isna(target_val) or target_val <= 0:
        return None, None

    # features
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return None, None

    feat = df[FEATURE_COLS].to_numpy(dtype=float)
    if len(feat) < INPUT_STEPS:
        return None, None

    X_seq = feat[:INPUT_STEPS, :]
    if not np.all(np.isfinite(X_seq)):
        return None, None

    return X_seq, float(target_val)


print(f"Loading cells from {INTERP_DIR} ...")

train_X_list, train_y_list, train_names = [], [], []
test_X_list, test_y_list, test_names = [], [], []

for csv_file in sorted(INTERP_DIR.glob("*.csv")):
    if csv_file.name.startswith("_"):
        continue  # skip summary file

    cell_name = csv_file.stem
    X_seq, y_target = load_cell(csv_file)
    if X_seq is None:
        print(f"  [SKIP] {cell_name}")
        continue

    if cell_name in TEST_NAMES:
        test_X_list.append(X_seq)
        test_y_list.append(y_target)
        test_names.append(cell_name)
        print(f"  [TEST]  {cell_name}  step_target={y_target:.0f}")
    elif "cycle" in cell_name.lower():
        train_X_list.append(X_seq)
        train_y_list.append(y_target)
        train_names.append(cell_name)
        print(f"  [TRAIN] {cell_name}  step_target={y_target:.0f}")
    else:
        print(f"  [SKIP]  {cell_name} (not cycle, not ref)")

X_train_all = np.stack(train_X_list, axis=0)   # (N_train, INPUT_STEPS, INPUT_FEATURES)
y_train_all = np.array(train_y_list)            # (N_train,)
X_test = np.stack(test_X_list, axis=0)          # (N_test, INPUT_STEPS, INPUT_FEATURES)
y_test = np.array(test_y_list)                  # (N_test,)

print(f"\nTrain: {X_train_all.shape[0]} cells,  Test: {X_test.shape[0]} cells")
print(f"Features: {FEATURE_COLS}")
print(f"Input shape per cell: ({INPUT_STEPS}, {INPUT_FEATURES})")

# ── Scale features (fit on train+test combined, same as your code) ─────
combined = np.concatenate([X_train_all, X_test], axis=0)
flat = combined.reshape(-1, INPUT_FEATURES)

# keep cap_ocv_dis (col 0) unscaled, scale the rest
unchanged = flat[:, [0]]
to_scale = flat[:, 1:]
scaler = StandardScaler()
scaled_part = scaler.fit_transform(to_scale)
flat_scaled = np.concatenate([unchanged, scaled_part], axis=1)

combined_scaled = flat_scaled.reshape(combined.shape)
X_train_all = combined_scaled[: len(train_X_list)]
X_test = combined_scaled[len(train_X_list):]

# ── Log-transform target ───────────────────────────────────────────────
y_train_all = np.log(y_train_all)
y_test = np.log(y_test)

# ── Train / val split ──────────────────────────────────────────────────
if USE_VALIDATION and len(X_train_all) > 10:
    X_train, y_train, X_val, y_val = create_balanced_val_split(
        X_train_all, y_train_all, val_fraction=VALIDATION_SPLIT, bins=min(3, len(X_train_all) // 3)
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

    loss_list, val_loss_list = [], []
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
                    torch.save(model.state_dict(), "savemodel/best_model_interp.pth")
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
    if USE_VALIDATION and os.path.exists("savemodel/best_model_interp.pth"):
        model.load_state_dict(torch.load("savemodel/best_model_interp.pth"))

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy().reshape(-1)
    y_pred_list.append(y_pred)

# ── Results ─────────────────────────────────────────────────────────────
y_pred_mean = np.mean(y_pred_list, axis=0)

print("\n" + "=" * 70)
print("RESULTS (log space)")
print("=" * 70)
for name, true, pred in zip(test_names, y_test, y_pred_mean):
    print(f"  {name:30s}  true={true:.4f}  pred={pred:.4f}")

print(f"\n  MAPE (log):  {np.mean(np.abs((y_test - y_pred_mean) / y_test)) * 100:.2f}%")

# Back to original scale (steps)
y_test_steps = np.exp(y_test)
y_pred_steps = np.exp(y_pred_mean)

print("\nRESULTS (original step scale)")
print("=" * 70)
for name, true, pred in zip(test_names, y_test_steps, y_pred_steps):
    print(f"  {name:30s}  true={true:.1f}  pred={pred:.1f}")

rmse = np.sqrt(np.mean((y_test_steps - y_pred_steps) ** 2))
mape = np.mean(np.abs((y_test_steps - y_pred_steps) / y_test_steps)) * 100
print(f"\n  RMSE (steps): {rmse:.2f}")
print(f"  MAPE (steps): {mape:.2f}%")

# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: bar chart true vs predicted
ax1 = axes[0]
x_pos = np.arange(len(test_names))
width = 0.35
ax1.bar(x_pos - width / 2, y_test_steps, width, label="True", color="steelblue", alpha=0.8)
ax1.bar(x_pos + width / 2, y_pred_steps, width, label="Predicted", color="coral", alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([n.replace("SPEED_LW_", "") for n in test_names], rotation=15)
ax1.set_ylabel("Steps to SOH target")
ax1.set_title(f"True vs Predicted steps  (RMSE={rmse:.1f}, MAPE={mape:.1f}%)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: ensemble spread
ax2 = axes[1]
preds_steps = np.exp(np.array(y_pred_list))  # (N_SEEDS, N_test)
for j, name in enumerate(test_names):
    short = name.replace("SPEED_LW_", "")
    ax2.scatter(np.full(N_SEEDS, j), preds_steps[:, j], alpha=0.4, s=20, color="coral")
    ax2.scatter(j, y_test_steps[j], marker="*", s=200, color="steelblue", zorder=5)
ax2.set_xticks(range(len(test_names)))
ax2.set_xticklabels([n.replace("SPEED_LW_", "") for n in test_names], rotation=15)
ax2.set_ylabel("Steps to SOH target")
ax2.set_title("Ensemble predictions (dots) vs true (stars)")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()