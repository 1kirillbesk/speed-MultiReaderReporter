import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


# ----------------------------
# 1) Ground-truth "original" exponential degradation
# ----------------------------
k_true = 0.25  # bigger => more curvature
def soh_true(t):
    t = np.asarray(t, dtype=float)
    return np.exp(-k_true * t)


# ----------------------------
# 2) Sample ONLY 5 discretized points from the true curve
# ----------------------------
t0, t1 = 0.0, 8.0
t_pts = np.array([0, 1, 2, 4, 8], dtype=float)  # 5 points (you can change these)
soh_pts = soh_true(t_pts)                        # these are your "measured" points

# Optional: add measurement noise to simulate reality
ADD_NOISE = False
if ADD_NOISE:
    rng = np.random.default_rng(0)
    soh_pts = soh_pts + rng.normal(0.0, 0.005, size=len(soh_pts))
    soh_pts = np.clip(soh_pts, 1e-6, 1.0)


# ----------------------------
# 3) Build interpolation models from the 5 points
# ----------------------------
def soh_linear(t):
    return np.interp(t, t_pts, soh_pts)

pchip = PchipInterpolator(t_pts, soh_pts, extrapolate=False)
def soh_pchip(t):
    return pchip(t)


# ----------------------------
# 4) Compare curves vs ground-truth on a dense grid
# ----------------------------
t_dense = np.linspace(t0, t1, 2000)
y_true  = soh_true(t_dense)
y_lin   = soh_linear(t_dense)
y_pchip = soh_pchip(t_dense)

rmse_lin = np.sqrt(np.mean((y_lin - y_true) ** 2))
rmse_pch = np.sqrt(np.mean((y_pchip - y_true) ** 2))

print("=== Curve error vs TRUE exponential (dense grid) ===")
print(f"RMSE Linear = {rmse_lin:.6f}")
print(f"RMSE PCHIP  = {rmse_pch:.6f}")


# ----------------------------
# 5) Compare t at a SOH threshold (crossing time)
# ----------------------------
target = 0.90

# True crossing time: SOH(t)=exp(-k t) => t = -ln(SOH)/k
t_true_cross = -np.log(target) / k_true

def first_crossing_time(f, t0, t1, target, ngrid=4000):
    """
    Find first t where f(t) <= target using grid bracket + brentq.
    """
    ts = np.linspace(t0, t1, ngrid)
    ys = f(ts)

    idx = np.where(ys <= target)[0]
    if len(idx) == 0:
        return np.nan
    j = int(idx[0])
    if j == 0:
        return float(ts[0])

    a, b = float(ts[j - 1]), float(ts[j])
    fa = float(f(a) - target)
    fb = float(f(b) - target)
    if fa * fb > 0:
        return float(ts[j])  # fallback

    return float(brentq(lambda x: float(f(x) - target), a, b))

t_lin_cross   = first_crossing_time(soh_linear, t0, t1, target)
t_pchip_cross = first_crossing_time(soh_pchip,  t0, t1, target)

print("\n=== t@SOH comparison ===")
print(f"True t@{target:.2f}   = {t_true_cross:.6f}")
print(f"Linear t@{target:.2f} = {t_lin_cross:.6f}   (error {t_lin_cross - t_true_cross:+.6f})")
print(f"PCHIP  t@{target:.2f} = {t_pchip_cross:.6f} (error {t_pchip_cross - t_true_cross:+.6f})")


# ----------------------------
# 6) Plot all: TRUE vs Linear vs PCHIP + the 5 points
# ----------------------------
plt.figure()
plt.scatter(t_pts, soh_pts, label="5 sampled points", zorder=3)
plt.plot(t_dense, y_true,  label=f"TRUE exp: exp(-{k_true} t)")
plt.plot(t_dense, y_lin,   label="Linear interpolation")
plt.plot(t_dense, y_pchip, label="PCHIP interpolation")

plt.axhline(target, linestyle="--", linewidth=1)
plt.axvline(t_true_cross, linestyle="--", linewidth=1)
plt.axvline(t_lin_cross, linestyle="--", linewidth=1)
plt.axvline(t_pchip_cross, linestyle="--", linewidth=1)

plt.xlabel("time")
plt.ylabel("SOH")
plt.ylim(0.0, 1.02)
plt.title("TRUE exponential vs Linear vs PCHIP (built from only 5 points)")
plt.legend()
plt.tight_layout()
plt.show()
