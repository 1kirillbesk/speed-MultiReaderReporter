"""
combined_cost_search.py
─────────────────────────────────────────────────────────────────────────────
Drop-in replacement for the surrogate search block in main().

Changes vs original:
  - Trains TWO XGB models on the SAME Monte Carlo sample grid:
        model_dist  : exp_conds  →  dist_feat          (minimize)
        model_var   : exp_conds  →  var_dQ_c @ 500k    (maximize)
  - Combined cost per candidate:
        cost = w1 * norm(pred_dist_feat) - w2 * norm(pred_var_dQc)
    where both predictions are min-max normalized to [0,1] across the
    candidate pool so the weights are dimensionless and comparable.
  - Returns top_k candidates ranked by combined cost (ascending).

Usage: replace the "Train surrogate model" block in main() with the
       call to  run_combined_search()  shown at the bottom of this file.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import xgboost as xgb

try:
    import shap
    _HAVE_SHAP = True
except ImportError:
    shap = None
    _HAVE_SHAP = False


RAW_CONDS  = ["soc_start", "soc_end", "c_rate_chg", "c_rate_dchg", "temp"]
FEAT_CONDS = ["soc", "dod", "c_rate_chg", "c_rate_dchg", "temp"]


# ─────────────────────────────────────────────────────────────────────────────
# shared feature engineering  (identical to make_features_from_raw in main.py)
# ─────────────────────────────────────────────────────────────────────────────
def _make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = df[RAW_CONDS].copy().apply(pd.to_numeric, errors="coerce")
    X["soc"] = 0.5 * (X["soc_start"] + X["soc_end"])
    X["dod"] = X["soc_end"] - X["soc_start"]
    X.loc[X["c_rate_chg"] == 15, "c_rate_chg"] = 1.5
    X = X.drop(columns=["soc_start", "soc_end"])
    return X[FEAT_CONDS].copy(), FEAT_CONDS


# ─────────────────────────────────────────────────────────────────────────────
# helper: min-max normalize a 1-D array to [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — already exists in main.py as train_xgb_no_val_and_shap
#           reproduced here so this file is self-contained, but you can
#           reuse the one from main.py if you prefer.
# ─────────────────────────────────────────────────────────────────────────────
def train_model_dist_feat(
    df_exp: pd.DataFrame,
    dist_df: pd.DataFrame,
    ref_names: list[str],
    *,
    random_state: int = 42,
    shap_max_display: int = 20,
) -> xgb.XGBRegressor:
    """
    XGB: exp_conds → dist_feat (distance-to-nearest-ref).
    Trained on ALL data (no val split) — same as original train_xgb_no_val_and_shap.
    """
    df = df_exp.merge(dist_df[["cell_name", "dist_feat"]], on="cell_name", how="inner")
    df = df[~df["cell_name"].isin(ref_names)].copy()

    y = pd.to_numeric(df["dist_feat"], errors="coerce")
    X_feat, feat_names = _make_features(df)

    ok = ~y.isna() & np.isfinite(X_feat.to_numpy()).all(axis=1)
    X_feat = X_feat.loc[ok].reset_index(drop=True)
    y = y.loc[ok].to_numpy(dtype=float)

    if len(X_feat) < 5:
        raise ValueError("Not enough rows to train dist_feat model.")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800, learning_rate=0.03, max_depth=4,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0,
        min_child_weight=1.0, gamma=0.0,
        tree_method="hist", random_state=random_state,
    )
    model.fit(X_feat, y)

    if _HAVE_SHAP:
        explainer = shap.KernelExplainer(
            lambda a: model.predict(np.asarray(a)), X_feat
        )
        sv = explainer.shap_values(X_feat)

        plt.figure()
        shap.summary_plot(sv, X_feat, show=False, max_display=shap_max_display)
        plt.title("SHAP beeswarm — dist_feat")
        plt.tight_layout()

        plt.figure()
        shap.summary_plot(sv, X_feat, plot_type="bar", show=False,
                          max_display=shap_max_display)
        plt.title("SHAP bar — dist_feat")
        plt.tight_layout()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — NEW: exp_conds → var_dQ_c @ 500k  (80/20 split + SHAP)
# ─────────────────────────────────────────────────────────────────────────────
def train_model_var_dqc(
    df_exp: pd.DataFrame,
    df_reg_table: pd.DataFrame,
    *,
    target_col: str = "var_dQ_c_at_thr500k",
    test_size: float = 0.30,
    random_state: int = 42,
    shap_max_display: int = 20,
) -> xgb.XGBRegressor:
    """
    XGB: exp_conds → var_dQ_c @ throughput=500k.
    Uses 70/30 train/test split with early stopping.
    """
    needed = ["cell_name"] + RAW_CONDS
    df = (
        df_exp[needed]
        .merge(df_reg_table[["cell_name", target_col]], on="cell_name", how="inner")
    )

    # strip bracket-encoded strings e.g. '[4.94e-3]'
    y = (
        df[target_col]
        .astype(str).str.strip().str.strip("[]")
        .pipe(pd.to_numeric, errors="coerce")
    )
    y = np.log(np.clip(np.abs(y), 1e-12, None))  # ← train on log scale
    X_feat, feat_names = _make_features(df)

    ok = ~y.isna() & np.isfinite(X_feat.to_numpy()).all(axis=1)
    X_feat = X_feat.loc[ok].reset_index(drop=True)
    y = y.loc[ok].to_numpy(dtype=float)

    n = len(y)
    if n < 10:
        raise ValueError(f"Only {n} valid rows — need at least 10 to train var_dQ_c model.")

    print(f"\n{'='*70}")
    print(f"  XGB model 2: exp_conds → {target_col}")
    print(f"  samples: {n}   features: {feat_names}")
    print(f"{'='*70}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feat, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800, learning_rate=0.03, max_depth=4,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0,
        min_child_weight=1.0, gamma=0.0,
        tree_method="hist", random_state=random_state,
        early_stopping_rounds=50, eval_metric="rmse",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    print(f"  Train  R²={r2_score(y_tr, y_pred_tr):.4f}   "
          f"MAE={mean_absolute_error(y_tr, y_pred_tr):.4g}")
    print(f"  Test   R²={r2_score(y_te, y_pred_te):.4f}   "
          f"MAE={mean_absolute_error(y_te, y_pred_te):.4g}")

    # parity plot
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_tr, y_pred_tr, s=20, alpha=0.6,
               label=f"train  R²={r2_score(y_tr, y_pred_tr):.3f}")
    ax.scatter(y_te, y_pred_te, s=30, alpha=0.9, marker="^",
               label=f"test   R²={r2_score(y_te, y_pred_te):.3f}")
    lims = [min(y.min(), y_pred_tr.min(), y_pred_te.min()),
            max(y.max(), y_pred_tr.max(), y_pred_te.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel(f"Actual  {target_col}")
    ax.set_ylabel(f"Predicted  {target_col}")
    ax.set_title(f"Parity — exp_conds → {target_col}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if _HAVE_SHAP:
        print("  Computing SHAP (KernelExplainer) for var_dQ_c model …")
        explainer = shap.KernelExplainer(
            lambda a: model.predict(np.asarray(a)), X_feat
        )
        sv = explainer.shap_values(X_feat)

        plt.figure()
        shap.summary_plot(sv, X_feat, show=False, max_display=shap_max_display)
        plt.title(f"SHAP beeswarm — {target_col}")
        plt.tight_layout()

        plt.figure()
        shap.summary_plot(sv, X_feat, plot_type="bar", show=False,
                          max_display=shap_max_display)
        plt.title(f"SHAP bar — {target_col}")
        plt.tight_layout()
        print("  SHAP done.")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Combined exhaustive search
# ─────────────────────────────────────────────────────────────────────────────
def exhaustive_best_conditions_combined(
    df_exp: pd.DataFrame,
    model_dist: xgb.XGBRegressor,
    model_var: xgb.XGBRegressor,
    ref_names: list[str],
    *,
    w1: float = 0.5,          # weight on normalized dist_feat  (minimize)
    w2: float = 0.5,          # weight on normalized var_dQ_c   (maximize → subtracted)
    n_samples_per_temp: int = 200,
    top_k: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate the same Monte Carlo candidate grid for BOTH models,
    predict with each, combine:

        cost = w1 * norm(pred_dist_feat) - w2 * norm(pred_var_dQc)

    Returns top_k rows sorted by cost ascending (best = lowest cost).

    Columns returned:
        soc_start, soc_end, c_rate_chg, c_rate_dchg, temp,
        soc, dod,
        pred_dist_feat, pred_var_dQc,
        norm_dist, norm_var,
        cost
    """
    rng = np.random.default_rng(random_state)

    allowed_temp      = np.array([15.0, 25.0, 40.0])
    allowed_soc_start = np.arange(0,  70, 10, dtype=float)
    allowed_soc_end   = np.arange(20, 110, 10, dtype=float)
    allowed_cur_cha   = np.arange(0.5, 1.75, 0.25, dtype=float)
    allowed_cur_dis   = np.arange(1.0, 3.25, 0.25, dtype=float)

    valid_pairs = np.array(
        [(s0, s1) for s0 in allowed_soc_start
                  for s1 in allowed_soc_end
                  if (s1 - s0) > 10],
        dtype=float,
    )

    # ── build candidate grid (same samples for both models) ──────────────────
    blocks = []
    for temp in allowed_temp:
        idx   = rng.integers(0, len(valid_pairs), size=n_samples_per_temp)
        block = pd.DataFrame({
            "soc_start":   valid_pairs[idx, 0],
            "soc_end":     valid_pairs[idx, 1],
            "c_rate_chg":  rng.choice(allowed_cur_cha,  size=n_samples_per_temp),
            "c_rate_dchg": rng.choice(allowed_cur_dis,  size=n_samples_per_temp),
            "temp":        np.full(n_samples_per_temp, temp),
        })
        blocks.append(block)

    X_mc_raw  = pd.concat(blocks, ignore_index=True)[RAW_CONDS].astype(float)
    X_mc_feat, _ = _make_features(X_mc_raw)

    # ── predictions ──────────────────────────────────────────────────────────
    pred_dist = model_dist.predict(X_mc_feat).astype(float)   # minimize
    pred_var  = model_var.predict(X_mc_feat).astype(float)
    pred_var = np.log(np.clip(np.abs(pred_var), 1e-12, None))

    # ── normalize to [0, 1] across the candidate pool ────────────────────────
    norm_dist = _minmax(pred_dist)          # 0 = best (smallest distance)
    norm_var  = _minmax(pred_var)           # 1 = best (largest var_dQ_c)

    # ── combined cost (lower = better) ───────────────────────────────────────
    cost = w1 * norm_dist - w2 * norm_var

    # ── assemble output ───────────────────────────────────────────────────────
    out = X_mc_raw.copy()
    out["soc"]            = X_mc_feat["soc"].to_numpy()
    out["dod"]            = X_mc_feat["dod"].to_numpy()
    out["pred_dist_feat"] = pred_dist
    out["pred_var_dQc"]   = pred_var
    out["norm_dist"]      = norm_dist
    out["norm_var"]       = norm_var
    out["cost"]           = cost

    best = out.nsmallest(top_k, "cost").reset_index(drop=True)

    print(f"\n{'='*90}")
    print(f"  Combined search  —  cost = {w1}*norm(dist_feat) - {w2}*norm(var_dQc)")
    print(f"  candidates: {len(out)}   top_k shown: {top_k}")
    print(f"{'='*90}")
    print(best[[
        "soc_start","soc_end","c_rate_chg","c_rate_dchg","temp",
        "pred_dist_feat","pred_var_dQc","norm_dist","norm_var","cost"
    ]].to_string(index=False))

    # ── scatter plot: dist vs var_dQ_c, highlight top_k ──────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pred_dist, pred_var, s=12, alpha=0.3, color="steelblue", label="all candidates")
    ax.scatter(
        best["pred_dist_feat"], best["pred_var_dQc"],
        s=80, alpha=0.95, color="red", edgecolors="k", linewidths=0.6,
        zorder=5, label=f"top {top_k} (lowest cost)",
    )
    ax.set_xlabel("pred_dist_feat  (minimize →)")
    ax.set_ylabel("pred_var_dQc  (← maximize)")
    ax.set_title(
        f"Combined objective space\n"
        f"cost = {w1}·norm(dist) − {w2}·norm(var_dQc)"
    )
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Top-level convenience wrapper — call this from main()
# ─────────────────────────────────────────────────────────────────────────────
def run_combined_search(
    df_exp: pd.DataFrame,
    dist_df: pd.DataFrame,
    df_reg_table: pd.DataFrame,
    ref_names: list[str],
    *,
    w1: float = 0.5,
    w2: float = 0.5,
    n_samples_per_temp: int = 200,
    top_k: int = 8,
    random_state: int = 42,
    target_col: str = "var_dQ_c_at_thr500k",
) -> pd.DataFrame:
    """
    Convenience wrapper that trains both models and runs the combined search.

    Parameters
    ----------
    df_exp        : exp condition rows (already built in main loop)
    dist_df       : output of the distance-to-ref computation in main()
    df_reg_table  : output of build_regression_table_cap93_and_var_at_thr (all cells)
    ref_names     : list of reference cell names
    w1            : weight for dist_feat term  (default 0.5)
    w2            : weight for var_dQ_c term   (default 0.5)

    Returns
    -------
    DataFrame of top_k candidates with all cost columns.
    """
    print("\n" + "="*90)
    print("Training Model 1: exp_conds → dist_feat")
    print("="*90)
    model_dist = train_model_dist_feat(
        df_exp, dist_df, ref_names, random_state=random_state
    )

    print("\n" + "="*90)
    print("Training Model 2: exp_conds → var_dQ_c @ 500k")
    print("="*90)
    model_var = train_model_var_dqc(
        df_exp, df_reg_table,
        target_col=target_col, random_state=random_state
    )

    best = exhaustive_best_conditions_combined(
        df_exp, model_dist, model_var, ref_names,
        w1=w1, w2=w2,
        n_samples_per_temp=n_samples_per_temp,
        top_k=top_k,
        random_state=random_state,
    )

    plt.show()
    return best


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO INTEGRATE INTO main()
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. At the top of main.py add:
#       from combined_cost_search import run_combined_search
#
# 2. You already build df_reg_table (called df_all) with:
#       df_all = build_regression_table_cap93_and_var_at_thr(
#           all_cells, traj_by_cell_reg, ...
#       )
#    If that call isn't in this version of main(), add it before the search.
#
# 3. Replace the existing surrogate block:
#
#       # OLD ─────────────────────────────────────────────────────────────────
#       model, explainer, X_feat, y, feat_names = train_xgb_no_val_and_shap(...)
#       best = exhaustive_best_conditions_for_distance(...)
#
#       # NEW ─────────────────────────────────────────────────────────────────
#       best = run_combined_search(
#           df_exp        = df_exp,
#           dist_df       = dist_df,
#           df_reg_table  = df_all,        # needs build_regression_table call
#           ref_names     = REF_NAMES,
#           w1            = 0.5,           # tweak as needed
#           w2            = 0.5,
#           n_samples_per_temp = 200,
#           top_k         = 8,
#       )