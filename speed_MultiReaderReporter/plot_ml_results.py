# speed_MultiReaderReporter/plot_ml_results.py
"""Figures for the ML condition-attribution results, publication oriented.

Consumes the two tables written by ml_condition_influence.py:
    ml_model_scores.csv            one row per (dataset, target, model)
    ml_permutation_importance.csv  one row per (dataset, target, model, condition)

Produces, per dataset:

  <ds>_model_scores.png
      cross-validated R2 per target and model, with the permutation p annotated.
      A bar is filled only where the model is significant AFTER multiple-testing
      correction; everything else is drawn hollow. The zero line is the "predict
      the mean" baseline - bars left of it are worse than useless.

  <ds>_importance.png
      permutation importance (mean +/- sd over repeats) for the models that
      survived correction, coloured by the SIGN of the Spearman correlation so
      the direction of each effect is readable.

and one overview figure across datasets.

MULTIPLE TESTING. Many (target, model) pairs are tested, so raw p values
overstate significance. This script applies Benjamini-Hochberg FDR across all
rows of ml_model_scores.csv and reports q values; `significant_fdr` uses q.

Usage:
    python speed_MultiReaderReporter/plot_ml_results.py
    python speed_MultiReaderReporter/plot_ml_results.py --src E:/download/ml_condition_influence
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POS, NEG, DEAD = "#c1443c", "#3b6ea5", "#b8b8b8"

def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        i = n - rank + 1
        prev = min(prev, p[idx] * n / i)
        q[idx] = prev
    return q

def plot_scores(ds: str, s: pd.DataFrame, dest: Path):
    s = s.sort_values(["target", "model"]).reset_index(drop=True)
    lab = [f"{r.target}  [{r.model}]" for r in s.itertuples()]
    y = np.arange(len(s))
    colours = [POS if sig else DEAD for sig in s.significant_fdr]

    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(s) + 2.0))
    ax.barh(y, s.cv_r2, color=colours, edgecolor="k", linewidth=0.5)
    for i, r in enumerate(s.itertuples()):
        x = r.cv_r2
        ax.text(x + (0.012 if x >= 0 else -0.012), i,
                f"p={r.p_perm:.3f}  q={r.q_fdr:.3f}",
                va="center", ha="left" if x >= 0 else "right", fontsize=7.5)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y, lab, fontsize=8)
    ax.set_xlabel("cross-validated $R^2$ on held-out cells")
    n = int(s.n_cells.max()); cvn = s.cv.iloc[0]
    ax.set_title(f"{ds}: model performance   (n={n} cells, {cvn}, "
                 f"{'filled' } = significant after BH-FDR)", fontsize=10)
    lo = min(-0.05, float(s.cv_r2.min()) - 0.1)
    ax.set_xlim(lo, max(0.6, float(s.cv_r2.max()) + 0.28))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = dest / f"{ds}_model_scores.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[OK] {p}")

def plot_importance(ds: str, imp: pd.DataFrame, dest: Path):
    keep = imp[imp.significant_fdr == 1]
    if keep.empty:
        print(f"[--] {ds}: nothing survived FDR; no importance figure")
        return
    keep = keep.sort_values(["target", "model", "importance"])
    groups = list(keep.groupby(["target", "model"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.3 * len(groups), 3.6),
                             squeeze=False)
    for k, ((tgt, mdl), g) in enumerate(groups):
        ax = axes[0][k]
        g = g.sort_values("importance")
        cols = [POS if d == "+" else NEG for d in g.direction]
        ax.barh(np.arange(len(g)), g.importance, xerr=g.importance_sd,
                color=cols, edgecolor="k", linewidth=0.5,
                error_kw=dict(lw=1, capsize=3))
        ax.set_yticks(np.arange(len(g)), g.condition, fontsize=9)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("permutation importance\n($\\Delta R^2$ when shuffled)", fontsize=8)
        ax.set_title(f"{tgt}\n{mdl}, cv $R^2$={g.cv_r2.iloc[0]:.2f}, "
                     f"q={g.q_fdr.iloc[0]:.3f}", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=POS, edgecolor="k", label="higher condition -> higher target"),
                        Patch(facecolor=NEG, edgecolor="k", label="higher condition -> lower target")],
               loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"{ds}: which condition carries the signal", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    p = dest / f"{ds}_importance.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[OK] {p}")

def plot_overview(S: pd.DataFrame, dest: Path):
    best = (S.sort_values("cv_r2", ascending=False)
              .groupby(["dataset", "target"], as_index=False).first())
    best = best.sort_values(["dataset", "cv_r2"])
    lab = [f"{r.dataset} | {r.target}" for r in best.itertuples()]
    colours = [POS if s else DEAD for s in best.significant_fdr]
    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(best) + 2))
    ax.barh(np.arange(len(best)), best.cv_r2, color=colours, edgecolor="k", linewidth=0.5)
    for i, r in enumerate(best.itertuples()):
        ax.text(r.cv_r2 + 0.012, i, f"{r.model}  n={r.n_cells}  q={r.q_fdr:.3f}",
                va="center", fontsize=7.5)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(np.arange(len(best)), lab, fontsize=8)
    ax.set_xlabel("best cross-validated $R^2$ per target")
    ax.set_title("Condition -> ageing: best model per target, all datasets\n"
                 "filled = significant after Benjamini-Hochberg FDR", fontsize=10)
    ax.set_xlim(min(-0.05, float(best.cv_r2.min()) - 0.1), float(best.cv_r2.max()) + 0.35)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = dest / "overview_model_scores.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[OK] {p}")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("E:/download/ml_condition_influence"))
    ap.add_argument("--dest", type=Path, default=None)
    args = ap.parse_args()
    dest = args.dest or args.src
    dest.mkdir(parents=True, exist_ok=True)

    S = pd.read_csv(args.src / "ml_model_scores.csv")
    I = pd.read_csv(args.src / "ml_permutation_importance.csv")

    # BH-FDR across every model fit, then propagate to the importance rows
    S["q_fdr"] = bh_fdr(S.p_perm.to_numpy())
    S["significant_fdr"] = ((S.q_fdr < 0.05) & (S.cv_r2 > 0)).astype(int)
    S.to_csv(dest / "ml_model_scores_fdr.csv", index=False)
    I = I.merge(S[["dataset", "target", "model", "q_fdr", "significant_fdr"]],
                on=["dataset", "target", "model"], how="left")
    I.to_csv(dest / "ml_permutation_importance_fdr.csv", index=False)

    print(f"{len(S)} model fits, "
          f"{int((S.p_perm < 0.05).sum())} at raw p<0.05, "
          f"{int(S.significant_fdr.sum())} surviving BH-FDR\n")

    for ds, s in S.groupby("dataset"):
        plot_scores(ds, s, dest)
        plot_importance(ds, I[I.dataset == ds], dest)
    plot_overview(S, dest)

    print("\nsurviving FDR:")
    sig = S[S.significant_fdr == 1].sort_values("cv_r2", ascending=False)
    print(sig[["dataset", "target", "model", "n_cells", "cv_r2", "p_perm", "q_fdr"]]
          .to_string(index=False) if not sig.empty else "  none")

if __name__ == "__main__":
    main()
