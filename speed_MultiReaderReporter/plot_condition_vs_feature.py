# speed_MultiReaderReporter/plot_condition_vs_feature.py
"""Plot one feature against every experimental condition, at a common SoH.

Reads <family>_features_at_soh<NN>.csv (written by
analyze_condition_influence_{jgne,lwhk}.py) and draws one panel per condition:
a point per cell, condition on x, the chosen feature on y, with a least-squares
line and the Spearman rho/p in the panel title.

Because the table holds one interpolated row per cell, every point is one cell
at exactly the requested SoH - so the comparison is like-for-like ageing.

Usage:
    python speed_MultiReaderReporter/plot_condition_vs_feature.py --family jgne
    python speed_MultiReaderReporter/plot_condition_vs_feature.py --family jgne \
        --feature mean_d_dqdv_m_c --target-soh 0.995 --label-cells
    python speed_MultiReaderReporter/plot_condition_vs_feature.py --family lwhk \
        --feature mean_d_dqdv_m_d --color-by fec
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path("E:/download/jgne/out_jgne")
DEFAULT_FEATURE = "mean_d_dqdv_m_c"

FAMILY_CONDITIONS = {
    "jgne": ["soc_start", "soc_end", "pause_h", "has_pulse", "low_soc", "temp_C", "c_rate_chg", "c_rate_dchg"],
    "lwhk": ["fec", "rest_min", "ref"],
}

def soh_tag(target_soh: float) -> str:
    return f"{target_soh:.3f}".replace("0.", "").rstrip("0") or "0"

def load_table(out_dir: Path, family: str, target_soh: float,
               prefix: str | None = None) -> pd.DataFrame:
    path = out_dir / f"{prefix or family}_features_at_soh{soh_tag(target_soh)}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run analyze_condition_influence_{family}.py "
            f"--target-soh {target_soh} first"
        )
    print(f"source : {path}")
    return pd.read_csv(path)

# Bookkeeping columns that are never a response variable.
SKIP_AS_FEATURE = {"cell", "label", "n_checkups", "soh_min", "soh_max",
                   "n_files", "first_test", "last_test", "has_feature_csv",
                   "pause_raw", "pause_h_2", "test_date", "ref_index"}

def _rho_matrix(df, features, conditions, spearmanr):
    """Spearman rho and p for every feature x condition pair."""
    rho = np.full((len(features), len(conditions)), np.nan)
    pval = np.full_like(rho, np.nan)
    for i, feat in enumerate(features):
        y_all = pd.to_numeric(df[feat], errors="coerce")
        for j, cond in enumerate(conditions):
            x_all = pd.to_numeric(df[cond], errors="coerce")
            m = x_all.notna() & y_all.notna()
            x, y = x_all[m], y_all[m]
            if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
                continue
            r, p = spearmanr(x, y)
            rho[i, j], pval[i, j] = r, p
    return rho, pval

def write_heatmap(df, features, conditions, args, dest, spearmanr):
    """Spearman rho over feature x condition; * marks p < 0.05."""
    rho, pval = _rho_matrix(df, features, conditions, spearmanr)

    fig, ax = plt.subplots(figsize=(1.5 + 1.15 * len(conditions),
                                    1.6 + 0.30 * len(features)))
    im = ax.imshow(rho, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(conditions)), conditions, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(features)), features, fontsize=8)
    for i in range(len(features)):
        for j in range(len(conditions)):
            if np.isnan(rho[i, j]):
                continue
            star = "*" if pval[i, j] < 0.05 else ""
            ax.text(j, i, f"{rho[i, j]:+.2f}{star}", ha="center", va="center",
                    fontsize=7, color="black" if abs(rho[i, j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax, label="Spearman rho", fraction=0.04)
    ax.set_title(f"{args.family.upper()} at SoH = {args.target_soh}\n"
                 f"* marks p < 0.05 (uncorrected)", fontsize=10)
    fig.tight_layout()

    png = dest / f"{args.family}_rho_heatmap_soh{soh_tag(args.target_soh)}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    long = pd.DataFrame([
        {"feature": f, "condition": c, "spearman_r": rho[i, j], "p": pval[i, j]}
        for i, f in enumerate(features) for j, c in enumerate(conditions)
    ]).dropna(subset=["spearman_r"])
    csv = dest / f"{args.family}_rho_heatmap_soh{soh_tag(args.target_soh)}.csv"
    long.reindex(long.spearman_r.abs().sort_values(ascending=False).index).to_csv(csv, index=False)
    print(f"[OK] {png}")
    print(f"[OK] {csv}")

    sig = long[long.p < 0.05]
    print(f"\n{len(sig)}/{len(long)} pair(s) at p < 0.05 (uncorrected; "
          f"~{0.05 * len(long):.0f} expected by chance)")
    if not sig.empty:
        print(sig.reindex(sig.spearman_r.abs().sort_values(ascending=False).index)
                 .head(15).to_string(index=False))

def plot_grid(df, features, conditions, args, dest, spearmanr):
    """One panel per (feature, condition): features down, conditions across."""
    nrow, ncol = len(features), len(conditions)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.7 * nrow),
                             squeeze=False)
    for i, feat in enumerate(features):
        y_all = pd.to_numeric(df[feat], errors="coerce")
        for j, cond in enumerate(conditions):
            ax = axes[i][j]
            x_all = pd.to_numeric(df[cond], errors="coerce")
            m = x_all.notna() & y_all.notna()
            x, y = x_all[m], y_all[m]
            ax.scatter(x, y, s=26, color="#3b6ea5", edgecolor="k",
                       linewidth=0.3, zorder=3)
            if len(x) >= 3 and x.nunique() >= 2 and y.nunique() >= 2:
                r, p = spearmanr(x, y)
                b, a = np.polyfit(x.astype(float), y.astype(float), 1)
                xs = np.linspace(x.min(), x.max(), 30)
                ax.plot(xs, a + b * xs, color="#c1443c", lw=1.1, zorder=2)
                ax.set_title(f"rho={r:+.2f}, p={p:.3f}", fontsize=7,
                             color="#b00" if p < 0.05 else "black")
            ax.grid(alpha=0.22, zorder=0)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(feat, fontsize=7)
            if i == nrow - 1:
                ax.set_xlabel(cond, fontsize=8)
    fig.suptitle(f"{args.family.upper()}: features vs conditions at "
                 f"SoH = {args.target_soh}  ({len(df)} cells)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    png = dest / f"{args.family}_features_vs_conditions_soh{soh_tag(args.target_soh)}.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[OK] {png}")

def usable_conditions(df: pd.DataFrame, conditions: list[str]) -> list[str]:
    out = []
    for c in conditions:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 3 and s.nunique(dropna=True) >= 2:
            out.append(c)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILY_CONDITIONS), required=True)
    ap.add_argument("--feature", default=DEFAULT_FEATURE,
                    help=f"feature(s) on the y axis, comma separated "
                         f"(default: {DEFAULT_FEATURE})")
    ap.add_argument("--all-features", action="store_true",
                    help="use every numeric feature in the table")
    ap.add_argument("--heatmap", action="store_true",
                    help="also write a Spearman rho heatmap over feature x condition")
    ap.add_argument("--target-soh", type=float, default=0.995)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dest", type=Path, default=None, help="output folder (default: --out-dir)")
    ap.add_argument("--color-by", default=None,
                    help="condition used to colour the points (default: none)")
    ap.add_argument("--label-cells", action="store_true",
                    help="annotate each point with its cell number")
    ap.add_argument("--prefix", default=None,
                    help="input/output filename prefix (default: the family name); "
                         "use lwhk_c2 for the C/2 discharge table")
    ap.add_argument("--list-features", action="store_true",
                    help="print the available feature columns and exit")
    args = ap.parse_args()

    df = load_table(args.out_dir, args.family, args.target_soh, args.prefix)
    # conditions come from the real family; the prefix only renames the outputs
    cond_list = FAMILY_CONDITIONS[args.family]
    args.family = args.prefix or args.family
    conditions = usable_conditions(df, cond_list)

    skip = set(cond_list) | SKIP_AS_FEATURE
    available = [c for c in df.columns
                 if c not in skip and pd.api.types.is_numeric_dtype(df[c])
                 and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 3]

    if args.list_features:
        print("available features:")
        for f in available:
            print("   ", f)
        return

    if args.all_features:
        features = available
    else:
        features = [f.strip() for f in args.feature.split(",") if f.strip()]
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"[ERROR] not a column: {missing}. "
                  f"Run with --list-features to see the options.")
            return

    features = [f for f in features
                if pd.to_numeric(df[f], errors="coerce").notna().sum() >= 3]
    if not features:
        print("[ERROR] no feature has enough usable values.")
        return

    from scipy.stats import spearmanr

    dest = args.dest or args.out_dir
    dest.mkdir(parents=True, exist_ok=True)

    if args.heatmap or len(features) > 1:
        write_heatmap(df, features, conditions, args, dest, spearmanr)

    if len(features) > 1:
        plot_grid(df, features, conditions, args, dest, spearmanr)
        return

    y_all = pd.to_numeric(df[features[0]], errors="coerce")
    args.feature = features[0]

    n = len(conditions)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), squeeze=False)

    colour_vals = None
    if args.color_by and args.color_by in df.columns:
        colour_vals = pd.to_numeric(df[args.color_by], errors="coerce")

    for k, cond in enumerate(conditions):
        ax = axes[k // ncol][k % ncol]
        x_all = pd.to_numeric(df[cond], errors="coerce")
        m = x_all.notna() & y_all.notna()
        x, y = x_all[m], y_all[m]

        if colour_vals is not None:
            sc = ax.scatter(x, y, c=colour_vals[m], cmap="viridis",
                            s=55, edgecolor="k", linewidth=0.4, zorder=3)
            fig.colorbar(sc, ax=ax, label=args.color_by, fraction=0.046)
        else:
            ax.scatter(x, y, s=55, color="#3b6ea5", edgecolor="k",
                       linewidth=0.4, zorder=3)

        if args.label_cells:
            for xi, yi, name in zip(x, y, df.loc[m, "cell"]):
                ax.annotate(str(name).split("_")[-1], (xi, yi), fontsize=7,
                            xytext=(3, 3), textcoords="offset points", alpha=0.75)

        title = f"{cond}   (n={len(x)})"
        if len(x) >= 3 and x.nunique() >= 2 and y.nunique() >= 2:
            rho, p = spearmanr(x, y)
            title = f"{cond}   rho={rho:+.2f}, p={p:.3f}, n={len(x)}"
            if x.nunique() >= 2:
                b, a = np.polyfit(x.astype(float), y.astype(float), 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, a + b * xs, color="#c1443c", lw=1.3, zorder=2)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel(cond)
        ax.set_ylabel(args.feature)
        ax.grid(alpha=0.25, zorder=0)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(f"{args.family.upper()}: {args.feature} vs experimental conditions "
                 f"at SoH = {args.target_soh}  ({len(df)} cells)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = dest / f"{args.family}_{args.feature}_vs_conditions_soh{soh_tag(args.target_soh)}.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[OK] {out_png}")

if __name__ == "__main__":
    main()
