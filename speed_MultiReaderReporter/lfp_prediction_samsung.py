# speed_MultiReaderReporter/lfp_prediction_samsung.py
"""Run the LFP lifetime-prediction + augmentation pipeline on the Samsung 35E cells.

Same machinery as the LW/LFP work - lfp_prediction_common.run_lfp_prediction_pipeline
from lfp_prediction_common_bundle/ - pointed at the Model2Life + MobilEM NMC cells
instead. Nothing in the model is chemistry-specific; what changes is the data and
four settings that the LFP defaults get wrong for this dataset:

  throughput   the pipeline left throughput_cum at 0.0 for all 101 cells, which
               makes load_lfp_cell_frames() drop every one of them as
               `invalid_target_throughput`. Run
               prepare_samsung_prediction_input.py first - it rebuilds the column
               from the cycling reports into `prediction_feature/`.

  input_steps  --input-steps is only a FALLBACK. With an SoH anchor,
               resolve_week_step_count() overrides it with the longest
               pre-anchor run over all cells (13 here) and
               build_sequence_for_frame() pads every shorter cell up to that
               length. So no cell is dropped for being short - but the Samsung
               cells have a MEDIAN of only 5 checkups at or above SoH 0.975,
               with a minimum of 2, so a lot of these sequences are mostly
               padding. That is the main weakness of this transfer and the
               reason the per-cell errors below are uneven.

  test cells   DEFAULT_TEST_NAMES is the three SPEED_LW_reference cells, which do
               not exist here. Test cells are instead drawn deterministically,
               stratified by cycling DOD so the held-out set spans the same
               conditions as the training set (--seed fixes the draw), or named
               outright with --test-cells.

  anchor       DEFAULT_ANCHOR_SOH 0.975 / DEFAULT_TARGET_SOH 0.95 both stay -
               all 101 cells reach 0.975 and 100 reach 0.95 - but these cells age
               far deeper (median final SoH 0.52), so a lower target is available
               and gives a longer extrapolation: try --target-soh 0.90.

The task is unchanged: from the first `input_steps` checkups up to the anchor
SoH, predict how long the cell takes to reach the target SoH - in weeks AND in
throughput, which are trained as two separate heads.

Usage:
    python speed_MultiReaderReporter/prepare_samsung_prediction_input.py
    python speed_MultiReaderReporter/lfp_prediction_samsung.py
    python speed_MultiReaderReporter/lfp_prediction_samsung.py --augmentation normal
    python speed_MultiReaderReporter/lfp_prediction_samsung.py --target-soh 0.90 --plots
    python speed_MultiReaderReporter/lfp_prediction_samsung.py \
        --test-cells LBE_Samsung_35E_127,LBE_Samsung_35E_155
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # must precede the bundle import, which pulls in pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = REPO_ROOT / "lfp_prediction_common_bundle" / "speed_MultiReaderReporter"
DEFAULT_OUT_DIR = Path("E:/download/samsung35e_combined")
DEFAULT_FEATURE_SUBDIR = "prediction_feature"
DEFAULT_DEST_SUBDIR = "prediction"
DEFAULT_COND_CSV = "condition_overview_samsung35e.csv"


def import_pipeline():
    """lfp_prediction_common lives only in the bundle on this branch."""
    if not BUNDLE_DIR.is_dir():
        raise SystemExit(f"[ERROR] bundle not found: {BUNDLE_DIR}\n"
                         f"        lfp_prediction_common.py is not on this branch "
                         f"outside lfp_prediction_common_bundle/.")
    if str(BUNDLE_DIR) not in sys.path:
        sys.path.insert(0, str(BUNDLE_DIR))
    import lfp_prediction_common as lpc
    return lpc


def weeks_to_target(feature_dir: Path, cell: str, target_soh: float,
                    cap_col: str = "cap_ocv_dis") -> float:
    """Weeks until this cell first reaches `target_soh`, by interpolation.

    Same quantity the model predicts, recomputed here so the test set can be
    picked BEFORE the pipeline runs. Monotone-decreasing SoH, as the pipeline
    enforces, so the inverse interpolation is well defined.
    """
    df = pd.read_csv(feature_dir / f"{cell}.csv", usecols=lambda c: c in ("CU_time", cap_col))
    if cap_col not in df.columns or "CU_time" not in df.columns:
        return np.nan
    t = pd.to_datetime(df["CU_time"], errors="coerce")
    cap = pd.to_numeric(df[cap_col], errors="coerce")
    ok = t.notna() & cap.notna() & (cap > 0)
    if ok.sum() < 2:
        return np.nan
    t, cap = t[ok].reset_index(drop=True), cap[ok].to_numpy(float)
    weeks = (t - t.iloc[0]).dt.total_seconds().to_numpy() / (7 * 24 * 3600)
    soh = np.minimum.accumulate(np.clip(cap / cap[0], 0.0, 1.05))
    if float(target_soh) < soh.min() or float(target_soh) > soh.max():
        return np.nan
    unique_soh, idx = np.unique(soh[::-1], return_index=True)
    return float(np.interp(float(target_soh), unique_soh, weeks[::-1][idx]))


def choose_extreme_cells(cells: list[str], feature_dir: Path, target_soh: float,
                         n_test: int, slowest: bool) -> list[str]:
    """The n slowest- (or fastest-) ageing cells, as the held-out reference set.

    This makes the task EXTRAPOLATION rather than interpolation: the model never
    sees a cell that lives as long as the ones it is asked to predict, so expect
    materially worse numbers than the stratified split. That is the point - it
    is the honest test of whether the features carry ageing rate, and it mirrors
    the LFP setup, where the three held-out SPEED_LW_reference cells were the
    slow-ageing references.
    """
    lives = {cell: weeks_to_target(feature_dir, cell, target_soh) for cell in cells}
    usable = {c: v for c, v in lives.items() if np.isfinite(v)}
    if not usable:
        raise SystemExit(f"[ERROR] no cell reaches SoH {target_soh}")
    ranked = sorted(usable, key=lambda c: usable[c], reverse=bool(slowest))
    picked = ranked[:int(n_test)]
    kind = "slowest" if slowest else "fastest"
    print(f"[cfg] {kind}-ageing test set, weeks to SoH {target_soh:g}: "
          f"{', '.join(f'{c.split()[-1].split(chr(95))[-1]}={usable[c]:.1f}' for c in picked)}")
    print(f"[cfg] training cells span {min(usable[c] for c in ranked[len(picked):]):.1f} - "
          f"{max(usable[c] for c in ranked[len(picked):]):.1f} weeks")
    return sorted(picked)


def choose_test_cells(cells: list[str], out_dir: Path, cond_csv: str,
                      n_test: int, seed: int) -> list[str]:
    """Deterministic held-out set, stratified by cycling DOD where it is known.

    Drawing at random over all 101 cells would frequently give a test set made
    only of 20% DOD cells, which are two thirds of the dataset and by far the
    slowest ageing; the error would then say nothing about the 50/75% groups.
    """
    path = out_dir / cond_csv
    strata: dict[str, list[str]] = {}
    if path.exists():
        cond = pd.read_csv(path)
        cond["_pref"] = cond["dod_pct"].notna().astype(int) if "dod_pct" in cond.columns else 0
        chosen = (cond.sort_values(["_pref", "n_files"], ascending=[False, False])
                      .drop_duplicates(subset="cell", keep="first"))
        dod = dict(zip(chosen["cell"], chosen.get("dod_pct", pd.Series(dtype=float))))
        for cell in cells:
            value = dod.get(cell, np.nan)
            key = "na" if pd.isna(value) else f"{float(value):g}"
            strata.setdefault(key, []).append(cell)
    else:
        print(f"[WARN] {path} not found; test cells drawn without stratification")
        strata["all"] = list(cells)

    rng = random.Random(int(seed))
    picked: list[str] = []
    # largest strata first, so the rounding remainder lands where there is room
    for key in sorted(strata, key=lambda k: (-len(strata[k]), k)):
        members = sorted(strata[key])
        share = len(members) / len(cells)
        take = int(round(n_test * share))
        take = max(1, take) if len(members) >= 3 else 0
        take = min(take, len(members) - 1, n_test - len(picked))
        if take <= 0:
            continue
        picked.extend(rng.sample(members, take))
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--feature-dir", type=Path, default=None,
                    help=f"default: <out-dir>/{DEFAULT_FEATURE_SUBDIR}")
    ap.add_argument("--dest", type=Path, default=None,
                    help=f"results folder (default: <out-dir>/{DEFAULT_DEST_SUBDIR})")
    ap.add_argument("--target-soh", type=float, default=0.95)
    ap.add_argument("--anchor-soh", type=float, default=0.975)
    ap.add_argument("--input-steps", type=int, default=4)
    ap.add_argument("--augmentation", default="none",
                    choices=["none", "normal", "cal", "hybrid", "codex"])
    ap.add_argument("--augmentation-samples", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--test-cells", default=None,
                    help="comma separated cell ids, overrides --test-selection")
    ap.add_argument("--test-selection", default="slowest",
                    choices=["slowest", "fastest", "stratified"],
                    help="how the held-out reference cells are chosen: the "
                         "slowest-ageing cells (default, matches the LFP "
                         "reference-cell setup), the fastest, or a DOD-stratified "
                         "random draw fixed by --seed")
    ap.add_argument("--seed", type=int, default=42,
                    help="fixes both the test-cell draw and the model seeds")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed-count", type=int, default=10,
                    help="models in the ensemble (default 10); lower it for a quick look")
    ap.add_argument("--plots", action="store_true",
                    help="also write the diagnostic figures (Agg backend)")
    args = ap.parse_args()

    feature_dir = args.feature_dir or (args.out_dir / DEFAULT_FEATURE_SUBDIR)
    dest = args.dest or (args.out_dir / DEFAULT_DEST_SUBDIR)
    if not feature_dir.is_dir():
        raise SystemExit(f"[ERROR] {feature_dir} not found - run "
                         f"prepare_samsung_prediction_input.py first")
    dest.mkdir(parents=True, exist_ok=True)

    lpc = import_pipeline()

    cells = sorted(p.stem for p in feature_dir.glob("*.csv") if not p.name.startswith("_"))
    if not cells:
        raise SystemExit(f"[ERROR] no feature csv under {feature_dir}")

    if args.test_cells:
        test_names = sorted({s.strip() for s in args.test_cells.split(",") if s.strip()})
        unknown = [c for c in test_names if c not in cells]
        if unknown:
            raise SystemExit(f"[ERROR] unknown test cell(s): {', '.join(unknown)}")
    elif args.test_selection in ("slowest", "fastest"):
        test_names = choose_extreme_cells(cells, feature_dir, float(args.target_soh),
                                          args.n_test,
                                          slowest=args.test_selection == "slowest")
    else:
        test_names = choose_test_cells(cells, args.out_dir, DEFAULT_COND_CSV,
                                       args.n_test, args.seed)
    print(f"[cfg] feature_dir   {feature_dir}  ({len(cells)} cells)")
    print(f"[cfg] anchor SoH    {args.anchor_soh}   target SoH {args.target_soh}")
    print(f"[cfg] input_steps   {args.input_steps}   augmentation {args.augmentation}")
    print(f"[cfg] test cells    {len(test_names)}: "
          f"{', '.join(c.split('_')[-1] for c in test_names)}")

    config = replace(lpc.TrainingConfig(),
                     input_steps=int(args.input_steps),
                     epochs=int(args.epochs),
                     seed_count=int(args.seed_count),
                     seed_source=int(args.seed))

    result = lpc.run_lfp_prediction_pipeline(
        interp_dir=feature_dir,
        target_soh=float(args.target_soh),
        fixed_anchor_soh=float(args.anchor_soh),
        training_config=config,
        explicit_test_names=test_names,
        train_substring="",              # every non-test cell trains
        augmentation_method=str(args.augmentation),
        augmentation_sample_count=args.augmentation_samples,
        show_plot=bool(args.plots),
        plot_title_prefix="Samsung 35E prediction",
    )

    meta = result["metadata"]
    selection_tag = "named" if args.test_cells else str(args.test_selection)
    tag = (f"soh{args.target_soh:g}_anchor{args.anchor_soh:g}"
           f"_steps{args.input_steps}_{args.augmentation}_{selection_tag}")

    print()
    print(f"[run] train real {meta['train_real_count']}  "
          f"augmented {meta['train_aug_count']}  test {meta['test_count']}")
    print(f"[run] sequence steps {meta['input_sequence_steps']}  "
          f"model {meta['prediction_model']}")
    dropped_frames = result["skipped_frames_df"]
    dropped_seqs = result["skipped_sequences_df"]
    if len(dropped_frames):
        print(f"[run] {len(dropped_frames)} cell(s) dropped while loading:")
        print(dropped_frames["reason"].value_counts().to_string())
    if len(dropped_seqs):
        print(f"[run] {len(dropped_seqs)} cell(s) dropped for a short sequence "
              f"(fewer than {args.input_steps} checkups above SoH {args.anchor_soh})")

    print()
    print(f"[weeks]      RMSE {result['rmse_weeks']:.2f} weeks   "
          f"MAPE {result['mape_weeks']:.1f}%")
    print(f"[throughput] RMSE {result['rmse_throughput'] / 3600.0:.0f} Ah   "
          f"MAPE {result['mape_throughput']:.1f}%")

    pred = pd.DataFrame({
        "cell": result["test_names"],
        "true_weeks": result["y_test_weeks"],
        "pred_weeks": result["y_pred_weeks"],
        "true_throughput_Ah": np.asarray(result["y_test_throughput"], float) / 3600.0,
        "pred_throughput_Ah": np.asarray(result["y_pred_throughput"], float) / 3600.0,
    })
    pred["err_weeks"] = pred["pred_weeks"] - pred["true_weeks"]
    pred["err_pct"] = 100.0 * pred["err_weeks"] / pred["true_weeks"]
    csv = dest / f"samsung_prediction_{tag}.csv"
    pred.to_csv(csv, index=False)
    print()
    print(pred.round(2).to_string(index=False))
    print(f"\n[OK] {csv}")

    meta_csv = dest / f"samsung_prediction_{tag}_metadata.csv"
    pd.DataFrame([{k: v for k, v in meta.items()
                   if isinstance(v, (int, float, str, bool))}]).to_csv(meta_csv, index=False)
    print(f"[OK] {meta_csv}")

    if args.plots:
        for num in plt.get_fignums():
            png = dest / f"samsung_prediction_{tag}_fig{num}.png"
            plt.figure(num).savefig(png, dpi=140, bbox_inches="tight")
            print(f"[OK] {png}")
        plt.close("all")


if __name__ == "__main__":
    main()
