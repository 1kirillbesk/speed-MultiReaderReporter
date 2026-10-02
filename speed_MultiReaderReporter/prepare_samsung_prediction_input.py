# speed_MultiReaderReporter/prepare_samsung_prediction_input.py
"""Make the Samsung 35E feature tables readable by lfp_prediction_common.

lfp_prediction_common.load_lfp_cell_frames() requires `throughput_cum` and
rejects any cell whose throughput at the target SoH is not finite and > 0. For
the Model2Life / MobilEM cells the pipeline left `throughput_cum` at 0.0 for all
101 cells, so every one of them would be dropped as `invalid_target_throughput`
before a single model is fit.

This script rebuilds that column the same way plot_capacity_trajectories.py
does for the `samsung35e` family: cumulative `throughput_Ah` from
<cell>/cycling/report.csv, counting only the ageing programmes (the checkup
programmes STD_Kap / *CUp / qOCV / Puls are excluded, otherwise the checkups
themselves add charge and the series stops being monotone). The result is
converted to AMPERE-SECONDS, which is the unit lfp_prediction_common assumes -
DEFAULT_INPUT_END_THROUGHPUT is 0.4e7, i.e. ~1100 Ah.

The curve columns (Vcha, dQdVcha, Q_intVcha) hold multi-line array strings and
are ~99% of the file size; they are dropped, since the prediction code only
reads scalar columns.

`SOH` and `weeks` are NOT written here - prepare_lfp_dataframe() derives them
itself from cap_ocv_dis and CU_time.

Usage:
    python speed_MultiReaderReporter/prepare_samsung_prediction_input.py
    python speed_MultiReaderReporter/prepare_samsung_prediction_input.py --cap-col cap_dis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path("E:/download/samsung35e_combined")
DEFAULT_DEST = "prediction_feature"
# same exclusion as the samsung35e preset in plot_capacity_trajectories.py
CHECKUP_PROGRAMME_RE = r"STD_Kap|CUp|qOCV|Puls"
DROP_COLUMNS = ("Vcha", "dQdVcha", "Q_intVcha")


def cumulative_throughput_As(out_dir: Path, cell: str,
                             times: pd.Series, exclude: str) -> np.ndarray:
    """Cumulative cycling throughput in ampere-seconds at each checkup time."""
    report = out_dir / cell / "cycling" / "report.csv"
    if not report.exists():
        report = out_dir / cell / "total" / "report.csv"
    if not report.exists():
        return np.full(len(times), np.nan)

    rep = pd.read_csv(report)
    if "throughput_Ah" not in rep.columns:
        return np.full(len(times), np.nan)
    rep = rep[rep["program"].astype(str) != "TOTAL"]
    if exclude:
        rep = rep[~rep["program"].astype(str).str.contains(exclude, case=False, na=False)]
    rep = rep.assign(end=pd.to_datetime(rep["end_time"], errors="coerce"))
    rep = rep.dropna(subset=["end"]).sort_values("end")
    if rep.empty:
        return np.full(len(times), np.nan)
    rep["cum_Ah"] = pd.to_numeric(rep["throughput_Ah"], errors="coerce").fillna(0.0).cumsum()

    out = []
    for t in times:
        prior = rep.loc[rep["end"] <= t, "cum_Ah"]
        out.append(float(prior.max()) if len(prior) else 0.0)
    return np.asarray(out, dtype=float) * 3600.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"pipeline output root (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--dest", type=Path, default=None,
                    help=f"destination folder (default: <out-dir>/{DEFAULT_DEST})")
    ap.add_argument("--pattern", default="LBE_Samsung_35E_*")
    ap.add_argument("--cap-col", default="cap_ocv_dis",
                    help="capacity column that must be present and positive "
                         "(default: cap_ocv_dis, the C/10 OCV discharge)")
    ap.add_argument("--exclude", default=CHECKUP_PROGRAMME_RE,
                    help="regex of programme names NOT counted as cycling throughput")
    args = ap.parse_args()

    src = args.out_dir / "cell_feature"
    dest = args.dest or (args.out_dir / DEFAULT_DEST)
    dest.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        print(f"[ERROR] {src} not found")
        return

    written, skipped = 0, []
    rows = []
    for path in sorted(src.glob(f"{args.pattern}.csv")):
        cell = path.stem
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

        if "CU_time" not in df.columns or args.cap_col not in df.columns:
            skipped.append((cell, "missing CU_time or capacity column"))
            continue
        t = pd.to_datetime(df["CU_time"], errors="coerce")
        cap = pd.to_numeric(df[args.cap_col], errors="coerce")
        keep = t.notna() & cap.notna() & (cap > 0)
        df = df.loc[keep].copy()
        t = t.loc[keep]
        if len(df) < 3:
            skipped.append((cell, f"only {len(df)} usable checkup(s)"))
            continue
        df = df.assign(_t=t).sort_values("_t").drop(columns="_t").reset_index(drop=True)
        t = t.sort_values().reset_index(drop=True)

        thr = cumulative_throughput_As(args.out_dir, cell, t, args.exclude)
        if not np.isfinite(thr).any() or np.nanmax(thr) <= 0:
            skipped.append((cell, "no cycling throughput in report.csv"))
            continue
        # A flat leading segment is fine, but the series must end above zero and
        # never go backwards, or the SoH->throughput interpolation is undefined.
        df["throughput_cum"] = np.maximum.accumulate(np.nan_to_num(thr, nan=0.0))

        df.to_csv(dest / f"{cell}.csv", index=False)
        written += 1
        soh = np.minimum.accumulate(
            np.clip(pd.to_numeric(df[args.cap_col], errors="coerce").to_numpy(float)
                    / float(pd.to_numeric(df[args.cap_col], errors="coerce").iloc[0]), 0, 1.05))
        rows.append({"cell": cell, "n": len(df), "soh_last": soh[-1],
                     "thr_last_Ah": df["throughput_cum"].iloc[-1] / 3600.0})

    print(f"[OK] {written} cell(s) -> {dest}")
    for cell, why in skipped:
        print(f"  [skip] {cell}: {why}")
    if rows:
        t = pd.DataFrame(rows)
        print(f"\ncheckups per cell: min {t.n.min()}, median {int(t.n.median())}, max {t.n.max()}")
        print(f"final SoH:         min {t.soh_last.min():.3f}, median {t.soh_last.median():.3f}")
        print(f"final throughput:  min {t.thr_last_Ah.min():.0f} Ah, "
              f"median {t.thr_last_Ah.median():.0f} Ah, max {t.thr_last_Ah.max():.0f} Ah")


if __name__ == "__main__":
    main()
