# speed_MultiReaderReporter/core/pipeline.py
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd

from .classify import is_checkup_run, configure_from_config
from .plotting import save_group_plot, save_grouped_checkup_plot
from .reports import write_report, write_grouped_report
from .capacity import compute_checkup_point_step19
from .soh import cumulative_throughput_until
from .model import RunRecord
from .grouping import prepare_grouping, compute_grouped_segments

def run_pipeline(runs: list[RunRecord], cfg: dict, out_root: Path):
    legend_ncol = int(cfg.get("legend", {}).get("ncol", 4))
    configure_from_config(cfg)

    # group by cell
    by_cell: dict[str, list[tuple[pd.DataFrame, str]]] = defaultdict(list)
    for r in runs:
        by_cell[r.cell].append((r.df, r.program))

    for cell, total_list in sorted(by_cell.items()):
        cell_dir = out_root / cell
        (cell_dir / "total").mkdir(parents=True, exist_ok=True)
        (cell_dir / "checkup").mkdir(parents=True, exist_ok=True)
        (cell_dir / "cycling").mkdir(parents=True, exist_ok=True)

        # split
        checkup_list, cycling_list = [], []
        for df, label in total_list:
            (checkup_list if is_checkup_run(label, df) else cycling_list).append((df, label))

        # plots
        save_group_plot(cell, total_list,   cell_dir / "total",   "total",   legend_ncol)
        save_group_plot(cell, checkup_list, cell_dir / "checkup", "checkup", legend_ncol)
        save_group_plot(cell, cycling_list, cell_dir / "cycling", "cycling", legend_ncol)

        # reports
        fmt = str(cfg.get("reports", {}).get("format", "csv")).lower()
        mat_var = str(cfg.get("reports", {}).get("mat_variable", "report"))

        write_report(total_list, cell_dir / "total" / "report", f"{cell} total", fmt=fmt, mat_variable=mat_var)
        write_report(checkup_list, cell_dir / "checkup" / "report", f"{cell} checkup", fmt=fmt, mat_variable=mat_var)
        write_report(cycling_list, cell_dir / "cycling" / "report", f"{cell} cycling", fmt=fmt, mat_variable=mat_var)

        # --- optional grouped report + per-checkup grouped plots ---
        prep = prepare_grouping(cfg)  # returns None when mode = off
        if prep is not None:
            # Compute segments once
            per_run, grouped_flat = compute_grouped_segments(checkup_list, prep.cfg)

            # Optional per-checkup debug plots
            if prep.do_plots:
                counts = Counter(run_label for run_label, _ in per_run)
                seen: dict[str, int] = defaultdict(int)
                for run_label, segs in per_run:
                    seen[run_label] += 1
                    total = counts[run_label]
                    occurrence = seen[run_label]
                    save_grouped_checkup_plot(
                        cell=cell,
                        run_label=run_label,
                        segments=segs,
                        out_dir=cell_dir / "checkup" / "grouped_plots",
                        max_points_per_segment=prep.max_points_per_segment,
                        occurrence_index=occurrence if total > 1 else None,
                        occurrence_total=total if total > 1 else None,
                    )

            # Optional grouped report
            if prep.do_report and grouped_flat:
                write_grouped_report(
                    grouped_flat,
                    cell_dir / "checkup" / "report_grouped",
                    f"{cell} checkup (grouped)",
                    fmt=fmt,
                    mat_variable=mat_var if fmt != "csv" else "report_grouped",
                )

        # SoH: step-19 capacity vs cumulative throughput (checkups only; need step_int)
        soh_points = []
        for df_chk, lbl_chk in checkup_list:
            if "cu" not in lbl_chk.lower():  # your rule of thumb
                continue
            if "step_int" not in df_chk.columns:
                continue
            res = compute_checkup_point_step19(
                df_chk,
                min_step_required=int(cfg.get("soh", {}).get("min_step_required", 20)),
                eod_v_cut=cfg.get("soh", {}).get("eod_v_cut_V", None),
                i_thresh=float(cfg.get("soh", {}).get("i_thresh_A", 0.0)),
            )
            if res is None:
                continue
            cap_ah, t_end = res
            x_thru = cumulative_throughput_until(total_list, t_end)
            soh_points.append((x_thru, cap_ah, lbl_chk, t_end))

        if soh_points:
            df_soh = pd.DataFrame(soh_points, columns=[
                "throughput_Ah", "discharge_capacity_Ah", "program", "discharge_end_time"
            ])
            df_soh.to_csv(cell_dir / "checkup" / "soh_discharge_capacity_vs_throughput.csv", index=False)
            # simple scatter
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            xs = [p[0] for p in soh_points]; ys = [p[1] for p in soh_points]
            plt.scatter(xs, ys)
            for x, y, name, _ in soh_points:
                plt.annotate(name, (x, y), fontsize=8, xytext=(5, 2), textcoords="offset points")
            plt.xlabel("Cumulative charge throughput up to discharge [Ah]")
            plt.ylabel("Discharge capacity (step 19) [Ah]")
            plt.title(f"Cell: {cell} — SoH: Capacity vs Throughput")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(cell_dir / "checkup" / "soh_discharge_capacity_vs_throughput.png", dpi=160)
            plt.close()
        else:
            print(f"[INFO] {cell}: no valid step-19 discharges found for SoH plot.")
