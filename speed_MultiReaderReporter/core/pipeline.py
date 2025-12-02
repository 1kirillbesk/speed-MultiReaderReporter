# speed_MultiReaderReporter/core/pipeline.py
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd

from .classify import is_checkup_run, configure_from_config
from .plotting import save_group_plot, save_grouped_checkup_plot
from .reports import write_report, write_grouped_report
from .capacity import compute_checkup_point_step19, compute_checkup_point_step6
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
        write_report(cycling_list, cell_dir / "cycling" / "report", f"{cell} cycling", fmt=fmt, mat_variable=mat_var)

        # --- optional grouped report + per-checkup grouped plots ---
        prep = prepare_grouping(cfg)  # returns None when mode = off
        per_run: list[tuple[str, list[tuple[pd.DataFrame, str]]]] = []
        grouped_flat: list[tuple[pd.DataFrame, str, str]] = []
        if prep is not None and checkup_list:
            # Compute segments once
            per_run, grouped_flat = compute_grouped_segments(checkup_list, prep.cfg)

        # Checkup report (with grouping when requested)
        if prep is not None and prep.do_report and per_run:
            write_report(
                checkup_list,
                cell_dir / "checkup" / "report",
                f"{cell} checkup",
                fmt=fmt,
                mat_variable=mat_var,
                grouped_runs=per_run,
            )
        else:
            write_report(
                checkup_list,
                cell_dir / "checkup" / "report",
                f"{cell} checkup",
                fmt=fmt,
                mat_variable=mat_var,
            )

        if prep is not None and per_run:
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

        # SoH: checkup discharge capacity vs cumulative throughput (step-19 for CU, step-6 for RPT)
        soh_points = []
        soh_rows = []
        soh_cfg = cfg.get("soh", {})
        export_soh_data = bool(soh_cfg.get("export_data", True))
        include_rpt = bool(soh_cfg.get("include_rpt", True))
        rpt_min_step = soh_cfg.get("rpt_min_step_required", None)
        if rpt_min_step is not None:
            rpt_min_step = int(rpt_min_step)
        rpt_trailing_step = soh_cfg.get("rpt_trailing_step_id", None)
        if rpt_trailing_step is not None:
            rpt_trailing_step = int(rpt_trailing_step)

        for df_chk, lbl_chk in checkup_list:
            label_lower = lbl_chk.lower()
            res = None
            source_type = None
            if "cu" in label_lower:
                if "step_int" not in df_chk.columns:
                    continue
                res = compute_checkup_point_step19(
                    df_chk,
                    min_step_required=int(soh_cfg.get("min_step_required", 20)),
                    eod_v_cut=soh_cfg.get("eod_v_cut_V", None),
                    i_thresh=float(soh_cfg.get("i_thresh_A", 0.0)),
                )
                source_type = "CU"
            elif include_rpt and "rpt" in label_lower:
                if "step_int" not in df_chk.columns:
                    continue
                res = compute_checkup_point_step6(
                    df_chk,
                    min_step_required=rpt_min_step,
                    eod_v_cut=soh_cfg.get("eod_v_cut_V", None),
                    i_thresh=float(soh_cfg.get("i_thresh_A", 0.0)),
                    trailing_step_id=rpt_trailing_step,
                    require_trailing_step=bool(soh_cfg.get("rpt_require_trailing_step", False)),
                )
                source_type = "RPT"
            if res is None:
                continue
            x_thru = cumulative_throughput_until(total_list, res.discharge_end_time)
            soh_points.append((x_thru, res.capacity_Ah, lbl_chk, res.discharge_end_time, source_type))
            if source_type == "RPT":
                print(f"[INFO] Added RPT SoH point for cell {cell}, step 6 discharge capacity = {res.capacity_Ah:.4f} Ah")
            soh_rows.append({
                "cell_id": cell,
                "program_name": lbl_chk,
                "source_type": source_type,
                "step_id": res.step_id,
                "discharge_end_time": res.discharge_end_time,
                "discharge_start_time": res.discharge_start_time,
                "throughput_Ah": x_thru,
                "discharge_capacity_Ah": res.capacity_Ah,
                "step_start_index": res.index_start,
                "step_end_index": res.index_end,
                "step_min_voltage_V": res.min_voltage_V,
                "step19_start_index": res.index_start if res.step_id == 19 else None,
                "step19_end_index": res.index_end if res.step_id == 19 else None,
                "step19_min_voltage_V": res.min_voltage_V if res.step_id == 19 else None,
            })

        if soh_points:
            df_soh = pd.DataFrame(soh_rows)
            soh_dir = cell_dir / "checkup"
            if export_soh_data:
                soh_data_path = soh_dir / "soh_scatter_data.csv"
                df_soh.to_csv(soh_data_path, index=False)
                print(f"[OK] wrote SoH data: {soh_data_path}")

            legacy_df = df_soh.rename(columns={"program_name": "program"})[
                ["throughput_Ah", "discharge_capacity_Ah", "program", "discharge_end_time"]
            ]
            legacy_df.to_csv(soh_dir / "soh_discharge_capacity_vs_throughput.csv", index=False)
            # simple scatter
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            xs = df_soh["throughput_Ah"].to_list(); ys = df_soh["discharge_capacity_Ah"].to_list()
            plt.scatter(xs, ys)
            for x, y, name in zip(xs, ys, df_soh["program_name"].to_list()):
                plt.annotate(name, (x, y), fontsize=8, xytext=(5, 2), textcoords="offset points")
            plt.xlabel("Cumulative charge throughput up to discharge [Ah]")
            plt.ylabel("Discharge capacity (checkup) [Ah]")
            plt.title(f"Cell: {cell} — SoH: Capacity vs Throughput")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(cell_dir / "checkup" / "soh_discharge_capacity_vs_throughput.png", dpi=160)
            plt.close()
        else:
            print(f"[INFO] {cell}: no valid checkup discharges found for SoH plot.")
