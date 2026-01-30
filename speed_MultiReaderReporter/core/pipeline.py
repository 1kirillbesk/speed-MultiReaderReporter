# speed_MultiReaderReporter/core/pipeline.py
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt
from .classify import *
from .plotting import *
from .reports import write_report, write_grouped_report
from .capacity import *
from .pulses import *
from .soh import cumulative_throughput_until
from .model import RunRecord
from .grouping import prepare_grouping, compute_grouped_segments
import re

log_path = Path('C:/Users/Public/Documents/RL_project/out_jgne') / "errors.log"
#C:/Users/Public/Documents/takedata/Speed/out_lw
logging.basicConfig(
    filename=str(log_path),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def run_pipeline(runs: list[RunRecord], cfg: dict, out_root: Path):
    legend_ncol = int(cfg.get("legend", {}).get("ncol", 4))
    configure_from_config(cfg)
    volt_lim = cfg.get("voltage")

    # group by cell
    by_cell: dict[str, list[tuple[pd.DataFrame, str]]] = defaultdict(list)
    for r in runs:
        by_cell[r.cell].append((r.df, r.program))

    for cell, total_list in sorted(by_cell.items()):
        try:
            cell_dir = out_root / cell
            (cell_dir / "total").mkdir(parents=True, exist_ok=True)
            (cell_dir / "checkup").mkdir(parents=True, exist_ok=True)
            (cell_dir / "cycling").mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.exception(f"[{cell}] Failed to create output directories: {e}")
            continue
        try:
            checkup_list, cycling_list = split_total_list(cell,total_list,cfg)
        except Exception as e:
            logging.exception(f"[{cell}] Failed while splitting total_list: {e}")
            continue

        # --- plots (non-critical) ---
        try:
            save_group_plot(cell, total_list, cell_dir / "total", "total", legend_ncol)
            save_group_plot(cell, checkup_list, cell_dir / "checkup", "checkup", legend_ncol)
            save_group_plot(cell, cycling_list, cell_dir / "cycling", "cycling", legend_ncol)
        except Exception as e:
            logging.exception(f"[{cell}] save_group_plot total failed: {e}")

        # --- reports (non-critical) ---
        try:
            fmt = str(cfg.get("reports", {}).get("format", "csv")).lower()
            mat_var = str(cfg.get("reports", {}).get("mat_variable", "report"))

            write_report(total_list, cell_dir / "total" / "report", f"{cell} total", fmt=fmt, mat_variable=mat_var)
            write_report(cycling_list, cell_dir / "cycling" / "report", f"{cell} cycling", fmt=fmt,
                         mat_variable=mat_var)
        except Exception as e:
            logging.exception(f"[{cell}] write_report total/cycling failed: {e}")

        # --- optional grouped report + per-checkup grouped plots ---
        try:
            prep = prepare_grouping(cfg)  # returns None when mode = off
            per_run = []
            grouped_flat = []

            if prep is not None and checkup_list:
                per_run, grouped_flat = compute_grouped_segments(checkup_list, prep.cfg)

            # Checkup report
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
                if prep.do_plots:
                    counts = Counter(run_label for run_label, _ in per_run)
                    seen = defaultdict(int)
                    for run_label, segs in per_run:
                        seen[run_label] += 1
                        total = counts[run_label]
                        occurrence = seen[run_label]
                        try:
                            save_grouped_checkup_plot(
                                cell=cell,
                                run_label=run_label,
                                segments=segs,
                                out_dir=cell_dir / "checkup" / "grouped_plots",
                                max_points_per_segment=prep.max_points_per_segment,
                                occurrence_index=occurrence if total > 1 else None,
                                occurrence_total=total if total > 1 else None,
                            )
                        except Exception as e:
                            logging.exception(f"[{cell}] save_grouped_checkup_plot failed for '{run_label}': {e}")

                if prep.do_report and grouped_flat:
                    try:
                        write_grouped_report(
                            grouped_flat,
                            cell_dir / "checkup" / "report_grouped",
                            f"{cell} checkup (grouped)",
                            fmt=fmt,
                            mat_variable=mat_var if fmt != "csv" else "report_grouped",
                        )
                    except Exception as e:
                        logging.exception(f"[{cell}] write_grouped_report failed: {e}")

        except Exception as e:
            logging.exception(f"[{cell}] grouped reporting/plots block failed: {e}")

        # --- SoH + throughput + features (this is critical; wrap per-run) ---
        try:
            soh_cfg = cfg.get("soh", {})
            export_soh_data = bool(soh_cfg.get("export_data", True))
            include_rpt = bool(soh_cfg.get("include_rpt", True))
            rpt_min_step = soh_cfg.get("rpt_min_step_required", None)
            if rpt_min_step is not None:
                rpt_min_step = int(rpt_min_step)
            rpt_trailing_step = soh_cfg.get("rpt_trailing_step_id", None)
            if rpt_trailing_step is not None:
                rpt_trailing_step = int(rpt_trailing_step)

            rows_through = []
            for df_cyc, lbl_cyc in cycling_list:
                try:
                    label_lower = lbl_cyc.lower()
                    experimental_condition = parse_experiment_label(lbl_cyc)

                    if "soc" in label_lower:
                        df_cyc = df_cyc.copy()
                        df_cyc["abs_time"] = pd.to_datetime(df_cyc["abs_time"], errors="coerce")
                        df_cyc = df_cyc.dropna(subset=["abs_time"])
                        df_cyc["relative_time_s"] = (df_cyc["abs_time"] - df_cyc["abs_time"].iloc[0]).dt.total_seconds()

                        throughput = compute_cum_abs_charge(df_cyc, time_col="abs_time", current_col="current_A")
                        rows_through.append(throughput)
                except Exception as e:
                    logging.exception(f"[{cell}] throughput calc failed for cycling run '{lbl_cyc}': {e}")
                    continue

            rows = []
            for df_chk, lbl_chk in checkup_list:
                try:
                    label_lower = lbl_chk.lower()
                    if "cu" in label_lower or "rpt" in label_lower:
                        df_chk = df_chk.copy()
                        df_chk["abs_time"] = pd.to_datetime(df_chk["abs_time"], errors="coerce")
                        df_chk = df_chk.dropna(subset=["abs_time"])
                        df_chk["relative_time_s"] = (df_chk["abs_time"] - df_chk["abs_time"].iloc[0]).dt.total_seconds()

                        ocv_features = extract_features(df_chk, cell, cfg)

                        df_filtered = df_chk[df_chk["procedure"] == "rul_Pulse"].reset_index(drop=True)
                        # df_filtered = (
                        #     df_chk
                        #     .loc[df_chk.index[df_chk["step_int"] == 26].max() + 1:]
                        #     .query("0 <= step_int <= 15")
                        # )
                        pulse_feature = analyze_df_pulse(df_filtered)

                        features = ocv_features | pulse_feature
                        rows.append(features)
                except Exception as e:
                    logging.exception(f"[{cell}] feature extraction failed for checkup run '{lbl_chk}': {e}")
                    continue

            if not rows:
                logging.warning(f"[{cell}] No feature rows produced; skipping summary + save.")
                continue

            df_summary = pd.DataFrame(rows)

            # throughput column (optional)
            try:
                df_summary = add_throughput_column(
                    df_summary=df_summary,
                    rows_through=rows_through,
                    df_time_col="CU_time",
                    dict_time_key="abs_time",
                    dict_value_key="throughput",
                    out_col="throughput_sum"
                )
            except Exception as e:
                logging.exception(f"[{cell}] add_throughput_column failed: {e}")

            # add experimental condition columns if available
            try:
                if experimental_condition is not None:
                    for k, v in experimental_condition.items():
                        df_summary[k] = v
            except Exception as e:
                logging.exception(f"[{cell}] adding experimental_condition columns failed: {e}")

        except Exception as e:
            logging.exception(f"[{cell}] SoH/feature pipeline failed: {e}")
            continue

        # --- derived features + plots (non-critical, but keep going if they fail) ---
        try:
            vol_high = volt_lim["high"];
            vol_low = volt_lim["low"]
            vol_mhigh = volt_lim["highm"];
            vol_mlow = volt_lim["lowm"]

            mean_mid_cha, var_mid_cha = window_delta_mean_var(df_summary, x_col="Vcha", y_col="dQdVcha", x_lo=vol_mlow,
                                                              x_hi=vol_mhigh)
            mean_low_cha, var_low_cha = window_delta_mean_var(df_summary, x_col="Vcha", y_col="dQdVcha", x_lo=vol_low,
                                                              x_hi=vol_mlow)
            mean_high_cha, var_high_cha = window_delta_mean_var(df_summary, x_col="Vcha", y_col="dQdVcha",
                                                                x_lo=vol_mhigh, x_hi=vol_high)
            mean_dQ_cha, var_dQ_cha = window_delta_mean_var(df_summary, x_col="Vcha", y_col="Q_intVcha", x_lo=vol_low,
                                                            x_hi=vol_high)

            df_summary["mean_d_dqdv_m_c"] = mean_mid_cha;
            df_summary["var_d_dqdv_m_c"] = var_mid_cha
            df_summary["mean_d_dqdv_l_c"] = mean_low_cha;
            df_summary["var_d_dqdv_l_c"] = var_low_cha
            df_summary["mean_d_dqdv_h_c"] = mean_high_cha;
            df_summary["var_d_dqdv_h_c"] = var_high_cha
            df_summary["mean_dQ_c"] = mean_dQ_cha;
            df_summary["var_dQ_c"] = var_dQ_cha

            mean_mid_dis, var_mid_dis = window_delta_mean_var(df_summary, x_col="Vdis", y_col="dQdVdis", x_lo=vol_mlow,
                                                              x_hi=vol_mhigh)
            mean_low_dis, var_low_dis = window_delta_mean_var(df_summary, x_col="Vdis", y_col="dQdVdis", x_lo=vol_low,
                                                              x_hi=vol_mlow)
            mean_high_dis, var_high_dis = window_delta_mean_var(df_summary, x_col="Vdis", y_col="dQdVdis",
                                                                x_lo=vol_mhigh, x_hi=vol_high)
            mean_dQ_dis, var_dQ_dis = window_delta_mean_var(df_summary, x_col="Vdis", y_col="Q_intVdis", x_lo=vol_low,
                                                            x_hi=vol_high)

            df_summary["mean_d_dqdv_m_d"] = mean_mid_dis;
            df_summary["var_d_dqdv_m_d"] = var_mid_dis
            df_summary["mean_d_dqdv_l_d"] = mean_low_dis;
            df_summary["var_d_dqdv_l_d"] = var_low_dis
            df_summary["mean_d_dqdv_h_d"] = mean_high_dis;
            df_summary["var_d_dqdv_h_d"] = var_high_dis
            df_summary["mean_dQ_d"] = mean_dQ_dis;
            df_summary["var_dQ_d"] = var_dQ_dis

            mean_mid_t, var_mid_t = window_delta_mean_var(df_summary, x_col="Vdis", y_col="dTdV", x_lo=vol_mlow,
                                                          x_hi=vol_mhigh)
            mean_low_t, var_low_t = window_delta_mean_var(df_summary, x_col="Vdis", y_col="dTdV", x_lo=vol_low,
                                                          x_hi=vol_mlow)
            mean_high_t, var_high_t = window_delta_mean_var(df_summary, x_col="Vdis", y_col="dTdV", x_lo=vol_mhigh,
                                                            x_hi=vol_high)
            mean_d_Qt, var_d_Qt = window_delta_mean_var(df_summary, x_col="Vdis", y_col="T_intV", x_lo=vol_low,
                                                        x_hi=vol_high)

            df_summary["mean_dqdv_mt"] = mean_mid_t;
            df_summary["var_d_dqdv_mt"] = var_mid_t
            df_summary["mean_dqdv_lt"] = mean_low_t;
            df_summary["var_d_dqdv_lt"] = var_low_t
            df_summary["mean_dqdv_ht"] = mean_high_t;
            df_summary["var_d_dqdv_ht"] = var_high_t
            df_summary["mean_d_Qt"] = mean_d_Qt;
            df_summary["var_d_Qt"] = var_d_Qt

        except Exception as e:
            logging.exception(f"[{cell}] derived feature calc failed: {e}")

        # plots (optional)
        try:
            plot_curves(df_summary, cfg, cell, x_col="Vcha", y_col="dQdVcha", kmax=5, kmin=5, peak_distance=10)
            plot_curves(df_summary, cfg, cell, x_col="Qcha", y_col="dVdQcha", kmax=5, kmin=5, peak_distance=10)
            plot_curves(df_summary, cfg, cell, x_col="Vdis", y_col="dQdVdis", kmax=5, kmin=5, peak_distance=10)
            plot_curves(df_summary, cfg, cell, x_col="Qdis", y_col="dVdQdis", kmax=5, kmin=5, peak_distance=10)
            plot_curves(df_summary, cfg, cell, x_col="VdisT", y_col="dTdV", kmax=5, kmin=5, peak_distance=10)
        except Exception as e:
            logging.exception(f"[{cell}] plot_curves failed: {e}")

        # drop + save (critical save should be protected too)
        # "Vcha", "dQdVcha", "Q_intVcha are taken",
        try:
            df_summary = df_summary.drop(columns=[
                "Qcha", "dVdQcha", "V_intQcha",
                "Vdis", "dQdVdis", "Q_intVdis", "Qdis", "dVdQdis", "V_intQdis",
                "VdisT", "dTdV", "T_intV"
            ], errors="ignore")

            cell_feature_dir = out_root / "cell_feature"
            cell_feature_dir.mkdir(parents=True, exist_ok=True)
            summary_path = cell_feature_dir / f"{cell}.csv"
            df_summary.to_csv(summary_path, index=False)
        except Exception as e:
            logging.exception(f"[{cell}] Failed to save df_summary csv: {e}")
            continue
                # plt.plot(df_chk["relative_time_s"],df_chk["voltage_V"])
                # plt.show()
        '''
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
        '''
        '''
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
        '''
        print(f"[INFO] {cell}: has been processed.")


def add_throughput_column(df_summary, rows_through, df_time_col="CU_time",
                          dict_time_key="abs_time", dict_value_key="throughput",
                          out_col="throughput_sum"):
    """
    df_summary: DataFrame with a time column (e.g., CU_time)
    rows_through: list[dict], each dict has {abs_time: ..., throughput: ...}

    Rule:
      For each row in df_summary, sum all dict throughputs whose abs_time <= df_time
      AND greater than the previous df_time (so each df row gets the sum in its interval).

    Adds:
      out_col: throughput summed per df row interval
      throughput_cum (optional): cumulative throughput up to that row time
    """

    df_out = df_summary.copy()

    # Ensure datetime and sort df_summary by time
    df_out[df_time_col] = pd.to_datetime(df_out[df_time_col])
    df_out = df_out.sort_values(df_time_col).reset_index(drop=True)

    # Build a DataFrame from list-of-dicts
    thr_df = pd.DataFrame(rows_through).copy()
    if thr_df.empty:
        df_out[out_col] = 0.0
        df_out["throughput_cum"] = 0.0
        return df_out

    thr_df[dict_time_key] = pd.to_datetime(thr_df[dict_time_key])
    thr_df[dict_value_key] = pd.to_numeric(thr_df[dict_value_key], errors="coerce").fillna(0.0)
    thr_df = thr_df.sort_values(dict_time_key).reset_index(drop=True)

    # If multiple entries share the same timestamp, sum them first
    thr_df = thr_df.groupby(dict_time_key, as_index=False)[dict_value_key].sum()

    # Cumulative throughput over time
    thr_df["throughput_cum"] = thr_df[dict_value_key].cumsum()

    # As-of merge: for each df row time, find last thr_df time <= it
    merged = pd.merge_asof(
        df_out[[df_time_col]],
        thr_df[[dict_time_key, "throughput_cum"]].rename(columns={dict_time_key: df_time_col}),
        on=df_time_col,
        direction="backward",
    )

    # Fill NaNs (means no abs_time <= row time)
    merged["throughput_cum"] = merged["throughput_cum"].fillna(0.0)

    # Per-row sum: difference of cumulative sums between rows
    df_out["throughput_cum"] = merged["throughput_cum"].values
    df_out[out_col] = df_out["throughput_cum"].diff().fillna(df_out["throughput_cum"]).astype(float)

    return df_out


def parse_experiment_label(label: str):
    """
    New format supports:
      ..._<x>c<y>_<socStart>soc<socEnd>_dyn   (dyn optional)
      ..._<pause>h                           (pause optional)

    Extracts:
      - soc_start, soc_end
      - c_rate_chg (x), c_rate_dchg (y)
      - dyn (bool)
      - pause (int)
    """

    result = {
        "soc_start": None,
        "soc_end": None,
        "c_rate_chg": None,
        "c_rate_dchg": None,
        "dyn": False,
        "pause": None,
    }

    lab = label.lower()

    # ---------- dyn flag ----------
    result["dyn"] = bool(re.search(r"(^|_)dyn($|_)", lab))

    # ---------- SOC: _20soc80_ ----------
    m_soc = re.search(r"_(\d+)soc(\d+)_", lab)
    if m_soc:
        result["soc_start"] = int(m_soc.group(1))
        result["soc_end"] = int(m_soc.group(2))

    # ---------- C-rate: _xcy_ ----------
    m_c = re.search(r"_(\d+)c(\d+)_", lab)
    if m_c:
        def parse_c(val: str):
            return int(val) / 10 if val.startswith("0") and len(val) > 1 else float(val)

        result["c_rate_chg"] = parse_c(m_c.group(1))
        result["c_rate_dchg"] = parse_c(m_c.group(2))

    # ---------- pause: _10h (must be at end) ----------
    m_pause = re.search(r"_(\d+)h$", lab)
    if m_pause:
        result["pause"] = int(m_pause.group(1))

    return result

