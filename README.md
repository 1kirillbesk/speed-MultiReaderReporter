# SPEED MultiReaderReporter

**SPEED MultiReaderReporter** is a unified tool for processing and analyzing battery experiment data stored in  
- **MATLAB `.mat` struct files**,  
- **CSV files (inside `.zip` archives or standalone)**, and  
- **Python pickle (`.pkl`) files**.  

It standardizes all data into a single format and produces per-cell plots, reports, and optional State-of-Health (SoH) analyses — all **fully configurable through `config.yaml`**. Inputs belonging to the same cell are automatically batched and processed together to keep memory bounded and to prevent plot/report overwrites when multiple raw files exist for one cell.

---

## Overview

SPEED MultiReaderReporter automatically:
- Detects input file formats (`.mat`, `.csv`, `.zip`, `.pkl`)
- Loads **all inputs for the same cell together**, processing one cell at a time to bound RAM and isolate outputs
- Standardizes each dataset into a common structure:
  - `abs_time` — absolute timestamp  
  - `current_A` — current in amperes 
  - `voltage_V` - voltage in volts
  - (optional) `step_int`
- Classifies experiments into **checkup** and **cycling**
- Generates:
  - Strom-vs-Zeit and Spannung-vs-Zeit plots for total, checkup, and cycling data
  - Throughput and current statistics (`report.csv` or `.mat`)
  - Optional **checkup-segment breakdowns** that split pauses into voltage-defined groups
  - Optional **SoH plots**: discharge capacity (step-19) vs cumulative charge throughput

Everything — input locations, classification rules, report format, and thresholds — is controlled through a single YAML configuration file.

---

## ⚙️ How to Use

### 1️⃣ Install dependencies

You’ll need **Python 3.10+**.  
Install dependencies using:

```bash
pip install -r requirements.txt
````

### 2️⃣ Configure `config.yaml`

Open the file `speed_MultiReaderReporter/config.yaml`.
It defines how the program behaves — no code changes required.

Example configuration:

```yaml
input:
  path: "./data"           # Folder or single file to process
  recurse: true            # Scan subfolders for data files

output:
  root: "./speed_MultiReaderReporter/out"  # Where to save results

classification:
  cycling_keywords: ["cyc"]                  # Force cycling when present in the name
  checkup_keywords: ["cu", "glu", "rpt"]   # Keywords to identify checkups
  duration_threshold_minutes: 60           # < 1 h = checkup
  step_min_required: 20                    # min step number to consider
  require_steps_19_22: true                # require both steps 19 & 22
  skip_glu: false                          # skip 'glu' keyword if true

soh:
  min_step_required: 20
  i_thresh_A: 0.0
  eod_v_cut_V: null                        # cutoff voltage (optional)

legend:
  ncol: 4                                  # legend columns on plots

reports:
  format: "csv"                            # csv | mat | both
  mat_variable: "report"                   # MATLAB struct variable name

logging:
  verbose: true
```

### 3️⃣ Run the program

From the project root:

```bash
cd speed_MultiReaderReporter
python main.py
```

The program will:

1. Read `config.yaml`
2. Automatically detect the file format of each input
3. Group inputs by cell and load only one cell’s data at a time
4. Standardize and process aggregated runs through the same pipeline
5. Save plots and reports into per-cell folders under the output directory

No command-line arguments are required — **the entire workflow is controlled via the YAML file**.

---

## 📁 Output Structure

Each cell’s results are saved to its own folder:

```
out/<CELL_NAME>/
  total/
    strom_vs_zeit.png
    report.csv or report.mat
  checkup/
    strom_vs_zeit.png
    report.csv or report.mat
    groups/
      strom_vs_zeit_[group-index].png         # if grouping plots are enabled
    soh_discharge_capacity_vs_throughput.csv
    soh_discharge_capacity_vs_throughput.png
  cycling/
    strom_vs_zeit.png
    report.csv or report.mat
```

* `CELL_NAME` is automatically extracted between the **first and second `=`** in each filename.
* The `out/` folder itself remains tracked in Git, but its contents are ignored (to avoid committing large data).

---

## 🔍 How Classification Works

The classification logic uses a combination of:

1. **Cycling keywords (override)** — if the program name contains any configured `cycling_keywords` (defaults to `cyc`), the run is forced to *cycling* and no checkup heuristics are evaluated.
2. **Program keywords** — if the filename or program name contains any of the configured `checkup_keywords` (e.g. `cu`, `glu`, `rpt`), the run is marked as a *checkup*.
3. **Step-aware rules** — if `step_int` exists, the run is considered a *checkup* if:

   * `max(step)` ≥ `step_min_required`, and
   * steps `19` and `22` are both present (if enabled).
4. **Duration rule** — any run shorter than `duration_threshold_minutes` is treated as a *checkup*.
5. **Otherwise** — it’s a *cycling* run.

Examples:

* `rul_inhomoSAM_cyc_1C_4OSOC100` → *cycling* even if it contains diagnostic-style steps or short durations because `cyc` matches `cycling_keywords`.
* `reference_cu_profile` → *checkup* when no cycling keyword is present and the usual keyword/step/duration rules apply.

These parameters can all be customized in `config.yaml` under the `classification:` section.

---

## 🔀 Checkup Grouping & Segment Reports

Some checkups contain repeated pause/charge cycles that need to be evaluated independently.
The `checkup_grouping` section in `config.yaml` enables automatic segmentation so plots and
reports reflect those finer-grained groups.

| Key | Description |
| --- | ----------- |
| `mode` | `off` (default), `plot`, `report`, or `both`. Controls whether segmentation is calculated, plotted, and/or included in reports. |
| `pause_label` | Pause state label to anchor each group. Defaults to `PAU`. |
| `min_points` | Minimum rows per segment; shorter pauses are merged forward. |
| `voltage_windows` | Acceptable pause-voltage ranges (defaults to `[[1.9, 2.1], [3.55, 3.65]]`). Groups only start when the first pause inside a new step lands within one of these windows. |
| `require_step_change` | When `true`, segments only start at step changes; the first pause within the new step is selected automatically. |
| `plot_max_points_per_segment` | Optional decimation for faster plotting of large pauses. |

Additional behavior:

* Checkups whose labels contain `glu` or `aer` are intentionally left unsplit so they retain their legacy behavior.
* Segment plots inherit the parent checkup name, and when multiple runs share a name the files are numbered to avoid overwriting.
* When grouping is enabled for reports, each segment contributes its own row alongside the per-checkup totals in both CSV and MATLAB exports.

---

## 🧪 Supported File Types

| Format          | Typical Source              | Loader          | Notes                              |
| --------------- | --------------------------- | --------------- | ---------------------------------- |
| `.mat`          | MATLAB `diga.daten` exports | `mat_loader`    | Full support for steps & voltage   |
| `.zip` / `.csv` | Machine exports             | `csvzip_loader` | Reads absolute time & current      |
| `.pkl`          | Python pickle RPT exports   | `pkl_loader`    | Treated as checkup runs by default |


---

## Changelog

- Aggregate all inputs per cell and run the pipeline once per cell to avoid overwriting outputs and to bound memory usage.
- Add `classification.cycling_keywords` (defaults to `cyc`) to force cycling classification before checkup heuristics.
- Update documentation and configuration examples to reflect the per-cell batching flow and cycling override.
