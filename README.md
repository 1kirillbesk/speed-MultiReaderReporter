# SPEED MultiReaderReporter

Unified tool for processing battery experiment data stored either in **MATLAB `.mat`** struct files (diga.daten) or **CSV files inside ZIP archives** (machine-exported).  
It standardizes everything into one format and generates per-cell plots and reports.

## ✳️ What it does

- Auto-detects inputs: **`.mat`**, **`.csv`**, **`.pkl`** or **`.zip`** (containing CSVs)
- Standardizes each run to a canonical dataframe:
  - `abs_time` (datetime), `current_A` (float)
  - optional: `voltage_V`, `step_int` (exact “Schritt” only)
- Classifies runs into **checkup** vs **cycling** (keywords + step-aware + duration fallback)
- Per-cell outputs:
  - Strom vs Zeit plots for **total**, **checkup**, **cycling**
  - Throughput reports (CSV and/or MAT struct)
  - **SoH** scatter: **discharge capacity (step-19)** vs **cumulative charge throughput** up to that discharge

## 📦 Requirements

- Python 3.10+ recommended
- See `requirements.txt` for packages

Install:
```bash
pip install -r requirements.txt
