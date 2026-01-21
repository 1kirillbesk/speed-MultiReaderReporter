# speed_MultiReaderReporter/main.py
from __future__ import annotations
from pathlib import Path
import sys
import yaml
import matplotlib.pyplot as plt
import pandas as pd

from loaders import csvzip_loader, mat_loader, pkl_loader
from utils.detect import discover_inputs
from core.pipeline import run_pipeline

# --- relative paths ---
here = Path(__file__).resolve().parent
sys.path.append(str(here))
sys.path.append(str(here / "core"))
sys.path.append(str(here / "loaders"))
sys.path.append(str(here / "utils"))

def main(dir_path):
    for csv_file in dir_path.glob("*.csv"):
        df = pd.read_csv(csv_file)
        df["cell_name"] = csv_file.name
        # plot the capacity


if __name__ == "__main__":
    dir_path = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
    main(dir_path)