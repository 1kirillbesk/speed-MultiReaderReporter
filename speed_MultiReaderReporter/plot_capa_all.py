from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# input / output folders
data_dir = Path(r"C:\Users\Public\Documents\RL_project\out_lw\cell_feature")
out_dir = Path(r"C:\Users\Public\Documents\RL_project\out_lw\feature_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# columns to look for
cap_cols = ["cap_dis", "cap_cha", "cap_ocv_dis", "cap_ocv_cha"]

for csv_file in data_dir.glob("*.csv"):
    df = pd.read_csv(csv_file)

    cols_present = [c for c in cap_cols if c in df.columns]
    if not cols_present:
        print(f"Skipping {csv_file.name} (no cap columns found)")
        continue

    plt.figure(figsize=(8, 5))

    for col in cols_present:
        y = pd.to_numeric(df[col], errors="coerce")
        plt.plot(y, label=col)

    plt.title(csv_file.name)
    plt.xlabel("Index")
    plt.ylabel("Capacity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save figure
    out_path = out_dir / f"{csv_file.stem}_capacity.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")
