import pandas as pd

# Load the CSV file
df = pd.read_csv("C:/Users/Public/Documents/takedata/Speed/out_lw/SPEED_LW_reference_1/total/report.csv")   # or the full path if needed

# Show first few rows
print(df.head())

# Show columns
print(df.columns)
