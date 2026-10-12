
import pandas as pd
from pathlib import Path

dataset_dir = Path("Dataset")
csv_files = sorted(dataset_dir.glob("*.csv"))

print("Number of CSV files found:", len(csv_files))

if not csv_files:
    print("ERROR: CSV files not found inside Dataset folder.")
else:
    for file in csv_files:
        print("\nFile:", file.name)

        df = pd.read_csv(file, nrows=5, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        print("Preview shape:", df.shape)
        print("First 5 columns:", df.columns[:5].tolist())
        print("Last column:", repr(df.columns[-1]))
        print(df.head(2))
