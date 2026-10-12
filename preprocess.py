
import pandas as pd
import numpy as np
from pathlib import Path

# Project folders
DATASET_DIR = Path("Dataset")
OUTPUT_DIR = Path("processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

csv_files = sorted(DATASET_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV files found inside Dataset folder."
    )

print("CSV files found:", len(csv_files))

# 1. Load and combine all CSV files
dataframes = []

for file in csv_files:
    print("Reading:", file.name)

    df = pd.read_csv(
        file,
        encoding="utf-8-sig",
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(f"Label column missing in {file.name}")

    df["Label"] = df["Label"].astype(str).str.strip()
    dataframes.append(df)

# 2. Verify column consistency
expected_columns = list(dataframes[0].columns)

for df in dataframes[1:]:
    if list(df.columns) != expected_columns:
        raise ValueError(
            "CSV files have different columns or column order."
        )

data = pd.concat(dataframes, ignore_index=True)
del dataframes

print("\nCombined dataset shape:", data.shape)

# 3. Clean column names
data.columns = data.columns.str.strip()
data = data.loc[:, ~data.columns.duplicated()]

# 4. Convert features to numeric
feature_columns = [
    col for col in data.columns if col != "Label"
]

for col in feature_columns:
    data[col] = pd.to_numeric(
        data[col], errors="coerce"
    )

# Replace infinite and missing values with zero
data[feature_columns] = data[feature_columns].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

print("Missing and infinite values handled.")

# 5. Remove duplicate rows
before = len(data)
data = data.drop_duplicates().reset_index(drop=True)

print("Duplicate rows removed:", before - len(data))

# 6. Remove all-zero features
feature_columns = [
    col for col in data.columns if col != "Label"
]

zero_columns = [
    col for col in feature_columns
    if (data[col] == 0).all()
]

data = data.drop(columns=zero_columns)

print("All-zero features removed:", zero_columns)
print("Shape after cleaning:", data.shape)

# 7. Convert labels into binary classes
data["Label"] = data["Label"].astype(str).str.strip()

data = data[data["Label"].ne("")].copy()

data["Label"] = np.where(
    data["Label"].str.upper() == "BENIGN",
    0,
    1
).astype(np.int32)

print("\nClass distribution:")
print(data["Label"].value_counts().sort_index())
print("Label mapping: Benign = 0, Malicious = 1")

# 8. Save cleaned data WITHOUT scaling
output_file = OUTPUT_DIR / "cicids2017_preprocessed.csv"

data.to_csv(output_file, index=False)

print("\nPreprocessing completed!")
print("Final dataset shape:", data.shape)
print("Features:", len(data.columns) - 1)
print("Scaling: NOT performed")
print("Saved to:", output_file)
