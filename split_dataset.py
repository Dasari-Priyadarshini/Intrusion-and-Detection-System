
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------
# 1. Configuration
# -----------------------------------------
INPUT_FILE = Path(
    "processed_data/cicids2017_preprocessed.csv"
)

OUTPUT_DIR = Path("split_data")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42

# -----------------------------------------
# 2. Load cleaned dataset
# -----------------------------------------
print("Loading preprocessed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

X = df.drop(columns=["Label"])
y = df["Label"].astype(np.int32)

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors="raise")

# -----------------------------------------
# 3. Split: 70% train, 15% validation, 15% test
# -----------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("\nFitting scaler on training data only...")

# -----------------------------------------
# 4. Scale without data leakage
# -----------------------------------------
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# 5. Save split data
# -----------------------------------------
np.savez_compressed(
    OUTPUT_DIR / "train.npz",
    X=X_train_scaled.astype(np.float32),
    y=y_train.to_numpy(dtype=np.int32)
)

np.savez_compressed(
    OUTPUT_DIR / "validation.npz",
    X=X_val_scaled.astype(np.float32),
    y=y_val.to_numpy(dtype=np.int32)
)

np.savez_compressed(
    OUTPUT_DIR / "test.npz",
    X=X_test_scaled.astype(np.float32),
    y=y_test.to_numpy(dtype=np.int32)
)

# -----------------------------------------
# 6. Print dataset details
# -----------------------------------------
print("\nSplit completed successfully!")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation class distribution:")
print(y_val.value_counts().sort_index())

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())

print("\nScaling: fitted on training data only")
print("SMOTE: not used")
print("Saved files inside:", OUTPUT_DIR)
