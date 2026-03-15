import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = "data/processed/nutrition5k_master.csv"
OUT_DIR = Path("data/processed")

SEED = 42
TEST_SIZE = 1000

df = pd.read_csv(INPUT_PATH)

rng = np.random.default_rng(SEED)
indices = rng.permutation(len(df))

test_indices = indices[:TEST_SIZE]
train_indices = indices[TEST_SIZE:]

test_df = df.iloc[test_indices].copy()
train_df = df.iloc[train_indices].copy()

train_path = OUT_DIR / "train_set.csv"
test_path = OUT_DIR / "test_set.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("total:", len(df))
print("train:", len(train_df))
print("test:", len(test_df))
print("Saved:", train_path)
print("Saved:", test_path)