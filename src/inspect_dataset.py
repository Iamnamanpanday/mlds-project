import pandas as pd
from pathlib import Path

# Get project folder automatically
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "train.xlsx"

print("\n========== DATASET INSPECTION ==========\n")

print("File:")
print(DATA_PATH)

# Check if file exists
if not DATA_PATH.exists():
    print("\n❌ train.xlsx not found!")
    exit()

# Read Excel
df = pd.read_excel(DATA_PATH)

print("\n========== SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DONE ==========")