import pandas as pd
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "train.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "compressed"
OUTPUT_FILE = OUTPUT_DIR / "train_1000.xlsx"

# Create compressed folder if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading dataset...")

df = pd.read_excel(INPUT_FILE)

print("Original dataset:")
print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ==========================================
# CHECK DATASET SIZE
# ==========================================

if len(df) < 1000:
    print("\n❌ Dataset contains fewer than 1000 rows.")
    print("Available rows:", len(df))
    exit()


# ==========================================
# CREATE 1000-ROW TRAINING DATASET
# ==========================================

train_1000 = df.sample(
    n=1000,
    random_state=42
).reset_index(drop=True)


# ==========================================
# SAVE
# ==========================================

train_1000.to_excel(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# SHOW RESULTS
# ==========================================

print("\n================================")
print("TRAINING DATASET CREATED")
print("================================")

print("Rows    :", len(train_1000))
print("Columns :", len(train_1000.columns))

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(train_1000.head())

print("\nDONE!")