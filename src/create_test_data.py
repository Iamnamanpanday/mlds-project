import pandas as pd
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "test.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "compressed"
OUTPUT_FILE = OUTPUT_DIR / "test_100.xlsx"

# Create output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD TEST DATA
# ==========================================

print("\nLoading test dataset...")

if not INPUT_FILE.exists():
    print("\n❌ test.xlsx was not found!")
    print("Put test.xlsx inside the data folder:")
    print(BASE_DIR / "data")
    exit()

df = pd.read_excel(INPUT_FILE)

print("\nOriginal test dataset:")
print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ==========================================
# CHECK SIZE
# ==========================================

if len(df) < 100:
    print("\n❌ Test dataset contains fewer than 100 records.")
    print("Available:", len(df))
    exit()


# ==========================================
# SELECT 100 RECORDS
# ==========================================

test_100 = df.sample(
    n=100,
    random_state=42
).reset_index(drop=True)


# ==========================================
# SAVE
# ==========================================

test_100.to_excel(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# DISPLAY INFORMATION
# ==========================================

print("\n================================")
print("TEST DATASET CREATED")
print("================================")

print("Rows    :", len(test_100))
print("Columns :", len(test_100.columns))

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst 5 records:")
print(test_100.head())

print("\n================================")
print("DONE!")
print("================================")