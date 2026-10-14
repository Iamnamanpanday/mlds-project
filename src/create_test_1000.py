import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "test.xlsx"
OUTPUT_FILE = BASE_DIR / "data" / "compressed" / "test_1000.xlsx"

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_excel(INPUT_FILE)

print("Original test records:", len(df))

if len(df) < 1000:
    raise ValueError(
        f"Only {len(df)} test records available."
    )

test_1000 = df.sample(
    n=1000,
    random_state=42
).reset_index(drop=True)

test_1000.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\n==============================")
print("TEST DATASET CREATED")
print("==============================")
print("Rows:", len(test_1000))
print("Columns:", len(test_1000.columns))
print("Saved:", OUTPUT_FILE)