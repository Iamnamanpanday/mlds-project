import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

SOURCE = BASE / "data" / "test.xlsx"
OUTPUT = BASE / "data" / "compressed" / "test_1200.xlsx"

df = pd.read_excel(SOURCE)

print("Original test dataset:", df.shape)

if len(df) < 1200:
    raise ValueError(f"Only {len(df)} rows available. Need at least 1200.")

test_1200 = df.iloc[:1200].copy()

test_1200.reset_index(drop=True, inplace=True)

test_1200.to_excel(
    OUTPUT,
    index=False
)

print("\nTest dataset created successfully")
print("Shape:", test_1200.shape)
print("Saved to:", OUTPUT)