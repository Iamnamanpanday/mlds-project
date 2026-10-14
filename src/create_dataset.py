import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE = BASE_DIR / "data" / "train.xlsx"
OUTPUT = BASE_DIR / "data" / "compressed"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(SOURCE)

print("Original dataset:", df.shape)

if len(df) < 2100:
    raise ValueError(
        f"Dataset only has {len(df)} rows. "
        "Need at least 2100."
    )

train = df.iloc[0:1600].copy()
validation = df.iloc[1600:2100].copy()

train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)

train.to_excel(
    OUTPUT / "train_1600.xlsx",
    index=False
)

validation.to_excel(
    OUTPUT / "validation_500.xlsx",
    index=False
)

print("\n==============================")
print("DATASETS CREATED")
print("==============================")

print(
    "Training:",
    train.shape,
    "→ train_1600.xlsx"
)

print(
    "Validation:",
    validation.shape,
    "→ validation_500.xlsx"
)

print("\nTraining rows: 1–1600")
print("Validation rows: 1601–2100")

print("\nDONE!")