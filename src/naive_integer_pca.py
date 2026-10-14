import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "compressed" / "train_1600.xlsx"
TEST_FILE = BASE_DIR / "data" / "compressed" / "test_1200.xlsx"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_excel(TRAIN_FILE)
test_df = pd.read_excel(TEST_FILE)

print("Training shape:", train_df.shape)
print("Testing shape :", test_df.shape)


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = train_df.columns[-1]

print("\nTarget column:", TARGET_COLUMN)


# ============================================================
# FEATURES / TARGET
# ============================================================

X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]


# Keep numeric columns only
numeric_columns = X_train.select_dtypes(
    include=np.number
).columns

X_train = X_train[numeric_columns]
X_test = X_test[numeric_columns]


print("Number of features:", X_train.shape[1])


# ============================================================
# STANDARDIZATION
# ============================================================

print("\nStandardizing data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# FLOAT PCA
# ============================================================

print("\nTraining floating-point PCA...")

pca = PCA(
    n_components=0.95,
    random_state=42
)

X_train_float = pca.fit_transform(X_train_scaled)
X_test_float = pca.transform(X_test_scaled)

W_float = pca.components_

print("\n========== FLOAT PCA ==========")
print("Original dimensions :", X_train.shape[1])
print("PCA dimensions      :", pca.n_components_)
print(
    "Explained variance  :",
    round(np.sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)


# ============================================================
# NAIVE INTEGER QUANTIZATION
# ============================================================

# Quantization scale
SCALE = 128

print("\n========== INTEGER QUANTIZATION ==========")
print("Scale:", SCALE)

# Convert floating-point PCA matrix into integers
W_int = np.round(W_float * SCALE).astype(np.int8)

# Convert back to floating point for comparison
W_dequantized = W_int.astype(np.float32) / SCALE


# ============================================================
# APPROXIMATION ERROR
# ============================================================

numerator = np.linalg.norm(
    W_float - W_dequantized,
    ord="fro"
)

denominator = np.linalg.norm(
    W_float,
    ord="fro"
)

relative_error = numerator / denominator

print(
    "Relative PCA approximation error:",
    round(relative_error * 100, 4),
    "%"
)


# ============================================================
# INTEGER PCA TRANSFORMATION
# ============================================================

print("\nTransforming data using integer PCA matrix...")

# Matrix multiplication using the quantized PCA matrix
#
# X_scaled @ W_int.T
#
# gives integer-like projected values.
#
# Divide by SCALE to recover approximately
# the same numerical range as normal PCA.

X_train_integer = (
    X_train_scaled @ W_int.astype(np.float32).T
) / SCALE

X_test_integer = (
    X_test_scaled @ W_int.astype(np.float32).T
) / SCALE


# ============================================================
# CLASSIFIER
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_integer,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test_integer)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n======================================")
print("       NAIVE INTEGER PCA")
print("======================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========\n")

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    f"Naive Integer PCA + Random Forest\n"
    f"INT8, Scale={SCALE}"
)

plt.tight_layout()

output_file = RESULTS_DIR / "naive_integer_confusion_matrix.png"

plt.savefig(
    output_file,
    dpi=300
)

plt.show()


# ============================================================
# SAVE INTEGER PCA MATRIX
# ============================================================

np.save(
    RESULTS_DIR / "naive_integer_pca_matrix.npy",
    W_int
)


# ============================================================
# SAVE METRICS
# ============================================================

print("\n========== RESULTS ==========")

print("Float PCA dimensions :", pca.n_components_)
print("Integer data type    :", W_int.dtype)
print("Quantization scale   :", SCALE)
print(
    "Relative error       :",
    round(relative_error * 100, 4),
    "%"
)
print(
    "Classification accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nConfusion matrix saved:")
print(output_file)

print("\nNaive Integer PCA complete!")