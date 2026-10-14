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



BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "compressed" / "train_1600.xlsx"
TEST_FILE = BASE_DIR / "data" / "compressed" / "test_1200.xlsx"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)



print("\nLoading datasets...")

train_df = pd.read_excel(TRAIN_FILE)
test_df = pd.read_excel(TEST_FILE)

print("Training shape :", train_df.shape)
print("Testing shape  :", test_df.shape)




print("\nColumns:")
print(train_df.columns.tolist())

# IMPORTANT:
# If your activity/target column has a different name,
# change TARGET_COLUMN below.

TARGET_COLUMN = train_df.columns[-1]

print("\nTarget column:", TARGET_COLUMN)


X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]


print("\nNumber of features:", X_train.shape[1])
print("Training samples :", X_train.shape[0])
print("Testing samples  :", X_test.shape[0])




numeric_columns = X_train.select_dtypes(
    include=np.number
).columns

X_train = X_train[numeric_columns]
X_test = X_test[numeric_columns]


print("\nNumeric features:", X_train.shape[1])




print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




print("\nApplying PCA...")

pca = PCA(
    n_components=0.95,
    random_state=42
)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


print("\n========== PCA RESULTS ==========")

print("Original features :", X_train.shape[1])
print("PCA components    :", pca.n_components_)
print(
    "Explained variance:",
    round(np.sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)




print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_pca, y_train)




print("\nMaking predictions...")

y_pred = model.predict(X_test_pca)



accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("       FLOAT PCA BASELINE")
print("======================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)




print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)




cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========\n")

print(cm)



disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Float PCA + Random Forest")

plt.tight_layout()

output_file = RESULTS_DIR / "baseline_confusion_matrix.png"

plt.savefig(
    output_file,
    dpi=300
)

plt.show()



pca_matrix_file = RESULTS_DIR / "float_pca_components.npy"

np.save(
    pca_matrix_file,
    pca.components_
)

print("\nPCA matrix saved to:")
print(pca_matrix_file)

print("\nConfusion matrix saved to:")
print(output_file)

print("\n BASELINE COMPLETE")