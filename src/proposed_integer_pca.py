import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "compressed"
RESULTS = BASE / "results"

RESULTS.mkdir(exist_ok=True)

TRAIN_FILE = DATA / "train_1600.xlsx"
VAL_FILE = DATA / "validation_500.xlsx"
TEST_FILE = DATA / "test_1200.xlsx"

SCALE = 128
PASSES = 2
COMPONENTS_TO_OPTIMIZE = 20

LAMBDA_ORTH = 0.05


def load_data(file):
    df = pd.read_excel(file)
    X = df.drop(columns=["Activity"]).values.astype(np.float32)
    y = df["Activity"].values
    return X, y


X_train, y_train = load_data(TRAIN_FILE)
X_val, y_val = load_data(VAL_FILE)
X_test, y_test = load_data(TEST_FILE)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)


pca = PCA(n_components=0.95)

pca.fit(X_train_s)

W_float = pca.components_.copy()

Z_train_float = pca.transform(X_train_s)
Z_val_float = pca.transform(X_val_s)

k = W_float.shape[0]

print("\nPCA components:", k)
print(
    "Explained variance:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)


W_int = np.round(W_float * SCALE).astype(np.int8)


def approximation_error(W):
    Wq = W.astype(np.float32) / SCALE

    return (
        np.linalg.norm(W_float - Wq)
        / np.linalg.norm(W_float)
    )


def orthogonality_error(W):
    Wq = W.astype(np.float32) / SCALE

    I = np.eye(Wq.shape[0])

    return (
        np.linalg.norm(Wq @ Wq.T - I)
        / np.sqrt(Wq.shape[0])
    )


def objective(W):

    error = approximation_error(W)

    orth = orthogonality_error(W)

    return error + LAMBDA_ORTH * orth


initial_error = approximation_error(W_int)
initial_orth = orthogonality_error(W_int)
initial_objective = objective(W_int)


print("\n========== INITIAL ==========")

print(
    "Approximation error:",
    initial_error * 100,
    "%"
)

print(
    "Orthogonality error:",
    initial_orth
)

print(
    "Initial objective:",
    initial_objective
)


start = time.perf_counter()

for p in range(PASSES):

    changes = 0

    for i in range(min(COMPONENTS_TO_OPTIMIZE, k)):

        for j in range(W_int.shape[1]):

            current = W_int[i, j]

            best_value = current
            best_objective = objective(W_int)

            for candidate in [
                current - 1,
                current + 1
            ]:

                if candidate < -128 or candidate > 127:
                    continue

                W_int[i, j] = candidate

                new_objective = objective(W_int)

                if new_objective < best_objective:

                    best_objective = new_objective
                    best_value = candidate

                W_int[i, j] = current

            if best_value != current:

                W_int[i, j] = best_value
                changes += 1

    print(
        f"Pass {p + 1}/{PASSES}:",
        changes,
        "changes"
    )


optimization_time = time.perf_counter() - start


final_error = approximation_error(W_int)
final_orth = orthogonality_error(W_int)
final_objective = objective(W_int)


print("\n========== OPTIMIZATION RESULTS ==========")

print(
    "Initial approximation error:",
    round(initial_error * 100, 6),
    "%"
)

print(
    "Final approximation error:",
    round(final_error * 100, 6),
    "%"
)

print(
    "Initial orthogonality error:",
    round(initial_orth, 6)
)

print(
    "Final orthogonality error:",
    round(final_orth, 6)
)

print(
    "Initial objective:",
    round(initial_objective, 8)
)

print(
    "Final objective:",
    round(final_objective, 8)
)

print(
    "Optimization time:",
    round(optimization_time, 2),
    "seconds"
)


def project(X, W):

    return (X @ W.T) / SCALE


Z_train_int = project(X_train_s, W_int)
Z_val_int = project(X_val_s, W_int)
Z_test_int = project(X_test_s, W_int)


clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

clf.fit(Z_train_int, y_train)


val_pred = clf.predict(Z_val_int)

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

print(
    "\nValidation accuracy:",
    round(val_accuracy * 100, 2),
    "%"
)


test_start = time.perf_counter()

y_pred = clf.predict(Z_test_int)

test_time = time.perf_counter() - test_start


test_accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========== FINAL TEST ==========")

print(
    "Test accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print(
    "Prediction time:",
    round(test_time, 4),
    "seconds"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


np.save(
    RESULTS / "proposed_integer_pca_matrix.npy",
    W_int
)

np.save(
    RESULTS / "proposed_float_pca_components.npy",
    W_float
)

print("\nSaved optimized PCA matrix.")