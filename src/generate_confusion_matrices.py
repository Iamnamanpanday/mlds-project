import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "compressed"
RESULTS = BASE / "results"

TRAIN_FILE = DATA / "train_1600.xlsx"
TEST_FILE = DATA / "test_1200.xlsx"

PROPOSED_FILE = RESULTS / "proposed_integer_pca_matrix.npy"

SCALE = 128
WARMUP = 5
REPEATS = 50


def load_data(file):
    df = pd.read_excel(file)

    X = df.drop(columns=["Activity"]).values.astype(np.float32)
    y = df["Activity"].values

    return X, y


print("Loading datasets...")

X_train, y_train = load_data(TRAIN_FILE)
X_test, y_test = load_data(TEST_FILE)

print("Training:", X_train.shape)
print("Testing :", X_test.shape)


print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


print("Training Float PCA...")

pca = PCA(n_components=0.95)

pca.fit(X_train_s)

W_float = pca.components_.astype(np.float32)

print("PCA components:", W_float.shape[0])
print(
    "Explained variance:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)


print("\nCreating Naive INT8 matrix...")

W_naive = np.round(
    W_float * SCALE
).astype(np.int8)


print("Loading Proposed INT8 matrix...")

W_proposed = np.load(PROPOSED_FILE).astype(np.int8)


def project_float(X):
    return X @ W_float.T


def project_naive(X):
    return (X @ W_naive.T) / SCALE


def project_proposed(X):
    return (X @ W_proposed.T) / SCALE


def benchmark_projection(function, X):

    for _ in range(WARMUP):
        function(X)

    times = []

    for _ in range(REPEATS):

        start = time.perf_counter()

        function(X)

        end = time.perf_counter()

        times.append(end - start)

    times = np.array(times)

    return {
        "mean_ms": times.mean() * 1000,
        "min_ms": times.min() * 1000,
        "max_ms": times.max() * 1000
    }


def matrix_memory(W):
    return W.nbytes


def approximation_error(W_int):

    W_quantized = W_int.astype(np.float32) / SCALE

    return (
        np.linalg.norm(W_float - W_quantized)
        / np.linalg.norm(W_float)
    ) * 100


print("\nBenchmarking projection latency...")


float_time = benchmark_projection(
    project_float,
    X_test_s
)

naive_time = benchmark_projection(
    project_naive,
    X_test_s
)

proposed_time = benchmark_projection(
    project_proposed,
    X_test_s
)


float_memory = matrix_memory(W_float)

naive_memory = matrix_memory(W_naive)

proposed_memory = matrix_memory(W_proposed)


naive_error = approximation_error(W_naive)

proposed_error = approximation_error(W_proposed)


print("\n==============================")
print(" MEMORY / ROM")
print("==============================")

print(
    "Float PCA:",
    round(float_memory / 1024, 2),
    "KiB"
)

print(
    "Naive INT8:",
    round(naive_memory / 1024, 2),
    "KiB"
)

print(
    "Proposed INT8:",
    round(proposed_memory / 1024, 2),
    "KiB"
)


print("\n==============================")
print(" APPROXIMATION ERROR")
print("==============================")

print(
    "Naive INT8:",
    round(naive_error, 4),
    "%"
)

print(
    "Proposed INT8:",
    round(proposed_error, 4),
    "%"
)


print("\n==============================")
print(" PROJECTION LATENCY")
print("==============================")

print(
    "Float PCA:",
    round(float_time["mean_ms"], 4),
    "ms"
)

print(
    "Naive INT8:",
    round(naive_time["mean_ms"], 4),
    "ms"
)

print(
    "Proposed INT8:",
    round(proposed_time["mean_ms"], 4),
    "ms"
)


print("\n==============================")
print(" LATENCY RANGE")
print("==============================")

print(
    "Float PCA:",
    round(float_time["min_ms"], 4),
    "-",
    round(float_time["max_ms"], 4),
    "ms"
)

print(
    "Naive INT8:",
    round(naive_time["min_ms"], 4),
    "-",
    round(naive_time["max_ms"], 4),
    "ms"
)

print(
    "Proposed INT8:",
    round(proposed_time["min_ms"], 4),
    "-",
    round(proposed_time["max_ms"], 4),
    "ms"
)


print("\n==============================")
print(" STORAGE REDUCTION")
print("==============================")

naive_reduction = (
    1 - naive_memory / float_memory
) * 100

proposed_reduction = (
    1 - proposed_memory / float_memory
) * 100

print(
    "Naive INT8:",
    round(naive_reduction, 2),
    "%"
)

print(
    "Proposed INT8:",
    round(proposed_reduction, 2),
    "%"
)


print("\n==============================")
print(" CLASSIFICATION ACCURACY")
print("==============================")


Z_train_float = project_float(X_train_s)
Z_test_float = project_float(X_test_s)

Z_train_naive = project_naive(X_train_s)
Z_test_naive = project_naive(X_test_s)

Z_train_proposed = project_proposed(X_train_s)
Z_test_proposed = project_proposed(X_test_s)


def train_and_test(Z_train, Z_test):

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(Z_train, y_train)

    prediction = clf.predict(Z_test)

    return accuracy_score(
        y_test,
        prediction
    ) * 100


float_accuracy = train_and_test(
    Z_train_float,
    Z_test_float
)

naive_accuracy = train_and_test(
    Z_train_naive,
    Z_test_naive
)

proposed_accuracy = train_and_test(
    Z_train_proposed,
    Z_test_proposed
)


print(
    "Float PCA:",
    round(float_accuracy, 2),
    "%"
)

print(
    "Naive INT8:",
    round(naive_accuracy, 2),
    "%"
)

print(
    "Proposed INT8:",
    round(proposed_accuracy, 2),
    "%"
)


print("\n==============================")
print(" FINAL COMPARISON")
print("==============================")

print(
    f"{'Method':<20}"
    f"{'Accuracy':<12}"
    f"{'Error':<12}"
    f"{'Memory':<14}"
    f"{'Latency(ms)':<15}"
)

print("-" * 73)

print(
    f"{'Float PCA':<20}"
    f"{float_accuracy:.2f}%"
    f"{'0.0000%':<12}"
    f"{float_memory / 1024:.2f} KiB"
    f"{float_time['mean_ms']:.4f}"
)

print(
    f"{'Naive INT8':<20}"
    f"{naive_accuracy:.2f}%"
    f"{naive_error:.4f}%"
    f"{naive_memory / 1024:.2f} KiB"
    f"{naive_time['mean_ms']:.4f}"
)

print(
    f"{'Proposed INT8':<20}"
    f"{proposed_accuracy:.2f}%"
    f"{proposed_error:.4f}%"
    f"{proposed_memory / 1024:.2f} KiB"
    f"{proposed_time['mean_ms']:.4f}"
)


print("\nBenchmark complete.")