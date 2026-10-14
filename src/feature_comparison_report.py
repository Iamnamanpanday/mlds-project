import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "compressed"
RESULTS = BASE / "results"
TRAIN_FILE = DATA / "train_1600.xlsx"
TEST_FILE = DATA / "test_1200.xlsx"
PROPOSED_FILE = RESULTS / "proposed_integer_pca_matrix.npy"
OUTPUT_FILE = RESULTS / "feature_comparison_report.json"
SCALE = 128
TIMING_REPEATS = 5


def load_data(path):
    frame = pd.read_excel(path)
    if "Activity" not in frame.columns:
        raise ValueError(f"Expected an Activity target column in {path}")
    features = frame.drop(columns=["Activity"]).to_numpy(dtype=np.float32)
    return features, frame["Activity"].to_numpy()


def random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


def build_model(
    name, note, components, classifier, X_train_scaled, X_test_raw, y_train, y_test, scaler,
    projection_storage_bytes=None,
):
    train_projection = X_train_scaled @ components.T
    model = classifier
    model.fit(train_projection, y_train)

    def predict():
        test_scaled = scaler.transform(X_test_raw)
        return model.predict(test_scaled @ components.T)

    predictions = predict()
    durations = []
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter()
        predict()
        durations.append(time.perf_counter() - started)

    return {
        "name": name,
        "note": note,
        "accuracyPct": float(accuracy_score(y_test, predictions) * 100),
        "weightedF1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "latencyMsPerSample": float(np.mean(durations) * 1000 / len(y_test)),
        "projectionBytes": int(components.nbytes if projection_storage_bytes is None else projection_storage_bytes),
    }


def main():
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and testing data have different feature counts")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=0.95, random_state=42).fit(X_train_scaled)
    float_components = pca.components_.astype(np.float32)
    naive_components = np.round(float_components * SCALE).astype(np.int8)
    proposed_components = np.load(PROPOSED_FILE).astype(np.int8)
    if proposed_components.shape != float_components.shape:
        raise ValueError(
            f"Proposed matrix shape {proposed_components.shape} does not match PCA shape {float_components.shape}"
        )

    proposed_float = proposed_components.astype(np.float32) / SCALE
    models = [
        build_model(
            "Proposed INT8 + Random Forest",
            "Your optimized quantized PCA projection with the project's 100-tree classifier.",
            proposed_float, random_forest(), X_train_scaled, X_test, y_train, y_test, scaler,
            projection_storage_bytes=proposed_components.nbytes,
        ),
        build_model(
            "Float PCA + Random Forest",
            "Full-precision PCA reference with the same 100-tree classifier.",
            float_components, random_forest(), X_train_scaled, X_test, y_train, y_test, scaler,
        ),
        build_model(
            "Naive INT8 PCA + Random Forest",
            "Directly rounded INT8 PCA projection with the same 100-tree classifier.",
            naive_components.astype(np.float32) / SCALE,
            random_forest(), X_train_scaled, X_test, y_train, y_test, scaler,
            projection_storage_bytes=naive_components.nbytes,
        ),
        build_model(
            "Float PCA + RBF SVM",
            "RBF support vector classifier on the shared 95%-variance PCA features.",
            float_components, SVC(kernel="rbf", C=1.0), X_train_scaled, X_test, y_train, y_test, scaler,
        ),
        build_model(
            "Float PCA + 5-NN",
            "Five-nearest-neighbor classifier on the shared 95%-variance PCA features.",
            float_components, KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            X_train_scaled, X_test, y_train, y_test, scaler,
        ),
        build_model(
            "Float PCA + Logistic Regression",
            "Multinomial logistic regression on the shared 95%-variance PCA features.",
            float_components,
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_scaled, X_test, y_train, y_test, scaler,
        ),
    ]

    proposed = models[0]
    comparisons = []
    for baseline in models[1:]:
        comparisons.append({
            "name": baseline["name"],
            "accuracy": {
                "proposed": proposed["accuracyPct"],
                "baseline": baseline["accuracyPct"],
                "proposedWins": proposed["accuracyPct"] >= baseline["accuracyPct"],
            },
            "latency": {
                "proposed": proposed["latencyMsPerSample"],
                "baseline": baseline["latencyMsPerSample"],
                "proposedWins": proposed["latencyMsPerSample"] <= baseline["latencyMsPerSample"],
            },
            "projection": {
                "proposed": proposed["projectionBytes"],
                "baseline": baseline["projectionBytes"],
                "proposedWins": proposed["projectionBytes"] <= baseline["projectionBytes"],
            },
        })

    report = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "trainingFile": TRAIN_FILE.name,
            "testingFile": TEST_FILE.name,
            "trainingSamples": int(len(y_train)),
            "testingSamples": int(len(y_test)),
            "features": int(X_train.shape[1]),
            "components": int(pca.n_components_),
            "explainedVariancePct": float(pca.explained_variance_ratio_.sum() * 100),
        },
        "methodology": {
            "timingRepeats": TIMING_REPEATS,
            "latency": "Mean per-sample StandardScaler + projection + classifier prediction time.",
            "projectionStorage": "PCA projection matrix bytes only; classifier and shared scaler storage excluded.",
            "comparison": "Check marks mean Proposed INT8 is equal or better for that metric; crosses mean the baseline is better.",
        },
        "models": models,
        "comparisons": comparisons,
    }

    RESULTS.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved five-baseline comparison to {OUTPUT_FILE}")
    for model in models:
        print(
            f"{model['name']}: accuracy={model['accuracyPct']:.2f}% | "
            f"latency={model['latencyMsPerSample']:.4f} ms/sample | "
            f"PCA matrix={model['projectionBytes'] / 1024:.2f} KiB"
        )


if __name__ == "__main__":
    main()