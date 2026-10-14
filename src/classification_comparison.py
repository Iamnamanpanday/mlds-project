import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "compressed"
RESULTS = BASE / "results"
OUTPUT_FILE = BASE / "classification" / "comparison.json"
TRAIN_FILE = DATA / "train_1600.xlsx"
TEST_FILE = DATA / "test_1200.xlsx"
PROPOSED_FILE = RESULTS / "proposed_integer_pca_matrix.npy"
SCALE = 128
TIMING_REPEATS = 5


def load_data(path):
    frame = pd.read_excel(path)
    if "Activity" not in frame.columns:
        raise ValueError(f"Expected an Activity target column in {path}")
    features = frame.drop(columns=["Activity"]).to_numpy(dtype=np.float32)
    return features, frame["Activity"].to_numpy()


def evaluate_model(name, note, classifier, components, component_bytes, train_scaled, test_raw, y_train, y_test, scaler):
    train_projection = train_scaled @ components.T
    classifier.fit(train_projection, y_train)

    def predict():
        test_scaled = scaler.transform(test_raw)
        return classifier.predict(test_scaled @ components.T)

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
        "weightedPrecisionPct": float(precision_score(y_test, predictions, average="weighted", zero_division=0) * 100),
        "weightedRecallPct": float(recall_score(y_test, predictions, average="weighted", zero_division=0) * 100),
        "weightedF1Pct": float(f1_score(y_test, predictions, average="weighted", zero_division=0) * 100),
        "latencyMsPerSample": float(np.mean(durations) * 1000 / len(y_test)),
        "projectionBytes": int(component_bytes),
    }


def comparison_metric(ours, baseline, higher_is_better=True):
    our_value = ours
    baseline_value = baseline
    return {
        "ourMethod": our_value,
        "baseline": baseline_value,
        "ourMethodWins": our_value >= baseline_value if higher_is_better else our_value <= baseline_value,
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
    naive_int8 = np.round(float_components * SCALE).astype(np.int8)
    proposed_int8 = np.load(PROPOSED_FILE).astype(np.int8)
    if proposed_int8.shape != float_components.shape:
        raise ValueError(
            f"Proposed matrix shape {proposed_int8.shape} does not match PCA shape {float_components.shape}"
        )

    proposed_components = proposed_int8.astype(np.float32) / SCALE
    models = [
        evaluate_model(
            "Proposed INT8 PCA + Random Forest (Our method)",
            "Optimized INT8 PCA projection with the project's 100-tree Random Forest.",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            proposed_components, proposed_int8.nbytes, X_train_scaled, X_test, y_train, y_test, scaler,
        ),
        evaluate_model(
            "Float PCA + Random Forest",
            "Full-precision PCA projection with the same 100-tree Random Forest.",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            float_components, float_components.nbytes, X_train_scaled, X_test, y_train, y_test, scaler,
        ),
        evaluate_model(
            "Naive INT8 PCA + Random Forest",
            "Directly rounded INT8 PCA projection with the same 100-tree Random Forest.",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            naive_int8.astype(np.float32) / SCALE, naive_int8.nbytes,
            X_train_scaled, X_test, y_train, y_test, scaler,
        ),
        evaluate_model(
            "Float PCA + RBF SVM",
            "RBF support vector classifier on shared 95%-variance PCA features.",
            SVC(kernel="rbf", C=1.0), float_components, float_components.nbytes,
            X_train_scaled, X_test, y_train, y_test, scaler,
        ),
        evaluate_model(
            "Float PCA + 5-NN",
            "Five-nearest-neighbor classifier on shared 95%-variance PCA features.",
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1), float_components, float_components.nbytes,
            X_train_scaled, X_test, y_train, y_test, scaler,
        ),
        evaluate_model(
            "Float PCA + Logistic Regression",
            "Logistic regression on shared 95%-variance PCA features.",
            LogisticRegression(max_iter=1000, random_state=42), float_components, float_components.nbytes,
            X_train_scaled, X_test, y_train, y_test, scaler,
        ),
    ]

    our_method = models[0]
    comparisons = []
    for baseline in models[1:]:
        comparisons.append({
            "name": baseline["name"],
            "accuracy": comparison_metric(our_method["accuracyPct"], baseline["accuracyPct"]),
            "weightedPrecision": comparison_metric(
                our_method["weightedPrecisionPct"], baseline["weightedPrecisionPct"],
            ),
            "weightedRecall": comparison_metric(
                our_method["weightedRecallPct"], baseline["weightedRecallPct"],
            ),
            "weightedF1": comparison_metric(our_method["weightedF1Pct"], baseline["weightedF1Pct"]),
            "latency": comparison_metric(
                our_method["latencyMsPerSample"], baseline["latencyMsPerSample"], higher_is_better=False,
            ),
            "projection": comparison_metric(
                our_method["projectionBytes"], baseline["projectionBytes"], higher_is_better=False,
            ),
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
            "classes": sorted(str(label) for label in np.unique(y_test)),
        },
        "methodology": {
            "timingRepeats": TIMING_REPEATS,
            "latency": "Mean per-sample StandardScaler + PCA projection + classifier prediction time.",
            "projectionStorage": "PCA projection matrix bytes only; classifier and shared scaler storage excluded.",
            "comparison": "Every baseline is compared with Proposed INT8 PCA + Random Forest (Our method).",
        },
        "distinctiveFeatures": [
            "Two local-search passes refine the first 20 PCA components after INT8 quantization.",
            "Each selected coefficient tests adjacent integer values (-1 and +1), keeping only objective-improving changes.",
            "The refinement objective combines relative projection approximation error with a 0.05-weighted orthogonality penalty.",
        ],
        "models": models,
        "comparisons": comparisons,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved classification comparison to {OUTPUT_FILE}")
    for model in models:
        print(
            f"{model['name']}: accuracy={model['accuracyPct']:.2f}% | "
            f"weighted F1={model['weightedF1Pct']:.2f}% | "
            f"latency={model['latencyMsPerSample']:.4f} ms/sample | "
            f"PCA matrix={model['projectionBytes'] / 1024:.2f} KiB"
        )


if __name__ == "__main__":
    main()
