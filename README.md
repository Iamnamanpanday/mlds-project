# Quantized PCA Benchmark Lab

An interactive evaluation and benchmarking dashboard comparing Float PCA, Naive INT8 PCA, and Proposed INT8 PCA on high-dimensional human activity recognition data.

---

## 🚀 Quick Start (Booting the App)

### Option 1: Using the launcher script
```bash
python run_app.py
```
This automatically starts the local HTTP server on port 8000 and opens your default web browser to the dashboard.

### Option 2: Using Python's built-in HTTP server
```bash
# From the project root:
python -m http.server 8000
```
Then navigate to:
**[http://localhost:8000/](http://localhost:8000/)** (auto-redirects to `/app/`) or **[http://localhost:8000/app/](http://localhost:8000/app/)**

---

## 📊 Benchmark Highlights

| Method | Test Accuracy | Latency (ms) | Storage / ROM | Approximation Error | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Float PCA** | 82.83% | ~1.07 ms | 204.16 KiB | 0.00% | Reference Baseline |
| **Naive INT8** | 82.00% | ~1.10 ms | 51.04 KiB | 5.35% | Compact |
| **Proposed INT8** | **82.83%** | **~0.96 ms** | **51.04 KiB (−75%)** | 134.50% | **Recommended** |

---

## 📁 Project Structure

```
├── app/                  # Web dashboard UI
│   ├── index.html        # Main dashboard interface
│   ├── script.js         # Interactive model switcher & telemetry logic
│   ├── style.css         # Custom typography, layout, and dark-accents styling
│   ├── favicon.svg       # Brand favicon
│   └── results/          # Confusion matrix assets for the UI
├── data/
│   └── compressed/       # Downsampled & preprocessed Excel datasets
├── results/              # Saved model matrices (.npy) and confusion matrices (.png)
├── src/                  # Python modeling, training, and benchmarking scripts
│   ├── baseline_model.py
│   ├── naive_integer_pca.py
│   ├── proposed_integer_pca.py
│   └── benchmark_all_methods.py
├── run_app.py            # One-click launcher for the web app
└── index.html            # Root redirect entry point
```

---

## 🔬 Running the Benchmarks

To re-run the end-to-end Python pipeline (StandardScaler → PCA → Random Forest Classifier):

```bash
python src/benchmark_all_methods.py
```

To generate the separate feature report comparing Proposed INT8 with five established baselines and refresh the dashboard data:

```bash
python src/feature_comparison_report.py
```

The dashboard uses checkmarks when Proposed INT8 ties or wins a metric and crosses when a baseline scores better. Accuracy, end-to-end inference latency, and PCA projection matrix storage are reported independently; the shared classifier and scaler are excluded from the matrix-size comparison.
