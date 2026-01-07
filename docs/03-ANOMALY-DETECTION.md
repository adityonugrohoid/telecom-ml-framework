## UC3 — Anomaly Detection

### Objective
Detect abnormal behavior without explicit labels.

### ML Type
- Unsupervised / semi-supervised

### Input View
- KPI time-series
- Multivariate metrics

### Forbidden
- Event flags
- Alarm labels
- Ticket data

### Label Definition
- None (evaluation only)

### Outputs
- Anomaly score
- Severity ranking

---
## Notebook Specifications

### 3. `03_anomaly_detection.ipynb` — UC3

| Property | Value |
| :--- | :--- |
| **Objective** | Detect abnormal cell behavior without labels |
| **ML Type** | Unsupervised / Semi-supervised |
| **Algorithm** | Isolation Forest / Autoencoder |
| **Input** | `features_anomaly.parquet` |
| **Target** | `label_anomaly` (for evaluation only) |

**🎯 What This Does:**
| Audience | Explanation |
| :--- | :--- |
| **Business/Layman** | *"Find cell towers that are behaving strangely BEFORE they fail completely. Like a health monitor that detects early warning signs."* |
| **Technical/ML** | *"Unsupervised anomaly detection on multivariate cell KPI time-series. Isolation Forest for baseline, LSTM Autoencoder for sequence modeling. Evaluated against ground-truth fault events."* |

**Sections:**
1. Setup & Configuration (Seaborn styling)
2. Data Loading & Time-series reshaping
3. EDA: Load patterns, throughput distributions
4. Unsupervised Training: Isolation Forest
5. Alternative: LSTM Autoencoder
6. Anomaly Scoring
7. Evaluation: Precision-Recall vs ground truth
8. Threshold Tuning & Alerting Strategy

**📊 Plotting Specifications:**

All visualizations should use a unified Seaborn configuration defined in **Cell 1** (after imports):

```python
# === UNIFIED PLOT STYLING (Cell 1) ===
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn context, style, and palette
sns.set_context("notebook")  # Switch to "talk" for presentations
sns.set_style("whitegrid")
sns.set_palette("husl")

# Optional: Set default figure size
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
```

**Rules for Subsequent Cells:**
- ✅ **DO**: Use `sns.plot_type()` functions directly (e.g., `sns.barplot()`, `sns.histplot()`)
- ✅ **DO**: Rely on the top-cell styling for all aesthetics
- ❌ **DON'T**: Modify font sizes, weights, padding, or colors in individual cells
- ❌ **DON'T**: Use `plt.rcParams` or `sns.set()` anywhere except Cell 1

**Context Switching:**
To switch from notebook to presentation mode, change **only Cell 1**:
```python
sns.set_context("talk")  # Larger fonts for presentations
```

**🔍 Model Interpretability (SHAP):**

Import SHAP for model interpretability:

```python
import shap
```

**⚠️ Version Compatibility Requirements:**

SHAP has specific version dependencies that must be pinned in `pyproject.toml`:

```toml
dependencies = [
    "numpy<2.0",      # SHAP color conversion bug with NumPy 2.x
    "xgboost<2.0",    # SHAP not compatible with XGBoost 2.x base_score format
    "numba>=0.59.0",  # Python 3.12 support
    "shap>=0.42.0",
]
```

**Why These Pins?**
- **numpy<2.0**: SHAP has color conversion issues with NumPy 2.x
- **xgboost<2.0**: SHAP doesn't support XGBoost 2.x's new base_score format
- **numba>=0.59.0**: Required for Python 3.12 compatibility

---
