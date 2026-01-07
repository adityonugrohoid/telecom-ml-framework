## UC1 — Churn Prediction

### Objective
Predict the probability that a customer will churn within a future window.

### ML Type
- Supervised classification

### Input View
- Customer daily snapshots
- Rolling QoE aggregates (7d, 30d)
- Ticket history
- Usage trend features

### Forbidden
- Future QoE
- Future tickets
- Explicit churn triggers

### Label Definition
- `churn = 1` if customer churns within next *N* days
- Evaluated at end-of-day

### Outputs
- Churn probability
- Feature importance (SHAP)

---
## Notebook Specifications

### 1. `01_churn_prediction.ipynb` — UC1

| Property | Value |
| :--- | :--- |
| **Objective** | Predict customer churn probability |
| **ML Type** | Binary Classification |
| **Algorithm** | XGBoost / LightGBM |
| **Input** | `features_churn_*.parquet` |
| **Target** | `is_churned` |

**🎯 What This Does:**
| Audience | Explanation |
| :--- | :--- |
| **Business/Layman** | *"Which customers are likely to cancel their subscription next month? This model identifies at-risk customers so the retention team can offer them incentives before they leave."* |
| **Technical/ML** | *"Binary classification on tabular customer features (tenure, avg MOS, session count) using gradient boosting. Evaluated with AUROC, Precision-Recall, and SHAP for interpretability."* |

**Sections:**
1. Setup & Configuration (Seaborn styling)
2. Data Loading & Schema Validation
3. EDA: Churn rate, feature distributions
4. Train/Test Split (time-based, prevent leakage)
5. Model Training with cross-validation
6. Evaluation: AUROC, Precision-Recall, Confusion Matrix
7. Interpretation: SHAP feature importance
8. Business Insights

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
