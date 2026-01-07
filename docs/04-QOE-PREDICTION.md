## UC4 — QoE Prediction

### Objective
Predict perceived user experience.

### ML Type
- Regression
- Multi-task learning

### Input View
- Network KPIs
- Session context
- Device capabilities
- Application type

### Forbidden
- Derived QoE values
- Churn outcomes

### Label Definition
- MOS score computed from physics
- QoE tier derived post-hoc

### Outputs
- Predicted MOS
- QoE class

---
## Notebook Specifications

### 4. `04_qoe_prediction.ipynb` — UC4

| Property | Value |
| :--- | :--- |
| **Objective** | Predict user-perceived QoE (MOS score) |
| **ML Type** | Regression |
| **Algorithm** | LightGBM / CatBoost |
| **Input** | `features_qoe.parquet` |
| **Target** | `mos_score` |

**🎯 What This Does:**
| Audience | Explanation |
| :--- | :--- |
| **Business/Layman** | *"Predict how happy a customer will be with their video call or streaming experience, based on network conditions. A score of 4+ means 'Excellent', below 2 means 'Frustrating'."* |
| **Technical/ML** | *"Regression on session-level features (throughput, latency, packet loss, device capability). Predicts continuous MOS (1.0-4.5). Evaluated with MAE, RMSE, R². Feature importance via permutation."* |

**Sections:**
1. Setup & Configuration (Seaborn styling)
2. Data Loading & QoE score distribution
3. EDA: Throughput vs QoE, latency vs QoE
4. Feature Engineering: App-specific features
5. Model Training: Regression (LightGBM or Neural Network)
6. Evaluation: RMSE, MAE, R²
7. Interpretation: Feature importance by app type
8. QoE Prediction Curves

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
