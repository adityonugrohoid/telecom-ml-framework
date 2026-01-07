## UC5 — Capacity Forecasting

### Objective
Forecast future traffic demand and capacity exhaustion.

### ML Type
- Time-series forecasting

### Input View
- Historical traffic KPIs
- Subscriber growth trends
- Regional aggregates

### Forbidden
- Future events
- Optimization actions

### Label Definition
- Future traffic values

### Outputs
- Forecast with confidence intervals
- Capacity exhaustion alerts

---
## Notebook Specifications

### 5. `05_capacity_forecasting.ipynb` — UC5

| Property | Value |
| :--- | :--- |
| **Objective** | Forecast future traffic demand |
| **ML Type** | Time-series Forecasting |
| **Algorithm** | Prophet / ARIMA / LSTM |
| **Input** | `features_capacity_h24.parquet` |
| **Target** | `load_target` (future load) |

**🎯 What This Does:**
| Audience | Explanation |
| :--- | :--- |
| **Business/Layman** | *"Predict how busy each cell tower will be tomorrow or next week. This helps the network team add capacity BEFORE congestion happens, not after customers complain."* |
| **Technical/ML** | *"Time-series forecasting with lag features (1h, 24h, 7d) and rolling statistics. Compares Prophet (additive seasonality) vs. LightGBM (tabular regression). Evaluated with MAPE and visual forecast plots."* |

**Sections:**
1. Setup & Configuration (Seaborn styling)
2. Data Loading & Time-series decomposition
3. EDA: Load patterns, seasonality, trend
4. Feature Engineering: Lag features, rolling aggregates
5. Model Training: Prophet / ARIMA / LSTM
6. Evaluation: MAPE, RMSE, forecast vs actual plots
7. Scenario Analysis: Weekday vs weekend forecasts
8. Confidence Intervals & Capacity Recommendations

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
- ✅ **DO**: Use `sns.plot_type()` functions directly (e.g., `sns.lineplot()`, `sns.histplot()`)
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
