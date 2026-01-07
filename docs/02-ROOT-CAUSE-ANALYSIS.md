## UC2 — ML-based RCA

### Objective
Identify root causes of incidents and rank causal candidates.

### ML Type
- Causal inference
- Graph ML
- Sequence analysis

### Input View
- Event timelines
- Alarm sequences
- KPI anomaly markers
- Network topology graph

### Forbidden
- Ground-truth root cause labels during inference
- Manual event annotations

### Label Definition
- Root cause known post-event
- Used only for training/evaluation

### Outputs
- Ranked root cause hypotheses
- Causal chains

---
## Notebook Specifications

### 2. `02_root_cause_analysis.ipynb` — UC2

| Property | Value |
| :--- | :--- |
| **Objective** | Rank causal candidates for network incidents |
| **ML Type** | Ranking / Causal Inference |
| **Algorithm** | Gradient Boosting or Graph Neural Network |
| **Input** | `features_rca.parquet` |
| **Target** | `is_root_cause` |

**🎯 What This Does:**
| Audience | Explanation |
| :--- | :--- |
| **Business/Layman** | *"When the network has problems, many alarms go off at once. This model finds the ORIGINAL cause (like a broken cable) instead of just the symptoms (slow internet for users)."* |
| **Technical/ML** | *"Supervised ranking model on event-alarm-ticket causal chains. Uses temporal lag features and severity encoding. Evaluated with Accuracy@K and Mean Reciprocal Rank."* |

**Sections:**
1. Setup & Configuration (Seaborn styling)
2. Data Loading & Causal Chain Visualization
3. EDA: Event severity distribution, alarm lag analysis
4. Feature Engineering: Sequence embeddings (optional)
5. Model Training: Ranking model or Classification
6. Evaluation: Accuracy@K, Mean Reciprocal Rank
7. Interpretation: Which features indicate root cause?
8. Causal Graph Visualization

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
