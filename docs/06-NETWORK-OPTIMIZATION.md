## UC6 — Network Optimization

### Objective
Recommend actions that improve network performance.

### ML Type
- Reinforcement learning
- Optimization algorithms

### Input View
- Current network state snapshot
- Interference matrix
- Load distribution

### Forbidden
- Ground-truth future KPIs
- Churn labels

### Reward Signal
- KPI improvement at *t+1 → t+k*

### Outputs
- Recommended actions
- Expected KPI deltas

---
## Notebook Specifications

### 6. `06_network_optimization.ipynb` — UC6

| Property | Value |
| :--- | :--- |
| **Objective** | Recommend network optimization actions |
| **ML Type** | Reinforcement Learning / Optimization |
| **Algorithm** | Q-Learning / Genetic Algorithm / Bayesian Optimization |
| **Input** | `features_network_state.parquet` |
| **Target** | KPI improvement (SINR, throughput, latency) |

**🎯 What This Does:**
| Audience | Explanation |
| :--- | :--- |
| **Business/Layman** | *"Automatically suggest the best actions to improve network quality—like adjusting power levels or rebalancing traffic between towers. Instead of engineers manually tuning parameters, this AI recommends optimal changes."* |
| **Technical/ML** | *"Reinforcement learning agent that observes network state (load, interference, SINR) and recommends parameter adjustments. Actions are evaluated by KPI delta at t+1. Uses Q-learning with state discretization or policy gradient methods."* |

**Sections:**
1. Setup & Configuration (Seaborn styling)
2. Data Loading & State Representation
3. EDA: Network state distributions, action space
4. Environment Setup: State, action, reward definitions
5. Baseline: Random/heuristic policy
6. Model: Q-Learning / DQN agent training
7. Evaluation: Reward convergence, KPI improvement
8. Action Recommendations & Expected Impact

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
- ✅ **DO**: Use `sns.plot_type()` functions directly (e.g., `sns.lineplot()`, `sns.barplot()`)
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
