# Getting Started with Telecom ML Framework

This guide will walk you through creating your first telecom ML project using this framework.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Creating Your First Project](#creating-your-first-project)
- [Understanding the Template](#understanding-the-template)
- [Customizing for Your Use Case](#customizing-for-your-use-case)
- [Example Walkthrough: Churn Prediction](#example-walkthrough-churn-prediction)
- [Common Customization Patterns](#common-customization-patterns)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## Prerequisites

Before starting, ensure you have:

### Required Software

1. **Python 3.11 or higher**
   ```bash
   python --version  # Should show 3.11.x or higher
   ```
   If not installed: [Download Python](https://www.python.org/downloads/)

2. **uv package manager**
   ```bash
   # Check if installed
   uv --version
   
   # If not installed (Linux/macOS):
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # If not installed (Windows):
   # Download from https://github.com/astral-sh/uv/releases
   ```

### Recommended Tools

- **Git** for version control
- **Jupyter Lab** (installed automatically via template)
- **VS Code** or **Cursor IDE** with Python extension

---

## Installation

### Step 1: Clone or Download the Framework

```bash
# Option A: Clone from GitHub
git clone https://github.com/adityonugrohoid/telecom-ml-framework.git
cd telecom-ml-framework

# Option B: Download ZIP and extract
# Then navigate to the extracted folder
```

### Step 2: Verify Framework Structure

```bash
ls -la
# You should see:
# - template/           (project template)
# - docs/              (documentation)
# - examples/          (usage examples)
# - README.md
# - LICENSE
```

You're ready to create your first project!

---

## Creating Your First Project

### Method 1: Manual Copy (Recommended for First-Time Users)

This method gives you full visibility into what's being created.

```bash
# 1. Navigate to where you want your projects
cd ..  # Go up one level from framework directory

# 2. Create a project directory from template
cp -r telecom-ml-framework/template/ churn-prediction
cd churn-prediction

# 3. You now have a complete project structure!
ls -la
```

### Method 2: Automated Script

For faster project creation with automatic variable substitution:

```bash
# From the framework root directory
python examples/create_project.py \
  --name churn-prediction \
  --use-case UC1 \
  --output ../my-projects/

cd ../my-projects/churn-prediction
```

**What the script does:**
- Copies template directory
- Renames `__project_name__` to your project name
- Substitutes placeholders in `pyproject.toml`, `README.md`, etc.
- Creates properly named Python package structure

---

## Understanding the Template

After creating your project, you'll have this structure:

```
churn-prediction/
├── README.md                    # Project documentation (contains placeholders)
├── QUICKSTART.md                # Quick setup commands
├── CONTRIBUTING.md              # For collaborative projects
├── pyproject.toml               # Dependencies managed by uv
├── .gitignore                   # Excludes data and Python artifacts
│
├── data/
│   ├── raw/                     # Generated synthetic data goes here
│   └── processed/               # Feature-engineered datasets
│
├── src/churn_prediction/        # Main Python package
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── data_generator.py        # Synthetic data generation
│   ├── features.py              # Feature engineering
│   └── models.py                # Model training/evaluation
│
├── notebooks/
│   └── (empty - you'll add analysis notebooks here)
│
└── tests/
    └── test_data_quality.py     # Data validation tests
```

### Key Files Explained

#### `pyproject.toml`
- Defines project metadata and dependencies
- Uses `uv` for fast, deterministic dependency resolution
- Pre-configured with SHAP-compatible versions:
  - `numpy<2.0` (SHAP color conversion compatibility)
  - `xgboost<2.0` (SHAP base_score format compatibility)
  - `numba>=0.59.0` (Python 3.12 support)

#### `src/your_project/data_generator.py`
- Base class `TelecomDataGenerator` with helper methods
- Domain physics functions: `generate_sinr()`, `sinr_to_throughput()`, etc.
- Example generator to customize for your use case
- Emphasizes **domain-informed** data generation over off-the-shelf tools

#### `src/your_project/config.py`
- Centralized configuration for all project settings
- Data paths, model hyperparameters, visualization settings
- Easy to modify without touching core logic

#### `src/your_project/features.py`
- Placeholder for feature engineering pipeline
- Should implement temporal aggregations, derived features
- Must prevent data leakage (no future data)

#### `src/your_project/models.py`
- Placeholder for model training and evaluation
- Should include cross-validation, hyperparameter tuning
- SHAP interpretability integration

---

## Customizing for Your Use Case

### Step 1: Choose Your Use Case

Review the [Use Cases documentation](USE_CASES.md) and select one:

- **UC1**: Churn Prediction (Classification)
- **UC2**: Root Cause Analysis (Ranking/Classification)
- **UC3**: Anomaly Detection (Unsupervised)
- **UC4**: QoE Prediction (Regression)
- **UC5**: Capacity Forecasting (Time-Series)
- **UC6**: Network Optimization (Reinforcement Learning)

Each has a detailed specification document (e.g., [01-CHURN-PREDICTION.md](01-CHURN-PREDICTION.md)).

### Step 2: Update Project Metadata

Edit `pyproject.toml`:

```toml
[project]
name = "churn-prediction"  # Your project name (lowercase, hyphens)
version = "0.1.0"
description = "Predict customer churn using QoE degradation patterns"
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]
```

### Step 3: Customize Data Generator

Edit `src/your_project/data_generator.py`:

1. **Subclass `TelecomDataGenerator`**:
   ```python
   class ChurnPredictionGenerator(TelecomDataGenerator):
       def generate(self) -> pd.DataFrame:
           # Your custom generation logic
           pass
   ```

2. **Use domain physics helpers**:
   - `generate_sinr()` - Signal quality
   - `sinr_to_throughput()` - Network speed
   - `generate_congestion_pattern()` - Time-of-day patterns
   - `compute_qoe_mos()` - User experience score

3. **Add use case-specific logic**:
   - For churn: Generate customer history, tenure, usage trends
   - For RCA: Generate alarm sequences, event timelines
   - For anomaly: Generate normal/abnormal cell behavior

### Step 4: Rename the Package

```bash
# Rename the source directory
mv src/__project_name__ src/churn_prediction

# Update imports in all Python files
# Change: from __project_name__ import X
# To:     from churn_prediction import X
```

### Step 5: Install Dependencies

```bash
# From project root
uv sync

# Verify installation
uv run python -c "import pandas; import xgboost; import shap; print('✓ All imports successful')"
```

---

## Example Walkthrough: Churn Prediction

Let's create a complete churn prediction project step-by-step.

### 1. Create Project

```bash
cd ..  # Navigate out of framework directory
cp -r telecom-ml-framework/template/ churn-prediction
cd churn-prediction
```

### 2. Rename Package

```bash
mv src/__project_name__ src/churn_prediction
```

### 3. Customize Data Generator

Edit `src/churn_prediction/data_generator.py`:

```python
class ChurnPredictionGenerator(TelecomDataGenerator):
    """Generate synthetic customer data with churn labels."""
    
    def generate(self) -> pd.DataFrame:
        n = self.n_samples
        
        # Customer attributes
        customer_ids = np.arange(1, n + 1)
        tenure_days = self.rng.integers(1, 730, n)
        
        # Generate timestamps (customer daily snapshots)
        base_date = pd.Timestamp("2024-01-01")
        timestamps = [base_date + pd.Timedelta(days=t) for t in range(30)]
        
        # Network experience features
        sinr = self.generate_sinr(n)
        throughput = self.sinr_to_throughput(sinr, 
            self.rng.choice(["4G", "5G"], n))
        
        # QoE degradation (churn signal)
        avg_qoe = self.rng.normal(3.5, 0.5, n)
        qoe_trend = self.rng.normal(0, 0.3, n)  # Negative = degrading
        
        # Churn label (occurs if QoE degrading + low tenure)
        churn_probability = 0.05 + 0.3 * (qoe_trend < -0.2) + 0.2 * (tenure_days < 90)
        is_churned = self.rng.random(n) < churn_probability
        
        return pd.DataFrame({
            "customer_id": customer_ids,
            "tenure_days": tenure_days,
            "avg_sinr_db": sinr,
            "avg_throughput_mbps": throughput,
            "avg_qoe_mos": avg_qoe,
            "qoe_trend_30d": qoe_trend,
            "is_churned": is_churned.astype(int),
        })

# Update main() to use new generator
def main():
    generator = ChurnPredictionGenerator(seed=42, n_samples=10_000)
    df = generator.generate()
    generator.save(df, "churn_features")
```

### 4. Generate Data

```bash
uv run python -m churn_prediction.data_generator

# Output:
# Generating synthetic telecom data...
# ✓ Saved 10,000 rows to data/raw/churn_features.parquet
```

### 5. Create Analysis Notebook

Create `notebooks/01_churn_analysis.ipynb` with these cells:

**Cell 1: Imports and Styling**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path

# Unified styling
sns.set_context("notebook")
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
```

**Cell 2: Load Data**
```python
data_path = Path("../data/raw/churn_features.parquet")
df = pd.read_parquet(data_path)
print(f"Loaded {len(df):,} rows")
df.head()
```

**Cell 3: EDA - Churn Rate**
```python
churn_rate = df['is_churned'].mean()
print(f"Churn rate: {churn_rate:.1%}")

sns.countplot(data=df, x='is_churned')
plt.title("Churn Distribution")
plt.show()
```

**Cell 4: Model Training**
```python
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Features and target
feature_cols = ['tenure_days', 'avg_sinr_db', 'avg_throughput_mbps', 
                'avg_qoe_mos', 'qoe_trend_30d']
X = df[feature_cols]
y = df['is_churned']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(classification_report(y_test, y_pred))
```

**Cell 5: SHAP Interpretability**
```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.title("Feature Importance (SHAP)")
plt.tight_layout()
plt.show()
```

### 6. Run the Notebook

```bash
uv run jupyter lab

# Open notebooks/01_churn_analysis.ipynb and run all cells
```

### 7. Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial commit: Churn prediction project from telecom-ml-framework"
```

🎉 **Congratulations!** You've created your first telecom ML project.

---

## Common Customization Patterns

### Adding More Features

Edit `src/your_project/features.py`:

```python
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling window aggregations."""
    df = df.sort_values(['customer_id', 'timestamp'])
    
    # 7-day rolling average QoE
    df['qoe_7d_avg'] = df.groupby('customer_id')['qoe_mos'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    
    # 30-day QoE trend
    df['qoe_30d_trend'] = df.groupby('customer_id')['qoe_mos'].transform(
        lambda x: x.rolling(30, min_periods=7).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0]
        )
    )
    
    return df
```

### Changing Model Algorithm

Update `src/your_project/models.py` to use LightGBM instead of XGBoost:

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
)
```

### Adjusting Configuration

Edit `src/your_project/config.py`:

```python
DATA_GEN_CONFIG = {
    "random_seed": 42,
    "n_samples": 50_000,  # Increase sample size
    "churn_rate": 0.15,   # Target churn rate
}

MODEL_CONFIG = {
    "algorithm": "xgboost",
    "cv_folds": 5,
    "hyperparameters": {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
    },
}
```

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Make sure you're using `uv run`:
```bash
# Wrong:
python -m your_project.data_generator

# Correct:
uv run python -m your_project.data_generator
```

### Issue: SHAP compatibility errors

**Error**: `AttributeError: module 'numpy' has no attribute 'float_'`

**Solution**: Verify version constraints in `pyproject.toml`:
```toml
dependencies = [
    "numpy>=1.24.0,<2.0",      # Must be < 2.0
    "xgboost>=1.7.0,<2.0",     # Must be < 2.0
]
```

Then reinstall:
```bash
uv sync --reinstall
```

### Issue: Data file not found

**Solution**: Generate data first:
```bash
uv run python -m your_project.data_generator
ls data/raw/  # Verify file exists
```

### Issue: Jupyter kernel not found

**Solution**: Install ipykernel in the uv environment:
```bash
uv sync  # Should install jupyter automatically
uv run jupyter lab  # Use uv run prefix
```

---

## Next Steps

### 1. Read Use Case Specifications
- Review [USE_CASES.md](USE_CASES.md) for detailed specifications
- Each use case document explains:
  - Business problem
  - ML framing
  - Feature engineering approach
  - Evaluation metrics
  - Interpretation requirements

### 2. Explore Other Use Cases
- Try creating projects for different use cases
- Each teaches different ML techniques:
  - Time-series forecasting (UC5)
  - Unsupervised learning (UC3)
  - Reinforcement learning (UC6)

### 3. Build Your Portfolio
- Push projects to GitHub
- Write detailed README for each project
- Add visualizations and insights
- Reference the [PORTFOLIO_OVERVIEW.md](PORTFOLIO_OVERVIEW.md) for context

### 4. Customize Further
- Add real data connectors (if you have access)
- Implement advanced feature engineering
- Add model serving capabilities
- Create dashboards with Streamlit or Dash

### 5. Share and Learn
- Open source your implementations
- Write blog posts about your learnings
- Contribute improvements back to this framework

---

## Additional Resources

### Framework Documentation
- [Main README](../README.md) - Framework overview
- [Portfolio Overview](PORTFOLIO_OVERVIEW.md) - Career context
- [Use Case Specifications](USE_CASES.md) - Detailed specs
- [GitHub Repository](https://github.com/adityonugrohoid/telecom-ml-framework) - Source code

### Telecom Domain Resources
- 3GPP Standards: [https://www.3gpp.org](https://www.3gpp.org)
- ITU-T QoE: [https://www.itu.int](https://www.itu.int)
- Network optimization guides

### ML Techniques
- SHAP Documentation: [https://shap.readthedocs.io](https://shap.readthedocs.io)
- XGBoost Guide: [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- Scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)

---

**Questions or Issues?**

- Check existing [GitHub issues](https://github.com/adityonugrohoid/telecom-ml-framework/issues)
- Open a new issue with detailed description
- Email: adityo.nugroho.id@gmail.com

---

*Happy building! 🚀*

