# Telecom ML Framework

> A production-ready framework for building AI/ML solutions to real-world telecom challenges, emphasizing domain expertise and practical problem-solving.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework Status](https://img.shields.io/badge/status-stable-green.svg)](https://github.com)

## 📋 Table of Contents

- [What Is This?](#what-is-this)
- [Who Should Use This?](#who-should-use-this)
- [What's Included](#whats-included)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Philosophy](#philosophy)
- [License](#license)

## What Is This?

**This is a FRAMEWORK, not an implementation.** 

The Telecom ML Framework provides:

✅ **6 Production-Ready ML Project Templates** covering the most common telecom AI/ML use cases  
✅ **Complete Technical Specifications** with problem framing, data requirements, and model architectures  
✅ **Domain-Informed Data Generators** embedding real telecom physics (SINR, QoE, congestion patterns)  
✅ **Unified Technical Standards** ensuring consistency across projects (dependencies, plotting, interpretability)  
✅ **Portfolio Documentation** demonstrating domain expertise and ML problem-solving approach

**What this is NOT:**
- ❌ Not a trained model or production system
- ❌ Not a Python package to install via pip
- ❌ Not a data science library with APIs

This framework serves as both a **project template generator** for rapid ML project creation and a **portfolio documentation hub** showcasing telecom domain expertise applied to ML.

## Who Should Use This?

This framework is designed for:

### 🎯 Primary Audience
- **Telecom professionals** transitioning to AI/ML who need structured project templates
- **Data scientists** entering telecom domain who need problem framing guidance
- **ML engineers** building telecom analytics solutions
- **Portfolio builders** demonstrating end-to-end ML thinking

### 💡 What You'll Learn
- How to frame business problems as ML tasks
- Domain-driven feature engineering for telecom data
- Proper handling of temporal leakage in time-series problems
- Model interpretability for business stakeholders
- Production-ready project structure and standards

## What's Included

### 🗂️ Framework Components

```
telecom-ml-framework/
├── template/                    # Project template (copy this to start)
│   ├── src/__project_name__/   # Python package structure
│   ├── notebooks/              # Jupyter notebook templates
│   ├── data/                   # Data directories
│   ├── tests/                  # Test templates
│   └── pyproject.toml          # Dependencies with SHAP compatibility
│
├── docs/                        # Documentation
│   ├── USE_CASES.md            # Index of all 6 use cases
│   ├── GETTING_STARTED.md      # Detailed usage guide
│   ├── PORTFOLIO_OVERVIEW.md   # Portfolio context
│   └── 01-06 use case specs    # Individual specifications
│
└── examples/                    # Usage examples
    └── create_project.py       # Template instantiation script
```

### 📚 6 Documented Use Cases

| # | Use Case | ML Type | Key Algorithms | Status |
|:---:|:---|:---|:---|:---:|
| **UC1** | [Churn Prediction](docs/01-CHURN-PREDICTION.md) | Binary Classification | XGBoost, LightGBM | ✅ Spec Complete |
| **UC2** | [Root Cause Analysis](docs/02-ROOT-CAUSE-ANALYSIS.md) | Ranking / Causal Inference | Gradient Boosting, GNN | ✅ Spec Complete |
| **UC3** | [Anomaly Detection](docs/03-ANOMALY-DETECTION.md) | Unsupervised Learning | Isolation Forest, LSTM AE | ✅ Spec Complete |
| **UC4** | [QoE Prediction](docs/04-QOE-PREDICTION.md) | Regression | LightGBM, CatBoost | ✅ Spec Complete |
| **UC5** | [Capacity Forecasting](docs/05-CAPACITY-FORECASTING.md) | Time-Series Forecasting | Prophet, ARIMA, LSTM | ✅ Spec Complete |
| **UC6** | [Network Optimization](docs/06-NETWORK-OPTIMIZATION.md) | Reinforcement Learning | Q-Learning, Genetic Algo | ✅ Spec Complete |

[📖 View detailed use case documentation →](docs/USE_CASES.md)

## Quick Start

### Prerequisites

- **Python 3.11+** ([download](https://www.python.org/downloads/))
- **uv** package manager ([install](https://github.com/astral-sh/uv))

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Method 1: Manual Template Copy (Recommended for Learning)

```bash
# 1. Copy the template directory
cp -r template/ ../my-churn-prediction
cd ../my-churn-prediction

# 2. Customize the project
# - Rename src/__project_name__/ to your project name
# - Update pyproject.toml with your details
# - Customize data_generator.py for your use case

# 3. Install dependencies
uv sync

# 4. Generate synthetic data
uv run python -m your_project_name.data_generator

# 5. Start working!
uv run jupyter lab notebooks/
```

### Method 2: Using the Example Script (Automated)

```bash
# Create a new project from template
python examples/create_project.py \
  --name churn-prediction \
  --use-case UC1 \
  --output ../my-projects/

cd ../my-projects/churn-prediction
uv sync
uv run python -m churn_prediction.data_generator
```

### Next Steps

1. **Read the documentation**: Start with [GETTING_STARTED.md](docs/GETTING_STARTED.md)
2. **Choose a use case**: Review [USE_CASES.md](docs/USE_CASES.md) to select your focus
3. **Customize the template**: Adapt data generation and features to your needs
4. **Build your portfolio**: Each project becomes a standalone repository

## Use Cases

### UC1: Churn Prediction
**Business Problem**: Which customers are likely to cancel their subscription?  
**ML Approach**: Binary classification with temporal feature engineering  
**Key Challenge**: Preventing future data leakage, handling class imbalance  
**Output**: Churn probability + SHAP interpretability for retention campaigns

[📄 Full Specification →](docs/01-CHURN-PREDICTION.md)

---

### UC2: Root Cause Analysis
**Business Problem**: When network issues occur, what was the original cause?  
**ML Approach**: Ranking/classification on event-alarm-ticket causal chains  
**Key Challenge**: Multi-label problem with correlated failure modes  
**Output**: Ranked root cause hypotheses with causal graphs

[📄 Full Specification →](docs/02-ROOT-CAUSE-ANALYSIS.md)

---

### UC3: Anomaly Detection
**Business Problem**: Detect cell towers behaving abnormally before they fail  
**ML Approach**: Unsupervised learning on multivariate KPI time-series  
**Key Challenge**: Defining "normal" in highly dynamic networks  
**Output**: Anomaly scores and severity ranking

[📄 Full Specification →](docs/03-ANOMALY-DETECTION.md)

---

### UC4: QoE Prediction
**Business Problem**: Predict user-perceived quality from network conditions  
**ML Approach**: Regression on session-level features (throughput, latency, loss)  
**Key Challenge**: QoE is subjective and application-dependent  
**Output**: Predicted MOS score and QoE class

[📄 Full Specification →](docs/04-QOE-PREDICTION.md)

---

### UC5: Capacity Forecasting
**Business Problem**: Predict future network load to plan capacity expansions  
**ML Approach**: Time-series forecasting with seasonal decomposition  
**Key Challenge**: Capturing diurnal patterns, weekend effects, growth trends  
**Output**: Load forecasts with confidence intervals

[📄 Full Specification →](docs/05-CAPACITY-FORECASTING.md)

---

### UC6: Network Optimization
**Business Problem**: Recommend parameter adjustments to improve KPIs  
**ML Approach**: Reinforcement learning with state-action-reward formulation  
**Key Challenge**: Delayed rewards, exploration vs exploitation  
**Output**: Recommended actions and expected KPI improvements

[📄 Full Specification →](docs/06-NETWORK-OPTIMIZATION.md)

## Project Structure

Each project created from this framework follows this structure:

```
your-project-name/
├── README.md                    # Project-specific documentation
├── QUICKSTART.md                # Quick setup guide
├── CONTRIBUTING.md              # Contribution guidelines
├── pyproject.toml               # Dependencies (uv-managed)
├── .gitignore                   # Python + data exclusions
│
├── data/
│   ├── raw/                     # Generated synthetic data
│   └── processed/               # Feature-engineered datasets
│
├── src/your_project_name/
│   ├── __init__.py
│   ├── config.py                # Centralized configuration
│   ├── data_generator.py        # Domain-informed data generation
│   ├── features.py              # Feature engineering pipeline
│   └── models.py                # ML model implementations
│
├── notebooks/
│   └── 01_analysis.ipynb        # Main analysis notebook
│
└── tests/
    └── test_data_quality.py     # Data validation tests
```

## Documentation

### 📖 Core Documentation
- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Step-by-step first project walkthrough
- **[Use Cases Index](docs/USE_CASES.md)** - Comparison and selection guide for all 6 use cases
- **[Portfolio Overview](docs/PORTFOLIO_OVERVIEW.md)** - Context and career transition narrative

### 🔧 Technical Specifications
Each use case has a detailed specification document covering:
- Objective and business context
- ML problem framing
- Input features and forbidden data (temporal leakage prevention)
- Label definitions
- Model architecture recommendations
- Evaluation metrics
- Notebook structure and plotting standards
- SHAP interpretability requirements

### 📝 Template Documentation
- **[Template README](template/README.md)** - How to use the template
- **[Template Quickstart](template/QUICKSTART.md)** - Fast setup commands
- **[Contributing Guide](template/CONTRIBUTING.md)** - For collaborative projects

## Philosophy

### Domain Expertise Over Code Complexity

This framework emphasizes:

✅ **Problem Framing** - Translating business problems into well-defined ML tasks  
✅ **Domain Knowledge** - Embedding telecom physics in data and features  
✅ **Interpretability** - SHAP explanations for business stakeholders  
✅ **Practical Solutions** - Fit-for-purpose algorithms, not bleeding-edge research  
✅ **End-to-End Thinking** - Data → Features → Model → Insights → Impact

### Why Synthetic Data?

Production telecom data is proprietary and sensitive. Instead of using off-the-shelf synthetic data tools, this framework provides **hand-crafted data generators** that:

- Embed real telecom physics (SINR, Shannon capacity, congestion patterns)
- Control data quality and realism (class imbalance, temporal patterns)
- Maintain interpretability (every data point has a clear causal story)
- Demonstrate domain expertise in how signals propagate and networks behave

### Technical Standards

All templates enforce:
- **Python 3.11+** for modern language features
- **uv** for fast, deterministic dependency management
- **SHAP-compatible versions**: `numpy<2.0`, `xgboost<2.0`, `numba>=0.59.0`
- **Unified plotting**: Seaborn with context switching (notebook vs presentation)
- **Testing**: pytest for data quality and pipeline validation
- **Linting**: Ruff for code quality

## Version History

### v1.0.0 (2025-01-07) - Framework Complete
- ✅ 6 use cases fully specified with problem framing
- ✅ Production-ready project template
- ✅ Domain-informed data generation helpers
- ✅ Unified technical standards (SHAP compatibility, plotting)
- ✅ Complete documentation and usage guides

### Roadmap
- **v1.1.0**: Add notebook templates for each use case
- **v1.2.0**: Enhanced create_project.py with interactive prompts
- **v2.0.0**: Cookiecutter integration for easier project generation

## Contributing

This is primarily a portfolio/framework project, but suggestions and improvements are welcome!

- **Found a bug?** Open an issue
- **Have an enhancement idea?** Start a discussion
- **Want to contribute?** See [CONTRIBUTING.md](template/CONTRIBUTING.md) for guidelines

## License

This framework is released under the **MIT License** - feel free to use it for learning, portfolio building, or commercial projects.

See [LICENSE](LICENSE) for full details.

## Acknowledgments

**Framework structure inspired by:**
- [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science) - Project templates for data science
- [scikit-learn-contrib](https://github.com/scikit-learn-contrib) - ML framework organization
- [FastAPI](https://github.com/tiangolo/fastapi) - Documentation best practices

**Telecom domain knowledge from:**
- 3GPP standards (LTE, 5G NR)
- ITU-T QoE recommendations
- 19+ years in network operations and optimization

## Author

**Adityo Nugroho**  
- Portfolio: https://adityonugrohoid.github.io  
- GitHub: https://github.com/adityonugrohoid  
- LinkedIn: https://www.linkedin.com/in/adityonugrohoid/

---

*Last Updated: January 2025 | Framework Status: Stable (v1.0.0)*
