# Quick Start Guide

This guide will help you get started with the project in under 5 minutes.

## Prerequisites

- **Python 3.11+** installed
- **uv** package manager ([install here](https://github.com/astral-sh/uv))

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/{USERNAME}/{REPO_NAME}.git
cd {REPO_NAME}
```

### 2. Install Dependencies

```bash
# uv will automatically create a virtual environment and install all dependencies
uv sync
```

That's it! All dependencies are now installed.

## Generate Data

```bash
# Generate synthetic telecom data
uv run python -m {project_name}.data_generator
```

This will create `data/raw/synthetic_data.parquet`.

**Expected output:**
```
Generating synthetic telecom data...
✓ Saved 10,000 rows to data/raw/synthetic_data.parquet
```

## Engineer Features

```bash
# Create engineered features from raw data
uv run python -m {project_name}.features
```

This will create `data/processed/engineered_features.parquet`.

## Run the Analysis

### Option 1: Jupyter Notebook (Recommended)

```bash
# Launch Jupyter Lab
uv run jupyter lab

# Open: notebooks/01_analysis.ipynb
```

### Option 2: Run Programmatically

```bash
# Run the notebook as a script
uv run jupyter nbconvert --to notebook --execute notebooks/01_analysis.ipynb
```

## Run Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov={project_name} --cov-report=html
```

## Project Structure

```
{project-name}/
├── data/
│   ├── raw/              # Generated synthetic data
│   └── processed/        # Feature-engineered datasets
├── src/{project_name}/   # Source code modules
│   ├── config.py         # Configuration
│   ├── data_generator.py # Data generation
│   ├── features.py       # Feature engineering
│   └── models.py         # ML models
├── notebooks/
│   └── 01_analysis.ipynb # Main analysis notebook
├── tests/                # Test suite
└── README.md             # Full documentation
```

## Next Steps

1. **Explore the data**: Open the notebook and run the EDA section
2. **Customize the generator**: Edit `src/{project_name}/data_generator.py` to adjust data characteristics
3. **Train models**: Follow the notebook's modeling section
4. **Read the full README**: See `README.md` for detailed documentation

## Troubleshooting

### Issue: "uv: command not found"

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Issue: "Module not found"

Make sure you're using `uv run`:
```bash
uv run python -m {project_name}.data_generator
```

### Issue: "Data file not found"

Generate data first:
```bash
uv run python -m {project_name}.data_generator
```

## Resources

- **Full Documentation**: See `README.md`
- **Use Case Specifications**: See `docs/USE_CASE_SPEC.md`
- **Contributing**: See `CONTRIBUTING.md`

---

**Questions?** Open an issue on GitHub or contact {YOUR_EMAIL}
