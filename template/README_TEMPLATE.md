# Project Template

This is the standardized template for all telecom ML projects in this workspace.

## What This Template Provides

- ✅ **Consistent structure** across all projects
- ✅ **Pre-configured tooling** (uv, ruff, pytest, GitHub Actions)
- ✅ **Domain-focused modules** (data generator with telecom physics)
- ✅ **Portfolio-ready documentation** (README emphasizes domain expertise)
- ✅ **Minimal but complete** setup for rapid project creation

## Using This Template

**📖 See the detailed guide**: [`..docs/TEMPLATE_USAGE.md`](../docs/TEMPLATE_USAGE.md)

### Quick Start

```bash
# Copy template to new project
cp -r _shared/TEMPLATE ../01-churn-prediction

# Navigate to new project
cd ../01-churn-prediction

# Rename package directory
mv src/__project_name__ src/churn_prediction

# Customize files (see TEMPLATE_USAGE.md for full checklist)
```

## Template Structure

```
TEMPLATE/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI
├── data/
│   ├── raw/                    # Generated synthetic data
│   └── processed/              # Feature-engineered data
├── src/__project_name__/       # ⚠️ MUST BE RENAMED
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration management
│   ├── data_generator.py       # Domain-informed data generation
│   ├── features.py             # Feature engineering
│   └── models.py               # ML model training/evaluation
├── notebooks/
│   └── (create 01_analysis.ipynb)
├── tests/
│   └── test_data_quality.py    # Data validation tests
├── docs/
│   └── (add use-case-specific docs)
├── .gitignore                  # Python + ML .gitignore
├── pyproject.toml              # uv dependency management
├── README.md                   # Main documentation (CUSTOMIZE!)
├── QUICKSTART.md               # Getting started guide
└── CONTRIBUTING.md             # Contribution guidelines
```

## Core Design Principles

### 1. Domain Expertise Over Code Complexity
- README focuses on business context and domain insights
- Data generators embed telecom physics
- Feature engineering reflects domain knowledge

### 2. Minimal But Complete
- ~5 core modules (config, data, features, models, __init__)
- Single notebook per use case
- Only essential dependencies

### 3. Ship Fast, Iterate Later
- Template provides working skeleton
- Customize only what's necessary for MVP
- Add complexity incrementally

### 4. Independent But Consistent
- Each project is fully self-contained
- No shared code dependencies (to prevent cascade breaking)
- Shared template ensures consistency

## Customization Checklist

When creating a new project:

- [ ] Copy template to new directory
- [ ] Rename `src/__project_name__/` to actual project name
- [ ] Update `pyproject.toml` (name, description, authors)
- [ ] Replace all placeholders in `README.md`
- [ ] Customize `data_generator.py` for use case
- [ ] Update `config.py` with use-case parameters
- [ ] Add use-case-specific features in `features.py`
- [ ] Create analysis notebook
- [ ] Update tests for use case
- [ ] Initialize Git repo
- [ ] Push to GitHub

**Detailed instructions**: See `../docs/TEMPLATE_USAGE.md`

## Placeholder Reference

Files with placeholders to replace:

| File | Placeholders |
|:---|:---|
| `README.md` | `{PROJECT_TITLE}`, `{BUSINESS_CONTEXT_PARAGRAPH}`, `{ML_TYPE}`, etc. |
| `pyproject.toml` | `{project-name}`, `{YOUR_NAME}`, `{YOUR_EMAIL}` |
| `src/__project_name__/*.py` | `{PROJECT_NAME}`, `{MODEL_ALGORITHM}`, etc. |

**Tip**: Search for `{` to find all placeholders.

## Dependencies

All projects use:
- **Python 3.11+**
- **uv** for package management
- **XGBoost/LightGBM** for ML (customize per use case)
- **pandas, numpy** for data processing
- **matplotlib, seaborn** for visualization
- **pytest** for testing
- **ruff** for linting/formatting

## Testing

Template includes basic data quality tests:

```bash
uv run pytest tests/ -v
```

Customize tests for your specific use case.

## CI/CD

GitHub Actions workflow included:
- Linting with ruff
- Testing with pytest
- Runs on push/PR to main

## Documentation Philosophy

README structure emphasizes:
1. **Business context** (why this problem matters)
2. **Domain approach** (how telecom expertise informs the solution)
3. **Key findings** (actionable insights)
4. **Technical details** (brief, not the focus)

This positions you as a **domain expert who uses ML**, not just a coder.

---

## Need Help?

- **Template Usage Guide**: `../docs/TEMPLATE_USAGE.md`
- **Use Case Specs**: `../docs/use-cases/XX-YOUR-USE-CASE.md`
- **Portfolio Overview**: `../docs/PORTFOLIO_OVERVIEW.md`

---

*This template is designed for rapid, consistent creation of portfolio-grade ML projects.*
