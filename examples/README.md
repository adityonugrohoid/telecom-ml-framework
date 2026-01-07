# Framework Usage Examples

This directory contains examples demonstrating how to use the Telecom ML Framework.

---

## 📋 Contents

### `create_project.py`
Automated script to create a new project from the template with variable substitution.

**Usage:**
```bash
# Basic usage - creates project in current directory
python examples/create_project.py --name churn-prediction --use-case UC1

# Specify output directory
python examples/create_project.py \
  --name churn-prediction \
  --use-case UC1 \
  --output ../my-projects/

# Full options
python examples/create_project.py \
  --name churn-prediction \
  --use-case UC1 \
  --output ../my-projects/ \
  --author "Your Name" \
  --email "your.email@example.com"
```

**What it does:**
1. Copies the `template/` directory
2. Renames `__project_name__` to your project name
3. Substitutes placeholders in configuration files:
   - `pyproject.toml` - project metadata
   - `README.md` - project name, author
   - `config.py` - use case specifics
4. Creates properly structured Python package

---

## 🎯 Use Case Codes

When using `--use-case`, specify one of:

- `UC1` - Churn Prediction (Binary Classification)
- `UC2` - Root Cause Analysis (Ranking/Classification)
- `UC3` - Anomaly Detection (Unsupervised)
- `UC4` - QoE Prediction (Regression)
- `UC5` - Capacity Forecasting (Time-Series)
- `UC6` - Network Optimization (Reinforcement Learning)

---

## 📝 Example Workflows

### Workflow 1: Quick Start

```bash
# Create project
python examples/create_project.py --name my-churn-model --use-case UC1

# Navigate and setup
cd my-churn-model
uv sync

# Generate data
uv run python -m my_churn_model.data_generator

# Start Jupyter
uv run jupyter lab
```

### Workflow 2: Multiple Projects

```bash
# Create all 6 use cases
for uc in UC1 UC2 UC3 UC4 UC5 UC6; do
  python examples/create_project.py \
    --use-case $uc \
    --output ../telecom-projects/
done
```

### Workflow 3: Custom Configuration

```bash
# Create with custom details
python examples/create_project.py \
  --name network-capacity-forecasting \
  --use-case UC5 \
  --author "Jane Smith" \
  --email "jane.smith@example.com" \
  --output ~/ml-projects/
```

---

## 🔍 Script Details

### Command-Line Arguments

| Argument | Required | Description | Example |
|:---|:---:|:---|:---|
| `--name` | ✅ | Project name (lowercase, hyphens) | `churn-prediction` |
| `--use-case` | ✅ | Use case code (UC1-UC6) | `UC1` |
| `--output` | ❌ | Output directory (default: current) | `../projects/` |
| `--author` | ❌ | Author name | `"Your Name"` |
| `--email` | ❌ | Author email | `your@email.com` |

### Substitutions Performed

The script replaces these placeholders:

**In `pyproject.toml`:**
- `{project-name}` → your project name
- `{YOUR_NAME}` → author name
- `{YOUR_EMAIL}` → author email
- `{PROJECT_DESCRIPTION}` → use case description

**In Python files:**
- `__project_name__` → `your_project_name` (package name)

**In README:**
- `{PROJECT_TITLE}` → Your Project Title
- `{BUSINESS_CONTEXT_PARAGRAPH}` → Use case context
- `{ML_TYPE}` → Algorithm type

---

## 🛠️ Extending the Examples

### Add More Examples

You can add additional example scripts:

- `batch_create_projects.py` - Create multiple projects at once
- `migrate_existing_project.py` - Update old projects to new template version
- `validate_project.py` - Check if a project follows framework standards

---

## 📚 Further Reading

- [Getting Started Guide](../docs/GETTING_STARTED.md) - Manual project creation
- [Use Cases Documentation](../docs/USE_CASES.md) - Detailed specifications
- [Template README](../template/README.md) - Template usage guide

---

*For questions or issues, see the [main framework README](../README.md).*

