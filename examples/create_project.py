#!/usr/bin/env python3
"""
Automated project creation from Telecom ML Framework template.

This script:
1. Copies the template directory
2. Renames __project_name__ to user's project name
3. Substitutes placeholders in configuration files
4. Creates properly structured Python package

Usage:
    python create_project.py --name churn-prediction --use-case UC1
    python create_project.py --name my-project --use-case UC2 --output ../projects/
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict


# Use case metadata
USE_CASES = {
    "UC1": {
        "name": "Churn Prediction",
        "ml_type": "Binary Classification",
        "description": "Predict customer churn probability using QoE degradation patterns",
        "algorithms": "XGBoost, LightGBM",
    },
    "UC2": {
        "name": "Root Cause Analysis",
        "ml_type": "Ranking / Causal Inference",
        "description": "Identify root causes of network incidents from alarm sequences",
        "algorithms": "Gradient Boosting, Graph Neural Networks",
    },
    "UC3": {
        "name": "Anomaly Detection",
        "ml_type": "Unsupervised Learning",
        "description": "Detect abnormal cell tower behavior without labels",
        "algorithms": "Isolation Forest, LSTM Autoencoder",
    },
    "UC4": {
        "name": "QoE Prediction",
        "ml_type": "Regression",
        "description": "Predict user-perceived quality from network KPIs",
        "algorithms": "LightGBM, CatBoost",
    },
    "UC5": {
        "name": "Capacity Forecasting",
        "ml_type": "Time-Series Forecasting",
        "description": "Forecast future traffic demand for capacity planning",
        "algorithms": "Prophet, ARIMA, LSTM",
    },
    "UC6": {
        "name": "Network Optimization",
        "ml_type": "Reinforcement Learning",
        "description": "Recommend parameter adjustments to improve network KPIs",
        "algorithms": "Q-Learning, Genetic Algorithms",
    },
}


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    # Check use case
    if args.use_case not in USE_CASES:
        print(f"❌ Error: Invalid use case '{args.use_case}'")
        print(f"   Valid options: {', '.join(USE_CASES.keys())}")
        sys.exit(1)

    # Check project name format
    if not args.name.replace("-", "").replace("_", "").isalnum():
        print(f"❌ Error: Project name must contain only alphanumeric, hyphens, underscores")
        print(f"   Got: '{args.name}'")
        sys.exit(1)

    # Check if output directory exists
    output_dir = Path(args.output)
    if not output_dir.exists():
        print(f"❌ Error: Output directory does not exist: {output_dir}")
        sys.exit(1)


def get_template_dir() -> Path:
    """Get path to template directory."""
    script_dir = Path(__file__).parent
    template_dir = script_dir.parent / "template"

    if not template_dir.exists():
        print(f"❌ Error: Template directory not found at {template_dir}")
        print("   Make sure you're running this from the framework root directory")
        sys.exit(1)

    return template_dir


def create_project_from_template(
    project_name: str,
    use_case: str,
    output_dir: Path,
    author: str,
    email: str,
) -> Path:
    """
    Create a new project from template with substitutions.

    Args:
        project_name: Name of the project (e.g., "churn-prediction")
        use_case: Use case code (UC1-UC6)
        output_dir: Directory to create project in
        author: Author name
        email: Author email

    Returns:
        Path to created project
    """
    template_dir = get_template_dir()
    project_path = output_dir / project_name

    # Check if project already exists
    if project_path.exists():
        print(f"❌ Error: Project directory already exists: {project_path}")
        sys.exit(1)

    print(f"📦 Creating project '{project_name}' from template...")

    # Copy template
    shutil.copytree(template_dir, project_path)
    print(f"   ✓ Copied template to {project_path}")

    # Rename __project_name__ to actual project name
    package_name = project_name.replace("-", "_")
    old_pkg_dir = project_path / "src" / "__project_name__"
    new_pkg_dir = project_path / "src" / package_name

    if old_pkg_dir.exists():
        old_pkg_dir.rename(new_pkg_dir)
        print(f"   ✓ Renamed package to '{package_name}'")

    # Prepare substitution variables
    use_case_info = USE_CASES[use_case]
    project_title = project_name.replace("-", " ").title()

    substitutions = {
        "{project-name}": project_name,
        "{project_name}": package_name,
        "__project_name__": package_name,
        "{PROJECT_TITLE}": f"{project_title} - {use_case_info['name']}",
        "{YOUR_NAME}": author,
        "{YOUR_EMAIL}": email,
        "{PROJECT_DESCRIPTION}": use_case_info["description"],
        "{ML_TYPE}": use_case_info["ml_type"],
        "{MODEL_ALGORITHM}": use_case_info["algorithms"].split(",")[0].strip(),
    }

    # Perform substitutions in key files
    files_to_substitute = [
        "pyproject.toml",
        "README.md",
        "QUICKSTART.md",
        "CONTRIBUTING.md",
    ]

    for filename in files_to_substitute:
        file_path = project_path / filename
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            for old, new in substitutions.items():
                content = content.replace(old, new)
            file_path.write_text(content, encoding="utf-8")

    # Also substitute in Python files
    for py_file in new_pkg_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for old, new in substitutions.items():
            content = content.replace(old, new)
        py_file.write_text(content, encoding="utf-8")

    print(f"   ✓ Applied variable substitutions")

    return project_path


def print_next_steps(project_path: Path, project_name: str) -> None:
    """Print next steps for the user."""
    package_name = project_name.replace("-", "_")

    print("\n" + "=" * 70)
    print("✅ Project created successfully!")
    print("=" * 70)
    print(f"\n📁 Project location: {project_path}")
    print(f"📦 Python package: {package_name}")
    print("\n📋 Next steps:")
    print(f"\n1. Navigate to project:")
    print(f"   cd {project_path}")
    print(f"\n2. Install dependencies:")
    print(f"   uv sync")
    print(f"\n3. Generate synthetic data:")
    print(f"   uv run python -m {package_name}.data_generator")
    print(f"\n4. Start Jupyter Lab:")
    print(f"   uv run jupyter lab")
    print(f"\n5. Create analysis notebook in notebooks/ directory")
    print(f"\n6. Follow the use case specification:")
    print(f"   See docs/USE_CASES.md for detailed guidance")
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create a new telecom ML project from template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python create_project.py --name churn-prediction --use-case UC1

  # Specify output directory
  python create_project.py --name my-project --use-case UC2 --output ../projects/

  # With author details
  python create_project.py --name qoe-prediction --use-case UC4 \\
    --author "Jane Smith" --email "jane@example.com"

Use Cases:
  UC1 - Churn Prediction (Binary Classification)
  UC2 - Root Cause Analysis (Ranking/Classification)
  UC3 - Anomaly Detection (Unsupervised)
  UC4 - QoE Prediction (Regression)
  UC5 - Capacity Forecasting (Time-Series)
  UC6 - Network Optimization (Reinforcement Learning)
        """,
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Project name (lowercase, use hyphens)",
    )
    parser.add_argument(
        "--use-case",
        required=True,
        choices=list(USE_CASES.keys()),
        help="Use case code (UC1-UC6)",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--author",
        default="Adityo Nugroho",
        help="Author name (default: 'Adityo Nugroho')",
    )
    parser.add_argument(
        "--email",
        default="adityo.nugroho.id@gmail.com",
        help="Author email (default: 'adityo.nugroho.id@gmail.com')",
    )

    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    # Convert output to Path
    output_dir = Path(args.output).resolve()

    # Show what we're about to do
    use_case_info = USE_CASES[args.use_case]
    print("\n" + "=" * 70)
    print("Telecom ML Framework - Project Creator")
    print("=" * 70)
    print(f"\n📌 Project name:    {args.name}")
    print(f"📌 Use case:        {args.use_case} - {use_case_info['name']}")
    print(f"📌 ML type:         {use_case_info['ml_type']}")
    print(f"📌 Output location: {output_dir}")
    print(f"📌 Author:          {args.author} <{args.email}>")
    print()

    # Create the project
    try:
        project_path = create_project_from_template(
            project_name=args.name,
            use_case=args.use_case,
            output_dir=output_dir,
            author=args.author,
            email=args.email,
        )

        # Print next steps
        print_next_steps(project_path, args.name)

    except Exception as e:
        print(f"\n❌ Error creating project: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

