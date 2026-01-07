"""
Configuration management for {PROJECT_NAME}.

This module centralizes all configuration parameters, making it easy to
adjust settings without modifying core logic.
"""

from pathlib import Path
from typing import Dict, Any
import yaml


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# ============================================================================
# DATA GENERATION CONFIG
# ============================================================================

DATA_GEN_CONFIG = {
    "random_seed": 42,
    "n_samples": 10_000,
    "test_size": 0.2,
    "validation_size": 0.1,
    
    # Use case specific parameters (customize per project)
    "use_case_params": {
        # Example: Churn prediction
        # "churn_rate": 0.15,
        # "observation_window_days": 30,
        # "prediction_window_days": 30,
    }
}


# ============================================================================
# FEATURE ENGINEERING CONFIG
# ============================================================================

FEATURE_CONFIG = {
    "categorical_features": [],
    "numerical_features": [],
    "datetime_features": [],
    
    # Rolling window aggregations
    "rolling_windows": [7, 30],  # days
    
    # Engineered features to create
    "create_features": True,
}


# ============================================================================
# MODEL TRAINING CONFIG
# ============================================================================

MODEL_CONFIG = {
    "algorithm": "{MODEL_ALGORITHM}",  # e.g., "xgboost", "lightgbm", "lstm"
    
    # Cross-validation
    "cv_folds": 5,
    "cv_strategy": "time_series",  # or "kfold", "stratified"
    
    # Hyperparameters (customize per model)
    "hyperparameters": {
        # Example for XGBoost:
        # "max_depth": 6,
        # "learning_rate": 0.1,
        # "n_estimators": 100,
        # "objective": "binary:logistic",
    },
    
    # Training
    "early_stopping_rounds": 10,
    "verbose": True,
}


# ============================================================================
# EVALUATION CONFIG
# ============================================================================

EVAL_CONFIG = {
    "primary_metric": "{PRIMARY_METRIC}",  # e.g., "roc_auc", "rmse", "f1"
    "threshold": 0.5,  # For classification
    
    # Metrics to compute
    "compute_metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ],
}


# ============================================================================
# VISUALIZATION CONFIG
# ============================================================================

VIZ_CONFIG = {
    "style": "seaborn-v0_8-darkgrid",
    "palette": "husl",
    "context": "notebook",
    "figure_size": (12, 6),
    "dpi": 100,
}


# ============================================================================
# UTILITIES
# ============================================================================

def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_custom_config(config_path: Path) -> Dict[str, Any]:
    """
    Load custom configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_config() -> Dict[str, Any]:
    """
    Get complete configuration dictionary.
    
    Returns:
        Merged configuration from all config sections
    """
    return {
        "data_gen": DATA_GEN_CONFIG,
        "features": FEATURE_CONFIG,
        "model": MODEL_CONFIG,
        "eval": EVAL_CONFIG,
        "viz": VIZ_CONFIG,
        "paths": {
            "root": PROJECT_ROOT,
            "data": DATA_DIR,
            "raw": RAW_DATA_DIR,
            "processed": PROCESSED_DATA_DIR,
            "notebooks": NOTEBOOKS_DIR,
        }
    }


if __name__ == "__main__":
    # Test configuration
    ensure_directories()
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Random seed: {DATA_GEN_CONFIG['random_seed']}")
