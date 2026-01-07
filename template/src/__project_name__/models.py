"""
ML model training and evaluation for {PROJECT_NAME}.

This module contains model training, evaluation, and interpretation logic.
Customize this based on your specific ML use case.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error
)

from .config import MODEL_CONFIG, EVAL_CONFIG, PROCESSED_DATA_DIR


class BaseModel:
    """
    Base class for ML models.
    Provides common functionality for training, evaluation, and saving.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or MODEL_CONFIG
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: Input dataframe with features and target
            target_col: Name of target column
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Train set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "classification"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            task_type: "classification" or "regression"
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        metrics = {}
        
        if task_type == "classification":
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            
            # ROC-AUC if binary classification
            if len(np.unique(y_test)) == 2:
                y_proba = self.model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
                
        elif task_type == "regression":
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_test, y_pred)
            metrics["r2"] = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.
        
        Returns:
            DataFrame with feature importances
        """
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("Model does not support feature importance")
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return importance_df
    
    def save(self, filepath: Path) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: Path) -> None:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to saved model
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")


# ============================================================================
# USE-CASE SPECIFIC MODELS (EXAMPLES)
# ============================================================================

class XGBoostClassifier(BaseModel):
    """
    XGBoost classifier for binary/multi-class classification.
    
    Use cases: Churn prediction, root cause analysis, anomaly detection
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: uv add xgboost")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train XGBoost classifier."""
        params = self.config.get("hyperparameters", {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "binary:logistic",
        })
        
        self.model = self.xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("✓ XGBoost model trained")


class LightGBMRegressor(BaseModel):
    """
    LightGBM regressor for regression tasks.
    
    Use cases: QoE prediction, capacity forecasting
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: uv add lightgbm")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train LightGBM regressor."""
        params = self.config.get("hyperparameters", {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100,
        })
        
        self.model = self.lgb.LGBMRegressor(**params)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("✓ LightGBM model trained")


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def cross_validate_model(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    scoring: str = "accuracy"
) -> Dict[str, Any]:
    """
    Perform cross-validation.
    
    Args:
        model: Model instance
        X: Features
        y: Target
        cv_folds: Number of CV folds
        scoring: Scoring metric
        
    Returns:
        Cross-validation results
    """
    scores = cross_val_score(model.model, X, y, cv=cv_folds, scoring=scoring)
    
    return {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "all_scores": scores,
    }


def print_metrics(metrics: Dict[str, float], title: str = "Model Performance") -> None:
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title to print
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:8.4f}")
    print(f"{'='*50}\n")


def main():
    """Example training workflow."""
    # This is a placeholder - customize based on your use case
    print("This is a template. Customize the training workflow for your specific use case.")
    print("\nExample usage:")
    print("""
    from models import XGBoostClassifier
    
    # Load data
    df = pd.read_parquet("data/processed/features.parquet")
    
    # Initialize model
    model = XGBoostClassifier()
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(df, target_col="is_churned")
    
    # Train
    model.train(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test, task_type="classification")
    print_metrics(metrics)
    
    # Feature importance
    importance = model.get_feature_importance()
    print(importance.head(10))
    """)


if __name__ == "__main__":
    main()
