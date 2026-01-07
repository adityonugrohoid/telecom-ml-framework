"""
{PROJECT_NAME} - {SHORT_DESCRIPTION}

A portfolio project demonstrating AI/ML application to telecom domain challenges.
"""

__version__ = "0.1.0"
__author__ = "{YOUR_NAME}"

from .config import get_config, ensure_directories
from .data_generator import TelecomDataGenerator
from .features import FeatureEngineer
from .models import BaseModel

__all__ = [
    "get_config",
    "ensure_directories",
    "TelecomDataGenerator",
    "FeatureEngineer",
    "BaseModel",
]
