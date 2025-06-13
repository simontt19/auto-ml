"""
Auto ML Core Package
Contains abstract base classes and core functionality for the ML framework.
"""

from .base_classes import BaseDataIngestion, BaseFeatureEngineering, BaseModelTraining
from .config import Config
from .exceptions import (
    AutoMLError, 
    ValidationError, 
    ConfigurationError, 
    DataIngestionError, 
    FeatureEngineeringError, 
    ModelTrainingError,
    ModelPersistenceError
)

__all__ = [
    'BaseDataIngestion',
    'BaseFeatureEngineering', 
    'BaseModelTraining',
    'Config',
    'AutoMLError',
    'ValidationError',
    'ConfigurationError',
    'DataIngestionError',
    'FeatureEngineeringError',
    'ModelTrainingError',
    'ModelPersistenceError'
] 