"""
Auto ML Framework
A comprehensive, production-ready machine learning framework.

This framework provides:
- Abstract base classes for extensible ML components
- Configuration management with YAML support
- Multiple dataset support
- Hyperparameter optimization
- Model persistence and versioning
- Production deployment capabilities
"""

from .core import (
    BaseDataIngestion,
    BaseFeatureEngineering,
    BaseModelTraining,
    ConfigManager,
    get_config,
    get_huggingface_token,
    is_debug_mode,
    get_log_level,
    AutoMLError,
    ValidationError,
    ConfigurationError,
    DataIngestionError,
    FeatureEngineeringError,
    ModelTrainingError,
    ModelPersistenceError
)

from .data.ingestion import AdultIncomeDataIngestion
from .features.engineering import StandardFeatureEngineering
from .models.training import ClassificationModelTraining
from .models.persistence import ModelPersistence
from .deployment.api import ModelAPI

__version__ = "0.1.0"

__all__ = [
    # Core components
    'BaseDataIngestion',
    'BaseFeatureEngineering',
    'BaseModelTraining',
    'ConfigManager',
    'get_config',
    'get_huggingface_token',
    'is_debug_mode',
    'get_log_level',
    'AutoMLError',
    'ValidationError',
    'ConfigurationError',
    'DataIngestionError',
    'FeatureEngineeringError',
    'ModelTrainingError',
    'ModelPersistenceError',
    
    # Concrete implementations
    'AdultIncomeDataIngestion',
    'StandardFeatureEngineering',
    'ClassificationModelTraining',
    'ModelPersistence',
    'ModelAPI',
] 