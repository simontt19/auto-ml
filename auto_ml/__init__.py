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
    Config,
    AutoMLError,
    ValidationError,
    ConfigurationError
)

from .data.ingestion import AdultIncomeDataIngestion
from .features.engineering import StandardFeatureEngineering
from .models.training import ClassificationModelTraining

__version__ = "0.1.0"

__all__ = [
    # Core components
    'BaseDataIngestion',
    'BaseFeatureEngineering',
    'BaseModelTraining',
    'Config',
    'AutoMLError',
    'ValidationError',
    'ConfigurationError',
    
    # Concrete implementations
    'AdultIncomeDataIngestion',
    'StandardFeatureEngineering',
    'ClassificationModelTraining',
] 