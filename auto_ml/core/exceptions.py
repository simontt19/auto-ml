"""
Custom exceptions for the Auto ML framework.
"""

class AutoMLError(Exception):
    """Base exception for all Auto ML framework errors."""
    pass

class ValidationError(AutoMLError):
    """Raised when data validation fails."""
    pass

class ConfigurationError(AutoMLError):
    """Raised when configuration is invalid or missing."""
    pass

class DataIngestionError(AutoMLError):
    """Raised when data ingestion fails."""
    pass

class FeatureEngineeringError(AutoMLError):
    """Raised when feature engineering fails."""
    pass

class ModelTrainingError(AutoMLError):
    """Raised when model training fails."""
    pass

class ModelPersistenceError(AutoMLError):
    """Raised when model persistence operations fail."""
    pass

class MonitoringError(AutoMLError):
    """Raised when monitoring and drift detection operations fail."""
    pass 