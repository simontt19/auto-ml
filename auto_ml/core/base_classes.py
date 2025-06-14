"""
Abstract base classes for the Auto ML framework.
Defines the interface for data ingestion, feature engineering, model training, and monitoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import logging
from .exceptions import DataIngestionError, FeatureEngineeringError, ModelTrainingError, MonitoringError
import numpy as np

logger = logging.getLogger(__name__)

class BaseDataIngestion(ABC):
    """
    Abstract base class for data ingestion.
    
    This class defines the interface for loading and preparing data from various sources.
    Concrete implementations should handle specific data formats and sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data ingestion.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        self.config = config or {}
        self.data_info: Dict[str, Any] = {}
    
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test dataframes
            
        Raises:
            DataIngestionError: If data loading fails
        """
        pass
    
    @abstractmethod
    def get_categorical_columns(self) -> List[str]:
        """
        Get list of categorical column names.
        
        Returns:
            List[str]: List of categorical column names
        """
        pass
    
    @abstractmethod
    def get_numerical_columns(self) -> List[str]:
        """
        Get list of numerical column names.
        
        Returns:
            List[str]: List of numerical column names
        """
        pass
    
    @abstractmethod
    def get_target_column(self) -> str:
        """
        Get target column name.
        
        Returns:
            str: Target column name
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate loaded data.
        
        Args:
            data (pd.DataFrame): Data to validate
            
        Raises:
            DataIngestionError: If validation fails
        """
        if data.empty:
            raise DataIngestionError("Loaded data is empty")
        
        if data.isnull().all().any():
            raise DataIngestionError("Data contains columns with all null values")
        
        logger.info(f"Data validation passed. Shape: {data.shape}")
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.
        
        Returns:
            Dict[str, Any]: Data information dictionary
        """
        return self.data_info

class BaseFeatureEngineering(ABC):
    """
    Abstract base class for feature engineering.
    
    This class defines the interface for transforming and engineering features.
    Concrete implementations should handle specific data types and transformations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineering.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        self.config = config or {}
        self.is_fitted = False
        self.feature_names: List[str] = []
    
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame, categorical_columns: List[str], 
                     numerical_columns: List[str]) -> pd.DataFrame:
        """
        Fit transformers and transform the data.
        
        Args:
            data (pd.DataFrame): Input data to transform
            categorical_columns (List[str]): List of categorical column names
            numerical_columns (List[str]): List of numerical column names
            
        Returns:
            pd.DataFrame: Transformed data with all features engineered
            
        Raises:
            FeatureEngineeringError: If transformation fails
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, categorical_columns: Optional[List[str]] = None,
                 numerical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform data using fitted transformers.
        
        Args:
            data (pd.DataFrame): Input data to transform
            categorical_columns (Optional[List[str]]): List of categorical column names
            numerical_columns (Optional[List[str]]): List of numerical column names
            
        Returns:
            pd.DataFrame: Transformed data with all features engineered
            
        Raises:
            FeatureEngineeringError: If transformation fails
        """
        pass
    
    @abstractmethod
    def get_feature_names(self, categorical_columns: List[str], 
                         numerical_columns: List[str]) -> List[str]:
        """
        Get list of feature names after engineering.
        
        Args:
            categorical_columns (List[str]): List of categorical column names
            numerical_columns (List[str]): List of numerical column names
            
        Returns:
            List[str]: List of engineered feature names
        """
        pass
    
    def validate_input(self, data: pd.DataFrame, categorical_columns: List[str], 
                      numerical_columns: List[str]) -> None:
        """
        Validate input data and column specifications.
        
        Args:
            data (pd.DataFrame): Input data to validate
            categorical_columns (List[str]): List of categorical column names
            numerical_columns (List[str]): List of numerical column names
            
        Raises:
            FeatureEngineeringError: If validation fails
        """
        if data.empty:
            raise FeatureEngineeringError("Input data cannot be empty")
        
        if not categorical_columns and not numerical_columns:
            raise FeatureEngineeringError("At least one categorical or numerical column must be specified")
        
        # Check if all specified columns exist in the data
        all_columns = categorical_columns + numerical_columns
        missing_columns = [col for col in all_columns if col not in data.columns]
        if missing_columns:
            raise FeatureEngineeringError(f"Columns not found in data: {missing_columns}")
        
        # Check for overlap between categorical and numerical columns
        overlap = set(categorical_columns) & set(numerical_columns)
        if overlap:
            raise FeatureEngineeringError(f"Columns cannot be both categorical and numerical: {overlap}")

class BaseModelTraining(ABC):
    """
    Abstract base class for model training.
    
    This class defines the interface for training and evaluating models.
    Concrete implementations should handle specific ML tasks and algorithms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model training.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        self.config = config or {}
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
    
    @abstractmethod
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and evaluate them.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            feature_names (List[str]): List of feature names
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of model performance results
            
        Raises:
            ModelTrainingError: If training fails
        """
        pass
    
    @abstractmethod
    def cross_validate_best_model(self, X: pd.DataFrame, y: pd.Series,
                                 feature_names: List[str], cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the best model.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target values
            feature_names (List[str]): List of feature names
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Cross-validation results
            
        Raises:
            ModelTrainingError: If cross-validation fails
        """
        pass
    
    @abstractmethod
    def train_final_model(self, X_full: pd.DataFrame, y_full: pd.Series,
                         feature_names: List[str]) -> Any:
        """
        Train the final model on the full dataset.
        
        Args:
            X_full (pd.DataFrame): Full dataset features
            y_full (pd.Series): Full dataset target
            feature_names (List[str]): List of feature names
            
        Returns:
            Any: Final trained model
            
        Raises:
            ModelTrainingError: If training fails
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the best model.
        
        Args:
            feature_names (List[str]): List of feature names
            
        Returns:
            Optional[pd.DataFrame]: Feature importance DataFrame or None if not available
        """
        pass
    
    def _select_best_model(self, metric: str = 'auc') -> None:
        """
        Select the best model based on a specified metric.
        
        Args:
            metric (str): Metric to use for model selection
        """
        if not self.results:
            raise ModelTrainingError("No models have been trained yet")
        
        best_score = -1
        best_model_name = None
        
        for model_name, metrics in self.results.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = model_name
        
        if best_model_name is None:
            raise ModelTrainingError(f"Could not find best model using metric: {metric}")
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model selected: {best_model_name} ({metric}: {best_score:.4f})")
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save training results to JSON file.
        
        Args:
            filename (Optional[str]): Output filename, auto-generated if None
            
        Returns:
            str: Path to the saved results file
        """
        import json
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_results_{timestamp}.json"
        
        try:
            # Convert results to serializable format
            results_serializable = {}
            for model_name, metrics in self.results.items():
                results_serializable[model_name] = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in metrics.items()
                }
            
            results_serializable['best_model'] = self.best_model_name
            results_serializable['timestamp'] = datetime.now().isoformat()
            
            with open(filename, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            logger.info(f"Results saved to {filename}")
            return filename
            
        except Exception as e:
            raise ModelTrainingError(f"Error saving results: {e}")

class BaseMonitoring(ABC):
    """
    Abstract base class for model monitoring and drift detection.
    
    This class defines the interface for monitoring model performance,
    detecting data drift, and generating alerts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize monitoring system.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        self.config = config or {}
        self.baseline_data: Optional[pd.DataFrame] = None
        self.monitoring_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
    
    @abstractmethod
    def set_baseline(self, data: pd.DataFrame, target_column: Optional[str] = None,
                    predictions: Optional[np.ndarray] = None) -> None:
        """
        Set baseline data for monitoring.
        
        Args:
            data (pd.DataFrame): Baseline data
            target_column (Optional[str]): Target column name
            predictions (Optional[np.ndarray]): Model predictions on baseline data
            
        Raises:
            MonitoringError: If baseline setting fails
        """
        pass
    
    @abstractmethod
    def detect_drift(self, current_data: pd.DataFrame, 
                    target_column: Optional[str] = None,
                    predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect drift in current data compared to baseline.
        
        Args:
            current_data (pd.DataFrame): Current data to check for drift
            target_column (Optional[str]): Target column name
            predictions (Optional[np.ndarray]): Model predictions on current data
            
        Returns:
            Dict[str, Any]: Drift detection results
            
        Raises:
            MonitoringError: If drift detection fails
        """
        pass
    
    @abstractmethod
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get summary of monitoring results.
        
        Returns:
            Dict[str, Any]: Monitoring summary
        """
        pass
    
    def validate_monitoring_data(self, data: pd.DataFrame) -> None:
        """
        Validate data for monitoring.
        
        Args:
            data (pd.DataFrame): Data to validate
            
        Raises:
            MonitoringError: If validation fails
        """
        if data.empty:
            raise MonitoringError("Monitoring data cannot be empty")
        
        if data.isnull().all().any():
            raise MonitoringError("Data contains columns with all null values")
        
        logger.info(f"Monitoring data validation passed. Shape: {data.shape}")
    
    def clear_history(self) -> None:
        """
        Clear monitoring history.
        """
        self.monitoring_history = []
        self.alerts = []
        logger.info("Monitoring history cleared") 