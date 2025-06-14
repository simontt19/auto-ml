"""
Breast Cancer Dataset Data Ingestion
Concrete implementation of BaseDataIngestion for the Breast Cancer dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from ...core.base_classes import BaseDataIngestion
from ...core.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

class BreastCancerDataIngestion(BaseDataIngestion):
    """
    Data ingestion implementation for the Breast Cancer dataset.
    
    This class handles loading and preparing the Breast Cancer dataset from scikit-learn.
    It provides a medical dataset for breast cancer diagnosis prediction.
    """
    
    # Dataset metadata for auto-discovery
    DATASET_METADATA = {
        'name': 'breast_cancer',
        'description': 'Breast cancer diagnosis dataset',
        'task_type': 'classification',
        'features': {
            'numerical': ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
                        'mean_smoothness', 'mean_compactness', 'mean_concavity',
                        'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension']
        },
        'target': 'diagnosis',
        'size': '~570 samples',
        'source': 'UCI Machine Learning Repository',
        'license': 'Public Domain',
        'tags': ['medical', 'cancer', 'classification', 'uci']
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Breast Cancer data ingestion.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        super().__init__(config)
        
        # Define column names for the Breast Cancer dataset
        self.columns = [
            'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
            'mean_smoothness', 'mean_compactness', 'mean_concavity',
            'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
            'diagnosis'
        ]
        
        # Define numerical columns (all features are numerical for Breast Cancer)
        self.numerical_columns = [
            'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
            'mean_smoothness', 'mean_compactness', 'mean_concavity',
            'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension'
        ]
        
        # No categorical columns in Breast Cancer dataset
        self.categorical_columns = []
        
        self.target_column = 'diagnosis'
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the Breast Cancer dataset from scikit-learn.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test dataframes
            
        Raises:
            DataIngestionError: If data loading fails
        """
        logger.info("Loading Breast Cancer dataset...")
        
        try:
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            
            # Load the Breast Cancer dataset
            cancer = load_breast_cancer()
            
            # Create DataFrame
            data = pd.DataFrame(
                cancer.data,
                columns=cancer.feature_names
            )
            
            # Add target column
            data['diagnosis'] = cancer.target
            
            # Convert to binary (0: malignant, 1: benign)
            # Note: scikit-learn uses 0 for malignant, 1 for benign
            # We'll keep this as is for consistency
            
            # Split into train and test
            train_data, test_data = train_test_split(
                data, 
                test_size=0.2, 
                random_state=42, 
                stratify=data[self.target_column]
            )
            
            # Reset indices
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            
            # Validate loaded data
            self.validate_data(train_data)
            self.validate_data(test_data)
            
            # Store data information
            self.data_info = {
                'train_shape': train_data.shape,
                'test_shape': test_data.shape,
                'categorical_columns': self.categorical_columns,
                'numerical_columns': self.numerical_columns,
                'target_column': self.target_column,
                'target_distribution_train': train_data[self.target_column].value_counts().to_dict(),
                'target_distribution_test': test_data[self.target_column].value_counts().to_dict()
            }
            
            logger.info(f"Training data shape: {train_data.shape}")
            logger.info(f"Test data shape: {test_data.shape}")
            logger.info(f"Target distribution in training: {self.data_info['target_distribution_train']}")
            logger.info(f"Target distribution in test: {self.data_info['target_distribution_test']}")
            
            return train_data, test_data
            
        except Exception as e:
            raise DataIngestionError(f"Failed to load Breast Cancer dataset: {e}")
    
    def get_categorical_columns(self) -> List[str]:
        """
        Get list of categorical column names.
        
        Returns:
            List[str]: List of categorical column names (empty for Breast Cancer)
        """
        return self.categorical_columns.copy()
    
    def get_numerical_columns(self) -> List[str]:
        """
        Get list of numerical column names.
        
        Returns:
            List[str]: List of numerical column names
        """
        return self.numerical_columns.copy()
    
    def get_target_column(self) -> str:
        """
        Get target column name.
        
        Returns:
            str: Target column name
        """
        return self.target_column 