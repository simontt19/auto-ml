"""
Wine Dataset Data Ingestion
Concrete implementation of BaseDataIngestion for the Wine quality dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from auto_ml.core.base_classes import BaseDataIngestion
from auto_ml.core.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

class WineDataIngestion(BaseDataIngestion):
    """
    Data ingestion implementation for the Wine quality dataset.
    
    This class handles loading and preparing the Wine quality dataset.
    It provides a dataset for wine quality prediction based on chemical properties.
    """
    
    # Dataset metadata for auto-discovery
    DATASET_METADATA = {
        'name': 'wine',
        'description': 'Wine quality dataset for quality prediction',
        'task_type': 'classification',
        'features': {
            'numerical': ['fixed_acidity', 'volatile_acidity', 'citric_acid', 
                        'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                        'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol']
        },
        'target': 'quality',
        'size': '~6K samples',
        'source': 'UCI Machine Learning Repository',
        'license': 'Public Domain',
        'tags': ['wine', 'quality', 'classification', 'uci']
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Wine data ingestion.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        super().__init__(config)
        
        # Define column names for the Wine dataset
        self.columns = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'ph', 'sulphates', 'alcohol', 'quality'
        ]
        
        # Define numerical columns (all features are numerical for Wine)
        self.numerical_columns = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'ph', 'sulphates', 'alcohol'
        ]
        
        # No categorical columns in Wine dataset
        self.categorical_columns = []
        
        self.target_column = 'quality'
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the Wine quality dataset.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test dataframes
            
        Raises:
            DataIngestionError: If data loading fails
        """
        logger.info("Loading Wine quality dataset...")
        
        try:
            import requests
            from sklearn.model_selection import train_test_split
            
            # Download Wine quality dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            
            # Create data directory if it doesn't exist
            Path('data').mkdir(exist_ok=True)
            
            # Download if not exists
            data_file = Path('data/winequality-red.csv')
            if not data_file.exists():
                logger.info("Downloading Wine quality dataset...")
                response = requests.get(url)
                response.raise_for_status()
                with open(data_file, 'w') as f:
                    f.write(response.text)
            
            # Load the data
            data = pd.read_csv(data_file, sep=';')
            
            # Convert quality to binary (high quality: 7-8, low quality: 3-6)
            data['quality'] = (data['quality'] >= 7).astype(int)
            
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
            raise DataIngestionError(f"Failed to load Wine quality dataset: {e}")
    
    def get_categorical_columns(self) -> List[str]:
        """
        Get list of categorical column names.
        
        Returns:
            List[str]: List of categorical column names (empty for Wine)
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