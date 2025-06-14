"""
Iris Dataset Data Ingestion
Concrete implementation of BaseDataIngestion for the Iris flower dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from auto_ml.core.base_classes import BaseDataIngestion
from auto_ml.core.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

class IrisDataIngestion(BaseDataIngestion):
    """
    Data ingestion implementation for the Iris flower dataset.
    
    This class handles loading and preparing the Iris dataset from scikit-learn.
    It provides a simple, clean dataset for testing and demonstration purposes.
    """
    
    # Dataset metadata for auto-discovery
    DATASET_METADATA = {
        'name': 'iris',
        'description': 'Iris flower dataset for species classification',
        'task_type': 'classification',
        'features': {
            'numerical': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        },
        'target': 'species',
        'size': '150 samples',
        'source': 'UCI Machine Learning Repository',
        'license': 'Public Domain',
        'tags': ['flowers', 'classification', 'uci', 'small']
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        self.numerical_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.categorical_columns = []
        self.target_column = 'species'
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading Iris dataset...")
        try:
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            iris = load_iris()
            data = pd.DataFrame(
                iris.data,
                columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            )
            data['species'] = iris.target
            species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
            data['species'] = data['species'].map(species_names)
            data['species'] = (data['species'] == 'setosa').astype(int)
            train_data, test_data = train_test_split(
                data, 
                test_size=0.2, 
                random_state=42, 
                stratify=data[self.target_column]
            )
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            self.validate_data(train_data)
            self.validate_data(test_data)
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
            raise DataIngestionError(f"Failed to load Iris dataset: {e}")
    def get_categorical_columns(self) -> List[str]:
        return self.categorical_columns.copy()
    def get_numerical_columns(self) -> List[str]:
        return self.numerical_columns.copy()
    def get_target_column(self) -> str:
        return self.target_column 