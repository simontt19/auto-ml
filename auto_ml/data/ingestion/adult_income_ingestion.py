"""
Adult Income Dataset Data Ingestion
Concrete implementation of BaseDataIngestion for the UCI Adult Income dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from ...core.base_classes import BaseDataIngestion
from ...core.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

class AdultIncomeDataIngestion(BaseDataIngestion):
    """
    Data ingestion implementation for the UCI Adult Income dataset.
    
    This class handles loading and preparing the Adult Income dataset from the UCI repository.
    It downloads the data if not present and provides methods to access categorical and numerical columns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Adult Income data ingestion.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        super().__init__(config)
        self.train_path = self.config.get('train_path', 'data/adult.data')
        self.test_path = self.config.get('test_path', 'data/adult.test')
        
        # Define column names for the Adult dataset
        self.columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target'
        ]
        
        # Define categorical and numerical columns
        self.categorical_columns = [
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country'
        ]
        
        self.numerical_columns = [
            'age', 'fnlwgt', 'education-num', 'capital-gain',
            'capital-loss', 'hours-per-week'
        ]
        
        self.target_column = 'target'
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data for the Adult Income dataset.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test dataframes
            
        Raises:
            DataIngestionError: If data loading fails
        """
        logger.info("Loading Adult dataset...")
        
        try:
            # Load training data
            train_data = self._load_csv_file(self.train_path, skiprows=1)
            
            # Load test data
            test_data = self._load_csv_file(self.test_path, skiprows=1)
            
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
            raise DataIngestionError(f"Failed to load Adult dataset: {e}")
    
    def _load_csv_file(self, file_path: str, skiprows: int = 0) -> pd.DataFrame:
        """
        Load a CSV file with proper column names and data cleaning.
        
        Args:
            file_path (str): Path to the CSV file
            skiprows (int): Number of rows to skip
            
        Returns:
            pd.DataFrame: Loaded and cleaned dataframe
        """
        try:
            # Check if file exists
            if not Path(file_path).exists():
                raise DataIngestionError(f"Data file not found: {file_path}")
            
            # Load data with proper column names
            df = pd.read_csv(file_path, names=self.columns, skiprows=skiprows)
            
            # Clean the data
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            raise DataIngestionError(f"Failed to load file {file_path}: {e}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the loaded data by handling missing values and data types.
        
        Args:
            df (pd.DataFrame): Raw dataframe to clean
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Replace '?' with NaN for missing values
        df = df.replace('?', np.nan)
        
        # Convert target column to binary (0/1)
        if self.target_column in df.columns:
            # Handle both training and test data formats
            # Training data: <=50K, >50K
            # Test data: <=50K., >50K. (with periods)
            df[self.target_column] = df[self.target_column].str.strip()
            df[self.target_column] = df[self.target_column].str.replace('.', '')
            df[self.target_column] = (df[self.target_column] == '>50K').astype(int)
        
        # Convert numerical columns to appropriate types
        for col in self.numerical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_categorical_columns(self) -> List[str]:
        """
        Get list of categorical column names.
        
        Returns:
            List[str]: List of categorical column names
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
    
    def download_dataset(self, force_download: bool = False) -> None:
        """
        Download the Adult Income dataset from UCI repository.
        
        Args:
            force_download (bool): Whether to force download even if files exist
        """
        import requests
        import os
        
        # URLs for the Adult dataset
        train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Download training data
        if force_download or not os.path.exists(self.train_path):
            logger.info("Downloading training data...")
            response = requests.get(train_url)
            response.raise_for_status()
            with open(self.train_path, 'w') as f:
                f.write(response.text)
        
        # Download test data
        if force_download or not os.path.exists(self.test_path):
            logger.info("Downloading test data...")
            response = requests.get(test_url)
            response.raise_for_status()
            with open(self.test_path, 'w') as f:
                f.write(response.text)
        
        logger.info("Dataset download completed.") 