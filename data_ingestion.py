"""
Data Ingestion Module
Handles loading and initial preprocessing of datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Handles data loading and initial preprocessing"""
    
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        self.column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
    
    def load_adult_dataset(self):
        """Load the Adult dataset from UCI"""
        logger.info("Loading Adult dataset...")
        
        try:
            # Load training data
            train_data = pd.read_csv(
                f"{self.data_path}/adult.data",
                names=self.column_names,
                skipinitialspace=True,
                na_values='?'
            )
            
            # Load test data - skip the header line and handle trailing periods
            test_data = pd.read_csv(
                f"{self.data_path}/adult.test",
                names=self.column_names,
                skipinitialspace=True,
                na_values='?',
                skiprows=1  # Skip the header row
            )
            
            logger.info(f"Training data shape: {train_data.shape}")
            logger.info(f"Test data shape: {test_data.shape}")
            
            # Clean income column (remove trailing periods in test set)
            train_data['income'] = train_data['income'].str.strip()
            test_data['income'] = test_data['income'].str.strip().str.rstrip('.')
            
            # Create binary target
            train_data['target'] = (train_data['income'] == '>50K').astype(int)
            test_data['target'] = (test_data['income'] == '>50K').astype(int)
            
            logger.info(f"Target distribution in training: {train_data['target'].value_counts().to_dict()}")
            logger.info(f"Target distribution in test: {test_data['target'].value_counts().to_dict()}")
            
            # Verify we have both classes in both datasets
            train_classes = train_data['target'].unique()
            test_classes = test_data['target'].unique()
            logger.info(f"Classes in training: {train_classes}")
            logger.info(f"Classes in test: {test_classes}")
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_feature_columns(self):
        """Get list of feature columns (excluding target)"""
        return [col for col in self.column_names if col != 'income']
    
    def get_categorical_columns(self):
        """Get list of categorical columns"""
        return [
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country'
        ]
    
    def get_numerical_columns(self):
        """Get list of numerical columns"""
        return [
            'age', 'fnlwgt', 'education-num', 'capital-gain',
            'capital-loss', 'hours-per-week'
        ]

if __name__ == "__main__":
    # Test the data ingestion
    ingestion = DataIngestion()
    train_data, test_data = ingestion.load_adult_dataset()
    print("Data ingestion test completed successfully!") 