"""
Feature Engineering Module
Handles data preprocessing, feature creation, and transformation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging
import traceback

logger = logging.getLogger(__name__)

# Constants for feature engineering
AGE_BINS = [0, 25, 35, 50, 65, 100]
AGE_LABELS = ['18-25', '26-35', '36-50', '51-65', '65+']
WORK_HOURS_BINS = [0, 35, 40, 50, 100]
WORK_HOURS_LABELS = ['Part-time', 'Full-time', 'Overtime', 'Extended']
UNSEEN_CATEGORY_VALUE = -1

# Education level mapping
EDUCATION_LEVEL_MAPPING = {
    1: 0, 2: 0, 3: 0, 4: 0,  # Preschool to 4th grade
    5: 1, 6: 1, 7: 1, 8: 1,  # 5th-8th grade
    9: 2, 10: 2, 11: 2, 12: 2, 13: 2,  # 9th-12th grade
    14: 3, 15: 3, 16: 3  # College and above
}

class FeatureEngineering:
    """
    Handles feature engineering and preprocessing for machine learning pipelines.
    
    This class provides comprehensive data transformation capabilities including:
    - Missing value imputation for categorical and numerical features
    - Categorical encoding with unseen category handling
    - Numerical scaling and normalization
    - Feature creation and engineering
    
    Attributes:
        label_encoders (Dict[str, LabelEncoder]): Dictionary of label encoders for categorical columns
        scaler (StandardScaler): StandardScaler for numerical features
        imputer_cat (SimpleImputer): Imputer for categorical features (mode strategy)
        imputer_num (SimpleImputer): Imputer for numerical features (median strategy)
        is_fitted (bool): Whether the transformers have been fitted
        cat_cols (Optional[List[str]]): Stored categorical columns
        num_cols (Optional[List[str]]): Stored numerical columns
    """
    
    def __init__(self) -> None:
        """Initialize the FeatureEngineering class with default transformers."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.imputer_num = SimpleImputer(strategy='median')
        self.is_fitted = False
        self.cat_cols: Optional[List[str]] = None
        self.num_cols: Optional[List[str]] = None
    
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
            ValueError: If input data is empty or columns are missing
            Exception: If any transformation step fails
        """
        self._validate_input(data, categorical_columns, numerical_columns)
        
        # Store columns for later use (inference)
        self.cat_cols = categorical_columns
        self.num_cols = numerical_columns
        
        logger.info("Starting feature engineering...")
        try:
            data_processed = data.copy()
            
            # Apply transformations in order
            data_processed = self._handle_missing_values(
                data_processed, categorical_columns, numerical_columns
            )
            data_processed = self._encode_categorical(
                data_processed, categorical_columns
            )
            data_processed = self._scale_numerical(
                data_processed, numerical_columns
            )
            data_processed = self._create_features(data_processed)
            
            self.is_fitted = True
            logger.info(f"Feature engineering completed. Final shape: {data_processed.shape}")
            return data_processed
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {e}\n{traceback.format_exc()}")
            raise
    
    def transform(self, data: pd.DataFrame, categorical_columns: List[str] = None, 
                 numerical_columns: List[str] = None) -> pd.DataFrame:
        """
        Transform data using fitted transformers.
        
        Args:
            data (pd.DataFrame): Input data to transform
            categorical_columns (List[str], optional): List of categorical column names
            numerical_columns (List[str], optional): List of numerical column names
            
        Returns:
            pd.DataFrame: Transformed data with all features engineered
            
        Raises:
            ValueError: If transformers are not fitted or input validation fails
            Exception: If any transformation step fails
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineering must be fitted before transform")
        
        # Use stored columns if not provided
        if categorical_columns is None:
            if self.cat_cols is None:
                raise ValueError("Categorical columns must be provided or fitted before transform")
            categorical_columns = self.cat_cols
        if numerical_columns is None:
            if self.num_cols is None:
                raise ValueError("Numerical columns must be provided or fitted before transform")
            numerical_columns = self.num_cols
        
        self._validate_input(data, categorical_columns, numerical_columns)
        
        logger.info("Transforming data using fitted transformers...")
        try:
            data_processed = data.copy()
            
            # Apply transformations in order (without fitting)
            data_processed = self._handle_missing_values(
                data_processed, categorical_columns, numerical_columns, is_fit=False
            )
            data_processed = self._encode_categorical(
                data_processed, categorical_columns, is_fit=False
            )
            data_processed = self._scale_numerical(
                data_processed, numerical_columns, is_fit=False
            )
            data_processed = self._create_features(data_processed)
            
            logger.info(f"Transform completed. Final shape: {data_processed.shape}")
            return data_processed
            
        except Exception as e:
            logger.error(f"Error in transform: {e}\n{traceback.format_exc()}")
            raise
    
    def _validate_input(self, data: pd.DataFrame, categorical_columns: List[str], 
                       numerical_columns: List[str]) -> None:
        """
        Validate input data and column specifications.
        
        Args:
            data (pd.DataFrame): Input data to validate
            categorical_columns (List[str]): List of categorical column names
            numerical_columns (List[str]): List of numerical column names
            
        Raises:
            ValueError: If validation fails
        """
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        if not categorical_columns and not numerical_columns:
            raise ValueError("At least one categorical or numerical column must be specified")
        
        # Check if all specified columns exist in the data
        all_columns = categorical_columns + numerical_columns
        missing_columns = [col for col in all_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in data: {missing_columns}")
        
        # Check for overlap between categorical and numerical columns
        overlap = set(categorical_columns) & set(numerical_columns)
        if overlap:
            raise ValueError(f"Columns cannot be both categorical and numerical: {overlap}")
    
    def _handle_missing_values(self, data: pd.DataFrame, categorical_columns: List[str], 
                              numerical_columns: List[str], is_fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values in categorical and numerical columns.
        
        Args:
            data (pd.DataFrame): Input data
            categorical_columns (List[str]): List of categorical column names
            numerical_columns (List[str]): List of numerical column names
            is_fit (bool): Whether to fit the imputers (True) or just transform (False)
            
        Returns:
            pd.DataFrame: Data with missing values imputed
        """
        if categorical_columns:
            if is_fit:
                self.imputer_cat.fit(data[categorical_columns])
            data[categorical_columns] = self.imputer_cat.transform(data[categorical_columns])
        
        if numerical_columns:
            if is_fit:
                self.imputer_num.fit(data[numerical_columns])
            data[numerical_columns] = self.imputer_num.transform(data[numerical_columns])
        
        return data
    
    def _encode_categorical(self, data: pd.DataFrame, categorical_columns: List[str], 
                           is_fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using label encoding.
        
        Args:
            data (pd.DataFrame): Input data
            categorical_columns (List[str]): List of categorical column names
            is_fit (bool): Whether to fit the encoders (True) or just transform (False)
            
        Returns:
            pd.DataFrame: Data with categorical columns encoded
        """
        for col in categorical_columns:
            if is_fit:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col])
            else:
                # Handle unseen categories by mapping them to a default value
                data[col] = data[col].map(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in self.label_encoders[col].classes_ else UNSEEN_CATEGORY_VALUE
                )
        
        return data
    
    def _scale_numerical(self, data: pd.DataFrame, numerical_columns: List[str], 
                        is_fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical variables using StandardScaler.
        
        Args:
            data (pd.DataFrame): Input data
            numerical_columns (List[str]): List of numerical column names
            is_fit (bool): Whether to fit the scaler (True) or just transform (False)
            
        Returns:
            pd.DataFrame: Data with numerical columns scaled
        """
        if numerical_columns:
            if is_fit:
                data[numerical_columns] = self.scaler.fit_transform(data[numerical_columns])
            else:
                data[numerical_columns] = self.scaler.transform(data[numerical_columns])
        
        return data
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with additional engineered features
        """
        # Age groups
        data['age_group'] = pd.cut(
            data['age'], 
            bins=AGE_BINS, 
            labels=AGE_LABELS
        )
        data['age_group'] = data['age_group'].cat.codes
        
        # Education level groups
        data['education_level'] = data['education-num'].map(EDUCATION_LEVEL_MAPPING)
        
        # Work hours categories
        data['work_hours_category'] = pd.cut(
            data['hours-per-week'],
            bins=WORK_HOURS_BINS,
            labels=WORK_HOURS_LABELS
        )
        data['work_hours_category'] = data['work_hours_category'].cat.codes
        
        # Capital gains/losses combined
        data['capital_net'] = data['capital-gain'] - data['capital-loss']
        
        # Wealth indicator (simplified)
        data['wealth_indicator'] = (
            (data['capital-gain'] > data['capital-gain'].median()) |
            (data['education-num'] > 12)
        ).astype(int)
        
        return data
    
    def get_feature_names(self, categorical_columns: List[str], 
                         numerical_columns: List[str]) -> List[str]:
        """
        Get list of all feature names after engineering.
        
        Args:
            categorical_columns (List[str]): List of categorical column names
            numerical_columns (List[str]): List of numerical column names
            
        Returns:
            List[str]: Complete list of feature names including engineered features
        """
        base_features = categorical_columns + numerical_columns
        engineered_features = [
            'age_group', 'education_level', 'work_hours_category',
            'capital_net', 'wealth_indicator'
        ]
        return base_features + engineered_features

if __name__ == "__main__":
    # Test feature engineering
    from data_ingestion import DataIngestion
    
    ingestion = DataIngestion()
    train_data, test_data = ingestion.load_adult_dataset()
    
    fe = FeatureEngineering()
    cat_cols = ingestion.get_categorical_columns()
    num_cols = ingestion.get_numerical_columns()
    
    train_processed = fe.fit_transform(train_data, cat_cols, num_cols)
    test_processed = fe.transform(test_data, cat_cols, num_cols)
    
    print("Feature engineering test completed successfully!") 