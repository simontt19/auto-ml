"""
Standard Feature Engineering Implementation
Concrete implementation of BaseFeatureEngineering for general ML tasks.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging
import traceback

from ...core.base_classes import BaseFeatureEngineering
from ...core.exceptions import FeatureEngineeringError

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

class StandardFeatureEngineering(BaseFeatureEngineering):
    """
    Standard feature engineering implementation for general ML tasks.
    
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the StandardFeatureEngineering class.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        super().__init__(config)
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
            FeatureEngineeringError: If transformation fails
        """
        self.validate_input(data, categorical_columns, numerical_columns)
        
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
            raise FeatureEngineeringError(f"Feature engineering failed: {e}")
    
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
        if not self.is_fitted:
            raise FeatureEngineeringError("FeatureEngineering must be fitted before transform")
        
        # Use stored columns if not provided
        if categorical_columns is None:
            if self.cat_cols is None:
                raise FeatureEngineeringError("Categorical columns must be provided or fitted before transform")
            categorical_columns = self.cat_cols
        if numerical_columns is None:
            if self.num_cols is None:
                raise FeatureEngineeringError("Numerical columns must be provided or fitted before transform")
            numerical_columns = self.num_cols
        
        self.validate_input(data, categorical_columns, numerical_columns)
        
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
            raise FeatureEngineeringError(f"Feature transformation failed: {e}")
    
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
        feature_names = []
        
        # Add encoded categorical features
        for col in categorical_columns:
            feature_names.append(col)
        
        # Add scaled numerical features
        for col in numerical_columns:
            feature_names.append(col)
        
        # Add engineered features
        feature_names.extend([
            'age_binned', 'work_hours_binned', 'education_level',
            'capital_net', 'income_ratio'
        ])
        
        return feature_names
    
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
            if col in data.columns:
                if is_fit:
                    # Fit the encoder
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(data[col].astype(str))
                
                # Transform the column
                encoded_values = self.label_encoders[col].transform(data[col].astype(str))
                data[col] = encoded_values
        
        return data
    
    def _scale_numerical(self, data: pd.DataFrame, numerical_columns: List[str], 
                        is_fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using standardization.
        
        Args:
            data (pd.DataFrame): Input data
            numerical_columns (List[str]): List of numerical column names
            is_fit (bool): Whether to fit the scaler (True) or just transform (False)
            
        Returns:
            pd.DataFrame: Data with numerical columns scaled
        """
        if numerical_columns:
            if is_fit:
                self.scaler.fit(data[numerical_columns])
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
        # Age binning
        if 'age' in data.columns:
            data['age_binned'] = pd.cut(
                data['age'], 
                bins=AGE_BINS, 
                labels=AGE_LABELS, 
                include_lowest=True
            ).cat.codes
        
        # Work hours binning
        if 'hours-per-week' in data.columns:
            data['work_hours_binned'] = pd.cut(
                data['hours-per-week'],
                bins=WORK_HOURS_BINS,
                labels=WORK_HOURS_LABELS,
                include_lowest=True
            ).cat.codes
        
        # Education level mapping
        if 'education-num' in data.columns:
            data['education_level'] = data['education-num'].map(EDUCATION_LEVEL_MAPPING).fillna(0)
        
        # Capital net (gain - loss)
        if 'capital-gain' in data.columns and 'capital-loss' in data.columns:
            data['capital_net'] = data['capital-gain'] - data['capital-loss']
        
        # Income ratio (capital gain / (capital gain + capital loss + 1))
        if 'capital-gain' in data.columns and 'capital-loss' in data.columns:
            total_capital = data['capital-gain'] + data['capital-loss'] + 1
            data['income_ratio'] = data['capital-gain'] / total_capital
        
        return data 