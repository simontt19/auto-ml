"""
Advanced Feature Engineering
Sophisticated feature engineering techniques for enterprise ML pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, chi2, 
    SelectFromModel, RFE, SelectPercentile
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, TargetEncoder
)
from sklearn.decomposition import PCA
from sklearn.feature_extraction import FeatureHasher
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

from auto_ml.core.base_classes import BaseFeatureEngineering
from auto_ml.core.exceptions import FeatureEngineeringError

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering(BaseFeatureEngineering):
    """
    Advanced feature engineering implementation with sophisticated techniques.
    
    Features:
    - Feature selection (mutual information, chi-square, recursive feature elimination)
    - Advanced encoding (target encoding, feature hashing)
    - Feature interaction detection
    - Feature importance analysis
    - Dimensionality reduction (PCA)
    - Clustering-based features
    - Statistical feature creation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize advanced feature engineering.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        super().__init__(config)
        
        # Advanced feature engineering settings
        self.feature_selection_method = config.get('feature_selection_method', 'mutual_info')
        self.feature_selection_k = config.get('feature_selection_k', 10)
        self.feature_selection_percentile = config.get('feature_selection_percentile', 80)
        self.use_target_encoding = config.get('use_target_encoding', False)
        self.use_feature_hashing = config.get('use_feature_hashing', False)
        self.use_pca = config.get('use_pca', False)
        self.pca_components = config.get('pca_components', 0.95)
        self.use_clustering_features = config.get('use_clustering_features', False)
        self.n_clusters = config.get('n_clusters', 3)
        self.use_interaction_features = config.get('use_interaction_features', False)
        self.use_statistical_features = config.get('use_statistical_features', False)
        
        # Store transformers for later use
        self.transformers = {}
        self.feature_names = []
        self.selected_features = []
        
    def engineer_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                         target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply advanced feature engineering techniques.
        
        Args:
            train_data (pd.DataFrame): Training data
            test_data (pd.DataFrame): Test data
            target_column (str): Target column name
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Engineered training and test data
            
        Raises:
            FeatureEngineeringError: If feature engineering fails
        """
        logger.info("Starting advanced feature engineering...")
        
        try:
            # Separate features and target
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[target_column])
            
            # Store original feature names
            self.feature_names = list(X_train.columns)
            
            # Step 1: Basic preprocessing
            X_train_processed, X_test_processed = self._basic_preprocessing(X_train, X_test)
            
            # Step 2: Advanced encoding
            if self.use_target_encoding:
                X_train_processed, X_test_processed = self._apply_target_encoding(
                    X_train_processed, X_test_processed, y_train
                )
            
            if self.use_feature_hashing:
                X_train_processed, X_test_processed = self._apply_feature_hashing(
                    X_train_processed, X_test_processed
                )
            
            # Step 3: Feature selection
            X_train_selected, X_test_selected = self._apply_feature_selection(
                X_train_processed, X_test_processed, y_train
            )
            
            # Step 4: Feature interactions
            if self.use_interaction_features:
                X_train_selected, X_test_selected = self._create_interaction_features(
                    X_train_selected, X_test_selected
                )
            
            # Step 5: Statistical features
            if self.use_statistical_features:
                X_train_selected, X_test_selected = self._create_statistical_features(
                    X_train_selected, X_test_selected
                )
            
            # Step 6: Clustering features
            if self.use_clustering_features:
                X_train_selected, X_test_selected = self._create_clustering_features(
                    X_train_selected, X_test_selected
                )
            
            # Step 7: Dimensionality reduction
            if self.use_pca:
                X_train_selected, X_test_selected = self._apply_pca(
                    X_train_selected, X_test_selected
                )
            
            # Step 8: Final scaling
            X_train_final, X_test_final = self._final_scaling(X_train_selected, X_test_selected)
            
            # Add target back
            train_engineered = X_train_final.copy()
            train_engineered[target_column] = y_train
            
            test_engineered = X_test_final.copy()
            test_engineered[target_column] = test_data[target_column]
            
            # Store feature information
            self.engineered_features_info = {
                'original_features': len(self.feature_names),
                'engineered_features': len(X_train_final.columns),
                'selected_features': len(self.selected_features),
                'feature_names': list(X_train_final.columns),
                'transformers_used': list(self.transformers.keys())
            }
            
            logger.info(f"Advanced feature engineering completed.")
            logger.info(f"Original features: {self.engineered_features_info['original_features']}")
            logger.info(f"Engineered features: {self.engineered_features_info['engineered_features']}")
            logger.info(f"Selected features: {self.engineered_features_info['selected_features']}")
            
            return train_engineered, test_engineered
            
        except Exception as e:
            raise FeatureEngineeringError(f"Advanced feature engineering failed: {e}")
    
    def _basic_preprocessing(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply basic preprocessing (missing values, encoding, scaling)."""
        logger.info("Applying basic preprocessing...")
        
        # Handle missing values
        X_train_clean = X_train.fillna(X_train.mode().iloc[0] if X_train.isnull().any().any() else 0)
        X_test_clean = X_test.fillna(X_test.mode().iloc[0] if X_test.isnull().any().any() else 0)
        
        # Identify categorical and numerical columns
        categorical_cols = X_train_clean.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X_train_clean.select_dtypes(include=['int64', 'float64']).columns
        
        # Encode categorical variables
        if len(categorical_cols) > 0:
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X_train_clean[col] = le.fit_transform(X_train_clean[col].astype(str))
                X_test_clean[col] = le.transform(X_test_clean[col].astype(str))
                label_encoders[col] = le
            self.transformers['label_encoders'] = label_encoders
        
        # REVIEW: chi2 feature selection requires non-negative features, so use MinMaxScaler for chi2, StandardScaler otherwise
        if len(numerical_cols) > 0:
            if getattr(self, 'feature_selection_method', None) == 'chi2':
                scaler = MinMaxScaler()
                logger.info("Using MinMaxScaler for chi2 feature selection (non-negative features required)")
            else:
                scaler = StandardScaler()
            X_train_clean[numerical_cols] = scaler.fit_transform(X_train_clean[numerical_cols])
            X_test_clean[numerical_cols] = scaler.transform(X_test_clean[numerical_cols])
            self.transformers['numerical_scaler'] = scaler
        
        return X_train_clean, X_test_clean
    
    def _apply_target_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply target encoding to categorical variables."""
        logger.info("Applying target encoding...")
        
        categorical_cols = X_train.select_dtypes(include=['int64']).columns
        
        for col in categorical_cols:
            # Calculate target means for each category
            target_means = y_train.groupby(X_train[col]).mean()
            
            # Apply to training data
            X_train[f'{col}_target_encoded'] = X_train[col].map(target_means)
            
            # Apply to test data (use training means)
            X_test[f'{col}_target_encoded'] = X_test[col].map(target_means)
            
            # Fill missing values with global mean
            global_mean = y_train.mean()
            X_train[f'{col}_target_encoded'].fillna(global_mean, inplace=True)
            X_test[f'{col}_target_encoded'].fillna(global_mean, inplace=True)
        
        return X_train, X_test
    
    def _apply_feature_hashing(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply feature hashing for high-dimensional categorical features."""
        logger.info("Applying feature hashing...")
        
        # Select categorical columns for hashing
        categorical_cols = X_train.select_dtypes(include=['int64']).columns[:5]  # Limit to first 5
        
        for col in categorical_cols:
            hasher = FeatureHasher(n_features=8, input_type='string')
            
            # Convert to string and hash
            train_hashed = hasher.transform(X_train[col].astype(str).values.reshape(-1, 1)).toarray()
            test_hashed = hasher.transform(X_test[col].astype(str).values.reshape(-1, 1)).toarray()
            
            # Add hashed features
            for i in range(8):
                X_train[f'{col}_hash_{i}'] = train_hashed[:, i]
                X_test[f'{col}_hash_{i}'] = test_hashed[:, i]
        
        return X_train, X_test
    
    def _apply_feature_selection(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply feature selection techniques."""
        logger.info(f"Applying feature selection using {self.feature_selection_method}...")
        
        if self.feature_selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=self.feature_selection_k)
        elif self.feature_selection_method == 'chi2':
            selector = SelectKBest(score_func=chi2, k=self.feature_selection_k)
        elif self.feature_selection_method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=self.feature_selection_k)
        elif self.feature_selection_method == 'percentile':
            selector = SelectPercentile(score_func=mutual_info_classif, percentile=self.feature_selection_percentile)
        elif self.feature_selection_method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=self.feature_selection_k)
        else:
            # No feature selection
            self.selected_features = list(X_train.columns)
            return X_train, X_test
        
        # Fit and transform
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        if hasattr(selector, 'get_support'):
            selected_mask = selector.get_support()
            self.selected_features = X_train.columns[selected_mask].tolist()
        else:
            self.selected_features = list(X_train.columns)
        
        # Convert back to DataFrame
        X_train_selected = pd.DataFrame(X_train_selected, columns=self.selected_features, index=X_train.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=self.selected_features, index=X_test.index)
        
        self.transformers['feature_selector'] = selector
        
        logger.info(f"Selected {len(self.selected_features)} features out of {len(X_train.columns)}")
        
        return X_train_selected, X_test_selected
    
    def _create_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create interaction features between numerical variables."""
        logger.info("Creating interaction features...")
        
        numerical_cols = X_train.select_dtypes(include=['float64']).columns
        
        if len(numerical_cols) >= 2:
            # Create pairwise interactions for first few numerical columns
            for i, col1 in enumerate(numerical_cols[:3]):
                for col2 in numerical_cols[i+1:4]:
                    interaction_name = f'{col1}_x_{col2}'
                    X_train[interaction_name] = X_train[col1] * X_train[col2]
                    X_test[interaction_name] = X_test[col1] * X_test[col2]
        
        return X_train, X_test
    
    def _create_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create statistical features (mean, std, min, max across rows)."""
        logger.info("Creating statistical features...")
        
        numerical_cols = X_train.select_dtypes(include=['float64']).columns
        
        if len(numerical_cols) > 1:
            # Row-wise statistics
            X_train['row_mean'] = X_train[numerical_cols].mean(axis=1)
            X_train['row_std'] = X_train[numerical_cols].std(axis=1)
            X_train['row_min'] = X_train[numerical_cols].min(axis=1)
            X_train['row_max'] = X_train[numerical_cols].max(axis=1)
            
            X_test['row_mean'] = X_test[numerical_cols].mean(axis=1)
            X_test['row_std'] = X_test[numerical_cols].std(axis=1)
            X_test['row_min'] = X_test[numerical_cols].min(axis=1)
            X_test['row_max'] = X_test[numerical_cols].max(axis=1)
        
        return X_train, X_test
    
    def _create_clustering_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create clustering-based features."""
        logger.info("Creating clustering features...")
        
        numerical_cols = X_train.select_dtypes(include=['float64']).columns
        
        if len(numerical_cols) >= 2:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            
            # Fit on training data
            cluster_labels_train = kmeans.fit_predict(X_train[numerical_cols])
            cluster_labels_test = kmeans.predict(X_test[numerical_cols])
            
            # Add cluster features
            X_train['cluster_label'] = cluster_labels_train
            X_test['cluster_label'] = cluster_labels_test
            
            # Add distance to cluster centers
            X_train['cluster_distance'] = kmeans.transform(X_train[numerical_cols]).min(axis=1)
            X_test['cluster_distance'] = kmeans.transform(X_test[numerical_cols]).min(axis=1)
            
            self.transformers['kmeans'] = kmeans
        
        return X_train, X_test
    
    def _apply_pca(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply Principal Component Analysis for dimensionality reduction."""
        logger.info("Applying PCA...")
        
        numerical_cols = X_train.select_dtypes(include=['float64']).columns
        
        if len(numerical_cols) > 1:
            if self.pca_components < 1:
                # Use explained variance ratio
                pca = PCA(n_components=self.pca_components)
            else:
                # Use number of components
                pca = PCA(n_components=int(self.pca_components))
            
            # Fit and transform
            pca_train = pca.fit_transform(X_train[numerical_cols])
            pca_test = pca.transform(X_test[numerical_cols])
            
            # Create new DataFrame with PCA components
            pca_columns = [f'pca_{i}' for i in range(pca_train.shape[1])]
            
            # Replace numerical columns with PCA components
            X_train_pca = X_train.drop(columns=numerical_cols)
            X_test_pca = X_test.drop(columns=numerical_cols)
            
            for i, col in enumerate(pca_columns):
                X_train_pca[col] = pca_train[:, i]
                X_test_pca[col] = pca_test[:, i]
            
            self.transformers['pca'] = pca
            
            logger.info(f"PCA reduced {len(numerical_cols)} features to {len(pca_columns)} components")
            
            return X_train_pca, X_test_pca
        
        return X_train, X_test
    
    def _final_scaling(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply final scaling to all features."""
        logger.info("Applying final scaling...")
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        self.transformers['final_scaler'] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if 'feature_selector' in self.transformers:
            selector = self.transformers['feature_selector']
            if hasattr(selector, 'scores_'):
                return dict(zip(self.feature_names, selector.scores_))
        
        return {}
    
    def get_engineered_features_info(self) -> Dict[str, Any]:
        """Get information about engineered features."""
        return getattr(self, 'engineered_features_info', {})
    
    def fit_transform(self, data: pd.DataFrame, categorical_columns: List[str], 
                     numerical_columns: List[str]) -> pd.DataFrame:
        """
        Fit the feature engineering pipeline and transform the data.
        
        Args:
            data (pd.DataFrame): Input data
            categorical_columns (List[str]): List of categorical column names
            numerical_columns (List[str]): List of numerical column names
            
        Returns:
            pd.DataFrame: Transformed data
        """
        # This is a simplified version for compatibility
        # In practice, you'd want to store the fitted state
        logger.info("Fit-transform not fully implemented for advanced feature engineering")
        return data
    
    def transform(self, data: pd.DataFrame, categorical_columns: Optional[List[str]] = None,
                 numerical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform data using the fitted feature engineering pipeline.
        
        Args:
            data (pd.DataFrame): Input data
            categorical_columns (Optional[List[str]]): List of categorical column names
            numerical_columns (Optional[List[str]]): List of numerical column names
            
        Returns:
            pd.DataFrame: Transformed data
        """
        # This is a simplified version for compatibility
        logger.info("Transform not fully implemented for advanced feature engineering")
        return data
    
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
        # Return the feature names from the last engineering operation
        if hasattr(self, 'engineered_features_info'):
            return self.engineered_features_info.get('feature_names', [])
        else:
            # Fallback to original feature names
            return categorical_columns + numerical_columns 