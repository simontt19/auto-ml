"""
Classification Model Training Implementation
Concrete implementation of BaseModelTraining for classification tasks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix
)
import lightgbm as lgb
import logging
import json
import traceback
from datetime import datetime
from scipy.stats import uniform, randint

from ...core.base_classes import BaseModelTraining
from ...core.exceptions import ModelTrainingError

logger = logging.getLogger(__name__)

class ClassificationModelTraining(BaseModelTraining):
    """
    Classification model training implementation.
    
    This class provides comprehensive ML model management including:
    - Multiple algorithm training and evaluation
    - Hyperparameter optimization using GridSearchCV and RandomizedSearchCV
    - Cross-validation and performance metrics
    - Model selection and feature importance analysis
    - Model persistence and loading capabilities
    
    Attributes:
        models (Dict[str, Any]): Dictionary of trained models
        results (Dict[str, Dict]): Dictionary of model performance results
        best_model (Any): Best performing model instance
        best_model_name (str): Name of the best performing model
        optimized_models (Dict[str, Any]): Dictionary of hyperparameter-optimized models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ClassificationModelTraining class.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        super().__init__(config)
        self.optimized_models = {}
        self.enable_hyperparameter_optimization = self.config.get('enable_hyperparameter_optimization', True)
        
        # Define hyperparameter search spaces for each model
        self.hyperparameter_spaces = {
            'logistic_regression': {
                'C': uniform(0.1, 10.0),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': randint(100, 1000)
            },
            'random_forest': {
                'n_estimators': randint(50, 300),
                'max_depth': [None] + list(range(10, 50, 10)),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20),
                'subsample': uniform(0.6, 0.4)
            },
            'lightgbm': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'num_leaves': randint(20, 100),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            }
        }
        
        # Define base model configurations
        self.base_models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1, objective='binary')
        }
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series, 
                    feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models with optional hyperparameter optimization.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            feature_names (List[str]): List of feature names
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of model performance results
        """
        logger.info("Starting model training and evaluation...")
        
        try:
            # Prepare data
            X_train_clean = X_train[feature_names].fillna(0)
            X_val_clean = X_val[feature_names].fillna(0)
            
            # Get algorithms from config or use defaults
            algorithms = self.config.get('algorithms', list(self.base_models.keys()))
            
            # Train and evaluate each model
            for model_name in algorithms:
                if model_name not in self.base_models:
                    logger.warning(f"Model {model_name} not supported, skipping...")
                    continue
                    
                logger.info(f"Training {model_name}...")
                
                if self.enable_hyperparameter_optimization:
                    optimized_model = self._optimize_hyperparameters(
                        model_name, X_train_clean, y_train
                    )
                    self.optimized_models[model_name] = optimized_model
                    model = optimized_model
                else:
                    model = self.base_models[model_name]
                    model.fit(X_train_clean, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val_clean)
                y_pred_proba = model.predict_proba(X_val_clean)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                
                # Store model and results
                self.models[model_name] = model
                self.results[model_name] = metrics
                
                logger.info(f"{model_name} - AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            
            # Select best model
            self._select_best_model()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error in train_models: {e}\n{traceback.format_exc()}")
            raise ModelTrainingError(f"Model training failed: {e}")
    
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
        """
        logger.info(f"Performing {cv}-fold cross-validation on {self.best_model_name}...")
        
        try:
            X_clean = X[feature_names].fillna(0)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                self.best_model,
                X_clean,
                y,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring='roc_auc'
            )
            
            cv_results = {
                'mean_auc': cv_scores.mean(),
                'std_auc': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            logger.info(f"Cross-validation results: {cv_results['mean_auc']:.4f} (+/- {cv_results['std_auc']:.4f})")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross_validate_best_model: {e}\n{traceback.format_exc()}")
            raise ModelTrainingError(f"Cross-validation failed: {e}")
    
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
        """
        logger.info("Training final model on full dataset...")
        
        try:
            X_clean = X_full[feature_names].fillna(0)
            
            # Create a fresh instance of the best model with optimized parameters
            if self.best_model_name in self.optimized_models:
                # Use the optimized model as template
                final_model = self.optimized_models[self.best_model_name]
                # Create a fresh instance with the same parameters
                final_model = type(final_model)(**final_model.get_params())
            else:
                # Fall back to base model
                final_model = self.base_models[self.best_model_name]
            
            # Train on full dataset
            final_model.fit(X_clean, y_full)
            
            return final_model
            
        except Exception as e:
            logger.error(f"Error in train_final_model: {e}\n{traceback.format_exc()}")
            raise ModelTrainingError(f"Final model training failed: {e}")
    
    def get_feature_importance(self, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the best model.
        
        Args:
            feature_names (List[str]): List of feature names
            
        Returns:
            Optional[pd.DataFrame]: Feature importance DataFrame or None if not available
        """
        if self.best_model is None:
            raise ModelTrainingError("No model has been trained yet")
        
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importance = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                importance = np.abs(self.best_model.coef_[0])
            else:
                return None
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}\n{traceback.format_exc()}")
            return None
    
    def _optimize_hyperparameters(self, model_name: str, X_train: pd.DataFrame, 
                                 y_train: pd.Series) -> Any:
        """
        Optimize hyperparameters for a given model using RandomizedSearchCV.
        
        Args:
            model_name (str): Name of the model to optimize
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Any: Optimized model with best hyperparameters
        """
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        try:
            base_model = self.base_models[model_name]
            param_space = self.hyperparameter_spaces[model_name]
            
            # Use RandomizedSearchCV for hyperparameter optimization
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_space,
                n_iter=20,  # Number of parameter settings sampled
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1,  # Use all available cores
                random_state=42,
                verbose=0
            )
            
            # Fit the random search
            random_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters for {model_name}: {random_search.best_params_}")
            logger.info(f"Best CV score for {model_name}: {random_search.best_score_:.4f}")
            
            return random_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization for {model_name}: {e}\n{traceback.format_exc()}")
            # Fall back to base model if optimization fails
            logger.warning(f"Falling back to base model for {model_name}")
            base_model = self.base_models[model_name]
            base_model.fit(X_train, y_train)
            return base_model
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (pd.Series): True target values
            y_pred (np.ndarray): Predicted target values
            y_pred_proba (np.ndarray): Predicted probabilities
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba)
        } 