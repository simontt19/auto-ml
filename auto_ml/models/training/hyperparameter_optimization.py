"""
Hyperparameter Optimization and AutoML
Advanced hyperparameter tuning, model selection, and ensemble methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    StratifiedKFold, KFold, train_test_split
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# LightGBM and XGBoost for advanced models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from auto_ml.core.base_classes import BaseModelTraining
from auto_ml.core.exceptions import ModelTrainingError

logger = logging.getLogger(__name__)

class HyperparameterOptimization(BaseModelTraining):
    """
    Advanced hyperparameter optimization and AutoML system.
    
    Features:
    - Automated hyperparameter tuning with Optuna
    - Model selection and comparison
    - Ensemble methods (Voting, Stacking, Bagging)
    - Advanced ML algorithms (LightGBM, XGBoost)
    - Cross-validation and performance evaluation
    - Automated feature importance analysis
    - Model interpretability tools
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hyperparameter optimization.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        super().__init__(config)
        
        # Optimization settings
        self.optimization_method = config.get('optimization_method', 'optuna')  # 'optuna', 'grid', 'random'
        self.n_trials = config.get('n_trials', 100)
        self.cv_folds = config.get('cv_folds', 5)
        self.scoring_metric = config.get('scoring_metric', 'f1')
        self.timeout = config.get('timeout', 3600)  # 1 hour timeout
        
        # Model selection settings
        self.use_ensemble = config.get('use_ensemble', True)
        self.ensemble_method = config.get('ensemble_method', 'voting')  # 'voting', 'stacking', 'bagging'
        self.top_k_models = config.get('top_k_models', 3)
        
        # Advanced algorithms
        self.use_lightgbm = config.get('use_lightgbm', True) and LIGHTGBM_AVAILABLE
        self.use_xgboost = config.get('use_xgboost', True) and XGBOOST_AVAILABLE
        
        # Storage
        self.best_models = {}
        self.optimization_results = {}
        self.ensemble_model = None
        self.feature_importance = {}
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and return their performance metrics.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation targets
            feature_names (List[str]): List of feature names
            
        Returns:
            Dict[str, Dict[str, float]]: Model performance metrics
        """
        logger.info("Training multiple models for comparison...")
        
        models_to_test = self._get_models_to_test()
        results = {}
        
        for model_name, model_config in models_to_test.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Create and train model with default parameters
                model = model_config['class'](random_state=42)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, average='weighted'),
                    'recall': recall_score(y_val, y_pred, average='weighted'),
                    'f1': f1_score(y_val, y_pred, average='weighted')
                }
                
                if y_pred_proba is not None:
                    metrics['auc'] = roc_auc_score(y_val, y_pred_proba)
                
                results[model_name] = metrics
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")
                continue
        
        return results
    
    def cross_validate_best_model(self, X: pd.DataFrame, y: pd.Series,
                                 feature_names: List[str], cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the best model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Targets
            feature_names (List[str]): List of feature names
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Use the best model from optimization if available, otherwise use RandomForest
        if hasattr(self, 'best_model') and self.best_model is not None:
            model = self.best_model
        else:
            model = RandomForestClassifier(random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, 
            cv=cv, 
            scoring=self.scoring_metric
        )
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'model_type': type(model).__name__
        }
    
    def train_final_model(self, X_full: pd.DataFrame, y_full: pd.Series,
                         feature_names: List[str]) -> Any:
        """
        Train the final model on the full dataset.
        
        Args:
            X_full (pd.DataFrame): Full training features
            y_full (pd.Series): Full training targets
            feature_names (List[str]): List of feature names
            
        Returns:
            Any: Trained final model
        """
        logger.info("Training final model on full dataset...")
        
        # Use the best model from optimization if available, otherwise use RandomForest
        if hasattr(self, 'best_model') and self.best_model is not None:
            # Clone the best model and retrain on full data
            final_model = type(self.best_model)(**self.best_model.get_params())
        else:
            final_model = RandomForestClassifier(random_state=42)
        
        # Train on full dataset
        final_model.fit(X_full, y_full)
        
        # Store the final model
        self.final_model = final_model
        
        return final_model
    
    def get_feature_importance(self, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the best model.
        
        Args:
            feature_names (List[str]): List of feature names
            
        Returns:
            Optional[pd.DataFrame]: Feature importance dataframe
        """
        if not hasattr(self, 'best_model') or self.best_model is None:
            return None
        
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                return importance_df
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    def train_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                   target_column: str) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization and model training.
        
        Args:
            train_data (pd.DataFrame): Training data
            test_data (pd.DataFrame): Test data
            target_column (str): Target column name
            
        Returns:
            Dict[str, Any]: Training results and model information
            
        Raises:
            ModelTrainingError: If training fails
        """
        logger.info("Starting hyperparameter optimization...")
        
        try:
            # Prepare data
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            
            # Step 1: Individual model optimization
            logger.info("Optimizing individual models...")
            individual_results = self._optimize_individual_models(X_train, y_train, X_test, y_test)
            
            # Step 2: Ensemble creation
            if self.use_ensemble and len(individual_results['models']) > 1:
                logger.info("Creating ensemble model...")
                ensemble_results = self._create_ensemble_model(
                    individual_results['models'], X_train, y_train, X_test, y_test
                )
            else:
                ensemble_results = None
            
            # Step 3: Final evaluation
            final_results = self._evaluate_final_models(
                individual_results, ensemble_results, X_test, y_test
            )
            
            # Step 4: Feature importance analysis
            feature_importance = self._analyze_feature_importance(
                individual_results['best_model'], X_train, y_train
            )
            
            # Compile results
            results = {
                'individual_models': individual_results,
                'ensemble_model': ensemble_results,
                'final_results': final_results,
                'feature_importance': feature_importance,
                'optimization_summary': {
                    'total_models_tested': len(individual_results['models']),
                    'best_individual_model': individual_results['best_model_name'],
                    'best_ensemble_model': ensemble_results['model_name'] if ensemble_results else None,
                    'optimization_time': time.time() - getattr(self, '_start_time', time.time()),
                    'config_used': self.config
                }
            }
            
            # Store results for later access
            self.optimization_results = results
            
            logger.info("Hyperparameter optimization completed successfully!")
            logger.info(f"Best individual model: {individual_results['best_model_name']}")
            if ensemble_results:
                logger.info(f"Best ensemble model: {ensemble_results['model_name']}")
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Hyperparameter optimization failed: {e}")
    
    def _optimize_individual_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Optimize individual models using specified method."""
        self._start_time = time.time()
        
        models_to_test = self._get_models_to_test()
        results = {'models': {}, 'best_model': None, 'best_model_name': None, 'best_score': 0}
        
        for model_name, model_config in models_to_test.items():
            logger.info(f"Optimizing {model_name}...")
            
            try:
                if self.optimization_method == 'optuna':
                    model_result = self._optimize_with_optuna(
                        model_name, model_config, X_train, y_train, X_test, y_test
                    )
                elif self.optimization_method == 'grid':
                    model_result = self._optimize_with_grid_search(
                        model_name, model_config, X_train, y_train, X_test, y_test
                    )
                else:  # random
                    model_result = self._optimize_with_random_search(
                        model_name, model_config, X_train, y_train, X_test, y_test
                    )
                
                results['models'][model_name] = model_result
                
                # Update best model
                if model_result['test_score'] > results['best_score']:
                    results['best_score'] = model_result['test_score']
                    results['best_model'] = model_result['model']
                    results['best_model_name'] = model_name
                    
            except Exception as e:
                logger.warning(f"Failed to optimize {model_name}: {e}")
                continue
        
        return results
    
    def _get_models_to_test(self) -> Dict[str, Dict[str, Any]]:
        """Get dictionary of models to test with their configurations."""
        models = {
            'RandomForest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'GradientBoosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'LogisticRegression': {
                'class': LogisticRegression,
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'SVM': {
                'class': SVC,
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                }
            },
            'KNN': {
                'class': KNeighborsClassifier,
                'params': {
                    'n_neighbors': [3, 5, 7, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
        
        # Add advanced models if available
        if self.use_lightgbm:
            models['LightGBM'] = {
                'class': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7, -1],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        
        if self.use_xgboost:
            models['XGBoost'] = {
                'class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        
        return models
    
    def _optimize_with_optuna(self, model_name: str, model_config: Dict[str, Any],
                             X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            # Get hyperparameters for this trial
            params = {}
            for param_name, param_values in model_config['params'].items():
                if isinstance(param_values, list):
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    elif isinstance(param_values[0], float):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    params[param_name] = param_values
            
            # Create and train model
            model = model_config['class'](**params, random_state=42)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.cv_folds, 
                scoring=self.scoring_metric
            )
            
            return cv_scores.mean()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner()
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Get best parameters and train final model
        best_params = study.best_params
        best_model = model_config['class'](**best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        test_score = self._calculate_score(y_test, y_pred)
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_cv_score': study.best_value,
            'test_score': test_score,
            'study': study,
            'predictions': y_pred
        }
    
    def _optimize_with_grid_search(self, model_name: str, model_config: Dict[str, Any],
                                  X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using GridSearchCV."""
        model = model_config['class'](random_state=42)
        
        grid_search = GridSearchCV(
            model, model_config['params'],
            cv=self.cv_folds,
            scoring=self.scoring_metric,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate
        y_pred = grid_search.predict(X_test)
        test_score = self._calculate_score(y_test, y_pred)
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_score': test_score,
            'grid_search': grid_search,
            'predictions': y_pred
        }
    
    def _optimize_with_random_search(self, model_name: str, model_config: Dict[str, Any],
                                    X_train: pd.DataFrame, y_train: pd.Series,
                                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using RandomizedSearchCV."""
        model = model_config['class'](random_state=42)
        
        random_search = RandomizedSearchCV(
            model, model_config['params'],
            n_iter=self.n_trials,
            cv=self.cv_folds,
            scoring=self.scoring_metric,
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        # Evaluate
        y_pred = random_search.predict(X_test)
        test_score = self._calculate_score(y_test, y_pred)
        
        return {
            'model': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'best_cv_score': random_search.best_score_,
            'test_score': test_score,
            'random_search': random_search,
            'predictions': y_pred
        }
    
    def _create_ensemble_model(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                              y_train: pd.Series, X_test: pd.DataFrame, 
                              y_test: pd.Series) -> Dict[str, Any]:
        """Create ensemble model using top performing models."""
        
        # Get top k models
        model_scores = [(name, result['test_score']) for name, result in models.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = model_scores[:self.top_k_models]
        
        logger.info(f"Creating ensemble with top {len(top_models)} models: {[name for name, _ in top_models]}")
        
        # Prepare models for ensemble
        ensemble_models = []
        for model_name, _ in top_models:
            model = models[model_name]['model']
            ensemble_models.append((model_name, model))
        
        if self.ensemble_method == 'voting':
            ensemble = VotingClassifier(
                estimators=ensemble_models,
                voting='soft' if hasattr(ensemble_models[0][1], 'predict_proba') else 'hard'
            )
        elif self.ensemble_method == 'stacking':
            # Use logistic regression as meta-learner
            meta_learner = LogisticRegression(random_state=42)
            ensemble = StackingClassifier(
                estimators=ensemble_models,
                final_estimator=meta_learner,
                cv=self.cv_folds
            )
        else:  # bagging
            # Use the best model with bagging
            best_model = ensemble_models[0][1]
            ensemble = BaggingClassifier(
                base_estimator=best_model,
                n_estimators=10,
                random_state=42
            )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ensemble.predict(X_test)
        test_score = self._calculate_score(y_test, y_pred)
        
        return {
            'model': ensemble,
            'model_name': f"{self.ensemble_method.capitalize()}Ensemble",
            'test_score': test_score,
            'base_models': [name for name, _ in top_models],
            'predictions': y_pred
        }
    
    def _evaluate_final_models(self, individual_results: Dict[str, Any],
                              ensemble_results: Optional[Dict[str, Any]],
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate all final models comprehensively."""
        
        results = {
            'individual_models': {},
            'ensemble_model': None,
            'best_overall': None,
            'best_overall_score': 0
        }
        
        # Evaluate individual models
        for model_name, model_result in individual_results['models'].items():
            y_pred = model_result['predictions']
            results['individual_models'][model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'test_score': model_result['test_score']
            }
            
            # Update best overall
            if model_result['test_score'] > results['best_overall_score']:
                results['best_overall_score'] = model_result['test_score']
                results['best_overall'] = model_name
        
        # Evaluate ensemble if available
        if ensemble_results:
            y_pred = ensemble_results['predictions']
            results['ensemble_model'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'test_score': ensemble_results['test_score']
            }
            
            # Check if ensemble is better
            if ensemble_results['test_score'] > results['best_overall_score']:
                results['best_overall_score'] = ensemble_results['test_score']
                results['best_overall'] = ensemble_results['model_name']
        
        return results
    
    def _analyze_feature_importance(self, best_model: Any, X_train: pd.DataFrame, 
                                  y_train: pd.Series) -> Dict[str, Any]:
        """Analyze feature importance of the best model."""
        
        importance_info = {}
        
        try:
            # Get feature importance if available
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                feature_names = X_train.columns
                
                # Sort by importance
                importance_pairs = list(zip(feature_names, importances))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                importance_info = {
                    'feature_importances': dict(importance_pairs),
                    'top_features': [name for name, _ in importance_pairs[:10]],
                    'importance_scores': [score for _, score in importance_pairs[:10]]
                }
                
            elif hasattr(best_model, 'coef_'):
                # For linear models
                coefs = best_model.coef_[0] if best_model.coef_.ndim > 1 else best_model.coef_
                feature_names = X_train.columns
                
                importance_pairs = list(zip(feature_names, abs(coefs)))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                importance_info = {
                    'feature_coefficients': dict(importance_pairs),
                    'top_features': [name for name, _ in importance_pairs[:10]],
                    'coefficient_scores': [score for _, score in importance_pairs[:10]]
                }
                
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            importance_info = {'error': str(e)}
        
        return importance_info
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate score based on the configured metric."""
        if self.scoring_metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.scoring_metric == 'precision':
            return precision_score(y_true, y_pred, average='weighted')
        elif self.scoring_metric == 'recall':
            return recall_score(y_true, y_pred, average='weighted')
        elif self.scoring_metric == 'f1':
            return f1_score(y_true, y_pred, average='weighted')
        else:
            return accuracy_score(y_true, y_pred)  # default
    
    def get_best_model(self) -> Any:
        """Get the best performing model."""
        if self.optimization_results.get('ensemble_model'):
            return self.optimization_results['ensemble_model']['model']
        elif self.optimization_results.get('individual_models', {}).get('best_model'):
            return self.optimization_results['individual_models']['best_model']
        else:
            return None
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        return self.optimization_results.get('optimization_summary', {})
    
    def save_optimization_results(self, filepath: str) -> None:
        """Save optimization results to file."""
        try:
            # Convert results to serializable format
            serializable_results = self._make_serializable(self.optimization_results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj 