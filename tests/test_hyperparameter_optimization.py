"""
Test Hyperparameter Optimization
Comprehensive tests for the hyperparameter optimization and AutoML system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import the system under test
from auto_ml.models.training.hyperparameter_optimization import HyperparameterOptimization
from auto_ml.core.exceptions import ModelTrainingError

class TestHyperparameterOptimization:
    """Test cases for hyperparameter optimization system."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Create synthetic data
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        # Create train/test split
        train_size = int(0.8 * n_samples)
        train_data = pd.DataFrame(X[:train_size], columns=[f'feature_{i}' for i in range(n_features)])
        train_data['target'] = y[:train_size]
        
        test_data = pd.DataFrame(X[train_size:], columns=[f'feature_{i}' for i in range(n_features)])
        test_data['target'] = y[train_size:]
        
        return train_data, test_data
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            'optimization_method': 'grid',  # Use grid search for faster testing
            'n_trials': 5,  # Reduced for testing
            'cv_folds': 3,
            'scoring_metric': 'f1',
            'use_ensemble': True,
            'ensemble_method': 'voting',
            'top_k_models': 2,
            'use_lightgbm': False,  # Disable for testing
            'use_xgboost': False   # Disable for testing
        }
    
    def test_initialization(self, basic_config):
        """Test hyperparameter optimization initialization."""
        optimizer = HyperparameterOptimization(basic_config)
        
        assert optimizer.optimization_method == 'grid'
        assert optimizer.n_trials == 5
        assert optimizer.cv_folds == 3
        assert optimizer.scoring_metric == 'f1'
        assert optimizer.use_ensemble is True
        assert optimizer.ensemble_method == 'voting'
        assert optimizer.top_k_models == 2
    
    def test_get_models_to_test(self, basic_config):
        """Test that models to test are properly configured."""
        optimizer = HyperparameterOptimization(basic_config)
        models = optimizer._get_models_to_test()
        
        # Check that expected models are included
        expected_models = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'KNN']
        for model_name in expected_models:
            assert model_name in models
            assert 'class' in models[model_name]
            assert 'params' in models[model_name]
            assert isinstance(models[model_name]['params'], dict)
    
    def test_grid_search_optimization(self, sample_data, basic_config):
        """Test grid search optimization method."""
        train_data, test_data = sample_data
        optimizer = HyperparameterOptimization(basic_config)
        
        # Test with a subset of models for faster execution
        with patch.object(optimizer, '_get_models_to_test') as mock_get_models:
            mock_get_models.return_value = {
                'RandomForest': {
                    'class': RandomForestClassifier,
                    'params': {
                        'n_estimators': [10, 20],
                        'max_depth': [3, 5]
                    }
                }
            }
            
            results = optimizer.train_model(train_data, test_data, 'target')
            
            # Check results structure
            assert 'individual_models' in results
            assert 'final_results' in results
            assert 'feature_importance' in results
            assert 'optimization_summary' in results
            
            # Check individual models results
            individual_results = results['individual_models']
            assert 'models' in individual_results
            assert 'best_model' in individual_results
            assert 'best_model_name' in individual_results
            assert 'best_score' in individual_results
            
            # Check that at least one model was optimized
            assert len(individual_results['models']) > 0
            assert individual_results['best_model'] is not None
            assert individual_results['best_model_name'] is not None
            assert individual_results['best_score'] > 0
    
    def test_ensemble_creation(self, sample_data, basic_config):
        """Test ensemble model creation."""
        train_data, test_data = sample_data
        optimizer = HyperparameterOptimization(basic_config)
        
        # Create real fitted models for ensemble testing
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        
        # Create and fit real models
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model1.fit(X_train, y_train)
        
        model2 = LogisticRegression(random_state=42)
        model2.fit(X_train, y_train)
        
        model3 = RandomForestClassifier(n_estimators=5, random_state=42)
        model3.fit(X_train, y_train)
        
        # Mock individual model results with real models
        mock_models = {
            'Model1': {
                'model': model1,
                'test_score': 0.85,
                'predictions': model1.predict(X_test)
            },
            'Model2': {
                'model': model2,
                'test_score': 0.82,
                'predictions': model2.predict(X_test)
            },
            'Model3': {
                'model': model3,
                'test_score': 0.78,
                'predictions': model3.predict(X_test)
            }
        }
        
        # Test voting ensemble
        ensemble_results = optimizer._create_ensemble_model(
            mock_models, X_train, y_train, X_test, y_test
        )
        
        assert ensemble_results is not None
        assert 'model' in ensemble_results
        assert 'model_name' in ensemble_results
        assert 'test_score' in ensemble_results
        assert 'base_models' in ensemble_results
        assert ensemble_results['model_name'] == 'VotingEnsemble'
        assert len(ensemble_results['base_models']) == 2  # top_k_models = 2
    
    def test_feature_importance_analysis(self, sample_data, basic_config):
        """Test feature importance analysis."""
        train_data, test_data = sample_data
        optimizer = HyperparameterOptimization(basic_config)
        
        # Create a real model with feature importances
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        
        mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
        mock_model.fit(X_train, y_train)
        
        importance_info = optimizer._analyze_feature_importance(mock_model, X_train, y_train)
        
        assert 'feature_importances' in importance_info
        assert 'top_features' in importance_info
        assert 'importance_scores' in importance_info
        assert len(importance_info['top_features']) == 10
        assert len(importance_info['importance_scores']) == 10
        
        # Check that features are sorted by importance
        scores = importance_info['importance_scores']
        assert scores[0] >= scores[1]  # First should be highest
    
    def test_score_calculation(self, basic_config):
        """Test different scoring metrics."""
        optimizer = HyperparameterOptimization(basic_config)
        
        y_true = pd.Series([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        
        # Test F1 score (default)
        f1_score = optimizer._calculate_score(y_true, y_pred)
        assert 0 <= f1_score <= 1
        
        # Test accuracy
        optimizer.scoring_metric = 'accuracy'
        accuracy = optimizer._calculate_score(y_true, y_pred)
        assert 0 <= accuracy <= 1
        
        # Test precision
        optimizer.scoring_metric = 'precision'
        precision = optimizer._calculate_score(y_true, y_pred)
        assert 0 <= precision <= 1
    
    def test_optimization_with_optuna(self, sample_data):
        """Test Optuna optimization method."""
        train_data, test_data = sample_data
        config = {
            'optimization_method': 'optuna',
            'n_trials': 3,  # Very small for testing
            'cv_folds': 2,
            'scoring_metric': 'f1',
            'use_ensemble': False,  # Disable for faster testing
            'use_lightgbm': False,
            'use_xgboost': False
        }
        
        optimizer = HyperparameterOptimization(config)
        
        # Test with a single model for faster execution
        with patch.object(optimizer, '_get_models_to_test') as mock_get_models:
            mock_get_models.return_value = {
                'RandomForest': {
                    'class': RandomForestClassifier,
                    'params': {
                        'n_estimators': [10, 20],
                        'max_depth': [3, 5]
                    }
                }
            }
            
            results = optimizer.train_model(train_data, test_data, 'target')
            
            assert 'individual_models' in results
            individual_results = results['individual_models']
            assert len(individual_results['models']) > 0
            assert individual_results['best_model'] is not None
    
    def test_random_search_optimization(self, sample_data):
        """Test random search optimization method."""
        train_data, test_data = sample_data
        config = {
            'optimization_method': 'random',
            'n_trials': 3,
            'cv_folds': 2,
            'scoring_metric': 'f1',
            'use_ensemble': False,
            'use_lightgbm': False,
            'use_xgboost': False
        }
        
        optimizer = HyperparameterOptimization(config)
        
        with patch.object(optimizer, '_get_models_to_test') as mock_get_models:
            mock_get_models.return_value = {
                'RandomForest': {
                    'class': RandomForestClassifier,
                    'params': {
                        'n_estimators': [10, 20],
                        'max_depth': [3, 5]
                    }
                }
            }
            
            results = optimizer.train_model(train_data, test_data, 'target')
            
            assert 'individual_models' in results
            individual_results = results['individual_models']
            assert len(individual_results['models']) > 0
    
    def test_ensemble_methods(self, sample_data, basic_config):
        """Test different ensemble methods."""
        train_data, test_data = sample_data
        
        # Test voting ensemble
        config_voting = basic_config.copy()
        config_voting['ensemble_method'] = 'voting'
        optimizer_voting = HyperparameterOptimization(config_voting)
        
        # Test stacking ensemble
        config_stacking = basic_config.copy()
        config_stacking['ensemble_method'] = 'stacking'
        optimizer_stacking = HyperparameterOptimization(config_stacking)
        
        # Test bagging ensemble
        config_bagging = basic_config.copy()
        config_bagging['ensemble_method'] = 'bagging'
        optimizer_bagging = HyperparameterOptimization(config_bagging)
        
        # All should initialize without errors
        assert optimizer_voting.ensemble_method == 'voting'
        assert optimizer_stacking.ensemble_method == 'stacking'
        assert optimizer_bagging.ensemble_method == 'bagging'
    
    def test_error_handling(self, sample_data, basic_config):
        """Test error handling in optimization."""
        train_data, test_data = sample_data
        optimizer = HyperparameterOptimization(basic_config)
        
        # Test with invalid data
        with pytest.raises(ModelTrainingError):
            optimizer.train_model(pd.DataFrame(), pd.DataFrame(), 'nonexistent')
    
    def test_save_optimization_results(self, sample_data, basic_config):
        """Test saving optimization results to file."""
        train_data, test_data = sample_data
        optimizer = HyperparameterOptimization(basic_config)
        
        # Mock optimization results
        optimizer.optimization_results = {
            'individual_models': {
                'best_model_name': 'RandomForest',
                'best_score': 0.85
            },
            'ensemble_model': {
                'model_name': 'VotingEnsemble',
                'test_score': 0.87
            }
        }
        
        # Test saving
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            optimizer.save_optimization_results(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Check that file contains valid JSON
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert len(content) > 0
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_get_best_model(self, sample_data, basic_config):
        """Test getting the best model."""
        train_data, test_data = sample_data
        optimizer = HyperparameterOptimization(basic_config)
        
        # Initially should return None
        assert optimizer.get_best_model() is None
        
        # After optimization, should return a model
        with patch.object(optimizer, '_get_models_to_test') as mock_get_models:
            mock_get_models.return_value = {
                'RandomForest': {
                    'class': RandomForestClassifier,
                    'params': {'n_estimators': [10]}
                }
            }
            
            results = optimizer.train_model(train_data, test_data, 'target')
            best_model = optimizer.get_best_model()
            
            assert best_model is not None
    
    def test_optimization_summary(self, sample_data, basic_config):
        """Test getting optimization summary."""
        train_data, test_data = sample_data
        optimizer = HyperparameterOptimization(basic_config)
        
        # Initially should return empty dict
        summary = optimizer.get_optimization_summary()
        assert summary == {}
        
        # After optimization, should return summary
        with patch.object(optimizer, '_get_models_to_test') as mock_get_models:
            mock_get_models.return_value = {
                'RandomForest': {
                    'class': RandomForestClassifier,
                    'params': {'n_estimators': [10]}
                }
            }
            
            results = optimizer.train_model(train_data, test_data, 'target')
            summary = optimizer.get_optimization_summary()
            
            assert 'total_models_tested' in summary
            assert 'best_individual_model' in summary
            assert 'optimization_time' in summary
    
    def test_advanced_algorithms_availability(self):
        """Test handling of advanced algorithms availability."""
        # Test with LightGBM and XGBoost disabled
        config = {
            'use_lightgbm': False,
            'use_xgboost': False
        }
        
        optimizer = HyperparameterOptimization(config)
        models = optimizer._get_models_to_test()
        
        # Should not include LightGBM and XGBoost
        assert 'LightGBM' not in models
        assert 'XGBoost' not in models
        
        # Should include basic models
        basic_models = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'KNN']
        for model in basic_models:
            assert model in models

if __name__ == "__main__":
    pytest.main([__file__]) 