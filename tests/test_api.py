#!/usr/bin/env python3
"""
Test API Endpoints
Comprehensive testing of the model serving API.
"""

import logging
import sys
import asyncio
import requests
import json
from pathlib import Path
import tempfile
import time

# Add the project root to the Python path (go up one level from tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_ml import (
    Config,
    AdultIncomeDataIngestion,
    StandardFeatureEngineering,
    ClassificationModelTraining,
    ModelPersistence,
    ModelAPI
)

def test_api():
    """Test the production Model API."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing production Model API...")
    
    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Using temporary directory: {temp_dir}")
            
            # 1. Train and save a model
            logger.info("1. Training and saving a model...")
            config = Config('config.yaml')
            ingestion = AdultIncomeDataIngestion(config.config)
            fe = StandardFeatureEngineering(config.config)
            mt = ClassificationModelTraining(config.config)
            mp = ModelPersistence(models_dir=f"{temp_dir}/models")
            
            # Load and process data
            train_data, test_data = ingestion.load_data()
            cat_cols = ingestion.get_categorical_columns()
            num_cols = ingestion.get_numerical_columns()
            
            train_processed = fe.fit_transform(train_data, cat_cols, num_cols)
            test_processed = fe.transform(test_data, cat_cols, num_cols)
            
            # Use small subset for quick testing
            train_subset = train_processed.head(1000)
            test_subset = test_processed.head(500)
            feature_names = fe.get_feature_names(cat_cols, num_cols)
            
            # Train models
            results = mt.train_models(
                train_subset, train_subset['target'],
                test_subset, test_subset['target'],
                feature_names
            )
            
            # Save the best model
            model_config = {
                'model_type': 'classification',
                'algorithms': ['lightgbm'],
                'hyperparameter_optimization': True
            }
            
            version = mp.save_model(
                model=mt.best_model,
                model_name="adult_income_classifier",
                model_type="classification",
                training_results=results[mt.best_model_name],
                feature_names=feature_names,
                model_config=model_config
            )
            logger.info(f"Model saved with version: {version}")
            
            # 2. Start the API server
            logger.info("2. Starting API server...")
            api = ModelAPI(models_dir=f"{temp_dir}/models", host="127.0.0.1", port=8001)
            
            # Start server in background
            import threading
            server_thread = threading.Thread(target=api.run, daemon=True)
            server_thread.start()
            
            # Wait for server to start
            time.sleep(3)
            
            # 3. Test API endpoints
            base_url = "http://127.0.0.1:8001"
            
            # Test health endpoint
            logger.info("3. Testing health endpoint...")
            response = requests.get(f"{base_url}/health")
            assert response.status_code == 200
            health_data = response.json()
            logger.info(f"Health check: {health_data}")
            
            # Test models endpoint
            logger.info("4. Testing models endpoint...")
            response = requests.get(f"{base_url}/models")
            assert response.status_code == 200
            models_data = response.json()
            logger.info(f"Available models: {list(models_data.keys())}")
            
            # Test model info endpoint
            logger.info("5. Testing model info endpoint...")
            response = requests.get(f"{base_url}/models/adult_income_classifier")
            assert response.status_code == 200
            model_info = response.json()
            logger.info(f"Model info: {model_info['model_name']} v{model_info['version']}")
            
            # 6. Test prediction endpoint
            logger.info("6. Testing prediction endpoint...")
            
            # Create sample features
            sample_features = {
                'age': 35,
                'workclass': 2,
                'fnlwgt': 200000,
                'education': 9,
                'education-num': 13,
                'marital-status': 2,
                'occupation': 3,
                'relationship': 1,
                'race': 4,
                'sex': 1,
                'capital-gain': 0,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 39,
                'age_binned': 2,
                'work_hours_binned': 1,
                'education_level': 3,
                'capital_net': 0,
                'income_ratio': 0.0
            }
            
            prediction_request = {
                "features": sample_features,
                "model_name": "adult_income_classifier"
            }
            
            response = requests.post(f"{base_url}/predict", json=prediction_request)
            assert response.status_code == 200
            prediction_data = response.json()
            logger.info(f"Prediction: {prediction_data['prediction']}, Probability: {prediction_data['probability']:.4f}")
            
            # 7. Test model loading/unloading
            logger.info("7. Testing model loading/unloading...")
            
            # Unload model
            response = requests.delete(f"{base_url}/models/adult_income_classifier/unload")
            assert response.status_code == 200
            logger.info("Model unloaded successfully")
            
            # Load model
            response = requests.post(f"{base_url}/models/adult_income_classifier/load")
            assert response.status_code == 200
            logger.info("Model loaded successfully")
            
            # Test prediction again
            response = requests.post(f"{base_url}/predict", json=prediction_request)
            assert response.status_code == 200
            logger.info("Prediction after reload successful")
            
            logger.info("✅ All API tests passed!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1) 