#!/usr/bin/env python3
"""
Test Model Persistence
Comprehensive testing of model saving, loading, and versioning.
"""

import logging
import sys
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path (go up one level from tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_ml import (
    ConfigManager,
    AdultIncomeDataIngestion,
    StandardFeatureEngineering,
    ClassificationModelTraining,
    ModelPersistence
)

def test_model_persistence():
    """Test the model persistence and versioning system."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing model persistence and versioning system...")
    
    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Using temporary directory: {temp_dir}")
            
            # 1. Initialize components
            logger.info("1. Initializing components...")
            config_manager = ConfigManager('configs')
            config = config_manager.load_settings()
            ingestion = AdultIncomeDataIngestion(config)
            fe = StandardFeatureEngineering(config)
            mt = ClassificationModelTraining(config)
            mp = ModelPersistence(models_dir=f"{temp_dir}/models")
            
            # 2. Load and process data
            logger.info("2. Loading and processing data...")
            train_data, test_data = ingestion.load_data()
            cat_cols = ingestion.get_categorical_columns()
            num_cols = ingestion.get_numerical_columns()
            
            train_processed = fe.fit_transform(train_data, cat_cols, num_cols)
            test_processed = fe.transform(test_data, cat_cols, num_cols)
            
            # Use small subset for quick testing
            train_subset = train_processed.head(1000)
            test_subset = test_processed.head(500)
            feature_names = fe.get_feature_names(cat_cols, num_cols)
            
            # 3. Train models
            logger.info("3. Training models...")
            results = mt.train_models(
                train_subset, train_subset['target'],
                test_subset, test_subset['target'],
                feature_names
            )
            
            # 4. Test model saving
            logger.info("4. Testing model saving...")
            model_config = {
                'model_type': 'classification',
                'algorithms': ['lightgbm'],
                'hyperparameter_optimization': True
            }
            
            # Save the best model
            version1 = mp.save_model(
                model=mt.best_model,
                model_name="adult_income_classifier",
                model_type="classification",
                training_results=results[mt.best_model_name],
                feature_names=feature_names,
                model_config=model_config
            )
            logger.info(f"Model saved with version: {version1}")
            
            # 5. Test model loading
            logger.info("5. Testing model loading...")
            loaded_model, metadata = mp.load_model("adult_income_classifier", version1)
            logger.info(f"Model loaded successfully. Version: {metadata['version']}")
            logger.info(f"Model size: {metadata['model_size_mb']:.2f} MB")
            
            # 6. Test model listing
            logger.info("6. Testing model listing...")
            models = mp.list_models()
            logger.info(f"Available models: {list(models.keys())}")
            
            # 7. Test model info
            logger.info("7. Testing model info...")
            model_info = mp.get_model_info("adult_income_classifier", version1)
            logger.info(f"Model info retrieved. AUC: {model_info['training_results']['auc']:.4f}")
            
            # 8. Test model export for deployment
            logger.info("8. Testing model export for deployment...")
            export_path = mp.export_model_for_deployment(
                "adult_income_classifier", 
                version1, 
                f"{temp_dir}/exports"
            )
            logger.info(f"Model exported to: {export_path}")
            
            # Verify export files
            export_files = list(Path(export_path).glob("*"))
            logger.info(f"Export files: {[f.name for f in export_files]}")
            
            # 9. Test saving multiple versions
            logger.info("9. Testing multiple version saving...")
            
            # Train a second model with different config
            mt2 = ClassificationModelTraining(config)
            results2 = mt2.train_models(
                train_subset, train_subset['target'],
                test_subset, test_subset['target'],
                feature_names
            )
            
            version2 = mp.save_model(
                model=mt2.best_model,
                model_name="adult_income_classifier",
                model_type="classification",
                training_results=results2[mt2.best_model_name],
                feature_names=feature_names,
                model_config=model_config
            )
            logger.info(f"Second model saved with version: {version2}")
            
            # 10. Test loading latest version
            logger.info("10. Testing loading latest version...")
            latest_model, latest_metadata = mp.load_model("adult_income_classifier")
            logger.info(f"Latest model loaded: {latest_metadata['version']}")
            
            # 11. Test model deletion
            logger.info("11. Testing model deletion...")
            mp.delete_model("adult_income_classifier", version1)
            logger.info(f"Model version {version1} deleted successfully")
            
            # Verify deletion
            remaining_models = mp.list_models()
            remaining_versions = remaining_models.get("adult_income_classifier", [])
            logger.info(f"Remaining versions: {[v['version'] for v in remaining_versions]}")
            
            logger.info("✅ All model persistence tests passed!")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise the exception for pytest to catch

if __name__ == "__main__":
    test_model_persistence() 