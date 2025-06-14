#!/usr/bin/env python3
"""
Test New Framework
Basic testing of the new framework structure.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path (go up one level from tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_ml import (
    ConfigManager,
    AdultIncomeDataIngestion,
    StandardFeatureEngineering,
    ClassificationModelTraining
)

def test_new_framework():
    """Test the new framework structure with Adult Income dataset."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing new Auto ML framework structure...")
    
    try:
        # 1. Test configuration
        logger.info("1. Testing configuration...")
        config_manager = ConfigManager('configs')
        config = config_manager.load_settings()
        logger.info(f"Configuration loaded successfully. Task type: {config.get('model', {}).get('task_type', 'classification')}")
        
        # 2. Test data ingestion
        logger.info("2. Testing data ingestion...")
        ingestion = AdultIncomeDataIngestion(config)
        train_data, test_data = ingestion.load_data()
        logger.info(f"Data loaded: Train {train_data.shape}, Test {test_data.shape}")
        
        # 3. Test feature engineering
        logger.info("3. Testing feature engineering...")
        fe = StandardFeatureEngineering(config)
        cat_cols = ingestion.get_categorical_columns()
        num_cols = ingestion.get_numerical_columns()
        
        train_processed = fe.fit_transform(train_data, cat_cols, num_cols)
        test_processed = fe.transform(test_data, cat_cols, num_cols)
        logger.info(f"Feature engineering completed: Train {train_processed.shape}, Test {test_processed.shape}")
        
        # 4. Test model training
        logger.info("4. Testing model training...")
        mt = ClassificationModelTraining(config)
        feature_names = fe.get_feature_names(cat_cols, num_cols)
        
        # Use a small subset for quick testing
        train_subset = train_processed.head(1000)
        test_subset = test_processed.head(500)
        
        results = mt.train_models(
            train_subset, train_subset['target'],
            test_subset, test_subset['target'],
            feature_names
        )
        logger.info(f"Model training completed. Best model: {mt.best_model_name}")
        
        # 5. Test feature importance
        logger.info("5. Testing feature importance...")
        feature_importance = mt.get_feature_importance(feature_names)
        if feature_importance is not None:
            logger.info(f"Top 3 features: {feature_importance.head(3)['feature'].tolist()}")
        
        logger.info("✅ All tests passed! New framework structure is working correctly.")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise the exception for pytest to catch

if __name__ == "__main__":
    test_new_framework() 