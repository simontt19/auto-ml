"""
Main ML Pipeline Orchestration
Coordinates the entire ML workflow from data ingestion to model deployment
"""

import logging
import argparse
import time
from datetime import datetime
from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from model_persistence import ModelPersistence

def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set log level for all handlers
    for handler in logging.root.handlers:
        handler.setLevel(numeric_level)

def main(data_path: str = None, log_level: str = "INFO", 
         enable_hyperparameter_optimization: bool = True,
         save_model: bool = True) -> None:
    """
    Execute the complete ML pipeline.
    
    Args:
        data_path (str): Path to the dataset (optional, uses default if None)
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_hyperparameter_optimization (bool): Whether to enable hyperparameter optimization
        save_model (bool): Whether to save the trained model
    """
    logger = logging.getLogger(__name__)
    
    # Set up logging
    setup_logging(log_level)
    
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("STARTING FULL ML PIPELINE")
    logger.info("=" * 60)
    logger.info("")
    
    try:
        # Step 1: Data Ingestion
        logger.info("==================== STEP 1: DATA INGESTION ====================")
        ingestion = DataIngestion()
        train_data, test_data = ingestion.load_adult_dataset()
        
        # Step 2: Feature Engineering
        logger.info("==================== STEP 2: FEATURE ENGINEERING ====================")
        fe = FeatureEngineering()
        cat_cols = ingestion.get_categorical_columns()
        num_cols = ingestion.get_numerical_columns()
        
        train_processed = fe.fit_transform(train_data, cat_cols, num_cols)
        test_processed = fe.transform(test_data, cat_cols, num_cols)
        
        # Get feature names
        feature_names = fe.get_feature_names(cat_cols, num_cols)
        
        # Step 3: Model Training & Evaluation
        logger.info("==================== STEP 3: MODEL TRAINING & EVALUATION ====================")
        mt = ModelTraining(enable_hyperparameter_optimization=enable_hyperparameter_optimization)
        results = mt.train_models(
            train_processed, train_processed['target'],
            test_processed, test_processed['target'],
            feature_names
        )
        
        # Step 4: Cross-validation
        logger.info("==================== STEP 4: CROSS-VALIDATION ====================")
        cv_results = mt.cross_validate_best_model(
            train_processed, train_processed['target'], feature_names
        )
        
        # Step 5: Feature Importance
        logger.info("==================== STEP 5: FEATURE IMPORTANCE ====================")
        feature_importance = mt.get_feature_importance(feature_names)
        if feature_importance is not None:
            logger.info("Top 10 most important features:")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Step 6: Final Model Training
        logger.info("==================== STEP 6: FINAL MODEL TRAINING ====================")
        final_model = mt.train_final_model(
            train_processed, train_processed['target'], feature_names
        )
        
        # Step 7: Model Persistence (if enabled)
        if save_model:
            logger.info("==================== STEP 7: MODEL PERSISTENCE ====================")
            mp = ModelPersistence()
            
            # Save the best model with all metadata
            version_id = mp.save_model(
                model=final_model,
                model_name=f"adult_income_{mt.best_model_name}",
                feature_engineering_pipeline=fe,
                feature_names=feature_names,
                model_metrics=mt.results[mt.best_model_name],
                model_params=final_model.get_params(),
                description=f"Best performing {mt.best_model_name} model for Adult Income prediction with hyperparameter optimization"
            )
            
            logger.info(f"Model saved successfully with version: {version_id}")
            
            # List all saved models
            models = mp.list_models()
            logger.info(f"Total saved models: {len(models)}")
        
        # Step 8: Save Results
        logger.info("==================== STEP 8: SAVING RESULTS ====================")
        results_file = mt.save_results()
        
        # Pipeline Summary
        logger.info("==================== PIPELINE SUMMARY ====================")
        logger.info("Data processed:")
        logger.info(f"  - Training samples: {len(train_processed)}")
        logger.info(f"  - Test samples: {len(test_processed)}")
        logger.info(f"  - Total features: {len(feature_names)}")
        logger.info("")
        
        logger.info("Model Performance:")
        for model_name, metrics in results.items():
            logger.info(f"  {model_name}:")
            logger.info(f"    - AUC: {metrics['auc']:.4f}")
            logger.info(f"    - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"    - F1-Score: {metrics['f1']:.4f}")
        
        logger.info("")
        logger.info(f"Best Model: {mt.best_model_name}")
        logger.info(f"  - AUC: {results[mt.best_model_name]['auc']:.4f}")
        logger.info(f"  - Cross-validation AUC: {cv_results['mean_auc']:.4f} (+/- {cv_results['std_auc']:.4f})")
        
        if feature_importance is not None:
            logger.info("")
            logger.info("Top 5 Most Important Features:")
            for idx, row in feature_importance.head(5).iterrows():
                logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Calculate and log duration
        duration = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Total duration: {time.strftime('%H:%M:%S', time.gmtime(duration))}")
        logger.info("=" * 60)
        
        print("\n🎉 Pipeline completed successfully!")
        print("Check 'pipeline.log' for detailed logs and results.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline for Adult Income Prediction")
    parser.add_argument("--data-path", type=str, help="Path to the dataset")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--no-hyperopt", action="store_true",
                       help="Disable hyperparameter optimization")
    parser.add_argument("--no-save", action="store_true",
                       help="Disable model saving")
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        log_level=args.log_level,
        enable_hyperparameter_optimization=not args.no_hyperopt,
        save_model=not args.no_save
    ) 