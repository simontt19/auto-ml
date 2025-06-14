#!/usr/bin/env python3
"""
Test Model Registry System
Comprehensive testing of the enterprise-grade model registry with metadata tracking, lineage, and governance.
"""

import logging
import sys
import tempfile
import json
from pathlib import Path
import shutil
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_ml import (
    ConfigManager,
    AdultIncomeDataIngestion,
    StandardFeatureEngineering,
    ClassificationModelTraining,
    ModelPersistence,
    ModelRegistry,
    ModelMetadata,
    ModelStatus,
    ModelStage
)
from auto_ml.core.user_management import UserManager, User, UserRole

def test_model_registry():
    """Test the comprehensive model registry system."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing comprehensive Model Registry system...")
    
    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Initialize model registry
            logger.info("1. Initializing model registry...")
            registry_path = temp_path / "registry"
            models_dir = temp_path / "models"
            
            registry = ModelRegistry(str(registry_path), str(models_dir))
            logger.info(f"Registry initialized at {registry_path}")
            
            # 2. Set up user management
            logger.info("2. Setting up user management...")
            user_manager = UserManager()
            
            # Create test user
            user_manager.create_user(
                username="testuser",
                email="test@example.com",
                role=UserRole.USER
            )
            
            # Create test project
            test_project = user_manager.create_project(
                name="Test Project",
                owner="testuser",
                description="Test project for registry"
            )
            logger.info(f"Created test project: {test_project}")
            
            # 3. Train a model for testing
            logger.info("3. Training a model for testing...")
            
            # Load configuration
            config_manager = ConfigManager('configs')
            config = config_manager.load_settings()
            
            # Load data
            ingestion = AdultIncomeDataIngestion(config)
            train_data, test_data = ingestion.load_data()
            
            # Feature engineering
            fe = StandardFeatureEngineering(config)
            cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']
            num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                       'capital-loss', 'hours-per-week']
            
            train_processed = fe.fit_transform(train_data, cat_cols, num_cols)
            test_processed = fe.transform(test_data)
            
            # Get feature names
            feature_names = list(train_processed.columns)
            feature_names.remove('target')
            
            # Model training
            mt = ClassificationModelTraining(config)
            results = mt.train_models(
                train_processed, train_processed['target'],
                test_processed, test_processed['target'],
                feature_names
            )
            
            # Save model using persistence
            mp = ModelPersistence(str(models_dir))
            version = mp.save_model(
                model=mt.best_model,
                model_name="test_classifier",
                model_type="classification",
                training_results=results[mt.best_model_name],
                feature_names=feature_names,
                model_config=config
            )
            logger.info(f"Model saved with version: {version}")

            # Find the actual model file path
            model_dir = models_dir / "test_classifier" / version
            model_path = model_dir / "model.pkl"
            
            # 4. Register model in registry
            logger.info("4. Registering model in registry...")
            
            model_id = registry.register_model(
                model_name="test_classifier",
                model_type="classification",
                owner="testuser",
                project_id=test_project,
                model_path=str(model_path),
                feature_names=feature_names,
                target_column="target",
                training_metrics=results[mt.best_model_name],
                hyperparameters=mt.best_model.get_params(),
                description="Test model for registry validation",
                framework="scikit-learn",
                algorithm="lightgbm",
                tags=["test", "classification", "income"]
            )
            
            logger.info(f"Model registered with ID: {model_id}")
            
            # 5. Test model retrieval
            logger.info("5. Testing model retrieval...")
            
            model = registry.get_model(model_id)
            print('DEBUG: model.status =', model.status, type(model.status))
            assert model is not None, "Model should be retrievable"
            assert model.model_name == "test_classifier", "Model name should match"
            assert model.owner == "testuser", "Owner should match"
            assert model.status == ModelStatus.TRAINED, "Status should be TRAINED"
            assert model.stage == ModelStage.DEVELOPMENT, "Stage should be DEVELOPMENT"
            
            logger.info(f"Model retrieved successfully: {model.model_name} v{model.version}")
            
            # 6. Test model listing with filters
            logger.info("6. Testing model listing with filters...")
            
            # List all models
            all_models = registry.list_models(limit=100)
            assert len(all_models) >= 1, "Should have at least one model"
            
            # Filter by owner
            user_models = registry.list_models(owner="testuser")
            assert len(user_models) >= 1, "Should find user's models"
            
            # Filter by project
            project_models = registry.list_models(project_id=test_project)
            assert len(project_models) >= 1, "Should find project models"
            
            # Filter by status
            trained_models = registry.list_models(status=ModelStatus.TRAINED)
            assert len(trained_models) >= 1, "Should find trained models"
            
            # Filter by stage
            dev_models = registry.list_models(stage=ModelStage.DEVELOPMENT)
            assert len(dev_models) >= 1, "Should find development models"
            
            # Filter by tags
            tagged_models = registry.list_models(tags=["test"])
            assert len(tagged_models) >= 1, "Should find tagged models"
            
            logger.info("Model listing and filtering tests passed")
            
            # 7. Test model status updates
            logger.info("7. Testing model status updates...")
            
            success = registry.update_model_status(
                model_id=model_id,
                status=ModelStatus.VALIDATED,
                updated_by="testuser",
                notes="Model validated through testing"
            )
            assert success, "Status update should succeed"
            
            # Verify status change
            updated_model = registry.get_model(model_id)
            assert updated_model.status == ModelStatus.VALIDATED, "Status should be updated"
            
            logger.info("Model status update test passed")
            
            # 8. Test model promotion
            logger.info("8. Testing model promotion...")
            
            success = registry.promote_model(
                model_id=model_id,
                target_stage=ModelStage.STAGING,
                approved_by="testuser",
                notes="Promoted to staging for testing"
            )
            assert success, "Model promotion should succeed"
            
            # Verify stage change
            promoted_model = registry.get_model(model_id)
            assert promoted_model.stage == ModelStage.STAGING, "Stage should be updated"
            assert promoted_model.approved_by == "testuser", "Approver should be set"
            
            logger.info("Model promotion test passed")
            
            # 9. Test performance recording
            logger.info("9. Testing performance recording...")
            
            # Record performance metrics
            performance_metrics = {
                "accuracy": 0.85,
                "precision": 0.80,
                "recall": 0.75,
                "f1": 0.77,
                "auc": 0.90
            }
            
            success = registry.record_performance(
                model_id=model_id,
                metrics=performance_metrics,
                data_version="v1.0",
                environment="staging",
                drift_score=0.02
            )
            assert success, "Performance recording should succeed"
            
            logger.info("Performance recording test passed")
            
            # 10. Test model lineage
            logger.info("10. Testing model lineage...")
            
            # Create a second model with lineage
            model_id_2 = registry.register_model(
                model_name="test_classifier_v2",
                model_type="classification",
                owner="testuser",
                project_id=test_project,
                model_path=str(model_path),  # Same model file for testing
                feature_names=feature_names,
                target_column="target",
                training_metrics=results[mt.best_model_name],
                hyperparameters=mt.best_model.get_params(),
                description="Second version of test model",
                framework="scikit-learn",
                algorithm="lightgbm",
                parent_model_id=model_id,
                tags=["test", "classification", "income", "v2"]
            )
            
            # Get lineage information
            lineage = registry.get_model_lineage(model_id_2)
            assert lineage is not None, "Lineage should exist"
            assert lineage.parent_model_id == model_id, "Parent model ID should match"
            
            logger.info("Model lineage test passed")
            
            # 11. Test model search
            logger.info("11. Testing model search...")
            
            # Search by name
            search_results = registry.search_models("test_classifier")
            assert len(search_results) >= 1, "Should find models by name"
            
            # Search by description
            search_results = registry.search_models("test model")
            assert len(search_results) >= 1, "Should find models by description"
            
            # Search by tags
            search_results = registry.search_models("classification")
            assert len(search_results) >= 1, "Should find models by tags"
            
            logger.info("Model search test passed")
            
            # 12. Test registry export
            logger.info("12. Testing registry export...")
            
            export_path = temp_path / "registry_export.json"
            success = registry.export_registry(str(export_path))
            assert success, "Registry export should succeed"
            assert export_path.exists(), "Export file should exist"
            
            # Verify export content
            with open(export_path, 'r') as f:
                export_data = json.load(f)
            
            assert "models" in export_data, "Export should contain models"
            assert "total_models" in export_data, "Export should contain total count"
            assert export_data["total_models"] >= 2, "Should have at least 2 models"
            
            logger.info("Registry export test passed")
            
            # 13. Test comprehensive model metadata
            logger.info("13. Testing comprehensive model metadata...")
            
            model = registry.get_model(model_id)
            
            # Verify all metadata fields
            assert model.model_id == model_id, "Model ID should match"
            assert model.model_name == "test_classifier", "Model name should match"
            assert model.model_type == "classification", "Model type should match"
            assert model.owner == "testuser", "Owner should match"
            assert model.project_id == test_project, "Project ID should match"
            assert model.framework == "scikit-learn", "Framework should match"
            assert model.algorithm == "lightgbm", "Algorithm should match"
            assert len(model.feature_names) > 0, "Feature names should be populated"
            assert model.target_column == "target", "Target column should match"
            assert len(model.training_metrics) > 0, "Training metrics should be populated"
            assert len(model.hyperparameters) > 0, "Hyperparameters should be populated"
            assert model.model_path == str(model_path), "Model path should match"
            assert model.model_size_mb > 0, "Model size should be positive"
            assert len(model.model_hash) > 0, "Model hash should be populated"
            assert model.created_at is not None, "Created timestamp should be set"
            assert model.updated_at is not None, "Updated timestamp should be set"
            assert "test" in model.tags, "Tags should contain 'test'"
            
            logger.info("Comprehensive metadata test passed")
            
            # 14. Test audit logging
            logger.info("14. Testing audit logging...")
            
            # The registry should have logged various events during our tests
            # We can verify this by checking that the model has been updated multiple times
            model = registry.get_model(model_id)
            assert model.updated_at != model.created_at, "Model should have been updated"
            
            logger.info("Audit logging test passed")
            
            logger.info("✅ All model registry tests passed!")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise the exception for pytest to catch

if __name__ == "__main__":
    test_model_registry() 