"""
Model Persistence and Loading Module
Handles saving, loading, and versioning of trained models
"""

import pickle
import joblib
import json
import os
import logging
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelPersistence:
    """
    Handles model persistence, loading, and versioning.
    
    This class provides comprehensive model management including:
    - Model serialization and deserialization
    - Model versioning and metadata tracking
    - Feature engineering pipeline persistence
    - Model performance history tracking
    - Deployment-ready model packaging
    
    Attributes:
        models_dir (str): Directory to store saved models
        metadata_file (str): File to store model metadata
        version_format (str): Format for model versioning
    """
    
    def __init__(self, models_dir: str = "models", version_format: str = "v{version}_{timestamp}"):
        """
        Initialize the ModelPersistence class.
        
        Args:
            models_dir (str): Directory to store saved models
            version_format (str): Format string for model versioning
        """
        self.models_dir = Path(models_dir)
        self.metadata_file = self.models_dir / "model_metadata.json"
        self.version_format = version_format
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize metadata if it doesn't exist
        if not self.metadata_file.exists():
            self._initialize_metadata()
    
    def _initialize_metadata(self) -> None:
        """Initialize the model metadata file."""
        initial_metadata = {
            "models": {},
            "latest_versions": {},
            "total_models": 0,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(initial_metadata, f, indent=2)
        
        logger.info(f"Initialized model metadata file: {self.metadata_file}")
    
    def save_model(self, model: Any, model_name: str, 
                  feature_engineering_pipeline: Optional[Any] = None,
                  feature_names: Optional[list] = None,
                  model_metrics: Optional[Dict] = None,
                  model_params: Optional[Dict] = None,
                  description: Optional[str] = None) -> str:
        """
        Save a trained model with metadata.
        
        Args:
            model (Any): Trained model object
            model_name (str): Name of the model
            feature_engineering_pipeline (Optional[Any]): Feature engineering pipeline
            feature_names (Optional[list]): List of feature names
            model_metrics (Optional[Dict]): Model performance metrics
            model_params (Optional[Dict]): Model hyperparameters
            description (Optional[str]): Description of the model
            
        Returns:
            str: Version identifier of the saved model
        """
        try:
            # Generate version
            version = self._get_next_version(model_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_id = self.version_format.format(version=version, timestamp=timestamp)
            
            # Create model directory
            model_dir = self.models_dir / model_name / version_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save feature engineering pipeline if provided
            pipeline_path = None
            if feature_engineering_pipeline is not None:
                pipeline_path = model_dir / "feature_pipeline.pkl"
                with open(pipeline_path, 'wb') as f:
                    pickle.dump(feature_engineering_pipeline, f)
            
            # Create metadata
            metadata = {
                "model_name": model_name,
                "version": version,
                "version_id": version_id,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(model_path),
                "pipeline_path": str(pipeline_path) if pipeline_path else None,
                "feature_names": feature_names,
                "model_metrics": model_metrics,
                "model_params": model_params,
                "description": description,
                "model_type": type(model).__name__,
                "file_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2)
            }
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update global metadata
            self._update_metadata(model_name, version_id, metadata)
            
            logger.info(f"Model {model_name} saved successfully as version {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            raise
    
    def load_model(self, model_name: str, version_id: Optional[str] = None) -> Tuple[Any, Dict]:
        """
        Load a saved model and its metadata.
        
        Args:
            model_name (str): Name of the model to load
            version_id (Optional[str]): Specific version to load, uses latest if None
            
        Returns:
            Tuple[Any, Dict]: Loaded model and metadata
        """
        try:
            # Get version to load
            if version_id is None:
                version_id = self._get_latest_version(model_name)
            
            # Load metadata
            metadata = self._get_model_metadata(model_name, version_id)
            if metadata is None:
                raise ValueError(f"Model {model_name} version {version_id} not found")
            
            # Load model
            model_path = Path(metadata["model_path"])
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Model {model_name} version {version_id} loaded successfully")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def load_feature_pipeline(self, model_name: str, version_id: Optional[str] = None) -> Optional[Any]:
        """
        Load the feature engineering pipeline for a model.
        
        Args:
            model_name (str): Name of the model
            version_id (Optional[str]): Specific version to load, uses latest if None
            
        Returns:
            Optional[Any]: Loaded feature engineering pipeline or None
        """
        try:
            # Get metadata
            if version_id is None:
                version_id = self._get_latest_version(model_name)
            
            metadata = self._get_model_metadata(model_name, version_id)
            if metadata is None:
                raise ValueError(f"Model {model_name} version {version_id} not found")
            
            # Load pipeline if it exists
            pipeline_path = metadata.get("pipeline_path")
            if pipeline_path and Path(pipeline_path).exists():
                with open(pipeline_path, 'rb') as f:
                    pipeline = pickle.load(f)
                logger.info(f"Feature pipeline for {model_name} version {version_id} loaded successfully")
                return pipeline
            else:
                logger.warning(f"No feature pipeline found for {model_name} version {version_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading feature pipeline for {model_name}: {e}")
            return None
    
    def list_models(self) -> Dict[str, Dict]:
        """
        List all saved models and their versions.
        
        Returns:
            Dict[str, Dict]: Dictionary of models and their metadata
        """
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            return metadata.get("models", {})
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {}
    
    def get_model_history(self, model_name: str) -> Dict[str, Dict]:
        """
        Get version history for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Dict]: Version history for the model
        """
        try:
            models = self.list_models()
            if model_name in models:
                return models[model_name]
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting model history for {model_name}: {e}")
            return {}
    
    def delete_model(self, model_name: str, version_id: str) -> bool:
        """
        Delete a specific model version.
        
        Args:
            model_name (str): Name of the model
            version_id (str): Version to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            # Get metadata
            metadata = self._get_model_metadata(model_name, version_id)
            if metadata is None:
                logger.warning(f"Model {model_name} version {version_id} not found")
                return False
            
            # Delete model directory
            model_dir = Path(metadata["model_path"]).parent
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
            
            # Update global metadata
            self._remove_from_metadata(model_name, version_id)
            
            logger.info(f"Model {model_name} version {version_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name} version {version_id}: {e}")
            return False
    
    def _get_next_version(self, model_name: str) -> int:
        """Get the next version number for a model."""
        models = self.list_models()
        if model_name in models:
            # Extract numeric part from version_id (e.g., 'v1_20250614_003024' -> 1)
            def extract_version_num(version_id):
                if version_id.startswith('v'):
                    parts = version_id.split('_')
                    try:
                        return int(parts[0][1:])
                    except Exception:
                        return 0
                try:
                    return int(version_id)
                except Exception:
                    return 0
            return max(extract_version_num(v) for v in models[model_name].keys()) + 1
        else:
            return 1
    
    def _get_latest_version(self, model_name: str) -> str:
        """Get the latest version ID for a model."""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            latest_versions = metadata.get("latest_versions", {})
            if model_name in latest_versions:
                return latest_versions[model_name]
            else:
                raise ValueError(f"No versions found for model {model_name}")
                
        except Exception as e:
            logger.error(f"Error getting latest version for {model_name}: {e}")
            raise
    
    def _get_model_metadata(self, model_name: str, version_id: str) -> Optional[Dict]:
        """Get metadata for a specific model version."""
        try:
            models = self.list_models()
            if model_name in models and version_id in models[model_name]:
                return models[model_name][version_id]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting model metadata: {e}")
            return None
    
    def _update_metadata(self, model_name: str, version_id: str, metadata: Dict) -> None:
        """Update the global metadata file."""
        try:
            with open(self.metadata_file, 'r') as f:
                global_metadata = json.load(f)
            
            # Update models
            if model_name not in global_metadata["models"]:
                global_metadata["models"][model_name] = {}
            
            global_metadata["models"][model_name][version_id] = metadata
            
            # Update latest version
            global_metadata["latest_versions"][model_name] = version_id
            
            # Update counters
            global_metadata["total_models"] = sum(len(versions) for versions in global_metadata["models"].values())
            global_metadata["last_updated"] = datetime.now().isoformat()
            
            # Save updated metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(global_metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            raise
    
    def _remove_from_metadata(self, model_name: str, version_id: str) -> None:
        """Remove a model version from metadata."""
        try:
            with open(self.metadata_file, 'r') as f:
                global_metadata = json.load(f)
            
            # Remove from models
            if model_name in global_metadata["models"]:
                if version_id in global_metadata["models"][model_name]:
                    del global_metadata["models"][model_name][version_id]
                
                # Update latest version if needed
                if not global_metadata["models"][model_name]:
                    del global_metadata["models"][model_name]
                    if model_name in global_metadata["latest_versions"]:
                        del global_metadata["latest_versions"][model_name]
                else:
                    # Find new latest version
                    latest_version = max(global_metadata["models"][model_name].keys())
                    global_metadata["latest_versions"][model_name] = latest_version
            
            # Update counters
            global_metadata["total_models"] = sum(len(versions) for versions in global_metadata["models"].values())
            global_metadata["last_updated"] = datetime.now().isoformat()
            
            # Save updated metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(global_metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error removing from metadata: {e}")
            raise
    
    def export_model_for_deployment(self, model_name: str, version_id: Optional[str] = None,
                                   export_dir: str = "deployment") -> str:
        """
        Export a model for deployment with all necessary components.
        
        Args:
            model_name (str): Name of the model to export
            version_id (Optional[str]): Specific version to export, uses latest if None
            export_dir (str): Directory to export to
            
        Returns:
            str: Path to the exported model package
        """
        try:
            # Load model and metadata
            model, metadata = self.load_model(model_name, version_id)
            
            # Create export directory
            export_path = Path(export_dir)
            export_path.mkdir(exist_ok=True)
            
            # Create deployment package
            package_name = f"{model_name}_{version_id}_deployment"
            package_path = export_path / package_name
            package_path.mkdir(exist_ok=True)
            
            # Copy model files
            model_path = Path(metadata["model_path"])
            deployment_model_path = package_path / "model.pkl"
            with open(model_path, 'rb') as src, open(deployment_model_path, 'wb') as dst:
                dst.write(src.read())
            
            # Copy feature pipeline if exists
            if metadata.get("pipeline_path"):
                pipeline_path = Path(metadata["pipeline_path"])
                deployment_pipeline_path = package_path / "feature_pipeline.pkl"
                with open(pipeline_path, 'rb') as src, open(deployment_pipeline_path, 'wb') as dst:
                    dst.write(src.read())
            
            # Create deployment metadata
            deployment_metadata = {
                "model_name": model_name,
                "version_id": version_id,
                "export_timestamp": datetime.now().isoformat(),
                "feature_names": metadata.get("feature_names"),
                "model_type": metadata.get("model_type"),
                "model_metrics": metadata.get("model_metrics"),
                "model_params": metadata.get("model_params")
            }
            
            deployment_metadata_path = package_path / "deployment_metadata.json"
            with open(deployment_metadata_path, 'w') as f:
                json.dump(deployment_metadata, f, indent=2)
            
            # Create requirements file
            requirements_path = package_path / "requirements.txt"
            with open(requirements_path, 'w') as f:
                f.write("pandas>=1.3.0\n")
                f.write("numpy>=1.21.0\n")
                f.write("scikit-learn>=1.0.0\n")
                f.write("lightgbm>=3.3.0\n")
                f.write("scipy>=1.7.0\n")
            
            logger.info(f"Model {model_name} version {version_id} exported for deployment to {package_path}")
            return str(package_path)
            
        except Exception as e:
            logger.error(f"Error exporting model for deployment: {e}")
            raise

if __name__ == "__main__":
    # Test model persistence
    from model_training import ModelTraining
    from data_ingestion import DataIngestion
    from feature_engineering import FeatureEngineering
    
    # Load and process data
    ingestion = DataIngestion()
    train_data, test_data = ingestion.load_adult_dataset()
    
    fe = FeatureEngineering()
    cat_cols = ingestion.get_categorical_columns()
    num_cols = ingestion.get_numerical_columns()
    
    train_processed = fe.fit_transform(train_data, cat_cols, num_cols)
    test_processed = fe.transform(test_data, cat_cols, num_cols)
    
    # Get feature names
    feature_names = fe.get_feature_names(cat_cols, num_cols)
    
    # Train a model
    mt = ModelTraining(enable_hyperparameter_optimization=False)  # Quick test
    results = mt.train_models(
        train_processed, train_processed['target'],
        test_processed, test_processed['target'],
        feature_names
    )
    
    # Test model persistence
    mp = ModelPersistence()
    
    # Save model
    version_id = mp.save_model(
        model=mt.best_model,
        model_name="test_model",
        feature_engineering_pipeline=fe,
        feature_names=feature_names,
        model_metrics=mt.results[mt.best_model_name],
        model_params=mt.best_model.get_params(),
        description="Test model for persistence functionality"
    )
    
    print(f"Model saved with version: {version_id}")
    
    # List models
    models = mp.list_models()
    print(f"Available models: {list(models.keys())}")
    
    # Load model
    loaded_model, metadata = mp.load_model("test_model", version_id)
    print(f"Model loaded successfully: {type(loaded_model).__name__}")
    
    # Load feature pipeline
    pipeline = mp.load_feature_pipeline("test_model", version_id)
    print(f"Feature pipeline loaded: {pipeline is not None}")
    
    print("Model persistence test completed successfully!") 