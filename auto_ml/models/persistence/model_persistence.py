"""
Model Persistence and Versioning System
Comprehensive model management for production deployment.
"""

import os
import json
import pickle
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import shutil
import hashlib
import yaml

from ...core.base_classes import BaseModelTraining
from ...core.exceptions import ModelPersistenceError

logger = logging.getLogger(__name__)

class ModelPersistence:
    """
    Model persistence and versioning system for production deployment.
    
    This class provides comprehensive model management including:
    - Model saving and loading with metadata
    - Version control and model registry
    - Model performance tracking
    - Deployment-ready model packaging
    - Model rollback capabilities
    
    Attributes:
        models_dir (str): Directory for storing models
        metadata_file (str): File for storing model metadata
        version_format (str): Format for version naming
        max_versions (int): Maximum number of versions to keep
    """
    
    def __init__(self, models_dir: str = "models", max_versions: int = 10):
        """
        Initialize the ModelPersistence system.
        
        Args:
            models_dir (str): Directory for storing models
            max_versions (int): Maximum number of versions to keep
        """
        self.models_dir = Path(models_dir)
        self.max_versions = max_versions
        self.metadata_file = self.models_dir / "model_registry.json"
        self.version_format = "v{version}_{timestamp}"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load model registry
        self.model_registry = self._load_model_registry()
    
    def save_model(self, model: Any, model_name: str, model_type: str,
                  training_results: Dict[str, Any], feature_names: List[str],
                  model_config: Dict[str, Any], version: Optional[str] = None) -> str:
        """
        Save a trained model with metadata and versioning.
        
        Args:
            model (Any): Trained model object
            model_name (str): Name of the model
            model_type (str): Type of model (e.g., 'classification', 'regression')
            training_results (Dict[str, Any]): Training performance results
            feature_names (List[str]): List of feature names used for training
            model_config (Dict[str, Any]): Model configuration
            version (Optional[str]): Specific version to save, auto-generated if None
            
        Returns:
            str: Version identifier of the saved model
            
        Raises:
            ModelPersistenceError: If saving fails
        """
        try:
            # Generate version if not provided
            if version is None:
                version = self._generate_version(model_name)
            
            # Create version directory
            version_dir = self.models_dir / model_name / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model file
            model_file = version_dir / "model.pkl"
            self._save_model_file(model, model_file)
            
            # Create metadata
            metadata = {
                'model_name': model_name,
                'model_type': model_type,
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'training_results': training_results,
                'feature_names': feature_names,
                'model_config': model_config,
                'model_file': str(model_file),
                'model_hash': self._calculate_model_hash(model_file),
                'model_size_mb': self._get_file_size_mb(model_file)
            }
            
            # Save metadata
            metadata_file = version_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update model registry
            self._update_model_registry(model_name, version, metadata)
            
            # Clean up old versions
            self._cleanup_old_versions(model_name)
            
            logger.info(f"Model {model_name} version {version} saved successfully")
            return version
            
        except Exception as e:
            raise ModelPersistenceError(f"Failed to save model {model_name}: {e}")
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model and its metadata.
        
        Args:
            model_name (str): Name of the model to load
            version (Optional[str]): Specific version to load, latest if None
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Model object and metadata
            
        Raises:
            ModelPersistenceError: If loading fails
        """
        try:
            # Get version to load
            if version is None:
                version = self._get_latest_version(model_name)
            
            # Load metadata
            metadata = self._load_model_metadata(model_name, version)
            
            # Load model
            model_file = Path(metadata['model_file'])
            model = self._load_model_file(model_file)
            
            # Verify model hash
            current_hash = self._calculate_model_hash(model_file)
            if current_hash != metadata['model_hash']:
                raise ModelPersistenceError(f"Model hash mismatch for {model_name} version {version}")
            
            logger.info(f"Model {model_name} version {version} loaded successfully")
            return model, metadata
            
        except Exception as e:
            raise ModelPersistenceError(f"Failed to load model {model_name}: {e}")
    
    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all available models and their versions.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary of models and their versions
        """
        return self.model_registry
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific model version.
        
        Args:
            model_name (str): Name of the model
            version (Optional[str]): Specific version, latest if None
            
        Returns:
            Dict[str, Any]: Model information and metadata
        """
        try:
            if version is None:
                version = self._get_latest_version(model_name)
            
            metadata = self._load_model_metadata(model_name, version)
            return metadata
            
        except Exception as e:
            raise ModelPersistenceError(f"Failed to get model info for {model_name}: {e}")
    
    def delete_model(self, model_name: str, version: str) -> None:
        """
        Delete a specific model version.
        
        Args:
            model_name (str): Name of the model
            version (str): Version to delete
            
        Raises:
            ModelPersistenceError: If deletion fails
        """
        try:
            version_dir = self.models_dir / model_name / version
            
            if not version_dir.exists():
                raise ModelPersistenceError(f"Model {model_name} version {version} not found")
            
            # Remove version directory
            shutil.rmtree(version_dir)
            
            # Update registry
            if model_name in self.model_registry:
                self.model_registry[model_name] = [
                    v for v in self.model_registry[model_name] 
                    if v['version'] != version
                ]
                
                # Remove model from registry if no versions left
                if not self.model_registry[model_name]:
                    del self.model_registry[model_name]
            
            # Save updated registry
            self._save_model_registry()
            
            logger.info(f"Model {model_name} version {version} deleted successfully")
            
        except Exception as e:
            raise ModelPersistenceError(f"Failed to delete model {model_name} version {version}: {e}")
    
    def export_model_for_deployment(self, model_name: str, version: str, 
                                   export_dir: str) -> str:
        """
        Export a model for deployment with all necessary files.
        
        Args:
            model_name (str): Name of the model
            version (str): Version to export
            export_dir (str): Directory to export to
            
        Returns:
            str: Path to exported model directory
            
        Raises:
            ModelPersistenceError: If export fails
        """
        try:
            # Load model and metadata
            model, metadata = self.load_model(model_name, version)
            
            # Create export directory
            export_path = Path(export_dir) / f"{model_name}_{version}"
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Copy model file
            model_file = export_path / "model.pkl"
            shutil.copy2(metadata['model_file'], model_file)
            
            # Create deployment metadata
            deployment_metadata = {
                'model_name': model_name,
                'version': version,
                'export_timestamp': datetime.now().isoformat(),
                'feature_names': metadata['feature_names'],
                'model_type': metadata['model_type'],
                'training_results': metadata['training_results'],
                'model_config': metadata['model_config']
            }
            
            # Save deployment metadata
            with open(export_path / "deployment_metadata.json", 'w') as f:
                json.dump(deployment_metadata, f, indent=2, default=str)
            
            # Create requirements file
            self._create_requirements_file(export_path)
            
            # Create README
            self._create_deployment_readme(export_path, deployment_metadata)
            
            logger.info(f"Model {model_name} version {version} exported to {export_path}")
            return str(export_path)
            
        except Exception as e:
            raise ModelPersistenceError(f"Failed to export model {model_name} version {version}: {e}")
    
    def _generate_version(self, model_name: str) -> str:
        """Generate a new version identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_num = self._get_next_version_number(model_name)
        return f"v{version_num}_{timestamp}"
    
    def _get_next_version_number(self, model_name: str) -> int:
        """Get the next version number for a model."""
        if model_name not in self.model_registry:
            return 1
        
        versions = self.model_registry[model_name]
        if not versions:
            return 1
        
        # Extract version numbers and find the highest
        version_numbers = []
        for version_info in versions:
            try:
                version_num = int(version_info['version'].split('_')[0][1:])
                version_numbers.append(version_num)
            except (ValueError, IndexError):
                continue
        
        return max(version_numbers) + 1 if version_numbers else 1
    
    def _save_model_file(self, model: Any, model_file: Path) -> None:
        """Save model to file using joblib."""
        joblib.dump(model, model_file)
    
    def _load_model_file(self, model_file: Path) -> Any:
        """Load model from file using joblib."""
        return joblib.load(model_file)
    
    def _calculate_model_hash(self, model_file: Path) -> str:
        """Calculate SHA256 hash of model file."""
        sha256_hash = hashlib.sha256()
        with open(model_file, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in megabytes."""
        return file_path.stat().st_size / (1024 * 1024)
    
    def _load_model_registry(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load model registry from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model registry: {e}")
        return {}
    
    def _save_model_registry(self) -> None:
        """Save model registry to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.model_registry, f, indent=2, default=str)
    
    def _update_model_registry(self, model_name: str, version: str, 
                              metadata: Dict[str, Any]) -> None:
        """Update model registry with new version."""
        if model_name not in self.model_registry:
            self.model_registry[model_name] = []
        
        # Add new version
        version_info = {
            'version': version,
            'timestamp': metadata['timestamp'],
            'model_type': metadata['model_type'],
            'training_results': metadata['training_results']
        }
        
        self.model_registry[model_name].append(version_info)
        
        # Sort by timestamp (newest first)
        self.model_registry[model_name].sort(
            key=lambda x: x['timestamp'], reverse=True
        )
        
        # Save registry
        self._save_model_registry()
    
    def _get_latest_version(self, model_name: str) -> str:
        """Get the latest version of a model."""
        if model_name not in self.model_registry or not self.model_registry[model_name]:
            raise ModelPersistenceError(f"No versions found for model {model_name}")
        
        return self.model_registry[model_name][0]['version']
    
    def _load_model_metadata(self, model_name: str, version: str) -> Dict[str, Any]:
        """Load metadata for a specific model version."""
        metadata_file = self.models_dir / model_name / version / "metadata.json"
        
        if not metadata_file.exists():
            raise ModelPersistenceError(f"Metadata not found for {model_name} version {version}")
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def _cleanup_old_versions(self, model_name: str) -> None:
        """Remove old versions beyond max_versions limit."""
        if model_name not in self.model_registry:
            return
        
        versions = self.model_registry[model_name]
        if len(versions) <= self.max_versions:
            return
        
        # Remove oldest versions
        versions_to_remove = versions[self.max_versions:]
        for version_info in versions_to_remove:
            version = version_info['version']
            try:
                self.delete_model(model_name, version)
            except Exception as e:
                logger.warning(f"Failed to cleanup old version {version}: {e}")
    
    def _create_requirements_file(self, export_path: Path) -> None:
        """Create requirements.txt file for deployment."""
        requirements = [
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "joblib>=1.1.0",
            "lightgbm>=3.3.0"
        ]
        
        with open(export_path / "requirements.txt", 'w') as f:
            f.write('\n'.join(requirements))
    
    def _create_deployment_readme(self, export_path: Path, metadata: Dict[str, Any]) -> None:
        """Create README file for deployment."""
        readme_content = f"""# Model Deployment Package

## Model Information
- **Model Name**: {metadata['model_name']}
- **Version**: {metadata['version']}
- **Type**: {metadata['model_type']}
- **Export Date**: {metadata['export_timestamp']}

## Performance Metrics
{json.dumps(metadata['training_results'], indent=2)}

## Usage
```python
import joblib
import json

# Load model
model = joblib.load('model.pkl')

# Load metadata
with open('deployment_metadata.json', 'r') as f:
    metadata = json.load(f)

# Make predictions
predictions = model.predict(features)
```

## Features
The model expects the following features:
{metadata['feature_names']}

## Installation
```bash
pip install -r requirements.txt
```
"""
        
        with open(export_path / "README.md", 'w') as f:
            f.write(readme_content) 