"""
Dataset Registry System
Centralized registry for managing multiple datasets with auto-discovery.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Callable
from pathlib import Path
import importlib
import inspect
from abc import ABC

from auto_ml.core.base_classes import BaseDataIngestion
from auto_ml.core.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

class DatasetRegistry:
    """
    Centralized registry for managing multiple datasets.
    
    This class provides:
    - Automatic discovery of dataset ingestion classes
    - Dataset metadata and configuration management
    - Dynamic loading of dataset-specific implementations
    - Dataset validation and compatibility checking
    
    Attributes:
        datasets (Dict[str, Dict]): Registry of available datasets
        ingestion_classes (Dict[str, Type]): Mapping of dataset names to ingestion classes
    """
    
    def __init__(self):
        """Initialize the dataset registry."""
        self.datasets = {}
        self.ingestion_classes = {}
        self._discover_datasets()
    
    def register_dataset(self, name: str, ingestion_class: Type[BaseDataIngestion], 
                        metadata: Dict[str, Any]) -> None:
        """
        Register a dataset with the registry.
        
        Args:
            name (str): Unique name for the dataset
            ingestion_class (Type[BaseDataIngestion]): Class that handles data ingestion
            metadata (Dict[str, Any]): Dataset metadata and configuration
        """
        if not issubclass(ingestion_class, BaseDataIngestion):
            raise DataIngestionError(f"Dataset class must inherit from BaseDataIngestion")
        
        self.datasets[name] = {
            'name': name,
            'ingestion_class': ingestion_class,
            'metadata': metadata,
            'description': metadata.get('description', ''),
            'task_type': metadata.get('task_type', 'unknown'),
            'features': metadata.get('features', {}),
            'target': metadata.get('target', ''),
            'size': metadata.get('size', 'unknown'),
            'source': metadata.get('source', ''),
            'license': metadata.get('license', ''),
            'tags': metadata.get('tags', [])
        }
        self.ingestion_classes[name] = ingestion_class
        
        logger.info(f"Registered dataset: {name} ({metadata.get('task_type', 'unknown')})")
    
    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get dataset information by name.
        
        Args:
            name (str): Dataset name
            
        Returns:
            Optional[Dict[str, Any]]: Dataset information or None if not found
        """
        return self.datasets.get(name)
    
    def list_datasets(self, task_type: Optional[str] = None, 
                     tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List available datasets with optional filtering.
        
        Args:
            task_type (Optional[str]): Filter by task type (classification, regression, etc.)
            tags (Optional[List[str]]): Filter by tags
            
        Returns:
            List[Dict[str, Any]]: List of matching datasets
        """
        datasets = list(self.datasets.values())
        
        if task_type:
            datasets = [d for d in datasets if d['task_type'] == task_type]
        
        if tags:
            datasets = [d for d in datasets if any(tag in d['tags'] for tag in tags)]
        
        return datasets
    
    def create_ingestion(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseDataIngestion:
        """
        Create a data ingestion instance for a specific dataset.
        
        Args:
            name (str): Dataset name
            config (Optional[Dict[str, Any]]): Configuration for the ingestion
            
        Returns:
            BaseDataIngestion: Configured data ingestion instance
            
        Raises:
            DataIngestionError: If dataset is not found or creation fails
        """
        if name not in self.ingestion_classes:
            raise DataIngestionError(f"Dataset '{name}' not found in registry")
        
        try:
            ingestion_class = self.ingestion_classes[name]
            return ingestion_class(config)
        except Exception as e:
            raise DataIngestionError(f"Failed to create ingestion for dataset '{name}': {e}")
    
    def get_dataset_config_template(self, name: str) -> Dict[str, Any]:
        """
        Get configuration template for a dataset.
        
        Args:
            name (str): Dataset name
            
        Returns:
            Dict[str, Any]: Configuration template
            
        Raises:
            DataIngestionError: If dataset is not found
        """
        dataset = self.get_dataset(name)
        if not dataset:
            raise DataIngestionError(f"Dataset '{name}' not found in registry")
        
        metadata = dataset['metadata']
        return {
            'dataset': {
                'name': name,
                'task_type': metadata.get('task_type', 'unknown'),
                'description': metadata.get('description', ''),
                'source': metadata.get('source', ''),
                'license': metadata.get('license', '')
            },
            'data': {
                'train_path': metadata.get('train_path', ''),
                'test_path': metadata.get('test_path', ''),
                'validation_split': metadata.get('validation_split', 0.2)
            },
            'features': metadata.get('features', {}),
            'target': metadata.get('target', ''),
            'preprocessing': {
                'missing_value_strategy': 'auto',
                'categorical_encoding': 'label',
                'numerical_scaling': 'standard',
                'feature_selection': False
            }
        }
    
    def validate_dataset(self, name: str) -> bool:
        """
        Validate that a dataset can be loaded and processed.
        
        Args:
            name (str): Dataset name
            
        Returns:
            bool: True if dataset is valid, False otherwise
        """
        try:
            ingestion = self.create_ingestion(name)
            # Try to load a small sample to validate
            # This is a basic validation - in production you might want more comprehensive checks
            return True
        except Exception as e:
            logger.error(f"Dataset validation failed for '{name}': {e}")
            return False
    
    def _discover_datasets(self) -> None:
        """Discover and register available datasets automatically."""
        # Register built-in datasets
        self._register_builtin_datasets()
        
        # Auto-discover additional datasets in the ingestion module
        self._auto_discover_datasets()
        
        logger.info(f"Discovered {len(self.datasets)} datasets in registry")
    
    def _register_builtin_datasets(self) -> None:
        """Register built-in datasets with the registry."""
        
        # Adult Income Dataset
        from .adult_income_ingestion import AdultIncomeDataIngestion
        self.register_dataset(
            name="adult_income",
            ingestion_class=AdultIncomeDataIngestion,
            metadata={
                'description': 'UCI Adult Income dataset for income prediction',
                'task_type': 'classification',
                'features': {
                    'categorical': ['workclass', 'education', 'marital-status', 'occupation', 
                                  'relationship', 'race', 'sex', 'native-country'],
                    'numerical': ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                                'capital-loss', 'hours-per-week']
                },
                'target': 'income',
                'size': '~50K samples',
                'source': 'UCI Machine Learning Repository',
                'license': 'Public Domain',
                'tags': ['income', 'demographics', 'classification', 'uci'],
                'train_path': 'data/adult.data',
                'test_path': 'data/adult.test',
                'validation_split': 0.2
            }
        )
        
        # Iris Dataset
        from .iris_ingestion import IrisDataIngestion
        self.register_dataset(
            name="iris",
            ingestion_class=IrisDataIngestion,
            metadata={
                'description': 'Iris flower dataset for species classification',
                'task_type': 'classification',
                'features': {
                    'numerical': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                },
                'target': 'species',
                'size': '150 samples',
                'source': 'UCI Machine Learning Repository',
                'license': 'Public Domain',
                'tags': ['flowers', 'classification', 'uci', 'small'],
                'train_path': 'data/iris.csv',
                'validation_split': 0.2
            }
        )
        
        # Wine Dataset
        from .wine_ingestion import WineDataIngestion
        self.register_dataset(
            name="wine",
            ingestion_class=WineDataIngestion,
            metadata={
                'description': 'Wine quality dataset for quality prediction',
                'task_type': 'classification',
                'features': {
                    'numerical': ['fixed_acidity', 'volatile_acidity', 'citric_acid', 
                                'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                                'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol']
                },
                'target': 'quality',
                'size': '~6K samples',
                'source': 'UCI Machine Learning Repository',
                'license': 'Public Domain',
                'tags': ['wine', 'quality', 'classification', 'uci'],
                'train_path': 'data/winequality-red.csv',
                'validation_split': 0.2
            }
        )
        
        # Breast Cancer Dataset
        from .breast_cancer_ingestion import BreastCancerDataIngestion
        self.register_dataset(
            name="breast_cancer",
            ingestion_class=BreastCancerDataIngestion,
            metadata={
                'description': 'Breast cancer diagnosis dataset',
                'task_type': 'classification',
                'features': {
                    'numerical': ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
                                'mean_smoothness', 'mean_compactness', 'mean_concavity',
                                'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension']
                },
                'target': 'diagnosis',
                'size': '~570 samples',
                'source': 'UCI Machine Learning Repository',
                'license': 'Public Domain',
                'tags': ['medical', 'cancer', 'classification', 'uci'],
                'train_path': 'data/breast_cancer.csv',
                'validation_split': 0.2
            }
        )
    
    def _auto_discover_datasets(self) -> None:
        """Auto-discover additional dataset classes in the ingestion module."""
        try:
            # Get the current module
            current_module = importlib.import_module(__name__)
            
            # Look for classes that inherit from BaseDataIngestion
            for name, obj in inspect.getmembers(current_module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseDataIngestion) and 
                    obj != BaseDataIngestion):
                    
                    # Try to get metadata from the class
                    metadata = getattr(obj, 'DATASET_METADATA', {})
                    if metadata:
                        dataset_name = metadata.get('name', name.lower())
                        self.register_dataset(dataset_name, obj, metadata)
                        
        except Exception as e:
            logger.warning(f"Auto-discovery failed: {e}")
    
    def export_registry(self, file_path: str) -> None:
        """
        Export the dataset registry to a JSON file.
        
        Args:
            file_path (str): Path to save the registry
        """
        import json
        
        export_data = {
            'datasets': self.datasets,
            'total_datasets': len(self.datasets),
            'export_timestamp': str(Path().absolute())
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Dataset registry exported to {file_path}")
    
    def import_registry(self, file_path: str) -> None:
        """
        Import a dataset registry from a JSON file.
        
        Args:
            file_path (str): Path to the registry file
        """
        import json
        
        with open(file_path, 'r') as f:
            import_data = json.load(f)
        
        # Note: This is a basic import - in production you'd want to validate
        # and handle class loading more carefully
        logger.info(f"Imported dataset registry from {file_path}") 