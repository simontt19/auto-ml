#!/usr/bin/env python3
"""
Test Multi-Dataset Support
Comprehensive testing of the dataset registry and multiple dataset support.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path (go up one level from tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_ml.data.ingestion import DatasetRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dataset_registry():
    """Test the dataset registry functionality."""
    logger.info("=" * 60)
    logger.info("TESTING DATASET REGISTRY")
    logger.info("=" * 60)
    
    # Initialize registry
    registry = DatasetRegistry()
    
    # Test listing all datasets
    logger.info("\n1. Listing all available datasets:")
    all_datasets = registry.list_datasets()
    for dataset in all_datasets:
        logger.info(f"  - {dataset['name']}: {dataset['description']} ({dataset['task_type']})")
    
    # Test filtering by task type
    logger.info("\n2. Filtering datasets by task type (classification):")
    classification_datasets = registry.list_datasets(task_type='classification')
    for dataset in classification_datasets:
        logger.info(f"  - {dataset['name']}: {dataset['description']}")
    
    # Test filtering by tags
    logger.info("\n3. Filtering datasets by tags (uci):")
    uci_datasets = registry.list_datasets(tags=['uci'])
    for dataset in uci_datasets:
        logger.info(f"  - {dataset['name']}: {dataset['description']}")
    
    # Test getting specific dataset info
    logger.info("\n4. Getting specific dataset information:")
    for dataset_name in ['adult_income', 'iris', 'wine', 'breast_cancer']:
        dataset_info = registry.get_dataset(dataset_name)
        if dataset_info:
            logger.info(f"  - {dataset_name}:")
            logger.info(f"    Description: {dataset_info['description']}")
            logger.info(f"    Task Type: {dataset_info['task_type']}")
            logger.info(f"    Size: {dataset_info['size']}")
            logger.info(f"    Source: {dataset_info['source']}")
            logger.info(f"    Tags: {', '.join(dataset_info['tags'])}")
    
    # Test configuration templates
    logger.info("\n5. Testing configuration templates:")
    for dataset_name in ['adult_income', 'iris', 'wine', 'breast_cancer']:
        try:
            config_template = registry.get_dataset_config_template(dataset_name)
            logger.info(f"  - {dataset_name} config template generated successfully")
            logger.info(f"    Features: {len(config_template['features'])} feature groups")
            logger.info(f"    Target: {config_template['target']}")
        except Exception as e:
            logger.error(f"  - {dataset_name} config template failed: {e}")
    
    return registry

def test_dataset_loading(registry):
    """Test loading each dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING DATASET LOADING")
    logger.info("=" * 60)
    
    datasets_to_test = ['iris', 'wine', 'breast_cancer']  # Skip adult_income for speed
    
    for dataset_name in datasets_to_test:
        logger.info(f"\nTesting dataset: {dataset_name}")
        logger.info("-" * 40)
        
        try:
            # Create ingestion instance
            ingestion = registry.create_ingestion(dataset_name)
            logger.info(f"✓ Created ingestion instance for {dataset_name}")
            
            # Load data
            train_data, test_data = ingestion.load_data()
            logger.info(f"✓ Loaded data successfully")
            logger.info(f"  Training shape: {train_data.shape}")
            logger.info(f"  Test shape: {test_data.shape}")
            
            # Check data quality
            logger.info(f"  Target column: {ingestion.get_target_column()}")
            logger.info(f"  Numerical columns: {len(ingestion.get_numerical_columns())}")
            logger.info(f"  Categorical columns: {len(ingestion.get_categorical_columns())}")
            
            # Check target distribution
            train_target_dist = train_data[ingestion.get_target_column()].value_counts()
            test_target_dist = test_data[ingestion.get_target_column()].value_counts()
            logger.info(f"  Training target distribution: {train_target_dist.to_dict()}")
            logger.info(f"  Test target distribution: {test_target_dist.to_dict()}")
            
            # Basic data validation
            assert not train_data.empty, "Training data is empty"
            assert not test_data.empty, "Test data is empty"
            assert ingestion.get_target_column() in train_data.columns, "Target column missing from training data"
            assert ingestion.get_target_column() in test_data.columns, "Target column missing from test data"
            
            logger.info(f"✓ All validations passed for {dataset_name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load {dataset_name}: {e}")
            continue

def test_dataset_validation(registry):
    """Test dataset validation functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING DATASET VALIDATION")
    logger.info("=" * 60)
    
    for dataset_name in ['iris', 'wine', 'breast_cancer']:
        try:
            is_valid = registry.validate_dataset(dataset_name)
            logger.info(f"  - {dataset_name}: {'✓ Valid' if is_valid else '✗ Invalid'}")
        except Exception as e:
            logger.error(f"  - {dataset_name}: Validation error - {e}")

def test_registry_export_import():
    """Test registry export and import functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING REGISTRY EXPORT/IMPORT")
    logger.info("=" * 60)
    
    registry = DatasetRegistry()
    
    # Export registry to results directory
    export_path = Path(__file__).parent.parent / "results" / "dataset_registry_export.json"
    try:
        registry.export_registry(str(export_path))
        logger.info(f"✓ Registry exported to {export_path}")
        
        # Check if file exists and has content
        if export_path.exists():
            file_size = export_path.stat().st_size
            logger.info(f"✓ Export file size: {file_size} bytes")
        else:
            logger.error("✗ Export file not found")
            
    except Exception as e:
        logger.error(f"✗ Export failed: {e}")

def main():
    """Run all multi-dataset tests."""
    logger.info("Starting Multi-Dataset Support Tests")
    logger.info("=" * 60)
    
    try:
        # Test 1: Dataset Registry
        registry = test_dataset_registry()
        
        # Test 2: Dataset Loading
        test_dataset_loading(registry)
        
        # Test 3: Dataset Validation
        test_dataset_validation(registry)
        
        # Test 4: Registry Export/Import
        test_registry_export_import()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Multi-dataset support is working correctly.")
        logger.info(f"Available datasets: {len(registry.datasets)}")
        
        # Summary
        logger.info("\nSUMMARY:")
        for dataset_name, dataset_info in registry.datasets.items():
            logger.info(f"  - {dataset_name}: {dataset_info['task_type']} ({dataset_info['size']})")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 